#!/usr/bin/env python
"""Out-of-sample test: blind Jev vs the Kalshi book, on markets settled after Jev's build.

For markets that settled in [LO, HI), take the book mid at AS_OF from Kalshi
candlesticks and ask Jev for P(YES) as of AS_OF from the market rules only (no
price — the price anchors it). Then score both against the real outcomes
(Brier), and simulate taking the side Jev favours whenever it disagrees with
the book by more than T (taker, fees included).

Result on 2026-09-25 (n=601, as-of 2026-09-18, settled Sep 19-25, 98% sports):
Brier book 0.157 vs Jev 0.209 vs 50/50 blend 0.171; trading disagreements lost
at every threshold. Blind Jev has no edge. See docs/JEV.md.

Usage:
    python scripts/jev_backtest.py --as-of 2026-09-18T12:00 --lo 2026-09-19 --hi 2026-09-26 --max 600 --out bt.json
    python scripts/jev_backtest.py --analyze bt.json
"""
import argparse
import asyncio
import collections
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.agent.jev import DECISIONS_URL, JEV_MODEL  # noqa: E402


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _request(m: dict, cat: str, as_of: datetime) -> dict:
    q = (m.get("title") or "") + (" :: " + m["yes_sub_title"] if m.get("yes_sub_title") else "")
    return {"model": JEV_MODEL,
            "state": {"today": as_of.date().isoformat(), "category": cat, "market_question": q,
                      "resolution_rules": (m.get("rules_primary") or "")[:1500],
                      "additional_rules": (m.get("rules_secondary") or "")[:800],
                      "market_closes": m.get("close_time")},
            "questions": {"resolves_yes": {"type": "noul",
                          "instructions": "Will this prediction market resolve YES under its exact rules?",
                          "criteria": {"true": "Resolves YES", "false": "Resolves NO"}}}}


async def collect(as_of: datetime, lo: datetime, hi: datetime, max_markets: int) -> list:
    import httpx
    from src.clients.kalshi_client import KalshiClient

    c = KalshiClient(); events = []; cur = None
    while True:
        p = {"status": "settled", "limit": 200, "min_close_ts": int(lo.timestamp())}
        if cur:
            p["cursor"] = cur
        r = await c._make_authenticated_request("GET", "/trade-api/v2/events", params=p)
        events += r["events"]; cur = r.get("cursor")
        if not cur or not r["events"]:
            break
    random.seed(7); random.shuffle(events)
    rows = []
    for e in events:
        if len(rows) >= max_markets:
            break
        try:
            d = await c._make_authenticated_request("GET", f"/trade-api/v2/events/{e['event_ticker']}",
                                                    params={"with_nested_markets": "true"})
        except Exception:
            continue
        ms = [m for m in (d.get("markets") or d.get("event", {}).get("markets") or [])
              if m.get("result") in ("yes", "no") and lo <= _ts(m["close_time"]) < hi]
        for m in ms[:6]:
            try:
                cs = await c._make_authenticated_request(
                    "GET", f"/trade-api/v2/series/{e['series_ticker']}/markets/{m['ticker']}/candlesticks",
                    params={"start_ts": int(as_of.timestamp()) - 6 * 3600, "end_ts": int(as_of.timestamp()),
                            "period_interval": 60})
                k = cs["candlesticks"][-1]
                yb, ya = float(k["yes_bid"]["close_dollars"] or 0), float(k["yes_ask"]["close_dollars"] or 0)
            except Exception:
                continue
            mid = (yb + ya) / 2
            if 0 < yb < ya <= 1 and ya - yb <= 0.10 and 0.03 <= mid <= 0.97:
                rows.append({"ticker": m["ticker"], "cat": e["category"], "yes_bid": yb, "yes_ask": ya,
                             "mid": mid, "y": int(m["result"] == "yes"), "req": _request(m, e["category"], as_of)})
    await c.close()
    key = os.environ["OPENROUTER_API_KEY"]; sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient(timeout=90) as h:
        async def one(r):
            async with sem:
                for _ in range(2):
                    try:
                        x = await h.post(DECISIONS_URL, json=r["req"], headers={"Authorization": f"Bearer {key}"})
                        r["jev"] = float(x.json()["answers"]["resolves_yes"]["noul"]); return
                    except Exception:
                        continue
        await asyncio.gather(*(one(r) for r in rows))
    rows = [r for r in rows if "jev" in r]
    for r in rows:
        r.pop("req")
    return rows


def analyze(rows: list) -> None:
    fee = lambda p: math.ceil(0.07 * p * (1 - p) * 100) / 100
    brier = lambda rs, k: sum((r[k] - r["y"]) ** 2 for r in rs) / len(rs)
    for r in rows:
        r["blend"] = 0.5 * r["jev"] + 0.5 * r["mid"]
    groups = [("ALL", rows)] + sorted(
        ((k, v) for k, v in collections.Counter(r["cat"] for r in rows).items()), key=lambda kv: -kv[1])
    for label, rs in groups:
        rs = rows if label == "ALL" else [r for r in rows if r["cat"] == label]
        if len(rs) >= 5:
            print(f"{label:24} n={len(rs):4}  Brier book {brier(rs, 'mid'):.4f}  jev {brier(rs, 'jev'):.4f}  blend {brier(rs, 'blend'):.4f}")
    for t in (0.05, 0.10, 0.20, 0.30):
        pnl = []
        for r in rows:
            d = r["jev"] - r["mid"]
            if d > t:
                pnl.append(r["y"] - r["yes_ask"] - fee(r["yes_ask"]))
            elif d < -t:
                p = 1 - r["yes_bid"]; pnl.append((1 - r["y"]) - p - fee(p))
        if pnl:
            print(f"  disagreement > {t:.2f}: {len(pnl)} trades, P&L ${sum(pnl):+.2f} ({sum(pnl)/len(pnl):+.3f}/trade)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", default="2026-09-18T12:00")
    ap.add_argument("--lo", default="2026-09-19")
    ap.add_argument("--hi", default="2026-09-26")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--out", default="data/runtime/jev_backtest.json")
    ap.add_argument("--analyze", help="analyze a saved results file instead of collecting")
    a = ap.parse_args()
    load_dotenv()
    if a.analyze:
        analyze(json.loads(Path(a.analyze).read_text())); return
    utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    rows = asyncio.run(collect(utc(a.as_of), utc(a.lo), utc(a.hi), a.max))
    Path(a.out).write_text(json.dumps(rows))
    print(f"scored {len(rows)} -> {a.out}")
    analyze(rows)


if __name__ == "__main__":
    main()
