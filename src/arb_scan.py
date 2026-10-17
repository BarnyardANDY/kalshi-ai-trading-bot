"""
Price-inconsistency scan (measurement only: never trades). Separate from every
other test: own process (``cli.py arb-scan``), database (``data/arb_scan.db``),
log (``logs/arb_scan.log``) and dashboard tab.

Looks across every open Kalshi event for prices that can't all be right at once,
where a set of trades pays more than it costs no matter what happens, after
Kalshi's taker fee on every leg:

1. **All NO, mutually exclusive events.** At most one outcome can be YES, so
   buying NO on all n outcomes pays at least n-1 dollars. Riskless if the asks
   plus fees cost less than n-1.
2. **All YES, mutually exclusive events.** Pays exactly 1 dollar IF one listed
   outcome must happen. That's only riskless when the list is complete (range
   brackets with open ends are; candidate lists may not be), so these are
   labelled "complete list" or "list may be incomplete".
3. **Ladders.** For "above X" markets in one event, the higher threshold can't be
   more likely than the lower one: YES(lower) + NO(higher) always pays at least
   1 dollar. Same in reverse for "below X" ladders.

Each finding is checked against the order books for how many contracts were
actually available at those prices, and tracked across scans so we know how
long it lasted.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DB_PATH = os.getenv("ARB_DB", "data/arb_scan.db")
TAKER_FEE_RATE = 0.07
EXCLUDE_PREFIXES = ("KXMVE",)
TZ = os.getenv("DASHBOARD_TZ", "America/New_York")


def _f(d: Dict[str, Any], dollars: str, cents: str) -> float:
    for k, scale in ((dollars, 1.0), (cents, 0.01)):
        v = d.get(k)
        if v not in (None, ""):
            try:
                return float(v) * scale
            except (TypeError, ValueError):
                pass
    return 0.0


def _ts(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def fee(p: float) -> float:
    return TAKER_FEE_RATE * p * (1 - p) if 0 < p < 1 else 0.0


def asks(m: Dict[str, Any]) -> Tuple[float, float]:
    return _f(m, "yes_ask_dollars", "yes_ask"), _f(m, "no_ask_dollars", "no_ask")


def _live(m: Dict[str, Any]) -> bool:
    return (m.get("status") or "active") in ("active", "open") and not (m.get("ticker") or "").startswith(EXCLUDE_PREFIXES)


def _complete_list(markets: List[Dict[str, Any]]) -> bool:
    """Range brackets with an open bottom and an open top cover every outcome."""
    kinds = {(m.get("strike_type") or "").lower() for m in markets}
    has_low = any(k.startswith("less") for k in kinds)
    has_high = any(k.startswith("greater") for k in kinds)
    return has_low and has_high and kinds <= {"less", "less_or_equal", "greater", "greater_or_equal", "between"}


def _strike(m: Dict[str, Any], key: str) -> Optional[float]:
    v = m.get(key)
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def check_event(ev: Dict[str, Any], min_edge: float = 0.0) -> List[Dict[str, Any]]:
    """Riskless combinations in one event whose guaranteed payout beats cost + fees by > ``min_edge``
    (dollars per set). Each leg is (ticker, side, ask)."""
    mk = [m for m in ev.get("markets") or [] if _live(m)]
    out: List[Dict[str, Any]] = []
    if len(mk) < 2:
        return out
    et = ev.get("event_ticker") or ""

    closes = {m["ticker"]: _ts(m.get("close_time")) for m in mk}

    def add(kind, legs, payoff, note=""):
        cost = sum(p for _, _, p in legs)
        fees = sum(fee(p) for _, _, p in legs)
        edge = payoff - cost - fees
        if edge > min_edge:
            pay_ts = max((closes.get(t) or 0) for t, _, _ in legs) or None
            out.append({"kind": kind, "event": et, "title": ev.get("title") or "", "legs": legs, "cost": cost,
                        "close_ts": pay_ts,
                        "fees": fees, "payoff": payoff, "edge": edge, "note": note,
                        "key": kind + "|" + "|".join(f"{t}:{s}" for t, s, _ in sorted(legs))})

    if ev.get("mutually_exclusive") and len(mk) == len([m for m in ev.get("markets") or []
                                                        if not (m.get("ticker") or "").startswith(EXCLUDE_PREFIXES)]):
        ya = [(m["ticker"], "yes", asks(m)[0]) for m in mk]
        na = [(m["ticker"], "no", asks(m)[1]) for m in mk]
        if all(0 < p < 1 for _, _, p in na):
            add("all NO (one winner at most)", na, float(len(mk) - 1))
        if all(0 < p < 1 for _, _, p in ya):
            add("all YES (exactly one winner)", ya, 1.0,
                "complete list" if _complete_list(mk) else "list may be incomplete")

    for direction, key in (("greater", "floor_strike"), ("less", "cap_strike")):
        rungs = [(m, _strike(m, key)) for m in mk if (m.get("strike_type") or "").lower().startswith(direction)]
        groups: Dict[str, list] = {}
        for m_, s_ in rungs:
            if s_ is not None:
                groups.setdefault(subject_key(m_), []).append((m_, s_))
        for lad in groups.values():
            lad.sort(key=lambda x: x[1])
            _ladder_pairs(lad, direction, add)
    return out


def subject_key(m: Dict[str, Any]) -> str:
    """What a rung is ABOUT, without its number. One event can hold ladders for many subjects
    (e.g. every NBA team's seed), and rungs only constrain each other within one subject."""
    norm = lambda x: re.sub(r"[-+]?\$?\d[\d,]*(\.\d+)?", "#", (x or "").lower()).strip()
    return "|".join([norm(m.get("title")), norm(m.get("yes_sub_title")),
                     json.dumps(m.get("custom_strike") or {}, sort_keys=True)])


def _ladder_pairs(lad, direction, add) -> None:
    """Every pair of rungs (sorted by strike) of one subject's ladder."""
    for i in range(len(lad)):
        for j in range(i + 1, len(lad)):
            lo, hi = lad[i][0], lad[j][0]
            if direction == "greater":     # above(lo) >= above(hi): YES lo + NO hi pays >= 1
                legs = [(lo["ticker"], "yes", asks(lo)[0]), (hi["ticker"], "no", asks(hi)[1])]
            else:                          # below(hi) >= below(lo): YES hi + NO lo pays >= 1
                legs = [(hi["ticker"], "yes", asks(hi)[0]), (lo["ticker"], "no", asks(lo)[1])]
            if all(0 < p < 1 for _, _, p in legs):
                add(f"ladder ({'above' if direction == 'greater' else 'below'} X)", legs, 1.0)


def depth_at_ask(book: Dict[str, Any], side: str, price: float) -> Optional[float]:
    """Contracts offered at ``price`` to buy ``side``. Kalshi's book lists bids only: an ask for
    YES at p is a NO bid at 1-p (and vice versa)."""
    ob = book.get("orderbook_fp") if isinstance(book.get("orderbook_fp"), dict) else book.get("orderbook", book)
    other = "no" if side == "yes" else "yes"
    want = round(1 - price, 4)
    for key, scale in ((f"{other}_dollars", 1.0), (other, 0.01)):
        if key in (ob or {}):
            total = 0.0
            for lvl in ob.get(key) or []:
                try:
                    if abs(float(lvl[0]) * scale - want) < 0.0005:
                        total += float(lvl[1])
                except (TypeError, ValueError, IndexError):
                    return None
            return total
    return None


# ----------------------------------------------------------------------------
# Storage and live loop
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS opps (key TEXT, kind TEXT, event TEXT, title TEXT, note TEXT,
                 legs TEXT, first_seen REAL, last_seen REAL, scans INTEGER, best_edge REAL, first_edge REAL,
                 contracts REAL, open INTEGER, PRIMARY KEY (key, first_seen))""")
    c.execute("CREATE TABLE IF NOT EXISTS scans (ts REAL PRIMARY KEY, events INTEGER, markets INTEGER, seconds REAL)")
    if "close_ts" not in {r[1] for r in c.execute("PRAGMA table_info(opps)")}:
        c.execute("ALTER TABLE opps ADD COLUMN close_ts REAL")
    return c


async def fetch_events(client, max_pages: int = 60) -> List[Dict[str, Any]]:
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = await client._make_authenticated_request("GET", "/trade-api/v2/events", params=params,
                                                         require_auth=False)
        page = resp.get("events") or []
        out.extend(page)
        cursor = resp.get("cursor")
        if not cursor or len(page) < 200:
            break
    return out


async def scan_once(client, now: float, path: Optional[str] = None, logger=None,
                    events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    t0 = time.time()
    events = events if events is not None else await fetch_events(client)
    found: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        for o in check_event(ev):
            found[o["key"]] = o
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        open_rows = {r["key"]: dict(r) for r in c.execute("SELECT * FROM opps WHERE open=1")}
        new = 0
        for k, o in found.items():
            if k in open_rows:
                r = open_rows[k]
                c.execute("UPDATE opps SET last_seen=?, scans=scans+1, best_edge=MAX(best_edge, ?) "
                          "WHERE key=? AND first_seen=?", (now, o["edge"], k, r["first_seen"]))
                continue
            contracts = None
            if o["note"] != "list may be incomplete":     # not riskless anyway: skip the order-book calls
                try:
                    depths = [depth_at_ask(await client.get_orderbook(t), side, p) for t, side, p in o["legs"]]
                    if depths and all(d is not None for d in depths):
                        contracts = min(depths)
                except Exception:
                    contracts = None
            c.execute("INSERT INTO opps (key, kind, event, title, note, legs, first_seen, last_seen, scans, best_edge,"
                      " first_edge, contracts, open, close_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
                      (k, o["kind"], o["event"], o["title"][:200], o["note"], json.dumps(o["legs"]), now, now, 1,
                       o["edge"], o["edge"], contracts, o.get("close_ts")))
            new += 1
            if logger:
                days = (o["close_ts"] - now) / 86400 if o.get("close_ts") else None
                logger.info(f"ARB {o['kind']} on {o['event']}: pays ${o['payoff']:.0f} for "
                            f"${o['cost'] + o['fees']:.3f} (edge {o['edge'] * 100:.1f}c per set, "
                            f"{contracts if contracts is not None else '?'} sets available, pays out in "
                            f"{f'{days:.0f} days' if days is not None else '?'}) {o['note']}")
        gone = [k for k in open_rows if k not in found]
        for k in gone:
            c.execute("UPDATE opps SET open=0 WHERE key=? AND first_seen=?", (k, open_rows[k]["first_seen"]))
        n_mk = sum(len(e.get("markets") or []) for e in events)
        c.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?,?)", (now, len(events), n_mk, time.time() - t0))
        c.commit()
    return {"events": len(events), "found": len(found), "new": new, "closed": len(gone)}


async def run_forever(logger=None, interval: float = 30.0, path: Optional[str] = None) -> None:
    from src.clients.kalshi_client import KalshiClient

    client = KalshiClient()
    last_log = 0.0
    try:
        while True:
            t0 = time.time()
            try:
                st = await scan_once(client, t0, path, logger)
                if logger and time.time() - last_log > 600:
                    last_log = time.time()
                    logger.info(f"ARB scan: {st['events']} events checked, {st['found']} inconsistencies open now "
                                f"({time.time() - t0:.0f}s per scan)")
            except Exception as e:
                if logger:
                    logger.warning(f"ARB scan failed: {e}")
            await asyncio.sleep(max(1.0, interval - (time.time() - t0)))
    finally:
        await client.close()


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def report(path: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.exists(path or DB_PATH):
        return {"scans": 0}
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        opps = pd.DataFrame([dict(r) for r in c.execute("SELECT * FROM opps")])
        sc = c.execute("SELECT COUNT(*), MIN(ts), MAX(ts), AVG(seconds), AVG(events) FROM scans").fetchone()
    hours = ((sc[2] or 0) - (sc[1] or 0)) / 3600 if sc[0] else 0.0
    out: Dict[str, Any] = {"scans": sc[0], "hours": hours, "scan_seconds": sc[3], "events_per_scan": sc[4],
                           "by_kind": pd.DataFrame(), "recent": pd.DataFrame(), "total": 0}
    if opps.empty:
        return out
    opps["lasted_min"] = (opps["last_seen"] - opps["first_seen"]) / 60
    opps["edge_c"] = 100 * opps["best_edge"]
    opps["profit_if_taken"] = opps["first_edge"] * opps["contracts"].fillna(0)
    opps["seen_et"] = pd.to_datetime(opps["first_seen"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    if "close_ts" not in opps:
        opps["close_ts"] = None
    opps["days_to_payout"] = (opps["close_ts"].astype(float) - opps["first_seen"]) / 86400
    days = max(hours / 24, 1 / 24)
    g = opps.groupby("kind").agg(found=("key", "size"), per_day=("key", lambda s: len(s) / days),
                                 median_edge_c=("edge_c", "median"), median_minutes=("lasted_min", "median"),
                                 with_10plus_sets=("contracts", lambda s: int((s.fillna(0) >= 10).sum())),
                                 profit_if_all_taken=("profit_if_taken", "sum")).reset_index()
    out.update(total=len(opps), by_kind=g,
               recent=opps.sort_values("first_seen", ascending=False)[
                   ["seen_et", "kind", "event", "title", "note", "edge_c", "contracts", "days_to_payout", "lasted_min",
                    "open"]].head(300))
    return out
