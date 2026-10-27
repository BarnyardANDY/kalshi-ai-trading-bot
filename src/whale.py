"""
Whale-follow study (measurement only: never trades). Separate from every other test:
own process (``cli.py whale-study``), database (``data/whale_study.db``), log
(``logs/whale_study.log``) and dashboard tab.

Kalshi is anonymous, so we can't follow a person, but its public trade tape shows
every trade. This watches the whole tape and asks one question:

    If you had copied every unusually large trade ONE MINUTE LATE, buying the same
    side at the real ask and paying the taker fee, then held to settlement,
    would you have made money?

How it works:

1. Poll Kalshi's global trade tape every few seconds.
2. Group prints in the same market, same direction, within 1 second into one
   "whale event" (a big order usually sweeps several price levels at once).
   Keep events of $1,000+ (cost to the buyer); the size split shows where size starts to matter.
3. Snapshot that market at +1 min (the copy price: the ask on the whale's side),
   +5, +30 and +60 min (did the price keep drifting the whale's way?), and the
   settlement result.
4. Copy P&L per contract = outcome - ask at +1 min - fee.
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import statistics
import time
from collections import defaultdict, deque
from contextlib import closing
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Optional

import pandas as pd

DB_PATH = os.getenv("WHALE_DB", "data/whale_study.db")
TAKER_FEE_RATE = 0.07
REPORT_PATH = os.getenv("WHALE_REPORT", "data/whale_report.json")   # written by the study; the dashboard just reads it
MIN_NOTIONAL = float(os.getenv("WHALE_MIN_NOTIONAL", "1000"))   # $250 logged ~200k trades/day: too heavy for 2GB
GROUP_SECONDS = 1.0   # prints within 1 s of an order's FIRST print = one sweep (never chains steady flow)
EXCLUDE_PREFIXES = ("KXMVE",)
CHECKPOINTS = (("entry", 60), ("mid_5m", 300), ("mid_30m", 1800), ("mid_60m", 3600))
TZ = os.getenv("DASHBOARD_TZ", "America/New_York")


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def _f(d: Dict[str, Any], *keys: str, cents: Iterable[str] = ()) -> float:
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    for k in cents:
        v = d.get(k)
        if v not in (None, ""):
            try:
                return float(v) / 100
            except (TypeError, ValueError):
                pass
    return 0.0


def ts_of(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def taker_fee(p: float) -> float:
    return TAKER_FEE_RATE * p * (1 - p) if 0 < p < 1 else 0.0


def trade_view(tr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalised trade: who bought which side, how many, at what price."""
    side = (tr.get("taker_side") or "").lower()
    t = ts_of(tr.get("created_time"))
    if side not in ("yes", "no") or t is None or not tr.get("ticker"):
        return None
    price = _f(tr, f"{side}_price_dollars", cents=(f"{side}_price",))
    count = _f(tr, "count_fp", "count")
    if not 0 < price < 1 or count <= 0:
        return None
    return {"ticker": tr["ticker"], "side": side, "ts": t, "price": price, "count": count,
            "id": tr.get("trade_id")}


def group_prints(trades: List[Dict[str, Any]], window: float = GROUP_SECONDS) -> List[Dict[str, Any]]:
    """Merge prints in the same market and direction within ``window`` seconds of the first one
    into one event (a large order sweeping several price levels at once)."""
    out: List[Dict[str, Any]] = []
    open_: Dict[tuple, Dict[str, Any]] = {}
    for v in sorted((x for x in (trade_view(t) for t in trades) if x), key=lambda x: x["ts"]):
        key = (v["ticker"], v["side"])
        ev = open_.get(key)
        if ev and v["ts"] - ev["ts"] < window:
            ev["contracts"] += v["count"]
            ev["notional"] += v["count"] * v["price"]
            ev["prints"] += 1
            ev["last_ts"] = v["ts"]
            ev["last_price"] = v["price"]
        else:
            if ev:
                out.append(ev)
            open_[key] = {"ticker": v["ticker"], "side": v["side"], "ts": v["ts"], "last_ts": v["ts"],
                          "contracts": v["count"], "notional": v["count"] * v["price"], "prints": 1,
                          "first_price": v["price"], "last_price": v["price"]}
    out.extend(open_.values())
    for ev in out:
        ev["vwap"] = ev["notional"] / ev["contracts"]
    return sorted(out, key=lambda e: e["ts"])


def side_quotes(m: Dict[str, Any], side: str) -> Dict[str, float]:
    bid = _f(m, f"{side}_bid_dollars", cents=(f"{side}_bid",))
    ask = _f(m, f"{side}_ask_dollars", cents=(f"{side}_ask",))
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else None}


def copy_pnl(side: str, entry_ask: Optional[float], result: str) -> Optional[float]:
    if not entry_ask or not 0.02 <= entry_ask <= 0.98 or result not in ("yes", "no"):
        return None
    return (1.0 if result == side else 0.0) - entry_ask - taker_fee(entry_ask)


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

COLUMNS = ["ts", "ticker", "event", "series", "title", "side", "contracts", "notional", "vwap", "first_price",
           "last_price", "prints", "typical_count", "close_ts", "entry_ask", "entry_bid", "entry_mid",
           "mid_5m", "mid_30m", "mid_60m", "result", "pnl_copy", "settled_ts"]


def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS whales (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, ticker TEXT,
                 event TEXT, series TEXT, title TEXT, side TEXT, contracts REAL, notional REAL, vwap REAL,
                 first_price REAL, last_price REAL, prints INTEGER, typical_count REAL, close_ts REAL,
                 entry_ask REAL, entry_bid REAL, entry_mid REAL, mid_5m REAL, mid_30m REAL, mid_60m REAL,
                 result TEXT, pnl_copy REAL, settled_ts REAL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_whales_ts ON whales(ts)")
    c.execute("CREATE TABLE IF NOT EXISTS series_meta (series TEXT PRIMARY KEY, category TEXT, title TEXT)")
    return c


def rows(path: Optional[str] = None, where: str = "", params=()) -> List[Dict[str, Any]]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(f"SELECT * FROM whales {where}", params)]


def _update(wid: int, path: Optional[str] = None, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn(path)) as c:
        c.execute(f"UPDATE whales SET {keys} WHERE id=?", (*fields.values(), wid))
        c.commit()


def series_meta(path: Optional[str] = None) -> Dict[str, str]:
    with closing(_conn(path)) as c:
        return {r[0]: r[1] for r in c.execute("SELECT series, category FROM series_meta")}


# ----------------------------------------------------------------------------
# Live loop (Kalshi public endpoints; read-only)
# ----------------------------------------------------------------------------

class Tape:
    """Pulls new prints from Kalshi's global trade tape and turns them into whale events."""

    def __init__(self, min_notional: float = MIN_NOTIONAL):
        self.min_notional = min_notional
        self.since = time.time() - 30
        self.seen: Deque[str] = deque(maxlen=200_000)
        self.seen_set: set = set()
        self.sizes: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
        self.pending: List[Dict[str, Any]] = []      # prints younger than the grouping window

    async def poll(self, client, max_pages: int = 20) -> List[Dict[str, Any]]:
        new, cursor = [], None
        for _ in range(max_pages):
            params = {"limit": 1000, "min_ts": int(self.since) - 2}
            if cursor:
                params["cursor"] = cursor
            resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets/trades",
                                                             params=params, require_auth=False)
            page = resp.get("trades") or []
            for tr in page:
                tid = tr.get("trade_id") or f"{tr.get('ticker')}|{tr.get('created_time')}|{tr.get('count')}"
                if tid in self.seen_set:
                    continue
                self.seen.append(tid)
                self.seen_set.add(tid)
                new.append(tr)
            cursor = resp.get("cursor")
            if not cursor or len(page) < 1000:
                break
        if len(self.seen_set) > len(self.seen) + 1000:      # keep the set in step with the bounded deque
            self.seen_set = set(self.seen)
        if new:
            self.since = max(ts_of(t.get("created_time")) or self.since for t in new)
        return new

    def events(self, new_trades: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        """Whale events whose grouping window has closed."""
        self.pending.extend(t for t in new_trades if not (t.get("ticker") or "").startswith(EXCLUDE_PREFIXES))
        cutoff = now - GROUP_SECONDS - 3
        ready = [t for t in self.pending if (ts_of(t.get("created_time")) or now) <= cutoff]
        self.pending = [t for t in self.pending if (ts_of(t.get("created_time")) or now) > cutoff]
        out = []
        for ev in group_prints(ready):
            hist = self.sizes[ev["ticker"]]
            ev["typical_count"] = statistics.median(hist) if len(hist) >= 10 else None
            if ev["notional"] >= self.min_notional:
                out.append(ev)
            hist.append(ev["contracts"])
        return out


async def run_forever(logger=None, interval: float = 5.0, path: Optional[str] = None,
                      min_notional: float = MIN_NOTIONAL) -> None:
    from src.clients.kalshi_client import KalshiClient

    client = KalshiClient()
    tape = Tape(min_notional)
    last_settle = 0.0
    last_report = 0.0
    report_task = None
    try:
        while True:
            t0 = time.time()
            try:
                if time.time() - last_report > 600 and (report_task is None or report_task.done()):
                    from src import report_runner
                    last_report = time.time()
                    report_task = asyncio.create_task(report_runner.run(
                        "src.whale", "WHALE", logger,
                        env={"WHALE_DB": path, "WHALE_REPORT": os.path.join(os.path.dirname(path), "whale_report.json")}
                        if path else None))
                new = await tape.poll(client)
                evs = tape.events(new, time.time())
                if evs:
                    with closing(_conn(path)) as c:
                        for e in evs:
                            c.execute("""INSERT INTO whales (ts, ticker, series, side, contracts, notional, vwap,
                                         first_price, last_price, prints, typical_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                      (e["ts"], e["ticker"], e["ticker"].split("-", 1)[0], e["side"], e["contracts"],
                                       e["notional"], e["vwap"], e["first_price"], e["last_price"], e["prints"],
                                       e["typical_count"]))
                        c.commit()
                    if logger:
                        big = max(evs, key=lambda e: e["notional"])
                        logger.info(f"WHALE {len(evs)} new events (largest ${big['notional']:,.0f}: "
                                    f"{big['side'].upper()} {big['contracts']:.0f} on {big['ticker']} @ {big['vwap']:.2f})")
                await checkpoints(client, time.time(), path, logger)
                if time.time() - last_settle > 600:
                    last_settle = time.time()
                    await settle(client, time.time(), path, logger)
                    await categories(client, path)
            except Exception as e:
                if logger:
                    logger.warning(f"WHALE cycle failed: {e}")
            await asyncio.sleep(max(0.0, interval - (time.time() - t0)))
    finally:
        await client.close()


async def _markets(client, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    got: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(tickers), 50):
        resp = await client.get_markets(limit=100, tickers=tickers[i:i + 50])
        for m in resp.get("markets") or []:
            got[m.get("ticker")] = m
    return got


async def checkpoints(client, now: float, path: Optional[str] = None, logger=None) -> None:
    """Fill in the +1/5/30/60 minute snapshots that have come due (each within a 10-minute grace)."""
    jobs = []
    for r in rows(path, "WHERE ts > ? AND (entry_ask IS NULL OR mid_5m IS NULL OR mid_30m IS NULL OR mid_60m IS NULL)",
                  (now - 3600 - 900,)):
        for col, lag in CHECKPOINTS:
            target = "entry_ask" if col == "entry" else col
            if r.get(target) is None and r["ts"] + lag <= now <= r["ts"] + lag + 600:
                jobs.append((r, col))
    if not jobs:
        return
    try:
        got = await _markets(client, sorted({r["ticker"] for r, _ in jobs}))
    except Exception as e:
        if logger:
            logger.warning(f"WHALE snapshot failed: {e}")
        return
    for r, col in jobs:
        m = got.get(r["ticker"])
        if not m:
            continue
        q = side_quotes(m, r["side"])
        res = (m.get("result") or "").lower()
        if col == "entry":
            _update(r["id"], path, entry_ask=q["ask"] or -1.0, entry_bid=q["bid"], entry_mid=q["mid"],
                    event=m.get("event_ticker"), close_ts=ts_of(m.get("close_time")),
                    title=" — ".join(x for x in (m.get("title"), m.get("yes_sub_title")) if x)[:200])
        else:
            val = (1.0 if res == r["side"] else 0.0) if res in ("yes", "no") else q["mid"]
            if val is not None:
                _update(r["id"], path, **{col: val})


async def settle(client, now: float, path: Optional[str] = None, logger=None) -> None:
    todo = rows(path, "WHERE result IS NULL AND entry_ask IS NOT NULL AND ts < ?", (now - 120,))
    if not todo:
        return
    try:
        got = await _markets(client, sorted({r["ticker"] for r in todo}))
    except Exception as e:
        if logger:
            logger.warning(f"WHALE settle lookup failed: {e}")
        return
    n = 0
    for r in todo:
        m = got.get(r["ticker"])
        res = (m or {}).get("result", "").lower() if m else ""
        if res in ("yes", "no"):
            _update(r["id"], path, result=res, pnl_copy=copy_pnl(r["side"], r["entry_ask"], res), settled_ts=now)
            n += 1
        elif r["close_ts"] and now - r["close_ts"] > 7 * 86400:
            _update(r["id"], path, result="unresolved", settled_ts=now)
    if n and logger:
        logger.info(f"WHALE {n} followed markets settled")


async def categories(client, path: Optional[str] = None, per_call: int = 20) -> None:
    known = series_meta(path)
    with closing(_conn(path)) as c:
        todo = [r[0] for r in c.execute("SELECT DISTINCT series FROM whales") if r[0] not in known][:per_call]
    for sr in todo:
        try:
            info = (await client._make_authenticated_request("GET", f"/trade-api/v2/series/{sr}",
                                                              require_auth=False)).get("series") or {}
        except Exception:
            info = {}
        with closing(_conn(path)) as c:
            c.execute("INSERT OR REPLACE INTO series_meta VALUES (?,?,?)",
                      (sr, info.get("category") or "Unknown", info.get("title") or ""))
            c.commit()


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

DIMENSIONS = {
    "Size of the trade ($)": "size_bucket",
    "Category": "category",
    "Price the whale paid": "whale_price_bucket",
    "Whale bought the favorite or the long shot": "fav_label",
    "Size vs. normal for that market": "multiple_bucket",
    "Time until the market closes": "time_left_bucket",
    "Time of day (Eastern)": "daypart",
    "Swept several prices at once?": "sweep_label",
    "Series (market type)": "series",
}


def _cut(v, edges, labels):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def features(all_rows: List[Dict[str, Any]], meta: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    meta = meta or {}
    df["category"] = df["series"].map(lambda s: meta.get(s) or "unknown")
    df["size_bucket"] = df["notional"].map(lambda v: _cut(v, [1000, 5000, 25000],
                                                          ["$250-999", "$1k-5k", "$5k-25k", "$25k+"]))
    df["whale_price_bucket"] = df["vwap"].map(lambda v: _cut(v, [0.25, 0.5, 0.75, 0.9],
                                                             ["<25c", "25-49c", "50-74c", "75-89c", "90c+"]))
    df["fav_label"] = df["vwap"].map(lambda v: "favorite (50c+)" if v >= 0.5 else "long shot (<50c)")
    mult = df["contracts"] / df["typical_count"].astype(float)
    df["multiple_bucket"] = mult.map(lambda v: _cut(v, [5, 20, 100], ["<5x normal", "5-20x", "20-100x", "100x+"]))
    left = (df["close_ts"].astype(float) - df["ts"]) / 3600
    df["time_left_bucket"] = left.map(lambda h: _cut(h, [1, 6, 24, 168], ["<1h", "1-6h", "6-24h", "1-7 days", "7 days+"]))
    t = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TZ)
    df["time_et"] = t.dt.tz_localize(None)
    df["daypart"] = t.dt.hour.map(lambda h: "night (0-6)" if h < 6 else "morning (6-12)" if h < 12
                                  else "afternoon (12-18)" if h < 18 else "evening (18-24)")
    df["sweep_label"] = df["prints"].map(lambda n: "yes (2+ prices)" if n and n > 1 else "no (single print)")
    entry = df["entry_ask"].astype(float).where(df["entry_ask"].astype(float) > 0)   # -1 = no ask at +1 min
    for col, lag in (("mid_5m", 5), ("mid_30m", 30), ("mid_60m", 60)):
        df[f"drift_{lag}m_c"] = 100 * (df[col].astype(float) - entry)
    df["copy_c"] = 100 * df["pnl_copy"].astype(float)
    return df


def summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"events": 0}
    s = df[df["pnl_copy"].notna()]
    out = {"events": len(df), "with_entry": int((df["entry_ask"].astype(float) > 0).sum()),
           "settled": len(s), "markets": s["ticker"].nunique() if len(s) else 0}
    if len(s):
        n_eff = max(1, s["event"].fillna(s["ticker"]).nunique())
        out["copy_c"] = s["copy_c"].mean()
        out["margin"] = 1.96 * s["copy_c"].std(ddof=1) / math.sqrt(n_eff) if len(s) > 1 else None
        out["win_rate"] = (s["pnl_copy"] > 0).mean()
        out["avg_entry"] = s["entry_ask"].mean()
    for lag in (5, 30, 60):
        col = f"drift_{lag}m_c"
        v = df[col].dropna()
        out[f"drift_{lag}m"] = v.mean() if len(v) else None
    return out


def slice_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df:
        return pd.DataFrame()
    s = df[df["pnl_copy"].notna()]
    rows_ = []
    for key, g in s.groupby(col, dropna=False):
        n = len(g)
        n_eff = max(1, g["event"].fillna(g["ticker"]).nunique())
        mean = g["copy_c"].mean()
        margin = 1.96 * (g["copy_c"].std(ddof=1) if n > 1 else 50.0) / math.sqrt(n_eff)
        rows_.append({"group": key, "whale trades": n, "markets": n_eff, "win rate": (g["pnl_copy"] > 0).mean(),
                      "avg copy price": g["entry_ask"].mean(), "c_per_contract": mean, "margin": margin,
                      "drift 30m (c)": g["drift_30m_c"].mean(),
                      "verdict": "clearly positive" if mean - margin > 0 else "clearly negative"
                      if mean + margin < 0 else "too early to tell"})
    t = pd.DataFrame(rows_)
    return t.sort_values("c_per_contract", ascending=False).reset_index(drop=True) if not t.empty else t


# ----------------------------------------------------------------------------
# Precomputed report (so the dashboard never loads hundreds of thousands of rows)
# ----------------------------------------------------------------------------

_REPORT_COLS = ("id, ts, ticker, event, series, side, contracts, notional, vwap, prints, typical_count, close_ts, "
                "entry_ask, mid_5m, mid_30m, mid_60m, result, pnl_copy, settled_ts")
LATEST_COLS = ["time_et", "ticker", "title", "side", "contracts", "notional", "vwap", "entry_ask", "drift_30m_c",
               "result", "copy_c"]


def build_report(path: Optional[str] = None) -> Dict[str, Any]:
    from src import edge_scan

    text_cols = ("ticker", "event", "series", "side", "result")
    with closing(_conn(path)) as c:
        total = c.execute("SELECT COUNT(*) FROM whales").fetchone()[0]
        # in chunks, with repeated text stored once: loading it all in one go needed ~4x the memory
        parts = []
        for ch in pd.read_sql(f"SELECT {_REPORT_COLS} FROM whales WHERE pnl_copy IS NOT NULL", c, chunksize=20000):
            parts.append(ch.astype({k: "category" for k in text_cols}))
        settled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_REPORT_COLS.split(", "))
        del parts
        settled = settled.astype({k: object for k in text_cols})
        latest = pd.read_sql("SELECT * FROM whales ORDER BY id DESC LIMIT 200", c)
    meta = series_meta(path)
    df = features(settled, meta)
    sm = summary(df)
    sm["events"] = total
    slices = {lab: slice_table(df, col).to_dict("records") for lab, col in DIMENSIONS.items()}
    if not df.empty:
        df["cluster"] = df["event"].fillna(df["ticker"])
        # every split and pair ('series' alone only: see edge_scan.scan)
        search = edge_scan.scan(df[df["pnl_copy"].notna()], DIMENSIONS, "copy_c", "cluster", min_clusters=50)
    else:
        search = {"tests": 0, "best": pd.DataFrame(), "passing": 0}
    bankroll = group_bankrolls(df, search)
    lf = features(latest, meta)
    lt = lf[LATEST_COLS].to_dict("records") if not lf.empty else []
    return {"summary": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in sm.items()},
            "slices": slices, "search": edge_scan.to_jsonable(search), "latest": lt, "bankroll": bankroll,
            "computed_ts": time.time(),
            "summary_line": f"{total:,} whale trades, {sm.get('settled', 0):,} settled, "
                            f"{search.get('tests', 0)} groups searched, {search.get('passing', 0)} survive"}


def group_bankrolls(df: pd.DataFrame, search: Dict[str, Any]) -> List[Dict[str, Any]]:
    """For every group that survives the split search: what $100 / $500 / $1,000 would be after copying
    each of its whale trades one minute late (same replay as the favorites tab). In-sample: these
    groups were FOUND in this data, so this flatters them by construction."""
    from src import favorites as F
    best = search.get("best")
    if df is None or df.empty or not isinstance(best, pd.DataFrame) or best.empty:
        return []
    out = []
    label_to_col = dict(DIMENSIONS)
    for row in best[best["survives the search"]].itertuples(index=False):
        cols = [label_to_col.get(x) for x in row.split.split(" × ")]
        vals = row.group.split(" × ")
        if None in cols or len(cols) != len(vals):
            continue
        m = df["pnl_copy"].notna()
        for c, v in zip(cols, vals):
            m &= df[c].astype(str) == v
        g = df[m]
        bets = pd.DataFrame({"id": g["id"], "arm": row.group, "ticker": g["ticker"], "side": g["side"],
                             "price": g["entry_ask"].astype(float), "depth": None, "entered_ts": g["ts"] + 60,
                             "settled_ts": g["settled_ts"], "result": g["result"]})
        sims = {b: F.simulate_bankroll(bets, b) for b in F.BANKROLLS}
        out.append({"split": row.split, "group": row.group, "since": float(g["ts"].min()) if len(g) else None,
                    "sims": {str(b): {k: v for k, v in r.items() if k != "curve"} for b, r in sims.items()}})
    return out


def write_report(path: Optional[str] = None, out: Optional[str] = None) -> Dict[str, Any]:
    from src import report_runner
    rep = build_report(path)
    report_runner.write_json(rep, out or REPORT_PATH)
    return rep


def read_report(out: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from src import report_runner
    return report_runner.read_json(out or REPORT_PATH)
