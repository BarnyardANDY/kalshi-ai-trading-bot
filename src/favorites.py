"""
Favorites paper test (measurement only: never places an order). Separate from every
other test: own process (``cli.py favorites``), database (``data/favorites.db``), log
(``logs/favorites.log``) and dashboard tab.

The idea came out of the whale study's "search every split": copying whales who
bought at 90c+ with 6-24 hours to go looked positive. Most of the other top groups
were expensive favorites too, which hints the edge may be "favorites are slightly
underpriced" rather than anything about whales. A result found by searching
hundreds of groups must be confirmed on NEW data, so this test only counts trades
from the day it starts, with rules fixed in advance:

* Arm "favorites": every open market closing in 6-24 hours whose favorite side
  can be bought for 90-98c (with a real bid behind it, gap <= 4c). One paper buy
  per market, at the ask, the first time it qualifies.
* Arm "whale": the same rules, but only right after a $1,000+ trade bought that
  favorite at 90c+, bought ONE MINUTE later at the then-current ask.

Each is held to settlement. Fee = Kalshi's taker fee rounded UP to the cent per
order, for a 10-contract order (what a small real test would pay).

Verdict rule, fixed now: after at least 14 days and 300 separate settled events in
an arm, that arm passes if its result per contract is positive with the 95% margin
(counting each event once) entirely above zero. Only then is a tiny live test sensible.
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import time
from contextlib import closing
from typing import Any, Dict, List, Optional

import pandas as pd

DB_PATH = os.getenv("FAV_DB", "data/favorites.db")
TZ = os.getenv("DASHBOARD_TZ", "America/New_York")

RULES = {
    "min_price": float(os.getenv("FAV_MIN_PRICE", "0.90")),
    "max_price": float(os.getenv("FAV_MAX_PRICE", "0.98")),
    "min_hours": float(os.getenv("FAV_MIN_HOURS", "6")),
    "max_hours": float(os.getenv("FAV_MAX_HOURS", "24")),
    "max_spread": 0.04,
    "order_size": 10,
    "whale_min_notional": 1000.0,
    "whale_delay_s": 60,
}
VERDICT = {"min_days": 14, "min_events": 300}
SCAN_SECONDS = 600
EXCLUDE_PREFIXES = ("KXMVE",)
ARMS = {"favorites": "All favorites (90-98c, 6-24h)", "whale": "Only after a whale bought the favorite"}


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def fee_per_contract(price: float, qty: int = RULES["order_size"]) -> float:
    """Kalshi taker fee for one order of ``qty`` contracts, rounded up to the cent, per contract."""
    if not 0 < price < 1 or qty <= 0:
        return 0.0
    raw = 0.07 * qty * price * (1 - price)
    return math.ceil(round(raw * 100, 6)) / 100 / qty


def pnl_per_contract(price: float, won: bool) -> float:
    return (1.0 if won else 0.0) - price - fee_per_contract(price)


def _f(m: Dict[str, Any], dollars: str, cents: str) -> float:
    v = m.get(dollars)
    if v not in (None, ""):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = m.get(cents)
    try:
        return float(v) / 100 if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def favorite(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The side that is the favorite and buyable within the rules, or None."""
    best = None
    for side in ("yes", "no"):
        ask = _f(m, f"{side}_ask_dollars", f"{side}_ask")
        bid = _f(m, f"{side}_bid_dollars", f"{side}_bid")
        if not (RULES["min_price"] <= ask <= RULES["max_price"]) or bid <= 0:
            continue
        spread = round(ask - bid, 4)
        if spread > RULES["max_spread"]:
            continue
        if best is None or ask > best["price"]:
            best = {"side": side, "price": ask, "bid": bid, "spread": spread}
    return best


def hours_left(m: Dict[str, Any], now: float) -> Optional[float]:
    from src.whale import ts_of
    c = ts_of(m.get("close_time"))
    return (c - now) / 3600 if c else None


def in_window(m: Dict[str, Any], now: float) -> bool:
    h = hours_left(m, now)
    return h is not None and RULES["min_hours"] <= h <= RULES["max_hours"]


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS bets (id INTEGER PRIMARY KEY AUTOINCREMENT, arm TEXT, ticker TEXT,
                 event TEXT, series TEXT, title TEXT, side TEXT, price REAL, bid REAL, spread REAL, fee REAL,
                 depth REAL, volume_24h REAL, entered_ts REAL, close_ts REAL, hours_left REAL,
                 whale_notional REAL, whale_price REAL, result TEXT, pnl REAL, settled_ts REAL,
                 UNIQUE (arm, ticker))""")
    c.execute("CREATE TABLE IF NOT EXISTS series_meta (series TEXT PRIMARY KEY, category TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS scans (ts REAL PRIMARY KEY, markets INTEGER, qualifying INTEGER)")
    return c


def started_ts(path: Optional[str] = None) -> Optional[float]:
    with closing(_conn(path)) as c:
        r = c.execute("SELECT v FROM meta WHERE k='started'").fetchone()
    return float(r[0]) if r else None


def _mark_started(path: Optional[str], now: float) -> None:
    with closing(_conn(path)) as c:
        c.execute("INSERT OR IGNORE INTO meta VALUES ('started', ?)", (str(now),))
        c.commit()


def _known(path: Optional[str], arm: str) -> set:
    with closing(_conn(path)) as c:
        return {r[0] for r in c.execute("SELECT ticker FROM bets WHERE arm=?", (arm,))}


def record(path: Optional[str], arm: str, m: Dict[str, Any], fav: Dict[str, Any], now: float,
           depth: Optional[float] = None, whale: Optional[Dict[str, Any]] = None) -> bool:
    from src.whale import ts_of
    title = " — ".join(x for x in (m.get("title"), m.get("yes_sub_title")) if x)[:200]
    vol = None
    for k in ("volume_24h_fp", "volume_24h"):
        try:
            vol = float(m[k])
            break
        except (KeyError, TypeError, ValueError):
            continue
    with closing(_conn(path)) as c:
        cur = c.execute(
            """INSERT OR IGNORE INTO bets (arm, ticker, event, series, title, side, price, bid, spread, fee, depth,
               volume_24h, entered_ts, close_ts, hours_left, whale_notional, whale_price)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (arm, m["ticker"], m.get("event_ticker"), m["ticker"].split("-", 1)[0], title, fav["side"], fav["price"],
             fav["bid"], fav["spread"], fee_per_contract(fav["price"]), depth,
             vol, now, ts_of(m.get("close_time")), hours_left(m, now),
             (whale or {}).get("notional"), (whale or {}).get("vwap")))
        c.commit()
        return cur.rowcount > 0


def rows(path: Optional[str] = None) -> pd.DataFrame:
    with closing(_conn(path)) as c:
        df = pd.read_sql("SELECT * FROM bets", c)
        meta = dict(c.execute("SELECT series, category FROM series_meta").fetchall())
    df["category"] = df["series"].map(lambda s: meta.get(s) or "unknown") if not df.empty else []
    return df


# ----------------------------------------------------------------------------
# Network (public endpoints; read-only -- this module never trades)
# ----------------------------------------------------------------------------

async def fetch_window(client, now: float, max_pages: int = 30) -> List[Dict[str, Any]]:
    from src.maker import slim
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"limit": 1000, "status": "open", "min_close_ts": int(now + RULES["min_hours"] * 3600),
                  "max_close_ts": int(now + RULES["max_hours"] * 3600)}
        if cursor:
            params["cursor"] = cursor
        resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets", params=params,
                                                         require_auth=False)
        page = resp.get("markets") or []
        cursor = resp.get("cursor")
        out.extend(slim(m) for m in page if not (m.get("ticker") or "").startswith(EXCLUDE_PREFIXES))
        del resp
        if not cursor or len(page) < 1000:
            break
    return out


async def _depth(client, ticker: str, side: str, price: float) -> Optional[float]:
    from src.arb_scan import depth_at_ask
    try:
        return depth_at_ask(await client.get_orderbook(ticker), side, price)
    except Exception:
        return None


async def scan_favorites(client, now: float, path: Optional[str] = None, logger=None,
                         markets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    markets = markets if markets is not None else await fetch_window(client, now)
    known = _known(path, "favorites")
    qualifying = new = 0
    book_calls = 0          # order-book lookups are capped per scan so a big first scan can't stall the whale arm
    for m in markets:
        if not in_window(m, now):
            continue
        fav = favorite(m)
        if not fav:
            continue
        qualifying += 1
        if m["ticker"] in known:
            continue
        depth = None
        if book_calls < 300:
            book_calls += 1
            depth = await _depth(client, m["ticker"], fav["side"], fav["price"])
        if record(path, "favorites", m, fav, now, depth):
            new += 1
    with closing(_conn(path)) as c:
        c.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?)", (now, len(markets), qualifying))
        c.commit()
    if logger:
        logger.info(f"FAV scan: {len(markets)} markets closing in 6-24h, {qualifying} with a 90-98c favorite, "
                    f"{new} new paper buys")
    return {"markets": len(markets), "qualifying": qualifying, "new": new}


async def whale_entries(client, pending: List[Dict[str, Any]], now: float, path: Optional[str] = None,
                        logger=None) -> List[Dict[str, Any]]:
    """Buy (on paper) the favorite a whale bought, one minute after the whale. Returns still-waiting ones."""
    from src.whale import _markets
    due = [p for p in pending if now - p["ts"] >= RULES["whale_delay_s"]]
    wait = [p for p in pending if now - p["ts"] < RULES["whale_delay_s"]]
    if not due:
        return wait
    known = _known(path, "whale")
    due = [p for p in due if p["ticker"] not in known]
    try:
        got = await _markets(client, sorted({p["ticker"] for p in due})) if due else {}
    except Exception as e:
        if logger:
            logger.warning(f"FAV whale lookup failed: {e}")
        return wait
    n = 0
    for p in due:
        m = got.get(p["ticker"])
        if not m or not in_window(m, now):
            continue
        fav = favorite(m)
        if not fav or fav["side"] != p["side"]:
            continue                                  # the whale's side is no longer a buyable 90-98c favorite
        depth = await _depth(client, m["ticker"], fav["side"], fav["price"])
        if record(path, "whale", m, fav, now, depth, whale=p):
            known.add(p["ticker"])
            n += 1
    if n and logger:
        logger.info(f"FAV whale arm: {n} new paper buys after whale trades")
    return wait


async def settle(client, now: float, path: Optional[str] = None, logger=None) -> int:
    from src.whale import _markets
    with closing(_conn(path)) as c:
        todo = c.execute("SELECT id, ticker, side, price, close_ts FROM bets WHERE result IS NULL").fetchall()
    if not todo:
        return 0
    try:
        got = await _markets(client, sorted({t for _, t, _, _, _ in todo}))
    except Exception as e:
        if logger:
            logger.warning(f"FAV settle lookup failed: {e}")
        return 0
    n = 0
    with closing(_conn(path)) as c:
        for bid_, ticker, side, price, close_ts in todo:
            res = ((got.get(ticker) or {}).get("result") or "").lower()
            if res in ("yes", "no"):
                c.execute("UPDATE bets SET result=?, pnl=?, settled_ts=? WHERE id=?",
                          (res, pnl_per_contract(price, res == side), now, bid_))
                n += 1
            elif close_ts and now - close_ts > 7 * 86400:
                c.execute("UPDATE bets SET result='unresolved', settled_ts=? WHERE id=?", (now, bid_))
        c.commit()
    if n and logger:
        logger.info(f"FAV {n} paper bets settled")
    return n


async def categories(client, path: Optional[str] = None, per_call: int = 20) -> None:
    with closing(_conn(path)) as c:
        known = {r[0] for r in c.execute("SELECT series FROM series_meta")}
        todo = [r[0] for r in c.execute("SELECT DISTINCT series FROM bets") if r[0] not in known][:per_call]
    for sr in todo:
        try:
            info = (await client._make_authenticated_request("GET", f"/trade-api/v2/series/{sr}",
                                                              require_auth=False)).get("series") or {}
        except Exception:
            info = {}
        with closing(_conn(path)) as c:
            c.execute("INSERT OR REPLACE INTO series_meta VALUES (?,?)", (sr, info.get("category") or "Unknown"))
            c.commit()


async def run_forever(logger=None, path: Optional[str] = None, tape_interval: float = 5.0) -> None:
    from src.clients.kalshi_client import KalshiClient
    from src.whale import Tape

    client = KalshiClient()
    tape = Tape(RULES["whale_min_notional"])
    pending: List[Dict[str, Any]] = []
    last_scan = last_settle = 0.0
    _mark_started(path, time.time())
    try:
        while True:
            t0 = time.time()
            try:
                evs = tape.events(await tape.poll(client), time.time())
                for e in evs:
                    if e["vwap"] >= RULES["min_price"]:
                        pending.append(e)
                pending = await whale_entries(client, pending, time.time(), path, logger)
                if time.time() - last_scan > SCAN_SECONDS:
                    last_scan = time.time()
                    await scan_favorites(client, time.time(), path, logger)
                if time.time() - last_settle > 600:
                    last_settle = time.time()
                    await settle(client, time.time(), path, logger)
                    await categories(client, path)
            except Exception as e:
                if logger:
                    logger.warning(f"FAV cycle failed: {e}")
            await asyncio.sleep(max(0.5, tape_interval - (time.time() - t0)))
    finally:
        await client.close()


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _cut(v, edges, labels):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


SPLITS = {
    "Price paid": "price_bucket",
    "Time of day bought (Eastern)": "daypart",
    "Category": "category",
    "Hours until close when bought": "hours_bucket",
    "Side bought": "side",
    "Buyable size at our price": "depth_bucket",
}


def features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["price_bucket"] = df["price"].map(lambda p: _cut(p, [0.93, 0.96], ["90-92c", "93-95c", "96-98c"]))
    df["hours_bucket"] = df["hours_left"].map(lambda h: _cut(h, [12], ["6-12h", "12-24h"]))
    df["depth_bucket"] = df["depth"].map(lambda d: _cut(d, [10, 100, 1000], ["<10", "10-99", "100-999", "1000+"]))
    df["c"] = 100 * df["pnl"].astype(float)
    df["cluster"] = df["event"].fillna(df["ticker"])
    t = pd.to_datetime(df["entered_ts"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    df["entered_et"] = t
    # same buckets as the whale study, so 'evening' means the same thing in both
    df["daypart"] = t.dt.hour.map(lambda h: "night (0-6)" if h < 6 else "morning (6-12)" if h < 12
                                  else "afternoon (12-18)" if h < 18 else "evening (18-24)")
    df["settled_et"] = pd.to_datetime(df["settled_ts"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    return df


def _stats(g: pd.DataFrame) -> Dict[str, Any]:
    from src.edge_scan import _cluster_stats
    s = g[g["result"].isin(["yes", "no"])]
    out: Dict[str, Any] = {"bets": len(g), "settled": len(s), "events": int(s["cluster"].nunique()) if len(s) else 0}
    if s.empty:
        return out
    mean, se, _ = _cluster_stats(s["c"], s["cluster"])
    won = s["pnl"] > 0
    avg_price = float(s["price"].mean())
    out.update(win_rate=float(won.mean()), avg_price=avg_price,
               breakeven=float((s["price"] + s["fee"]).mean()), losses=int((~won).sum()),
               c_per_contract=mean, margin=1.96 * se if math.isfinite(se) else None,
               pnl_10=float(s["pnl"].sum() * RULES["order_size"]))
    return out


def verdict(st: Dict[str, Any], days: float) -> str:
    if st.get("settled", 0) == 0:
        return "waiting for results"
    m, c = st.get("margin"), st.get("c_per_contract")
    enough = days >= VERDICT["min_days"] and st["events"] >= VERDICT["min_events"]
    if m is not None and c + m < 0:
        return "losing (clearly)" if enough else "losing so far"
    if not enough:
        return (f"too early: {st['events']} of {VERDICT['min_events']} events, "
                f"day {days:.0f} of {VERDICT['min_days']}")
    if m is not None and c - m > 0:
        return "PASSED: worth a tiny live test"
    return "no edge (positive margin not reached)"


BANKROLLS = (100, 500, 1000)
BET_SHARE = 0.05          # each bet uses up to 5% of the STARTING bankroll (no compounding)


def simulate_bankroll(g: pd.DataFrame, start: float, share: float = BET_SHARE) -> Dict[str, Any]:
    """Replay one arm's paper bets in time order with ``start`` dollars.

    Each bet buys as many contracts as ``share`` of the starting bankroll pays for, capped by
    the contracts actually offered at our price (when known) and by the cash on hand; Kalshi's
    fee is charged for that order size. Cash comes back when the market settles; a bet with no
    cash free is skipped. Open bets count at what they cost."""
    out: Dict[str, Any] = {"start": start, "value": start, "pnl": 0.0, "ret": 0.0, "max_drop": 0.0,
                           "taken": 0, "skipped": 0, "capped": 0, "open": 0, "contracts_per_bet": 0, "curve": []}
    if g is None or g.empty:
        return out
    evs = []
    for r in g.itertuples(index=False):
        evs.append((float(r.entered_ts), 0, r))
        if r.result in ("yes", "no", "unresolved") and pd.notna(r.settled_ts):
            evs.append((float(r.settled_ts), 1, r))
    evs.sort(key=lambda e: (e[0], -e[1]))          # same moment: settle first, freeing cash
    cash, held, peak, max_drop, sizes = float(start), {}, float(start), 0.0, []
    curve = []
    for ts, kind, r in evs:
        key = getattr(r, "id", None) or (r.arm, r.ticker)     # whale copies repeat tickers: use the row id
        if kind == 0:
            per = r.price + fee_per_contract(r.price, 1)
            n = int(share * start // per)
            depth = r.depth if pd.notna(r.depth) else None
            if depth is not None and n > depth:
                n = int(depth)
                out["capped"] += 1
            while n > 0 and n * r.price + fee_per_contract(r.price, n) * n > cash:
                n -= 1
            if n <= 0:
                out["skipped"] += 1
                continue
            cost = n * r.price + fee_per_contract(r.price, n) * n
            cash -= cost
            held[key] = (n, cost)
            sizes.append(n)
            out["taken"] += 1
        elif key in held:
            n, cost = held.pop(key)
            if r.result == "unresolved":
                cash += cost                           # never resolved: treat as money back
            elif r.result == r.side:
                cash += n * 1.0
        value = cash + sum(c for _, c in held.values())
        peak = max(peak, value)
        max_drop = max(max_drop, peak - value)
        curve.append((ts, value))
    value = cash + sum(c for _, c in held.values())
    out.update(value=value, pnl=value - start, ret=(value - start) / start, max_drop=max_drop,
               open=len(held), contracts_per_bet=(sum(sizes) / len(sizes)) if sizes else 0, curve=curve)
    return out


def report(path: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.exists(path or DB_PATH):
        return {"started": None}
    df = features(rows(path))
    start = started_ts(path)
    days = (time.time() - start) / 86400 if start else 0.0
    with closing(_conn(path)) as c:
        sc = c.execute("SELECT COUNT(*), AVG(markets), AVG(qualifying) FROM scans").fetchone()
    out: Dict[str, Any] = {"started": start, "days": days, "scans": sc[0], "markets_per_scan": sc[1] or 0,
                           "qualifying_per_scan": sc[2] or 0, "arms": {}, "splits": {}, "df": df}
    for arm in ARMS:
        g = df[df["arm"] == arm] if not df.empty else df
        st = _stats(g) if not g.empty else {"bets": 0, "settled": 0, "events": 0}
        st["verdict"] = verdict(st, days)
        out["arms"][arm] = st
        sp = {}
        if not g.empty:
            for lab, col in SPLITS.items():
                rows_ = []
                for key, gg in g.groupby(col):
                    r = _stats(gg)
                    if r.get("settled"):
                        rows_.append({"group": key, **{k: r.get(k) for k in ("settled", "events", "win_rate",
                                                                             "avg_price", "c_per_contract", "margin")}})
                sp[lab] = pd.DataFrame(rows_)
        out["splits"][arm] = sp
        out.setdefault("bankroll", {})[arm] = {b: simulate_bankroll(g, b) for b in BANKROLLS}
    return out
