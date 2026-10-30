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
from typing import Any, Dict, List, Optional, Tuple

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
ARMS = {"favorites": "All favorites (90-98c, 6-24h)", "whale": "Only after a whale bought the favorite",
        "fav_93_95": "93-95c favorites only (6-24h)", "fav_1_6h": "Favorites 1-6h before close (90-98c)",
        "fav_econ": "Economics favorites only (90-98c, 6-24h)"}
# each scanned arm's own rules (the whale arm uses the 'favorites' rules plus its whale trigger)
ARM_RULES = {
    "favorites": {"min_price": RULES["min_price"], "max_price": RULES["max_price"],
                  "min_hours": RULES["min_hours"], "max_hours": RULES["max_hours"]},
    # added Oct 27 from the one-week check-in, each judged ONLY on bets from its own start date:
    "fav_93_95": {"min_price": 0.93, "max_price": 0.955, "min_hours": 6.0, "max_hours": 24.0},
    "fav_1_6h": {"min_price": 0.90, "max_price": 0.98, "min_hours": 1.0, "max_hours": 6.0},
    # added Oct 29 from the stress test (Economics carried two-thirds of the profit), fresh data only:
    "fav_econ": {"min_price": 0.90, "max_price": 0.98, "min_hours": 6.0, "max_hours": 24.0,
                 "category": "Economics"},
}
NEW_ARMS = ("fav_93_95", "fav_1_6h", "fav_econ")
ARM_ADDED_BECAUSE = {"fav_93_95": "the one-week check-in", "fav_1_6h": "the one-week check-in",
                     "fav_econ": "the stress test"}


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


def favorite(m: Dict[str, Any], rules: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """The side that is the favorite and buyable within the rules, or None."""
    r = rules or ARM_RULES["favorites"]
    best = None
    for side in ("yes", "no"):
        ask = _f(m, f"{side}_ask_dollars", f"{side}_ask")
        bid = _f(m, f"{side}_bid_dollars", f"{side}_bid")
        if not (r["min_price"] <= ask <= r["max_price"]) or bid <= 0:
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


def in_window(m: Dict[str, Any], now: float, rules: Optional[Dict[str, float]] = None) -> bool:
    r = rules or ARM_RULES["favorites"]
    h = hours_left(m, now)
    return h is not None and r["min_hours"] <= h <= r["max_hours"]


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
    if "book_ask" not in {r[1] for r in c.execute("PRAGMA table_info(bets)")}:
        c.execute("ALTER TABLE bets ADD COLUMN book_ask REAL")      # real best price in the order book at entry
    return c


def _start_key(arm: Optional[str]) -> str:
    return f"started_{arm}" if arm in NEW_ARMS else "started"


def started_ts(path: Optional[str] = None, arm: Optional[str] = None) -> Optional[float]:
    """When the test (or one of the arms added later) started counting bets."""
    with closing(_conn(path)) as c:
        r = c.execute("SELECT v FROM meta WHERE k=?", (_start_key(arm),)).fetchone()
    return float(r[0]) if r else None


def _mark_started(path: Optional[str], now: float) -> None:
    with closing(_conn(path)) as c:
        c.execute("INSERT OR IGNORE INTO meta VALUES ('started', ?)", (str(now),))
        for arm in NEW_ARMS:
            c.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (_start_key(arm), str(now)))
        c.commit()


def _known(path: Optional[str], arm: str) -> set:
    with closing(_conn(path)) as c:
        return {r[0] for r in c.execute("SELECT ticker FROM bets WHERE arm=?", (arm,))}


def record(path: Optional[str], arm: str, m: Dict[str, Any], fav: Dict[str, Any], now: float,
           depth: Optional[float] = None, whale: Optional[Dict[str, Any]] = None,
           book_ask: Optional[float] = None) -> bool:
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
               volume_24h, entered_ts, close_ts, hours_left, whale_notional, whale_price, book_ask)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (arm, m["ticker"], m.get("event_ticker"), m["ticker"].split("-", 1)[0], title, fav["side"], fav["price"],
             fav["bid"], fav["spread"], fee_per_contract(fav["price"]), depth,
             vol, now, ts_of(m.get("close_time")), hours_left(m, now),
             (whale or {}).get("notional"), (whale or {}).get("vwap"), book_ask))
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
        params = {"limit": 1000, "status": "open",
                  "min_close_ts": int(now + min(r["min_hours"] for r in ARM_RULES.values()) * 3600),
                  "max_close_ts": int(now + max(r["max_hours"] for r in ARM_RULES.values()) * 3600)}
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


def best_ask(book: Dict[str, Any], side: str) -> Optional[float]:
    """Cheapest price you could really buy ``side`` at. Kalshi lists bids only: buying YES at p means
    hitting a NO bid at 1 - p (and vice versa). None if unreadable or nothing offered."""
    from src.maker import _book_levels
    other = _book_levels(book or {}, "no" if side == "yes" else "yes")
    if not other:
        return None
    top = max((pr for pr, sz in other if 0 < pr < 1 and sz > 0), default=None)
    return round(1 - top, 4) if top is not None else None


async def _book_info(client, ticker: str, side: str, price: float) -> Tuple[Optional[float], Optional[float]]:
    """(contracts offered at our price, real best price) from one order-book read; (None, None) on failure."""
    from src.arb_scan import depth_at_ask
    try:
        book = await client.get_orderbook(ticker)
    except Exception:
        return None, None
    return depth_at_ask(book, side, price), best_ask(book, side)


async def scan_favorites(client, now: float, path: Optional[str] = None, logger=None,
                         markets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    markets = markets if markets is not None else await fetch_window(client, now)
    known = {arm: _known(path, arm) for arm in ARM_RULES}
    qualifying = new = 0
    new_by_arm = {arm: 0 for arm in ARM_RULES}
    depth_cache: Dict[tuple, Optional[float]] = {}
    book_calls = 0          # order-book lookups are capped per scan so a big first scan can't stall the whale arm
    # categories for the arms limited to one, looked up before deciding (only for markets that qualify)
    need = [m["ticker"].split("-", 1)[0] for m in markets
            if any("category" in r and in_window(m, now, r) and favorite(m, r) for r in ARM_RULES.values())]
    cats = await ensure_categories(client, need, path) if need else series_categories(path)
    for m in markets:
        for arm, rules in ARM_RULES.items():
            if not in_window(m, now, rules):
                continue
            if "category" in rules and cats.get(m["ticker"].split("-", 1)[0]) != rules["category"]:
                continue
            fav = favorite(m, rules)
            if not fav:
                continue
            if arm == "favorites":
                qualifying += 1
            if m["ticker"] in known[arm]:
                continue
            key = (m["ticker"], fav["side"], fav["price"])
            if key not in depth_cache:
                depth_cache[key] = (None, None)
                if book_calls < 300:
                    book_calls += 1
                    depth_cache[key] = await _book_info(client, m["ticker"], fav["side"], fav["price"])
            depth, real_ask = depth_cache[key]
            if record(path, arm, m, fav, now, depth, book_ask=real_ask):
                new_by_arm[arm] += 1
                if arm == "favorites":
                    new += 1
    with closing(_conn(path)) as c:
        c.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?)", (now, len(markets), qualifying))
        c.commit()
    if logger:
        logger.info(f"FAV scan: {len(markets)} markets closing in 1-24h, {qualifying} with a 90-98c favorite 6-24h out; "
                    "new paper buys: " + ", ".join(f"{a} {n}" for a, n in new_by_arm.items()))
    return {"markets": len(markets), "qualifying": qualifying, "new": new, "new_by_arm": new_by_arm}


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
        depth, real_ask = await _book_info(client, m["ticker"], fav["side"], fav["price"])
        if record(path, "whale", m, fav, now, depth, whale=p, book_ask=real_ask):
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


def series_categories(path: Optional[str] = None) -> Dict[str, str]:
    with closing(_conn(path)) as c:
        return dict(c.execute("SELECT series, category FROM series_meta").fetchall())


async def ensure_categories(client, series: List[str], path: Optional[str] = None,
                            limit: int = 80) -> Dict[str, str]:
    """Look up the category of series we haven't seen yet (a few per scan), so an arm limited to one
    category can decide on the spot. Returns {series: category} for everything known."""
    known = series_categories(path)
    todo = [x for x in dict.fromkeys(series) if x not in known][:limit]
    for sr in todo:
        try:
            info = (await client._make_authenticated_request("GET", f"/trade-api/v2/series/{sr}",
                                                              require_auth=False)).get("series") or {}
        except Exception:
            continue                                   # try again next scan
        known[sr] = info.get("category") or "Unknown"
        with closing(_conn(path)) as c:
            c.execute("INSERT OR REPLACE INTO series_meta VALUES (?,?)", (sr, known[sr]))
            c.commit()
    return known


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


def series_family(series: Optional[str]) -> str:
    """One name for markets on the same underlying: KXNASDAQ100U / KXNASDAQ100D / KXNASDAQ100 are all the
    Nasdaq-100, KXBTCD / KXBTC are both bitcoin. Merging too much only widens the margin (the safe side)."""
    s_ = (series or "?").upper()
    for suffix in ("U", "D"):
        if len(s_) > 5 and s_.endswith(suffix):
            s_ = s_[:-1]
    return s_


def stress(g: pd.DataFrame) -> Dict[str, Any]:
    """Ways the result could be flattered, each checked: bets that move together counted as one,
    the best days removed, the worst day, which market types carry it, and a stingier money replay."""
    from src.edge_scan import _cluster_stats
    s = g[g["result"].isin(["yes", "no"])] if g is not None and not g.empty else pd.DataFrame()
    out: Dict[str, Any] = {"settled": len(s)}
    if s.empty or len(s) < 2:
        return out

    def stat(sub, col):
        mean, se, k = _cluster_stats(sub["c"], sub[col])
        m = 1.96 * se if math.isfinite(se) else None
        return {"c_per_contract": mean, "margin": m, "groups": k,
                "low_end": (mean - m) if m is not None else None}

    out["margins"] = [
        {"counted as one": "each Kalshi event (as on the cards above)", **stat(s, "cluster")},
        {"counted as one": "same underlying, same day (e.g. all Nasdaq-100 markets on one day)",
         **stat(s, "cluster_underlying_day")},
        {"counted as one": "same category, same day (e.g. ALL financial markets on one day)",
         **stat(s, "cluster_category_day")},
    ]
    byday = s.groupby("day").agg(pnl_10=("pnl", lambda x: float(x.sum()) * RULES["order_size"]),
                                 bets=("pnl", "size"), losses=("pnl", lambda x: int((x <= 0).sum())))
    byday = byday.sort_values("pnl_10")
    out["days"] = int(len(byday))
    out["days_positive"] = int((byday["pnl_10"] > 0).sum())
    w = byday.iloc[0]
    b_ = byday.iloc[-1]
    out["worst_day"] = {"day": byday.index[0], "pnl_10": float(w["pnl_10"]), "bets": int(w["bets"]),
                        "losses": int(w["losses"])}
    out["best_day"] = {"day": byday.index[-1], "pnl_10": float(b_["pnl_10"]), "bets": int(b_["bets"]),
                       "losses": int(b_["losses"])}
    # Remove the best day(s). Removing 5 of only ~11 days takes out nearly half the sample and is too harsh
    # for a strategy whose bad days are rare and large, so: the best 1 and 2 days, and the best 5 only
    # once there are 25+ days.
    out["without_best"] = []
    for n in (1, 2, 5):
        if n == 5 and len(byday) < 25:
            continue
        if len(byday) <= n:
            continue
        best = list(byday.index[-n:])
        rest = s[~s["day"].isin(best)]
        if len(rest) > 1:
            out["without_best"].append({"days removed": n, "removed": best, **stat(rest, "cluster_underlying_day")})
    cats = []
    total = float(s["pnl"].sum())
    for cat, gg in s.groupby(s["category"].fillna("unknown")):
        st_ = stat(gg, "cluster_underlying_day") if len(gg) > 1 else {"c_per_contract": float(gg["c"].mean()),
                                                                        "margin": None, "groups": 1, "low_end": None}
        cats.append({"category": cat, "settled": len(gg), "win rate": float((gg["pnl"] > 0).mean()),
                     **st_, "share of profit": float(gg["pnl"].sum()) / total if total else None})
    out["categories"] = sorted(cats, key=lambda r: -r["settled"])
    out["bankroll_stingy"] = {b: {k: v for k, v in simulate_bankroll(g, b, unknown_depth=10).items() if k != "curve"}
                              for b in BANKROLLS}
    out["real_price"] = real_price_check(s)
    return out


def real_price_check(s: pd.DataFrame) -> Dict[str, Any]:
    """Was our recorded price actually for sale in Kalshi's order book when we 'bought'? (The market summary
    can lag the book: that is what made the Polymarket gaps look real.) 'Really there' = contracts were offered
    at our price; judged only on bets whose book we read. Also: the result counting only those bets, and the
    result if you had paid the real best price (recorded from Oct 29 on)."""
    from src.edge_scan import _cluster_stats
    if "book_ask" not in s:
        s = s.assign(book_ask=None)
    checked = s[s["depth"].notna()]
    out: Dict[str, Any] = {"settled": len(s), "checked": len(checked)}
    if checked.empty:
        return out
    there = checked[checked["depth"].astype(float) > 0]
    out["really_there"] = len(there) / len(checked)

    def stat(sub, col="c"):
        if len(sub) < 2:
            return {}
        mean, se, k = _cluster_stats(sub[col].astype(float), sub["cluster_underlying_day"])
        m = 1.96 * se if math.isfinite(se) else None
        return {"c_per_contract": mean, "margin": m, "low_end": (mean - m) if m is not None else None, "bets": len(sub)}

    out["only_really_there"] = stat(there)
    real = s[s["book_ask"].notna()].copy()
    out["with_real_price"] = len(real)
    if len(real):
        won = real["result"] == real["side"]
        real["c_real"] = 100 * (won.astype(float) - real["book_ask"].astype(float)
                                - real["book_ask"].astype(float).map(fee_per_contract))
        real["diff_c"] = 100 * (real["book_ask"].astype(float) - real["price"].astype(float))
        out["at_real_price"] = stat(real, "c_real")
        out["median_real_minus_ours_c"] = float(real["diff_c"].median())
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("close_ts", "series", "category"):
        if col not in df:
            df[col] = None
    df["price_bucket"] = df["price"].map(lambda p: _cut(p, [0.93, 0.96], ["90-92c", "93-95c", "96-98c"]))
    df["hours_bucket"] = df["hours_left"].map(lambda h: _cut(h, [12], ["6-12h", "12-24h"]))
    df["depth_bucket"] = df["depth"].map(lambda d: _cut(d, [10, 100, 1000], ["<10", "10-99", "100-999", "1000+"]))
    df["c"] = 100 * df["pnl"].astype(float)
    df["cluster"] = df["event"].fillna(df["ticker"])
    # the day the bet's market closed (Eastern): bets on the same underlying that day move together
    when = df["close_ts"].astype(float).fillna(df["entered_ts"].astype(float))
    df["day"] = pd.to_datetime(when, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    df["family"] = df["series"].map(series_family)
    df["cluster_underlying_day"] = df["family"] + "|" + df["day"]
    df["cluster_category_day"] = df["category"].fillna("unknown").astype(str) + "|" + df["day"]
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


def simulate_bankroll(g: pd.DataFrame, start: float, share: float = BET_SHARE,
                      unknown_depth: Optional[float] = None) -> Dict[str, Any]:
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
            depth = r.depth if pd.notna(r.depth) else unknown_depth     # None = assume no limit
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
    out["arm_started"], out["arm_days"] = {}, {}
    for arm in ARMS:
        g = df[df["arm"] == arm] if not df.empty else df
        a_start = started_ts(path, arm) if arm in NEW_ARMS else start
        a_days = (time.time() - a_start) / 86400 if a_start else 0.0
        out["arm_started"][arm], out["arm_days"][arm] = a_start, a_days
        st = _stats(g) if not g.empty else {"bets": 0, "settled": 0, "events": 0}
        st["verdict"] = verdict(st, a_days)
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
        out.setdefault("stress", {})[arm] = stress(g)
    return out
