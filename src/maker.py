"""
Maker test: the "patient favorite buyer" (paper only).

Research on every Kalshi trade (Bürgi, Deng & Whelan, 2025) found that people
who TAKE prices (buy at the ask) lose about 32% on average, people who MAKE
prices (post an order and wait) lose about 10%, cheap long shots are badly
overpriced, and makers buying favourites (>= 50c) earned roughly +2-2.6%.
Our own record agreed: long shots lost in every niche, and buying at the ask
lost while selling into other people's bids made money.

So this strategy does the opposite of the old bot:

1. Universe: open markets that settle within ``MAKER_MAX_HOURS`` (any
   category), with some trading activity.
2. Only the FAVOURITE side, priced ``MAKER_MIN_PRICE``-``MAKER_MAX_PRICE``.
3. Never pay the ask: post a buy at the best bid (one cent better when the
   spread is wide) and wait for a seller to come to us.
4. Spread out: one order per event, a few per series, a cap on open orders,
   a small fixed size each.
5. Honest paper fills, from Kalshi's public trade tape: an order only fills
   if, after we posted, someone sold BELOW our price (they would have hit us
   first) or enough contracts traded AT our price to clear the queue that was
   ahead of us plus our own size.
6. Hold to settlement. Charge the maker fee on every fill (conservative: many
   markets charge makers nothing).

Version 2 (``MAKER_VARIANT=v2``, own database ``data/maker_v2.db``) answers what
version 1's record showed: most fills happened because the price dropped through
our bid during live games (we were picked off), and many bets piled onto the
same match. V2 therefore:

* takes at most ONE bet per match (all of a match's markets share the middle
  part of the event ticker, e.g. 26OCT16MEMTULN),
* only quotes BEFORE the event: the market's expected end must be at least
  4 hours away, and resting orders are pulled once it is under 3 hours,
* only quotes where the bid/ask gap is 3c+, so it is always first in line
  (never joins a 1,000-contract queue in a busy market),
* pulls an order as soon as our side's price falls 2c below where it was when
  posted, or the best bid drops below our price,
* checks every 15 seconds instead of every minute.

Nothing in this module can place a real order: it never calls place_order.
Everything is stored in ``data/maker.db`` (v1) or ``data/maker_v2.db`` (v2).
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_PATH = os.getenv("MAKER_DB", "data/maker.db")   # version 1 (kept for compatibility)


def variant() -> str:
    """Which version this process runs: 'v1' (original) or 'v2'."""
    return (os.getenv("MAKER_VARIANT") or "v1").lower()


def db_path(v: Optional[str] = None) -> str:
    v = (v or variant()).lower()
    if v == "v2":
        return os.getenv("MAKER_V2_DB", "data/maker_v2.db")
    return os.getenv("MAKER_DB", "data/maker.db")


def match_key(event: str) -> str:
    """Shared id of a game across its many Kalshi events (winner, totals, maps, ...)."""
    parts = (event or "").split("-")
    return parts[1] if len(parts) > 1 else (event or "")
MAKER_FEE_RATE = 0.0175          # Kalshi maker fee where charged (quarter of the 7% taker rate)
EXCLUDE_PREFIXES = ("KXMVE",)
_last_full_settle = 0.0    # multi-leg combo markets: illiquid, not real single bets


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

def _rc(key: str):
    from src import runtime_config
    return runtime_config.get(key)


def settings() -> Dict[str, Any]:
    return {
        "enabled": bool(_rc("MAKER_ENABLED")),
        "min_price": float(_rc("MAKER_MIN_PRICE")),
        "max_price": float(_rc("MAKER_MAX_PRICE")),
        "max_hours": float(_rc("MAKER_MAX_HOURS")),
        "min_minutes": 30.0,
        "contracts": int(_rc("MAKER_CONTRACTS")),
        "max_open": int(_rc("MAKER_MAX_OPEN")),
        "order_minutes": float(_rc("MAKER_ORDER_MINUTES")),
        "max_per_series": int(_rc("MAKER_MAX_PER_SERIES")),
        "min_volume_24h": float(_rc("MAKER_MIN_VOLUME_24H")),
        **(V2_RULES if variant() == "v2" else {"v2": False}),
    }


V2_RULES = {
    "v2": True,
    "min_spread": 0.03,          # always first in line
    "pre_event_hours": 4.0,      # expected end at least this far away to post
    "pull_within_hours": 3.0,    # pull resting orders once the expected end is this close
    "pull_move": 0.02,           # pull if our side's mid falls this much below where it was
    "interval": 15,              # seconds between checks
}


# ----------------------------------------------------------------------------
# Small parsers (pure)
# ----------------------------------------------------------------------------

def _num(d: Dict[str, Any], *keys: str, cents: Tuple[str, ...] = ()) -> float:
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


def quotes(m: Dict[str, Any]) -> Dict[str, float]:
    return {
        "yes_bid": _num(m, "yes_bid_dollars", cents=("yes_bid",)),
        "yes_ask": _num(m, "yes_ask_dollars", cents=("yes_ask",)),
        "no_bid": _num(m, "no_bid_dollars", cents=("no_bid",)),
        "no_ask": _num(m, "no_ask_dollars", cents=("no_ask",)),
    }


def volume_24h(m: Dict[str, Any]) -> float:
    return _num(m, "volume_24h_fp", "volume_24h")


def ts_of(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def series_of(ticker: str) -> str:
    return (ticker or "").split("-", 1)[0].upper()


def maker_fee(price: float, contracts: int) -> float:
    if contracts <= 0 or not 0 < price < 1:
        return 0.0
    return math.ceil(round(MAKER_FEE_RATE * contracts * price * (1 - price) * 100, 6)) / 100


def quote_for(m: Dict[str, Any], cfg: Dict[str, Any], now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The order we'd post on this market, or None if it doesn't qualify."""
    now = now or time.time()
    t = m.get("ticker") or ""
    if not t or t.startswith(EXCLUDE_PREFIXES):
        return None
    if (m.get("status") or "active") not in ("active", "open"):
        return None
    close = ts_of(m.get("close_time"))
    if close is None:
        return None
    hours = (close - now) / 3600
    if hours * 60 < cfg["min_minutes"] or hours > cfg["max_hours"]:
        return None
    if volume_24h(m) < cfg["min_volume_24h"]:
        return None
    q = quotes(m)
    yes_mid = (q["yes_bid"] + q["yes_ask"]) / 2 if q["yes_bid"] > 0 and q["yes_ask"] > 0 else 0
    if yes_mid <= 0:
        return None
    side = "yes" if yes_mid >= 0.5 else "no"
    bid, ask = (q["yes_bid"], q["yes_ask"]) if side == "yes" else (q["no_bid"], q["no_ask"])
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    spread = round(ask - bid, 2)
    improve = spread >= 0.03
    price = round(bid + (0.01 if improve else 0.0), 2)
    if not cfg["min_price"] <= price <= cfg["max_price"]:
        return None
    if cfg.get("v2"):
        if spread < cfg["min_spread"] - 1e-9:
            return None
        end = ts_of(m.get("expected_expiration_time")) or close
        if (end - now) / 3600 < cfg["pre_event_hours"]:
            return None
    return {"ticker": t, "event": m.get("event_ticker") or t.rsplit("-", 1)[0], "series": series_of(t),
            "side": side, "price": price, "improved": improve, "spread": spread, "close_ts": close,
            "title": " — ".join(x for x in (m.get("title"), m.get("yes_sub_title")) if x)[:200],
            "volume_24h": volume_24h(m), **snapshot(m, side)}


def snapshot(m: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Market state when we posted, kept for slicing results later."""
    q = quotes(m)
    bid, ask = (q["yes_bid"], q["yes_ask"]) if side == "yes" else (q["no_bid"], q["no_ask"])
    last = _num(m, "last_price_dollars", cents=("last_price",))
    prev = _num(m, "previous_price_dollars", cents=("previous_price",))
    if side == "no":
        last = 1 - last if last else 0.0
        prev = 1 - prev if prev else 0.0
    return {"yes_bid": q["yes_bid"], "yes_ask": q["yes_ask"],
            "side_mid_post": round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else None,
            "side_last": last or None, "side_prev_day": prev or None,
            "open_interest": _num(m, "open_interest_fp", "open_interest") or None,
            "liquidity": _num(m, "liquidity_dollars", cents=("liquidity",)) or None,
            "expected_exp_ts": ts_of(m.get("expected_expiration_time")),
            "open_ts": ts_of(m.get("open_time"))}


def side_mid(m: Dict[str, Any], side: str) -> Optional[float]:
    q = quotes(m)
    bid, ask = (q["yes_bid"], q["yes_ask"]) if side == "yes" else (q["no_bid"], q["no_ask"])
    return round((bid + ask) / 2, 4) if bid > 0 and ask > 0 else None


def _book_levels(orderbook: Dict[str, Any], side: str) -> Optional[List[Tuple[float, float]]]:
    """(price in dollars, contracts) bid levels for ``side``, or None if the book can't be read.

    Accepts the shapes Kalshi has used: {"orderbook": {...}} or {"orderbook_fp": {...}},
    with levels under "<side>_dollars" ([["0.70", "120"], ...]) or "<side>" ([[70, 120], ...]).
    """
    ob = None
    for key in ("orderbook_fp", "orderbook"):
        if isinstance(orderbook.get(key), dict):
            ob = orderbook[key]
            break
    if ob is None:
        ob = orderbook
    for key, scale in ((f"{side}_dollars", 1.0), (f"{side}_dollars_fp", 1.0), (side, 0.01)):
        if key not in ob:
            continue
        levels = ob.get(key) or []   # an empty / null list is a real empty book
        out = []
        for lvl in levels:
            try:
                out.append((float(lvl[0]) * scale, float(lvl[1])))
            except (TypeError, ValueError, IndexError):
                return None
        return out
    return None


def queue_ahead(orderbook: Dict[str, Any], side: str, price: float) -> Optional[float]:
    """Contracts already bidding at ``price`` on ``side`` (they fill before us).
    None if the book's format isn't recognised (the caller then skips the order:
    assuming nobody is ahead would make fills look easier than they are)."""
    levels = _book_levels(orderbook, side)
    if levels is None:
        return None
    return sum(q for p, q in levels if abs(p - price) < 0.0005)


def _trade_side_price(tr: Dict[str, Any], side: str) -> Optional[float]:
    """Price on OUR side at which a seller of our side traded, or None if the trade
    wasn't someone selling our side.

    A resting YES bid is hit by a taker buying NO (taker_side == 'no'); a resting NO bid
    by a taker buying YES. The price on our side is that trade's yes/no price.
    """
    taker = (tr.get("taker_side") or "").lower()
    if (side == "yes" and taker != "no") or (side == "no" and taker != "yes"):
        return None
    p = _num(tr, f"{side}_price_dollars", cents=(f"{side}_price",))
    return p if p > 0 else None


def check_fill(trades: Iterable[Dict[str, Any]], side: str, price: float, qty: int,
               ahead: float, posted_ts: float, ticker: Optional[str] = None) -> Optional[float]:
    r = fill_detail(trades, side, price, qty, ahead, posted_ts, ticker)
    return r[0] if r else None


def fill_detail(trades: Iterable[Dict[str, Any]], side: str, price: float, qty: int,
                ahead: float, posted_ts: float, ticker: Optional[str] = None) -> Optional[Tuple[float, str, str, float]]:
    """Timestamp at which a resting buy would have filled, or None.

    Fills if someone sold our side BELOW our price after we posted (they'd have
    hit our better bid first), or enough sold AT our price to clear the queue
    ahead of us plus our own size. Conservative: assumes nobody ahead cancelled.
    """
    at_price = 0.0
    for tr in sorted(trades, key=lambda x: ts_of(x.get("created_time")) or 0):
        if ticker and tr.get("ticker") and tr["ticker"] != ticker:
            continue  # someone else's market
        t = ts_of(tr.get("created_time"))
        if t is None or t <= posted_ts:
            continue
        p = _trade_side_price(tr, side)
        if p is None:
            continue
        n = _num(tr, "count_fp", "count")
        if p < price - 0.0005:
            return t, f"sold through at {p:.2f}", "through", p
        if abs(p - price) < 0.0005:
            at_price += n
            if at_price >= ahead + qty:
                return t, f"{at_price:.0f} sold at our price ({ahead:.0f} ahead)", "queue", p
    return None


def settle_pnl(side: str, price: float, qty: int, fee: float, result: str) -> Optional[float]:
    r = (result or "").lower()
    if r in ("yes", "no"):
        payout = qty if r == side else 0
        return round(payout - qty * price - fee, 4)
    if r in ("void", "voided", "cancelled", "canceled"):
        return 0.0
    return None


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or db_path()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, event TEXT, series TEXT, title TEXT, side TEXT,
        price REAL, qty INTEGER, queue_ahead REAL, improved INTEGER, spread REAL,
        posted_ts REAL, close_ts REAL, status TEXT,           -- open / filled / cancelled / settled
        fill_ts REAL, fee REAL, result TEXT, pnl REAL, settled_ts REAL, note TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
    have = {r[1] for r in c.execute("PRAGMA table_info(orders)")}
    for col, typ in EXTRA_COLUMNS:
        if col not in have:
            c.execute(f"ALTER TABLE orders ADD COLUMN {col} {typ}")
    c.execute("""CREATE TABLE IF NOT EXISTS series_meta (series TEXT PRIMARY KEY, category TEXT,
                 title TEXT, fetched_ts REAL)""")
    return c


# Extra facts kept per order for slicing results (added to older databases automatically).
EXTRA_COLUMNS = [
    ("yes_bid", "REAL"), ("yes_ask", "REAL"), ("side_mid_post", "REAL"), ("side_last", "REAL"),
    ("side_prev_day", "REAL"), ("volume_24h", "REAL"), ("open_interest", "REAL"), ("liquidity", "REAL"),
    ("expected_exp_ts", "REAL"), ("open_ts", "REAL"),
    ("fill_kind", "TEXT"), ("fill_trade_price", "REAL"),
    ("mid_5m", "REAL"), ("mid_30m", "REAL"),   # our side's midpoint 5 / 30 min after the fill
]


def rows(status: Optional[str] = None, path: Optional[str] = None) -> List[Dict[str, Any]]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        q = "SELECT * FROM orders" + (" WHERE status=?" if status else "") + " ORDER BY posted_ts"
        return [dict(r) for r in c.execute(q, (status,) if status else ())]


_SNAPSHOT_KEYS = ("yes_bid", "yes_ask", "side_mid_post", "side_last", "side_prev_day", "volume_24h",
                  "open_interest", "liquidity", "expected_exp_ts", "open_ts")


def _insert(o: Dict[str, Any], ahead: float, now: float, path: Optional[str] = None) -> None:
    extra = [k for k in _SNAPSHOT_KEYS if k in o]
    cols = ["ticker", "event", "series", "title", "side", "price", "qty", "queue_ahead", "improved", "spread",
            "posted_ts", "close_ts", "status", *extra]
    vals = [o["ticker"], o["event"], o["series"], o["title"], o["side"], o["price"], o["qty"], ahead,
            int(o["improved"]), o["spread"], now, o["close_ts"], "open", *[o[k] for k in extra]]
    with closing(_conn(path)) as c:
        c.execute(f"INSERT INTO orders ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        c.commit()


def series_meta(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    with closing(_conn(path)) as c:
        return {r[0]: {"category": r[1], "title": r[2]} for r in c.execute("SELECT series, category, title FROM series_meta")}


def _update(oid: int, path: Optional[str] = None, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn(path)) as c:
        c.execute(f"UPDATE orders SET {keys} WHERE id=?", (*fields.values(), oid))
        c.commit()


# ----------------------------------------------------------------------------
# Network (Kalshi public endpoints; read-only)
# ----------------------------------------------------------------------------

async def fetch_universe(client, cfg: Dict[str, Any], now: Optional[float] = None,
                         max_pages: int = 25) -> List[Dict[str, Any]]:
    now = now or time.time()
    out, cursor = [], None
    # Kalshi lists the soonest-closing markets first, so v2 (which only wants markets whose event
    # is 4h+ away) must start its window there, or the page cap fills up with games already on.
    earliest = cfg["min_minutes"] * 60
    if cfg.get("v2"):
        earliest = max(earliest, cfg["pre_event_hours"] * 3600)
    for _ in range(max_pages):
        params = {"limit": 1000, "status": "open",
                  "min_close_ts": int(now + earliest),
                  "max_close_ts": int(now + cfg["max_hours"] * 3600)}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets", params=params,
                                                             require_auth=False)
        except Exception:
            if "min_close_ts" not in params:
                raise
            # Older API: no close-time filter. Fetch open markets; quote_for() filters by time.
            params.pop("min_close_ts"), params.pop("max_close_ts")
            resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets", params=params,
                                                             require_auth=False)
        page = resp.get("markets") or []
        out.extend(page)
        cursor = resp.get("cursor")
        if not cursor or len(page) < 1000:
            break
    return out


async def fetch_trades(client, ticker: str, since_ts: float, max_pages: int = 5) -> List[Dict[str, Any]]:
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"ticker": ticker, "min_ts": int(since_ts) - 1, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = await client._make_authenticated_request("GET", "/trade-api/v2/markets/trades", params=params,
                                                         require_auth=False)
        page = resp.get("trades") or []
        out.extend(page)
        cursor = resp.get("cursor")
        if not cursor or len(page) < 1000:
            break
    return out


# ----------------------------------------------------------------------------
# One cycle
# ----------------------------------------------------------------------------

def pick(candidates: List[Dict[str, Any]], open_rows: List[Dict[str, Any]], cfg: Dict[str, Any],
         blocked: Iterable[str] = (), blocked_matches: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Choose new orders: one per event, a few per series, up to the open-order cap.
    Busier markets first (more chances to fill, fairer prices)."""
    events = {r["event"] for r in open_rows}
    per_series: Dict[str, int] = {}
    for r in open_rows:
        per_series[r["series"]] = per_series.get(r["series"], 0) + 1
    room = cfg["max_open"] - len(open_rows)
    blocked = set(blocked)
    matches = set(blocked_matches) | {match_key(r["event"]) for r in open_rows}
    chosen = []
    for q in sorted(candidates, key=lambda x: -x["volume_24h"]):
        if room <= 0:
            break
        if q["event"] in events or q["ticker"] in blocked:
            continue
        if cfg.get("v2"):
            mk = match_key(q["event"])
            if mk in matches:
                continue
        if per_series.get(q["series"], 0) >= cfg["max_per_series"]:
            continue
        chosen.append(q)
        events.add(q["event"])
        if cfg.get("v2"):
            matches.add(match_key(q["event"]))
        per_series[q["series"]] = per_series.get(q["series"], 0) + 1
        room -= 1
    return chosen


async def run_cycle(client, logger=None, now: Optional[float] = None, path: Optional[str] = None,
                    universe: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    """Check fills, settle, cancel stale orders, post new ones. Returns counts."""
    now = now or time.time()
    cfg = settings()
    stats = {"filled": 0, "settled": 0, "cancelled": 0, "posted": 0}

    # 1. Fills for resting orders
    for r in rows("open", path):
        try:
            trades = await fetch_trades(client, r["ticker"], r["posted_ts"])
        except Exception as e:
            if logger:
                logger.warning(f"MAKER trades fetch failed for {r['ticker']}: {e}")
            continue
        fd = fill_detail(trades, r["side"], r["price"], r["qty"], r["queue_ahead"] or 0, r["posted_ts"],
                         r["ticker"])
        t = fd[0] if fd else None
        if t is not None and t <= min(now, r["close_ts"] or now):
            fee = maker_fee(r["price"], r["qty"])
            _update(r["id"], path, status="filled", fill_ts=t, fee=fee, note=fd[1], fill_kind=fd[2],
                    fill_trade_price=fd[3])
            stats["filled"] += 1
            if logger:
                logger.info(f"MAKER FILL {r['ticker']} {r['side'].upper()} {r['qty']} @ {r['price']:.2f}")

    # 2. Settle filled positions. Many markets close early once the outcome is
    # known, so check every held market now and then, not only after close_time.
    global _last_full_settle
    held = rows("filled", path)
    due = [r for r in held if not r["close_ts"] or now >= r["close_ts"]]
    if now - _last_full_settle > 10 * 60:
        due, _last_full_settle = held, now
    for i in range(0, len(due), 50):
        batch = due[i:i + 50]
        try:
            resp = await client.get_markets(limit=100, tickers=[r["ticker"] for r in batch])
            got = {m.get("ticker"): m for m in resp.get("markets") or []}
        except Exception as e:
            if logger:
                logger.warning(f"MAKER settle lookup failed: {e}")
            continue
        for r in batch:
            m = got.get(r["ticker"])
            if not m:
                continue
            pnl = settle_pnl(r["side"], r["price"], r["qty"], r["fee"] or 0.0, m.get("result") or "")
            if pnl is None:
                continue
            _update(r["id"], path, status="settled", result=(m.get("result") or "").lower(), pnl=pnl,
                    settled_ts=now)
            stats["settled"] += 1
            if logger:
                logger.info(f"MAKER SETTLED {r['ticker']} {r['side'].upper()} result={m.get('result')} "
                            f"pnl={pnl:+.2f}")

    # 2b. Mark-outs: where our side's price stood 5 and 30 minutes after each fill.
    # Falling right after we buy = we were picked off by better-informed sellers.
    await _markouts(client, now, path, logger)
    # 2c. Category of each series we trade (looked up once per series).
    await _series_categories(client, now, path)

    # 3. Fresh market view
    if universe is None:
        try:
            universe = await fetch_universe(client, cfg, now)
        except Exception as e:
            if logger:
                logger.warning(f"MAKER universe fetch failed: {e}")
            universe = []
    by_ticker = {m.get("ticker"): m for m in universe}

    # 4. Cancel stale orders: too old, about to close, or the bid moved above us.
    # V2 also pulls orders when the event is getting close or the price turns against us,
    # using a fresh look at each resting order's market (not the few-minutes-old universe).
    if cfg.get("v2"):
        open_tickers = [r["ticker"] for r in rows("open", path)]
        for i in range(0, len(open_tickers), 50):
            try:
                resp = await client.get_markets(limit=100, tickers=open_tickers[i:i + 50])
                for m in resp.get("markets") or []:
                    by_ticker[m.get("ticker")] = m
            except Exception as e:
                if logger:
                    logger.warning(f"MAKER fresh quotes failed: {e}")
    for r in rows("open", path):
        reason = None
        if cfg.get("v2"):
            end = r.get("expected_exp_ts") or r["close_ts"]
            m = by_ticker.get(r["ticker"])
            if end and (end - now) / 3600 < cfg["pull_within_hours"]:
                reason = "event getting close"
            elif m:
                q = quotes(m)
                bid = q["yes_bid"] if r["side"] == "yes" else q["no_bid"]
                mid = side_mid(m, r["side"])
                # Kalshi's book doesn't contain our paper order, so compare with the best bid
                # that was there when we posted (1c under ours when we improved on it).
                book_bid_then = r["price"] - (0.01 if r.get("improved") else 0.0)
                if 0 < bid < book_bid_then - 0.0005:
                    reason = "price fell below our bid"
                elif mid is not None and r.get("side_mid_post") and mid <= r["side_mid_post"] - cfg["pull_move"]:
                    reason = "price moving against us"
            if reason:
                _update(r["id"], path, status="cancelled", note=reason)
                stats["cancelled"] += 1
                continue
        if now - r["posted_ts"] > cfg["order_minutes"] * 60:
            reason = "expired"
        elif r["close_ts"] and r["close_ts"] - now < cfg["min_minutes"] * 60 / 2:
            reason = "closing"
        else:
            m = by_ticker.get(r["ticker"])
            if m:
                q = quotes(m)
                bid = q["yes_bid"] if r["side"] == "yes" else q["no_bid"]
                if bid > r["price"] + 0.0105:
                    reason = "bid moved away"
        if reason:
            _update(r["id"], path, status="cancelled", note=reason)
            stats["cancelled"] += 1

    # 5. Post new orders
    if not cfg["enabled"]:
        return stats
    open_now = rows("open", path)
    held = {r["ticker"] for r in rows("filled", path)}
    used_matches = ({match_key(r["event"]) for r in rows(path=path) if r["status"] != "cancelled"}
                    if cfg.get("v2") else set())
    candidates = [q for q in (quote_for(m, cfg, now) for m in universe) if q]
    for q in pick(candidates, open_now, cfg, blocked=held, blocked_matches=used_matches):
        # Re-quote on the live book (the universe can be a few minutes old).
        try:
            fresh = (await client.get_market(q["ticker"])).get("market", {})
        except Exception:
            continue
        live = quote_for(fresh, cfg, now) if fresh else None
        if not live or live["side"] != q["side"]:
            continue
        q = live
        q["qty"] = cfg["contracts"]
        ahead = 0.0
        if not q["improved"]:
            try:
                book = await client.get_orderbook(q["ticker"])
            except Exception:
                continue
            ahead = queue_ahead(book, q["side"], q["price"])
            if ahead is None:
                if logger:
                    logger.warning(f"MAKER skip {q['ticker']}: unreadable order book keys={list(book)[:5]}")
                continue
            if ahead == 0:
                # Joining the best bid means someone is already there; an empty level here
                # means the book moved. Re-check next cycle rather than guess.
                continue
        _insert(q, ahead, now, path)
        stats["posted"] += 1
        if logger:
            logger.info(f"MAKER POST {q['ticker']} {q['side'].upper()} {q['qty']} @ {q['price']:.2f} "
                        f"(spread {q['spread']*100:.0f}c, {ahead:.0f} ahead) {q['title'][:70]}")
    return stats


async def _markouts(client, now: float, path: Optional[str], logger=None) -> None:
    jobs = []
    for r in rows(path=path):
        if not r.get("fill_ts"):
            continue
        for col, lag in (("mid_5m", 300), ("mid_30m", 1800)):
            if r.get(col) is None and r["fill_ts"] + lag <= now < r["fill_ts"] + lag + 3600:
                jobs.append((r, col))
    for i in range(0, len(jobs), 50):
        batch = jobs[i:i + 50]
        try:
            resp = await client.get_markets(limit=100, tickers=sorted({r["ticker"] for r, _ in batch}))
            got = {m.get("ticker"): m for m in resp.get("markets") or []}
        except Exception as e:
            if logger:
                logger.warning(f"MAKER markout lookup failed: {e}")
            continue
        for r, col in batch:
            m = got.get(r["ticker"])
            if not m:
                continue
            res = (m.get("result") or "").lower()
            if res in ("yes", "no"):
                val = 1.0 if res == r["side"] else 0.0       # already decided
            else:
                val = side_mid(m, r["side"])
            if val is not None:
                _update(r["id"], path, **{col: val})


async def _series_categories(client, now: float, path: Optional[str], per_cycle: int = 10) -> None:
    known = series_meta(path)
    todo = sorted({r["series"] for r in rows(path=path)} - set(known))[:per_cycle]
    for sr in todo:
        try:
            resp = await client._make_authenticated_request("GET", f"/trade-api/v2/series/{sr}", require_auth=False)
            info = resp.get("series") or {}
        except Exception:
            info = {}
        with closing(_conn(path)) as c:
            c.execute("INSERT OR REPLACE INTO series_meta VALUES (?,?,?,?)",
                      (sr, info.get("category") or "Unknown", info.get("title") or "", now))
            c.commit()


# ----------------------------------------------------------------------------
# Report (pure over stored rows)
# ----------------------------------------------------------------------------

def report(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    posted = len(all_rows)
    filled = [r for r in all_rows if r["status"] in ("filled", "settled")]
    settled = [r for r in all_rows if r["status"] == "settled"]
    contracts = sum(r["qty"] for r in settled)
    pnl = sum(r["pnl"] or 0 for r in settled)
    cost = sum(r["qty"] * r["price"] for r in settled)
    wins = sum(1 for r in settled if (r["pnl"] or 0) > 0)
    buckets = []
    for lo, hi in ((0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.951)):
        b = [r for r in settled if lo <= r["price"] < hi]
        c = sum(r["qty"] for r in b)
        buckets.append({"price range": f"{lo*100:.0f}-{min(hi, 0.95)*100:.0f}c", "settled": len(b),
                        "win rate": (sum(1 for r in b if (r["pnl"] or 0) > 0) / len(b)) if b else None,
                        "avg price": (sum(r["price"] for r in b) / len(b)) if b else None,
                        "c_per_contract": (100 * sum(r["pnl"] or 0 for r in b) / c) if c else None,
                        "pnl": sum(r["pnl"] or 0 for r in b)})
    return {
        "posted": posted,
        "open": sum(1 for r in all_rows if r["status"] == "open"),
        "filled": len(filled),
        "fill_rate": len(filled) / max(1, posted - sum(1 for r in all_rows if r["status"] == "open")),
        "settled": len(settled),
        "awaiting": sum(1 for r in all_rows if r["status"] == "filled"),
        "contracts": contracts,
        "pnl": pnl,
        "c_per_contract": 100 * pnl / contracts if contracts else None,
        "return_on_cost": pnl / cost if cost else None,
        "win_rate": wins / len(settled) if settled else None,
        "avg_price": (sum(r["price"] for r in settled) / len(settled)) if settled else None,
        "buckets": buckets,
        "events": len({r["event"] for r in settled}),
    }


async def run_forever(logger=None, interval: Optional[int] = None) -> None:
    interval = interval or int(settings().get("interval", 60))
    from src.clients.kalshi_client import KalshiClient
    client = KalshiClient()
    universe, universe_ts = [], 0.0
    try:
        while True:
            now = time.time()
            try:
                if now - universe_ts > 3 * 60:
                    universe = await fetch_universe(client, settings(), now)
                    universe_ts = now
                    if logger:
                        cfg = settings()
                        n_ok = sum(1 for m in universe if quote_for(m, cfg, now))
                        logger.info(f"MAKER universe: {len(universe)} markets closing within "
                                    f"{cfg['max_hours']:.0f}h; {n_ok} qualify right now")
                stats = await run_cycle(client, logger, now, universe=universe)
                if logger and any(stats.values()):
                    logger.info(f"MAKER cycle: {stats}")
            except Exception as e:
                if logger:
                    logger.error(f"MAKER cycle failed: {e}")
            await asyncio.sleep(interval)
    finally:
        await client.close()
