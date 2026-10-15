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

Nothing in this module can place a real order: it never calls place_order.
Everything is stored in ``data/maker.db``.
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

DB_PATH = os.getenv("MAKER_DB", "data/maker.db")
MAKER_FEE_RATE = 0.0175          # Kalshi maker fee where charged (quarter of the 7% taker rate)
EXCLUDE_PREFIXES = ("KXMVE",)    # multi-leg combo markets: illiquid, not real single bets


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
    return {"ticker": t, "event": m.get("event_ticker") or t.rsplit("-", 1)[0], "series": series_of(t),
            "side": side, "price": price, "improved": improve, "spread": spread, "close_ts": close,
            "title": " — ".join(x for x in (m.get("title"), m.get("yes_sub_title")) if x)[:200],
            "volume_24h": volume_24h(m)}


def queue_ahead(orderbook: Dict[str, Any], side: str, price: float) -> float:
    """Contracts already bidding at ``price`` on ``side`` (they fill before us)."""
    ob = orderbook.get("orderbook", orderbook) or {}
    levels = ob.get(f"{side}_dollars")
    total = 0.0
    if levels:
        for lvl in levels:
            try:
                if abs(float(lvl[0]) - price) < 0.0005:
                    total += float(lvl[1])
            except (TypeError, ValueError, IndexError):
                continue
        return total
    for lvl in ob.get(side) or []:
        try:
            if abs(float(lvl[0]) / 100 - price) < 0.0005:
                total += float(lvl[1])
        except (TypeError, ValueError, IndexError):
            continue
    return total


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
               ahead: float, posted_ts: float) -> Optional[float]:
    """Timestamp at which a resting buy would have filled, or None.

    Fills if someone sold our side BELOW our price after we posted (they'd have
    hit our better bid first), or enough sold AT our price to clear the queue
    ahead of us plus our own size. Conservative: assumes nobody ahead cancelled.
    """
    at_price = 0.0
    for tr in sorted(trades, key=lambda x: ts_of(x.get("created_time")) or 0):
        t = ts_of(tr.get("created_time"))
        if t is None or t <= posted_ts:
            continue
        p = _trade_side_price(tr, side)
        if p is None:
            continue
        n = _num(tr, "count_fp", "count")
        if p < price - 0.0005:
            return t
        if abs(p - price) < 0.0005:
            at_price += n
            if at_price >= ahead + qty:
                return t
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
    path = path or DB_PATH
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
    return c


def rows(status: Optional[str] = None, path: Optional[str] = None) -> List[Dict[str, Any]]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        q = "SELECT * FROM orders" + (" WHERE status=?" if status else "") + " ORDER BY posted_ts"
        return [dict(r) for r in c.execute(q, (status,) if status else ())]


def _insert(o: Dict[str, Any], ahead: float, now: float, path: Optional[str] = None) -> None:
    with closing(_conn(path)) as c:
        c.execute("""INSERT INTO orders (ticker, event, series, title, side, price, qty, queue_ahead,
                     improved, spread, posted_ts, close_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open')""",
                  (o["ticker"], o["event"], o["series"], o["title"], o["side"], o["price"], o["qty"],
                   ahead, int(o["improved"]), o["spread"], now, o["close_ts"]))
        c.commit()


def _update(oid: int, path: Optional[str] = None, **fields) -> None:
    keys = ", ".join(f"{k}=?" for k in fields)
    with closing(_conn(path)) as c:
        c.execute(f"UPDATE orders SET {keys} WHERE id=?", (*fields.values(), oid))
        c.commit()


# ----------------------------------------------------------------------------
# Network (Kalshi public endpoints; read-only)
# ----------------------------------------------------------------------------

async def fetch_universe(client, cfg: Dict[str, Any], now: Optional[float] = None,
                         max_pages: int = 10) -> List[Dict[str, Any]]:
    now = now or time.time()
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"limit": 1000, "status": "open",
                  "min_close_ts": int(now + cfg["min_minutes"] * 60),
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
         blocked: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Choose new orders: one per event, a few per series, up to the open-order cap.
    Busier markets first (more chances to fill, fairer prices)."""
    events = {r["event"] for r in open_rows}
    per_series: Dict[str, int] = {}
    for r in open_rows:
        per_series[r["series"]] = per_series.get(r["series"], 0) + 1
    room = cfg["max_open"] - len(open_rows)
    blocked = set(blocked)
    chosen = []
    for q in sorted(candidates, key=lambda x: -x["volume_24h"]):
        if room <= 0:
            break
        if q["event"] in events or q["ticker"] in blocked:
            continue
        if per_series.get(q["series"], 0) >= cfg["max_per_series"]:
            continue
        chosen.append(q)
        events.add(q["event"])
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
        t = check_fill(trades, r["side"], r["price"], r["qty"], r["queue_ahead"] or 0, r["posted_ts"])
        if t is not None and t <= min(now, r["close_ts"] or now):
            fee = maker_fee(r["price"], r["qty"])
            _update(r["id"], path, status="filled", fill_ts=t, fee=fee)
            stats["filled"] += 1
            if logger:
                logger.info(f"MAKER FILL {r['ticker']} {r['side'].upper()} {r['qty']} @ {r['price']:.2f}")

    # 2. Settle filled positions after close
    for r in rows("filled", path):
        if r["close_ts"] and now < r["close_ts"]:
            continue
        try:
            m = (await client.get_market(r["ticker"])).get("market", {})
        except Exception:
            continue
        pnl = settle_pnl(r["side"], r["price"], r["qty"], r["fee"] or 0.0, m.get("result") or "")
        if pnl is None:
            continue
        _update(r["id"], path, status="settled", result=(m.get("result") or "").lower(), pnl=pnl, settled_ts=now)
        stats["settled"] += 1
        if logger:
            logger.info(f"MAKER SETTLED {r['ticker']} {r['side'].upper()} result={m.get('result')} pnl={pnl:+.2f}")

    # 3. Fresh market view
    if universe is None:
        try:
            universe = await fetch_universe(client, cfg, now)
        except Exception as e:
            if logger:
                logger.warning(f"MAKER universe fetch failed: {e}")
            universe = []
    by_ticker = {m.get("ticker"): m for m in universe}

    # 4. Cancel stale orders: too old, about to close, or the bid moved above us
    for r in rows("open", path):
        reason = None
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
    candidates = [q for q in (quote_for(m, cfg, now) for m in universe) if q]
    for q in pick(candidates, open_now, cfg, blocked=held):
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
                ahead = queue_ahead(await client.get_orderbook(q["ticker"]), q["side"], q["price"])
            except Exception:
                continue
        _insert(q, ahead, now, path)
        stats["posted"] += 1
        if logger:
            logger.info(f"MAKER POST {q['ticker']} {q['side'].upper()} {q['qty']} @ {q['price']:.2f} "
                        f"(spread {q['spread']*100:.0f}c, {ahead:.0f} ahead) {q['title'][:70]}")
    return stats


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


async def run_forever(logger=None, interval: int = 60) -> None:
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
                        logger.info(f"MAKER universe: {len(universe)} markets closing within "
                                    f"{settings()['max_hours']:.0f}h")
                stats = await run_cycle(client, logger, now, universe=universe)
                if logger and any(stats.values()):
                    logger.info(f"MAKER cycle: {stats}")
            except Exception as e:
                if logger:
                    logger.error(f"MAKER cycle failed: {e}")
            await asyncio.sleep(interval)
    finally:
        await client.close()
