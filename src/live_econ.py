"""
Live test of the Economics-favorites rule, with real money, small and fenced in.

Rules: exactly the paper 'Economics favorites only' arm (src/favorites.py, ARM_RULES['fav_econ']):
the favorite side of an Economics market closing in 6-24 hours, priced 90-98c, a real bid within
4c, one bet per market, held to settlement. Separate process (``cli.py live-econ``, tmux 'live'),
own database (``data/live_econ.db``), own dashboard tab.

SAFETY (the bot once placed real trades nobody asked for; none of that can happen here):

* SHADOW by default: it decides exactly what it would buy and records it, but sends nothing.
* Real orders only while ARMED: ``cli.py live-econ --arm`` asks you to type the dollar cap back,
  and writes data/live_arm.json. ``--disarm`` stops new orders within one cycle, no restart.
  The arm file is re-read before EVERY order.
* Hard limits: total money at risk (open positions at cost) <= the cap (code refuses caps over
  $250); 3-5 contracts per bet; at most 10 orders per scan; buys only, never sells.
* Daily stop: once today's settled losses reach the daily limit, no new orders until tomorrow.
* Every order is 'immediate or cancel' at the price we saw in the REAL order book (never worse):
  whatever doesn't fill at once is cancelled, so nothing sits on Kalshi's books.
* Three order errors in a row -> it disarms itself and says so.
* Kalshi's hard lock (settings.trading.live_trading_enabled) is switched on in THIS process only for
  the instant of each armed order, and straight back off.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DB_PATH = os.getenv("LIVE_DB", "data/live_econ.db")
ARM_PATH = os.getenv("LIVE_ARM", "data/live_arm.json")
TZ = os.getenv("DASHBOARD_TZ", "America/New_York")
SCAN_SECONDS = 600
CAP_MAX = 250.0                 # the code refuses any cap above this
CONTRACTS_MIN, CONTRACTS_MAX = 3, 5
MAX_ORDERS_PER_SCAN = 10
MAX_ERRORS_IN_A_ROW = 3
DEFAULTS = {"cap": 100.0, "contracts": 3, "daily_loss_stop": 15.0}


def rules() -> Dict[str, Any]:
    from src.favorites import ARM_RULES
    return dict(ARM_RULES["fav_econ"])


# ----------------------------------------------------------------------------
# Arm / disarm
# ----------------------------------------------------------------------------

def read_arm(path: Optional[str] = None) -> Dict[str, Any]:
    """{'armed': bool, cap, contracts, daily_loss_stop, ...}; anything unreadable counts as NOT armed."""
    try:
        with open(path or ARM_PATH) as f:
            a = json.load(f)
    except (OSError, ValueError):
        return {"armed": False, **DEFAULTS}
    out = {**DEFAULTS, **{k: a.get(k) for k in ("cap", "contracts", "daily_loss_stop", "armed_at", "reason")
                          if a.get(k) is not None}}
    ok = (a.get("armed") is True and isinstance(out["cap"], (int, float)) and 0 < out["cap"] <= CAP_MAX
          and isinstance(out["contracts"], int) and CONTRACTS_MIN <= out["contracts"] <= CONTRACTS_MAX
          and isinstance(out["daily_loss_stop"], (int, float)) and out["daily_loss_stop"] > 0)
    out["armed"] = bool(ok)
    return out


def write_arm(armed: bool, cap: float = DEFAULTS["cap"], contracts: int = DEFAULTS["contracts"],
              daily_loss_stop: float = DEFAULTS["daily_loss_stop"], reason: str = "",
              path: Optional[str] = None) -> Dict[str, Any]:
    if armed:
        if not 0 < cap <= CAP_MAX:
            raise ValueError(f"cap must be between $1 and ${CAP_MAX:.0f}")
        if not CONTRACTS_MIN <= contracts <= CONTRACTS_MAX:
            raise ValueError(f"contracts per bet must be {CONTRACTS_MIN}-{CONTRACTS_MAX}")
        if daily_loss_stop <= 0:
            raise ValueError("daily loss stop must be above $0")
    data = {"armed": bool(armed), "cap": float(cap), "contracts": int(contracts),
            "daily_loss_stop": float(daily_loss_stop), "armed_at": time.time() if armed else None,
            "reason": reason}
    p = path or ARM_PATH
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)
    return data


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, mode TEXT,
                 ticker TEXT, event TEXT, series TEXT, title TEXT, side TEXT, scan_price REAL, limit_price REAL,
                 depth REAL, count_req INTEGER, client_order_id TEXT, order_id TEXT, status TEXT,
                 fill_count REAL, avg_price REAL, price_source TEXT, cost REAL, fee REAL, close_ts REAL,
                 result TEXT, pnl REAL, settled_ts REAL, error TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_live_ticker ON orders(ticker)")
    c.execute("CREATE TABLE IF NOT EXISTS log (ts REAL, level TEXT, msg TEXT)")
    return c


def orders(path: Optional[str] = None, where: str = "", params=()) -> List[Dict[str, Any]]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(f"SELECT * FROM orders {where}", params)]


def note(path: Optional[str], level: str, msg: str, logger=None) -> None:
    with closing(_conn(path)) as c:
        c.execute("INSERT INTO log VALUES (?,?,?)", (time.time(), level, msg))
        c.commit()
    if logger:
        (logger.warning if level != "info" else logger.info)(f"LIVE {msg}")


def money_at_risk(path: Optional[str] = None) -> float:
    """Cost of live positions not yet settled (what the cap limits)."""
    with closing(_conn(path)) as c:
        r = c.execute("SELECT COALESCE(SUM(cost), 0) FROM orders WHERE mode='live' AND fill_count > 0 "
                      "AND result IS NULL").fetchone()
    return float(r[0] or 0)


def _today_start(now: float) -> float:
    d = pd.Timestamp(now, unit="s", tz="UTC").tz_convert(TZ).normalize()
    return d.timestamp()


def settled_pnl_today(path: Optional[str], now: float, mode: str = "live") -> float:
    with closing(_conn(path)) as c:
        r = c.execute("SELECT COALESCE(SUM(pnl), 0) FROM orders WHERE mode=? AND settled_ts >= ? AND pnl IS NOT NULL",
                      (mode, _today_start(now))).fetchone()
    return float(r[0] or 0)


# ----------------------------------------------------------------------------
# Order details
# ----------------------------------------------------------------------------

def fee_total(price: float, count: float) -> float:
    """Kalshi taker fee for one order, rounded up to the cent."""
    if not 0 < price < 1 or count <= 0:
        return 0.0
    return math.ceil(round(0.07 * count * price * (1 - price) * 100, 6)) / 100


def parse_fills(resp: Dict[str, Any], side: str) -> Tuple[float, Optional[float], str]:
    """(contracts filled, average price paid per contract for OUR side, where the price came from)."""
    n = cost = 0.0
    src = "fills"
    for f in (resp or {}).get("fills") or []:
        try:
            cnt = float(f.get("count_fp") if f.get("count_fp") is not None else f.get("count") or 0)
        except (TypeError, ValueError):
            continue
        price = None
        for k, scale in ((f"{side}_price_dollars", 1.0), (f"{side}_price_fixed", 1.0), (f"{side}_price", 0.01)):
            if f.get(k) is not None:
                try:
                    price = float(f[k]) * scale
                    break
                except (TypeError, ValueError):
                    pass
        if price is None and f.get("price") is not None:          # YES-leg price in dollars
            try:
                yes = float(f["price"])
                price = yes if side == "yes" else 1 - yes
                src = "fills (YES-leg price)"
            except (TypeError, ValueError):
                pass
        if price is None or cnt <= 0:
            continue
        n += cnt
        cost += cnt * price
    return n, (cost / n if n else None), src


# ----------------------------------------------------------------------------
# One scan
# ----------------------------------------------------------------------------

async def candidates(client, now: float, path: Optional[str] = None,
                     markets: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(market, favorite side) pairs that meet the Economics-arm rules and haven't been bet yet."""
    from src import favorites as F
    r = rules()
    markets = markets if markets is not None else await F.fetch_window(client, now)
    done = {o["ticker"] for o in orders(path)}
    elig = []
    for m in markets:
        if m["ticker"] in done or not F.in_window(m, now, r):
            continue
        fav = F.favorite(m, r)
        if fav:
            elig.append((m, fav))
    if not elig:
        return []
    cats = await F.ensure_categories(client, [m["ticker"].split("-", 1)[0] for m, _ in elig], path=F.DB_PATH)
    return [(m, fav) for m, fav in elig if cats.get(m["ticker"].split("-", 1)[0]) == r["category"]]


async def place(client, ticker: str, side: str, count: int, price: float) -> Dict[str, Any]:
    """One immediate-or-cancel buy at ``price`` or better. Kalshi's hard lock is lifted for this call only."""
    from src.config.settings import settings
    cents = int(round(price * 100))
    if not (1 <= cents <= 98) or side not in ("yes", "no") or not (1 <= count <= CONTRACTS_MAX):
        raise ValueError(f"refusing order: side={side} count={count} price={price}")
    prev = getattr(settings.trading, "live_trading_enabled", False)
    settings.trading.live_trading_enabled = True
    try:
        return await client.place_order(
            ticker=ticker, client_order_id=f"econ-{uuid.uuid4().hex[:20]}", side=side, action="buy", count=count,
            type_="market",                       # the client sends this as immediate-or-cancel at our price
            yes_price=cents if side == "yes" else None, no_price=cents if side == "no" else None)
    finally:
        settings.trading.live_trading_enabled = prev


async def scan(client, now: float, path: Optional[str] = None, logger=None, arm_path: Optional[str] = None,
               markets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    from src import favorites as F
    from src.arb_scan import depth_at_ask
    arm = read_arm(arm_path)
    mode = "live" if arm["armed"] else "shadow"
    stats = {"mode": mode, "candidates": 0, "ordered": 0, "skipped_cap": 0, "skipped_book": 0, "stopped": None}
    if mode == "live" and settled_pnl_today(path, now) <= -arm["daily_loss_stop"]:
        stats["stopped"] = f"daily loss stop reached ({settled_pnl_today(path, now):+.2f} today)"
        note(path, "warn", stats["stopped"] + ": no new orders until tomorrow", logger)
        return stats
    cands = await candidates(client, now, path, markets)
    stats["candidates"] = len(cands)
    errors_in_a_row = 0
    for m, fav in cands[:MAX_ORDERS_PER_SCAN]:
        try:
            book = await client.get_orderbook(m["ticker"])
        except Exception:
            book = None
        real = F.best_ask(book, fav["side"]) if book else None
        r = rules()
        # buy only at the price we saw, or better, and only if that price really is in the book
        if real is None or not (r["min_price"] <= real <= r["max_price"]) or real > fav["price"] + 1e-9:
            stats["skipped_book"] += 1
            continue
        depth = depth_at_ask(book, fav["side"], real) or 0
        arm = read_arm(arm_path) if mode == "live" else arm          # re-read before every real order
        live_now = mode == "live" and arm["armed"]
        count = int(min(arm["contracts"], depth))
        if count < 1:
            stats["skipped_book"] += 1
            continue
        est_cost = count * real + fee_total(real, count)
        if live_now and money_at_risk(path) + est_cost > arm["cap"] + 1e-9:
            stats["skipped_cap"] += 1
            continue
        row = {"ts": now, "mode": "live" if live_now else "shadow", "ticker": m["ticker"],
               "event": m.get("event_ticker"), "series": m["ticker"].split("-", 1)[0],
               "title": " — ".join(x for x in (m.get("title"), m.get("yes_sub_title")) if x)[:200],
               "side": fav["side"], "scan_price": fav["price"], "limit_price": real, "depth": depth,
               "count_req": count, "close_ts": _close_ts(m)}
        if not live_now:
            row.update(status="shadow", fill_count=count, avg_price=real, price_source="shadow (book)",
                       cost=count * real + fee_total(real, count), fee=fee_total(real, count))
        else:
            try:
                resp = await place(client, m["ticker"], fav["side"], count, real)
                if resp.get("blocked") or resp.get("error"):
                    raise RuntimeError(resp.get("error") or "blocked")
                o = resp.get("order") or resp
                filled = float(o.get("fill_count_fp") or o.get("fill_count") or 0)
                row.update(status="sent", order_id=o.get("order_id"), fill_count=filled,
                           avg_price=real if filled else None, price_source="order limit (until fills read)",
                           cost=(filled * real + fee_total(real, filled)) if filled else 0.0,
                           fee=fee_total(real, filled) if filled else 0.0)
                errors_in_a_row = 0
                stats["ordered"] += 1
            except Exception as e:
                errors_in_a_row += 1
                row.update(status="error", error=repr(e)[:300], fill_count=0, cost=0.0)
                note(path, "warn", f"order failed on {m['ticker']}: {e!r}", logger)
                if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                    write_arm(False, reason=f"auto-disarmed after {errors_in_a_row} order errors in a row",
                              path=arm_path)
                    note(path, "warn", "AUTO-DISARMED after repeated order errors; back to shadow mode", logger)
                    _insert(path, row)
                    break
        _insert(path, row)
    msg = (f"scan ({stats['mode']}): {stats['candidates']} new Economics favorites qualified; "
           + (f"{stats['ordered']} real orders placed" if stats["mode"] == "live" else "recorded as shadow bets")
           + (f"; {stats['skipped_cap']} skipped (cap reached)" if stats["skipped_cap"] else "")
           + (f"; {stats['skipped_book']} skipped (price not really in the book)" if stats["skipped_book"] else ""))
    note(path, "info", msg)                    # shows under 'Recent events' on the dashboard
    if logger:
        logger.info(f"LIVE {msg}")
    return stats


def _close_ts(m: Dict[str, Any]) -> Optional[float]:
    from src.whale import ts_of
    return ts_of(m.get("close_time"))


def _insert(path: Optional[str], row: Dict[str, Any]) -> None:
    cols = list(row)
    with closing(_conn(path)) as c:
        c.execute(f"INSERT INTO orders ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                  [row[k] for k in cols])
        c.commit()


async def reconcile_fills(client, path: Optional[str] = None, logger=None) -> int:
    """Replace the estimated price with what Kalshi actually charged, from the order's fills."""
    todo = orders(path, "WHERE mode='live' AND status='sent' AND order_id IS NOT NULL")
    n = 0
    for o in todo:
        try:
            resp = await client.get_fills(order_id=o["order_id"])
        except Exception:
            continue
        cnt, avg, src = parse_fills(resp, o["side"])
        if cnt == 0 and (o["fill_count"] or 0) > 0:
            # Kalshi said it filled but its fill record isn't there (yet): keep the estimate, never zero it.
            if time.time() - o["ts"] < 3600:
                continue
            with closing(_conn(path)) as c:
                c.execute("UPDATE orders SET status='filled (price estimated)' WHERE id=?", (o["id"],))
                c.commit()
            n += 1
            continue
        status = "filled" if cnt >= (o["count_req"] or 0) - 1e-9 and cnt > 0 else "partial" if cnt > 0 else "no fill"
        cost = cnt * (avg or 0) + fee_total(avg or 0, cnt)
        with closing(_conn(path)) as c:
            c.execute("UPDATE orders SET status=?, fill_count=?, avg_price=?, price_source=?, cost=?, fee=? WHERE id=?",
                      (status, cnt, avg, src if avg is not None else None, cost, fee_total(avg or 0, cnt), o["id"]))
            c.commit()
        n += 1
    return n


async def settle(client, now: float, path: Optional[str] = None, logger=None) -> int:
    from src.whale import _markets
    todo = orders(path, "WHERE result IS NULL AND fill_count > 0 AND status != 'error'")
    if not todo:
        return 0
    got = await _markets(client, sorted({o["ticker"] for o in todo}))
    n = 0
    with closing(_conn(path)) as c:
        for o in todo:
            res = ((got.get(o["ticker"]) or {}).get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            payout = float(o["fill_count"]) * (1.0 if res == o["side"] else 0.0)
            c.execute("UPDATE orders SET result=?, pnl=?, settled_ts=? WHERE id=?",
                      (res, payout - float(o["cost"] or 0), now, o["id"]))
            n += 1
        c.commit()
    if n and logger:
        logger.info(f"LIVE {n} positions settled")
    return n


LOCK_PATH = os.getenv("LIVE_LOCK", "data/live_econ.lock")


def acquire_single_instance(lock_path: Optional[str] = None):
    """Only one live process may run: two at once could both buy the same market before either records it.
    Returns the open lock file (keep it open for the life of the process), or None if another copy holds it."""
    import fcntl
    p = lock_path or LOCK_PATH
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    f = open(p, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


async def run_forever(logger=None, path: Optional[str] = None, arm_path: Optional[str] = None,
                      lock_path: Optional[str] = None) -> None:
    from src.clients.kalshi_client import KalshiClient
    from src.config.settings import settings
    lock = acquire_single_instance(lock_path)
    if lock is None:
        note(path, "warn", "another live process is already running: this copy did NOT start", logger)
        return
    settings.trading.live_trading_enabled = False          # off for this whole process except inside place()
    client = KalshiClient()
    last_scan = last_settle = 0.0
    note(path, "info", f"started in {'LIVE (armed)' if read_arm(arm_path)['armed'] else 'SHADOW'} mode", logger)
    was_armed = read_arm(arm_path)["armed"]
    try:
        while True:
            t0 = time.time()
            try:
                a = read_arm(arm_path)
                if a["armed"] != was_armed:            # record every switch between live and shadow
                    was_armed = a["armed"]
                    note(path, "warn" if not a["armed"] else "info",
                         f"now LIVE: armed, cap ${a['cap']:,.0f}, {a['contracts']} per bet" if a["armed"]
                         else f"now SHADOW: disarmed ({a.get('reason') or 'no reason given'})", logger)
                if t0 - last_scan > SCAN_SECONDS:
                    last_scan = t0
                    await scan(client, t0, path, logger, arm_path)
                if t0 - last_settle > 300:
                    last_settle = t0
                    await reconcile_fills(client, path, logger)
                    await settle(client, t0, path, logger)
            except Exception as e:
                note(path, "warn", f"cycle failed: {e!r}", logger)
            await asyncio.sleep(20)
    finally:
        await client.close()
        lock.close()


# ----------------------------------------------------------------------------
# Report (dashboard)
# ----------------------------------------------------------------------------

def report(path: Optional[str] = None, arm_path: Optional[str] = None) -> Dict[str, Any]:
    arm = read_arm(arm_path)
    out: Dict[str, Any] = {"arm": arm, "orders": pd.DataFrame(), "log": pd.DataFrame()}
    if not os.path.exists(path or DB_PATH):
        return out
    df = pd.DataFrame(orders(path))
    with closing(_conn(path)) as c:
        log = pd.read_sql("SELECT * FROM log ORDER BY ts DESC LIMIT 30", c)
    if not log.empty:
        log["time"] = pd.to_datetime(log["ts"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    out["log"] = log
    now = time.time()
    for mode in ("live", "shadow"):
        g = df[df["mode"] == mode] if not df.empty else df
        fil = g[g["fill_count"].fillna(0) > 0] if not g.empty else g
        settled = fil[fil["pnl"].notna()] if not fil.empty else fil
        out[mode] = {"orders": len(g), "filled": len(fil),
                     "contracts": float(fil["fill_count"].sum()) if not fil.empty else 0.0,
                     "settled": len(settled), "wins": int((settled["pnl"] > 0).sum()) if not settled.empty else 0,
                     "pnl": float(settled["pnl"].sum()) if not settled.empty else 0.0,
                     "pnl_today": settled_pnl_today(path, now, mode),
                     "at_risk": float(fil[fil["pnl"].isna()]["cost"].sum()) if not fil.empty else 0.0,
                     "c_per_contract": (100 * float(settled["pnl"].sum()) / float(settled["fill_count"].sum()))
                     if not settled.empty and settled["fill_count"].sum() else None}
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
        out["orders"] = df.sort_values("ts", ascending=False)
    out["vs_paper"] = vs_paper(df)
    return out


def vs_paper(df: pd.DataFrame) -> Dict[str, Any]:
    """Live fills next to the paper Economics arm on the SAME markets: did real money pay the same price
    and get the same result as paper? That's what the live test is for."""
    from src import favorites as F
    if df is None or df.empty:
        return {}
    live = df[(df["mode"] == "live") & (df["fill_count"].fillna(0) > 0)]
    if live.empty or not os.path.exists(F.DB_PATH):
        return {}
    paper = F.rows()
    paper = paper[paper["arm"] == "fav_econ"][["ticker", "price", "pnl"]].rename(
        columns={"price": "paper_price", "pnl": "paper_pnl"})
    m = live.merge(paper, on="ticker", how="inner")
    if m.empty:
        return {"matched": 0}
    m["price_diff_c"] = 100 * (m["avg_price"].astype(float) - m["paper_price"].astype(float))
    return {"matched": len(m), "median_price_diff_c": float(m["price_diff_c"].median()),
            "live_pnl_per_contract_c": (100 * m["pnl"].sum() / m["fill_count"].sum()) if m["pnl"].notna().any() else None,
            "paper_pnl_per_contract_c": (100 * m["paper_pnl"].mean()) if m["paper_pnl"].notna().any() else None}
