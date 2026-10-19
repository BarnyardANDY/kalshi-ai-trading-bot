"""
Speed-gap study for Kalshi's 15-minute crypto markets (measurement only: never trades).

The "Dexter's Lab" style bots win by noticing that bitcoin has already moved on
the big exchanges while the 15-minute "up or down" market still shows the old
odds. This module measures whether such gaps exist on Kalshi today, how big and
how long they are, and whether acting on them would actually have made money
after Kalshi's taker fee, if you were 0, 1, 3 or 10 seconds late.

Every second, for each series (KXBTC15M, KXETH15M, KXSOL15M):

* spot price = median of Coinbase, Kraken and Bitstamp (Kalshi settles on the
  CF Benchmarks index, which is built from exchanges like these),
* the open 15-minute market's bid/ask on Kalshi,
* our fair probability that the market settles YES: a random-walk model using
  the last 15 minutes of realised volatility, the time left, and the fact that
  settlement is the AVERAGE of the final 60 seconds.

Samples go to ``data/gap_study.db``; outcomes are filled in when each market
settles. ``report()`` then answers: is our model as good as Kalshi's own price
(if not, any "gap" is just our error), how often does Kalshi's ask sit below
fair value by more than the fee, for how long, and what would buying at those
moments have earned.
"""
from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import statistics
import time
from collections import deque
from contextlib import closing
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

DB_PATH = os.getenv("GAP_DB", "data/gap_study.db")
REPORT_PATH = os.getenv("GAP_REPORT", "data/gap_report.json")   # written by the study itself; the dashboard just reads it
TAKER_FEE_RATE = 0.07

SERIES: Dict[str, Dict[str, str]] = {
    "KXBTC15M": {"coinbase": "BTC-USD", "kraken": "XBTUSD", "bitstamp": "btcusd"},
    "KXETH15M": {"coinbase": "ETH-USD", "kraken": "ETHUSD", "bitstamp": "ethusd"},
    "KXSOL15M": {"coinbase": "SOL-USD", "kraken": "SOLUSD", "bitstamp": "solusd"},
}


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def taker_fee(price: float) -> float:
    """Per-contract taker fee (dollars) at ``price``."""
    return TAKER_FEE_RATE * price * (1 - price) if 0 < price < 1 else 0.0


def fair_prob_up(spot: float, strike: float, tau: float, sigma: float,
                 final_avg_so_far: Optional[float] = None) -> float:
    """P(average of the final 60 s of the price ends above ``strike``).

    ``sigma`` is the per-second standard deviation of log returns; ``tau`` the
    seconds left. Inside the final minute, ``final_avg_so_far`` is the average
    of the seconds already observed.
    """
    if spot <= 0 or strike <= 0:
        return 0.5
    s2 = (sigma * spot) ** 2
    if tau > 60:
        mean, var = spot, s2 * ((tau - 60) + 60 / 3)
    else:
        tau = max(0.0, tau)
        done = 60 - tau
        a = final_avg_so_far if final_avg_so_far is not None else spot
        mean = (done * a + tau * spot) / 60
        var = s2 * (tau / 60) ** 2 * (tau / 3)
    if var <= 1e-18:
        return 0.999 if mean > strike else 0.001
    p = _phi((mean - strike) / math.sqrt(var))
    return min(0.999, max(0.001, p))


def realised_sigma(points: List[Tuple[float, float]], floor: float = 2e-5) -> float:
    """Per-second volatility of log price from (time, price) samples, whatever their spacing."""
    sq = dt = 0.0
    n = 0
    for (t1, a), (t2, b) in zip(points, points[1:]):
        if a > 0 and b > 0 and 0 < t2 - t1 <= 10:
            sq += math.log(b / a) ** 2
            dt += t2 - t1
            n += 1
    if n < 30 or dt <= 0:
        return max(floor, 5e-5)
    return max(floor, math.sqrt(sq / dt))


def parse_kraken(data: Dict[str, Any]) -> Dict[str, float]:
    """{kraken pair as requested (upper, no slash): last price} from a multi-pair Ticker response."""
    out = {}
    for name, v in ((data or {}).get("result") or {}).items():
        try:
            price = float(v["c"][0])
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        n = name.upper()
        for want in ("XBTUSD", "ETHUSD", "SOLUSD"):
            base = want[:3]
            if (base in n or (base == "XBT" and "XXBT" in n)) and n.endswith("USD"):
                out[want] = price
    return out


def episodes(samples: List[Dict[str, Any]], threshold: float, max_gap_s: float = 3.0) -> List[Dict[str, Any]]:
    """Runs of consecutive samples on one market where buying a side at the ask beats fair
    value by at least ``threshold`` after the fee. ``samples`` sorted by time for ONE ticker."""
    out: List[Dict[str, Any]] = []
    cur: Dict[str, Optional[Dict[str, Any]]] = {"yes": None, "no": None}
    last_ts = None
    for s in samples:
        if last_ts is not None and s["ts"] - last_ts > max_gap_s:
            for side in cur:
                if cur[side]:
                    out.append(cur[side])
                    cur[side] = None
        last_ts = s["ts"]
        p = s.get("p_yes")
        for side, ask, fair in (("yes", s.get("yes_ask"), p),
                                ("no", s.get("no_ask"), None if p is None else 1 - p)):
            edge = None
            if fair is not None and ask and 0 < ask < 1:
                edge = fair - ask - taker_fee(ask)
            if edge is not None and edge >= threshold:
                if cur[side] is None:
                    cur[side] = {"ticker": s["ticker"], "series": s["series"], "side": side, "start": s["ts"],
                                 "end": s["ts"], "max_edge": edge, "ask0": ask, "tau0": s.get("tau")}
                else:
                    cur[side]["end"] = s["ts"]
                    cur[side]["max_edge"] = max(cur[side]["max_edge"], edge)
            elif cur[side] is not None:
                out.append(cur[side])
                cur[side] = None
    out.extend(e for e in cur.values() if e)
    for e in out:
        e["duration"] = e["end"] - e["start"] + 1
    return out


def simulate(ep: Dict[str, Any], samples: List[Dict[str, Any]], result: str, delay: float) -> Optional[float]:
    """Per-contract P&L (dollars) of buying the episode's side at the ask seen ``delay``
    seconds after it began, held to settlement. None if no quote that late."""
    t = ep["start"] + delay
    for s in samples:
        if s["ts"] >= t:
            ask = s.get(f"{ep['side']}_ask")
            if not ask or not 0 < ask < 1:
                return None
            won = (result == ep["side"])
            return (1.0 if won else 0.0) - ask - taker_fee(ask)
    return None


def _mid(s: Dict[str, Any]) -> Optional[float]:
    b, a = s.get("yes_bid"), s.get("yes_ask")
    return (b + a) / 2 if b and a and 0 < b < a < 1 else None


def kalshi_lag(samples: List[Dict[str, Any]], min_jump: float = 0.04,
               waits: Tuple[int, ...] = (0, 1, 2, 3, 5, 10)) -> List[Dict[str, Any]]:
    """Every sudden 1-second jump in our fair value (exchanges moved), and how much of that jump
    Kalshi's mid had priced in 0..10 s later. Also how much of the jump our own fair value KEPT
    (if it falls back, the jump was exchange noise, not news). One ticker, sorted by time."""
    out = []
    n = len(samples)
    for i in range(1, n):
        a, b = samples[i - 1], samples[i]
        if a.get("p_yes") is None or b.get("p_yes") is None or not 0 < b["ts"] - a["ts"] <= 1.5:
            continue
        if (b.get("tau") or 0) < 90:
            continue
        jump = b["p_yes"] - a["p_yes"]
        base = _mid(a)
        if abs(jump) < min_jump or base is None:
            continue
        rec: Dict[str, Any] = {"ticker": b["ticker"], "series": b["series"], "jump": jump}
        j = i
        for w in waits:
            while j < n and samples[j]["ts"] < b["ts"] + w:
                j += 1
            if j >= n or samples[j]["ts"] > b["ts"] + w + 2:
                break
            m = _mid(samples[j])
            if m is not None:
                rec[f"k{w}"] = (m - base) / jump
            if samples[j].get("p_yes") is not None:
                rec[f"m{w}"] = (samples[j]["p_yes"] - a["p_yes"]) / jump
        out.append(rec)
    return out


def anchored_samples(samples: List[Dict[str, Any]], lag: float = 5.0) -> List[Dict[str, Any]]:
    """Samples whose fair value is Kalshi's own mid ``lag`` seconds ago PLUS how much our model moved
    since. Removes our model's standing disagreement with Kalshi: a gap only appears when the
    exchanges have just moved and Kalshi hasn't followed yet (the 'speed' trade in its pure form)."""
    out = []
    j = 0
    for s in samples:
        while j + 1 < len(samples) and samples[j + 1]["ts"] <= s["ts"] - lag:
            j += 1
        a = samples[j]
        if not (lag <= s["ts"] - a["ts"] <= lag + 3) or a.get("p_yes") is None or s.get("p_yes") is None:
            out.append({**s, "p_yes": None})
            continue
        m = _mid(a)
        fair = None if m is None else min(0.99, max(0.01, m + s["p_yes"] - a["p_yes"]))
        out.append({**s, "p_yes": fair})
    return out


def _clustered(pnl: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[float], int]:
    """(mean c/contract, 95% margin in c, markets) with every market's gaps counted together:
    gaps on the same 15-minute market share one outcome, so they aren't independent bets."""
    if not pnl:
        return None, None, 0
    n = len(pnl)
    mean = sum(v for _, v in pnl) / n
    by: Dict[str, float] = {}
    for k, v in pnl:
        by[k] = by.get(k, 0.0) + (v - mean)
    k = len(by)
    if n < 2 or k < 2:
        return 100 * mean, None, k
    var = sum(x * x for x in by.values()) / (n * n) * k / (k - 1)
    return 100 * mean, 196 * math.sqrt(var), k


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS samples (ts REAL, series TEXT, ticker TEXT, close_ts REAL,
                 strike REAL, spot REAL, n_src INTEGER, tau REAL, sigma REAL, p_yes REAL,
                 yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_samples_ticker ON samples(ticker, ts)")
    c.execute("""CREATE TABLE IF NOT EXISTS outcomes (ticker TEXT PRIMARY KEY, series TEXT, close_ts REAL,
                 result TEXT, strike REAL)""")
    return c


# ----------------------------------------------------------------------------
# Network (public endpoints; read-only)
# ----------------------------------------------------------------------------

async def _spot_prices(http, pairs: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[float, int]]:
    """{series: (median spot, number of exchanges that answered)}."""
    async def get(url):
        try:
            r = await http.get(url)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    tasks = {}
    for sr, p in pairs.items():
        tasks[("coinbase", sr)] = get(f"https://api.exchange.coinbase.com/products/{p['coinbase']}/ticker")
        tasks[("bitstamp", sr)] = get(f"https://www.bitstamp.net/api/v2/ticker/{p['bitstamp']}/")
    kr_pairs = ",".join(p["kraken"] for p in pairs.values())
    tasks[("kraken", "*")] = get(f"https://api.kraken.com/0/public/Ticker?pair={kr_pairs}")
    keys = list(tasks)
    results = dict(zip(keys, await asyncio.gather(*tasks.values())))
    kr = parse_kraken(results.get(("kraken", "*")) or {})
    out = {}
    for sr, p in pairs.items():
        vals = []
        cb = results.get(("coinbase", sr))
        bs = results.get(("bitstamp", sr))
        try:
            vals.append(float(cb["price"]))
        except (TypeError, KeyError, ValueError):
            pass
        try:
            vals.append(float(bs["last"]))
        except (TypeError, KeyError, ValueError):
            pass
        if p["kraken"] in kr:
            vals.append(kr[p["kraken"]])
        if vals:
            out[sr] = (statistics.median(vals), len(vals))
    return out


def _ts(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _num(m: Dict[str, Any], dollars: str, cents: str) -> float:
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


async def _current_market(client, series: str, now: float) -> Optional[Dict[str, Any]]:
    resp = await client._make_authenticated_request(
        "GET", "/trade-api/v2/markets", params={"series_ticker": series, "status": "open", "limit": 50},
        require_auth=False)
    live = [m for m in resp.get("markets") or [] if (_ts(m.get("close_time")) or 0) > now]
    live.sort(key=lambda m: _ts(m.get("close_time")) or 0)
    return live[0] if live else None


class Tracker:
    """Per-series rolling state."""

    def __init__(self, series: str):
        self.series = series
        self.prices: Deque[Tuple[float, float]] = deque(maxlen=1000)
        self.market: Optional[Dict[str, Any]] = None
        self.strike: Optional[float] = None
        self.open_seen: Dict[str, float] = {}

    def final_avg(self, close_ts: float) -> Optional[float]:
        vals = [p for t, p in self.prices if close_ts - 60 <= t <= close_ts]
        return sum(vals) / len(vals) if vals else None


async def run_forever(logger=None, interval: float = 1.0, series: Optional[List[str]] = None,
                      path: Optional[str] = None) -> None:
    import httpx
    from src.clients.kalshi_client import KalshiClient

    series = series or list(SERIES)
    client = KalshiClient()
    trackers = {s: Tracker(s) for s in series}
    pending: Dict[str, Tuple[str, float, Optional[float]]] = {}   # ticker -> (series, close_ts, strike)
    last_outcome_check = 0.0
    last_report = 0.0
    report_task = None
    buffer: List[tuple] = []
    try:
        async with httpx.AsyncClient(timeout=2.0, headers={"User-Agent": "kalshi-gap-study"}) as http:
            while True:
                t0 = time.time()
                try:
                    spots = await _spot_prices(http, {s: SERIES[s] for s in series})
                    for s, tr in trackers.items():
                        now = time.time()
                        if s in spots:
                            tr.prices.append((now, spots[s][0]))
                        m = tr.market
                        if m is None or (_ts(m.get("close_time")) or 0) <= now:
                            if m is not None:
                                pending[m["ticker"]] = (s, _ts(m.get("close_time")) or now, tr.strike)
                            tr.market = await _current_market(client, s, now)
                            tr.strike = None
                            if tr.market is None:
                                continue
                            if logger:
                                logger.info(f"GAP {s}: now tracking {tr.market['ticker']}")
                        mk = (await client._make_authenticated_request(
                            "GET", f"/trade-api/v2/markets/{tr.market['ticker']}", require_auth=False)).get("market") or {}
                        if not mk:
                            continue
                        close_ts = _ts(mk.get("close_time")) or 0
                        open_ts = _ts(mk.get("open_time"))
                        if tr.strike is None:
                            fs = mk.get("floor_strike")
                            if fs not in (None, ""):
                                tr.strike = float(fs)
                            elif open_ts and now - open_ts <= 5 and s in spots:
                                tr.strike = spots[s][0]      # we saw the window open ourselves
                        spot = spots.get(s, (None, 0))
                        tau = close_ts - now
                        p_yes = None
                        sigma = realised_sigma(list(tr.prices)[-900:])
                        if tr.strike and spot[0]:
                            p_up = fair_prob_up(spot[0], tr.strike, tau, sigma,
                                                tr.final_avg(close_ts) if tau <= 60 else None)
                            st = (mk.get("strike_type") or "").lower()
                            p_yes = 1 - p_up if st.startswith("less") else p_up
                        buffer.append((now, s, mk.get("ticker"), close_ts, tr.strike, spot[0], spot[1], tau,
                                       sigma, p_yes, _num(mk, "yes_bid_dollars", "yes_bid"),
                                       _num(mk, "yes_ask_dollars", "yes_ask"), _num(mk, "no_bid_dollars", "no_bid"),
                                       _num(mk, "no_ask_dollars", "no_ask")))
                    if len(buffer) >= 30:
                        with closing(_conn(path)) as c:
                            c.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", buffer)
                            c.commit()
                        buffer.clear()
                    if time.time() - last_outcome_check > 60 and pending:
                        last_outcome_check = time.time()
                        for tk, (s, cts, strike) in list(pending.items()):
                            try:
                                m = (await client.get_market(tk)).get("market") or {}
                            except Exception:
                                continue
                            res = (m.get("result") or "").lower()
                            if res in ("yes", "no"):
                                with closing(_conn(path)) as c:
                                    c.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?)",
                                              (tk, s, cts, res, strike))
                                    c.commit()
                                pending.pop(tk)
                                if logger:
                                    logger.info(f"GAP {tk} settled {res.upper()}")
                            elif time.time() - cts > 6 * 3600:
                                pending.pop(tk)
                    if time.time() - last_report > 600 and (report_task is None or report_task.done()):
                        last_report = time.time()
                        # in a separate low-priority process, so the 1-second sampling never waits on it
                        from src import report_runner
                        report_task = asyncio.create_task(report_runner.run(
                            "src.crypto_gap", "GAP", logger,
                            env={"GAP_DB": path, "GAP_REPORT": os.path.join(os.path.dirname(path), "gap_report.json")}
                            if path else None))
                except Exception as e:
                    if logger:
                        logger.warning(f"GAP sample failed: {e}")
                await asyncio.sleep(max(0.0, interval - (time.time() - t0)))
    finally:
        if buffer:
            with closing(_conn(path)) as c:
                c.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", buffer)
                c.commit()
        await client.close()


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _tau_bucket(tau: Optional[float]) -> str:
    if tau is None:
        return "unknown"
    return "final minute" if tau <= 60 else "1-5 min left" if tau <= 300 else "5-15 min left"


COIN = {"KXBTC15M": "bitcoin", "KXETH15M": "ether", "KXSOL15M": "solana"}


def _gap_rows(label: str, eps_all, hours: float, delays) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gaps": label, "count": len(eps_all),
                           "per day": len(eps_all) / (hours / 24) if hours >= 1 else None,
                           "markets": len({e["ticker"] for e, _ in eps_all}),
                           "median seconds": statistics.median([e["duration"] for e, _ in eps_all]) if eps_all else None,
                           "share lasting 3s+": (sum(1 for e, _ in eps_all if e["duration"] >= 3) / len(eps_all))
                           if eps_all else None}
    for d in delays:
        mean, margin, _ = _clustered([(e["ticker"], sim[d]) for e, sim in eps_all if sim[d] is not None])
        row[f"{d}s late: c/contract"] = mean
        row[f"{d}s late: ± (c)"] = margin
    return row


def report(path: Optional[str] = None, thresholds=(0.02, 0.05, 0.10), delays=(0, 1, 3, 10),
           fast_thresholds=(0.03, 0.05), waits=(0, 1, 2, 3, 5, 10)) -> Dict[str, Any]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        outcomes = {r["ticker"]: dict(r) for r in c.execute("SELECT * FROM outcomes")}
        n_samples = c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        span = c.execute("SELECT MIN(ts), MAX(ts) FROM samples").fetchone()
    hours = ((span[1] or 0) - (span[0] or 0)) / 3600 if span and span[0] else 0.0

    def ticker_samples():
        # one market at a time, so days of 1-second samples never sit in memory at once
        with closing(_conn(path)) as c2:
            c2.row_factory = sqlite3.Row
            for tk in outcomes:
                yield tk, [dict(r) for r in c2.execute("SELECT * FROM samples WHERE ticker=? ORDER BY ts", (tk,))]

    cal: Dict[str, List[Tuple[float, float]]] = {}
    eps_by_th: Dict[float, list] = {th: [] for th in thresholds}
    fast_by_th: Dict[float, list] = {th: [] for th in fast_thresholds}
    jumps: List[Dict[str, Any]] = []
    for tk, ss in ticker_samples():
        res = outcomes[tk]["result"]
        for th in thresholds:
            for e in episodes(ss, th):
                eps_by_th[th].append((e, {d: simulate(e, ss, res, d) for d in delays}))
        anch = anchored_samples(ss)
        for th in fast_thresholds:
            for e in episodes(anch, th):
                fast_by_th[th].append((e, {d: simulate(e, ss, res, d) for d in delays}))
        jumps.extend(kalshi_lag(ss, waits=waits))
        y = 1.0 if res == "yes" else 0.0
        for s in ss[::15]:                                     # one sample per ~15 s
            if s["p_yes"] is None or not s["yes_bid"] or not s["yes_ask"]:
                continue
            mid = (s["yes_bid"] + s["yes_ask"]) / 2
            for b in (_tau_bucket(s["tau"]), f"{COIN.get(s['series'], s['series'])}, any time"):
                cal.setdefault(b, []).append(((s["p_yes"] - y) ** 2, (mid - y) ** 2))
    calibration = [{"time left": b, "samples": len(v), "our model Brier": sum(a for a, _ in v) / len(v),
                    "Kalshi price Brier": sum(m for _, m in v) / len(v)} for b, v in sorted(cal.items())]

    # 2. gaps vs. our model (as before, now with honest per-market margins)
    gaps = [_gap_rows(f"{th*100:.0f}c+ after fee", eps_by_th[th], hours, delays) for th in thresholds]
    # 2b. the 5c gaps split by coin and time left: where (if anywhere) does it work?
    mid_th = thresholds[1] if len(thresholds) > 1 else thresholds[0]
    split = []
    for name, keyf in (("coin", lambda e: COIN.get(e["series"], e["series"])),
                       ("time left", lambda e: _tau_bucket(e.get("tau0"))),
                       ("side bought", lambda e: e["side"].upper())):
        groups: Dict[str, list] = {}
        for e, sim in eps_by_th[mid_th]:
            groups.setdefault(keyf(e), []).append((e, sim))
        for g, lst in sorted(groups.items()):
            r = _gap_rows(f"{name}: {g}", lst, hours, delays)
            split.append(r)
    # 3. fast-move gaps: Kalshi's own price as the baseline, plus the exchanges' move in the last 5 s
    fast = [_gap_rows(f"fast move, {th*100:.0f}c+ after fee", fast_by_th[th], hours, delays)
            for th in fast_thresholds]
    # 4. how fast does Kalshi follow a sudden move on the exchanges?
    lag = []
    for w in waits:
        ks = [max(-1.0, min(2.0, j[f"k{w}"])) for j in jumps if f"k{w}" in j]
        ms = [max(-1.0, min(2.0, j[f"m{w}"])) for j in jumps if f"m{w}" in j]
        lag.append({"seconds after the jump": w, "jumps": len(ks),
                    "share Kalshi has priced in": sum(ks) / len(ks) if ks else None,
                    "share our fair value kept": sum(ms) / len(ms) if ms else None})
    best = None
    for r in gaps + fast + split:
        v, m = r.get("1s late: c/contract"), r.get("1s late: ± (c)")
        if v is not None and m is not None and (best is None or v - m > best[0]):
            best = (v - m, r["gaps"], v, m)
    return {"samples": n_samples, "hours": hours, "markets_settled": len(outcomes),
            "calibration": calibration, "gaps": gaps, "gaps_split": split, "fast_gaps": fast, "lag": lag,
            "jumps": len(jumps),
            "summary_line": f"{n_samples:,} samples, {len(outcomes)} markets settled, {len(jumps)} sudden moves"
                            + (f"; best at 1s late: {best[1]} {best[2]:+.2f}c ±{best[3]:.2f}" if best else "")}


async def _report_safely(path: Optional[str], logger=None) -> None:
    try:
        rep = await asyncio.to_thread(write_report, path)
        if logger:
            logger.info(f"GAP report saved: {rep['samples']:,} samples, {rep['markets_settled']} markets settled")
    except Exception as e:
        if logger:
            logger.warning(f"GAP report failed: {e!r}")


def write_report(path: Optional[str] = None, out: Optional[str] = None) -> Dict[str, Any]:
    """Compute the report and save it as JSON for the dashboard (keeps the heavy work out of it)."""
    import json
    rep = report(path)
    rep["computed_ts"] = time.time()
    out = out or REPORT_PATH
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f)
    os.replace(tmp, out)
    return rep


def read_report(out: Optional[str] = None) -> Optional[Dict[str, Any]]:
    import json
    try:
        with open(out or REPORT_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
