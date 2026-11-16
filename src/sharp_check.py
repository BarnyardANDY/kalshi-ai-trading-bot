"""
Sharp-sportsbook check (measurement only: never places an order anywhere). Separate from every other
test: own process (``cli.py sharp-check``), database (``data/sharp_check.db``), log and dashboard tab.

Idea: Pinnacle is the sharpest sportsbook, so its margin-free ("de-vigged") price is the best public
estimate of a game's true chance. If Kalshi's REAL order book sells a team's YES for less than that
estimate by more than Kalshi's fee, buying it has positive expected value. Our Polymarket check showed
the trap: most apparent gaps are stale prices or have nothing for sale. So every comparison uses the
real Kalshi order book (price AND contracts offered) read right after the sportsbook price.

What it records and reports:
  * every pre-game moment a Kalshi game-winner market is matched to a Pinnacle price (a "view")
  * paper bets: the FIRST view per market where  fair - real ask - fee >= threshold  and >= 3 contracts
    are for sale, held to the result (thresholds 2, 3, 5 and 8 cents)
  * judged with the same pre-set rule as every other test: 14 days and 300 separate events
  * how often a gap appears and how many dollars a day it would be worth
  * which price was closer to the real outcome: Pinnacle's or Kalshi's (Brier score)

Two odds providers are supported (free tiers): The Odds API (bookmaker 'pinnacle') and pinnapi.
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("SHARP_DB", "data/sharp_check.db")
REPORT_PATH = os.getenv("SHARP_REPORT", "data/sharp_report.json")
KEY_PATH = os.getenv("SHARP_KEY_FILE", "data/sharp_key.json")      # {"provider": ..., "key": ...}, chmod 600, git-ignored
MIN_DEPTH = 3                      # contracts that must be for sale at our price
THRESHOLDS = (0.02, 0.03, 0.05, 0.08)
VERDICT = {"min_days": 14, "min_events": 300}
HOURS_AHEAD = 30                   # only look at games starting within this long
STOP_BEFORE_START = 300            # no views in the last 5 minutes before the start (prices are being pulled)
ORDER_SIZE = 3

# our league -> Kalshi game-winner series, The Odds API sport key, pinnapi sport id (UNCONFIRMED: use --probe)
LEAGUES: Dict[str, Dict[str, Any]] = {
    "nfl": {"kalshi": "KXNFLGAME", "odds_api": "americanfootball_nfl", "pinnapi": os.getenv("PINNAPI_NFL")},
    "nba": {"kalshi": "KXNBAGAME", "odds_api": "basketball_nba", "pinnapi": os.getenv("PINNAPI_NBA")},
    "nhl": {"kalshi": "KXNHLGAME", "odds_api": "icehockey_nhl", "pinnapi": os.getenv("PINNAPI_NHL")},
    "mlb": {"kalshi": "KXMLBGAME", "odds_api": "baseball_mlb", "pinnapi": os.getenv("PINNAPI_MLB")},
    "cfb": {"kalshi": "KXNCAAFGAME", "odds_api": "americanfootball_ncaaf", "pinnapi": None},
}
# requests per day we allow ourselves (free tiers: The Odds API 500 a MONTH; pinnapi 100 a day)
DAILY_BUDGET = {"odds_api": int(os.getenv("SHARP_BUDGET", "15")), "pinnapi": int(os.getenv("SHARP_BUDGET", "90"))}


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def taker_fee(price: float, qty: int = ORDER_SIZE) -> float:
    """Kalshi taker fee per contract for an order of ``qty``, rounded up to the cent."""
    if not 0 < price < 1 or qty <= 0:
        return 0.0
    return math.ceil(round(0.07 * qty * price * (1 - price) * 100, 6)) / 100 / qty


def devig_two_way(o_home: float, o_away: float) -> Optional[Tuple[float, float]]:
    """Margin-free probabilities from two decimal odds (proportional method). None for odd-looking inputs."""
    try:
        if o_home <= 1.0 or o_away <= 1.0:
            return None
        a, b = 1 / o_home, 1 / o_away
    except (TypeError, ZeroDivisionError):
        return None
    total = a + b
    if not 1.0 <= total <= 1.15:                    # below 1 = arbitrage glitch; far above = a stale or wrong line
        return None
    return a / total, b / total


def _num(v) -> Optional[float]:
    if isinstance(v, dict):
        v = v.get("price", v.get("odds"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts(v) -> Optional[float]:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v) / (1000 if v > 1e11 else 1)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00").replace(" ", "T")).timestamp()
    except ValueError:
        return None


def parse_odds_api(resp: Any) -> List[Dict[str, Any]]:
    """The Odds API v4 /odds response -> [{home, away, start_ts, o_home, o_away}] from bookmaker 'pinnacle'."""
    out = []
    for ev in resp if isinstance(resp, list) else []:
        book = next((b for b in ev.get("bookmakers") or [] if b.get("key") == "pinnacle"), None)
        mk = next((m for m in (book or {}).get("markets") or [] if m.get("key") == "h2h"), None)
        if not mk:
            continue
        prices = {o.get("name"): _num(o.get("price")) for o in mk.get("outcomes") or []}
        oh, oa = prices.get(ev.get("home_team")), prices.get(ev.get("away_team"))
        st = _ts(ev.get("commence_time"))
        if oh and oa and st and ev.get("home_team") and ev.get("away_team") and len(prices) == 2:
            out.append({"home": ev["home_team"], "away": ev["away_team"], "start_ts": st, "o_home": oh, "o_away": oa,
                        "updated": _ts(book.get("last_update"))})
    return out


_START_KEYS = ("starts", "start_time", "starts_at", "start", "commence_time", "event_start", "date", "start_ts")


def parse_pinnapi(resp: Any) -> List[Dict[str, Any]]:
    """pinnapi /kit/v1/markets response (shape unconfirmed, so parsed loosely) -> the same event dicts."""
    evs = resp.get("events") if isinstance(resp, dict) else resp
    out = []
    for ev in evs if isinstance(evs, list) else []:
        periods = ev.get("periods") or {}
        full = periods.get("num_0") if isinstance(periods, dict) else None
        ml = (full or {}).get("money_line") or ev.get("money_line") or {}
        oh, oa = _num(ml.get("home")), _num(ml.get("away"))
        st = next((t for t in (_ts(ev.get(k)) for k in _START_KEYS) if t), None)
        if oh and oa and st and ev.get("home") and ev.get("away") and not ml.get("draw"):
            out.append({"home": str(ev["home"]), "away": str(ev["away"]), "start_ts": st, "o_home": oh, "o_away": oa,
                        "updated": None})
    return out


_STOP = {"the", "at", "vs", "winner", "will", "win", "game", "to", "of", "fc"}


def tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in _STOP and len(t) > 1}


def _market_text(ev: Dict[str, Any], mk: Dict[str, Any]) -> str:
    return " ".join(str(x or "") for x in (mk.get("yes_sub_title"), mk.get("subtitle"), mk.get("title")))


def match_kalshi(game: Dict[str, Any], k_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The Kalshi event for a Pinnacle game: two team markets, each matched by team-name words.
    Both teams must match different markets with a clear best score, or the game is skipped."""
    best = None
    for ev in k_events:
        mks = [m for m in ev.get("markets") or [] if m.get("ticker")]
        if len(mks) != 2:
            continue
        close = _ts(mks[0].get("expected_expiration_time")) or _ts(mks[0].get("close_time"))
        if close and abs(close - game["start_ts"]) > 2 * 86400 + 6 * 3600:
            continue
        ev_words = tokens(ev.get("title", ""))
        scores = {}
        for side in ("home", "away"):
            tw = tokens(game[side])
            scores[side] = [len(tw & tokens(_market_text(ev, m))) + 0.1 * len(tw & ev_words) for m in mks]
        for hi, ai in ((0, 1), (1, 0)):
            if scores["home"][hi] > 0 and scores["away"][ai] > 0 and scores["home"][hi] > scores["home"][ai] \
                    and scores["away"][ai] > scores["away"][hi]:
                total = scores["home"][hi] + scores["away"][ai]
                if best is None or total > best[0]:
                    best = (total, {"event": ev.get("event_ticker"), "home": mks[hi]["ticker"], "away": mks[ai]["ticker"]})
    return best[1] if best else None


def view_edge(fair: float, ask: Optional[float]) -> Optional[float]:
    """Expected profit per contract if bought at the real ask: fair chance - price - fee."""
    return None if ask is None or not 0 < ask < 1 else fair - ask - taker_fee(ask)


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS views(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, league TEXT, event TEXT, ticker TEXT, team TEXT,
  start_ts REAL, fair REAL, ask REAL, depth REAL, fee REAL, edge REAL, provider TEXT, odds_age REAL,
  result TEXT, settled_ts REAL, UNIQUE(ticker, ts));
CREATE TABLE IF NOT EXISTS requests(day TEXT, provider TEXT, n INTEGER, PRIMARY KEY(day, provider));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS log(ts REAL, level TEXT, msg TEXT);
"""


def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = sqlite3.connect(path, timeout=30)
    c.executescript(SCHEMA)
    return c


def note(path, level, msg, logger=None) -> None:
    with closing(_conn(path)) as c:
        c.execute("INSERT INTO log VALUES(?,?,?)", (time.time(), level, msg))
        c.commit()
    if logger:
        getattr(logger, "warning" if level == "warn" else "info")(f"SHARP {msg}")


def used_today(path, provider: str, now: float) -> int:
    day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    with closing(_conn(path)) as c:
        r = c.execute("SELECT n FROM requests WHERE day=? AND provider=?", (day, provider)).fetchone()
    return r[0] if r else 0


def count_request(path, provider: str, now: float) -> None:
    day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    with closing(_conn(path)) as c:
        c.execute("INSERT INTO requests VALUES(?,?,1) ON CONFLICT(day, provider) DO UPDATE SET n=n+1", (day, provider))
        c.commit()


def load_key(path: Optional[str] = None) -> Optional[Dict[str, str]]:
    try:
        with open(path or KEY_PATH) as f:
            d = json.load(f)
        return d if d.get("provider") in DAILY_BUDGET and d.get("key") else None
    except (OSError, ValueError):
        return None


def save_key(provider: str, key: str, path: Optional[str] = None) -> None:
    if provider not in DAILY_BUDGET:
        raise ValueError(f"provider must be one of {sorted(DAILY_BUDGET)}")
    p = path or KEY_PATH
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump({"provider": provider, "key": key.strip()}, f)
    os.chmod(p, 0o600)


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------

async def fetch_sharp(http, provider: str, key: str, league: str) -> Tuple[List[Dict[str, Any]], Any]:
    """Pinnacle prices for one league. Returns (events, raw response)."""
    cfg = LEAGUES[league]
    if provider == "odds_api":
        r = await http.get(f"https://api.the-odds-api.com/v4/sports/{cfg['odds_api']}/odds",
                           params={"regions": "eu", "markets": "h2h", "bookmakers": "pinnacle", "oddsFormat": "decimal",
                                   "apiKey": key})
        r.raise_for_status()
        return parse_odds_api(r.json()), r.json()
    sid = cfg.get("pinnapi")
    if not sid:
        raise ValueError(f"no pinnapi sport id set for {league} (set PINNAPI_{league.upper()}; find it with --probe)")
    r = await http.get("https://pinnapi.com/kit/v1/markets", params={"sport_id": sid, "event_type": "prematch"},
                       headers={"x-portal-apikey": key})
    r.raise_for_status()
    return parse_pinnapi(r.json()), r.json()


async def kalshi_events(client, series: str) -> List[Dict[str, Any]]:
    from src.poly_check import kalshi_events as ke
    return await ke(client, series)


async def real_ask(client, ticker: str) -> Tuple[Optional[float], float]:
    """(cheapest YES price really offered, contracts offered at it) from the live order book."""
    from src.poly_check import book_asks
    try:
        book = await client.get_orderbook(ticker)
    except Exception:
        return None, 0.0
    asks = book_asks(book.get("orderbook", book) if isinstance(book, dict) else {})
    if not asks:
        return None, 0.0
    best = asks[0][0]
    return best, sum(sz for pr, sz in asks if abs(pr - best) < 1e-6)


async def snapshot(client, http, provider: str, key: str, league: str, now: float, path=None, logger=None) -> Dict[str, int]:
    """One sportsbook read for a league + a real order-book read for every matched side, stored as views."""
    st = {"games": 0, "matched": 0, "views": 0, "gaps": 0, "requested": 0}
    k_events = await kalshi_events(client, LEAGUES[league]["kalshi"])      # free: skip the league if Kalshi has no games
    if not k_events:
        return st
    events, _ = await fetch_sharp(http, provider, key, league)
    count_request(path, provider, now)
    st["requested"] = 1
    upcoming = [g for g in events if now + STOP_BEFORE_START < g["start_ts"] <= now + HOURS_AHEAD * 3600]
    st["games"] = len(upcoming)
    if not upcoming:
        return st
    for g in upcoming:
        fair = devig_two_way(g["o_home"], g["o_away"])
        m = match_kalshi(g, k_events)
        if not fair or not m:
            continue
        st["matched"] += 1
        for side, p in (("home", fair[0]), ("away", fair[1])):
            ask, depth = await real_ask(client, m[side])
            fee = taker_fee(ask) if ask else None
            edge = view_edge(p, ask)
            with closing(_conn(path)) as c:
                c.execute("INSERT OR IGNORE INTO views(ts,league,event,ticker,team,start_ts,fair,ask,depth,fee,edge,"
                          "provider,odds_age) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (now, league, m["event"], m[side], g[side], g["start_ts"], p, ask, depth, fee, edge, provider,
                           (now - g["updated"]) if g.get("updated") else None))
                c.commit()
            st["views"] += 1
            if edge is not None and edge >= 0.03 and depth >= MIN_DEPTH:
                st["gaps"] += 1
    return st


async def settle(client, now: float, path=None, logger=None) -> int:
    from src.whale import _markets
    with closing(_conn(path)) as c:
        todo = [r[0] for r in c.execute("SELECT DISTINCT ticker FROM views WHERE result IS NULL AND start_ts < ?",
                                        (now - 2 * 3600,))]
    if not todo:
        return 0
    got = await _markets(client, todo)
    n = 0
    with closing(_conn(path)) as c:
        for t in todo:
            res = ((got.get(t) or {}).get("result") or "").lower()
            if res in ("yes", "no"):
                n += c.execute("UPDATE views SET result=?, settled_ts=? WHERE ticker=?", (res, now, t)).rowcount
        c.commit()
    return n


# ----------------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------------

def next_wait(provider: str, requests_used: int) -> float:
    """Seconds until the next round, so the day's request budget lasts all day (30 min when nothing is on)."""
    if requests_used <= 0:
        return 1800.0
    return max(900.0, 86400.0 * requests_used / max(1, DAILY_BUDGET[provider]))


async def run_forever(logger=None, path: Optional[str] = None) -> None:
    import httpx
    from src.clients.kalshi_client import KalshiClient

    cfg = load_key()
    if not cfg:
        note(path, "warn", "no odds key saved: run `python cli.py sharp-check --set-key`", logger)
        return
    provider, key = cfg["provider"], cfg["key"]
    client = KalshiClient()
    last_settle = 0.0
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "kalshi-sharp-check"}) as http:
            note(path, "info", f"started: {provider}, budget {DAILY_BUDGET[provider]} requests/day", logger)
            while True:
                t0 = time.time()
                leagues = [lg for lg in LEAGUES if provider == "odds_api" or LEAGUES[lg].get("pinnapi")]
                done, used = [], 0
                for lg in leagues:
                    if used_today(path, provider, time.time()) >= DAILY_BUDGET[provider]:
                        note(path, "info", "daily request budget used; waiting for tomorrow", logger)
                        break
                    try:
                        s = await snapshot(client, http, provider, key, lg, time.time(), path, logger)
                        used += s["requested"]
                        if s["games"]:
                            done.append(f"{lg}: {s['matched']}/{s['games']} games matched, {s['gaps']} gaps")
                    except Exception as e:
                        note(path, "warn", f"{lg} read failed: {e!r}"[:300], logger)
                note(path, "info", "round: " + ("; ".join(done) if done else "no games in the next day"), logger)
                if time.time() - last_settle > 600:
                    last_settle = time.time()
                    try:
                        await settle(client, last_settle, path, logger)
                    except Exception as e:
                        note(path, "warn", f"settle failed: {e!r}"[:200], logger)
                await asyncio.sleep(max(60.0, next_wait(provider, used) - (time.time() - t0)))
    finally:
        await client.close()


async def probe(logger=None, path: Optional[str] = None) -> Dict[str, Any]:
    """One read per league, nothing stored: does the odds feed parse, and do the games match Kalshi?"""
    import httpx
    from src.clients.kalshi_client import KalshiClient
    cfg = load_key()
    if not cfg:
        return {"error": "no odds key saved: run `python cli.py sharp-check --set-key`"}
    client, out = KalshiClient(), {"provider": cfg["provider"]}
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "kalshi-sharp-check"}) as http:
            for lg in LEAGUES:
                if cfg["provider"] == "pinnapi" and not LEAGUES[lg].get("pinnapi"):
                    continue
                try:
                    events, raw = await fetch_sharp(http, cfg["provider"], cfg["key"], lg)
                    k_events = await kalshi_events(client, LEAGUES[lg]["kalshi"])
                    shown = []
                    for g in events[:40]:
                        m = match_kalshi(g, k_events)
                        shown.append((g, m))
                    first_raw = (raw[0] if isinstance(raw, list) and raw else
                                 (raw.get("events") or [None])[0] if isinstance(raw, dict) else None)
                    out[lg] = {"sharp_games": len(events), "kalshi_events": len(k_events),
                               "matched": sum(1 for _, m in shown if m),
                               "examples": [f"{g['away']} @ {g['home']} -> {(m or {}).get('event')}" for g, m in shown[:3]],
                               "unmatched": [f"{g['away']} @ {g['home']}" for g, m in shown if not m][:3],
                               "kalshi_sample": [f"{e.get('title')} | {[mk.get('yes_sub_title') for mk in e.get('markets') or []]}"
                                                 for e in k_events[:2]],
                               "raw_keys": sorted(first_raw.keys())[:20] if isinstance(first_raw, dict) else None}
                except Exception as e:
                    out[lg] = {"error": repr(e)[:300]}
    finally:
        await client.close()
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def views_df(path: Optional[str] = None):
    import pandas as pd
    with closing(_conn(path)) as c:
        return pd.read_sql("SELECT * FROM views", c)


def _stats(g) -> Dict[str, Any]:
    from src.edge_scan import _cluster_stats
    out: Dict[str, Any] = {"bets": len(g), "settled": 0, "events": 0}
    s = g[g["result"].isin(["yes", "no"])]
    if s.empty:
        return out
    won = (s["result"] == "yes").astype(float)
    c = won - s["ask"] - s["fee"]
    mean, se, _ = _cluster_stats(c, s["event"])
    out.update(settled=len(s), events=int(s["event"].nunique()), win_rate=float(won.mean()),
               avg_price=float(s["ask"].mean()), avg_fair=float(s["fair"].mean()), c_per_contract=float(mean),
               margin=1.96 * se if math.isfinite(se) else None, pnl_3=float(c.sum() * ORDER_SIZE))
    return out


def verdict(st: Dict[str, Any], days: float) -> str:
    if not st.get("settled"):
        return "waiting for results"
    m, c = st.get("margin"), st["c_per_contract"]
    enough = days >= VERDICT["min_days"] and st["events"] >= VERDICT["min_events"]
    if m is not None and c + m < 0:
        return "losing (clearly)" if enough else "losing so far"
    if not enough:
        return f"too early: {st['events']} of {VERDICT['min_events']} events, day {days:.0f} of {VERDICT['min_days']}"
    return "PASSED: worth a tiny live test" if m is not None and c - m > 0 else "no edge (positive margin not reached)"


def bets_at(df, threshold: float):
    """First view per market with a real gap >= threshold and enough contracts for sale."""
    ok = df[(df["edge"] >= threshold) & (df["depth"] >= MIN_DEPTH) & df["ask"].notna()]
    return ok.sort_values("ts").groupby("ticker", as_index=False).first()


def report(path: Optional[str] = None) -> Dict[str, Any]:
    df = views_df(path)
    out: Dict[str, Any] = {"views": len(df), "provider": (load_key() or {}).get("provider")}
    with closing(_conn(path)) as c:
        out["log"] = [dict(zip(("ts", "level", "msg"), r)) for r in
                      c.execute("SELECT ts, level, msg FROM log ORDER BY ts DESC LIMIT 15")]
    if df.empty:
        return out
    days = max(0.0, (df["ts"].max() - df["ts"].min()) / 86400)
    out.update(days=days, games=int(df["event"].nunique()), markets=int(df["ticker"].nunique()),
               first_ts=float(df["ts"].min()), last_ts=float(df["ts"].max()))
    out["thresholds"] = []
    for t in THRESHOLDS:
        b = bets_at(df, t)
        st = _stats(b) if not b.empty else {"bets": 0, "settled": 0, "events": 0}
        per_day = len(b) / max(days, 1.0)
        st.update(threshold=t, per_day=per_day, verdict=verdict(st, days),
                  dollars_per_day=(st["c_per_contract"] * ORDER_SIZE * per_day) if st.get("c_per_contract") is not None else None)
        out["thresholds"].append(st)
    s = df[df["result"].isin(["yes", "no"]) & df["ask"].notna()]
    if not s.empty:
        y = (s["result"] == "yes").astype(float)
        out["calibration"] = {"samples": len(s), "events": int(s["event"].nunique()),
                              "Pinnacle Brier": float(((s["fair"] - y) ** 2).mean()),
                              "Kalshi Brier": float(((s["ask"] - y) ** 2).mean()),
                              "avg Kalshi ask minus Pinnacle fair (c)": float((s["ask"] - s["fair"]).mean() * 100)}
    out["gap_sizes"] = {"views with a real gap >= 3c": int(((df["edge"] >= 0.03) & (df["depth"] >= MIN_DEPTH)).sum()),
                        "views with a gap but under 3 contracts for sale": int(((df["edge"] >= 0.03) & (df["depth"] < MIN_DEPTH)).sum()),
                        "views with no Kalshi price for sale": int(df["ask"].isna().sum())}
    return out


def write_report(path: Optional[str] = None, out: Optional[str] = None) -> Dict[str, Any]:
    rep = report(path)
    out = out or REPORT_PATH
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(rep, f, default=str)
    os.replace(out + ".tmp", out)
    return rep
