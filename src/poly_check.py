"""
Polymarket price-check study (measurement only: never trades on either site). Separate from
every other test: own process (``cli.py poly-check``), database (``data/poly_check.db``), log
(``logs/poly_check.log``) and dashboard tab.

Research on prediction markets finds Polymarket usually moves first (more money trades there),
especially near the end. Our crypto speed-gap study failed mostly because OUR fair value was
worse than Kalshi's own price. Polymarket's live crowd price may be a far better fair value. So:

1. Match Kalshi game-winner markets (NFL, NBA, NHL, MLB, college football) with the same game
   on Polymarket (league + date + the two team codes).
2. Sample both sites' prices: every 5 s from an hour before the game until it ends, every
   minute earlier the same day.
3. Report: (a) which price predicts the winner better (Brier), (b) when Polymarket jumps,
   how fast Kalshi follows, (c) when Kalshi's ask sits below Polymarket's middle price by more
   than the fee: would buying on KALSHI ONLY (0, 5, 15, 30 s late, held to the result) have paid?

Only Kalshi would ever be traded; Polymarket is just the price check.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import statistics
import time
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("POLY_DB", "data/poly_check.db")
REPORT_PATH = os.getenv("POLY_REPORT", "data/poly_report.json")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
TAKER_FEE_RATE = 0.07

# league -> Polymarket tag slugs to try, Polymarket slug prefixes, Kalshi game-winner series
LEAGUES: Dict[str, Dict[str, Any]] = {
    "nfl": {"tags": ["nfl"], "prefixes": ["nfl"], "kalshi": "KXNFLGAME"},
    "nba": {"tags": ["nba"], "prefixes": ["nba"], "kalshi": "KXNBAGAME"},
    "nhl": {"tags": ["nhl"], "prefixes": ["nhl"], "kalshi": "KXNHLGAME"},
    "mlb": {"tags": ["mlb"], "prefixes": ["mlb"], "kalshi": "KXMLBGAME"},
    "cfb": {"tags": ["cfb", "ncaaf", "college-football"], "prefixes": ["cfb", "ncaaf"], "kalshi": "KXNCAAFGAME"},
}
# team codes that differ between the two sites (both sides normalised through this)
ALIASES = {"jac": "jax", "wsh": "was", "lvr": "lv", "gs": "gsw", "sa": "sas", "no": "nop", "phx": "pho",
           "bkn": "bk", "cws": "chw", "az": "ari"}
LIVE_EVERY = 2.0           # seconds between samples while a game is on (a home-built bot reacts in 1-3 s)
PRE_EVERY = 5.0            # ...in the hour before it starts
HOT_BEFORE = 3600          # sample every 5 s from an hour before the start...
HOT_AFTER = 5 * 3600       # ...until 5 hours after it
DAY_WINDOW = 24 * 3600     # every minute for games starting within a day


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def taker_fee(p: float) -> float:
    return TAKER_FEE_RATE * p * (1 - p) if 0 < p < 1 else 0.0


# per-league codes that differ between the sites, mapped to one shared code (seen in the live probe:
# Polymarket nhl-mon-stl / nhl-lak-chi / nhl-min-utah / nhl-ott-las vs Kalshi MTL, LA, UTA, VGK)
LEAGUE_ALIASES: Dict[str, Dict[str, str]] = {
    "nhl": {"mon": "mtl", "lak": "la", "utah": "uta", "las": "vgk", "vgs": "vgk", "nj": "njd", "tb": "tbl",
            "sj": "sjs", "cls": "cbj", "wsh": "was", "lv": "vgk"},
}


def lnorm(league: str, code: str) -> str:
    c = norm(code)
    return LEAGUE_ALIASES.get(league, {}).get(c, c)


def code_variants(league: str, code: str) -> List[str]:
    """Every spelling of a team code that maps to the same team (for building page names)."""
    canon = lnorm(league, code)
    out = {code.lower(), norm(code), canon}
    out |= {k for k, v in ALIASES.items() if v == canon}
    out |= {k for k, v in LEAGUE_ALIASES.get(league, {}).items() if v == canon}
    return sorted(out)


def norm(code: str) -> str:
    c = (code or "").lower()
    return ALIASES.get(c, c)


def _jl(v) -> List[Any]:
    """Gamma sends some lists as JSON-encoded strings."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else []
        except ValueError:
            return []
    return []


def _iso_ts(v) -> Optional[float]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00").replace(" ", "T")).timestamp()
    except ValueError:
        return None


_POLY_SLUG = re.compile(r"^([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-(\d{4})-(\d{2})-(\d{2})$")
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
_KALSHI_EVENT = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z0-9]+)$")


def parse_poly_slug(slug: str) -> Optional[Dict[str, Any]]:
    m = _POLY_SLUG.match(slug or "")
    if not m:
        return None
    return {"prefix": m.group(1), "teams": {norm(m.group(2)), norm(m.group(3))},
            "date": f"{m.group(4)}-{m.group(5)}-{m.group(6)}"}


def parse_kalshi_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = _KALSHI_EVENT.search(ev.get("event_ticker") or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    date = f"20{m.group(1)}-{_MONTHS[m.group(2)]:02d}-{m.group(3)}"
    teams = {}
    for mk in ev.get("markets") or []:
        code = (mk.get("ticker") or "").rsplit("-", 1)[-1]
        if code and code != (ev.get("event_ticker") or ""):
            teams[norm(code)] = mk
    if len(teams) != 2:
        return None
    return {"date": date, "teams": teams}


def dates_close(a: str, b: str, days: int = 1) -> bool:
    try:
        da, db = datetime.strptime(a, "%Y-%m-%d"), datetime.strptime(b, "%Y-%m-%d")
    except ValueError:
        return False
    return abs((da - db).days) <= days


def moneyline(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The game-winner market of a Polymarket game event: two team outcomes and their tokens."""
    for mk in ev.get("markets") or []:
        smt = (mk.get("sportsMarketType") or "").lower()
        outs, toks = _jl(mk.get("outcomes")), _jl(mk.get("clobTokenIds"))
        if len(outs) != 2 or len(toks) != 2 or {str(o).lower() for o in outs} == {"yes", "no"}:
            continue
        if smt and smt != "moneyline":
            continue
        if mk.get("closed"):
            continue
        return {"outcomes": [str(o) for o in outs], "tokens": [str(t) for t in toks],
                "start": _iso_ts(mk.get("gameStartTime")) or _iso_ts(mk.get("eventStartTime"))
                or _iso_ts(ev.get("startTime")),
                "prices": [float(x) for x in _jl(mk.get("outcomePrices"))] if _jl(mk.get("outcomePrices")) else None}
    return None


def lnorm_any(code: str) -> str:
    """Shared code under any league's aliases (team names are looked up per league, so this is safe)."""
    c = norm(code)
    for m in LEAGUE_ALIASES.values():
        c = m.get(c, c)
    return c


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2}


def assign_outcomes(kteams: Dict[str, Dict[str, Any]], ml: Dict[str, Any],
                    team_names: Dict[str, str]) -> Tuple[Optional[Dict[str, int]], str]:
    """Which Polymarket outcome is which Kalshi team: by team name, else by price (if unambiguous)."""
    codes = list(kteams)
    outs = [o.lower() for o in ml["outcomes"]]
    # 1) names: Polymarket's team list, or words shared with Kalshi's market titles
    pick: Dict[str, int] = {}
    for code in codes:
        mk = kteams[code]
        # the market's OWN team name only: Kalshi's game title names both teams, so it can't tell them apart
        hints = _words(team_names.get(code, "")) | _words(mk.get("yes_sub_title") or "") | \
            _words(next((n for k, n in team_names.items() if k != code and lnorm_any(k) == lnorm_any(code)), ""))
        hits = [i for i, o in enumerate(outs) if _words(o) & hints or o in (team_names.get(code, "").lower())]
        if len(hits) == 1:
            pick[code] = hits[0]
    if len(pick) == 2 and len(set(pick.values())) == 2:
        return pick, "names"
    if len(pick) == 1:
        (c, i), = pick.items()
        other = [x for x in codes if x != c][0]
        return {c: i, other: 1 - i}, "names"
    # 2) prices: the assignment where the two sites agree, only if the other one clearly doesn't
    if ml.get("prices"):
        def kmid(mk):
            b, a = _q(mk, "yes_bid"), _q(mk, "yes_ask")
            return (b + a) / 2 if b and a else None
        a, b = codes
        ma, mb = kmid(kteams[a]), kmid(kteams[b])
        if ma is not None and mb is not None:
            p0, p1 = ml["prices"]
            straight = abs(ma - p0) + abs(mb - p1)
            swapped = abs(ma - p1) + abs(mb - p0)
            if straight + 0.15 < swapped and straight < 0.16:
                return {a: 0, b: 1}, "prices"
            if swapped + 0.15 < straight and swapped < 0.16:
                return {a: 1, b: 0}, "prices"
    return None, "ambiguous"


def _q(m: Dict[str, Any], key: str) -> float:
    v = m.get(f"{key}_dollars")
    if v not in (None, ""):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = m.get(key)
    try:
        return float(v) / 100 if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS pairs (pair_id TEXT PRIMARY KEY, league TEXT, poly_slug TEXT,
                 kalshi_event TEXT, team_a TEXT, team_b TEXT, k_ticker_a TEXT, k_ticker_b TEXT,
                 poly_token_a TEXT, poly_token_b TEXT, start_ts REAL, matched_how TEXT, created_ts REAL,
                 result_a TEXT, settled_ts REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS samples (ts REAL, pair_id TEXT, live INTEGER,
                 ka_bid REAL, ka_ask REAL, kb_bid REAL, kb_ask REAL,
                 pa_bid REAL, pa_ask REAL, pb_bid REAL, pb_ask REAL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_poly_samples ON samples(pair_id, ts)")
    c.execute("CREATE TABLE IF NOT EXISTS discovery (ts REAL PRIMARY KEY, info TEXT)")
    return c


def pairs(path: Optional[str] = None, where: str = "", params=()) -> List[Dict[str, Any]]:
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(f"SELECT * FROM pairs {where}", params)]


# ----------------------------------------------------------------------------
# Network (public, read-only endpoints on both sites)
# ----------------------------------------------------------------------------

async def poly_events(http, league: str, seen_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = LEAGUES[league]
    seen, out = set(), []
    raw = 0
    sample: List[str] = []
    for tag in cfg["tags"]:
        for offset in range(0, 600, 200):
            try:
                r = await http.get(f"{GAMMA}/events", params={"tag_slug": tag, "closed": "false", "limit": 200,
                                                             "offset": offset})
                r.raise_for_status()
                page = r.json()
            except Exception:
                break
            if not isinstance(page, list) or not page:
                break
            raw += len(page)
            for ev in page:
                if len(sample) < 4 and ev.get("slug"):
                    sample.append(ev["slug"])
                info = parse_poly_slug(ev.get("slug") or "")
                if info and info["prefix"] in cfg["prefixes"] and ev.get("slug") not in seen:
                    seen.add(ev["slug"])
                    out.append(ev)
            if len(page) < 200:
                break
    if seen_info is not None:
        seen_info.update(tag_events=raw, tag_sample=sample)
    return out


_SLUG_MISS: Dict[str, float] = {}          # slugs looked up and not found: don't ask again for 2 hours


async def poly_events_by_slug(http, league: str, kal_evs: List[Dict[str, Any]], have: set,
                              now: Optional[float] = None, days_ahead: int = 3) -> List[Dict[str, Any]]:
    """Look up Kalshi games directly by Polymarket's game-page name ('nfl-ari-dal-2026-11-01'),
    for leagues whose tag search finds nothing. Only games in the next few days, both team orders,
    the Kalshi date and the next day (a late game can carry the next day's date in UTC)."""
    now = now or time.time()
    today = datetime.fromtimestamp(now).date()
    cands: List[str] = []
    for ev in kal_evs:
        kp = parse_kalshi_event(ev)
        if not kp:
            continue
        d0 = datetime.strptime(kp["date"], "%Y-%m-%d").date()
        if not 0 <= (d0 - today).days <= days_ahead:
            continue
        raw = [((m.get("ticker") or "").rsplit("-", 1)[-1]).lower() for m in kp["teams"].values()]
        codes = [code_variants(league, c) for c in raw]
        for d in (d0, d0.fromordinal(d0.toordinal() + 1)):
            for prefix in LEAGUES[league]["prefixes"]:
                for a in codes[0]:
                    for b in codes[1]:
                        for x, y in ((a, b), (b, a)):
                            slug = f"{prefix}-{x}-{y}-{d.isoformat()}"
                            if slug not in have and now - _SLUG_MISS.get(slug, 0) > 7200:
                                cands.append(slug)
    out = []
    for i in range(0, len(cands), 20):
        chunk = cands[i:i + 20]
        try:
            r = await http.get(f"{GAMMA}/events", params=[("slug", x) for x in chunk])
            r.raise_for_status()
            page = r.json()
        except Exception:
            continue
        found = {ev.get("slug") for ev in page if isinstance(ev, dict)} if isinstance(page, list) else set()
        for x in chunk:
            if x not in found:
                _SLUG_MISS[x] = now
        out.extend(ev for ev in (page if isinstance(page, list) else []) if ev.get("slug") in set(chunk))
    return out


async def poly_team_names(http, league: str) -> Dict[str, str]:
    """{team code: team name} from Polymarket's team list (best effort)."""
    try:
        r = await http.get(f"{GAMMA}/teams", params={"league": league, "limit": 500})
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}
    out = {}
    for t in data if isinstance(data, list) else []:
        code = t.get("abbreviation") or t.get("abbr")
        name = " ".join(x for x in (t.get("name"), t.get("alias"), t.get("nickname")) if x)
        if code and name:
            out[norm(code)] = name
    return out


async def kalshi_events(client, series: str) -> List[Dict[str, Any]]:
    out, cursor = [], None
    for _ in range(5):
        params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
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


async def poly_prices(http, tokens: List[str]) -> Dict[str, Dict[str, float]]:
    """{token: {"ask": best ask, "bid": best bid}} via the batch price endpoint."""
    out: Dict[str, Dict[str, float]] = {}
    for i in range(0, len(tokens), 240):
        chunk = tokens[i:i + 240]
        body = [{"token_id": t, "side": s} for t in chunk for s in ("BUY", "SELL")]
        r = await http.post(f"{CLOB}/prices", json=body)
        r.raise_for_status()
        data = r.json() or {}
        for t in chunk:
            d = data.get(t) or {}
            try:
                vals = [float(d[k]) for k in ("BUY", "SELL") if d.get(k) is not None]
            except (TypeError, ValueError):
                continue
            # The live feed labels the two sides the opposite way from its docs (BUY came back below SELL),
            # so don't trust the labels: the lower price is the best bid, the higher the best ask.
            if len(vals) == 2:
                out[t] = {"bid": min(vals), "ask": max(vals)}
    return out


def match_games(poly_evs: List[Dict[str, Any]], kal_evs: List[Dict[str, Any]], league: str,
                team_names: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pairs of (Polymarket game, Kalshi game), one-to-one, and counts of why others didn't pair."""
    stats: Dict[str, Any] = {"poly_games": 0, "kalshi_games": 0, "matched": 0, "no_kalshi_match": 0,
                             "no_moneyline": 0, "ambiguous_sides": 0}
    kparsed = [(ev, parse_kalshi_event(ev)) for ev in kal_evs]
    kparsed = [(ev, p) for ev, p in kparsed if p]
    stats["kalshi_games"] = len(kparsed)
    polys = []
    for pev in poly_evs:
        info = parse_poly_slug(pev.get("slug") or "")
        if info:
            polys.append((pev, info))
    stats["poly_games"] = len(polys)
    # every (Polymarket, Kalshi) pair with both teams the same and dates a day apart at most;
    # the same date first, and each game used once (a series plays the same teams on consecutive days)
    cand = []
    for pi, (pev, info) in enumerate(polys):
        for ki, (kev, kp) in enumerate(kparsed):
            if ({lnorm(league, t) for t in info["teams"]} == {lnorm(league, t) for t in kp["teams"]}
                    and dates_close(info["date"], kp["date"])):
                cand.append((0 if info["date"] == kp["date"] else 1, pi, ki))
    cand.sort()
    used_p, used_k, out = set(), set(), []
    for _, pi, ki in cand:
        if pi in used_p or ki in used_k:
            continue
        pev, _info = polys[pi]
        kev, kp = kparsed[ki]
        ml = moneyline(pev)
        if not ml:
            stats["no_moneyline"] += 1
            used_p.add(pi)
            continue
        sides, how = assign_outcomes(kp["teams"], ml, team_names)
        if not sides:
            stats["ambiguous_sides"] += 1
            used_p.add(pi)
            continue
        used_p.add(pi)
        used_k.add(ki)
        a, b = sorted(kp["teams"])
        out.append({"pair_id": f"{kev['event_ticker']}|{pev['slug']}", "league": league, "poly_slug": pev["slug"],
                    "kalshi_event": kev["event_ticker"], "team_a": a, "team_b": b,
                    "k_ticker_a": kp["teams"][a]["ticker"], "k_ticker_b": kp["teams"][b]["ticker"],
                    "poly_token_a": ml["tokens"][sides[a]], "poly_token_b": ml["tokens"][sides[b]],
                    "start_ts": ml["start"], "matched_how": how})
        stats["matched"] += 1
    in_cand = {pi for _, pi, _ in cand}
    stats["no_kalshi_match"] = sum(1 for i in range(len(polys)) if i not in in_cand)
    stats["unmatched_poly"] = [polys[i][0].get("slug") for i in range(len(polys)) if i not in used_p][:4]
    stats["unmatched_kalshi"] = [kparsed[i][0].get("event_ticker") for i in range(len(kparsed)) if i not in used_k][:4]
    return out, stats


async def find_games(client, http, league: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Tag search, then a direct look-up by game-page name for Kalshi games still unmatched."""
    cfg = LEAGUES[league]
    seen: Dict[str, Any] = {}
    pev = await poly_events(http, league, seen)
    kev = await kalshi_events(client, cfg["kalshi"])
    names = await poly_team_names(http, league)
    found, stats = match_games(pev, kev, league, names)
    done = {p["kalshi_event"] for p in found}
    have = {e.get("slug") for e in pev}
    extra = await poly_events_by_slug(http, league, [e for e in kev if e.get("event_ticker") not in done], have)
    if extra:
        found, stats = match_games(pev + extra, kev, league, names)
    stats.update(seen, by_name_lookup=len(extra), team_names=len(names))
    return found, stats


async def discover(client, http, path: Optional[str] = None, logger=None) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    now = time.time()
    for league, cfg in LEAGUES.items():
        try:
            found, stats = await find_games(client, http, league)
        except Exception as e:
            info[league] = {"error": repr(e)[:200]}
            continue
        stats = {k: v for k, v in stats.items() if not isinstance(v, list)}
        with closing(_conn(path)) as c:
            for p in found:
                c.execute("""INSERT OR IGNORE INTO pairs (pair_id, league, poly_slug, kalshi_event, team_a, team_b,
                             k_ticker_a, k_ticker_b, poly_token_a, poly_token_b, start_ts, matched_how, created_ts)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (p["pair_id"], p["league"], p["poly_slug"], p["kalshi_event"], p["team_a"], p["team_b"],
                           p["k_ticker_a"], p["k_ticker_b"], p["poly_token_a"], p["poly_token_b"], p["start_ts"],
                           p["matched_how"], now))
            c.commit()
        info[league] = stats
    with closing(_conn(path)) as c:
        c.execute("INSERT OR REPLACE INTO discovery VALUES (?,?)", (now, json.dumps(info)))
        c.commit()
    if logger:
        logger.info("POLY discovery: " + "; ".join(
            f"{lg} {s.get('matched', 0)}/{s.get('poly_games', 0)} matched" if "error" not in s else f"{lg} error {s['error']}"
            for lg, s in info.items()))
    return info


async def sample(client, http, pl: List[Dict[str, Any]], now: float, path: Optional[str] = None) -> int:
    from src.whale import _markets
    if not pl:
        return 0
    # both sites read at the same moment (one after the other gave Polymarket's price a slight head start)
    ks, ps = await asyncio.gather(
        _markets(client, sorted({t for p in pl for t in (p["k_ticker_a"], p["k_ticker_b"])})),
        poly_prices(http, sorted({t for p in pl for t in (p["poly_token_a"], p["poly_token_b"])})))
    rows = []
    for p in pl:
        ka, kb = ks.get(p["k_ticker_a"]), ks.get(p["k_ticker_b"])
        pa, pb = ps.get(p["poly_token_a"]), ps.get(p["poly_token_b"])
        if not (ka and kb and pa and pb):
            continue
        live = 1 if p["start_ts"] and now >= p["start_ts"] else 0
        rows.append((now, p["pair_id"], live, _q(ka, "yes_bid"), _q(ka, "yes_ask"), _q(kb, "yes_bid"),
                     _q(kb, "yes_ask"), pa.get("bid"), pa.get("ask"), pb.get("bid"), pb.get("ask")))
    if rows:
        with closing(_conn(path)) as c:
            c.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            c.commit()
    return len(rows)


async def settle(client, now: float, path: Optional[str] = None, logger=None) -> int:
    from src.whale import _markets
    todo = pairs(path, "WHERE result_a IS NULL AND (start_ts IS NULL OR start_ts < ?)", (now - 2 * 3600,))
    if not todo:
        return 0
    got = await _markets(client, sorted({t for p in todo for t in (p["k_ticker_a"], p["k_ticker_b"])}))
    n = 0
    with closing(_conn(path)) as c:
        for p in todo:
            ra = ((got.get(p["k_ticker_a"]) or {}).get("result") or "").lower()
            rb = ((got.get(p["k_ticker_b"]) or {}).get("result") or "").lower()
            res = "yes" if ra == "yes" else "no" if rb == "yes" else "tie" if ra == rb == "no" else None
            if res:
                c.execute("UPDATE pairs SET result_a=?, settled_ts=? WHERE pair_id=?", (res, now, p["pair_id"]))
                n += 1
            elif p["start_ts"] and now - p["start_ts"] > 7 * 86400:
                c.execute("UPDATE pairs SET result_a='unresolved', settled_ts=? WHERE pair_id=?", (now, p["pair_id"]))
        c.commit()
    if n and logger:
        logger.info(f"POLY {n} games settled")
    return n


async def run_forever(logger=None, path: Optional[str] = None) -> None:
    import httpx
    from src.clients.kalshi_client import KalshiClient
    from src import report_runner

    client = KalshiClient()
    last_disc = last_slow = last_pre = last_settle = last_report = 0.0
    report_task = None
    try:
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "kalshi-poly-check"}) as http:
            while True:
                t0 = time.time()
                try:
                    if t0 - last_disc > 600:
                        last_disc = t0
                        await discover(client, http, path, logger)
                    open_pairs = pairs(path, "WHERE result_a IS NULL")
                    live = [p for p in open_pairs if p["start_ts"] and p["start_ts"] <= t0 <= p["start_ts"] + HOT_AFTER]
                    await sample(client, http, live, time.time(), path)            # every 2 s during games
                    if t0 - last_pre >= PRE_EVERY - 0.2:
                        last_pre = t0
                        pre = [p for p in open_pairs if p["start_ts"] and p["start_ts"] - HOT_BEFORE <= t0 < p["start_ts"]]
                        await sample(client, http, pre, time.time(), path)
                    if t0 - last_slow > 60:
                        last_slow = t0
                        day = [p for p in open_pairs if p["start_ts"] and
                               0 < p["start_ts"] - HOT_BEFORE - t0 <= DAY_WINDOW]
                        await sample(client, http, day, time.time(), path)
                    if t0 - last_settle > 600:
                        last_settle = t0
                        await settle(client, t0, path, logger)
                    if t0 - last_report > 600 and (report_task is None or report_task.done()):
                        last_report = t0
                        report_task = asyncio.create_task(report_runner.run(
                            "src.poly_check", "POLY", logger,
                            env={"POLY_DB": path, "POLY_REPORT": os.path.join(os.path.dirname(path), "poly_report.json")}
                            if path else None))
                except Exception as e:
                    if logger:
                        logger.warning(f"POLY cycle failed: {e!r}")
                await asyncio.sleep(max(0.3, LIVE_EVERY - (time.time() - t0)))
    finally:
        await client.close()


async def probe(logger=None) -> Dict[str, Any]:
    """One discovery pass + one price read, printed: does matching work on the live sites?"""
    import httpx
    from src.clients.kalshi_client import KalshiClient
    client = KalshiClient()
    out: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "kalshi-poly-check"}) as http:
            for league, cfg in LEAGUES.items():
                try:
                    found, stats = await find_games(client, http, league)
                    stats["examples"] = [f"{p['kalshi_event']} <-> {p['poly_slug']} ({p['matched_how']})"
                                         for p in found[:2]]
                    if found:
                        p = found[0]
                        pr = await poly_prices(http, [p["poly_token_a"]])
                        stats["price_check"] = {p["team_a"]: pr.get(p["poly_token_a"])}
                    out[league] = stats
                except Exception as e:
                    out[league] = {"error": repr(e)[:300]}
    finally:
        await client.close()
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _mid(b, a) -> Optional[float]:
    return (b + a) / 2 if b and a and 0 < b <= a < 1 else None


def _clustered(vals: List[Tuple[str, float]]) -> Tuple[Optional[float], Optional[float], int]:
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(v for _, v in vals) / n
    by: Dict[str, float] = {}
    for k, v in vals:
        by[k] = by.get(k, 0.0) + (v - mean)
    k = len(by)
    if n < 2 or k < 2:
        return 100 * mean, None, k
    var = sum(x * x for x in by.values()) / (n * n) * k / (k - 1)
    return 100 * mean, 196 * math.sqrt(var), k


def side_views(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Both teams of one sample as (Kalshi bid/ask, Polymarket mid) views; 'a' wins if result_a == yes."""
    out = []
    for side in ("a", "b"):
        km = _mid(s[f"k{side}_bid"], s[f"k{side}_ask"])
        pm = _mid(s[f"p{side}_bid"], s[f"p{side}_ask"])
        out.append({"side": side, "k_bid": s[f"k{side}_bid"], "k_ask": s[f"k{side}_ask"], "k_mid": km, "p_mid": pm})
    return out


def gap_episodes(ss: List[Dict[str, Any]], threshold: float, max_gap_s: float = 20.0) -> List[Dict[str, Any]]:
    """Runs where Kalshi's ask for a team is below Polymarket's middle price by more than fee + threshold."""
    out, cur, last = [], {"a": None, "b": None}, None
    for s in ss:
        if last is not None and s["ts"] - last > max_gap_s:
            out.extend(e for e in cur.values() if e)
            cur = {"a": None, "b": None}
        last = s["ts"]
        for v in side_views(s):
            ask, pm = v["k_ask"], v["p_mid"]
            edge = (pm - ask - taker_fee(ask)) if pm is not None and ask and 0 < ask < 1 else None
            if edge is not None and edge >= threshold:
                if cur[v["side"]] is None:
                    cur[v["side"]] = {"side": v["side"], "start": s["ts"], "end": s["ts"], "live": s["live"],
                                      "max_edge": edge}
                else:
                    cur[v["side"]]["end"] = s["ts"]
                    cur[v["side"]]["max_edge"] = max(cur[v["side"]]["max_edge"], edge)
            elif cur[v["side"]] is not None:
                out.append(cur[v["side"]])
                cur[v["side"]] = None
    out.extend(e for e in cur.values() if e)
    for e in out:
        e["duration"] = e["end"] - e["start"]
    return out


def late_tolerance(delay: float) -> float:
    """How far past N seconds a recorded price may be and still count as 'N s late'. Tight, so older
    5-second data is never passed off as '2 s late'."""
    return max(2.0, 0.5 * delay)


def buy_late(ep: Dict[str, Any], ss: List[Dict[str, Any]], won_a: bool, delay: float) -> Optional[float]:
    for s in ss:
        if s["ts"] >= ep["start"] + delay:
            if s["ts"] > ep["start"] + delay + late_tolerance(delay):
                return None
            ask = s[f"k{ep['side']}_ask"]
            if not ask or not 0 < ask < 1:
                return None
            won = won_a if ep["side"] == "a" else not won_a
            return (1.0 if won else 0.0) - ask - taker_fee(ask)
    return None


def poly_jumps(ss: List[Dict[str, Any]], min_jump: float = 0.03,
               waits=(0, 2, 5, 10, 15, 30, 60)) -> List[Dict[str, Any]]:
    """When Polymarket's middle price for team a jumps 3c+ between samples ~5 s apart: how much of it
    Kalshi's middle price has followed after N seconds, and how much Polymarket kept."""
    out = []
    n = len(ss)
    for i in range(1, n):
        a, b = ss[i - 1], ss[i]
        if not 0 < b["ts"] - a["ts"] <= 12:
            continue
        pa0, pa1 = _mid(a["pa_bid"], a["pa_ask"]), _mid(b["pa_bid"], b["pa_ask"])
        k0 = _mid(a["ka_bid"], a["ka_ask"])
        if pa0 is None or pa1 is None or k0 is None or abs(pa1 - pa0) < min_jump:
            continue
        jump = pa1 - pa0
        rec: Dict[str, Any] = {"jump": jump}
        j = i
        for w in waits:
            while j < n and ss[j]["ts"] < b["ts"] + w:
                j += 1
            if j >= n or ss[j]["ts"] > b["ts"] + w + late_tolerance(w):
                break
            km = _mid(ss[j]["ka_bid"], ss[j]["ka_ask"])
            pm = _mid(ss[j]["pa_bid"], ss[j]["pa_ask"])
            if km is not None:
                rec[f"k{w}"] = (km - k0) / jump
            if pm is not None:
                rec[f"p{w}"] = (pm - pa0) / jump
        out.append(rec)
    return out


def preview(path: Optional[str] = None) -> Dict[str, Any]:
    """What the prices recorded so far look like, before any game has finished: how far apart the two
    sites usually are, and how often Kalshi's ask sits below Polymarket's middle by more than the fee."""
    diffs: List[float] = []
    gaps2 = gaps5 = n = 0
    games, live_games = set(), set()
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        for s in c.execute("SELECT * FROM samples"):
            s = dict(s)
            km, pm = _mid(s["ka_bid"], s["ka_ask"]), _mid(s["pa_bid"], s["pa_ask"])
            if km is None or pm is None:
                continue
            n += 1
            games.add(s["pair_id"])
            if s["live"]:
                live_games.add(s["pair_id"])
            diffs.append(abs(km - pm) * 100)
            best = 0.0
            for v in side_views(s):
                if v["p_mid"] is not None and v["k_ask"] and 0 < v["k_ask"] < 1:
                    best = max(best, v["p_mid"] - v["k_ask"] - taker_fee(v["k_ask"]))
            gaps2 += best >= 0.02
            gaps5 += best >= 0.05
    if not n:
        return {"samples_with_both_prices": 0}
    return {"samples_with_both_prices": n, "games_sampled": len(games), "games_sampled_live": len(live_games),
            "median_diff_c": statistics.median(diffs), "share_diff_over_3c": sum(d > 3 for d in diffs) / n,
            "share_gap_2c": gaps2 / n, "share_gap_5c": gaps5 / n}


def report(path: Optional[str] = None, thresholds=(0.02, 0.05), delays=(0, 2, 5, 10, 15, 30),
           waits=(0, 2, 5, 10, 15, 30, 60)) -> Dict[str, Any]:
    if not os.path.exists(path or DB_PATH):
        return {"samples": 0}
    with closing(_conn(path)) as c:
        n_samples = c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        span = c.execute("SELECT MIN(ts), MAX(ts) FROM samples").fetchone()
        disc = c.execute("SELECT info FROM discovery ORDER BY ts DESC LIMIT 1").fetchone()
    allp = pairs(path)
    settled = [p for p in allp if p["result_a"] in ("yes", "no")]
    hours = ((span[1] or 0) - (span[0] or 0)) / 3600 if span and span[0] else 0.0
    cal: Dict[str, List[Tuple[float, float]]] = {}
    eps: Dict[float, list] = {th: [] for th in thresholds}
    jumps: List[Dict[str, Any]] = []
    with closing(_conn(path)) as c:
        c.row_factory = sqlite3.Row
        for p in settled:
            ss = [dict(r) for r in c.execute("SELECT * FROM samples WHERE pair_id=? ORDER BY ts", (p["pair_id"],))]
            if not ss:
                continue
            won_a = p["result_a"] == "yes"
            for th in thresholds:
                for e in gap_episodes(ss, th):
                    eps[th].append((p["pair_id"], e, {d: buy_late(e, ss, won_a, d) for d in delays}))
            jumps.extend(poly_jumps(ss, waits=waits))
            last_t = -1e18
            for s in ss:
                if s["ts"] - last_t < 60:                         # one sample a minute for accuracy
                    continue
                last_t = s["ts"]
                ya = 1.0 if won_a else 0.0
                km, pm = _mid(s["ka_bid"], s["ka_ask"]), _mid(s["pa_bid"], s["pa_ask"])
                if km is None or pm is None:
                    continue
                for b in ("during the game" if s["live"] else "before the game", "all"):
                    cal.setdefault(b, []).append(((pm - ya) ** 2, (km - ya) ** 2))
    calibration = [{"when": b, "samples": len(v), "Polymarket Brier": sum(x for x, _ in v) / len(v),
                    "Kalshi Brier": sum(y for _, y in v) / len(v)} for b, v in sorted(cal.items())]
    gaps = []
    for th in thresholds:
        for live_lab, keep in (("any time", lambda e: True), ("during the game", lambda e: e["live"]),
                               ("before the game", lambda e: not e["live"])):
            lst = [(pid, e, sim) for pid, e, sim in eps[th] if keep(e)]
            row: Dict[str, Any] = {"gaps": f"{th * 100:.0f}c+ after fee, {live_lab}", "count": len(lst),
                                   "games": len({pid for pid, _, _ in lst}),
                                   "median seconds": statistics.median([e["duration"] for _, e, _ in lst]) if lst else None}
            for d in delays:
                mean, margin, _ = _clustered([(pid, sim[d]) for pid, _, sim in lst if sim[d] is not None])
                row[f"{d}s late: c/contract"] = mean
                row[f"{d}s late: ± (c)"] = margin
            gaps.append(row)
    lag = []
    for w in waits:
        ks = [max(-1.0, min(2.0, j[f"k{w}"])) for j in jumps if f"k{w}" in j]
        ps = [max(-1.0, min(2.0, j[f"p{w}"])) for j in jumps if f"p{w}" in j]
        lag.append({"seconds after the jump": w, "jumps": len(ks),
                    "share Kalshi has followed": sum(ks) / len(ks) if ks else None,
                    "share Polymarket kept": sum(ps) / len(ps) if ps else None})
    best = None
    for r in gaps:
        v, m = r.get("5s late: c/contract"), r.get("5s late: ± (c)")
        if v is not None and m is not None and (best is None or v - m > best[0]):
            best = (v - m, r["gaps"], v, m)
    return {"samples": n_samples, "hours": hours, "games_matched": len(allp), "games_settled": len(settled),
            "by_league": {lg: sum(1 for p in allp if p["league"] == lg) for lg in LEAGUES},
            "discovery": json.loads(disc[0]) if disc else {}, "calibration": calibration, "gaps": gaps,
            "lag": lag, "jumps": len(jumps), "computed_ts": time.time(), "preview": preview(path),
            "summary_line": f"{n_samples:,} samples, {len(allp)} games matched, {len(settled)} settled"
                            + (f"; best at 5s late: {best[1]} {best[2]:+.2f}c ±{best[3]:.2f}" if best else "")}


def write_report(path: Optional[str] = None, out: Optional[str] = None) -> Dict[str, Any]:
    from src import report_runner
    rep = report(path)
    report_runner.write_json(rep, out or REPORT_PATH)
    return rep


def read_report(out: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from src import report_runner
    return report_runner.read_json(out or REPORT_PATH)
