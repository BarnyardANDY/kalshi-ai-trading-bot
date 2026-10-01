"""
Sports: price Kalshi game-winner markets from the sportsbook consensus.

The idea is not to out-predict professional sportsbooks; it's to check
whether Kalshi's price LAGS them. For each Kalshi "X wins" market we find the
same game in The Odds API (https://the-odds-api.com, free key), take every
bookmaker's two-way moneyline, remove the bookmaker margin ("devig") so the
two sides sum to 100%, and average them, weighting Pinnacle (the sharpest
book) most. That consensus is the bot's probability.

Runs in SHADOW mode by default (see SHADOW_NICHES in runtime_config): the
prediction is recorded and graded by the learning loop against Kalshi's own
price, but no trades are placed until the record shows the consensus beats
Kalshi.

API usage is budgeted: one request per league per ODDS_REFRESH_MINUTES
(free tier = 500 requests/month). Set ODDS_API_KEY in .env.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

# Kalshi series -> (league key used in settings, The Odds API sport key)
SERIES_LEAGUE: Dict[str, Tuple[str, str]] = {
    "KXNFLGAME": ("nfl", "americanfootball_nfl"),
    "KXNCAAFGAME": ("ncaaf", "americanfootball_ncaaf"),
    "KXMLBGAME": ("mlb", "baseball_mlb"),
    "KXNBAGAME": ("nba", "basketball_nba"),
    "KXNHLGAME": ("nhl", "icehockey_nhl"),
}
ALL_LEAGUES = ["nfl", "ncaaf", "mlb", "nba", "nhl"]
BOOKMAKERS = "pinnacle,draftkings,fanduel,betmgm,williamhill_us,betrivers,bovada,lowvig"
SHARP_WEIGHT = {"pinnacle": 3.0, "lowvig": 1.5}
_ET = ZoneInfo("America/New_York")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

_odds_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_quota: Dict[str, Any] = {"remaining": None, "used": None}


# ----------------------------------------------------------------------------
# Kalshi side
# ----------------------------------------------------------------------------

def league_of(ticker: str) -> Optional[Tuple[str, str]]:
    return SERIES_LEAGUE.get((ticker or "").split("-", 1)[0].upper())


def norm(name: str) -> str:
    s = (name or "").lower().replace("&", "and")
    s = re.sub(r"\bst\.?(?=\s|$)", "state", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _abbr_nickname(side: str) -> Optional[str]:
    """'KC Chiefs' -> 'Chiefs' (NFL rules write teams as ABBR + nickname)."""
    m = re.match(r"^([A-Z]{2,4})\s+(\S.*)$", side.strip())
    return m.group(2).strip() if m else None


def _same_team(label: str, side: str) -> bool:
    a, b = norm(label), norm(side)
    return bool(a) and (a == b or b.startswith(a) or a.startswith(b))


def parse_game(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """{yes, no, date, nickname, no_nickname} for a Kalshi game-winner market."""
    rules = market.get("rules_primary") or ""
    yes = (market.get("yes_sub_title") or "").strip()
    if not yes:
        m = re.match(r"\s*(.+?) wins\b", market.get("title") or "")
        yes = m.group(1).strip() if m else ""
    m = re.search(r"wins the (.+?) vs (.+?) game\b", rules)
    sides = [m.group(1).strip(), m.group(2).strip()] if m else []
    if sides:
        # trailing sport words: "NO Saints Pro Football" -> "NO Saints"
        sides[1] = re.sub(r"\s+(Pro|College|professional|college)\b.*$", "", sides[1]).strip()

    # Which side is YES? Prefer the ticker's team code (e.g. -LV), then the name.
    abbr = (market.get("ticker") or "").rsplit("-", 1)[-1].upper()
    yes_idx = None
    for i, side in enumerate(sides):
        if side.split(" ", 1)[0].upper() == abbr and _abbr_nickname(side):
            yes_idx = i
    if yes_idx is None:
        for i, side in enumerate(sides):
            if _same_team(yes, side):
                yes_idx = i
    nickname = no_nickname = None
    no = ""
    if len(sides) == 2 and yes_idx is not None:
        no_side = sides[1 - yes_idx]
        nickname = _abbr_nickname(sides[yes_idx])
        no_nickname = _abbr_nickname(no_side)
        no = no_side
    # Kalshi's no_sub_title is often just the YES team again; only trust a real opponent.
    nst = (market.get("no_sub_title") or "").strip()
    if not no and nst and not _same_team(yes, nst):
        no = nst

    day = None
    d = re.search(r"scheduled for ([A-Z][a-z]{2})\w* (\d{1,2}), (\d{4})", rules)
    if d and d.group(1).upper() in _MONTHS:
        day = datetime(int(d.group(3)), _MONTHS[d.group(1).upper()], int(d.group(2))).date()
    if not day:
        t = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", market.get("event_ticker") or market.get("ticker") or "")
        if t and t.group(2) in _MONTHS:
            day = datetime(2000 + int(t.group(1)), _MONTHS[t.group(2)], int(t.group(3))).date()
    if not yes or not day:
        return None
    return {"yes": yes, "no": no, "date": day, "nickname": nickname, "no_nickname": no_nickname}


# ----------------------------------------------------------------------------
# Sportsbook side
# ----------------------------------------------------------------------------

def _team_matches(label: str, team: str, nickname: Optional[str] = None) -> bool:
    lt, tt = norm(label), norm(team)
    if not lt:
        return False
    if nickname and tt.endswith(norm(nickname)):
        return True
    if tt.startswith(lt):
        return True
    toks = lt.split()
    return len(toks) > 1 and all(t in tt.split() for t in toks)


def find_event(game: Dict[str, Any], events: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], str]]:
    """(odds event, odds name of the YES team) for a parsed Kalshi game."""
    hits = []
    for ev in events:
        try:
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if abs((start.astimezone(_ET).date() - game["date"]).days) > 1:
            continue
        teams = [ev.get("home_team", ""), ev.get("away_team", "")]
        yes_team = [t for t in teams if _team_matches(game["yes"], t, game.get("nickname"))]
        if len(yes_team) != 1:
            continue
        other = [t for t in teams if t != yes_team[0]][0]
        if game.get("no") and not _team_matches(game["no"], other, game.get("no_nickname")):
            continue
        same_day = start.astimezone(_ET).date() == game["date"]
        hits.append((0 if same_day else 1, ev, yes_team[0]))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None  # ambiguous (e.g. doubleheader); skip rather than guess
    return hits[0][1], hits[0][2]


def consensus(event: Dict[str, Any], yes_team: str) -> Optional[Dict[str, Any]]:
    """Devigged, Pinnacle-weighted P(yes_team wins) across bookmakers."""
    num = den = 0.0
    books = []
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != "h2h":
                continue
            prices = {o.get("name"): o.get("price") for o in mk.get("outcomes") or []}
            if len(prices) != 2 or yes_team not in prices or any(not p or p <= 1 for p in prices.values()):
                continue
            inv = {k: 1.0 / v for k, v in prices.items()}
            p = inv[yes_team] / sum(inv.values())
            w = SHARP_WEIGHT.get(bk.get("key"), 1.0)
            num += w * p
            den += w
            books.append(bk.get("key"))
    if not den:
        return None
    return {"prob": num / den, "books": books, "sharp": any(b in SHARP_WEIGHT for b in books)}


async def fetch_odds(sport_key: str, api_key: str, refresh_minutes: float) -> List[Dict[str, Any]]:
    hit = _odds_cache.get(sport_key)
    if hit and time.time() - hit[0] < refresh_minutes * 60:
        return hit[1]
    if _quota["remaining"] is not None and _quota["remaining"] <= 5:
        return hit[1] if hit else []
    url = (f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
           f"?apiKey={api_key}&markets=h2h&oddsFormat=decimal&bookmakers={BOOKMAKERS}")
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url)
        if r.status_code == 200:
            _quota["remaining"] = _int(r.headers.get("x-requests-remaining"))
            _quota["used"] = _int(r.headers.get("x-requests-used"))
            data = r.json()
            _odds_cache[sport_key] = (time.time(), data)
            return data
    except Exception:
        pass
    # On failure keep serving the last good copy and don't retry immediately
    _odds_cache[sport_key] = (time.time(), hit[1] if hit else [])
    return hit[1] if hit else []


def _int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def quota() -> Dict[str, Any]:
    return dict(_quota)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

async def predict_sports(markets, kalshi_client, logger) -> Dict[str, Tuple[float, float]]:
    """(probability, confidence) for Kalshi game markets from sportsbook consensus."""
    from src import runtime_config as rc

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        logger.info("SPORTS: no ODDS_API_KEY in .env; skipping sports (get a free key at the-odds-api.com)")
        return {}
    leagues = {x for x in str(rc.get("SPORTS_LEAGUES")).split(",") if x}
    refresh = float(rc.get("ODDS_REFRESH_MINUTES"))

    out: Dict[str, Tuple[float, float]] = {}
    matched = unmatched = 0
    now = datetime.now(timezone.utc)
    for mk in markets:
        lg = league_of(mk.market_id)
        if not lg or lg[0] not in leagues:
            continue
        try:
            info = (await kalshi_client.get_market(mk.market_id)).get("market", {})
        except Exception:
            continue
        game = parse_game(info)
        if not game:
            unmatched += 1
            continue
        events = await fetch_odds(lg[1], api_key, refresh)
        found = find_event(game, events)
        if not found:
            unmatched += 1
            continue
        ev, yes_team = found
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if start <= now:
            continue  # pre-game only; live odds move faster than our refresh
        cons = consensus(ev, yes_team)
        if not cons:
            unmatched += 1
            continue
        matched += 1
        conf = 0.85 if cons["sharp"] else 0.7
        p = min(max(cons["prob"], 0.01), 0.99)
        out[mk.market_id] = (p, conf)
        logger.info(
            f"SPORTS {mk.market_id}: books {p:.1%} ({len(cons['books'])} books"
            f"{', incl. Pinnacle' if 'pinnacle' in cons['books'] else ''}) vs Kalshi ~{mk.yes_price:.0%} "
            f"[{ev.get('away_team')} @ {ev.get('home_team')}, {start.astimezone(_ET):%a %b %d %I:%M%p} ET]"
        )
    if matched or unmatched:
        q = quota()
        logger.info(f"SPORTS: matched {matched} markets to sportsbook odds, {unmatched} unmatched; "
                    f"Odds API requests left this month: {q['remaining']}")
    return out
