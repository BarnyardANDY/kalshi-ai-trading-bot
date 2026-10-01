"""
Stock-index markets (S&P 500, Nasdaq-100), priced from the options market.

Kalshi lists the index level at a set time (4pm close, or hourly) as ranges
("between 8,050 and 8,074.99") and above/below lines. Options traders already
publish how much they expect the index to move: the VIX (S&P 500) and VXN
(Nasdaq-100) are annualized implied volatilities. From the current index
level and that volatility, a lognormal model gives the probability of every
range at the target time, the same way the weather model turns a forecast
and its uncertainty into range odds.

The test is whether Kalshi's prices LAG that options-implied view. Runs in
shadow mode by default (graded, never traded).

Time to the target is measured in trading time: minutes of regular sessions
(9:30-16:00 ET, weekdays) remaining, plus a share of a day's variance for
each overnight gap. Market holidays are ignored (a small overstatement of
uncertainty on those days).

Quotes come from Yahoo Finance's public chart endpoint; during market hours a
quote older than 20 minutes is treated as stale and the market is skipped.
"""
from __future__ import annotations

import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

ET = ZoneInfo("America/New_York")
# Kalshi series -> (index symbol, implied-vol symbol, label)
SERIES: Dict[str, Tuple[str, str, str]] = {
    "KXINX": ("^GSPC", "^VIX", "S&P 500"),
    "KXINXU": ("^GSPC", "^VIX", "S&P 500"),
    "KXNASDAQ100": ("^NDX", "^VXN", "Nasdaq-100"),
    "KXNASDAQ100U": ("^NDX", "^VXN", "Nasdaq-100"),
}
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
SESSION_MIN = 390.0
OVERNIGHT_SHARE = 0.2  # share of a day's variance that arrives overnight
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
_cache: Dict[str, Tuple[float, Any]] = {}


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

def series_info(ticker: str) -> Optional[Tuple[str, str, str]]:
    return SERIES.get((ticker or "").split("-", 1)[0].upper())


def parse_target(ticker: str) -> Optional[datetime]:
    """KXINX-26AUG19H1600-B8062 -> 2026-08-19 16:00 ET."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})H(\d{2})(\d{2})", ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return datetime(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)),
                        int(m.group(4)), int(m.group(5)), tzinfo=ET)
    except ValueError:
        return None


def parse_range(title: str = "", market: Optional[Dict[str, Any]] = None) -> Optional[Tuple[float, float]]:
    """(low, high) index levels for which the market resolves YES."""
    if market:
        st = (market.get("strike_type") or "").lower()
        fl, cap = market.get("floor_strike"), market.get("cap_strike")
        if st.startswith("greater") and fl is not None:
            return (float(fl), math.inf)
        if st.startswith("less") and (cap is not None or fl is not None):
            return (-math.inf, float(cap if cap is not None else fl))
        if st == "between" and fl is not None and cap is not None:
            return (float(fl), float(cap))
    t = (title or "").replace(",", "")
    m = re.search(r"between\s+([\d.]+)\s+and\s+([\d.]+)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"\babove\s+([\d.]+)", t)
    if m:
        return (float(m.group(1)), math.inf)
    m = re.search(r"\bbelow\s+([\d.]+)", t)
    if m:
        return (-math.inf, float(m.group(1)))
    return None


# ----------------------------------------------------------------------------
# Model (pure)
# ----------------------------------------------------------------------------

def _sessions(day: date) -> Tuple[datetime, datetime]:
    return (datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET),
            datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET))


def variance_days(now: datetime, target: datetime) -> float:
    """Trading-day-equivalents of variance between now and target (ET)."""
    now, target = now.astimezone(ET), target.astimezone(ET)
    if target <= now:
        return 0.0
    minutes = 0.0
    overnights = 0
    d = now.date()
    while d <= target.date():
        if d.weekday() < 5:
            open_, close = _sessions(d)
            lo, hi = max(open_, now), min(close, target)
            if hi > lo:
                minutes += (hi - lo).total_seconds() / 60
            # an overnight gap is crossed if this session's open lies after now and at/before target
            if now < open_ <= target:
                overnights += 1
        d += timedelta(days=1)
    return (1 - OVERNIGHT_SHARE) * minutes / SESSION_MIN + OVERNIGHT_SHARE * overnights


def range_probability(lo: float, hi: float, spot: float, vol_annual: float, var_days: float) -> float:
    """P(lo < X < hi) for a driftless lognormal index level."""
    if var_days <= 0:
        return 1.0 if lo < spot < hi else 0.0
    s = vol_annual * math.sqrt(var_days / 252.0)
    mu = math.log(spot) - 0.5 * s * s

    def cdf(x: float) -> float:
        if x == math.inf:
            return 1.0
        if x == -math.inf or x <= 0:
            return 0.0
        return 0.5 * (1 + math.erf((math.log(x) - mu) / (s * math.sqrt(2))))

    return max(0.0, min(1.0, cdf(hi) - cdf(lo)))


# ----------------------------------------------------------------------------
# Quotes (network, cached)
# ----------------------------------------------------------------------------

def parse_yahoo_chart(data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(price, unix time) from a Yahoo /v8/finance/chart response."""
    try:
        meta = data["chart"]["result"][0]["meta"]
        return float(meta["regularMarketPrice"]), float(meta.get("regularMarketTime") or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


async def fetch_quote(symbol: str, ttl: float = 60.0) -> Optional[Tuple[float, float]]:
    hit = _cache.get(symbol)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('^', '%5E')}?interval=1m&range=1d"
    q = None
    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": _UA}) as c:
            r = await c.get(url)
        if r.status_code == 200:
            q = parse_yahoo_chart(r.json())
    except Exception:
        q = None
    if q:
        _cache[symbol] = (time.time(), q)
        return q
    return hit[1] if hit else None


def market_open(now: datetime) -> bool:
    now = now.astimezone(ET)
    if now.weekday() >= 5:
        return False
    o, c = _sessions(now.date())
    return o <= now <= c


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------

async def snapshot(index_sym: str, vol_sym: str, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Current index level and implied vol, or None if missing/stale."""
    now = now or datetime.now(timezone.utc)
    spot = await fetch_quote(index_sym, ttl=60)
    vol = await fetch_quote(vol_sym, ttl=600)
    if not spot or not vol or spot[0] <= 0 or vol[0] <= 0:
        return None
    age_min = (now.timestamp() - spot[1]) / 60 if spot[1] else None
    if market_open(now) and (age_min is None or age_min > 20):
        return None
    return {"spot": spot[0], "vol": vol[0] / 100.0, "quote_age_min": age_min}


async def predict_stocks(markets, kalshi_client, logger) -> Dict[str, Tuple[float, float]]:
    """(probability, confidence) for index range markets, one view per target time."""
    from src import runtime_config as rc

    scale = float(rc.get("STOCKS_VOL_SCALE"))
    events: Dict[str, List] = {}
    for mk in markets:
        if series_info(mk.market_id) and parse_target(mk.market_id) and parse_range(mk.title):
            events.setdefault(mk.market_id.rsplit("-", 1)[0], []).append(mk)

    now = datetime.now(timezone.utc)
    out: Dict[str, Tuple[float, float]] = {}
    for event, rungs in events.items():
        target = parse_target(rungs[0].market_id)
        if not target or target <= now:
            continue
        idx, vsym, label = series_info(rungs[0].market_id)
        snap = await snapshot(idx, vsym, now)
        if not snap:
            logger.info(f"STOCKS {event}: no fresh {label} / implied-vol quote, skipping")
            continue
        vd = variance_days(now, target)
        vol = snap["vol"] * scale
        parts = []
        for mk in rungs:
            lo, hi = parse_range(mk.title)
            p = min(max(range_probability(lo, hi, snap["spot"], vol, vd), 0.01), 0.99)
            out[mk.market_id] = (p, 0.8)
            parts.append(p)
        one_sd = snap["spot"] * vol * math.sqrt(vd / 252.0)
        logger.info(
            f"STOCKS {event}: {label} {snap['spot']:,.2f}, implied vol {vol:.1%} -> "
            f"±{one_sd:,.0f} pts (1 s.d.) by {target:%a %b %d %I:%M%p} ET; priced {len(parts)} ranges"
        )
    return out


async def describe(market: Dict[str, Any]) -> str:
    info = series_info(market.get("ticker") or "")
    target = parse_target(market.get("ticker") or "")
    rng = parse_range(market.get("title") or "", market)
    if not info or not target or not rng:
        return "Could not read the index, time or range from this market."
    from src import runtime_config as rc

    now = datetime.now(timezone.utc)
    if target <= now:
        return "Target time has passed."
    snap = await snapshot(info[0], info[1], now)
    if not snap:
        return f"No fresh {info[2]} or implied-volatility quote available."
    vol = snap["vol"] * float(rc.get("STOCKS_VOL_SCALE"))
    vd = variance_days(now, target)
    p = range_probability(rng[0], rng[1], snap["spot"], vol, vd)
    return (f"{info[2]} now {snap['spot']:,.2f}; options-implied vol {vol:.1%}; "
            f"{vd:.2f} trading days of variance to {target:%b %d %I:%M%p} ET -> {p:.1%} for this range.")
