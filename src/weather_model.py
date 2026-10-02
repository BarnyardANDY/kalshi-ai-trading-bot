"""
Daily high/low temperature markets, priced from weather forecasts (no AI).

Kalshi lists each city-day as a set of ranges ("77° or above", "75° to 76°",
"68° or below") that settle on the official daily climate report for one
weather station (e.g. CLINYC = Central Park). For each city-day we:

1. Read the station and whether it's the daily HIGH or LOW from the rules.
2. Forecast that station's temperature for the date from several free
   sources: Open-Meteo's multi-model forecast (GFS, ECMWF, ICON, GEM) and the
   National Weather Service point forecast.
3. Turn the forecast into a probability for each range with a normal
   distribution: the center is the blended forecast, the spread is the
   disagreement between models, never below a floor that grows with lead time
   (typical forecast error is ~2°F a day ahead, more further out).
4. For TODAY, use temperatures already observed at the station: the day's high
   can't end below the highest reading so far (and the low can't end above
   the lowest), so ranges that are already impossible get ~0%.

Settlement values are whole degrees Fahrenheit, so "75° to 76°" means the
reported value is 75 or 76, i.e. the continuous temperature is in [74.5, 76.5).

Everything network-related is best-effort and cached; a missing source just
means a wider spread or a skipped city.
"""
from __future__ import annotations

import math
import re
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

# Climate-report code (as written in Kalshi's rules) -> (ICAO station, lat, lon, tz)
STATIONS: Dict[str, Tuple[str, float, float, str]] = {
    "CLINYC": ("KNYC", 40.7789, -73.9692, "America/New_York"),
    "CLIMDW": ("KMDW", 41.7868, -87.7522, "America/Chicago"),
    "CLIMIA": ("KMIA", 25.7932, -80.2906, "America/New_York"),
    "CLIAUS": ("KAUS", 30.1945, -97.6699, "America/Chicago"),
    "CLIDEN": ("KDEN", 39.8466, -104.6562, "America/Denver"),
    "CLILAX": ("KLAX", 33.9382, -118.3866, "America/Los_Angeles"),
    "CLIPHL": ("KPHL", 39.8733, -75.2268, "America/New_York"),
    "CLIBOS": ("KBOS", 42.3606, -71.0097, "America/New_York"),
    "CLIDCA": ("KDCA", 38.8483, -77.0341, "America/New_York"),
    "CLIATL": ("KATL", 33.6301, -84.4418, "America/New_York"),
    "CLIDFW": ("KDFW", 32.8998, -97.0403, "America/Chicago"),
    "CLIPHX": ("KPHX", 33.4278, -112.0038, "America/Phoenix"),
    "CLISEA": ("KSEA", 47.4447, -122.3144, "America/Los_Angeles"),
    "CLISFO": ("KSFO", 37.6197, -122.3656, "America/Los_Angeles"),
    "CLIMSP": ("KMSP", 44.8831, -93.2289, "America/Chicago"),
    "CLILAS": ("KLAS", 36.0719, -115.1634, "America/Los_Angeles"),
    "CLIMSY": ("KMSY", 29.9934, -90.2580, "America/Chicago"),
    "CLIOKC": ("KOKC", 35.3931, -97.6007, "America/Chicago"),
    "CLISAT": ("KSAT", 29.5337, -98.4698, "America/Chicago"),
    "CLIHOU": ("KHOU", 29.6375, -95.2825, "America/Chicago"),
    "CLIEWR": ("KEWR", 40.6925, -74.1687, "America/New_York"),
    "CLISAN": ("KSAN", 32.7336, -117.1831, "America/Los_Angeles"),
    "CLITTN": ("KTTN", 40.2767, -74.8135, "America/New_York"),
}

_UA = "kalshi-ai-trading-bot (weather research)"
_TIMEOUT = 12.0
_CACHE_TTL = 30 * 60
_cache: Dict[str, Tuple[float, Any]] = {}
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


# ----------------------------------------------------------------------------
# Parsing Kalshi markets
# ----------------------------------------------------------------------------

def parse_station(rules: str) -> Optional[str]:
    m = re.search(r"\((CLI[A-Z]{3,4})\)", rules or "")
    return m.group(1) if m else None


def parse_kind(rules: str) -> Optional[str]:
    r = (rules or "").lower()
    if "maximum temperature" in r or "highest temperature" in r:
        return "high"
    if "minimum temperature" in r or "lowest temperature" in r:
        return "low"
    return None


def parse_event_date(event_or_ticker: str) -> Optional[date]:
    """KXHIGHNY-26SEP29(-...) -> 2026-09-29."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)", event_or_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def parse_range(title: str = "", market: Optional[Dict[str, Any]] = None) -> Optional[Tuple[float, float]]:
    """Inclusive whole-degree range (lo, hi) that resolves YES; inf for open ends."""
    if market:
        st = (market.get("strike_type") or "").lower()
        fl, cap = market.get("floor_strike"), market.get("cap_strike")
        if st == "greater" and fl is not None:
            return (math.floor(float(fl)) + 1, math.inf)
        if st == "less" and (cap is not None or fl is not None):
            return (-math.inf, math.ceil(float(cap if cap is not None else fl)) - 1)
        if st == "between" and fl is not None and cap is not None:
            return (float(fl), float(cap))
    t = title or ""
    m = re.search(r">\s*(-?\d+)°", t)
    if m:
        return (int(m.group(1)) + 1, math.inf)
    m = re.search(r"<\s*(-?\d+)°", t)
    if m:
        return (-math.inf, int(m.group(1)) - 1)
    m = re.search(r"(-?\d+)\s*-\s*(-?\d+)°", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(-?\d+)°\s*or above", t)
    if m:
        return (int(m.group(1)), math.inf)
    m = re.search(r"(-?\d+)°\s*or below", t)
    if m:
        return (-math.inf, int(m.group(1)))
    m = re.search(r"(-?\d+)°\s*to\s*(-?\d+)°", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


# ----------------------------------------------------------------------------
# Probability model (pure)
# ----------------------------------------------------------------------------

def _cdf(x: float, mu: float, sigma: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def range_probability(lo: float, hi: float, mu: float, sigma: float,
                      kind: str = "high", observed: Optional[float] = None) -> float:
    """P(reported whole-degree value in [lo, hi]).

    ``observed`` is the extreme seen so far today: for a high, the final
    value is at least it; for a low, at most it. The forecast distribution is
    truncated accordingly.
    """
    a = lo - 0.5 if lo != -math.inf else -math.inf
    b = hi + 0.5 if hi != math.inf else math.inf
    if observed is not None:
        if kind == "high":
            a = max(a, observed - 0.5)
            lower, upper = observed - 0.5, math.inf
        else:
            b = min(b, observed + 0.5)
            lower, upper = -math.inf, observed + 0.5
        if a >= b:
            return 0.0
        mass = _cdf(upper, mu, sigma) - _cdf(lower, mu, sigma)
        if mass < 1e-6:
            # Forecast is badly off (obs already beyond it): recentre on obs.
            mu = observed
            mass = _cdf(upper, mu, sigma) - _cdf(lower, mu, sigma)
        return max(0.0, min(1.0, (_cdf(b, mu, sigma) - _cdf(a, mu, sigma)) / mass))
    return max(0.0, min(1.0, _cdf(b, mu, sigma) - _cdf(a, mu, sigma)))


def spread_floor(lead_days: int) -> float:
    return {0: 1.8, 1: 2.5, 2: 3.3}.get(lead_days, 4.2 if lead_days > 2 else 1.8)


NWS_WEIGHT = 0.65        # NWS point forecasts are tuned to the exact location
OUTLIER_F = 6.0          # model values this far from the anchor are dropped


def blend(model_values: List[float], nws_value: Optional[float], lead_days: int) -> Optional[Tuple[float, float]]:
    """(mu, sigma) from global-model forecasts and the NWS point forecast.

    Global models run on coarse grids and can misplace a station's local
    climate badly (coastal LAX vs inland LA: one model said 102F when the NWS
    said 89F). So: anchor on the NWS forecast when present, drop model values
    more than OUTLIER_F away from it, use the MEDIAN of the rest, and weight
    the NWS forecast more heavily. Without an NWS value, anchor on the model
    median.
    """
    vals = [v for v in model_values if v is not None]
    if not vals and nws_value is None:
        return None
    anchor = nws_value if nws_value is not None else statistics.median(vals)
    kept = [v for v in vals if abs(v - anchor) <= OUTLIER_F]
    model_mid = statistics.median(kept) if kept else None
    if model_mid is not None and nws_value is not None:
        mu = NWS_WEIGHT * nws_value + (1 - NWS_WEIGHT) * model_mid
    else:
        mu = model_mid if model_mid is not None else nws_value
    pts = kept + ([nws_value] if nws_value is not None else [])
    disagreement = statistics.pstdev(pts) if len(pts) > 1 else 0.0
    sigma = max(spread_floor(lead_days), 1.2 * disagreement)
    return mu, sigma


# ----------------------------------------------------------------------------
# Data sources (network, cached, best-effort)
# ----------------------------------------------------------------------------

async def _get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True,
                                     headers={"User-Agent": _UA, **(headers or {})}) as c:
            r = await c.get(url)
            if r.status_code == 200:
                data = r.json()
                _cache[url] = (time.time(), data)
                return data
    except Exception:
        pass
    return None


def parse_open_meteo(data: Dict[str, Any], kind: str) -> Dict[date, List[float]]:
    """{date: [value per model]} from a multi-model Open-Meteo daily response."""
    daily = (data or {}).get("daily") or {}
    days = daily.get("time") or []
    key = "temperature_2m_max" if kind == "high" else "temperature_2m_min"
    out: Dict[date, List[float]] = {}
    for k, series in daily.items():
        if not k.startswith(key) or not isinstance(series, list):
            continue
        for d, v in zip(days, series):
            if v is not None:
                out.setdefault(date.fromisoformat(d), []).append(float(v))
    return out


def parse_nws_forecast(data: Dict[str, Any], kind: str) -> Dict[date, float]:
    """{date: forecast} from an NWS /forecast response.

    Daytime periods carry the high for their date. A night period starting on
    the evening of D carries the low that usually occurs on the morning of D+1.
    """
    out: Dict[date, float] = {}
    for p in ((data or {}).get("properties") or {}).get("periods") or []:
        try:
            start = datetime.fromisoformat(p["startTime"])
            temp = float(p["temperature"])
        except (KeyError, ValueError, TypeError):
            continue
        if (p.get("temperatureUnit") or "F") == "C":
            temp = temp * 9 / 5 + 32
        if kind == "high" and p.get("isDaytime"):
            out.setdefault(start.date(), temp)
        elif kind == "low" and not p.get("isDaytime"):
            out.setdefault(start.date() + timedelta(days=1), temp)
    return out


def standard_time_zone(day: date, tz: str) -> timezone:
    """Fixed LOCAL STANDARD time for ``tz`` (climate reports ignore daylight saving,
    so during DST the official 'day' runs 1am-1am on the wall clock)."""
    zone = ZoneInfo(tz)
    noon = datetime.combine(day, datetime.min.time().replace(hour=12), zone)
    return timezone(noon.utcoffset() - (noon.dst() or timedelta(0)))


def parse_observed_extreme(data: Dict[str, Any], day: date, tz: str, kind: str) -> Optional[float]:
    """Highest (or lowest) °F observed at the station so far on climate-day ``day``."""
    zone = standard_time_zone(day, tz)
    vals = []
    for f in (data or {}).get("features") or []:
        p = f.get("properties") or {}
        t = (p.get("temperature") or {}).get("value")
        if t is None:
            continue
        try:
            ts = datetime.fromisoformat(p["timestamp"]).astimezone(zone)
        except (KeyError, ValueError):
            continue
        if ts.date() == day:
            vals.append(float(t) * 9 / 5 + 32)
    if not vals:
        return None
    return round(max(vals) if kind == "high" else min(vals))


async def fetch_inputs(code: str, day: date, kind: str) -> Dict[str, Any]:
    icao, lat, lon, tz = STATIONS[code]
    om = await _get_json(
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min"
        "&temperature_unit=fahrenheit&timezone=" + tz.replace("/", "%2F") +
        "&forecast_days=4&models=gfs_seamless,ecmwf_ifs025,icon_seamless,gem_seamless"
    )
    models = parse_open_meteo(om, kind).get(day, []) if om else []

    nws_val = None
    pts = await _get_json(f"https://api.weather.gov/points/{lat},{lon}")
    fc_url = ((pts or {}).get("properties") or {}).get("forecast")
    if fc_url:
        fc = await _get_json(fc_url)
        nws_val = parse_nws_forecast(fc, kind).get(day)

    observed = None
    today = datetime.now(ZoneInfo(tz)).date()
    if day == today:
        start = datetime.combine(day, datetime.min.time(), standard_time_zone(day, tz)).astimezone(timezone.utc)
        obs = await _get_json(
            f"https://api.weather.gov/stations/{icao}/observations?start="
            + start.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        observed = parse_observed_extreme(obs, day, tz, kind)
    return {"models": models, "nws": nws_val, "observed": observed,
            "lead": (day - today).days, "tz": tz, "icao": icao}


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------

def _close_passed(market: Dict[str, Any]) -> bool:
    ct = market.get("close_time")
    if not ct:
        return False
    try:
        return datetime.fromisoformat(ct.replace("Z", "+00:00")) <= datetime.now(timezone.utc)
    except ValueError:
        return False


async def forecast_for_event(sample_market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Forecast summary for the city-day a market belongs to, or None."""
    rules = sample_market.get("rules_primary") or ""
    code, kind = parse_station(rules), parse_kind(rules)
    day = parse_event_date(sample_market.get("event_ticker") or sample_market.get("ticker") or "")
    if not code or code not in STATIONS or not kind or not day:
        return None
    inp = await fetch_inputs(code, day, kind)
    if inp["lead"] < 0:
        return None  # the day is over; the value is known, nothing to forecast
    bl = blend(inp["models"], inp["nws"], inp["lead"])
    if not bl:
        return None
    mu, sigma = bl
    phase = ""
    if inp["lead"] == 0:
        hour = local_hour(inp["tz"])
        mu, sigma, phase = same_day_adjust(kind, mu, sigma, hour, inp.get("observed"))
    return {"code": code, "kind": kind, "day": day, "mu": mu, "sigma": sigma, "phase": phase, **inp}


def local_hour(tz: str, now: Optional[datetime] = None) -> float:
    t = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(tz))
    return t.hour + t.minute / 60


def same_day_adjust(kind: str, mu: float, sigma: float, hour: float,
                    observed: Optional[float]) -> Tuple[float, float, str]:
    """Shrink same-day uncertainty as the day plays out.

    Highs usually peak mid/late afternoon: by ~4-5pm local the day's high is
    essentially the highest reading so far (the official value can be a degree
    above hourly readings, which miss brief peaks). Lows usually happen near
    dawn, so by mid-morning the low is mostly set (though a late cold front can
    still lower it before midnight).
    """
    def lerp(h, h0, h1, v0, v1):
        if h <= h0:
            return v0
        if h >= h1:
            return v1
        return v0 + (v1 - v0) * (h - h0) / (h1 - h0)

    if kind == "high":
        if observed is not None and hour >= 16.5:
            return observed + 0.4, 0.7, "after peak: high ~= observed"
        floor = lerp(hour, 10, 16.5, 1.8, 0.8)
        if observed is not None and observed > mu:
            mu = observed + 0.5  # already warmer than forecast
        # model disagreement matters less as observations accumulate
        return mu, max(floor, min(sigma, floor * 1.6)), f"same-day {hour:.0f}h"
    if observed is not None and hour >= 10:
        return min(mu, observed - 0.3), 1.0, "after dawn: low mostly set"
    floor = lerp(hour, 5, 10, 1.8, 1.0)
    if observed is not None and observed < mu:
        mu = observed - 0.5
    return mu, max(floor, min(sigma, floor * 1.6)), f"same-day {hour:.0f}h"


def describe(fc: Dict[str, Any]) -> str:
    models = ", ".join(f"{v:.0f}" for v in fc["models"]) or "none"
    obs = (f"; observed so far today: {fc['observed']:.0f}°F"
           if fc.get("observed") is not None else "")
    nws = f"{fc['nws']:.0f}°F" if fc.get("nws") is not None else "n/a"
    return (f"Forecast {fc['kind']} for {fc['icao']} on {fc['day']} (day+{fc['lead']}): "
            f"models [{models}]°F, NWS {nws} -> center {fc['mu']:.1f}°F, spread ±{fc['sigma']:.1f}{obs}"
            + (f" [{fc['phase']}]" if fc.get("phase") else ""))


async def predict_weather(markets, kalshi_client, logger) -> Dict[str, Tuple[float, float]]:
    """(probability, confidence) for weather markets, one forecast per city-day."""
    events: Dict[str, List] = {}
    for mk in markets:
        if parse_range(mk.title) is not None:
            events.setdefault(mk.market_id.rsplit("-", 1)[0], []).append(mk)

    out: Dict[str, Tuple[float, float]] = {}
    for event, rungs in events.items():
        try:
            sample = (await kalshi_client.get_market(rungs[0].market_id)).get("market", {})
        except Exception as e:
            logger.warning(f"WX {event}: could not load market details: {e}")
            continue
        if _close_passed(sample):
            continue
        fc = await forecast_for_event(sample)
        if not fc:
            logger.info(f"WX {event}: no station mapping or forecast available, skipping")
            continue
        conf = 0.8 if fc["lead"] <= 1 else 0.6
        parts = []
        for mk in rungs:
            rng = parse_range(mk.title)
            p = range_probability(rng[0], rng[1], fc["mu"], fc["sigma"], fc["kind"], fc.get("observed"))
            p = min(max(p, 0.01), 0.99)
            out[mk.market_id] = (p, conf)
            lab = (f"≥{rng[0]:.0f}" if rng[1] == math.inf else
                   f"≤{rng[1]:.0f}" if rng[0] == -math.inf else f"{rng[0]:.0f}-{rng[1]:.0f}")
            parts.append(f"{lab}: {p:.0%}")
        logger.info(f"WX {event}: {describe(fc)} -> " + ", ".join(parts))
    return out
