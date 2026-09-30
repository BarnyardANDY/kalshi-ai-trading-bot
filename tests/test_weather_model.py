"""Weather range pricing from forecasts (offline)."""
import asyncio
import math
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src import weather_model as W
from src.utils.database import Market

RULES = ("If the maximum temperature recorded at New York City (CLINYC) for Sep 29, 2026, "
         "is greater than 76° fahrenheit according to The Weather Company, then the market resolves to Yes.")


def test_parse_rules_and_dates():
    assert W.parse_station(RULES) == "CLINYC"
    assert W.parse_kind(RULES) == "high"
    assert W.parse_kind("If the minimum temperature recorded at ...") == "low"
    assert W.parse_event_date("KXHIGHNY-26SEP29") == date(2026, 9, 29)
    assert W.parse_event_date("KXHIGHNY-26SEP29-T76") == date(2026, 9, 29)
    assert W.parse_event_date("KXRT-DIG-45") is None


def test_parse_ranges_from_titles_and_strikes():
    assert W.parse_range("Will the maximum temperature be >76° on Sep 29, 2026? — 77° or above") == (77, math.inf)
    assert W.parse_range("Will the maximum temperature be <69° on Sep 29, 2026? — 68° or below") == (-math.inf, 68)
    assert W.parse_range("Will the maximum temperature be 75-76° on Sep 29, 2026? — 75° to 76°") == (75, 76)
    assert W.parse_range(market={"strike_type": "greater", "floor_strike": 76}) == (77, math.inf)
    assert W.parse_range(market={"strike_type": "less", "cap_strike": 69}) == (-math.inf, 68)
    assert W.parse_range(market={"strike_type": "between", "floor_strike": 75, "cap_strike": 76}) == (75, 76)
    assert W.parse_range("Digger Rotten Tomatoes score? — Above 45") is None


def test_ranges_partition_to_one():
    mu, s = 72.3, 2.5
    buckets = [(-math.inf, 68), (69, 70), (71, 72), (73, 74), (75, 76), (77, math.inf)]
    total = sum(W.range_probability(lo, hi, mu, s) for lo, hi in buckets)
    assert total == pytest.approx(1.0, abs=1e-9)
    # the bucket holding the forecast is the favorite
    probs = [W.range_probability(lo, hi, mu, s) for lo, hi in buckets]
    assert max(probs) == probs[2]


def test_observed_high_rules_out_lower_ranges():
    # Already hit 75 today: nothing below 75 is possible anymore
    assert W.range_probability(-math.inf, 68, 72, 2.5, "high", observed=75) == 0.0
    assert W.range_probability(71, 72, 72, 2.5, "high", observed=75) == 0.0
    p75 = W.range_probability(75, 76, 72, 2.5, "high", observed=75)
    p77 = W.range_probability(77, math.inf, 72, 2.5, "high", observed=75)
    assert p75 + p77 == pytest.approx(1.0, abs=1e-9) and p75 > p77


def test_observed_low_caps_higher_ranges():
    assert W.range_probability(60, math.inf, 55, 2.0, "low", observed=52) == 0.0


def test_blend_and_spread():
    mu, s = W.blend([70, 72, 74, 72], 74, lead_days=1)
    assert mu == pytest.approx(0.5 * 72 + 0.5 * 74)
    assert s == W.spread_floor(1)  # models agree closely -> floor applies
    mu, s = W.blend([60, 80], None, lead_days=0)
    assert s == pytest.approx(1.2 * 10)  # big disagreement widens the spread
    assert W.blend([], None, 1) is None


def test_parse_sources():
    om = {"daily": {"time": ["2026-09-29", "2026-09-30"],
                    "temperature_2m_max_gfs_seamless": [74.1, 70.0],
                    "temperature_2m_max_ecmwf_ifs025": [75.3, None],
                    "temperature_2m_min_gfs_seamless": [60.0, 58.0]}}
    assert W.parse_open_meteo(om, "high") == {date(2026, 9, 29): [74.1, 75.3], date(2026, 9, 30): [70.0]}
    nws = {"properties": {"periods": [
        {"startTime": "2026-09-29T06:00:00-04:00", "isDaytime": True, "temperature": 75, "temperatureUnit": "F"},
        {"startTime": "2026-09-29T18:00:00-04:00", "isDaytime": False, "temperature": 61, "temperatureUnit": "F"},
    ]}}
    assert W.parse_nws_forecast(nws, "high") == {date(2026, 9, 29): 75.0}
    assert W.parse_nws_forecast(nws, "low") == {date(2026, 9, 30): 61.0}
    obs = {"features": [
        {"properties": {"timestamp": "2026-09-29T14:00:00+00:00", "temperature": {"value": 22.0}}},
        {"properties": {"timestamp": "2026-09-29T18:00:00+00:00", "temperature": {"value": 24.4}}},
        {"properties": {"timestamp": "2026-09-29T19:00:00+00:00", "temperature": {"value": None}}},
    ]}
    assert W.parse_observed_extreme(obs, date(2026, 9, 29), "America/New_York", "high") == 76


def test_predict_weather_end_to_end(monkeypatch):
    tomorrow = datetime.now(ZoneInfo("America/New_York")).date() + timedelta(days=1)
    tag = tomorrow.strftime("%y%b%d").upper()
    event = f"KXHIGHNY-{tag}"

    async def fake_inputs(code, day, kind):
        return {"models": [72.0, 73.0, 72.5], "nws": 73.0, "observed": None, "lead": 1,
                "tz": "America/New_York", "icao": "KNYC"}
    monkeypatch.setattr(W, "fetch_inputs", fake_inputs)

    class K:
        async def get_market(self, t):
            return {"market": {"ticker": t, "event_ticker": event, "rules_primary": RULES,
                               "close_time": (datetime.now(ZoneInfo("UTC")) + timedelta(days=1)).isoformat()}}

    class Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass

    def mk(suffix, title):
        return Market(f"{event}-{suffix}", title, 0.2, 0.8, 500, int(time.time()) + 86400,
                      "weather", "active", datetime.now())
    ms = [mk("T76", "Will the maximum temperature be >76°? — 77° or above"),
          mk("B72.5", "Will the maximum temperature be 72-73°? — 72° to 73°"),
          mk("T69", "Will the maximum temperature be <69°? — 68° or below")]
    out = asyncio.run(W.predict_weather(ms, K(), Log()))
    assert set(out) == {m.market_id for m in ms}
    assert out[f"{event}-B72.5"][0] > out[f"{event}-T76"][0] > 0
    assert out[f"{event}-B72.5"][1] == 0.8
