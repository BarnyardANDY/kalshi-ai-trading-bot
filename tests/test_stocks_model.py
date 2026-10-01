"""Stock-index range pricing from implied volatility (offline)."""
import asyncio
import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from src import stocks_model as M
from src.utils.database import Market

ET = M.ET


def test_parse_target_and_ranges():
    assert M.parse_target("KXINX-26AUG19H1600-B8062") == datetime(2026, 8, 19, 16, 0, tzinfo=ET)
    assert M.parse_target("KXINXU-26OCT02H1100-T7869.9999") == datetime(2026, 10, 2, 11, 0, tzinfo=ET)
    assert M.parse_target("KXRT-DIG-45") is None
    assert M.parse_range("Will the S&P 500 be between 8050 and 8074.9999 on Aug 19? — 8,050 to 8,074.9999") == (8050, 8074.9999)
    assert M.parse_range("Will the S&P 500 be above 8074.9999 on Aug 19, 2026 at 4pm EDT?") == (8074.9999, math.inf)
    assert M.parse_range("Will the S&P 500 be below 7375 on Aug 19, 2026 at 4pm EDT?") == (-math.inf, 7375)
    assert M.parse_range(market={"strike_type": "greater_or_equal", "floor_strike": 7870}) == (7870, math.inf)
    assert M.series_info("KXNASDAQ100-26OCT02H1600-B1") == ("^NDX", "^VXN", "Nasdaq-100")
    assert M.series_info("KXINXY-26DEC31-T1") is None


def test_variance_days():
    mon_open = datetime(2026, 10, 5, 9, 30, tzinfo=ET)          # Monday
    assert M.variance_days(mon_open, datetime(2026, 10, 5, 16, 0, tzinfo=ET)) == pytest.approx(0.8)
    noon = datetime(2026, 10, 5, 12, 45, tzinfo=ET)
    assert M.variance_days(noon, datetime(2026, 10, 5, 16, 0, tzinfo=ET)) == pytest.approx(0.4)
    # Friday after close -> Monday close: one overnight (weekend) + one full session
    fri = datetime(2026, 10, 2, 17, 0, tzinfo=ET)
    assert M.variance_days(fri, datetime(2026, 10, 5, 16, 0, tzinfo=ET)) == pytest.approx(0.8 + 0.2)
    assert M.variance_days(noon, noon - timedelta(hours=1)) == 0.0


def test_ranges_sum_to_one_and_center_is_favorite():
    spot, vol, vd = 8000.0, 0.16, 1.0
    edges = [-math.inf] + [7900 + 25 * i for i in range(9)] + [math.inf]
    probs = [M.range_probability(lo, hi, spot, vol, vd) for lo, hi in zip(edges[:-1], edges[1:])]
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert max(probs) in (probs[4], probs[5])  # ranges around 8000
    # 1 s.d. for one day at 16% vol is ~1% of the index
    assert M.range_probability(-math.inf, spot * (1 - 0.16 / math.sqrt(252)), spot, vol, vd) == pytest.approx(0.16, abs=0.01)


def test_yahoo_parse():
    data = {"chart": {"result": [{"meta": {"regularMarketPrice": 7998.5, "regularMarketTime": 1790000000}}]}}
    assert M.parse_yahoo_chart(data) == (7998.5, 1790000000.0)
    assert M.parse_yahoo_chart({}) is None


def test_stale_quote_skipped_during_market_hours(monkeypatch):
    open_now = datetime(2026, 10, 5, 14, 0, tzinfo=ET).astimezone(timezone.utc)

    async def fake(sym, ttl=60):
        return (8000.0 if sym == "^GSPC" else 16.0, open_now.timestamp() - 3600)  # an hour old
    monkeypatch.setattr(M, "fetch_quote", fake)
    assert asyncio.run(M.snapshot("^GSPC", "^VIX", open_now)) is None

    async def fresh(sym, ttl=60):
        return (8000.0 if sym == "^GSPC" else 16.0, open_now.timestamp() - 60)
    monkeypatch.setattr(M, "fetch_quote", fresh)
    s = asyncio.run(M.snapshot("^GSPC", "^VIX", open_now))
    assert s["spot"] == 8000.0 and s["vol"] == pytest.approx(0.16)


def test_predict_stocks_end_to_end(monkeypatch):
    target = (datetime.now(ET) + timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0)
    tag = target.strftime("%y%b%d").upper() + "H1600"
    event = f"KXINX-{tag}"

    async def fake(sym, ttl=60):
        return (8000.0 if sym == "^GSPC" else 16.0, time.time())
    monkeypatch.setattr(M, "fetch_quote", fake)

    class Log:
        def info(self, *a, **k): pass

    def mk(suffix, title):
        return Market(f"{event}-{suffix}", title, 0.2, 0.8, 100, int(target.timestamp()), "stocks", "active",
                      datetime.now())
    ms = [mk("B8012", "Will the S&P 500 be between 8000 and 8024.9999?"),
          mk("T8500", "Will the S&P 500 be above 8500 on ...?"),
          mk("T7500", "Will the S&P 500 be below 7500 on ...?")]
    out = asyncio.run(M.predict_stocks(ms, None, Log()))
    assert set(out) == {m.market_id for m in ms}
    assert out[f"{event}-B8012"][0] > out[f"{event}-T8500"][0]
    assert out[f"{event}-T8500"][0] == 0.01  # 6% away in a few days: floor
