"""Stocks can be paper-tested on the bot's own view despite the auto-pause, skipping long shots."""
import asyncio
import time
from datetime import datetime

from src import learning, runtime_config
from src.strategies.portfolio import immediate
from src.utils.database import Market

TICKER = "KXINXU-26OCT14H1600-T6800"


class FakeKalshi:
    def __init__(self, bid, ask):
        self.bid, self.ask = bid, ask

    async def get_market(self, ticker):
        return {"market": {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "status": "active",
                           "yes_bid_dollars": f"{self.bid:.2f}", "yes_ask_dollars": f"{self.ask:.2f}",
                           "no_bid_dollars": f"{1 - self.ask:.2f}", "no_ask_dollars": f"{1 - self.bid:.2f}",
                           "close_time": "2026-10-14T20:00:00Z"}}


def _run(monkeypatch, tmp_path, bot_prob, bid, ask, override):
    monkeypatch.setattr(learning, "DB_PATH", str(tmp_path / "l.db"))

    async def _noop(*a, **k):
        return 0

    async def _pred(markets, *a, **k):
        return {m.market_id: (bot_prob, 0.8) for m in markets}
    monkeypatch.setattr(learning, "settle_predictions", _noop)
    monkeypatch.setattr(learning, "niche_params", lambda n: {"trust": 0.0, "paused": True, "events": 120})
    monkeypatch.setattr("src.stocks_model.predict_stocks", _pred)
    runtime_config.save({"NICHES": "stocks", "SHADOW_NICHES": "", "TRADE_DESPITE_PAUSE": override})
    m = Market(TICKER, "S&P 500 above 6800?", (bid + ask) / 2, 1 - (bid + ask) / 2, 500,
               int(time.time()) + 86400, "stocks", "active", datetime.now())
    return asyncio.run(immediate.create_market_opportunities_from_markets([m], None, FakeKalshi(bid, ask), None))


def test_paused_stocks_stay_off_without_override(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, 0.60, 0.39, 0.41, "") == []


def test_override_trades_bot_view(monkeypatch, tmp_path):
    ops = _run(monkeypatch, tmp_path, 0.60, 0.39, 0.41, "stocks")
    assert len(ops) == 1 and ops[0].recommended_side == "YES"


def test_override_still_skips_long_shots(monkeypatch, tmp_path):
    # bot likes YES at a 14c ask with enough edge to pass every other gate
    assert _run(monkeypatch, tmp_path, 0.38, 0.12, 0.14, "stocks") == []
    runtime_config.save({"STOCKS_MIN_SIDE_PRICE": 0})
    assert len(_run(monkeypatch, tmp_path, 0.38, 0.12, 0.14, "stocks")) == 1   # only the long-shot rule blocked it


def test_apply_uses_raw_view_when_overridden(monkeypatch):
    monkeypatch.setattr(learning, "niche_params", lambda n: {"trust": 0.0})
    runtime_config.save({"TRADE_DESPITE_PAUSE": "stocks"})
    assert learning.apply("stocks", 0.6, 0.4) == 0.6
    assert learning.apply("weather", 0.6, 0.4) == 0.4
