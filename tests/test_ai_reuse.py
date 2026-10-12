"""Trump-style AI markets reuse a recent answer instead of re-asking every cycle."""
import asyncio
import time
from datetime import datetime

from src import learning, runtime_config
from src.strategies.portfolio import immediate
from src.utils.database import Market


class FakeKalshi:
    def __init__(self):
        self.yes_bid, self.yes_ask = 0.39, 0.41

    async def get_market(self, ticker):
        return {"market": {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "status": "active",
                           "yes_bid_dollars": str(self.yes_bid), "yes_ask_dollars": str(self.yes_ask),
                           "no_bid_dollars": str(round(1 - self.yes_ask, 2)),
                           "no_ask_dollars": str(round(1 - self.yes_bid, 2)),
                           "close_time": "2026-10-20T00:00:00Z"}}


class FakeAI:
    def __init__(self):
        self.calls = 0

    async def get_completion(self, prompt, **kw):
        self.calls += 1
        return '{"probability": 0.42, "confidence": 0.7, "reasoning": "x"}'


def test_ai_answer_reused_until_price_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "DB_PATH", str(tmp_path / "l.db"))

    async def _noop(*a, **k):
        return 0

    async def _research(*a, **k):
        return "news"
    monkeypatch.setattr(learning, "settle_predictions", _noop)
    monkeypatch.setattr("src.niche_research.build_research_context", _research)
    runtime_config.save({"NICHES": "trump_mentions", "SHADOW_NICHES": ""})
    immediate._AI_CACHE.clear()

    m = Market("KXTRUMPMENTION-26OCT20-TARIFF", "Will Trump say tariff?", 0.4, 0.6, 500,
               int(time.time()) + 86400 * 5, "trump_mentions", "active", datetime.now())
    k, ai = FakeKalshi(), FakeAI()
    run = lambda: asyncio.run(immediate.create_market_opportunities_from_markets([m], ai, k, None))
    run()
    run()
    assert ai.calls == 1                      # second cycle reused the answer
    k.yes_bid, k.yes_ask = 0.54, 0.56         # price moved 15 pts -> ask again
    run()
    assert ai.calls == 2
    runtime_config.save({"AI_REPRICE_MINUTES": 0})
    run()
    assert ai.calls == 3                      # 0 = ask every cycle
