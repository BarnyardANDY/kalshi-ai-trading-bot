"""Exits must never send real orders in paper mode."""
import asyncio
from datetime import datetime

from src.config.settings import settings
from src.jobs.execute import place_sell_limit_order
from src.utils.database import Position


class Client:
    def __init__(self): self.orders = []
    async def place_order(self, **k):
        self.orders.append(k); return {"order": {"order_id": "x"}}


def _pos():
    return Position(market_id="KXRT-DIG-70", side="YES", entry_price=0.06, quantity=17, timestamp=datetime.now())


def test_paper_mode_sell_is_simulated(monkeypatch):
    monkeypatch.setattr(settings.trading, "live_trading_enabled", False)
    c = Client()
    assert asyncio.run(place_sell_limit_order(_pos(), 0.5, None, c)) is True
    assert c.orders == []


def test_live_mode_sell_sends_order(monkeypatch):
    monkeypatch.setattr(settings.trading, "live_trading_enabled", True)
    c = Client()
    asyncio.run(place_sell_limit_order(_pos(), 0.5, None, c))
    assert len(c.orders) == 1 and c.orders[0]["action"] == "sell"
