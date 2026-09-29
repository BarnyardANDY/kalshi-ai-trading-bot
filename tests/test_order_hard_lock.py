"""KalshiClient.place_order must refuse unless live trading is on."""
import asyncio

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings


def _client(sent):
    c = KalshiClient.__new__(KalshiClient)  # skip key loading / network setup
    import structlog
    c.logger = structlog.get_logger("test")
    async def fake_request(method, path, **k):
        sent.append((method, path, k.get("json_data")))
        return {"order_id": "abc"}
    c._make_authenticated_request = fake_request
    return c


def test_blocked_in_paper_mode(monkeypatch):
    monkeypatch.setattr(settings.trading, "live_trading_enabled", False)
    sent = []
    r = asyncio.run(_client(sent).place_order("KXRT-DIG-45", "id1", "no", "sell", 4, "limit", no_price=24))
    assert r.get("blocked") and sent == []


def test_allowed_in_live_mode(monkeypatch):
    monkeypatch.setattr(settings.trading, "live_trading_enabled", True)
    sent = []
    asyncio.run(_client(sent).place_order("KXRT-DIG-45", "id1", "yes", "buy", 1, "limit", yes_price=30))
    assert len(sent) == 1 and sent[0][1].endswith("/portfolio/events/orders")


def test_paper_run_forces_lock_even_if_env_says_live(monkeypatch):
    import cli
    monkeypatch.setattr(settings.trading, "live_trading_enabled", True)
    class Args: live=False; paper=True; beast=False; disciplined=False; safe_compounder=True; loop=False; interval=300; log_level="INFO"
    monkeypatch.setattr(cli, "_run_safe_compounder", lambda **k: None)
    cli.cmd_run(Args())
    assert settings.trading.live_trading_enabled is False
