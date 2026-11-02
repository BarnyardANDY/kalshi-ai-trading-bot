import asyncio
import json
import time
from datetime import datetime, timezone

import pytest

from src import favorites as F
from src import live_econ as L
from src.config.settings import settings

NOW = time.time()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def mk(t, yes_bid, yes_ask, hours=12.0):
    return {"ticker": t, "event_ticker": t.rsplit("-", 1)[0], "title": "Will it", "close_time": iso(NOW + hours * 3600),
            "yes_bid_dollars": f"{yes_bid:.2f}", "yes_ask_dollars": f"{yes_ask:.2f}",
            "no_bid_dollars": f"{1 - yes_ask:.2f}", "no_ask_dollars": f"{1 - yes_bid:.2f}"}


class Client:
    """Fake Kalshi: Economics = KXAAAGASD; YES offered at 0.95 (NO bid 0.05) with 120 contracts."""

    def __init__(self, markets, book_no_bid=0.05, book_size=120, order_resp=None, fail=False):
        self.markets = {m["ticker"]: m for m in markets}
        self.book_no_bid, self.book_size = book_no_bid, book_size
        self.order_resp, self.fail = order_resp, fail
        self.placed, self.flag_during = [], []

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
        if endpoint.endswith("/markets"):
            return {"markets": list(self.markets.values())}
        sr = endpoint.rsplit("/", 1)[-1]
        return {"series": {"category": "Economics" if sr == "KXAAAGASD" else "Sports"}}

    async def get_orderbook(self, t):
        return {"orderbook": {"no_dollars": [[f"{self.book_no_bid:.2f}", self.book_size]], "yes_dollars": [["0.03", 50]]}}

    async def place_order(self, **kw):
        self.placed.append(kw)
        self.flag_during.append(settings.trading.live_trading_enabled)
        if self.fail:
            raise RuntimeError("HTTP 500")
        return self.order_resp or {"order": {"order_id": f"oid{len(self.placed)}", "fill_count": kw["count"]}}

    async def get_fills(self, order_id=None, **kw):
        return {"fills": [{"count": 3, "yes_price_dollars": "0.94"}]}

    async def get_markets(self, limit=100, tickers=None):
        return {"markets": [self.markets[t] for t in tickers or [] if t in self.markets]}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "DB_PATH", str(tmp_path / "fav.db"))
    return str(tmp_path / "live.db"), str(tmp_path / "arm.json")


def econ():
    return mk("KXAAAGASD-26NOV03-4.30", .94, .95)


def test_not_armed_by_default_and_bad_arm_files_count_as_not_armed(tmp_path):
    p = str(tmp_path / "arm.json")
    assert L.read_arm(p)["armed"] is False
    for bad in ({"armed": True, "cap": 1000, "contracts": 3, "daily_loss_stop": 15},     # cap over the limit
                {"armed": True, "cap": 100, "contracts": 20, "daily_loss_stop": 15},      # too many contracts
                {"armed": "yes", "cap": 100, "contracts": 3, "daily_loss_stop": 15}):     # not literally true
        with open(p, "w") as f:
            json.dump(bad, f)
        assert L.read_arm(p)["armed"] is False
    with open(p, "w") as f:
        f.write("{garbage")
    assert L.read_arm(p)["armed"] is False
    with pytest.raises(ValueError):
        L.write_arm(True, cap=500, path=p)
    assert L.write_arm(True, cap=100, path=p)["armed"] and L.read_arm(p)["armed"]


def test_shadow_mode_never_sends_an_order(paths):
    db, arm = paths
    c = Client([econ(), mk("KXNBAGAME-26NOV03BOSNYK-BOS", .94, .95)])
    st = asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert st["mode"] == "shadow" and c.placed == []
    o = L.orders(db)
    assert len(o) == 1 and o[0]["ticker"].startswith("KXAAAGASD")                  # Economics only
    assert o[0]["mode"] == "shadow" and o[0]["count_req"] == 3 and o[0]["limit_price"] == pytest.approx(0.95)
    assert asyncio.run(L.scan(c, NOW + 600, db, arm_path=arm))["candidates"] == 0     # one bet per market


def test_armed_sends_one_immediate_or_cancel_buy_and_relocks(paths):
    db, arm = paths
    L.write_arm(True, cap=100, contracts=3, path=arm)
    settings.trading.live_trading_enabled = False
    c = Client([econ()])
    st = asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert st["mode"] == "live" and st["ordered"] == 1
    kw = c.placed[0]
    assert kw["side"] == "yes" and kw["action"] == "buy" and kw["count"] == 3 and kw["yes_price"] == 95
    assert kw["type_"] == "market"                          # sent as immediate-or-cancel at that price
    assert c.flag_during == [True] and settings.trading.live_trading_enabled is False
    o = L.orders(db)[0]
    assert o["mode"] == "live" and o["status"] == "sent" and o["fill_count"] == 3


def test_buys_only_at_the_price_seen_and_really_in_the_book(paths):
    db, arm = paths
    L.write_arm(True, path=arm)
    c = Client([econ()], book_no_bid=0.03)                 # real YES price 0.97, worse than the 0.95 we saw
    st = asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert st["skipped_book"] == 1 and c.placed == []


def test_cap_limits_money_at_risk(paths):
    db, arm = paths
    L.write_arm(True, cap=5, contracts=3, path=arm)       # one 3-contract bet at 95c costs ~$2.86
    c = Client([econ(), mk("KXAAAGASD-26NOV03-4.40", .94, .95), mk("KXAAAGASD-26NOV03-4.50", .94, .95)])
    st = asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert st["ordered"] == 1 and st["skipped_cap"] == 2
    assert L.money_at_risk(db) <= 5


def test_daily_loss_stop(paths):
    db, arm = paths
    L.write_arm(True, daily_loss_stop=15, path=arm)
    L._insert(db, {"ts": NOW, "mode": "live", "ticker": "OLD-1", "side": "yes", "fill_count": 10, "cost": 9.5,
                   "result": "no", "pnl": -16.0, "settled_ts": NOW, "status": "filled"})
    c = Client([econ()])
    st = asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert st["stopped"] and c.placed == []


def test_three_errors_in_a_row_disarm(paths):
    db, arm = paths
    L.write_arm(True, cap=100, path=arm)
    c = Client([mk(f"KXAAAGASD-26NOV03-4.{i}0", .94, .95) for i in range(5)], fail=True)
    asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert len(c.placed) == 3 and L.read_arm(arm)["armed"] is False
    assert "auto-disarmed" in L.read_arm(arm)["reason"]
    assert settings.trading.live_trading_enabled is False


def test_disarm_takes_effect_before_the_next_order(paths):
    db, arm = paths
    L.write_arm(True, path=arm)
    c = Client([mk(f"KXAAAGASD-26NOV03-4.{i}0", .94, .95) for i in range(3)])
    orig = c.place_order

    async def place_then_disarm(**kw):
        r = await orig(**kw)
        L.write_arm(False, reason="user", path=arm)
        return r
    c.place_order = place_then_disarm
    asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    assert len(c.placed) == 1                              # the arm file is re-read before every real order
    assert [o["mode"] for o in L.orders(db)].count("live") == 1


def test_fills_and_settlement(paths):
    db, arm = paths
    L.write_arm(True, path=arm)
    c = Client([econ()])
    asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    asyncio.run(L.reconcile_fills(c, db))
    o = L.orders(db)[0]
    assert o["status"] == "filled" and o["avg_price"] == pytest.approx(0.94)
    assert o["cost"] == pytest.approx(3 * 0.94 + L.fee_total(0.94, 3))
    c.markets[o["ticker"]]["result"] = "yes"
    assert asyncio.run(L.settle(c, NOW + 90000, db)) == 1
    assert L.orders(db)[0]["pnl"] == pytest.approx(3 - o["cost"])


def test_missing_fill_record_never_zeroes_a_reported_fill(paths):
    db, arm = paths
    L.write_arm(True, path=arm)
    c = Client([econ()])
    asyncio.run(L.scan(c, NOW, db, arm_path=arm))

    async def no_fills(order_id=None, **kw):
        return {"fills": []}
    c.get_fills = no_fills
    asyncio.run(L.reconcile_fills(c, db))
    o = L.orders(db)[0]
    assert o["fill_count"] == 3 and o["status"] == "sent"


def test_parse_fills_handles_no_side_and_yes_leg_price():
    assert L.parse_fills({"fills": [{"count": 2, "no_price": 96}]}, "no") == (2.0, pytest.approx(0.96), "fills")
    n, avg, src = L.parse_fills({"fills": [{"count_fp": "3.00", "price": "0.04"}]}, "no")
    assert n == 3 and avg == pytest.approx(0.96) and "YES-leg" in src


def test_report_and_mode(paths):
    db, arm = paths
    c = Client([econ()])
    asyncio.run(L.scan(c, NOW, db, arm_path=arm))
    rep = L.report(db, arm)
    assert rep["arm"]["armed"] is False and rep["shadow"]["orders"] == 1 and rep["live"]["orders"] == 0


def test_never_sells():
    src = open(L.__file__).read()
    assert 'action="sell"' not in src and "action='sell'" not in src


def test_every_scan_is_logged_for_the_dashboard(paths):
    db, arm = paths
    asyncio.run(L.scan(Client([econ()]), NOW, db, arm_path=arm))
    log = L.report(db, arm)["log"]
    assert any("scan (shadow): 1 new Economics favorites qualified" in m for m in log["msg"])


def test_only_one_live_process_can_run(tmp_path):
    lp = str(tmp_path / "live.lock")
    first = L.acquire_single_instance(lp)
    assert first is not None
    assert L.acquire_single_instance(lp) is None            # a second copy is refused
    first.close()
    again = L.acquire_single_instance(lp)                   # free once the first one stops
    assert again is not None
    again.close()


def test_second_copy_exits_without_trading(tmp_path, monkeypatch):
    lp, db = str(tmp_path / "live.lock"), str(tmp_path / "live.db")
    held = L.acquire_single_instance(lp)
    import src.clients.kalshi_client as kc
    monkeypatch.setattr(kc, "KalshiClient", lambda: (_ for _ in ()).throw(AssertionError("must not start")))
    asyncio.run(L.run_forever(None, db, str(tmp_path / "arm.json"), lock_path=lp))
    assert any("already running" in m for m in L.report(db, str(tmp_path / "arm.json"))["log"]["msg"])
    held.close()
