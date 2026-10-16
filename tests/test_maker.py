"""Maker test: patient favorite buyer (paper only)."""
import asyncio
from datetime import datetime, timezone

import pytest

from src import maker as MK
from src import runtime_config

NOW = 1_800_000_000.0


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def mkt(ticker="KXNFLGAME-26OCT18BUFMIA-BUF", yb=0.70, ya=0.72, hours=5, vol=500, **kw):
    m = {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "status": "active",
         "title": "Buffalo vs Miami", "yes_sub_title": "Buffalo",
         "yes_bid_dollars": f"{yb:.2f}", "yes_ask_dollars": f"{ya:.2f}",
         "no_bid_dollars": f"{1 - ya:.2f}", "no_ask_dollars": f"{1 - yb:.2f}",
         "close_time": iso(NOW + hours * 3600), "volume_24h_fp": str(vol)}
    m.update(kw)
    return m


CFG = {"enabled": True, "min_price": 0.55, "max_price": 0.95, "max_hours": 48, "min_minutes": 30,
       "contracts": 5, "max_open": 40, "order_minutes": 60, "max_per_series": 3, "min_volume_24h": 100}


def test_quote_joins_bid_on_favorite():
    q = MK.quote_for(mkt(), CFG, NOW)
    assert q["side"] == "yes" and q["price"] == 0.70 and not q["improved"]


def test_quote_improves_wide_spread_and_picks_no_favorite():
    q = MK.quote_for(mkt(yb=0.20, ya=0.25), CFG, NOW)       # NO is the favourite: bid .75 ask .80
    assert q["side"] == "no" and q["price"] == 0.76 and q["improved"]


@pytest.mark.parametrize("m", [
    mkt(yb=0.40, ya=0.45, ticker="KXA-1-X"),   # favourite NO at .55 bid ok... but check long-shot side never chosen
])
def test_never_buys_long_shot(m):
    q = MK.quote_for(m, CFG, NOW)
    assert q is None or q["price"] >= 0.55


def test_quote_filters():
    assert MK.quote_for(mkt(hours=100), CFG, NOW) is None             # too far out
    assert MK.quote_for(mkt(hours=0.2), CFG, NOW) is None             # about to close
    assert MK.quote_for(mkt(vol=10), CFG, NOW) is None                # no activity
    assert MK.quote_for(mkt(yb=0.97, ya=0.98), CFG, NOW) is None      # above max price
    assert MK.quote_for(mkt(ticker="KXMVESPORTS-1-A"), CFG, NOW) is None  # combo market
    assert MK.quote_for(mkt(yb=0.51, ya=0.53), CFG, NOW) is None      # coin flip: below min price


def test_queue_ahead_both_formats():
    assert MK.queue_ahead({"orderbook": {"yes_dollars": [["0.69", 50], ["0.70", 120]]}}, "yes", 0.70) == 120
    assert MK.queue_ahead({"orderbook": {"no": [[29, 7], [30, 40]]}}, "no", 0.30) == 40


def tr(ts, taker, yes_price, count):
    return {"created_time": iso(ts), "taker_side": taker, "yes_price_dollars": f"{yes_price:.2f}",
            "no_price_dollars": f"{1 - yes_price:.2f}", "count_fp": str(count)}


def test_fill_needs_queue_cleared_or_trade_through():
    posted = NOW
    # people BUYING yes don't fill our yes bid; selling at our price must clear 100 ahead + our 5
    trades = [tr(NOW + 10, "yes", 0.72, 50), tr(NOW + 20, "no", 0.70, 60), tr(NOW + 30, "no", 0.70, 40)]
    assert MK.check_fill(trades, "yes", 0.70, 5, 100, posted) is None
    trades.append(tr(NOW + 40, "no", 0.70, 5))
    assert MK.check_fill(trades, "yes", 0.70, 5, 100, posted) == pytest.approx(NOW + 40, abs=1)
    # a sale below our price means we were hit first
    assert MK.check_fill([tr(NOW + 5, "no", 0.69, 1)], "yes", 0.70, 5, 999, posted) == pytest.approx(NOW + 5, abs=1)
    # trades before we posted don't count
    assert MK.check_fill([tr(NOW - 5, "no", 0.60, 100)], "yes", 0.70, 5, 0, posted) is None


def test_fill_for_no_side():
    # our NO bid at .30 is hit by a taker buying YES at .70
    assert MK.check_fill([tr(NOW + 5, "yes", 0.71, 3)], "no", 0.30, 5, 0, NOW) == pytest.approx(NOW + 5, abs=1)
    assert MK.check_fill([tr(NOW + 5, "no", 0.71, 3)], "no", 0.30, 5, 0, NOW) is None


def test_settle_pnl_with_fee():
    fee = MK.maker_fee(0.70, 5)
    assert fee == 0.02
    assert MK.settle_pnl("yes", 0.70, 5, fee, "yes") == pytest.approx(5 - 3.5 - 0.02)
    assert MK.settle_pnl("yes", 0.70, 5, fee, "no") == pytest.approx(-3.5 - 0.02)
    assert MK.settle_pnl("no", 0.70, 5, fee, "") is None


def test_pick_spreads_risk():
    cands = [MK.quote_for(mkt(ticker=f"KXNFLGAME-26OCT18G{i}-A", vol=1000 - i), CFG, NOW) for i in range(6)]
    cands.append(MK.quote_for(mkt(ticker="KXNFLGAME-26OCT18G0-B"), CFG, NOW))   # same event as G0-A
    chosen = MK.pick(cands, [], CFG)
    assert len(chosen) == 3                                   # max 3 per series
    assert len({c["event"] for c in chosen}) == 3


class FakeClient:
    """Read-only fake. Has no place_order: the maker must never need one."""
    def __init__(self, markets, trades=None, result=""):
        self.markets = {m["ticker"]: m for m in markets}
        self.trades = trades or []
        self.result = result

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True, **kw):
        assert method == "GET"
        if endpoint.endswith("/markets/trades"):
            return {"trades": [t for t in self.trades if params["ticker"] == t.get("ticker", params["ticker"])]}
        return {"markets": list(self.markets.values())}

    async def get_markets(self, limit=100, tickers=None, **kw):
        return {"markets": [dict(self.markets[t], result=self.result) for t in (tickers or []) if t in self.markets]}

    async def get_market(self, ticker):
        m = dict(self.markets[ticker])
        m["result"] = self.result
        return {"market": m}

    async def get_orderbook(self, ticker):
        return {"orderbook": {"yes_dollars": [["0.70", 20]], "no_dollars": [["0.28", 10]]}}


def test_full_cycle_post_fill_settle(tmp_path):
    db = str(tmp_path / "m.db")
    runtime_config.save({"MAKER_MIN_VOLUME_24H": 100})
    c = FakeClient([mkt()])
    s = asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    assert s["posted"] == 1
    r = MK.rows("open", db)[0]
    assert r["side"] == "yes" and r["price"] == 0.70 and r["queue_ahead"] == 20
    # 25 sold at our price after we posted: clears 20 ahead + our 5
    c.trades = [tr(NOW + 60, "no", 0.70, 25)]
    asyncio.run(MK.run_cycle(c, now=NOW + 120, path=db))
    assert MK.rows("filled", db)
    # settles after close
    c.result = "yes"
    asyncio.run(MK.run_cycle(c, now=NOW + 6 * 3600, path=db, universe=[]))
    done = MK.rows("settled", db)[0]
    assert done["pnl"] == pytest.approx(5 - 3.5 - 0.02)
    rep = MK.report(MK.rows(path=db))
    assert rep["settled"] == 1 and rep["c_per_contract"] == pytest.approx(100 * 1.48 / 5)


def test_stale_order_cancelled_and_reposted(tmp_path):
    db = str(tmp_path / "m.db")
    c = FakeClient([mkt()])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    asyncio.run(MK.run_cycle(c, now=NOW + 61 * 60, path=db))
    st = [r["status"] for r in MK.rows(path=db)]
    assert st.count("cancelled") == 1 and st.count("open") == 1


def test_disabled_posts_nothing(tmp_path):
    runtime_config.save({"MAKER_ENABLED": False})
    s = asyncio.run(MK.run_cycle(FakeClient([mkt()]), now=NOW, path=str(tmp_path / "m.db")))
    assert s["posted"] == 0


def test_trades_from_other_markets_ignored():
    other = dict(tr(NOW + 5, "no", 0.10, 999), ticker="KXOTHER-1-A")
    assert MK.check_fill([other], "yes", 0.70, 5, 0, NOW, "KXNFLGAME-26OCT18BUFMIA-BUF") is None
    mine = dict(tr(NOW + 5, "no", 0.69, 1), ticker="KXNFLGAME-26OCT18BUFMIA-BUF")
    assert MK.check_fill([mine], "yes", 0.70, 5, 0, NOW, "KXNFLGAME-26OCT18BUFMIA-BUF") is not None


def test_settles_early_closing_market(tmp_path):
    db = str(tmp_path / "m.db")
    MK._last_full_settle = 0.0
    c = FakeClient([mkt(hours=30)])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    c.trades = [tr(NOW + 60, "no", 0.69, 1)]
    asyncio.run(MK.run_cycle(c, now=NOW + 120, path=db))
    c.result = "no"                                  # game over long before close_time
    MK._last_full_settle = 0.0
    asyncio.run(MK.run_cycle(c, now=NOW + 3 * 3600, path=db, universe=[]))
    r = MK.rows("settled", db)[0]
    assert r["pnl"] == pytest.approx(-3.5 - 0.02) and "sold through" in (r["note"] or "")
