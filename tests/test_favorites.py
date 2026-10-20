import asyncio
import time
from datetime import datetime, timezone

import pytest

from src import favorites as F

NOW = time.time()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def mk(t, yes_bid, yes_ask, hours=12.0, **kw):
    d = {"ticker": t, "event_ticker": t.rsplit("-", 1)[0], "title": "Will it", "close_time": iso(NOW + hours * 3600),
         "yes_bid_dollars": f"{yes_bid:.2f}", "yes_ask_dollars": f"{yes_ask:.2f}",
         "no_bid_dollars": f"{1 - yes_ask:.2f}", "no_ask_dollars": f"{1 - yes_bid:.2f}", "volume_24h_fp": "500"}
    d.update(kw)
    return d


class Fake:
    def __init__(self, markets):
        self.markets = {m["ticker"]: m for m in markets}

    async def get_orderbook(self, t):
        return {"orderbook": {"yes_dollars": [["0.04", 300]], "no_dollars": [["0.05", 120]]}}   # YES ask .95 = NO bid .05

    async def get_markets(self, limit=100, tickers=None):
        return {"markets": [self.markets[t] for t in tickers or [] if t in self.markets]}

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
        if endpoint.endswith("/markets"):
            return {"markets": list(self.markets.values())}
        return {"series": {"category": "Sports"}}

    async def close(self):
        pass


def test_fee_rounds_up_per_order():
    assert F.fee_per_contract(0.95) == pytest.approx(0.004)        # 0.07*10*.95*.05 = $0.033 -> $0.04 / 10
    assert F.fee_per_contract(0.98) == pytest.approx(0.002)
    assert F.pnl_per_contract(0.95, True) == pytest.approx(0.046)
    assert F.pnl_per_contract(0.95, False) == pytest.approx(-0.954)


def test_favorite_picks_buyable_side_within_rules():
    assert F.favorite(mk("A-1", 0.94, 0.95))["side"] == "yes"
    assert F.favorite(mk("A-2", 0.04, 0.06))["side"] == "no"        # NO ask 0.96, NO bid 0.94
    assert F.favorite(mk("A-3", 0.97, 0.99)) is None                 # 99c: above the 98c cap
    assert F.favorite(mk("A-4", 0.80, 0.92)) is None                 # 12c gap: no real price
    assert F.favorite(mk("A-5", 0.60, 0.62)) is None                 # not a 90c+ favorite


def test_window():
    assert F.in_window(mk("A-1", .94, .95, hours=12), NOW)
    assert not F.in_window(mk("A-1", .94, .95, hours=3), NOW)
    assert not F.in_window(mk("A-1", .94, .95, hours=30), NOW)


def test_scan_records_once_then_settles(tmp_path):
    db = str(tmp_path / "f.db")
    ms = [mk("E1-A", .94, .95), mk("E2-B", .04, .06), mk("E3-C", .60, .62), mk("E4-D", .94, .95, hours=2)]
    c = Fake(ms)
    r = asyncio.run(F.scan_favorites(c, NOW, db))
    assert r == {"markets": 4, "qualifying": 2, "new": 2}
    assert asyncio.run(F.scan_favorites(c, NOW + 600, db))["new"] == 0          # one bet per market
    df = F.rows(db)
    assert set(df["side"]) == {"yes", "no"} and df.loc[df["side"] == "yes", "depth"].iloc[0] == 120
    c.markets["E1-A"]["result"] = "yes"                                          # favorite wins
    c.markets["E2-B"]["result"] = "yes"                                          # favorite (NO) loses
    assert asyncio.run(F.settle(c, NOW + 90000, db)) == 2
    df = F.rows(db).set_index("ticker")
    assert df.loc["E1-A", "pnl"] == pytest.approx(F.pnl_per_contract(0.95, True))
    assert df.loc["E2-B", "pnl"] == pytest.approx(F.pnl_per_contract(0.96, False))


def test_whale_arm_buys_a_minute_later_only_if_still_a_favorite(tmp_path):
    db = str(tmp_path / "f.db")
    c = Fake([mk("E1-A", .94, .95), mk("E2-B", .70, .72)])
    pend = [{"ticker": "E1-A", "side": "yes", "ts": NOW, "notional": 5000, "vwap": 0.95},
            {"ticker": "E2-B", "side": "yes", "ts": NOW, "notional": 5000, "vwap": 0.93},   # fell below 90c since
            {"ticker": "E1-A", "side": "yes", "ts": NOW + 50, "notional": 2000, "vwap": 0.95}]
    left = asyncio.run(F.whale_entries(c, pend, NOW + 61, db))
    assert len(left) == 1                                                        # the later one still waits
    df = F.rows(db)
    assert list(df["ticker"]) == ["E1-A"] and df["arm"].iloc[0] == "whale" and df["whale_notional"].iloc[0] == 5000
    asyncio.run(F.whale_entries(c, left, NOW + 120, db))
    assert len(F.rows(db)) == 1                                                  # one per market per arm


def test_report_and_verdict(tmp_path):
    db = str(tmp_path / "f.db")
    c = Fake([mk(f"E{i}-X", .94, .95) for i in range(40)])
    asyncio.run(F.scan_favorites(c, NOW, db))
    for i, m in enumerate(c.markets.values()):
        m["result"] = "yes" if i else "no"                                       # 39 wins, 1 loss
    asyncio.run(F.settle(c, NOW + 90000, db))
    asyncio.run(F.categories(c, db))
    with F.closing(F._conn(db)) as con:
        con.execute("INSERT OR IGNORE INTO meta VALUES ('started', ?)", (str(NOW),))
        con.commit()
    rep = F.report(db)
    st = rep["arms"]["favorites"]
    assert st["settled"] == 40 and st["events"] == 40 and st["losses"] == 1
    assert st["win_rate"] == pytest.approx(39 / 40)
    assert st["verdict"].startswith("too early")
    assert rep["arms"]["whale"]["verdict"] == "waiting for results"
    assert not rep["splits"]["favorites"]["Category"].empty
    assert F.verdict({"settled": 400, "events": 400, "c_per_contract": 1.0, "margin": 0.5}, 15) .startswith("PASSED")
    assert F.verdict({"settled": 400, "events": 400, "c_per_contract": 1.0, "margin": 1.5}, 15).startswith("no edge")
    assert F.verdict({"settled": 50, "events": 50, "c_per_contract": -9.0, "margin": 3.0}, 3) == "losing so far"


def test_never_trades():
    src = open(F.__file__).read()
    assert "place_order" not in src and "create_order" not in src


def test_time_of_day_split_uses_eastern_time_and_whale_buckets():
    import pandas as pd
    # 23:30 UTC on Oct 20 = 19:30 Eastern (EDT) -> evening; 14:00 UTC = 10:00 Eastern -> morning
    df = pd.DataFrame({"price": [0.95, 0.95], "hours_left": [12, 12], "depth": [50, 50], "pnl": [0.046, 0.046],
                       "event": ["E1", "E2"], "ticker": ["E1-A", "E2-A"], "settled_ts": [None, None],
                       "entered_ts": [pd.Timestamp("2026-10-20 23:30", tz="UTC").timestamp(),
                                      pd.Timestamp("2026-10-20 14:00", tz="UTC").timestamp()]})
    f = F.features(df)
    assert list(f["daypart"]) == ["evening (18-24)", "morning (6-12)"]
    assert F.SPLITS["Time of day bought (Eastern)"] == "daypart"
