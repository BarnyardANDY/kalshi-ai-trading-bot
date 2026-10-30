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
    assert (r["markets"], r["qualifying"], r["new"]) == (4, 2, 2)
    # E1-A (95c, 12h) also fits the 93-95c arm; E4-D (95c, 2h) fits only the 1-6h arm
    assert r["new_by_arm"] == {"favorites": 2, "fav_93_95": 1, "fav_1_6h": 1, "fav_econ": 0}   # Fake says "Sports"
    assert asyncio.run(F.scan_favorites(c, NOW + 600, db))["new"] == 0          # one bet per market
    df = F.rows(db)
    df = df[df["arm"] == "favorites"]
    assert set(df["side"]) == {"yes", "no"} and df.loc[df["side"] == "yes", "depth"].iloc[0] == 120
    c.markets["E1-A"]["result"] = "yes"                                          # favorite wins
    c.markets["E2-B"]["result"] = "yes"                                          # favorite (NO) loses
    assert asyncio.run(F.settle(c, NOW + 90000, db)) == 3                        # E1-A in two arms + E2-B
    df = F.rows(db)
    df = df[df["arm"] == "favorites"].set_index("ticker")
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


def _bets(rows):
    import pandas as pd
    cols = ["arm", "ticker", "side", "price", "depth", "entered_ts", "settled_ts", "result"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_bankroll_replay_sizes_bets_and_returns_cash():
    # $100 at 5% = $5 a bet -> 5 contracts at 95c; fee for 5 at .95 = ceil(1.66c) = 2c total
    g = _bets([("favorites", "A", "yes", 0.95, None, 0, 100, "yes"),
               ("favorites", "B", "yes", 0.95, None, 10, 200, "no")])
    r = F.simulate_bankroll(g, 100)
    cost = 5 * 0.95 + 0.02
    assert r["taken"] == 2 and r["contracts_per_bet"] == 5
    assert r["value"] == pytest.approx(100 - cost + 5 - cost)
    assert r["max_drop"] > 0 and r["open"] == 0


def test_bankroll_replay_caps_at_available_size_and_skips_when_cash_is_tied_up():
    g = _bets([("favorites", "A", "yes", 0.95, 2, 0, 50, "yes")])
    r = F.simulate_bankroll(g, 100)
    assert r["contracts_per_bet"] == 2 and r["capped"] == 1
    many = _bets([("favorites", f"M{i}", "yes", 0.95, None, i, None, None) for i in range(30)])
    r = F.simulate_bankroll(many, 100)                         # 5% each: cash runs out after ~20
    assert r["taken"] < 30 and r["skipped"] == 30 - r["taken"] and r["open"] == r["taken"]
    assert r["value"] == pytest.approx(100)                   # open bets count at cost


def test_report_includes_bankrolls(tmp_path):
    db = str(tmp_path / "f.db")
    c = Fake([mk("E1-A", .94, .95)])
    asyncio.run(F.scan_favorites(c, NOW, db))
    rep = F.report(db)
    assert set(rep["bankroll"]["favorites"]) == set(F.BANKROLLS)


def test_new_arms_have_their_own_rules():
    m93 = mk("A-1", 0.93, 0.94)
    assert F.favorite(m93, F.ARM_RULES["fav_93_95"]) is not None
    assert F.favorite(mk("A-2", 0.96, 0.97), F.ARM_RULES["fav_93_95"]) is None         # 97c: outside 93-95c
    assert F.favorite(mk("A-3", 0.90, 0.91), F.ARM_RULES["fav_93_95"]) is None
    assert F.in_window(mk("A-4", .94, .95, hours=3), NOW, F.ARM_RULES["fav_1_6h"])
    assert not F.in_window(mk("A-4", .94, .95, hours=3), NOW, F.ARM_RULES["favorites"])
    assert not F.in_window(mk("A-5", .94, .95, hours=8), NOW, F.ARM_RULES["fav_1_6h"])


def test_new_arms_start_their_own_clock(tmp_path):
    db = str(tmp_path / "f.db")
    with F.closing(F._conn(db)) as con:                        # the original test began 8 days earlier
        con.execute("INSERT INTO meta VALUES ('started', ?)", (str(NOW - 8 * 86400),))
        con.commit()
    F._mark_started(db, NOW)                                    # new code starts: originals keep their date
    assert F.started_ts(db) == pytest.approx(NOW - 8 * 86400)
    assert F.started_ts(db, "fav_93_95") == pytest.approx(NOW)
    c = Fake([mk("E1-X", .94, .95)])
    asyncio.run(F.scan_favorites(c, NOW, db))
    rep = F.report(db)
    assert rep["arm_days"]["favorites"] == pytest.approx(8, abs=0.1)
    assert rep["arm_days"]["fav_93_95"] < 0.1
    assert "day 0 of 14" in rep["arms"]["fav_93_95"]["verdict"] or rep["arms"]["fav_93_95"]["verdict"] == "waiting for results"


def _stress_rows():
    """10 days. Each day: 8 Nasdaq bets on 8 different events (they move together) + 4 weather bets.
    Day 3 is a crash: every Nasdaq bet loses. Otherwise everything wins."""
    import pandas as pd
    from datetime import datetime
    rows, i = [], 0
    for d in range(10):
        close = datetime(2026, 10, 20 + d, 16).timestamp()
        for k in range(12):
            nas = k < 8
            series = ["KXNASDAQ100U", "KXNASDAQ100D", "KXNASDAQ100"][k % 3] if nas else "KXHIGHMIA"
            won = not (nas and d == 3)
            rows.append({"id": i, "arm": "favorites", "ticker": f"T{i}", "event": f"EV{d}-{k}", "series": series,
                         "category": "Financials" if nas else "Climate and Weather", "side": "yes", "price": 0.95,
                         "fee": F.fee_per_contract(0.95), "depth": None if k % 2 else 50, "hours_left": 12,
                         "entered_ts": close - 12 * 3600, "close_ts": close, "settled_ts": close + 600,
                         "result": "yes" if won else "no", "pnl": F.pnl_per_contract(0.95, won)})
            i += 1
    return F.features(pd.DataFrame(rows))


def test_series_family_merges_versions_of_one_underlying():
    assert F.series_family("KXNASDAQ100U") == F.series_family("KXNASDAQ100D") == F.series_family("KXNASDAQ100")
    assert F.series_family("KXBTCD") == F.series_family("KXBTC")
    assert F.series_family("KXHIGHMIA") != F.series_family("KXHIGHNY")


def test_stress_counts_bets_that_move_together_as_one():
    st = F.stress(_stress_rows())
    by_event, by_day, by_cat = st["margins"]
    assert by_event["groups"] == 120 and by_day["groups"] == 20 and by_cat["groups"] == 20
    assert by_day["margin"] > by_event["margin"]           # the crash day counts once, not 8 times
    assert st["worst_day"]["day"] == "2026-10-23" and st["worst_day"]["losses"] == 8
    assert st["days"] == 10 and st["days_positive"] == 9
    assert [w["days removed"] for w in st["without_best"]] == [1, 2]        # 5 only once there are 25+ days


def test_stress_categories_and_stingy_replay():
    st = F.stress(_stress_rows())
    cats = {c["category"]: c for c in st["categories"]}
    assert cats["Financials"]["win rate"] == pytest.approx(72 / 80)
    assert cats["Climate and Weather"]["win rate"] == 1.0
    assert sum(c["share of profit"] for c in cats.values()) == pytest.approx(1.0)
    # $1,000 at 5% = ~52 contracts a bet; half the bets have unknown size -> capped at 10 in the stingy replay
    normal = F.simulate_bankroll(_stress_rows(), 1000)
    stingy = st["bankroll_stingy"][1000]
    assert stingy["contracts_per_bet"] < normal["contracts_per_bet"]


def test_economics_arm_only_takes_economics_and_looks_categories_up_first(tmp_path):
    db = str(tmp_path / "f.db")

    class EconFake(Fake):
        def __init__(self, markets):
            super().__init__(markets)
            self.series_calls = []

        async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
            if endpoint.endswith("/markets"):
                return {"markets": list(self.markets.values())}
            sr = endpoint.rsplit("/", 1)[-1]
            self.series_calls.append(sr)
            return {"series": {"category": "Economics" if sr == "KXAAAGASD" else "Sports"}}

    c = EconFake([mk("KXAAAGASD-26OCT30-4.30", .94, .95), mk("KXNBAGAME-26OCT30BOSNYK-BOS", .94, .95)])
    r = asyncio.run(F.scan_favorites(c, NOW, db))
    assert r["new_by_arm"]["fav_econ"] == 1 and r["new_by_arm"]["favorites"] == 2
    df = F.rows(db)
    assert list(df[df["arm"] == "fav_econ"]["ticker"]) == ["KXAAAGASD-26OCT30-4.30"]
    assert set(c.series_calls) == {"KXAAAGASD", "KXNBAGAME"}
    asyncio.run(F.scan_favorites(c, NOW + 600, db))
    assert len(c.series_calls) == 2                         # known categories aren't asked again
    assert "fav_econ" in F.NEW_ARMS and F.ARM_ADDED_BECAUSE["fav_econ"] == "the stress test"


def test_best_ask_reads_the_other_sides_bids():
    book = {"orderbook": {"yes_dollars": [["0.03", 10], ["0.04", 300]], "no_dollars": [["0.05", 120], ["0.04", 9]]}}
    assert F.best_ask(book, "yes") == pytest.approx(0.95)       # best NO bid 0.05 -> YES offered at 0.95
    assert F.best_ask(book, "no") == pytest.approx(0.96)        # best YES bid 0.04 -> NO offered at 0.96
    assert F.best_ask({"orderbook": {"yes_dollars": [], "no_dollars": []}}, "yes") is None


def test_scan_records_the_real_best_price(tmp_path):
    db = str(tmp_path / "f.db")
    asyncio.run(F.scan_favorites(Fake([mk("E1-A", .94, .95)]), NOW, db))
    r = F.rows(db)
    r = r[r["arm"] == "favorites"].iloc[0]
    assert r["depth"] == 120 and r["book_ask"] == pytest.approx(0.95)


def test_real_price_check_separates_bets_that_were_really_buyable():
    df = _stress_rows()
    df["book_ask"] = None
    df.loc[df.index[:20], "book_ask"] = 0.97                    # these 20 really cost 2c more than we recorded
    df.loc[df["depth"].notna() & (df.index % 4 == 0), "depth"] = 0     # some recorded prices weren't for sale
    rp = F.real_price_check(df[df["result"].isin(["yes", "no"])])
    checked = df["depth"].notna().sum()
    there = (df["depth"].fillna(-1) > 0).sum()
    assert rp["checked"] == checked and rp["really_there"] == pytest.approx(there / checked)
    assert rp["only_really_there"]["bets"] == there
    assert rp["with_real_price"] == 20 and rp["median_real_minus_ours_c"] == pytest.approx(2.0)
    assert F.stress(df)["real_price"]["checked"] == checked
