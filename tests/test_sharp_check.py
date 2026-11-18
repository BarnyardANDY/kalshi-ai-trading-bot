import asyncio
import time

import pytest

from src import sharp_check as S

NOW = time.time()
START = NOW + 6 * 3600


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def test_devig_two_way_removes_the_margin_and_rejects_odd_inputs():
    h, a = S.devig_two_way(1.909, 1.909)
    assert h == pytest.approx(0.5) and a == pytest.approx(0.5)
    h, a = S.devig_two_way(1.50, 2.70)
    assert h + a == pytest.approx(1.0) and h > 0.6
    assert S.devig_two_way(0.9, 2.0) is None
    assert S.devig_two_way(1.2, 1.2) is None            # sums to 1.67: not a real two-way line
    assert S.devig_two_way(1.9, 2.2) is None            # sums to 0.98: arbitrage glitch, ignore


def test_parse_odds_api_takes_only_pinnacle_h2h():
    resp = [{"home_team": "Boston Celtics", "away_team": "New York Knicks", "commence_time": iso(START),
             "bookmakers": [
                 {"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [
                     {"name": "Boston Celtics", "price": 1.1}, {"name": "New York Knicks", "price": 9.0}]}]},
                 {"key": "pinnacle", "last_update": iso(NOW - 60), "markets": [{"key": "h2h", "outcomes": [
                     {"name": "Boston Celtics", "price": 1.62}, {"name": "New York Knicks", "price": 2.35}]}]}]},
            {"home_team": "A", "away_team": "B", "commence_time": iso(START), "bookmakers": []}]
    evs = S.parse_odds_api(resp)
    assert len(evs) == 1 and evs[0]["o_home"] == 1.62 and evs[0]["o_away"] == 2.35
    assert evs[0]["start_ts"] == pytest.approx(START, abs=1)


def test_parse_pinnapi_loose_shape_and_draw_markets_skipped():
    resp = {"events": [
        {"home": "Boston Celtics", "away": "New York Knicks", "starts": iso(START),
         "periods": {"num_0": {"money_line": {"home": 1.62, "away": 2.35}}}},
        {"home": "X", "away": "Y", "starts": iso(START),
         "periods": {"num_0": {"money_line": {"home": 2.0, "away": 3.0, "draw": 3.3}}}}]}
    evs = S.parse_pinnapi(resp)
    assert [e["home"] for e in evs] == ["Boston Celtics"]


def kev(title, a, b, close=START):
    return {"event_ticker": "KXNBAGAME-26NOV20BOSNYK", "title": title, "markets": [
        {"ticker": "KXNBAGAME-26NOV20BOSNYK-BOS", "yes_sub_title": a, "close_time": iso(close + 3 * 3600)},
        {"ticker": "KXNBAGAME-26NOV20BOSNYK-NYK", "yes_sub_title": b, "close_time": iso(close + 3 * 3600)}]}


def test_match_kalshi_assigns_each_team_to_its_own_market():
    g = {"home": "Boston Celtics", "away": "New York Knicks", "start_ts": START}
    m = S.match_kalshi(g, [kev("New York at Boston Winner?", "Boston", "New York")])
    assert m["home"].endswith("-BOS") and m["away"].endswith("-NYK")
    assert S.match_kalshi(g, [kev("Los Angeles at Denver", "Denver", "Los Angeles")]) is None
    swapped = S.match_kalshi(g, [kev("x", "New York", "Boston")])             # names, not ticker order, decide
    assert swapped["home"].endswith("-NYK") and swapped["away"].endswith("-BOS")


def test_view_edge_includes_the_fee():
    e = S.view_edge(0.60, 0.55)
    assert e == pytest.approx(0.05 - S.taker_fee(0.55))
    assert S.view_edge(0.60, None) is None


class Client:
    def __init__(self, ask_home=0.55, ask_away=0.45, size=10):
        self.book = {"KXNBAGAME-26NOV20BOSNYK-BOS": ask_home, "KXNBAGAME-26NOV20BOSNYK-NYK": ask_away}
        self.size = size
        self.markets = {}

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
        return {"events": [kev("New York at Boston Winner?", "Boston", "New York")]}

    async def get_orderbook(self, t):
        ask = self.book[t]
        return {"orderbook": {"no_dollars": [[f"{1 - ask:.2f}", self.size]], "yes_dollars": [[f"{ask - 0.02:.2f}", 5]]}}

    async def get_markets(self, limit=100, tickers=None):
        return {"markets": [self.markets[t] for t in tickers or [] if t in self.markets]}


class Http:
    def __init__(self, resp):
        self.resp = resp

    async def get(self, url, params=None, headers=None):
        outer = self

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return outer.resp
        return R()


ODDS = [{"home_team": "Boston Celtics", "away_team": "New York Knicks", "commence_time": iso(START),
         "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
             {"name": "Boston Celtics", "price": 1.62}, {"name": "New York Knicks", "price": 2.35}]}]}]}]


def test_snapshot_records_views_from_the_real_book_and_counts_the_request(tmp_path):
    db = str(tmp_path / "s.db")
    c = Client(ask_home=0.52, ask_away=0.45)        # Pinnacle fair ~ Boston 0.59 / New York 0.41
    st = asyncio.run(S.snapshot(c, Http(ODDS), "odds_api", "k", "nba", NOW, db))
    assert st["matched"] == 1 and st["views"] == 2 and st["requested"] == 1
    assert S.used_today(db, "odds_api", NOW) == 1
    df = S.views_df(db).set_index("team")
    assert df.loc["Boston Celtics", "fair"] == pytest.approx(0.5937, abs=0.01)
    assert df.loc["Boston Celtics", "ask"] == pytest.approx(0.52) and df.loc["Boston Celtics", "depth"] == 10
    assert df.loc["Boston Celtics", "edge"] > 0.03 and st["gaps"] == 1
    assert df.loc["New York Knicks", "edge"] < 0


def test_games_too_close_to_the_start_or_leagues_with_no_kalshi_games_cost_no_request(tmp_path):
    db = str(tmp_path / "s.db")
    late = [dict(ODDS[0], commence_time=iso(NOW + 60))]
    st = asyncio.run(S.snapshot(Client(), Http(late), "odds_api", "k", "nba", NOW, db))
    assert st["games"] == 0 and st["views"] == 0

    class Empty(Client):
        async def _make_authenticated_request(self, *a, **k):
            return {"events": []}
    st = asyncio.run(S.snapshot(Empty(), Http(ODDS), "odds_api", "k", "nba", NOW, db))
    assert st["requested"] == 0 and S.used_today(db, "odds_api", NOW) == 1      # only the first call counted


def test_report_bets_only_with_real_depth_and_judges_by_results(tmp_path):
    db = str(tmp_path / "s.db")
    rows = [  # ticker, team, fair, ask, depth, result
        ("T1", "A", 0.62, 0.55, 10, "yes"), ("T2", "B", 0.62, 0.55, 1, "no"),      # T2: gap but only 1 contract
        ("T3", "C", 0.62, 0.55, 10, "no"), ("T4", "D", 0.50, 0.60, 10, "yes")]    # T4: Kalshi dearer: no bet
    with S._conn(db) as c:
        for i, (t, team, fair, ask, depth, res) in enumerate(rows):
            fee = S.taker_fee(ask)
            c.execute("INSERT INTO views(ts,league,event,ticker,team,start_ts,fair,ask,depth,fee,edge,provider,result,settled_ts)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (NOW + i, "nba", "E" + t, t, team, START, fair, ask, depth, fee, S.view_edge(fair, ask), "odds_api",
                       res, NOW))
    rep = S.report(db)
    t3 = next(t for t in rep["thresholds"] if t["threshold"] == 0.03)
    assert t3["bets"] == 2 and t3["settled"] == 2                                  # T1 and T3 only
    assert t3["win_rate"] == pytest.approx(0.5)
    assert rep["gap_sizes"]["views with a gap but under 3 contracts for sale"] == 1
    assert rep["calibration"]["Pinnacle Brier"] > 0
    assert "too early" in t3["verdict"] or "losing" in t3["verdict"]


def test_settle_sets_results_for_finished_games(tmp_path):
    db = str(tmp_path / "s.db")
    c = Client()
    asyncio.run(S.snapshot(c, Http(ODDS), "odds_api", "k", "nba", NOW, db))
    c.markets = {"KXNBAGAME-26NOV20BOSNYK-BOS": {"ticker": "KXNBAGAME-26NOV20BOSNYK-BOS", "result": "yes"},
                 "KXNBAGAME-26NOV20BOSNYK-NYK": {"ticker": "KXNBAGAME-26NOV20BOSNYK-NYK", "result": "no"}}
    assert asyncio.run(S.settle(c, START + 5 * 3600, db)) == 2
    df = S.views_df(db).set_index("team")
    assert df.loc["Boston Celtics", "result"] == "yes" and df.loc["New York Knicks", "result"] == "no"


def test_key_file_is_private_and_rejects_unknown_providers(tmp_path):
    p = str(tmp_path / "k.json")
    S.save_key("odds_api", " abc123 ", p)
    assert S.load_key(p) == {"provider": "odds_api", "key": "abc123"}
    import os
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError):
        S.save_key("bogus", "x", p)


def test_never_places_orders():
    src = open(S.__file__).read()
    assert "place_order" not in src and "create_order" not in src


def test_errors_never_show_the_api_key():
    key = "abc123SECRETkey"
    e = RuntimeError(f"Client error '401' for url 'https://api.x.com/v4/odds?regions=eu&apiKey={key}&b=1'")
    msg = S.safe_err(e, key)
    assert key not in msg and "apiKey=***" in msg
    assert key not in S.safe_err(RuntimeError(f"boom {key}"), key)
