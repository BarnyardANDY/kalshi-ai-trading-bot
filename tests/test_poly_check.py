import asyncio
import json
import time

import pytest

from src import poly_check as P

NOW = time.time()


def kev(ticker, a, b, a_bid=0.60, a_ask=0.62, a_name=None, b_name=None):
    return {"event_ticker": ticker, "title": "Away at Home",
            "markets": [{"ticker": f"{ticker}-{a}", "yes_sub_title": a_name or a, "yes_bid_dollars": f"{a_bid:.2f}",
                         "yes_ask_dollars": f"{a_ask:.2f}"},
                        {"ticker": f"{ticker}-{b}", "yes_sub_title": b_name or b,
                         "yes_bid_dollars": f"{1 - a_ask:.2f}", "yes_ask_dollars": f"{1 - a_bid:.2f}"}]}


def pev(slug, outcomes, prices, tokens=("t1", "t2"), smt="moneyline", start="2026-11-01T18:00:00Z"):
    return {"slug": slug, "startTime": start,
            "markets": [{"sportsMarketType": "spreads", "outcomes": json.dumps(outcomes),
                         "clobTokenIds": json.dumps(["x1", "x2"]), "outcomePrices": json.dumps(["0.5", "0.5"])},
                        {"sportsMarketType": smt, "outcomes": json.dumps(outcomes), "clobTokenIds": json.dumps(list(tokens)),
                         "outcomePrices": json.dumps([str(p) for p in prices]), "gameStartTime": "2026-11-01 18:00:00+00"}]}


def test_parsers():
    assert P.parse_poly_slug("nfl-buf-mia-2026-11-01") == {"prefix": "nfl", "teams": {"buf", "mia"}, "date": "2026-11-01"}
    assert P.parse_poly_slug("nfl-buf-mia-2026-11-01-more-markets") is None
    k = P.parse_kalshi_event(kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA"))
    assert k["date"] == "2026-11-01" and set(k["teams"]) == {"buf", "mia"}
    assert P.parse_kalshi_event(kev("KXMLBGAME-26OCT271910LADTOR", "LAD", "TOR"))["date"] == "2026-10-27"
    assert P.norm("JAC") == "jax" and P.norm("WSH") == "was"
    assert P.dates_close("2026-11-01", "2026-11-02") and not P.dates_close("2026-11-01", "2026-11-03")


def test_moneyline_skips_spreads_and_yes_no():
    ml = P.moneyline(pev("nfl-buf-mia-2026-11-01", ["Bills", "Dolphins"], [0.6, 0.4]))
    assert ml["tokens"] == ["t1", "t2"] and ml["outcomes"] == ["Bills", "Dolphins"]
    assert ml["start"] == pytest.approx(P._iso_ts("2026-11-01T18:00:00Z"))
    assert P.moneyline(pev("x-a-b-2026-11-01", ["Yes", "No"], [0.5, 0.5])) is None


def test_match_by_team_names():
    k = [kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA")]
    p = [pev("nfl-buf-mia-2026-11-01", ["Dolphins", "Bills"], [0.4, 0.6])]
    found, st = P.match_games(p, k, "nfl", {"buf": "Buffalo Bills", "mia": "Miami Dolphins"})
    assert st["matched"] == 1 and found[0]["matched_how"] == "names"
    f = found[0]
    assert (f["team_a"], f["poly_token_a"], f["team_b"], f["poly_token_b"]) == ("buf", "t2", "mia", "t1")


def test_match_by_prices_when_names_unknown_and_skip_when_ambiguous():
    k = [kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA", a_bid=0.70, a_ask=0.72)]
    p = [pev("nfl-buf-mia-2026-11-01", ["Dolphins", "Bills"], [0.29, 0.71])]
    found, st = P.match_games(p, k, "nfl", {})
    assert found[0]["matched_how"] == "prices" and found[0]["poly_token_a"] == "t2"
    k2 = [kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA", a_bid=0.49, a_ask=0.51)]
    p2 = [pev("nfl-buf-mia-2026-11-01", ["Dolphins", "Bills"], [0.5, 0.5])]
    found, st = P.match_games(p2, k2, "nfl", {})
    assert not found and st["ambiguous_sides"] == 1


def test_match_prefers_same_date_for_back_to_back_games():
    k = [kev("KXMLBGAME-26OCT27LADTOR", "LAD", "TOR"), kev("KXMLBGAME-26OCT28LADTOR", "LAD", "TOR")]
    p = [pev("mlb-lad-tor-2026-10-28", ["Dodgers", "Blue Jays"], [0.6, 0.4])]
    found, _ = P.match_games(p, k, "mlb", {"lad": "Los Angeles Dodgers", "tor": "Toronto Blue Jays"})
    assert found[0]["kalshi_event"] == "KXMLBGAME-26OCT28LADTOR"


def test_unmatched_teams_are_counted():
    found, st = P.match_games([pev("nfl-buf-nyj-2026-11-01", ["Bills", "Jets"], [0.6, 0.4])],
                              [kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA")], "nfl", {})
    assert not found and st["no_kalshi_match"] == 1


def _s(ts, ka, pa, live=1):
    """sample: Kalshi team-a bid/ask and Polymarket team-a bid/ask; team b mirrors."""
    return {"ts": ts, "pair_id": "G", "live": live, "ka_bid": ka[0], "ka_ask": ka[1], "kb_bid": round(1 - ka[1], 2),
            "kb_ask": round(1 - ka[0], 2), "pa_bid": pa[0], "pa_ask": pa[1], "pb_bid": round(1 - pa[1], 2),
            "pb_ask": round(1 - pa[0], 2)}


def test_gap_episode_and_late_buy():
    ss = [_s(0, (0.49, 0.50), (0.49, 0.51)), _s(5, (0.49, 0.50), (0.59, 0.61)),     # Poly jumps to 0.60
          _s(10, (0.52, 0.53), (0.59, 0.61)), _s(15, (0.58, 0.59), (0.59, 0.61))]
    eps = P.gap_episodes(ss, 0.05)
    assert len(eps) == 1 and eps[0]["side"] == "a" and eps[0]["start"] == 5 and eps[0]["end"] == 10
    assert P.buy_late(eps[0], ss, True, 0) == pytest.approx(1 - 0.50 - P.taker_fee(0.50))
    assert P.buy_late(eps[0], ss, True, 5) == pytest.approx(1 - 0.53 - P.taker_fee(0.53))
    assert P.buy_late(eps[0], ss, False, 0) == pytest.approx(-0.50 - P.taker_fee(0.50))


def test_poly_jumps_measure_kalshi_follow():
    ss = [_s(0, (0.49, 0.51), (0.49, 0.51)), _s(5, (0.49, 0.51), (0.59, 0.61)),
          _s(10, (0.54, 0.56), (0.59, 0.61)), _s(20, (0.59, 0.61), (0.59, 0.61))]
    j = P.poly_jumps(ss, waits=(0, 5, 15))
    assert len(j) == 1 and j[0]["k0"] == pytest.approx(0) and j[0]["k5"] == pytest.approx(0.5)
    assert j[0]["k15"] == pytest.approx(1.0) and j[0]["p15"] == pytest.approx(1.0)


def test_report_end_to_end(tmp_path):
    db = str(tmp_path / "p.db")
    with P.closing(P._conn(db)) as c:
        c.execute("INSERT INTO pairs (pair_id, league, poly_slug, kalshi_event, team_a, team_b, k_ticker_a, k_ticker_b,"
                  " poly_token_a, poly_token_b, start_ts, matched_how, created_ts, result_a) VALUES "
                  "('G','nfl','s','e','buf','mia','ka','kb','pa','pb',0,'names',0,'yes')")
        rows = [_s(0, (0.49, 0.50), (0.49, 0.51)), _s(5, (0.49, 0.50), (0.59, 0.61)),
                _s(70, (0.58, 0.59), (0.59, 0.61)), _s(140, (0.70, 0.71), (0.70, 0.72))]
        c.executemany("INSERT INTO samples VALUES (:ts,:pair_id,:live,:ka_bid,:ka_ask,:kb_bid,:kb_ask,:pa_bid,:pa_ask,"
                      ":pb_bid,:pb_ask)", rows)
        c.commit()
    rep = P.report(db)
    assert rep["games_settled"] == 1 and rep["samples"] == 4
    g5 = [g for g in rep["gaps"] if g["gaps"] == "5c+ after fee, any time"][0]
    assert g5["count"] == 1 and g5["games"] == 1
    assert {r["when"] for r in rep["calibration"]} == {"during the game", "all"}
    out = str(tmp_path / "r.json")
    P.write_report(db, out)
    assert P.read_report(out)["games_settled"] == 1


class FakeKalshi:
    def __init__(self):
        self.markets = {}

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
        if params and params.get("series_ticker") == "KXNFLGAME":
            ev = kev("KXNFLGAME-26NOV01BUFMIA", "BUF", "MIA")
            for m in ev["markets"]:
                self.markets[m["ticker"]] = m
            return {"events": [ev]}
        return {"events": []}

    async def get_markets(self, limit=100, tickers=None):
        return {"markets": [self.markets[t] for t in tickers or [] if t in self.markets]}

    async def close(self):
        pass


class FakeResp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeHttp:
    def __init__(self, tag_events=True, slug_events=None):
        self.tag_events, self.slug_events, self.slug_calls = tag_events, slug_events or {}, []

    async def get(self, url, params=None):
        if isinstance(params, list):                         # look-up by game-page name
            asked = [v for k, v in params if k == "slug"]
            self.slug_calls.append(asked)
            return FakeResp([self.slug_events[x] for x in asked if x in self.slug_events])
        if url.endswith("/events") and params.get("tag_slug") == "nfl" and self.tag_events:
            return FakeResp([pev("nfl-buf-mia-2026-11-01", ["Bills", "Dolphins"], [0.61, 0.39])])
        if url.endswith("/teams") and params.get("league") == "nfl":
            return FakeResp([{"abbreviation": "BUF", "name": "Bills"}, {"abbreviation": "MIA", "name": "Dolphins"}])
        return FakeResp([])

    async def post(self, url, json=None):
        # labelled the way the live feed does it: BUY below SELL
        return FakeResp({"t1": {"BUY": 0.60, "SELL": 0.62}, "t2": {"BUY": 0.38, "SELL": 0.40}})


def test_discover_sample_settle(tmp_path):
    db = str(tmp_path / "p.db")
    k, h = FakeKalshi(), FakeHttp()
    info = asyncio.run(P.discover(k, h, db))
    assert info["nfl"]["matched"] == 1
    pl = P.pairs(db)
    assert len(pl) == 1 and pl[0]["poly_token_a"] == "t1"
    assert asyncio.run(P.sample(k, h, pl, NOW, db)) == 1
    k.markets[pl[0]["k_ticker_a"]]["result"] = "no"
    k.markets[pl[0]["k_ticker_b"]]["result"] = "yes"
    assert asyncio.run(P.settle(k, NOW + 9e6, db)) == 1
    assert P.pairs(db)[0]["result_a"] == "no"


def test_tie_is_recorded_as_tie(tmp_path):
    db = str(tmp_path / "p.db")
    k, h = FakeKalshi(), FakeHttp()
    asyncio.run(P.discover(k, h, db))
    pl = P.pairs(db)
    k.markets[pl[0]["k_ticker_a"]]["result"] = "no"
    k.markets[pl[0]["k_ticker_b"]]["result"] = "no"
    asyncio.run(P.settle(k, NOW + 9e6, db))
    assert P.pairs(db)[0]["result_a"] == "tie"


def test_never_trades():
    src = open(P.__file__).read()
    assert "place_order" not in src and "create_order" not in src


def test_one_kalshi_game_pairs_with_one_polymarket_game():
    # the live probe paired Kalshi's single World Series game with BOTH of Polymarket's (today + tomorrow)
    k = [kev("KXMLBGAME-26OCT272000MILTB", "MIL", "TB")]
    p = [pev("mlb-mil-tb-2026-10-28", ["Brewers", "Rays"], [0.55, 0.45], tokens=("x", "y")),
         pev("mlb-mil-tb-2026-10-27", ["Brewers", "Rays"], [0.55, 0.45])]
    found, st = P.match_games(p, k, "mlb", {"mil": "Milwaukee Brewers", "tb": "Tampa Bay Rays"})
    assert len(found) == 1 and found[0]["poly_slug"] == "mlb-mil-tb-2026-10-27"
    assert st["unmatched_poly"] == ["mlb-mil-tb-2026-10-28"]


def test_prices_take_lower_as_bid_whatever_the_labels():
    class H:
        async def post(self, url, json=None):
            return FakeResp({"t": {"BUY": 0.17, "SELL": 0.18}})
    assert asyncio.run(P.poly_prices(H(), ["t"])) == {"t": {"bid": 0.17, "ask": 0.18}}


def test_lookup_by_game_page_name_when_tag_search_finds_nothing(tmp_path):
    from datetime import datetime, timedelta
    d = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    kd = (datetime.now() + timedelta(days=1)).strftime("%y%b%d").upper()
    k = FakeKalshi()
    ev = kev(f"KXNFLGAME-{kd}ARIDAL", "ARI", "DAL")

    async def req(method, endpoint, params=None, require_auth=True):
        if params and params.get("series_ticker") == "KXNFLGAME":
            for m in ev["markets"]:
                k.markets[m["ticker"]] = m
            return {"events": [ev]}
        return {"events": []}
    k._make_authenticated_request = req
    target = pev(f"nfl-ari-dal-{d}", ["Cardinals", "Cowboys"], [0.4, 0.6])
    h = FakeHttp(tag_events=False, slug_events={f"nfl-ari-dal-{d}": target})
    P._SLUG_MISS.clear()
    found, st = asyncio.run(P.find_games(k, h, "nfl"))
    assert st["matched"] == 1 and st["by_name_lookup"] == 1 and st["tag_events"] == 0
    assert any(f"nfl-ari-dal-{d}" in c for c in h.slug_calls)
    # misses aren't asked again for two hours
    h2 = FakeHttp(tag_events=False)
    asyncio.run(P.find_games(k, h2, "nfl"))
    asyncio.run(P.find_games(k, h2, "nfl"))
    assert len([c for c in h2.slug_calls if c]) <= len([c for c in h.slug_calls if c])


def test_nhl_codes_that_differ_between_sites_still_match():
    # seen live: Polymarket nhl-mon-stl / nhl-lak-chi / nhl-min-utah / nhl-ott-las vs Kalshi MTL, LA, UTA, VGK
    k = [kev("KXNHLGAME-26OCT27MTLSTL", "MTL", "STL", a_name="Montreal"),
         kev("KXNHLGAME-26OCT27LACHI", "LA", "CHI", a_name="Los Angeles"),
         kev("KXNHLGAME-26OCT27MINUTA", "MIN", "UTA", a_name="Minnesota"),
         kev("KXNHLGAME-26OCT27OTTVGK", "OTT", "VGK", a_name="Ottawa")]
    p = [pev("nhl-mon-stl-2026-10-27", ["Canadiens", "Blues"], [0.6, 0.4], tokens=("m1", "m2")),
         pev("nhl-lak-chi-2026-10-27", ["Kings", "Blackhawks"], [0.6, 0.4], tokens=("l1", "l2")),
         pev("nhl-min-utah-2026-10-27", ["Wild", "Utah"], [0.6, 0.4], tokens=("u1", "u2")),
         pev("nhl-ott-las-2026-10-27", ["Senators", "Golden Knights"], [0.6, 0.4], tokens=("o1", "o2"))]
    names = {"mon": "Montreal Canadiens", "stl": "St. Louis Blues", "lak": "Los Angeles Kings",
             "chi": "Chicago Blackhawks", "min": "Minnesota Wild", "utah": "Utah Mammoth",
             "ott": "Ottawa Senators", "las": "Vegas Golden Knights"}
    found, st = P.match_games(p, k, "nhl", names)
    assert st["matched"] == 4 and st["no_kalshi_match"] == 0
    mtl = [f for f in found if f["kalshi_event"].endswith("MTLSTL")][0]
    assert mtl["poly_token_a"] == "m1" and mtl["team_a"] == "mtl"          # Montreal = Canadiens


def test_aliases_stay_in_their_league():
    assert P.lnorm("nhl", "lak") == "la" and P.lnorm("nba", "lak") == "lak"
    assert "mon" in P.code_variants("nhl", "MTL") and "lak" in P.code_variants("nhl", "LA")
