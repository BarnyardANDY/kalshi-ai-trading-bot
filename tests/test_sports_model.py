"""Sports: matching Kalshi games to sportsbook odds and devigging (offline)."""
import asyncio
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from src import sports_model as S
from src.utils.database import Market

NFL = {"ticker": "KXNFLGAME-26OCT05ATLNO-NO", "event_ticker": "KXNFLGAME-26OCT05ATLNO",
       "title": "New Orleans wins", "yes_sub_title": "New Orleans", "no_sub_title": "Atlanta",
       "rules_primary": "If New Orleans wins the ATL Falcons vs NO Saints Pro Football game originally "
                        "scheduled for Oct 5, 2026, then the market resolves to Yes."}
NCAAF = {"ticker": "KXNCAAFGAME-26OCT10BSUFRES-FRES", "title": "Fresno St. wins", "yes_sub_title": "Fresno St.",
         "rules_primary": "If Fresno St. wins the Boise St. vs Fresno St. college football game originally "
                          "scheduled for Oct 10, 2026, then the market resolves to Yes."}
MLB = {"ticker": "KXMLBGAME-26OCT051905BOSBAL-BOS", "title": "Boston wins", "yes_sub_title": "Boston",
       "rules_primary": "If Boston wins the Boston vs Baltimore professional baseball game originally "
                        "scheduled for Oct 5, 2026 at 7:05 PM EDT, then the market resolves to Yes."}


def _ev(home, away, start, books):
    return {"home_team": home, "away_team": away, "commence_time": start,
            "bookmakers": [{"key": k, "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": ph}, {"name": away, "price": pa}]}]} for k, ph, pa in books]}


def test_parse_games():
    g = S.parse_game(NFL)
    assert g["yes"] == "New Orleans" and g["nickname"] == "Saints"
    assert g["no"] == "ATL Falcons" and g["no_nickname"] == "Falcons" and g["date"] == date(2026, 10, 5)
    g = S.parse_game(NCAAF)
    assert g["yes"] == "Fresno St." and g["no"] == "Boise St." and g["date"] == date(2026, 10, 10)
    g = S.parse_game(MLB)
    assert g["yes"] == "Boston" and g["no"] == "Baltimore"


def test_league_of():
    assert S.league_of("KXNFLGAME-26OCT05ATLNO-NO") == ("nfl", "americanfootball_nfl")
    assert S.league_of("KXRT-DIG-45") is None


def test_find_event_and_devig():
    events = [
        _ev("New Orleans Saints", "Atlanta Falcons", "2026-10-05T17:00:00Z",
            [("pinnacle", 1.80, 2.10), ("draftkings", 1.77, 2.10)]),
        _ev("Boise State Broncos", "Fresno State Bulldogs", "2026-10-11T02:30:00Z", [("fanduel", 1.5, 2.6)]),
    ]
    ev, team = S.find_event(S.parse_game(NFL), events)
    assert team == "New Orleans Saints"
    c = S.consensus(ev, team)
    pin = (1 / 1.80) / (1 / 1.80 + 1 / 2.10)
    dk = (1 / 1.77) / (1 / 1.77 + 1 / 2.10)
    assert c["prob"] == pytest.approx((3 * pin + dk) / 4) and c["sharp"]
    # late game: UTC date is next day, Eastern date matches
    ev, team = S.find_event(S.parse_game(NCAAF), events)
    assert team == "Fresno State Bulldogs"
    assert S.consensus(ev, team)["prob"] == pytest.approx((1 / 2.6) / (1 / 1.5 + 1 / 2.6))


def test_ambiguous_city_resolved_by_opponent_and_doubleheader_skipped():
    la = {"ticker": "X-LAD", "yes_sub_title": "Los Angeles D", "rules_primary":
          "If Los Angeles D wins the Los Angeles D vs San Diego professional baseball game originally "
          "scheduled for Oct 5, 2026 at 7 PM EDT, then the market resolves to Yes."}
    events = [_ev("Los Angeles Dodgers", "San Diego Padres", "2026-10-05T23:00:00Z", [("fanduel", 1.7, 2.2)]),
              _ev("Los Angeles Angels", "Seattle Mariners", "2026-10-05T23:00:00Z", [("fanduel", 2.0, 1.8)])]
    ev, team = S.find_event(S.parse_game(la), events)
    assert team == "Los Angeles Dodgers"
    double = events[:1] + [_ev("Los Angeles Dodgers", "San Diego Padres", "2026-10-05T18:00:00Z", [])]
    assert S.find_event(S.parse_game(la), double) is None


def test_predict_sports_end_to_end(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test")
    start = (datetime.now(timezone.utc) + timedelta(days=2)).replace(microsecond=0)
    day = start.astimezone(S._ET)
    rules = (f"If New Orleans wins the ATL Falcons vs NO Saints Pro Football game originally scheduled for "
             f"{day:%b} {day.day}, {day.year}, then the market resolves to Yes.")
    info = dict(NFL, rules_primary=rules)

    async def fake_fetch(sport, key, refresh):
        assert sport == "americanfootball_nfl"
        return [_ev("New Orleans Saints", "Atlanta Falcons", start.isoformat().replace("+00:00", "Z"),
                    [("pinnacle", 1.80, 2.10)])]
    monkeypatch.setattr(S, "fetch_odds", fake_fetch)

    class K:
        async def get_market(self, t):
            return {"market": info}

    class Log:
        def info(self, *a, **k): pass

    mk = Market(NFL["ticker"], "New Orleans wins", 0.555, 0.445, 9000, int(time.time()) + 86400 * 3,
                "sports", "active", datetime.now())
    out = asyncio.run(S.predict_sports([mk], K(), Log()))
    assert out[NFL["ticker"]][0] == pytest.approx((1 / 1.80) / (1 / 1.80 + 1 / 2.10))
    assert out[NFL["ticker"]][1] == 0.85


def test_no_key_skips(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    class Log:
        def info(self, *a, **k): pass
    assert asyncio.run(S.predict_sports([], None, Log())) == {}


def test_shadow_setting_default():
    from src import runtime_config as RC
    assert RC.get("SHADOW_NICHES") == "sports,stocks"
    assert RC.get("SPORTS_LEAGUES") == "nfl,ncaaf"


def test_real_kalshi_shape_no_subtitle_repeats_yes_team():
    # As seen live: no_sub_title is the YES team again, and teams are "KC Chiefs" / "LV Raiders"
    m = {"ticker": "KXNFLGAME-26OCT04KCLV-LV", "title": "Las Vegas wins", "yes_sub_title": "Las Vegas",
         "no_sub_title": "Las Vegas",
         "rules_primary": "If Las Vegas wins the KC Chiefs vs LV Raiders Pro Football game originally "
                          "scheduled for Oct 4, 2026, then the market resolves to Yes."}
    g = S.parse_game(m)
    assert g["nickname"] == "Raiders" and g["no_nickname"] == "Chiefs"
    events = [_ev("Las Vegas Raiders", "Kansas City Chiefs", "2026-10-04T20:25:00Z",
                  [("pinnacle", 2.6, 1.55)]),
              _ev("Pittsburgh Steelers", "Cleveland Browns", "2026-10-02T00:15:00Z", [("fanduel", 1.7, 2.2)])]
    ev, team = S.find_event(g, events)
    assert team == "Las Vegas Raiders"
    assert S.consensus(ev, team)["prob"] == pytest.approx((1 / 2.6) / (1 / 2.6 + 1 / 1.55))
    pit = {"ticker": "KXNFLGAME-26OCT01PITCLE-PIT", "yes_sub_title": "Pittsburgh", "no_sub_title": "Pittsburgh",
           "rules_primary": "If Pittsburgh wins the PIT Steelers vs CLE Browns Pro Football game originally "
                            "scheduled for Oct 1, 2026, then the market resolves to Yes."}
    ev, team = S.find_event(S.parse_game(pit), events)
    assert team == "Pittsburgh Steelers"


def test_college_and_mlb_without_nicknames():
    m = {"ticker": "KXNCAAFGAME-26OCT03ALAMSST-ALA", "yes_sub_title": "Alabama", "no_sub_title": "Alabama",
         "rules_primary": "If Alabama wins the Alabama vs Mississippi St. college football game originally "
                          "scheduled for Oct 3, 2026, then the market resolves to Yes."}
    g = S.parse_game(m)
    assert g["yes"] == "Alabama" and g["no"] == "Mississippi St."
    events = [_ev("Alabama Crimson Tide", "Mississippi State Bulldogs", "2026-10-03T23:30:00Z",
                  [("draftkings", 1.2, 4.8)])]
    assert S.find_event(g, events)[1] == "Alabama Crimson Tide"
