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
    assert g == {"yes": "New Orleans", "no": "Atlanta", "date": date(2026, 10, 5), "nickname": "Saints"}
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
    assert RC.get("SHADOW_NICHES") == "sports"
    assert RC.get("SPORTS_LEAGUES") == "nfl,ncaaf"
