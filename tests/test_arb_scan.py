"""Price-inconsistency scan."""
import asyncio

import pytest

from src import arb_scan as A


def m(t, ya, na, st="between", floor=None, cap=None):
    d = {"ticker": t, "status": "active", "yes_ask_dollars": f"{ya:.2f}", "no_ask_dollars": f"{na:.2f}",
         "strike_type": st}
    if floor is not None:
        d["floor_strike"] = floor
    if cap is not None:
        d["cap_strike"] = cap
    return d


def test_all_no_in_one_winner_event():
    # 3 outcomes: NO asks .55 + .55 + .55 = 1.65 < 2 guaranteed
    ev = {"event_ticker": "E1", "mutually_exclusive": True,
          "markets": [m("E1-A", .50, .55), m("E1-B", .50, .55), m("E1-C", .50, .55)]}
    o = [x for x in A.check_event(ev) if x["kind"].startswith("all NO")][0]
    assert o["payoff"] == 2 and o["edge"] == pytest.approx(2 - 1.65 - 3 * A.fee(0.55))


def test_fair_event_has_nothing():
    ev = {"event_ticker": "E2", "mutually_exclusive": True,
          "markets": [m("E2-A", .51, .50), m("E2-B", .51, .50)]}   # YES sums to 1.02, NO sums to 1.00
    assert A.check_event(ev) == []


def test_all_yes_labels_completeness():
    complete = {"event_ticker": "W", "mutually_exclusive": True,
                "markets": [m("W-T60", .30, .72, "less", cap=60), m("W-B61", .30, .72),
                            m("W-T62", .30, .72, "greater", floor=62)]}
    o = [x for x in A.check_event(complete) if x["kind"].startswith("all YES")][0]
    assert o["note"] == "complete list"
    names = {"event_ticker": "N", "mutually_exclusive": True,
             "markets": [m("N-A", .40, .62), m("N-B", .40, .62)]}
    o = [x for x in A.check_event(names) if x["kind"].startswith("all YES")][0]
    assert o["note"] == "list may be incomplete"


def test_ladder_out_of_order():
    # "above 50" YES ask .40 but "above 55" NO ask .55: together .95 for a guaranteed $1
    ev = {"event_ticker": "L", "mutually_exclusive": False,
          "markets": [m("L-50", .40, .62, "greater", floor=50), m("L-55", .47, .55, "greater", floor=55)]}
    o = A.check_event(ev)
    assert len(o) == 1 and o[0]["kind"] == "ladder (above X)"
    assert o[0]["legs"] == [("L-50", "yes", .40), ("L-55", "no", .55)]
    sane = {"event_ticker": "L2", "markets": [m("L2-50", .60, .42, "greater", floor=50),
                                                m("L2-55", .45, .57, "greater", floor=55)]}
    assert A.check_event(sane) == []


def test_depth_at_ask():
    book = {"orderbook": {"yes_dollars": [["0.40", 30]], "no_dollars": [["0.60", 12], ["0.59", 100]]}}
    assert A.depth_at_ask(book, "yes", 0.40) == 12      # YES ask .40 = NO bid .60
    assert A.depth_at_ask(book, "no", 0.60) == 30


class FakeClient:
    async def get_orderbook(self, t):
        return {"orderbook": {"yes_dollars": [["0.45", 50]], "no_dollars": [["0.45", 40]]}}


def test_scan_tracks_episodes(tmp_path):
    db = str(tmp_path / "a.db")
    ev = {"event_ticker": "E1", "title": "Who wins", "mutually_exclusive": True,
          "markets": [m("E1-A", .50, .55), m("E1-B", .50, .55), m("E1-C", .50, .55)]}
    c = FakeClient()
    s1 = asyncio.run(A.scan_once(c, 1000.0, db, events=[ev]))
    s2 = asyncio.run(A.scan_once(c, 1030.0, db, events=[ev]))
    assert s1["new"] >= 1 and s2["new"] == 0
    asyncio.run(A.scan_once(c, 1060.0, db, events=[]))
    rep = A.report(db)
    k = rep["by_kind"].set_index("kind")
    assert k.loc["all NO (one winner at most)", "found"] == 1
    assert k.loc["all NO (one winner at most)", "median_minutes"] == pytest.approx(0.5)
    r = rep["recent"]
    assert r[r["kind"].str.startswith("all NO")]["contracts"].iloc[0] == 50
