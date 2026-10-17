"""Maker v2: one bet per match, before the event only, first in line, pulls orders when price turns."""
import asyncio

import pytest

from src import maker as MK
from tests.test_maker import CFG, FakeClient, NOW, iso, mkt, tr

V2 = {**CFG, **MK.V2_RULES}


@pytest.fixture(autouse=True)
def _v2(monkeypatch):
    monkeypatch.setenv("MAKER_VARIANT", "v2")
    MK._last_full_settle = 0.0


def wide(ticker="KXNFLGAME-26OCT18BUFMIA-BUF", end_h=10, **kw):
    # 70/74 book: YES favourite, 4c gap -> we bid 71, first in line
    return mkt(ticker=ticker, yb=0.70, ya=0.74, hours=end_h + 2,
               expected_expiration_time=iso(NOW + end_h * 3600), **kw)


def test_settings_switch_to_v2():
    assert MK.settings()["v2"] is True and MK.settings()["interval"] == 15
    assert MK.db_path().endswith("maker_v2.db")


def test_v2_quote_rules():
    assert MK.quote_for(wide(), V2, NOW)["price"] == 0.71
    assert MK.quote_for(mkt(yb=0.70, ya=0.72, hours=12), V2, NOW) is None        # 2c gap: busy market, skip
    assert MK.quote_for(wide(end_h=2), V2, NOW) is None                           # event about to start/live
    assert MK.quote_for(wide(end_h=2), CFG, NOW) is not None                      # v1 would still quote it


def test_one_bet_per_match():
    a = MK.quote_for(wide("KXEPLGAME-26OCT18ARSCHE-ARS"), V2, NOW)
    b = MK.quote_for(wide("KXEPLTOTAL-26OCT18ARSCHE-3"), V2, NOW)          # same match, other event
    c = MK.quote_for(wide("KXEPLGAME-26OCT18LIVMCI-LIV"), V2, NOW)
    assert MK.match_key(a["event"]) == MK.match_key(b["event"]) == "26OCT18ARSCHE"
    assert [q["ticker"] for q in MK.pick([a, b, c], [], V2)] == [a["ticker"], c["ticker"]]
    assert [q["ticker"] for q in MK.pick([b, c], [], V2, blocked_matches={"26OCT18ARSCHE"})] == [c["ticker"]]


def test_pulls_order_when_price_turns(tmp_path):
    db = str(tmp_path / "v2.db")
    c = FakeClient([wide()])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    assert MK.rows("open", db)[0]["price"] == 0.71
    c.markets["KXNFLGAME-26OCT18BUFMIA-BUF"].update(yes_bid_dollars="0.68", yes_ask_dollars="0.72")  # mid 72 -> 70
    asyncio.run(MK.run_cycle(c, now=NOW + 15, path=db, universe=[]))
    r = MK.rows(path=db)[0]
    assert r["status"] == "cancelled" and r["note"] in ("price fell below our bid", "price moving against us")


def test_fill_before_pull_still_counts(tmp_path):
    db = str(tmp_path / "v2.db")
    c = FakeClient([wide()])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    c.trades = [tr(NOW + 5, "no", 0.69, 3)]           # a seller hit through us before we could react
    c.markets["KXNFLGAME-26OCT18BUFMIA-BUF"].update(yes_bid_dollars="0.66", yes_ask_dollars="0.70")
    asyncio.run(MK.run_cycle(c, now=NOW + 15, path=db, universe=[]))
    assert MK.rows(path=db)[0]["status"] == "filled"


def test_pulls_before_event(tmp_path):
    db = str(tmp_path / "v2.db")
    c = FakeClient([wide(end_h=5)])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    asyncio.run(MK.run_cycle(c, now=NOW + 2.5 * 3600, path=db, universe=[]))
    r = MK.rows(path=db)[0]
    assert r["status"] == "cancelled" and r["note"] == "event getting close"


def test_unchanged_market_keeps_order(tmp_path):
    db = str(tmp_path / "v2.db")
    c = FakeClient([wide()])
    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    asyncio.run(MK.run_cycle(c, now=NOW + 15, path=db, universe=[]))
    asyncio.run(MK.run_cycle(c, now=NOW + 30, path=db, universe=[]))
    assert MK.rows(path=db)[0]["status"] == "open"
