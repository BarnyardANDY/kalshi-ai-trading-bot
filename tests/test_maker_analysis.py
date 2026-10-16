"""Maker test: extra tracking and slice-and-dice."""
import asyncio
import sqlite3

import pytest

from src import maker as MK
from src import maker_analysis as MA
from tests.test_maker import FakeClient, NOW, mkt, tr


def test_old_database_gets_new_columns(tmp_path):
    db = tmp_path / "old.db"
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, event TEXT, series TEXT,
                 title TEXT, side TEXT, price REAL, qty INTEGER, queue_ahead REAL, improved INTEGER, spread REAL,
                 posted_ts REAL, close_ts REAL, status TEXT, fill_ts REAL, fee REAL, result TEXT, pnl REAL,
                 settled_ts REAL, note TEXT)""")
    c.execute("INSERT INTO orders (ticker, event, series, side, price, qty, queue_ahead, improved, spread, posted_ts,"
              " close_ts, status, fill_ts, fee, result, pnl, settled_ts, note) VALUES "
              "('KXA-1-X','KXA-1','KXA','yes',0.6,5,0,1,0.05,?,?,'settled',?,0.02,'yes',1.98,?,'sold through at 0.58')",
              (NOW, NOW + 7200, NOW + 60, NOW + 3000))
    c.commit()
    c.close()
    rows = MK.rows(path=str(db))
    assert "mid_5m" in rows[0] and rows[0]["mid_5m"] is None
    f = MA.features(rows, {})
    assert f.loc[0, "fill_kind"] == "price dropped through our bid"   # derived from the old note
    assert f.loc[0, "fill_to_result"] == "<1h (live)"


def test_cycle_records_snapshot_markouts_and_category(tmp_path):
    db = str(tmp_path / "m.db")
    MK._last_full_settle = 0.0
    m = mkt(open_interest_fp="1234", previous_price_dollars="0.66", expected_expiration_time=None)
    c = FakeClient([m])

    async def series_req(method, endpoint, params=None, require_auth=True, **kw):
        if endpoint.startswith("/trade-api/v2/series/"):
            return {"series": {"category": "Sports", "title": "NFL games"}}
        return await FakeClient._make_authenticated_request(c, method, endpoint, params, require_auth)
    c._make_authenticated_request = series_req

    asyncio.run(MK.run_cycle(c, now=NOW, path=db))
    r = MK.rows("open", db)[0]
    assert r["open_interest"] == 1234 and r["side_prev_day"] == pytest.approx(0.66)
    assert r["side_mid_post"] == pytest.approx(0.71)
    c.trades = [tr(NOW + 60, "no", 0.69, 1)]
    asyncio.run(MK.run_cycle(c, now=NOW + 120, path=db))
    r = MK.rows("filled", db)[0]
    assert r["fill_kind"] == "through" and r["fill_trade_price"] == pytest.approx(0.69)
    # 5 minutes after the fill the price has dropped to .66/.68 -> mid .67
    c.markets[r["ticker"]].update(yes_bid_dollars="0.66", yes_ask_dollars="0.68")
    asyncio.run(MK.run_cycle(c, now=NOW + 60 + 400, path=db))
    r = MK.rows("filled", db)[0]
    assert r["mid_5m"] == pytest.approx(0.67)
    assert MK.series_meta(db)["KXNFLGAME"]["category"] == "Sports"
    f = MA.features(MK.rows(path=db), MK.series_meta(db))
    filled = f[f["status"] == "filled"].iloc[0]
    assert filled["markout_5m"] == "fell 2-4c" and filled["category"] == "Sports"


def _settled(n, price, won, event_prefix, **extra):
    out = []
    for i in range(n):
        out.append(dict(id=i, ticker=f"{event_prefix}{i}-A", event=f"{event_prefix}{i}", series="KXS", title="t",
                        side="yes", price=price, qty=5, queue_ahead=0, improved=1, spread=0.03,
                        posted_ts=NOW + i * 3600, close_ts=NOW + 99999, status="settled",
                        fill_ts=NOW + i * 3600 + 60, fee=0.02, result="yes" if won(i) else "no",
                        pnl=(5 - 5 * price - 0.02) if won(i) else (-5 * price - 0.02),
                        settled_ts=NOW + i * 3600 + 1800, note="sold through at 0.5", **extra))
    return out


def test_slice_verdicts():
    good = _settled(200, 0.60, lambda i: i % 10 < 8, "G")          # wins 80% at 60c: clearly positive
    bad = _settled(200, 0.90, lambda i: i % 10 < 7, "B")           # wins 70% at 90c: clearly negative
    meh = _settled(6, 0.70, lambda i: i % 3 != 0, "M")             # too few to say
    f = MA.features(good + bad + meh, {})
    t = MA.slice_table(f, "price_bucket").set_index("group")
    assert t.loc["55-64c", "verdict"] == "clearly positive"
    assert t.loc["85-95c", "verdict"] == "clearly negative"
    assert t.loc["65-74c", "verdict"] == "too early to tell"
    assert set(MA.DIMENSIONS.values()) <= set(f.columns)
