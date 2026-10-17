"""Whale-follow study: grouping, snapshots, settlement, copy P&L, slices."""
import asyncio
from datetime import datetime, timezone

import pytest

from src import whale as W

NOW = 1_800_000_000.0


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def tr(ts, ticker, side, price, count, tid):
    other = "no" if side == "yes" else "yes"
    return {"trade_id": tid, "ticker": ticker, "taker_side": side, "created_time": iso(ts),
            f"{side}_price_dollars": f"{price:.2f}", f"{other}_price_dollars": f"{1 - price:.2f}",
            "count_fp": str(count)}


def test_group_prints_merges_a_sweep():
    trades = [tr(NOW, "A-1-X", "yes", 0.60, 1000, "1"), tr(NOW + 0.5, "A-1-X", "yes", 0.61, 500, "2"),
              tr(NOW + 1, "A-1-X", "no", 0.39, 10, "3"), tr(NOW + 10, "A-1-X", "yes", 0.62, 5, "4")]
    evs = W.group_prints(trades)
    big = [e for e in evs if e["contracts"] == 1500][0]
    assert big["prints"] == 2 and big["notional"] == pytest.approx(600 + 305)
    assert big["vwap"] == pytest.approx(905 / 1500)
    assert len(evs) == 3


def test_copy_pnl():
    assert W.copy_pnl("yes", 0.62, "yes") == pytest.approx(1 - 0.62 - W.taker_fee(0.62))
    assert W.copy_pnl("yes", 0.62, "no") == pytest.approx(-0.62 - W.taker_fee(0.62))
    assert W.copy_pnl("yes", 0.99, "yes") is None          # nothing left to gain
    assert W.copy_pnl("yes", -1.0, "yes") is None          # no ask at +1 min


class FakeClient:
    def __init__(self):
        self.trades = []
        self.market = {"ticker": "KXNBAGAME-26OCT20BOSNYK-BOS", "event_ticker": "KXNBAGAME-26OCT20BOSNYK",
                       "title": "Boston at New York", "yes_sub_title": "Boston", "close_time": iso(NOW + 86400),
                       "yes_bid_dollars": "0.63", "yes_ask_dollars": "0.65", "no_bid_dollars": "0.35",
                       "no_ask_dollars": "0.37", "result": ""}

    async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
        assert method == "GET"
        if endpoint.endswith("/markets/trades"):
            return {"trades": list(self.trades)}
        return {"series": {"category": "Sports"}}

    async def get_markets(self, limit=100, tickers=None, **kw):
        return {"markets": [dict(self.market)] if self.market["ticker"] in (tickers or []) else []}


def test_tape_to_settlement(tmp_path):
    db = str(tmp_path / "w.db")
    c = FakeClient()
    tape = W.Tape(min_notional=250)
    tk = c.market["ticker"]
    # normal flow, then a whale buys 2,000 YES at 60-61c
    c.trades = [tr(NOW - 100 + i, tk, "yes", 0.60, 5, f"n{i}") for i in range(20)]
    c.trades += [tr(NOW, tk, "yes", 0.60, 1500, "w1"), tr(NOW + 0.3, tk, "yes", 0.61, 500, "w2")]
    new = asyncio.run(tape.poll(c))
    assert len(new) == 22
    assert asyncio.run(tape.poll(c)) == []                 # de-duplicated
    evs = tape.events(new, NOW + 10)
    assert len(evs) == 1 and evs[0]["contracts"] == 2000 and evs[0]["typical_count"] == 5
    with W.closing(W._conn(db)) as con:
        e = evs[0]
        con.execute("INSERT INTO whales (ts, ticker, series, side, contracts, notional, vwap, first_price, "
                    "last_price, prints, typical_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (e["ts"], tk, tk.split("-")[0], e["side"], e["contracts"], e["notional"], e["vwap"],
                     e["first_price"], e["last_price"], e["prints"], e["typical_count"]))
        con.commit()
    asyncio.run(W.checkpoints(c, NOW + 61, db))
    r = W.rows(db)[0]
    assert r["entry_ask"] == 0.65 and r["event"] == "KXNBAGAME-26OCT20BOSNYK"
    c.market.update(yes_bid_dollars="0.70", yes_ask_dollars="0.72")
    asyncio.run(W.checkpoints(c, NOW + 1801, db))
    assert W.rows(db)[0]["mid_30m"] == pytest.approx(0.71)
    c.market["result"] = "yes"
    asyncio.run(W.settle(c, NOW + 90000, db))
    asyncio.run(W.categories(c, db))
    r = W.rows(db)[0]
    assert r["result"] == "yes" and r["pnl_copy"] == pytest.approx(1 - 0.65 - W.taker_fee(0.65))
    df = W.features(W.rows(db), W.series_meta(db))
    assert df.loc[0, "size_bucket"] == "$1k-5k" and df.loc[0, "category"] == "Sports"
    assert df.loc[0, "drift_30m_c"] == pytest.approx(6.0)
    assert df.loc[0, "multiple_bucket"] == "100x+"
    sm = W.summary(df)
    assert sm["settled"] == 1 and sm["copy_c"] == pytest.approx(100 * (0.35 - W.taker_fee(0.65)))
    t = W.slice_table(df, "size_bucket")
    assert t.loc[0, "group"] == "$1k-5k"


def test_run_loop_records_whale(tmp_path, monkeypatch):
    import time as _t
    now = _t.time()
    c = FakeClient()
    tk = c.market["ticker"]
    c.trades = [tr(now - 30, tk, "no", 0.35, 3000, "big")]
    c.close = lambda: asyncio.sleep(0)
    monkeypatch.setattr("src.clients.kalshi_client.KalshiClient", lambda: c)
    db = str(tmp_path / "w.db")

    async def go():
        try:
            await asyncio.wait_for(W.run_forever(None, interval=0.2, path=db), timeout=0.8)
        except asyncio.TimeoutError:
            pass
    asyncio.run(go())
    r = W.rows(db)
    assert len(r) == 1 and r[0]["side"] == "no" and r[0]["notional"] == pytest.approx(1050)
