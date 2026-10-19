"""Speed-gap study: fair value, gap episodes, late-entry P&L, report."""
import math
import sqlite3

import pytest

from src import crypto_gap as G


def test_fair_prob_basics():
    assert G.fair_prob_up(100.0, 100.0, 600, 1e-4) == pytest.approx(0.5, abs=1e-6)
    hi = G.fair_prob_up(100.2, 100.0, 600, 1e-4)
    assert 0.5 < hi < 1
    # less time left -> more certain
    assert G.fair_prob_up(100.2, 100.0, 120, 1e-4) > hi
    # final minute: half the average is already locked in above the strike
    p = G.fair_prob_up(99.98, 100.0, 30, 1e-4, final_avg_so_far=100.3)
    assert p > 0.99


def test_realised_sigma_uses_time():
    pts = [(i * 2.0, 100 * math.exp(0.001 * ((-1) ** i))) for i in range(100)]  # +-0.2% every 2 s
    s = G.realised_sigma(pts)
    assert s == pytest.approx(math.sqrt((0.002 ** 2) / 2), rel=0.05)


def test_parse_kraken():
    d = {"result": {"XXBTZUSD": {"c": ["67000.1", "1"]}, "XETHZUSD": {"c": ["2500.5", "2"]},
                    "SOLUSD": {"c": ["150.2", "3"]}}}
    assert G.parse_kraken(d) == {"XBTUSD": 67000.1, "ETHUSD": 2500.5, "SOLUSD": 150.2}


def _s(ts, p_yes, yes_ask, no_ask, tk="KXBTC15M-X"):
    return {"ts": ts, "ticker": tk, "series": "KXBTC15M", "p_yes": p_yes, "yes_ask": yes_ask, "no_ask": no_ask,
            "yes_bid": yes_ask - 0.01, "no_bid": no_ask - 0.01, "tau": 300}


def test_episodes_and_late_entry():
    # fair 70%, ask 55c for 3 seconds (edge ~ .70-.55-.017 = 13c), then the market catches up to 70c
    ss = [_s(0, 0.70, 0.60, 0.42), _s(1, 0.70, 0.55, 0.47), _s(2, 0.70, 0.55, 0.47), _s(3, 0.70, 0.56, 0.46),
          _s(4, 0.70, 0.70, 0.32), _s(5, 0.70, 0.71, 0.31)]
    eps = G.episodes(ss, 0.10)
    assert len(eps) == 1 and eps[0]["side"] == "yes" and eps[0]["duration"] == 3
    assert G.simulate(eps[0], ss, "yes", 0) == pytest.approx(1 - 0.55 - G.taker_fee(0.55))
    assert G.simulate(eps[0], ss, "yes", 3) == pytest.approx(1 - 0.70 - G.taker_fee(0.70))   # too late: paid up
    assert G.simulate(eps[0], ss, "no", 0) == pytest.approx(-0.55 - G.taker_fee(0.55))


def test_gap_breaks_on_missing_samples():
    ss = [_s(0, 0.7, 0.5, 0.5), _s(10, 0.7, 0.5, 0.5)]
    assert len(G.episodes(ss, 0.05)) == 2


def test_report_end_to_end(tmp_path):
    db = str(tmp_path / "g.db")
    c = G._conn(db)
    rows = []
    for i in range(120):
        ask = 0.55 if 30 <= i < 35 else 0.70
        rows.append((i, "KXBTC15M", "T1", 900, 100.0, 100.1, 3, 900 - i, 1e-4, 0.70, ask - 0.01, ask, 0.29, 0.31))
    c.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.execute("INSERT INTO outcomes VALUES ('T1','KXBTC15M',900,'yes',100.0)")
    c.commit()
    rep = G.report(db)
    g5 = [g for g in rep["gaps"] if g["gaps"] == "5c+ after fee"][0]
    assert g5["count"] == 1 and g5["median seconds"] == 5 and g5["markets"] == 1
    assert g5["0s late: c/contract"] == pytest.approx(100 * (0.45 - G.taker_fee(0.55)))
    assert rep["calibration"][0]["samples"] > 0


def test_run_loop_with_fake_feeds(tmp_path, monkeypatch):
    import asyncio
    import time as _t
    from datetime import datetime, timezone

    close = _t.time() + 600
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")
    market = {"ticker": "KXBTC15M-26OCT171215-15", "close_time": iso(close), "open_time": iso(close - 900),
              "floor_strike": 67000.0, "strike_type": "greater_or_equal",
              "yes_bid_dollars": "0.48", "yes_ask_dollars": "0.50", "no_bid_dollars": "0.50", "no_ask_dollars": "0.52"}

    class FakeKalshi:
        async def _make_authenticated_request(self, method, endpoint, params=None, require_auth=True):
            if endpoint.endswith("/markets"):
                return {"markets": [market]}
            return {"market": market}

        async def get_market(self, t):
            return {"market": market}

        async def close(self):
            pass

    class Resp:
        def __init__(self, j):
            self.status_code, self._j = 200, j

        def json(self):
            return self._j

    class FakeHttp:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            if "coinbase" in url:
                return Resp({"price": "67100.0"})
            if "bitstamp" in url:
                return Resp({"last": "67110.0"})
            return Resp({"result": {"XXBTZUSD": {"c": ["67090.0", "1"]}}})

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeHttp)
    monkeypatch.setattr("src.clients.kalshi_client.KalshiClient", FakeKalshi)
    db = str(tmp_path / "g.db")

    async def go():
        try:
            await asyncio.wait_for(G.run_forever(None, interval=0.2, series=["KXBTC15M"], path=db), timeout=1.1)
        except asyncio.TimeoutError:
            pass
    asyncio.run(go())
    rows = sqlite3.connect(db).execute("SELECT spot, n_src, strike, p_yes, yes_ask FROM samples").fetchall()
    assert len(rows) >= 3
    spot, n, strike, p, ask = rows[-1]
    assert spot == 67100.0 and n == 3 and strike == 67000.0 and p > 0.5 and ask == 0.50


def test_report_written_and_read(tmp_path):
    db = str(tmp_path / "g.db")
    G._conn(db).close()
    out = str(tmp_path / "r.json")
    G.write_report(db, out)
    r = G.read_report(out)
    assert r["samples"] == 0 and "computed_ts" in r
    assert G.read_report(str(tmp_path / "missing.json")) is None


def _ss(points):
    """(ts, p_yes, yes_bid, yes_ask, tau) -> sample dicts for one ticker."""
    return [{"ts": t, "ticker": "T", "series": "KXBTC15M", "p_yes": p, "yes_bid": b, "yes_ask": a, "tau": tau,
             "no_ask": round(1 - b, 2) if b else None} for t, p, b, a, tau in points]


def test_kalshi_lag_measures_catch_up_and_our_persistence():
    pts = [(0, 0.50, 0.49, 0.51, 600), (1, 0.60, 0.49, 0.51, 599),   # we jump 10c; Kalshi still at 0.50
           (2, 0.60, 0.53, 0.55, 598), (3, 0.60, 0.59, 0.61, 597)]  # Kalshi: 40% after 1 s, 100% after 2 s
    j = G.kalshi_lag(_ss(pts), waits=(0, 1, 2))
    assert len(j) == 1
    assert j[0]["k0"] == pytest.approx(0.0) and j[0]["k1"] == pytest.approx(0.4) and j[0]["k2"] == pytest.approx(1.0)
    assert j[0]["m2"] == pytest.approx(1.0)


def test_kalshi_lag_ignores_small_moves_and_final_seconds():
    assert G.kalshi_lag(_ss([(0, 0.50, 0.49, 0.51, 600), (1, 0.52, 0.49, 0.51, 599)])) == []
    assert G.kalshi_lag(_ss([(0, 0.50, 0.49, 0.51, 60), (1, 0.70, 0.49, 0.51, 59)])) == []


def test_anchored_samples_use_kalshi_mid_plus_our_move():
    pts = [(t, 0.80 if t < 6 else 0.90, 0.49, 0.51, 600 - t) for t in range(10)]   # our model says 0.80 throughout
    a = G.anchored_samples(_ss(pts), lag=5)
    assert a[0]["p_yes"] is None                               # nothing 5 s earlier yet
    assert a[5]["p_yes"] == pytest.approx(0.50)                # Kalshi's 0.50, our model unchanged: no gap
    assert a[6]["p_yes"] == pytest.approx(0.60)                # our model +10c in the last 5 s


def test_clustered_margin_counts_a_market_once():
    same = [("A", 1.0)] * 50 + [("B", -1.0)] * 50              # two markets, perfectly correlated within each
    _, m_clu, k = G._clustered(same)
    assert k == 2 and m_clu > 100                              # huge: really just two bets
    indep = [(str(i), 1.0 if i % 2 else -1.0) for i in range(100)]
    _, m_ind, _ = G._clustered(indep)
    assert m_ind < 25
