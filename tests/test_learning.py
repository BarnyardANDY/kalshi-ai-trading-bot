"""Self-improvement loop: recording, settling, learning, applying (offline)."""
import asyncio
import random

from src import learning as L


def _row(mid, ev, niche, p, m, y, thr=None, liked=None, nl=None, ts=1.0):
    return (mid, ev, niche, ts, p, m, y, thr, liked, nl)


def _history(n_events, bot_good, niche="trump_mentions", seed=1):
    """Synthetic record: truth ~ U(0,1); bot sees truth clearly (good) or noise (bad)."""
    rng = random.Random(seed)
    rows = []
    for e in range(n_events):
        for k in range(3):
            truth = rng.random()
            y = 1 if rng.random() < truth else 0
            market = min(max(truth + rng.gauss(0, 0.15), 0.02), 0.98)
            bot = truth if bot_good else rng.random()
            rows.append(_row(f"M{e}-{k}", f"E{e}", niche, bot, market, y))
    return rows


def test_no_opinion_before_min_events():
    p = L.learn(_history(3, bot_good=True), min_events=8, default_trust=0.6)["trump_mentions"]
    assert p["learned"] is False and p["trust"] == 0.6 and p["paused"] is False
    assert p["events"] == 3


def test_trusts_a_bot_that_beats_the_market():
    p = L.learn(_history(40, bot_good=True), min_events=8)["trump_mentions"]
    assert p["learned"] and p["trust"] >= 0.7 and not p["paused"]
    assert p["brier_ours"] < p["brier_market"]


def test_pauses_a_bot_that_is_just_noise():
    p = L.learn(_history(40, bot_good=False), min_events=8)["trump_mentions"]
    assert p["learned"] and p["trust"] <= 0.2 and p["paused"]


def test_events_weighted_not_rungs():
    rows = [_row(f"KXRT-A-{t}", "KXRT-A", "rotten_tomatoes", .5, .5, 1, t) for t in range(40, 90, 5)]
    rows += [_row("KXRT-B-50", "KXRT-B", "rotten_tomatoes", .5, .5, 0, 50)]
    p = L.learn(rows, min_events=1)["rotten_tomatoes"]
    assert p["events"] == 2 and p["markets"] == 11


def test_rt_drift_from_brackets():
    rows = []
    # Three films predicted at 90% early; each ended in (70, 75] -> drift ~ -17.5 clipped to -10
    for f in "ABC":
        for thr, y in ((65, 1), (70, 1), (75, 0), (80, 0)):
            rows.append(_row(f"KXRT-{f}-{thr}", f"KXRT-{f}", "rotten_tomatoes", .5, .5, y, thr, 18, 2))
    assert L._rt_drift(rows, min_events=4) == -10.0
    rows = []
    for f in "ABC":
        for thr, y in ((80, 1), (85, 0)):
            rows.append(_row(f"KXRT-{f}-{thr}", f"KXRT-{f}", "rotten_tomatoes", .5, .5, y, thr, 17, 3))
    # early 85%, final bracket (80,85] -> midpoint 83 -> drift -2
    assert L._rt_drift(rows, min_events=4) == -2.0


def test_record_dedupes_and_settle_fills_outcome(tmp_path):
    db = str(tmp_path / "l.db")
    assert L.record_prediction("KXRT-X-80", "KXRT-X", "rotten_tomatoes", .6, .5, .7,
                               "2020-01-01T00:00:00Z", 80, 20, 5, path=db, now=100.0)
    assert not L.record_prediction("KXRT-X-80", "KXRT-X", "rotten_tomatoes", .61, .5, path=db, now=200.0)
    assert L.record_prediction("KXRT-X-80", "KXRT-X", "rotten_tomatoes", .7, .5, path=db, now=300.0)

    class K:
        async def get_market(self, t):
            return {"market": {"ticker": t, "result": "yes"}}
    n = asyncio.run(L.settle_predictions(K(), path=db, force=True))
    assert n == 1
    rows = L.load_rows(db)
    assert len(rows) == 2 and all(r[6] == 1 for r in rows)
    # grading uses the latest prediction per market
    assert L._latest_per_market(rows)[0][4] == .7


def test_unsettled_markets_stay_open(tmp_path):
    db = str(tmp_path / "l.db")
    L.record_prediction("KXTRUMPSAY-1-A", "KXTRUMPSAY-1", "trump_mentions", .3, .2, path=db, now=1.0)

    class K:
        async def get_market(self, t):
            return {"market": {"ticker": t, "result": ""}}
    assert asyncio.run(L.settle_predictions(K(), path=db, force=True)) == 0
    assert L.pending_count(db) == {"trump_mentions": 1}


def test_apply_blends_with_trust(monkeypatch):
    monkeypatch.setattr(L, "niche_params", lambda n: {"trust": 0.25, "paused": False})
    assert abs(L.apply("x", 0.9, 0.5) - 0.6) < 1e-9


def test_would_have_traded_takes_first_disagreement_and_scores_after_fees():
    from src import learning
    # (market_id, event, niche, ts, our, mkt, outcome, thr, liked, not_liked)
    rows = [
        ("W-1", "W", "weather", 1, 0.50, 0.45, 0, None, None, None),   # gap 5: not a bet
        ("W-1", "W", "weather", 2, 0.60, 0.40, 0, None, None, None),   # first bet: YES at 40, loses
        ("W-1", "W", "weather", 3, 0.90, 0.40, 0, None, None, None),   # later: ignored
        ("W-2", "W", "weather", 2, 0.10, 0.30, 0, None, None, None),   # NO at 70, wins
        ("W-3", "W", "weather", 2, 0.30, 0.01, 1, None, None, None),   # 1c: no realistic fill
    ]
    rep = learning.would_have_traded(rows, 0.10)["weather"]
    assert rep["bets"] == 2 and rep["wins"] == 1 and rep["events"] == 1
    fee = 0.07 * 0.4 * 0.6 + 0.07 * 0.3 * 0.7
    assert abs(rep["pnl"] - ((0 - 0.40) + (1 - 0.70) - fee)) < 1e-9
    assert rep["underdog_bets"] == 0


def test_would_have_traded_empty():
    from src import learning
    assert learning.would_have_traded([], 0.1) == {}
