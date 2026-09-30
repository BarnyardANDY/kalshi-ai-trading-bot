"""Hold-to-settlement for niche positions and realistic paper fills."""
from datetime import datetime

import pytest

from src import hold_policy as H
from src import paper_fills as F
from src.utils.database import Position

BOOK = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44",
        "no_bid_dollars": "0.56", "no_ask_dollars": "0.60"}


def _pos(side="YES", entry=0.44, qty=5):
    return Position(market_id="KXRT-DIG-60", side=side, entry_price=entry, quantity=qty, timestamp=datetime.now())


def test_kalshi_fee_matches_formula():
    # 0.07 * 5 * 0.44 * 0.56 = 0.08624 -> rounds UP to $0.09
    assert F.kalshi_fee(0.44, 5) == 0.09
    assert F.kalshi_fee(0.50, 1) == 0.02   # 0.0175 -> 0.02
    assert F.kalshi_fee(0.0, 5) == 0.0


def test_entry_pays_ask_plus_fee_and_exit_gets_bid_minus_fee():
    assert F.entry_fill(BOOK, "YES", 5) == pytest.approx(0.44 + 0.09 / 5)
    assert F.entry_fill(BOOK, "NO", 5) == pytest.approx(0.60 + F.kalshi_fee(0.60, 5) / 5)
    assert F.exit_fill(BOOK, "YES", 5) == pytest.approx(0.40 - F.kalshi_fee(0.40, 5) / 5)
    # Round trip with no price move loses money (spread + fees), as in reality
    assert F.exit_fill(BOOK, "YES", 5) < F.entry_fill(BOOK, "YES", 5)


def test_no_ask_or_bid_means_no_fill():
    empty = {"yes_bid_dollars": "0", "yes_ask_dollars": "0", "no_bid_dollars": "0", "no_ask_dollars": "0"}
    assert F.entry_fill(empty, "YES", 1) is None and F.exit_fill(empty, "NO", 1) is None
    assert F.entry_fill({"market": BOOK}, "YES", 5) is not None  # wrapper accepted


def test_resolution_settles_case_insensitive():
    assert H.resolution(_pos("YES"), {"result": "yes"}) == 1.0
    assert H.resolution(_pos("NO"), {"result": "yes"}) == 0.0
    assert H.resolution(_pos("NO"), {"result": "NO"}) == 1.0
    assert H.resolution(_pos(), {"result": ""}) is None


def _patch_learning(monkeypatch, our_prob, trust=1.0):
    from src import learning
    monkeypatch.setattr(learning, "latest_prediction",
                        lambda mid, **k: {"our_prob": our_prob, "market_prob": 0.42, "ts": 0, "niche": "rotten_tomatoes"})
    monkeypatch.setattr(learning, "niche_params", lambda n: {"trust": trust, "paused": False})


def test_holds_through_small_moves(monkeypatch):
    _patch_learning(monkeypatch, 0.55)  # we think YES worth 55c; bid 40c
    ok, reason, _ = H.niche_exit_decision(_pos("YES"), BOOK, "rotten_tomatoes")
    assert ok is False and reason.startswith("hold")


def test_exits_when_bid_beats_our_value(monkeypatch):
    _patch_learning(monkeypatch, 0.30)  # new reviews: now worth 30c, bid is 40c
    ok, reason, price = H.niche_exit_decision(_pos("YES"), BOOK, "rotten_tomatoes")
    assert ok and reason.startswith("value_exit") and price == 0.40


def test_no_side_uses_no_bid(monkeypatch):
    _patch_learning(monkeypatch, 0.30)  # NO worth 70c; NO bid 56c -> hold
    ok, _, _ = H.niche_exit_decision(_pos("NO", 0.60), BOOK, "rotten_tomatoes")
    assert ok is False


def test_holds_without_fresh_estimate(monkeypatch):
    from src import learning
    monkeypatch.setattr(learning, "latest_prediction", lambda mid, **k: None)
    ok, reason, _ = H.niche_exit_decision(_pos(), BOOK, "rotten_tomatoes")
    assert ok is False and "no fresh estimate" in reason


def test_settles_even_without_estimate(monkeypatch):
    from src import learning
    monkeypatch.setattr(learning, "latest_prediction", lambda mid, **k: None)
    ok, reason, price = H.niche_exit_decision(_pos("YES"), dict(BOOK, result="yes"), "rotten_tomatoes")
    assert ok and reason == "market_resolution" and price == 1.0
