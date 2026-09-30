"""Edge after costs, slippage and resolution-time guards."""
from datetime import datetime, timedelta, timezone

import pytest

from src import cost_edge as C

BOOK = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.46",
        "no_bid_dollars": "0.54", "no_ask_dollars": "0.60"}


def test_fee_per_contract():
    assert C.fee_per_contract(0.5) == pytest.approx(0.0175)
    assert C.fee_per_contract(0.0) == 0.0


def test_net_edge_uses_ask_and_fee():
    e = C.net_edges(0.55, BOOK)
    # Midpoint is 0.43 -> "12 pt edge"; against the 46c ask + ~1.7c fee it's ~7 pts
    assert e["net_yes"] == pytest.approx(0.55 - (0.46 + 0.07 * 0.46 * 0.54))
    assert e["best_side"] == "YES" and e["best_net"] < 0.10
    no_fee = C.net_edges(0.55, BOOK, fee_aware=False)
    assert no_fee["net_yes"] == pytest.approx(0.09)


def test_net_edge_picks_no_side():
    e = C.net_edges(0.20, BOOK)
    assert e["best_side"] == "NO"
    assert e["net_no"] == pytest.approx(0.80 - (0.60 + 0.07 * 0.60 * 0.40))


def test_no_book_means_no_edge():
    empty = {"yes_bid_dollars": "0", "yes_ask_dollars": "0", "no_bid_dollars": "0", "no_ask_dollars": "0"}
    assert C.net_edges(0.9, empty)["best_net"] is None
    assert C.slippage(empty, "YES") is None


def test_slippage_is_ask_above_mid():
    assert C.slippage(BOOK, "YES") == pytest.approx(0.03)
    assert C.slippage({"market": BOOK}, "NO") == pytest.approx(0.03)


def test_days_to_close():
    now = datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert C.days_to_close({"close_time": "2026-10-05T14:00:00Z"}, now) == pytest.approx(5 + 14 / 24)
    assert C.days_to_close({}, now) is None
