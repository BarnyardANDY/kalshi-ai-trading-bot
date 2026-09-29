"""Regression: markets must not silently default to a 50c price."""
from src.utils.market_prices import get_mid_prices


def test_v2_dollar_fields_use_book_midpoint():
    m = {"yes_bid_dollars": "0.03", "yes_ask_dollars": "0.05",
         "no_bid_dollars": "0.95", "no_ask_dollars": "0.97"}
    yes, no = get_mid_prices(m)
    assert abs(yes - 0.04) < 1e-9
    assert abs(no - 0.96) < 1e-9


def test_accepts_get_market_wrapper():
    yes, no = get_mid_prices({"market": {"yes_bid_dollars": 0.60, "yes_ask_dollars": 0.62,
                                         "no_bid_dollars": 0.38, "no_ask_dollars": 0.40}})
    assert abs(yes - 0.61) < 1e-9 and abs(no - 0.39) < 1e-9


def test_legacy_cent_fields():
    yes, no = get_mid_prices({"yes_bid": 20, "yes_ask": 22, "no_bid": 78, "no_ask": 80})
    assert abs(yes - 0.21) < 1e-9 and abs(no - 0.79) < 1e-9


def test_one_sided_book_derives_other_side():
    yes, no = get_mid_prices({"yes_bid_dollars": 0, "yes_ask_dollars": 0,
                              "no_bid_dollars": 0.90, "no_ask_dollars": 0.92})
    assert abs(no - 0.91) < 1e-9 and abs(yes - 0.09) < 1e-9


def test_no_price_returns_zero_not_fifty():
    assert get_mid_prices({}) == (0.0, 0.0)
