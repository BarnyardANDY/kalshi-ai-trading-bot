"""
Realistic paper fills: what a real order would actually have cost or paid.

Paper trades used to fill at the bid/ask midpoint with no fees, which made
small trades look profitable when real ones would lose. Now:

* Buying pays the current ASK for that side, plus Kalshi's trading fee.
* Selling receives the current BID for that side, minus the fee.
* Settlement pays $1 / $0 with no fee (Kalshi doesn't charge to settle).

Kalshi's taker fee per order is round_up(0.07 x contracts x P x (1 - P)) to
the next cent, where P is the price in dollars. Override the 0.07 rate with
KALSHI_FEE_RATE if Kalshi changes it.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

from src.utils.market_prices import get_market_prices


def fee_rate() -> float:
    try:
        return float(os.getenv("KALSHI_FEE_RATE", "0.07"))
    except ValueError:
        return 0.07


def kalshi_fee(price: float, contracts: int) -> float:
    """Total fee in dollars for one order of ``contracts`` at ``price``."""
    if contracts <= 0 or not 0 < price < 1:
        return 0.0
    raw = fee_rate() * contracts * price * (1 - price)
    return math.ceil(round(raw * 100, 6)) / 100


def _market(market: Dict[str, Any]) -> Dict[str, Any]:
    return market.get("market", market) if isinstance(market.get("market"), dict) else market


def entry_fill(market: Dict[str, Any], side: str, contracts: int) -> Optional[float]:
    """Per-contract cost of buying now (ask + fee share). None if no ask."""
    yes_bid, yes_ask, no_bid, no_ask = get_market_prices(_market(market))
    ask = yes_ask if side.upper() == "YES" else no_ask
    if not 0 < ask < 1:
        return None
    return ask + kalshi_fee(ask, contracts) / max(1, contracts)


def exit_fill(market: Dict[str, Any], side: str, contracts: int) -> Optional[float]:
    """Per-contract proceeds of selling now (bid - fee share). None if no bid."""
    yes_bid, yes_ask, no_bid, no_ask = get_market_prices(_market(market))
    bid = yes_bid if side.upper() == "YES" else no_bid
    if not 0 < bid < 1:
        return None
    return max(0.0, bid - kalshi_fee(bid, contracts) / max(1, contracts))
