"""
Exit rules for niche positions: hold to settlement unless the bot's own view says sell.

The original bot closed positions on stop-loss / take-profit / time limits.
For markets where the edge is "this is mispriced until it resolves", that
turned a prediction strategy into churn: enter, exit on a small wiggle,
re-enter. Niche positions now:

* settle at $1 / $0 when Kalshi resolves the market (no fee);
* otherwise are held, EXCEPT when selling now beats holding by a clear
  margin: the bid for our side exceeds what the bot currently thinks the
  side is worth by more than HOLD_EXIT_MARGIN (default 5c, enough to cover
  fees and noise). That covers both "the market overshot our estimate" and
  "new information (e.g. more reviews) turned our estimate against us".

The "worth" is the bot's latest recorded estimate (from the learning log),
blended with the market price using the learned trust weight.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from src.utils.market_prices import get_market_prices, get_mid_prices


def exit_margin() -> float:
    try:
        from src import runtime_config
        return max(0.0, float(runtime_config.get("HOLD_EXIT_MARGIN")))
    except ValueError:
        return 0.05


def resolution(position, market: Dict[str, Any]) -> Optional[float]:
    """Settlement price for the position if Kalshi has resolved the market."""
    result = (market.get("result") or "").strip().lower()
    if result in ("yes", "no"):
        return 1.0 if result == position.side.lower() else 0.0
    return None


def niche_exit_decision(position, market: Dict[str, Any], niche: str) -> Tuple[bool, str, float]:
    """(should_exit, reason, per-contract price) for an open niche position."""
    settle = resolution(position, market)
    if settle is not None:
        return True, "market_resolution", settle

    from src import learning  # local import: keeps this module light for tests

    pred = learning.latest_prediction(position.market_id)
    yes_mid, _ = get_mid_prices(market)
    if not pred or yes_mid <= 0:
        return False, "hold (no fresh estimate)", 0.0
    fair_yes = learning.apply(niche, pred["our_prob"], yes_mid)
    fair = fair_yes if position.side.upper() == "YES" else 1 - fair_yes

    yes_bid, _, no_bid, _ = get_market_prices(market)
    bid = yes_bid if position.side.upper() == "YES" else no_bid
    if bid > 0 and bid > fair + exit_margin():
        return True, f"value_exit (bid {bid:.2f} > fair {fair:.2f})", bid
    return False, f"hold (fair {fair:.2f}, bid {bid:.2f})", bid
