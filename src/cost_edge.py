"""
Edge after real trading costs, and other entry guards.

The bot's probability is compared with the price you'd actually PAY, not the
bid/ask midpoint:

    net edge (YES) = P(yes)      - (yes ask + Kalshi fee per contract)
    net edge (NO)  = 1 - P(yes)  - (no ask  + Kalshi fee per contract)

A "10-point edge" against a 6c-wide book is really ~7 points before fees,
so trades that only looked good against the midpoint are filtered out.

Also here:
* slippage: how far the ask sits above the midpoint the signal was judged
  on (a wide or thin book), and the execution-time check that the ask hasn't
  moved past the price the trade was sized at;
* days to close: skip markets that won't settle for a long time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.utils.market_prices import get_market_prices, get_mid_prices


def _m(market: Dict[str, Any]) -> Dict[str, Any]:
    return market["market"] if isinstance(market.get("market"), dict) else market


def fee_per_contract(price: float) -> float:
    """Kalshi taker fee per contract at ``price`` (before rounding up to the cent per order)."""
    from src.paper_fills import fee_rate

    if not 0 < price < 1:
        return 0.0
    return fee_rate() * price * (1 - price)


def net_edges(prob_yes: float, market: Dict[str, Any], fee_aware: bool = True) -> Dict[str, Any]:
    """Edge for each side against its ask (plus fee when ``fee_aware``)."""
    mk = _m(market)
    _, yes_ask, _, no_ask = get_market_prices(mk)
    yes_mid, no_mid = get_mid_prices(mk)
    out: Dict[str, Any] = {"yes_ask": yes_ask, "no_ask": no_ask, "yes_mid": yes_mid, "no_mid": no_mid}
    for side, p, ask in (("yes", prob_yes, yes_ask), ("no", 1 - prob_yes, no_ask)):
        if 0 < ask < 1:
            cost = ask + (fee_per_contract(ask) if fee_aware else 0.0)
            out[f"{side}_cost"] = cost
            out[f"net_{side}"] = p - cost
        else:
            out[f"{side}_cost"] = None
            out[f"net_{side}"] = None
    candidates = [(s, out[f"net_{s}"]) for s in ("yes", "no") if out[f"net_{s}"] is not None]
    if candidates:
        side, best = max(candidates, key=lambda x: x[1])
        out["best_side"], out["best_net"] = side.upper(), best
    else:
        out["best_side"], out["best_net"] = None, None
    return out


def slippage(market: Dict[str, Any], side: str) -> Optional[float]:
    """How far the ask for ``side`` sits above that side's midpoint (None if no book)."""
    mk = _m(market)
    _, yes_ask, _, no_ask = get_market_prices(mk)
    yes_mid, no_mid = get_mid_prices(mk)
    ask, mid = (yes_ask, yes_mid) if side.upper() == "YES" else (no_ask, no_mid)
    if not (0 < ask < 1) or mid <= 0:
        return None
    return max(0.0, ask - mid)


def days_to_close(market: Dict[str, Any], now: Optional[datetime] = None) -> Optional[float]:
    mk = _m(market)
    ct = mk.get("close_time") or mk.get("expiration_time")
    if not ct:
        return None
    try:
        close = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    return (close - now).total_seconds() / 86400


def side_ask(market: Dict[str, Any], side: str) -> Optional[float]:
    mk = _m(market)
    _, yes_ask, _, no_ask = get_market_prices(mk)
    ask = yes_ask if side.upper() == "YES" else no_ask
    return ask if 0 < ask < 1 else None
