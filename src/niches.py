"""
Niche focus: trade only a chosen slice of Kalshi instead of every market.

Set ``NICHES`` in ``.env`` to a comma-separated list of niche names, e.g.

    NICHES=rotten_tomatoes,trump_mentions

When set, market ingestion fetches only these niches' series (fast, and it
never misses them the way a capped crawl of all events can), and the trading
loop only analyzes markets that belong to them. Leave ``NICHES`` empty to keep
the original trade-everything behavior.

Each niche also gets a research step (see ``src/niche_research.py``) that
feeds fresh, niche-specific data to the AI before it estimates a probability.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Niche:
    name: str
    label: str
    # Series whose open markets are always included.
    series: Tuple[str, ...]
    # Series-ticker prefixes that identify this niche (for membership checks
    # and for filtering dynamically discovered series).
    prefixes: Tuple[str, ...]
    # Kalshi /series filters used to discover extra series (e.g. one-off
    # event-specific series). Each entry is a dict of query params.
    discovery: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    # Lowercase words that must ALL appear in a discovered series' title or
    # ticker for it to count (guards broad discovery like category=Mentions).
    discovery_must_contain: Tuple[str, ...] = field(default_factory=tuple)


NICHES: Dict[str, Niche] = {
    "rotten_tomatoes": Niche(
        name="rotten_tomatoes",
        label="Rotten Tomatoes scores",
        # All current RT markets live in the single KXRT series; the ~130
        # older per-film series (KXRTSMURFS, RTWICKED, ...) are closed, and
        # polling each one every cycle made ingestion slow.
        series=("KXRT",),
        prefixes=("KXRT",),
    ),
    "trump_mentions": Niche(
        name="trump_mentions",
        label="Trump mention markets",
        series=("KXTRUMPMENTION", "KXTRUMPSAY", "KXTRUMPTOPIC", "KXTRUMPMENTIONB"),
        prefixes=("KXTRUMPMENTION", "KXTRUMPSAY", "KXTRUMPTOPIC", "KXDJT"),
        discovery=({"category": "Mentions"},),
        discovery_must_contain=("trump",),
    ),
}


def enabled_niches() -> List[Niche]:
    """Niches listed in the NICHES env var (unknown names are ignored)."""
    raw = os.getenv("NICHES", "")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    return [NICHES[n] for n in names if n in NICHES]


def niche_min_volume() -> int:
    """Minimum contracts traded for a niche market to be analyzed.

    Niche markets are thinner than headline markets, so the global 200-contract
    floor would hide most of them.
    """
    try:
        return int(os.getenv("NICHE_MIN_VOLUME", "20"))
    except ValueError:
        return 20


def series_of(ticker: str) -> str:
    """Series ticker of a market/event ticker (text before the first '-')."""
    return (ticker or "").split("-", 1)[0].upper()


def niche_for_ticker(ticker: str, niches: Optional[Iterable[Niche]] = None) -> Optional[Niche]:
    """Return the niche a market ticker belongs to, if any."""
    series = series_of(ticker)
    if not series:
        return None
    pool = list(niches) if niches is not None else list(NICHES.values())
    for niche in pool:
        discovered = _SERIES_CACHE.get(niche.name, (0, []))[1]
        if series in niche.series or series in discovered:
            return niche
        if any(series.startswith(p) for p in niche.prefixes):
            return niche
    return None


def market_display_title(market: Dict[str, Any]) -> str:
    """Title that names the specific outcome, not just the shared question.

    Multi-outcome events share one title ("Verity Rotten Tomatoes score?")
    across many markets; the outcome lives in ``yes_sub_title`` ("Above 90").
    Without it the AI cannot tell the markets apart.
    """
    title = (market.get("title") or "").strip()
    sub = (market.get("yes_sub_title") or market.get("subtitle") or "").strip()
    if sub and sub.lower() not in title.lower():
        return f"{title} — {sub}" if title else sub
    return title


# ----------------------------------------------------------------------------
# Discovery (network)
# ----------------------------------------------------------------------------

_SERIES_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_SERIES_CACHE_TTL = 6 * 3600


async def _discover_series(kalshi_client, niche: Niche, logger=None) -> List[str]:
    """Static series plus any matching series found via Kalshi's /series search."""
    cached = _SERIES_CACHE.get(niche.name)
    if cached and time.time() - cached[0] < _SERIES_CACHE_TTL:
        return cached[1]

    found = list(niche.series)
    for params in niche.discovery:
        try:
            resp = await kalshi_client._make_authenticated_request(
                "GET", "/trade-api/v2/series", params=params
            )
        except Exception as e:  # discovery is best-effort
            if logger:
                logger.warning(f"Series discovery failed for {niche.name} {params}: {e}")
            continue
        for s in resp.get("series", []) or []:
            ticker = (s.get("ticker") or "").upper()
            if not ticker or ticker in found:
                continue
            text = f"{ticker} {s.get('title', '')}".lower()
            if niche.discovery_must_contain and not all(w in text for w in niche.discovery_must_contain):
                continue
            if niche.discovery_must_contain or any(ticker.startswith(p) for p in niche.prefixes) or params.get("tags"):
                found.append(ticker)

    _SERIES_CACHE[niche.name] = (time.time(), found)
    return found


async def fetch_niche_markets(kalshi_client, niches: List[Niche], logger=None) -> List[Dict[str, Any]]:
    """All open markets in the given niches, each tagged with ``_niche``."""
    markets: List[Dict[str, Any]] = []
    seen = set()
    for niche in niches:
        series_list = await _discover_series(kalshi_client, niche, logger)
        for series in series_list:
            cursor = None
            for _ in range(20):  # page cap per series
                try:
                    resp = await kalshi_client.get_markets(
                        limit=200, cursor=cursor, series_ticker=series, status="open"
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to fetch markets for series {series}: {e}")
                    break
                page = resp.get("markets", []) or []
                for m in page:
                    t = m.get("ticker")
                    if not t or t in seen:
                        continue
                    if m.get("status") not in ("active", "open"):
                        continue
                    seen.add(t)
                    m["_niche"] = niche.name
                    markets.append(m)
                cursor = resp.get("cursor")
                # Kalshi can hand back a cursor on the last page; stop on a short page.
                if not cursor or len(page) < 200:
                    break
        if logger:
            count = sum(1 for m in markets if m.get("_niche") == niche.name)
            logger.info(f"Niche {niche.name}: {count} open markets across {len(series_list)} series")
    return markets
