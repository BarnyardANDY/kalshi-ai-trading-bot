"""
Big scheduled market-moving events, fixed in advance (no hindsight): US elections, the monthly jobs
report, CPI, and Fed rate decisions. Used by the paper arm 'fav_cal', which sits out any bet that is
still open when one of these happens.

Sources (checked Nov 5, 2026): BLS 2026 release schedule (bls.gov/schedule/2026), Federal Reserve FOMC
calendar (2026 and the tentative 2027 dates). Times are US Eastern. Add new dates here as they are
published; a bet is only judged against dates that were in this list when it was placed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

TZ = "America/New_York"

# (Eastern date-time, name, how many hours the event "lasts" -- election results come in over a night)
EVENTS: List[Tuple[str, str, float]] = [
    ("2026-10-02 08:30", "Jobs report (Sep)", 0),
    ("2026-10-14 08:30", "CPI (Sep)", 0),
    ("2026-10-28 14:00", "Fed rate decision", 0),
    ("2026-11-03 06:00", "US midterm elections", 30),      # polls open through results the next morning
    ("2026-11-06 08:30", "Jobs report (Oct)", 0),
    ("2026-11-10 08:30", "CPI (Oct)", 0),
    ("2026-12-04 08:30", "Jobs report (Nov)", 0),
    ("2026-12-09 14:00", "Fed rate decision", 0),
    ("2026-12-10 08:30", "CPI (Nov)", 0),
    ("2027-01-27 14:00", "Fed rate decision", 0),
]
AFTER_CLOSE_HOURS = 12.0      # many markets settle on next-morning data, so stay out a bit past the close


def _ts(s: str) -> float:
    return pd.Timestamp(s, tz=TZ).timestamp()


_SPANS = [(_ts(t), _ts(t) + dur * 3600, name) for t, name, dur in EVENTS]


def event_during(start_ts: float, close_ts: Optional[float]) -> Optional[str]:
    """Name of a big event that happens while a bet bought at ``start_ts`` is still open
    (until ``close_ts`` + AFTER_CLOSE_HOURS), or None."""
    if close_ts is None:
        return None
    end = close_ts + AFTER_CLOSE_HOURS * 3600
    for a, b, name in _SPANS:
        if a <= end and b >= start_ts:
            return name
    return None
