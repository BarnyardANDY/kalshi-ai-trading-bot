"""
Slice-and-dice for the maker test: which kinds of bets made money and which didn't.

``features()`` turns stored orders into one row per order with readable labels
(time of day, category, price range, how it filled, what the price did right
after, ...). ``slice_table()`` groups settled bets by any of those labels and
says, honestly, whether each group's result is clearly positive, clearly
negative, or still too noisy to tell.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

TZ = os.getenv("DASHBOARD_TZ", "America/New_York")

# label -> column, in the order shown in the dashboard
DIMENSIONS: Dict[str, str] = {
    "Category (sports, politics, ...)": "category",
    "Series (market type)": "series",
    "Price paid": "price_bucket",
    "Time of day posted (Eastern)": "daypart",
    "Hour posted (Eastern)": "hour_et",
    "Day of week": "weekday",
    "Gap between bid and ask": "spread_bucket",
    "Place in line when posted": "queue_bucket",
    "How it filled": "fill_kind",
    "Wait from post to fill": "wait_bucket",
    "Fill to result (live vs. ahead of time)": "fill_to_result",
    "Near the expected end when filled": "near_end",
    "Price 5 min after fill": "markout_5m",
    "Price 30 min after fill": "markout_30m",
    "Price trend before we posted (vs. yesterday)": "trend_24h",
    "Market activity (24h volume)": "volume_bucket",
    "Side bought": "side",
    "Bid 1c better than best?": "improved_label",
}


def _cut(v: Optional[float], edges: List[float], labels: List[str]) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def _fill_kind(row) -> str:
    k = row.get("fill_kind")
    if isinstance(k, str) and k:
        return {"through": "price dropped through our bid", "queue": "waited in line at our price"}.get(k, k)
    note = row.get("note") or ""
    if note.startswith("sold through"):
        return "price dropped through our bid"
    if re.match(r"\d+ sold at our", note):
        return "waited in line at our price"
    return "unknown"


def _markout(mid: Optional[float], price: float) -> str:
    if mid is None or (isinstance(mid, float) and math.isnan(mid)):
        return "unknown"
    d = round((mid - price) * 100, 1)
    if d <= -5:
        return "fell 5c+"
    if d <= -2:
        return "fell 2-4c"
    if d < 2:
        return "about flat"
    return "rose 2c+"


def features(rows: List[Dict[str, Any]], meta: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    meta = meta or {}
    for col in ("fill_ts", "settled_ts", "expected_exp_ts", "mid_5m", "mid_30m", "side_mid_post",
                "side_prev_day", "volume_24h", "fill_kind", "note", "pnl", "result"):
        if col not in df:
            df[col] = None
    posted = pd.to_datetime(df["posted_ts"], unit="s", utc=True).dt.tz_convert(TZ)
    df["posted_et"] = posted.dt.tz_localize(None)
    df["hour_et"] = posted.dt.hour.map(lambda h: f"{h:02d}:00")
    df["weekday"] = posted.dt.day_name()
    df["daypart"] = posted.dt.hour.map(lambda h: "night (0-6)" if h < 6 else "morning (6-12)" if h < 12
                                       else "afternoon (12-18)" if h < 18 else "evening (18-24)")
    df["category"] = df["series"].map(lambda s: (meta.get(s) or {}).get("category") or "unknown")
    df["price_bucket"] = df["price"].map(lambda p: _cut(p, [0.65, 0.75, 0.85], ["55-64c", "65-74c", "75-84c", "85-95c"]))
    df["spread_bucket"] = df["spread"].map(lambda v: _cut(round(v * 100) if v is not None else None, [2, 3, 6, 11],
                                                          ["1c", "2c", "3-5c", "6-10c", "11c+"]))
    df["queue_bucket"] = df["queue_ahead"].map(lambda q: "first in line" if q == 0 else
                                               _cut(q, [100, 1000], ["1-99 ahead", "100-999 ahead", "1000+ ahead"]))
    df["fill_kind"] = df.apply(lambda r: _fill_kind(r) if pd.notna(r.get("fill_ts")) else "not filled", axis=1)
    wait = (df["fill_ts"].astype(float) - df["posted_ts"]) / 60
    df["wait_bucket"] = wait.map(lambda m: _cut(m, [5, 15, 60], ["<5 min", "5-15 min", "15-60 min", "60+ min"]))
    ftr = (df["settled_ts"].astype(float) - df["fill_ts"].astype(float)) / 3600
    df["fill_to_result"] = ftr.map(lambda h: _cut(h, [1, 3, 12], ["<1h (live)", "1-3h", "3-12h", "12h+ (ahead of time)"]))
    near = (df["expected_exp_ts"].astype(float) - df["fill_ts"].astype(float)) / 3600
    df["near_end"] = near.map(lambda h: _cut(h, [3], ["within 3h of expected end", "earlier"]))
    df["markout_5m"] = [_markout(m, p) for m, p in zip(df["mid_5m"], df["price"])]
    df["markout_30m"] = [_markout(m, p) for m, p in zip(df["mid_30m"], df["price"])]
    trend = (df["side_mid_post"].astype(float) - df["side_prev_day"].astype(float)) * 100
    df["trend_24h"] = trend.map(lambda d: _cut(d, [-3, 3], ["falling 3c+", "about flat", "rising 3c+"]))
    df["volume_bucket"] = df["volume_24h"].astype(float).map(
        lambda v: _cut(v, [500, 5000, 50000], ["100-499", "500-4,999", "5,000-49,999", "50,000+"]))
    df["improved_label"] = df["improved"].map(lambda v: "yes (first in line)" if v else "no (joined the line)")
    df["c_per_contract"] = 100 * df["pnl"].astype(float) / df["qty"]
    df["won"] = df["pnl"].astype(float) > 0
    return df


def slice_table(df: pd.DataFrame, col: str, min_bets: int = 1) -> pd.DataFrame:
    """Settled bets grouped by ``col``, with a plain-language verdict per group."""
    if df is None or df.empty or col not in df:
        return pd.DataFrame()
    s = df[df["status"] == "settled"]
    if s.empty:
        return pd.DataFrame()
    out = []
    for key, g in s.groupby(col, dropna=False):
        n = len(g)
        if n < min_bets:
            continue
        c = g["c_per_contract"].astype(float)
        mean = c.mean()
        # 95% margin, counting each event once (bets on the same event move together)
        n_eff = max(1, g["event"].nunique())
        margin = 1.96 * (c.std(ddof=1) if n > 1 else 50.0) / math.sqrt(n_eff)
        verdict = ("clearly positive" if mean - margin > 0 else "clearly negative" if mean + margin < 0
                   else "too early to tell")
        out.append({"group": key, "bets": n, "events": n_eff, "win rate": g["won"].mean(),
                    "avg price": g["price"].mean(), "c_per_contract": mean, "margin": margin,
                    "pnl": g["pnl"].astype(float).sum(), "verdict": verdict})
    t = pd.DataFrame(out)
    return t.sort_values("c_per_contract", ascending=False).reset_index(drop=True) if not t.empty else t


def export_csv(df: pd.DataFrame, path: str) -> str:
    keep = [c for c in df.columns if not c.endswith("_ts") or c in ("posted_ts",)]
    df[keep].to_csv(path, index=False)
    return path
