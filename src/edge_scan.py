"""
"Is there ANY group that makes money?" -- an honest search across every split.

Given one row per bet (``value_col`` = cents per contract) and the columns you
could have known BEFORE betting, ``scan()`` tries every single column and every
pair of columns, and returns the best-looking groups.

Searching hundreds of groups guarantees a few will look good by luck. So each
group gets two tests:

* ``95% sure`` -- the usual single-test bar (mean minus 95% margin above zero).
* ``survives the search`` -- the same bar raised for the number of groups tried
  (Bonferroni). Only a group that passes this is worth real attention.

Margins are cluster-robust: bets on the same event move together, so they count
roughly once, not once per bet.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _cluster_stats(v: pd.Series, clusters: pd.Series) -> Tuple[float, float, int]:
    """(mean, standard error, number of clusters), with errors summed within each cluster."""
    n = len(v)
    mean = float(v.mean())
    k = int(clusters.nunique())
    if n < 2 or k < 2:
        return mean, float("inf"), k
    resid_sum = (v - mean).groupby(clusters.values).sum()
    var = float((resid_sum ** 2).sum()) / (n * n) * (k / (k - 1))
    return mean, math.sqrt(var) if var > 0 else float("inf"), k


def scan(df: pd.DataFrame, dims: Dict[str, str], value_col: str, cluster_col: str,
         min_clusters: int = 30, pairs: bool = True, top: int = 12,
         solo: Tuple[str, ...] = ("series",)) -> Dict[str, object]:
    """Best groups across every split (and every pair of splits) of ``dims``. Columns in ``solo``
    (e.g. hundreds of market types) are tried on their own but never paired: paired, they make
    thousands of tiny groups that can never pass and only slow things down.

    Returns {"tests": number of groups tried, "best": DataFrame, "passing": count that survive the search}.
    """
    empty = {"tests": 0, "best": pd.DataFrame(), "passing": 0, "positive_95": 0}
    if df is None or df.empty or value_col not in df or cluster_col not in df:
        return empty
    usable = [(lab, col) for lab, col in dims.items() if col in df]
    keep = list(dict.fromkeys([col for _, col in usable] + [value_col, cluster_col]))
    d = df.loc[df[value_col].notna(), keep]                 # only the columns needed: keeps memory low
    if d.empty:
        return empty
    d = d.astype({value_col: float, **{col: "category" for _, col in usable}})
    d[cluster_col] = d[cluster_col].fillna("?").astype(str).astype("category")
    combos: List[Tuple[List[str], List[str]]] = [([lab], [col]) for lab, col in usable]
    if pairs:
        combos += [([l1, l2], [c1, c2]) for (l1, c1), (l2, c2) in itertools.combinations(usable, 2)
                   if c1 not in solo and c2 not in solo]
    d = d.rename(columns={value_col: "_v", cluster_col: "_c"})
    frames = []
    for labels, cols in combos:
        # All groups at once: first sum within each (group, event), then combine per group.
        # (Looping over groups one by one took minutes on a small server.)
        pc = d.groupby(cols + ["_c"], observed=True, dropna=False)["_v"].agg(["sum", "count"]).reset_index()
        g = pc.groupby(cols, observed=True, dropna=False)
        tot = g.agg(n=("count", "sum"), s=("sum", "sum"), k=("_c", "size"))
        tot = tot[tot["k"] >= max(2, min_clusters)]
        if tot.empty:
            continue
        pc = pc.join(tot[["n", "s"]], on=cols, how="inner")
        pc["r2"] = (pc["sum"] - pc["count"] * pc["s"] / pc["n"]) ** 2     # each event's summed error, squared
        tot["ss"] = pc.groupby(cols, observed=True, dropna=False)["r2"].sum()
        tot["mean"] = tot["s"] / tot["n"]
        tot["se"] = (tot["ss"] / tot["n"] ** 2 * tot["k"] / (tot["k"] - 1)) ** 0.5
        tot = tot[tot["se"] > 0].reset_index()
        keys = tot[cols].astype(str)
        bad = keys.isin(["unknown", "nan", "None"]).any(axis=1)
        tot, keys = tot[~bad], keys[~bad]
        if tot.empty:
            continue
        frames.append(pd.DataFrame({
            "split": " × ".join(labels),
            "group": keys.agg(" × ".join, axis=1).values,
            "bets": tot["n"].astype(int).values, "events": tot["k"].astype(int).values,
            "c_per_contract": tot["mean"].values, "margin_95": 1.96 * tot["se"].values,
            "z": (tot["mean"] / tot["se"]).values}))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if rows.empty:
        return empty
    t = rows
    m = len(t)
    z_bar = NormalDist().inv_cdf(1 - 0.025 / m)          # Bonferroni: one-sided 2.5% shared across all tries
    t["95% sure"] = t["z"] > 1.96
    t["survives the search"] = t["z"] > z_bar
    best = t.sort_values("z", ascending=False).head(top).reset_index(drop=True)
    return {"tests": m, "z_needed": z_bar, "best": best, "passing": int(t["survives the search"].sum()),
            "positive_95": int(t["95% sure"].sum()), "expected_by_luck": round(0.025 * m, 1)}


def to_jsonable(res: Dict[str, object]) -> Dict[str, object]:
    out = dict(res)
    b = out.get("best")
    out["best"] = b.to_dict("records") if isinstance(b, pd.DataFrame) else []
    return out


def from_jsonable(res: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not res:
        return {"tests": 0, "best": pd.DataFrame(), "passing": 0}
    out = dict(res)
    out["best"] = pd.DataFrame(out.get("best") or [])
    return out
