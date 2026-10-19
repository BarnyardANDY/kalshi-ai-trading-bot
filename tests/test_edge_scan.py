import numpy as np
import pandas as pd

from src import edge_scan


def _df(seed=0, n_events=400, per=3, edge_group=None, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        a = ["x", "y"][e % 2]
        b = ["p", "q", "r"][e % 3]
        shock = rng.normal(0, 30)                                    # bets on one event move together
        for _ in range(per):
            v = shock + rng.normal(0, 10) + (edge if (a, b) == edge_group else 0.0)
            rows.append({"ev": f"E{e}", "a": a, "b": b, "v": v})
    return pd.DataFrame(rows)


def test_no_edge_means_nothing_survives():
    r = edge_scan.scan(_df(), {"A": "a", "B": "b"}, "v", "ev", min_clusters=30)
    assert r["tests"] == 2 + 3 + 6
    assert r["passing"] == 0


def test_a_real_pocket_survives_the_search():
    r = edge_scan.scan(_df(edge_group=("y", "q"), edge=40.0, n_events=1200), {"A": "a", "B": "b"}, "v", "ev",
                       min_clusters=30)
    assert r["passing"] >= 1
    assert r["best"].iloc[0]["group"] == "y × q"


def test_small_groups_and_unknowns_are_skipped():
    d = _df(n_events=40)
    d.loc[d["a"] == "x", "a"] = "unknown"
    r = edge_scan.scan(d, {"A": "a"}, "v", "ev", min_clusters=30, pairs=False)
    assert r["tests"] == 0


def test_json_round_trip():
    r = edge_scan.scan(_df(), {"A": "a"}, "v", "ev", pairs=False)
    back = edge_scan.from_jsonable(edge_scan.to_jsonable(r))
    assert isinstance(back["best"], pd.DataFrame) and len(back["best"]) == len(r["best"])
