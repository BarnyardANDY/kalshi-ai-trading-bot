"""Jev forecaster: request shape, response parsing, and the stricter-only floor."""
import asyncio

import pytest

from src.agent.jev import build_jev_request, parse_jev_yes_pct, with_jev_floor
from src.agent.verify import aggregate_verdict, make_operator_llm, run_verify

RESEARCH = {
    "resolution_criteria": "YES if X happens by Oct 1",
    "true_yes_pct": 2,
    "catalyst": "none",
    "direction": "none",
    "is_frontrunner": False,
    "verdict": "structural longshot",
}
SKEPTIC = {
    "strongest_yes_path": "surprise announcement",
    "adjusted_true_yes_pct": 3,
    "survives": True,
    "recommend": "BUY_NO",
    "size_hint": "half",
}


def test_request_carries_facts_but_no_prices_or_probabilities():
    req = build_jev_request("Will X?", RESEARCH, SKEPTIC, facts=["fact A"], today="2026-09-25")
    assert req["model"] == "~typesafe/jev-latest"
    assert req["questions"]["resolves_yes"]["type"] == "noul"
    state = req["state"]
    assert state["facts"] == ["fact A"] and state["today"] == "2026-09-25"
    assert state["resolution_criteria"] == RESEARCH["resolution_criteria"]
    flat = str(state)
    assert "true_yes" not in flat and "no_ask" not in flat and "adjusted" not in flat


def test_parse_yes_pct_and_malformed_fails_loudly():
    assert parse_jev_yes_pct({"answers": {"resolves_yes": {"noul": 0.07}}}) == pytest.approx(7.0)
    with pytest.raises(ValueError):
        parse_jev_yes_pct({"error": {"message": "User not found."}})
    with pytest.raises(ValueError):
        parse_jev_yes_pct({"answers": {"resolves_yes": {"noul": 1.5}}})


def _run(jev_pct, no_ask=0.90):
    record = {}

    async def forecast(_req):
        return jev_pct

    llm = with_jev_floor(
        make_operator_llm({"research": RESEARCH, "skeptic": SKEPTIC}), "Will X?",
        forecast=forecast, record=record,
    )
    verdict = asyncio.run(run_verify({"ticker": "T", "question": "Will X?", "no_ask": no_ask}, llm))
    return verdict, record


def test_jev_higher_than_skeptic_tightens_gate_to_pass():
    verdict, record = _run(jev_pct=8.0)  # true-YES 8% vs implied 10% -> edge +2 < 5
    assert record["jev_true_yes_pct"] == 8.0
    assert verdict["true_yes"] == 8.0 and verdict["recommend"] == "PASS"


def test_jev_lower_than_skeptic_never_loosens():
    verdict, _ = _run(jev_pct=0.5)
    base = aggregate_verdict({"ticker": "T", "no_ask": 0.90}, RESEARCH, SKEPTIC)
    assert verdict["true_yes"] == 3.0 and verdict["edge_pts"] == base["edge_pts"]
    assert verdict["recommend"] == "BUY_NO"


def test_classify_request_one_noul_question_per_text():
    from src.agent.jev import build_classify_request
    req = build_classify_request("Trump says 'tariff'", ["big tariffs", "trade deals"], {"window": "Sep"})
    assert req["state"] == {"market_rule": "Trump says 'tariff'", "context": {"window": "Sep"}}
    assert list(req["questions"]) == ["q0", "q1"]
    assert all(q["type"] == "noul" for q in req["questions"].values())
    assert "big tariffs" in req["questions"]["q0"]["instructions"]


def test_parse_classify_orders_by_question_and_fails_loudly():
    from src.agent.jev import parse_classify
    resp = {"answers": {"q0": {"noul": 0.91}, "q1": {"noul": 0.03}}}
    assert parse_classify(resp, 2) == [0.91, 0.03]
    with pytest.raises(ValueError):
        parse_classify({"answers": {"q0": {"noul": 0.9}}}, 2)
