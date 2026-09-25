"""
Jev (TypeSafe's calibrated decisions model) as an independent forecaster for the verify gate.

Jev is served on OpenRouter as ``~typesafe/jev-latest`` through the alpha Decisions
endpoint (NOT chat/completions): you send a ``state`` of facts plus typed
``questions`` and get calibrated probabilities back. It does no research of its
own — it reasons over exactly the facts it is handed.

How it plugs into ``verify``: the researcher/skeptic (operator file or LLM) supply
the facts; Jev independently turns those facts into P(YES); the skeptic's
``adjusted_true_yes_pct`` is floored at Jev's number. Jev can only make the gate
STRICTER, never looser. Jev never sees the market price or the operator's
probability estimates — both would anchor it (observed: passing the price pulls
its answer toward the book).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.agent.verify import RESEARCH_SCHEMA, SKEPTIC_SCHEMA

JEV_MODEL = "~typesafe/jev-latest"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

LlmCall = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def build_jev_request(
    question: str,
    research: Dict[str, Any],
    skeptic: Dict[str, Any],
    facts: Optional[List[str]] = None,
    today: Optional[str] = None,
) -> Dict[str, Any]:
    """Decisions-API request asking P(YES) from researched facts. PURE.

    Only qualitative evidence goes in the state — no prices, no probabilities.
    """
    state: Dict[str, Any] = {
        "today": today or date.today().isoformat(),
        "market_question": question,
        "resolution_criteria": research.get("resolution_criteria", ""),
        "catalyst": research.get("catalyst", ""),
        "research_verdict": research.get("verdict", ""),
        "strongest_yes_path": skeptic.get("strongest_yes_path", ""),
    }
    if facts:
        state["facts"] = list(facts)
    return {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            "resolves_yes": {
                "type": "noul",
                "instructions": (
                    "Given only these facts, will this prediction market resolve YES "
                    "under its exact resolution criteria?"
                ),
                "criteria": {
                    "true": "The resolution criteria are met by the deadline",
                    "false": "They are not met",
                },
            }
        },
    }


def parse_jev_yes_pct(resp: Dict[str, Any]) -> float:
    """Extract P(YES) as a percent 0-100 from a Decisions response. PURE.

    Raises ValueError on a malformed/error response — a missing forecast must
    fail loudly, never silently let the gate through.
    """
    try:
        p = float(resp["answers"]["resolves_yes"]["noul"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed Jev response: {str(resp)[:300]}") from e
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Jev probability out of range: {p}")
    return p * 100.0


async def jev_yes_pct(request: Dict[str, Any], api_key: Optional[str] = None) -> float:
    """POST one Decisions request and return Jev's P(YES) percent."""
    import httpx

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — Jev unavailable")
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            DECISIONS_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=request,
        )
    try:
        body = r.json()
    except ValueError:
        body = {"error": r.text[:300]}
    if r.status_code != 200:
        raise RuntimeError(f"Jev HTTP {r.status_code}: {str(body)[:300]}")
    return parse_jev_yes_pct(body)


def with_jev_floor(
    llm_call: LlmCall,
    question: str,
    forecast: Callable[[Dict[str, Any]], Awaitable[float]] = jev_yes_pct,
    facts: Optional[List[str]] = None,
    record: Optional[Dict[str, Any]] = None,
) -> LlmCall:
    """Wrap an ``llm_call`` so the skeptic's true-YES is floored at Jev's forecast.

    ``forecast`` takes a Decisions request and returns a percent (injected for
    tests). ``record`` (optional dict) receives ``jev_true_yes_pct`` for display.
    """
    captured: Dict[str, Any] = {}

    async def wrapped(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        out = await llm_call(prompt, schema)
        if schema is RESEARCH_SCHEMA:
            captured["research"] = out
            return out
        if schema is SKEPTIC_SCHEMA:
            req = build_jev_request(question, captured.get("research", {}), out, facts)
            jev_pct = await forecast(req)
            if record is not None:
                record["jev_true_yes_pct"] = jev_pct
            floored = max(float(out["adjusted_true_yes_pct"]), jev_pct)
            return {**out, "adjusted_true_yes_pct": floored}
        return out

    return wrapped
