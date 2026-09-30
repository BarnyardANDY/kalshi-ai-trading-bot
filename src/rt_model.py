"""
Rotten Tomatoes: price every threshold of a film from ONE coherent view.

Kalshi lists each film as a ladder of markets ("Above 45", "Above 50", ...
"Above 90"). Pricing each rung with its own AI call produced contradictory
bets (e.g. NO on "Above 45" together with YES on "Above 70"). Here we:

1. Group the ladder by film (event).
2. Skip the film until Rotten Tomatoes shows at least RT_MIN_REVIEWS critic
   reviews. Before reviews, there is nothing to know that the market doesn't.
3. Build a statistical baseline from the fresh/rotten counts: sample the
   underlying "fresh rate" from a Beta posterior, simulate the reviews still
   to come, and read off P(final score > threshold) for every rung.
4. Ask the AI once per film to adjust that baseline using the news, then
   force the answers to be consistent (a higher threshold can never be more
   likely than a lower one).

The result is a probability for every rung, which the normal edge, sizing
and risk checks then use as usual.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.niche_research import build_research_context, parse_rt_market, rotten_tomatoes_score
from src.niches import market_display_title

Prediction = Tuple[float, float]  # (probability, confidence)

# event -> (timestamp, liked, not_liked, thresholds, probs, confidence)
_LADDER_CACHE: Dict[str, tuple] = {}


def rt_reprice_minutes() -> float:
    """Re-ask the AI about a film at most this often unless its reviews change."""
    try:
        from src import runtime_config
        return max(0.0, float(runtime_config.get("RT_REPRICE_MINUTES")))
    except ValueError:
        return 60.0


def rt_min_reviews() -> int:
    try:
        from src import runtime_config
        return max(0, int(runtime_config.get("RT_MIN_REVIEWS")))
    except ValueError:
        return 5


def event_of(ticker: str) -> str:
    """KXRT-DIG-45 -> KXRT-DIG."""
    return ticker.rsplit("-", 1)[0] if ticker.count("-") >= 2 else ticker


def threshold_of(ticker: str, title: str = "") -> Optional[int]:
    """Threshold of an RT ladder rung, from the title ('Above 45') or ticker suffix."""
    m = re.search(r"\babove\s+(\d{1,3})\b", title or "", re.IGNORECASE)
    if m:
        return int(m.group(1))
    tail = ticker.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() and 0 <= int(tail) <= 100 else None


def baseline_probs(
    liked: int,
    not_liked: int,
    thresholds: Iterable[int],
    extra_reviews: Optional[int] = None,
    samples: int = 20000,
    seed: int = 7,
    shift: float = 0.0,
) -> Dict[int, float]:
    """P(final Tomatometer > t) for each t, from the current fresh/rotten split.

    The final score is ``round(100 * fresh / total)`` after ``extra_reviews``
    more reviews arrive (default: as many again as exist now, at least 10).
    "Above t" means the displayed integer score is greater than t.
    """
    n = liked + not_liked
    m = extra_reviews if extra_reviews is not None else max(10, n)
    rng = np.random.default_rng(seed)
    p = rng.beta(liked + 1, not_liked + 1, size=samples)
    future_fresh = rng.binomial(m, p) if m > 0 else np.zeros(samples)
    final = np.rint(np.clip(100 * (liked + future_fresh) / max(1, n + m) + shift, 0, 100))
    return {int(t): float(np.mean(final > t)) for t in thresholds}


def make_monotone(probs: Dict[int, float]) -> Dict[int, float]:
    """Force P(>t) to be non-increasing in t and within [0.01, 0.99]."""
    out: Dict[int, float] = {}
    prev = 1.0
    for t in sorted(probs):
        p = min(max(float(probs[t]), 0.01), 0.99)
        p = min(p, prev)
        out[t] = p
        prev = p
    return out


def _parse_ai(text: Optional[str], thresholds: List[int]) -> Tuple[Optional[Dict[int, float]], Optional[float]]:
    if not text:
        return None, None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None, None
    raw = data.get("probabilities") or {}
    probs: Dict[int, float] = {}
    for t in thresholds:
        v = raw.get(str(t), raw.get(t))
        if isinstance(v, (int, float)) and 0 <= v <= 1:
            probs[t] = float(v)
    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None
    if len(probs) < len(thresholds):
        return None, conf
    return probs, conf


async def predict_rt_ladders(markets, xai_client, kalshi_client, logger) -> Dict[str, Prediction]:
    """Coherent (probability, confidence) for every RT ladder market given.

    ``markets`` are ``Market`` rows from the DB (all in the RT niche). Markets
    whose film has too few reviews, or whose RT page can't be confirmed, get
    no prediction and are therefore skipped by the caller.
    """
    ladders: Dict[str, List] = {}
    for mk in markets:
        t = threshold_of(mk.market_id, mk.title)
        if t is not None:
            ladders.setdefault(event_of(mk.market_id), []).append((t, mk))

    min_reviews = rt_min_reviews()
    out: Dict[str, Prediction] = {}
    for event, rungs in ladders.items():
        rungs.sort(key=lambda x: x[0])
        # One market fetch per film for the rules (name, resolution date).
        try:
            sample = (await kalshi_client.get_market(rungs[0][1].market_id)).get("market", {})
        except Exception as e:
            logger.warning(f"RT {event}: could not load market details: {e}")
            continue
        info = parse_rt_market(sample)
        name = info.get("name")
        if not name:
            continue
        year = None
        if info.get("date"):
            y = re.search(r"(\d{4})$", info["date"])
            year = int(y.group(1)) if y else None
        rt = await rotten_tomatoes_score(name, year)
        if not rt or rt.get("year_unverified"):
            logger.info(f"RT {event} ({name}): no confirmed RT page for this year, skipping film")
            continue
        liked, not_liked = rt.get("liked"), rt.get("not_liked")
        reviews = rt.get("reviews") or 0
        if liked is None or not_liked is None:
            # Fall back to score x reviews when the split isn't published.
            if rt.get("score") is not None and reviews:
                liked = round(rt["score"] / 100 * reviews)
                not_liked = reviews - liked
            else:
                liked, not_liked = 0, 0
        if liked + not_liked < min_reviews:
            logger.info(
                f"RT {event} ({name}): {liked + not_liked} reviews < {min_reviews} minimum, waiting for reviews"
            )
            continue

        thresholds = [t for t, _ in rungs]
        cached = _LADDER_CACHE.get(event)
        if (
            cached
            and cached[1] == liked
            and cached[2] == not_liked
            and set(thresholds) <= set(cached[4])
            and time.time() - cached[0] < rt_reprice_minutes() * 60
        ):
            # Same reviews as last time: reuse the view instead of paying for
            # another AI call every cycle.
            for t, mk in rungs:
                out[mk.market_id] = (cached[4][t], cached[5])
            continue
        drift = None
        try:
            from src.learning import niche_params
            drift = niche_params("rotten_tomatoes").get("rt_drift")
        except Exception:
            pass
        base = baseline_probs(liked, not_liked, thresholds, shift=drift or 0.0)
        drift_note = (
            f"\n        LEARNED FROM PAST RESULTS: films in this bot's record ended on average "
            f"{drift:+.1f} points from their score at prediction time; the baseline "
            f"already includes this shift."
            if drift else ""
        )
        prices = {t: mk.yes_price for t, mk in rungs}
        research = await build_research_context("rotten_tomatoes", sample)
        ladder_lines = "\n".join(
            f"  Above {t}: market {prices[t]*100:.0f}c | statistical baseline {base[t]*100:.0f}%"
            for t in thresholds
        )
        prompt = f"""
        ROTTEN TOMATOES LADDER — {name}

        Kalshi lists these markets for {name}'s final Tomatometer
        (YES pays if the displayed score is ABOVE the number at resolution):
{ladder_lines}

        The statistical baseline uses only the current {liked} fresh / {not_liked}
        rotten split and assumes about as many more reviews arrive before
        resolution. It ignores who has/hasn't reviewed yet and the common
        tendency for scores to slip as more (often less enthusiastic) critics
        weigh in.{drift_note}

        FRESH RESEARCH:
        {research}

        Give your probability for EVERY threshold, consistent with a single
        view of the final score (higher thresholds can't be more likely).
        Move away from the baseline only for concrete reasons in the research.
        Reply with JSON only:
        {{"probabilities": {{{", ".join(f'"{t}": 0.0' for t in thresholds)}}},
          "confidence": 0.0,
          "reasoning": "1-2 sentences"}}
        """
        text = await xai_client.get_completion(prompt, max_tokens=3000, temperature=0.1)
        probs, conf = _parse_ai(text, thresholds)
        if probs is None:
            logger.info(f"RT {event} ({name}): AI answer unusable, using statistical baseline")
            probs, conf = base, 0.5
        probs = make_monotone(probs)
        conf = conf if conf is not None else 0.5
        logger.info(
            f"RT {event} ({name}): {liked}F/{not_liked}R -> "
            + ", ".join(f">{t}: {probs[t]:.0%}" for t in thresholds)
            + f" (confidence {conf:.0%})"
        )
        _LADDER_CACHE[event] = (time.time(), liked, not_liked, thresholds, probs, conf)
        for t, mk in rungs:
            out[mk.market_id] = (probs[t], conf)
    return out
