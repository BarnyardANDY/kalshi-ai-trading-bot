"""
Self-improvement loop for the niche strategies.

The bot writes down every probability it forms for a niche market, whether it
trades or not. When Kalshi settles those markets, each prediction is graded
against the outcome and against the market price at the time. Three lessons
are then fed back into the next decisions:

1. **Trust weight** (per niche). The bot's final probability becomes
       trust * its_estimate + (1 - trust) * market_price
   with ``trust`` chosen to minimize log-loss on its own settled record.
   If it has been beating the market, trust rises toward 1; if the market has
   been more accurate, trust falls toward 0 and the bot mostly defers.
2. **Auto-pause** (per niche). Once a niche has enough settled events and the
   record shows no clear advantage over the market price (low learned trust,
   or a blend that barely beats the market), new trades in that niche stop. Predictions keep being recorded and graded
   ("shadow mode"), so trading resumes by itself if the record improves.
3. **Rotten Tomatoes drift**. Comparing each film's score when predicted with
   where it finally landed teaches how much scores tend to move after early
   reviews (often down). The statistical baseline is shifted by that amount.

Honesty rules: nothing is learned from fewer than ``LEARN_MIN_EVENTS`` settled
events (a film or a Trump event counts once, however many rungs it had), and
every learned number is clipped to a sane range.

Stored in ``data/learning.db`` (separate from trading_system.db, so resetting
paper trades never erases what the bot has learned).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("LEARNING_DB", "data/learning.db")
_SETTLE_EVERY_SEC = 30 * 60
_last_settle = 0.0
_params_cache: Dict[str, Any] = {"ts": 0.0, "params": None}


def _min_events() -> int:
    try:
        return max(1, int(os.getenv("LEARN_MIN_EVENTS", "8")))
    except ValueError:
        return 8


def _default_trust() -> float:
    try:
        from src import runtime_config
        return min(1.0, max(0.0, float(runtime_config.get("LEARN_DEFAULT_TRUST"))))
    except ValueError:
        return 0.6


def _conn(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            event TEXT NOT NULL,
            niche TEXT NOT NULL,
            ts REAL NOT NULL,
            close_ts REAL,
            our_prob REAL NOT NULL,
            market_prob REAL NOT NULL,
            confidence REAL,
            threshold INTEGER,
            rt_liked INTEGER,
            rt_not_liked INTEGER,
            outcome INTEGER,
            settled_ts REAL
        )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS ix_pred_market ON predictions(market_id)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_pred_open ON predictions(outcome)")
    return c


def _close_ts(close_time: Optional[str]) -> Optional[float]:
    if not close_time:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------

def record_prediction(
    market_id: str,
    event: str,
    niche: str,
    our_prob: float,
    market_prob: float,
    confidence: Optional[float] = None,
    close_time: Optional[str] = None,
    threshold: Optional[int] = None,
    rt_liked: Optional[int] = None,
    rt_not_liked: Optional[int] = None,
    path: Optional[str] = None,
    now: Optional[float] = None,
) -> bool:
    """Store a prediction. Skips near-duplicates (same market, <1h, <3pt change)."""
    now = now or time.time()
    with closing(_conn(path)) as c:
        row = c.execute(
            "SELECT ts, our_prob FROM predictions WHERE market_id=? ORDER BY ts DESC LIMIT 1",
            (market_id,),
        ).fetchone()
        if row and now - row[0] < 3600 and abs(row[1] - our_prob) < 0.03:
            return False
        c.execute(
            """INSERT INTO predictions (market_id, event, niche, ts, close_ts, our_prob,
               market_prob, confidence, threshold, rt_liked, rt_not_liked)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (market_id, event, niche, now, _close_ts(close_time), float(our_prob),
             float(market_prob), confidence, threshold, rt_liked, rt_not_liked),
        )
        c.commit()
    return True


# ----------------------------------------------------------------------------
# Settling
# ----------------------------------------------------------------------------

async def settle_predictions(kalshi_client, logger=None, path: Optional[str] = None,
                             force: bool = False, max_markets: int = 150) -> int:
    """Fill in outcomes for predictions whose markets have settled on Kalshi.

    Throttled to once per 30 minutes unless ``force``. Returns markets settled.
    """
    global _last_settle
    if not force and time.time() - _last_settle < _SETTLE_EVERY_SEC:
        return 0
    _last_settle = time.time()
    now = time.time()
    with closing(_conn(path)) as c:
        ids = [r[0] for r in c.execute(
            """SELECT DISTINCT market_id FROM predictions
               WHERE outcome IS NULL AND (close_ts IS NULL OR close_ts < ?)
               LIMIT ?""", (now, max_markets))]
    settled = 0
    for mid in ids:
        try:
            m = (await kalshi_client.get_market(mid)).get("market", {})
        except Exception:
            continue
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue
        with closing(_conn(path)) as c:
            c.execute("UPDATE predictions SET outcome=?, settled_ts=? WHERE market_id=? AND outcome IS NULL",
                      (1 if result == "yes" else 0, now, mid))
            c.commit()
        settled += 1
    if settled:
        _params_cache["ts"] = 0.0  # force re-learning
        if logger:
            logger.info(f"🧠 Learning: {settled} predicted markets settled; re-learning")
    return settled


# ----------------------------------------------------------------------------
# Learning (pure)
# ----------------------------------------------------------------------------

def _ll(p: float, y: int) -> float:
    p = min(max(p, 0.01), 0.99)
    return -math.log(p if y else 1 - p)


def _latest_per_market(rows: List[tuple]) -> List[tuple]:
    """Grade the LAST prediction made before close for each market."""
    best: Dict[str, tuple] = {}
    for r in rows:
        if r[0] not in best or r[3] > best[r[0]][3]:
            best[r[0]] = r
    return list(best.values())


def learn(rows: List[tuple], min_events: Optional[int] = None,
          default_trust: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """Learned parameters per niche from settled rows.

    Row layout: (market_id, event, niche, ts, our_prob, market_prob, outcome,
                 threshold, rt_liked, rt_not_liked)
    """
    min_events = min_events or _min_events()
    default_trust = _default_trust() if default_trust is None else default_trust
    by_niche: Dict[str, List[tuple]] = defaultdict(list)
    for r in _latest_per_market(rows):
        by_niche[r[2]].append(r)

    params: Dict[str, Dict[str, Any]] = {}
    for niche, rs in by_niche.items():
        # Weight each market by 1/(markets in its event) so a 10-rung film
        # counts as one piece of evidence, not ten.
        per_event = defaultdict(int)
        for r in rs:
            per_event[r[1]] += 1
        w = [1.0 / per_event[r[1]] for r in rs]
        n_events = len(per_event)
        tw = sum(w)

        def loss(trust: float) -> float:
            return sum(wi * _ll(trust * r[4] + (1 - trust) * r[5], r[6]) for wi, r in zip(w, rs)) / tw

        market_ll = loss(0.0)
        raw_ll = loss(1.0)
        brier_ours = sum(wi * (r[4] - r[6]) ** 2 for wi, r in zip(w, rs)) / tw
        brier_mkt = sum(wi * (r[5] - r[6]) ** 2 for wi, r in zip(w, rs)) / tw
        p: Dict[str, Any] = {
            "events": n_events,
            "markets": len(rs),
            "logloss_ours": round(raw_ll, 4),
            "logloss_market": round(market_ll, 4),
            "brier_ours": round(brier_ours, 4),
            "brier_market": round(brier_mkt, 4),
            "trust": default_trust,
            "paused": False,
            "learned": False,
        }
        if n_events >= min_events:
            grid = [i / 10 for i in range(11)]
            best = min(grid, key=loss)
            p["trust"] = best
            p["learned"] = True
            p["logloss_blended"] = round(loss(best), 4)
            # Pause unless the record shows a real advantage: the bot must earn
            # meaningful trust AND the blend must beat the market by a margin
            # (a sliver of improvement is noise, not edge).
            p["paused"] = best <= 0.2 or (market_ll - loss(best)) < 0.005

        if niche == "rotten_tomatoes":
            p["rt_drift"] = _rt_drift(rs, min_events)
        params[niche] = p
    return params


def _rt_drift(rs: List[tuple], min_events: int) -> Optional[float]:
    """Average points the Tomatometer moved from prediction time to resolution.

    The final score of a film is bracketed by its rungs: above every threshold
    that settled YES and at/below every one that settled NO.
    """
    films: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"yes": [], "no": [], "early": None})
    for r in rs:
        ev, thr, y, liked, not_liked = r[1], r[7], r[6], r[8], r[9]
        if thr is None:
            continue
        f = films[ev]
        (f["yes"] if y else f["no"]).append(thr)
        if liked is not None and not_liked and liked + not_liked > 0:
            f["early"] = 100.0 * liked / (liked + not_liked)
        elif liked is not None and liked > 0 and not not_liked:
            f["early"] = 100.0
    diffs = []
    for f in films.values():
        if f["early"] is None or not (f["yes"] or f["no"]):
            continue
        lo = max(f["yes"]) + 1 if f["yes"] else 0
        hi = min(f["no"]) if f["no"] else 100
        if lo > hi:
            continue
        diffs.append((lo + hi) / 2 - f["early"])
    if len(diffs) < max(3, min_events // 2):
        return None
    return round(max(-10.0, min(10.0, sum(diffs) / len(diffs))), 1)


def would_have_traded(rows: List[tuple], min_gap: float = 0.10) -> Dict[str, Dict[str, Any]]:
    """What following the bot would have earned, per niche, ignoring every other gate.

    For each settled market, take the FIRST prediction where the bot disagreed
    with the market by at least ``min_gap`` (the moment it would first have
    bet), buy the side the bot preferred at the market's midpoint, pay Kalshi's
    fee, and hold to settlement. Real fills at the ask would be a little worse.
    Answers the question the Brier score can't: on the bets the bot wants to
    make, who turns out right - the bot or the market?
    """
    first: Dict[str, tuple] = {}
    for r in sorted(rows, key=lambda r: r[3]):
        mid, our, mkt = r[0], r[4], r[5]
        if mid in first or our is None or mkt is None or abs(our - mkt) < min_gap:
            continue
        price = mkt if our > mkt else 1 - mkt
        if not 0.03 <= price <= 0.97:
            continue  # no realistic fill that close to 0 or 100
        first[mid] = r
    out: Dict[str, Dict[str, Any]] = {}
    for r in first.values():
        niche, our, mkt, y = r[2], r[4], r[5], r[6]
        yes = our > mkt
        price = mkt if yes else 1 - mkt
        won = (y == 1) if yes else (y == 0)
        pnl = (1.0 if won else 0.0) - price - 0.07 * mkt * (1 - mkt)
        o = out.setdefault(niche, {"bets": 0, "wins": 0, "pnl": 0.0, "price": 0.0, "events": set(),
                                   "underdog_bets": 0, "underdog_pnl": 0.0})
        o["bets"] += 1
        o["wins"] += int(won)
        o["pnl"] += pnl
        o["price"] += price
        o["events"].add(r[1])
        if price <= 0.25:
            o["underdog_bets"] += 1
            o["underdog_pnl"] += pnl
    for o in out.values():
        n = o["bets"]
        o["events"] = len(o["events"])
        o["win_rate"] = o["wins"] / n
        o["avg_price"] = o.pop("price") / n
        o["cents_per_contract"] = 100 * o["pnl"] / n
        u = o["underdog_bets"]
        o["underdog_cents_per_contract"] = 100 * o["underdog_pnl"] / u if u else None
    return out


def load_rows(path: Optional[str] = None) -> List[tuple]:
    with closing(_conn(path)) as c:
        return c.execute(
            """SELECT market_id, event, niche, ts, our_prob, market_prob, outcome,
                      threshold, rt_liked, rt_not_liked
               FROM predictions WHERE outcome IS NOT NULL AND (close_ts IS NULL OR ts <= close_ts)"""
        ).fetchall()


def current_params(path: Optional[str] = None, max_age: float = 600) -> Dict[str, Dict[str, Any]]:
    """Learned parameters, recomputed at most every ``max_age`` seconds."""
    if _params_cache["params"] is not None and time.time() - _params_cache["ts"] < max_age:
        return _params_cache["params"]
    try:
        params = learn(load_rows(path))
    except Exception:
        params = {}
    _params_cache.update(ts=time.time(), params=params)
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/learned_params.json", "w") as f:
            json.dump(params, f, indent=2)
    except Exception:
        pass
    return params


def niche_params(niche: str) -> Dict[str, Any]:
    return current_params().get(niche) or {
        "trust": _default_trust(), "paused": False, "learned": False, "events": 0,
    }


def apply(niche: str, our_prob: float, market_prob: float) -> float:
    """Blend the bot's estimate with the market using the learned trust."""
    t = niche_params(niche)["trust"]
    return t * our_prob + (1 - t) * market_prob


def pending_count(path: Optional[str] = None) -> Dict[str, int]:
    with closing(_conn(path)) as c:
        return dict(c.execute(
            "SELECT niche, COUNT(DISTINCT market_id) FROM predictions WHERE outcome IS NULL GROUP BY niche"
        ).fetchall())


def latest_prediction(market_id: str, max_age_hours: float = 12.0,
                      path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Most recent recorded estimate for a market, if fresh enough."""
    with closing(_conn(path)) as c:
        row = c.execute(
            "SELECT ts, our_prob, market_prob, niche FROM predictions WHERE market_id=? ORDER BY ts DESC LIMIT 1",
            (market_id,),
        ).fetchone()
    if not row or time.time() - row[0] > max_age_hours * 3600:
        return None
    return {"ts": row[0], "our_prob": row[1], "market_prob": row[2], "niche": row[3]}
