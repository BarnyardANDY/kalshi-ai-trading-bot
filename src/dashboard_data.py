"""
Read-only data for the control-panel dashboard (scripts/control_panel.py).

Everything here reads local files the bot already writes: trading_system.db
(paper/live trades and positions), data/learning.db (predictions and
grades), logs/ (activity and AI spend). Nothing here talks to Kalshi.
"""
from __future__ import annotations

import os
import pickle
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.niches import niche_for_ticker

TRADING_DB = os.getenv("TRADING_DB", "trading_system.db")
_AI_USAGE_FILES = ("logs/daily_openrouter_usage.pkl", "logs/daily_ai_usage.pkl")


def _niche(ticker: str) -> str:
    n = niche_for_ticker(ticker or "")
    return n.name if n else "other"


def _read(db: str, sql: str, params=()) -> pd.DataFrame:
    if not os.path.exists(db):
        return pd.DataFrame()
    try:
        with closing(sqlite3.connect(db)) as c:
            return pd.read_sql_query(sql, c, params=params)
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------------

def trades(db: str = TRADING_DB) -> pd.DataFrame:
    """Closed trades with niche, exit reason and per-contract P&L."""
    df = _read(db, "SELECT market_id, side, entry_price, exit_price, quantity, pnl, "
                   "entry_timestamp, exit_timestamp, rationale FROM trade_logs")
    if df.empty:
        return df
    df["niche"] = df["market_id"].map(_niche)
    df["exit_time"] = pd.to_datetime(df["exit_timestamp"], errors="coerce", format="mixed")
    df["entry_time"] = pd.to_datetime(df["entry_timestamp"], errors="coerce", format="mixed")
    df["exit_reason"] = df["rationale"].fillna("").map(
        lambda r: (re.search(r"EXIT:\s*([a-z_]+)", r) or [None, "unknown"])[1])
    df["settled"] = df["exit_reason"].eq("market_resolution")
    df["pnl_per_contract_c"] = 100 * df["pnl"] / df["quantity"].where(df["quantity"] > 0)
    return df.sort_values("exit_time").reset_index(drop=True)


def kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"trades": 0, "win_rate": None, "edge_per_contract_c": None, "total_pnl": 0.0,
                "max_drawdown": 0.0, "days_traded": 0, "settled": 0}
    equity = df["pnl"].cumsum()
    drawdown = (equity - equity.cummax().clip(lower=0)).min()
    contracts = df["quantity"].sum()
    return {
        "trades": int(len(df)),
        "win_rate": float((df["pnl"] > 0).mean()),
        "edge_per_contract_c": float(100 * df["pnl"].sum() / contracts) if contracts else None,
        "total_pnl": float(df["pnl"].sum()),
        "max_drawdown": float(min(0.0, drawdown)),
        "days_traded": int(df["exit_time"].dt.date.nunique()),
        "settled": int(df["settled"].sum()),
    }


def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["exit_time", "equity"])
    out = df[["exit_time", "pnl"]].copy()
    out["equity"] = out["pnl"].cumsum()
    return out


def pnl_by_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["day", "pnl", "trades"])
    g = df.groupby(df["exit_time"].dt.date).agg(pnl=("pnl", "sum"), trades=("pnl", "size"))
    return g.reset_index().rename(columns={"exit_time": "day"})


def by_niche(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.groupby("niche").agg(
        trades=("pnl", "size"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        total_pnl=("pnl", "sum"),
        contracts=("quantity", "sum"),
        settled=("settled", "sum"),
    )
    g["pnl_per_contract_c"] = 100 * g["total_pnl"] / g["contracts"]
    return g.reset_index().sort_values("total_pnl", ascending=False)


def open_positions(db: str = TRADING_DB) -> pd.DataFrame:
    df = _read(db, "SELECT market_id, side, entry_price, quantity, timestamp, confidence "
                   "FROM positions WHERE status='open' AND live=1 ORDER BY timestamp DESC")
    if df.empty:
        return df
    df["niche"] = df["market_id"].map(_niche)
    df["cost"] = df["entry_price"] * df["quantity"]
    df["max_payout"] = df["quantity"].astype(float)
    return df


# ----------------------------------------------------------------------------
# Learning / predictions
# ----------------------------------------------------------------------------

def learning_summary() -> pd.DataFrame:
    from src import learning

    params = learning.current_params(max_age=0)
    pending = learning.pending_count()
    rows = []
    for n in sorted(set(params) | set(pending)):
        p = params.get(n, {})
        rows.append({
            "niche": n,
            "settled_events": p.get("events", 0),
            "waiting_markets": pending.get(n, 0),
            "bot_brier": p.get("brier_ours"),
            "market_brier": p.get("brier_market"),
            "trust_in_bot": p.get("trust"),
            "learned": p.get("learned", False),
            "paused_by_learning": p.get("paused", False),
            "rt_drift_pts": p.get("rt_drift"),
        })
    return pd.DataFrame(rows)


def recent_predictions(limit: int = 200) -> pd.DataFrame:
    from src import learning

    df = _read(learning.DB_PATH, "SELECT market_id, niche, ts, our_prob, market_prob, confidence, outcome "
                                 "FROM predictions ORDER BY ts DESC LIMIT ?", (limit,))
    if df.empty:
        return df
    df["time"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                  .dt.tz_convert(os.getenv("DASHBOARD_TZ", "America/New_York")).dt.tz_localize(None))
    df["result"] = df["outcome"].map({1: "YES", 0: "NO"}).fillna("pending")
    return df.drop(columns=["ts", "outcome"])


def would_have_traded(min_gap: float = 0.10) -> pd.DataFrame:
    from src import learning

    rep = learning.would_have_traded(learning.load_rows(), min_gap)
    rows = [{"niche": n, "bets": o["bets"], "events": o["events"], "win_rate": o["win_rate"],
             "avg_price": o["avg_price"], "cents_per_contract": o["cents_per_contract"],
             "dollars_per_contract_each": o["pnl"], "underdog_bets": o["underdog_bets"],
             "underdog_cents": o["underdog_cents_per_contract"]}
            for n, o in sorted(rep.items())]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------------

def ai_spend_today() -> Dict[str, Any]:
    today = datetime.now().strftime("%Y-%m-%d")
    cost = reqs = 0.0
    limit = None
    for path in _AI_USAGE_FILES:
        try:
            with open(path, "rb") as f:
                t = pickle.load(f)
        except Exception:
            continue
        if getattr(t, "date", None) == today:
            cost = max(cost, float(getattr(t, "total_cost", 0.0)))
            reqs = max(reqs, float(getattr(t, "request_count", 0)))
            limit = getattr(t, "daily_limit", limit)
    return {"cost": cost, "requests": int(reqs), "limit": limit}


_BOT_MARKERS = (b"BEAST MODE TRADING BOT STARTED", b"Trading Cycle")


def _head_tail(path: str, n: int = 200_000) -> bytes:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(min(n, size))
        if size <= n:
            return head
        f.seek(max(n, size - n))
        return head + b"\n" + f.read()


def _bot_log(logs_dir: str = "logs") -> Optional[str]:
    """Newest log file written by the trading loop (CLI commands also create logs).

    A busy cycle can write more than the tail we read, so the start-up banner
    at the head of the file counts too.
    """
    import glob

    files = sorted(glob.glob(os.path.join(logs_dir, "trading_system*.log")), key=os.path.getmtime, reverse=True)
    for path in files[:12]:
        try:
            text = _head_tail(path)
        except OSError:
            continue
        if any(m in text for m in _BOT_MARKERS):
            return path
    return None


def _last_prediction_ts() -> Optional[float]:
    """Time of the bot's most recent prediction (written only by the running bot)."""
    try:
        from src import learning
        df = _read(learning.DB_PATH, "SELECT MAX(ts) AS ts FROM predictions")
        v = df["ts"].iloc[0] if not df.empty else None
        return float(v) if v is not None and v == v else None
    except Exception:
        return None


def bot_status(log_file: Optional[str] = None, prediction_ts: Optional[float] = None) -> Dict[str, Any]:
    """Last activity time and trading mode, from the bot's log and its predictions."""
    out: Dict[str, Any] = {"last_activity": None, "minutes_ago": None, "mode": "unknown"}
    log_file = log_file or _bot_log()
    stamps = []
    if log_file:
        try:
            stamps.append(os.path.getmtime(log_file))
        except OSError:
            log_file = None
    pts = prediction_ts if prediction_ts is not None else _last_prediction_ts()
    if pts:
        stamps.append(pts)
    if stamps:
        last = max(stamps)
        out["last_activity"] = datetime.fromtimestamp(last)
        out["minutes_ago"] = max(0.0, (datetime.now().timestamp() - last) / 60)
    if log_file:
        try:
            text = _head_tail(log_file, 400_000).decode("utf-8", "ignore")
            modes = re.findall(r"Trading Mode: (LIVE|PAPER)|Live mode: (True|False)|live_mode=(True|False)", text)
            if modes:
                last_mode = next(x for x in modes[-1] if x)
                out["mode"] = "LIVE" if last_mode in ("LIVE", "True") else "paper"
        except OSError:
            pass
    return out
