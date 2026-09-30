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
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    df["result"] = df["outcome"].map({1: "YES", 0: "NO"}).fillna("pending")
    return df.drop(columns=["ts", "outcome"])


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


def _bot_log(logs_dir: str = "logs") -> Optional[str]:
    """Newest log file written by the trading loop (CLI commands also create logs)."""
    import glob

    files = sorted(glob.glob(os.path.join(logs_dir, "trading_system*.log")), key=os.path.getmtime, reverse=True)
    for path in files[:8]:
        try:
            with open(path, "rb") as f:
                f.seek(max(0, os.path.getsize(path) - 200_000))
                if b"Trading Cycle" in f.read():
                    return path
        except OSError:
            continue
    return None


def bot_status(log_file: Optional[str] = None) -> Dict[str, Any]:
    """Last activity time and trading mode, read from the bot's log."""
    out: Dict[str, Any] = {"last_activity": None, "minutes_ago": None, "mode": "unknown"}
    log_file = log_file or _bot_log()
    if not log_file:
        return out
    try:
        mtime = os.path.getmtime(log_file)
    except OSError:
        return out
    out["last_activity"] = datetime.fromtimestamp(mtime)
    out["minutes_ago"] = (datetime.now().timestamp() - mtime) / 60
    try:
        with open(log_file, "rb") as f:
            f.seek(max(0, os.path.getsize(log_file) - 400_000))
            tail = f.read().decode("utf-8", "ignore")
        modes = re.findall(r"Live mode: (True|False)|live_mode=(True|False)", tail)
        if modes:
            last = modes[-1][0] or modes[-1][1]
            out["mode"] = "LIVE" if last == "True" else "paper"
    except OSError:
        pass
    return out
