"""
Live strategy settings, editable while the bot runs (from the dashboard).

Stored in ``data/runtime_config.json``. The bot re-reads the file whenever it
changes (checked by modification time), so a change takes effect on the next
cycle (about a minute) with no restart. Lookup order for every setting:

    runtime_config.json  ->  .env / environment  ->  built-in default

Only strategy knobs live here. Live trading itself can NOT be switched on
from this file: that still requires starting the bot with ``--live``.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

PATH = os.getenv("RUNTIME_CONFIG", "data/runtime_config.json")

ALL_NICHES = ["rotten_tomatoes", "trump_mentions", "weather"]

# key, label, type, default, min, max, step, unit, group, help
SPEC: List[Dict[str, Any]] = [
    {"key": "PAUSED", "label": "Pause new trades", "type": "bool", "default": False, "group": "Master",
     "help": "Stop opening new positions. Predictions and learning keep running; open positions are still managed."},
    {"key": "NICHES", "label": "Active niches", "type": "niches", "default": "rotten_tomatoes,trump_mentions,weather",
     "group": "Master", "help": "Which market groups the bot scans. New niches appear within ~5 minutes."},

    {"key": "MIN_EDGE", "label": "Min edge", "type": "float", "default": 0.10, "min": 0.0, "max": 0.5,
     "step": 0.01, "unit": "pts", "scale": 100, "group": "Signal gates",
     "help": "Bot probability minus market price required to bet (10 = 10 percentage points)."},
    {"key": "MIN_CONFIDENCE", "label": "Min confidence", "type": "float", "default": 0.60, "min": 0.0, "max": 1.0,
     "step": 0.05, "unit": "%", "scale": 100, "group": "Signal gates",
     "help": "Skip predictions the bot is less sure about than this."},
    {"key": "MIN_PRICE", "label": "Min entry price", "type": "float", "default": 0.05, "min": 0.01, "max": 0.5,
     "step": 0.01, "unit": "¢", "scale": 100, "group": "Signal gates",
     "help": "Skip markets priced below this (long shots)."},
    {"key": "MAX_PRICE", "label": "Max entry price", "type": "float", "default": 0.95, "min": 0.5, "max": 0.99,
     "step": 0.01, "unit": "¢", "scale": 100, "group": "Signal gates",
     "help": "Skip markets priced above this (near-certain; small upside, big downside)."},
    {"key": "NICHE_MIN_VOLUME", "label": "Min market volume", "type": "int", "default": 20, "min": 0, "max": 100000,
     "step": 10, "unit": "ct", "group": "Signal gates",
     "help": "Ignore markets that have traded fewer contracts than this (thin books = wide spreads)."},

    {"key": "MAX_POSITION_PCT", "label": "Max position size", "type": "float", "default": 3.0, "min": 0.5, "max": 20.0,
     "step": 0.5, "unit": "% of balance", "group": "Sizing & exits",
     "help": "Hard cap on any single bet as a share of your balance."},
    {"key": "HOLD_EXIT_MARGIN", "label": "Early-exit margin", "type": "float", "default": 0.05, "min": 0.0, "max": 0.5,
     "step": 0.01, "unit": "¢", "scale": 100, "group": "Sizing & exits",
     "help": "Positions hold to settlement unless the bid beats the bot's value by this much."},
    {"key": "LEARN_DEFAULT_TRUST", "label": "Starting trust in bot", "type": "float", "default": 0.60, "min": 0.0,
     "max": 1.0, "step": 0.05, "unit": "%", "scale": 100, "group": "Sizing & exits",
     "help": "Blend of bot view vs market price until a niche has enough settled results to learn its own."},

    {"key": "RT_MIN_REVIEWS", "label": "RT: min reviews", "type": "int", "default": 5, "min": 0, "max": 200,
     "step": 1, "unit": "reviews", "group": "Niche settings",
     "help": "Don't trade a film until Rotten Tomatoes shows this many critic reviews."},
    {"key": "RT_REPRICE_MINUTES", "label": "RT: re-price every", "type": "int", "default": 60, "min": 5, "max": 1440,
     "step": 5, "unit": "min", "group": "Niche settings",
     "help": "Re-ask the AI about a film at most this often unless new reviews land."},
    {"key": "DAILY_AI_COST_LIMIT", "label": "Daily AI budget", "type": "float", "default": 2.0, "min": 0.0,
     "max": 50.0, "step": 0.5, "unit": "$", "group": "Niche settings",
     "help": "AI spend cap per day (Rotten Tomatoes + Trump). Weather uses no AI."},
]
_BY_KEY = {s["key"]: s for s in SPEC}

_cache: Dict[str, Any] = {"mtime": None, "data": {}}


def _read_file() -> Dict[str, Any]:
    try:
        mtime = os.path.getmtime(PATH)
    except OSError:
        _cache.update(mtime=None, data={})
        return {}
    if mtime != _cache["mtime"]:
        try:
            with open(PATH) as f:
                data = json.load(f)
            _cache.update(mtime=mtime, data=data if isinstance(data, dict) else {})
        except (OSError, ValueError):
            pass  # keep last good copy on a partial/corrupt write
    return _cache["data"]


def _coerce(spec: Dict[str, Any], value: Any) -> Any:
    t = spec["type"]
    if t == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if t == "niches":
        items = value if isinstance(value, list) else str(value).split(",")
        return ",".join(n.strip().lower() for n in items if n and n.strip())
    v = float(value)
    if "min" in spec:
        v = max(spec["min"], v)
    if "max" in spec:
        v = min(spec["max"], v)
    return int(round(v)) if t == "int" else v


def get(key: str, default: Any = None) -> Any:
    """Current value: runtime file, then environment, then built-in default."""
    spec = _BY_KEY.get(key)
    data = _read_file()
    if key in data:
        try:
            return _coerce(spec, data[key]) if spec else data[key]
        except (TypeError, ValueError):
            pass
    env = os.getenv(key)
    if env not in (None, ""):
        try:
            return _coerce(spec, env) if spec else env
        except (TypeError, ValueError):
            pass
    if default is not None or not spec:
        return default
    return spec["default"]


def current() -> Dict[str, Any]:
    return {s["key"]: get(s["key"]) for s in SPEC}


def overrides() -> Dict[str, Any]:
    """Settings explicitly saved from the dashboard."""
    return dict(_read_file())


def save(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and persist updates atomically. Returns the saved file content."""
    data = dict(_read_file())
    for k, v in updates.items():
        if k not in _BY_KEY:
            raise KeyError(f"Unknown setting {k}")
        data[k] = _coerce(_BY_KEY[k], v)
    if "MIN_PRICE" in data and "MAX_PRICE" in data and data["MIN_PRICE"] >= data["MAX_PRICE"]:
        raise ValueError("Min entry price must be below max entry price")
    d = os.path.dirname(PATH) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".runtime_config.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, PATH)
    _cache["mtime"] = None  # force reload
    return data


def reset(keys: Optional[List[str]] = None) -> None:
    """Drop dashboard overrides (all, or the given keys) so .env/defaults apply."""
    data = dict(_read_file())
    for k in (keys or list(data)):
        data.pop(k, None)
    d = os.path.dirname(PATH) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".runtime_config.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, PATH)
    _cache["mtime"] = None


def paused() -> bool:
    return bool(get("PAUSED"))
