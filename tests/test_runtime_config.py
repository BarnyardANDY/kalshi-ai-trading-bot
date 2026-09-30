"""Live, dashboard-editable settings."""
import json
import os
import time

import pytest

from src import runtime_config as RC


def test_defaults_then_env_then_file(monkeypatch):
    monkeypatch.delenv("MIN_EDGE", raising=False)
    assert RC.get("MIN_EDGE") == 0.10
    monkeypatch.setenv("MIN_EDGE", "0.2")
    assert RC.get("MIN_EDGE") == 0.2
    RC.save({"MIN_EDGE": 0.15})
    assert RC.get("MIN_EDGE") == 0.15          # dashboard wins over .env
    RC.reset(["MIN_EDGE"])
    assert RC.get("MIN_EDGE") == 0.2           # back to .env


def test_values_are_clamped_and_typed():
    RC.save({"MIN_EDGE": 5, "RT_MIN_REVIEWS": "7.4", "PAUSED": "true"})
    assert RC.get("MIN_EDGE") == 0.5 and RC.get("RT_MIN_REVIEWS") == 7 and RC.paused() is True


def test_price_bounds_validated():
    with pytest.raises(ValueError):
        RC.save({"MIN_PRICE": 0.5, "MAX_PRICE": 0.5})
    with pytest.raises(KeyError):
        RC.save({"LIVE_TRADING_ENABLED": True})  # can't switch live on from here


def test_bot_sees_changes_without_restart(monkeypatch):
    from src.niches import enabled_niches
    monkeypatch.setenv("NICHES", "rotten_tomatoes")
    assert [n.name for n in enabled_niches()] == ["rotten_tomatoes"]
    RC.save({"NICHES": ["weather", "trump_mentions"]})
    assert [n.name for n in enabled_niches()] == ["weather", "trump_mentions"]
    time.sleep(0.01)
    RC.save({"NICHES": "weather"})
    assert [n.name for n in enabled_niches()] == ["weather"]


def test_empty_niches_env_still_means_all_markets(monkeypatch):
    from src.niches import enabled_niches
    monkeypatch.setenv("NICHES", "")
    assert enabled_niches() == []


def test_corrupt_file_keeps_last_good(monkeypatch):
    RC.save({"MIN_EDGE": 0.12})
    assert RC.get("MIN_EDGE") == 0.12
    with open(RC.PATH, "w") as f:
        f.write("{not json")
    os.utime(RC.PATH, (time.time() + 5, time.time() + 5))
    assert RC.get("MIN_EDGE") == 0.12
