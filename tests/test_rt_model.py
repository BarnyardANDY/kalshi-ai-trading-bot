"""Per-film Rotten Tomatoes ladder pricing (offline)."""
import asyncio
import time
from datetime import datetime

from src import niche_research as R
from src import rt_model as M
from src.utils.database import Market

RULES = "If Digger has a Tomatometer score of above {t} on Oct 5, 2026 at 10:00 AM ET, then the market resolves to Yes."


def _mk(t, price):
    return Market(f"KXRT-DIG-{t}", f"Digger Rotten Tomatoes score? — Above {t}", price, 1 - price,
                  5000, int(time.time()) + 86400 * 6, "rotten_tomatoes", "active", datetime.now())


class FakeKalshi:
    async def get_market(self, ticker):
        t = ticker.rsplit("-", 1)[1]
        return {"market": {"ticker": ticker, "title": "Digger Rotten Tomatoes score?", "yes_sub_title": f"Above {t}",
                           "rules_primary": RULES.format(t=t), "close_time": "2026-10-05T14:00:00Z"}}


class Log:
    def __init__(self): self.lines = []
    def info(self, m, *a, **k): self.lines.append(m)
    def warning(self, m, *a, **k): self.lines.append(m)


def _patch_rt(monkeypatch, liked, not_liked):
    page = ('<title>Digger | Rotten Tomatoes</title>{"releaseYear":"2026","criticsScore":'
            f'{{"likedCount":{liked},"notLikedCount":{not_liked},"reviewCount":{liked + not_liked},"score":"0"}}}}')
    async def fake_get(url):
        return page if url.endswith("/m/digger") else None
    monkeypatch.setattr(R, "_get", fake_get)
    R._cache.clear()


def test_threshold_and_event():
    assert M.threshold_of("KXRT-DIG-45", "Digger Rotten Tomatoes score? — Above 45") == 45
    assert M.threshold_of("KXRT-DIG-45") == 45
    assert M.threshold_of("KXRTCOMPARE-26-WICKED", "Highest rated movie") is None
    assert M.event_of("KXRT-DIG-45") == "KXRT-DIG"


def test_baseline_is_monotone_and_sensible():
    b = M.baseline_probs(45, 5, [50, 70, 85, 90, 95])
    vals = [b[t] for t in sorted(b)]
    assert vals == sorted(vals, reverse=True)
    assert b[50] > 0.99 and b[95] < 0.2


def test_make_monotone_fixes_contradictions():
    fixed = M.make_monotone({45: 0.3, 70: 0.6, 90: 0.0})
    assert fixed[45] == 0.3 and fixed[70] == 0.3 and fixed[90] == 0.01


def test_waits_for_reviews(monkeypatch):
    _patch_rt(monkeypatch, 2, 1)
    called = []
    class AI:
        async def get_completion(self, *a, **k):
            called.append(1); return None
    log = Log()
    out = asyncio.run(M.predict_rt_ladders([_mk(45, .79), _mk(70, .06)], AI(), FakeKalshi(), log))
    assert out == {} and not called
    assert any("waiting for reviews" in l for l in log.lines)


def test_one_ai_call_per_film_and_coherent(monkeypatch):
    _patch_rt(monkeypatch, 30, 20)
    calls = []
    class AI:
        async def get_completion(self, prompt, **k):
            calls.append(prompt)
            # Deliberately contradictory: >70 more likely than >45
            return '{"probabilities": {"45": 0.40, "60": 0.30, "70": 0.55}, "confidence": 0.7}'
    ms = [_mk(45, .79), _mk(60, .2), _mk(70, .06)]
    out = asyncio.run(M.predict_rt_ladders(ms, AI(), FakeKalshi(), Log()))
    assert len(calls) == 1
    assert "30 fresh / 20" in calls[0] and "Above 70" in calls[0]
    p = {k.rsplit('-', 1)[1]: v[0] for k, v in out.items()}
    assert p["45"] >= p["60"] >= p["70"]
    assert p["70"] <= 0.30


def test_falls_back_to_baseline_when_ai_fails(monkeypatch):
    _patch_rt(monkeypatch, 40, 10)
    class AI:
        async def get_completion(self, *a, **k):
            return "sorry"
    out = asyncio.run(M.predict_rt_ladders([_mk(70, .5), _mk(90, .1)], AI(), FakeKalshi(), Log()))
    assert set(out) == {"KXRT-DIG-70", "KXRT-DIG-90"}
    assert out["KXRT-DIG-70"][0] > out["KXRT-DIG-90"][0]
    assert out["KXRT-DIG-70"][1] == 0.5
