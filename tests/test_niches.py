"""Niche selection and research parsing (offline, fixture-based)."""
import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from src import niches as N
from src import niche_research as R


def test_niche_for_ticker():
    assert N.niche_for_ticker("KXRT-VER-90").name == "rotten_tomatoes"
    assert N.niche_for_ticker("KXTRUMPMENTION-26SEP24-UKRA").name == "trump_mentions"
    assert N.niche_for_ticker("KXDJTJUSTICEMENTION-26-X").name == "trump_mentions"
    assert N.niche_for_ticker("KXFEDHIKE-2-26DEC31") is None
    assert N.niche_for_ticker("") is None


def test_niche_for_ticker_respects_enabled_subset():
    only_rt = [N.NICHES["rotten_tomatoes"]]
    assert N.niche_for_ticker("KXTRUMPMENTION-26SEP24-UKRA", only_rt) is None


def test_enabled_niches_env(monkeypatch):
    monkeypatch.setenv("NICHES", "rotten_tomatoes, bogus ,TRUMP_MENTIONS")
    assert [n.name for n in N.enabled_niches()] == ["rotten_tomatoes", "trump_mentions"]
    monkeypatch.setenv("NICHES", "")
    assert N.enabled_niches() == []


def test_display_title_includes_outcome():
    m = {"title": "Verity Rotten Tomatoes score?", "yes_sub_title": "Above 90"}
    assert N.market_display_title(m) == "Verity Rotten Tomatoes score? — Above 90"
    assert N.market_display_title({"title": "Will X happen?"}) == "Will X happen?"


RT_MARKET = {
    "ticker": "KXRT-VER-90",
    "title": "Verity Rotten Tomatoes score?",
    "yes_sub_title": "Above 90",
    "rules_primary": "If Verity has a Tomatometer score of above 90 on Oct 5, 2026 at 10:00 AM ET, then the market resolves to Yes.",
}


def test_parse_rt_market():
    info = R.parse_rt_market(RT_MARKET)
    assert info == {"name": "Verity", "op": "above", "threshold": 90, "date": "Oct 5, 2026"}


def test_rt_slugs():
    c = R.rt_slug_candidates("Springsteen: Deliver Me from Nowhere")
    assert c[0] == "springsteen_deliver_me_from_nowhere"
    assert "smurfs" == R.rt_slug_candidates("Smurfs")[0]


def test_parse_rt_page_scorecard_json():
    page = ('<html><title>Verity | Rotten Tomatoes</title><script id="media-scorecard-json">'
            '{"audienceScore":{"score":"70"},"criticsScore":{"averageRating":"7.1","likedCount":45,'
            '"notLikedCount":5,"reviewCount":50,"score":"90","scorePercent":"90%"}}</script></html>')
    d = R.parse_rt_page(page)
    assert d["score"] == 90 and d["reviews"] == 50
    assert d["liked"] == 45 and d["not_liked"] == 5 and d["exact_pct"] == 90.0
    assert d["page_title"] == "Verity | Rotten Tomatoes"


def test_parse_rt_page_attribute_fallback():
    d = R.parse_rt_page('<score-board tomatometerscore="83"></score-board> 120 Reviews')
    assert d["score"] == 83 and d["reviews"] == 120


def _rss(items):
    body = "".join(
        f"<item><title>{t} - {s}</title><source>{s}</source><pubDate>{format_datetime(d)}</pubDate></item>"
        for t, s, d in items
    )
    return f'<?xml version="1.0"?><rss><channel>{body}</channel></rss>'


def test_parse_news_rss_sorts_and_filters_old():
    now = datetime.now(timezone.utc)
    xml = _rss([
        ("Old story", "AP", now - timedelta(days=30)),
        ("Newer story", "CNN", now - timedelta(hours=2)),
        ("Mid story", "BBC", now - timedelta(days=3)),
    ])
    items = R.parse_news_rss(xml, max_items=5, max_age_days=7)
    assert [i["title"] for i in items] == ["Newer story", "Mid story"]
    assert items[0]["source"] == "CNN" and items[0]["age"] == "2h ago"
    assert R.parse_news_rss("not xml") == []


def test_parse_mention_market():
    m = {"title": "What will Donald Trump say during remarks at Xi Jinping's state arrival ceremony originally scheduled for September 24, 2026?",
         "yes_sub_title": "United Nations / NATO"}
    info = R.parse_mention_market(m)
    assert info["phrase"] == "United Nations / NATO"
    assert info["event"].startswith("remarks at Xi Jinping")


def test_build_context_offline(monkeypatch):
    async def no_net(url):
        return None
    monkeypatch.setattr(R, "_get", no_net)
    R._cache.clear()
    ctx = asyncio.run(R.build_research_context("rotten_tomatoes", RT_MARKET))
    assert "Resolution rules: If Verity" in ctx
    assert "above 90%" in ctx
    assert "Could not find a current Rotten Tomatoes page" in ctx


def test_build_context_uses_scraped_score(monkeypatch):
    page = '<title>Verity</title>{"releaseYear":"2026","criticsScore":{"likedCount":40,"notLikedCount":10,"reviewCount":50,"score":"80"}}'
    async def fake_get(url):
        return page if "rottentomatoes.com/m/verity" in url else None
    monkeypatch.setattr(R, "_get", fake_get)
    R._cache.clear()
    ctx = asyncio.run(R.build_research_context("rotten_tomatoes", RT_MARKET))
    assert "Tomatometer 80%" in ctx and "40 fresh / 10 rotten" in ctx


def test_ingest_keeps_outcome_and_niche(monkeypatch):
    from src.jobs import ingest

    monkeypatch.setenv("NICHES", "rotten_tomatoes")
    monkeypatch.setenv("NICHE_MIN_VOLUME", "20")

    saved = {}

    class FakeDB:
        async def upsert_markets(self, markets):
            saved["markets"] = markets

    class Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def debug(self, *a, **k): pass

    m = {
        "ticker": "KXRT-VER-85", "title": "Verity Rotten Tomatoes score?", "yes_sub_title": "Above 85",
        "yes_bid_dollars": "0.08", "yes_ask_dollars": "0.15", "no_bid_dollars": "0.85", "no_ask_dollars": "0.92",
        "volume_fp": "62.6", "close_time": "2026-10-05T14:00:00Z", "status": "active", "_niche": "rotten_tomatoes",
    }
    q = asyncio.Queue()
    asyncio.run(ingest.process_and_queue_markets([m], FakeDB(), q, set(), Log()))
    mk = saved["markets"][0]
    assert mk.title == "Verity Rotten Tomatoes score? — Above 85"
    assert mk.category == "rotten_tomatoes"
    assert q.qsize() == 1  # volume 62 passes the niche floor (20), not the global 100


def test_fetch_stops_on_short_page_even_with_cursor():
    calls = []

    class FakeClient:
        async def _make_authenticated_request(self, *a, **k):
            return {"series": []}

        async def get_markets(self, **k):
            calls.append(k["series_ticker"])
            return {"markets": [{"ticker": "KXRT-VER-90", "status": "active"}], "cursor": "always-more"}

    N._SERIES_CACHE.clear()
    out = asyncio.run(N.fetch_niche_markets(FakeClient(), [N.NICHES["rotten_tomatoes"]]))
    assert calls == ["KXRT"] and len(out) == 1 and out[0]["_niche"] == "rotten_tomatoes"


def test_rt_prefers_current_year_over_same_name_old_film(monkeypatch):
    old = '<title>Digger | Rotten Tomatoes</title>{"releaseYear":"1993","criticsScore":{"reviewCount":0}}'
    new = '<title>Digger | Rotten Tomatoes</title>{"releaseYear":"2026","criticsScore":{"likedCount":30,"notLikedCount":20,"reviewCount":50,"score":"60"}}'
    async def fake_get(url):
        if url.endswith("/m/digger"):
            return old
        if url.endswith("/m/digger_2026"):
            return new
        return None
    monkeypatch.setattr(R, "_get", fake_get)
    R._cache.clear()
    d = asyncio.run(R.rotten_tomatoes_score("Digger", 2026))
    assert d["url"].endswith("/m/digger_2026") and d["score"] == 60 and d["year"] == 2026


def test_rt_year_from_theaters_release_text():
    page = 'Release Date (Theaters)</dt><dd>Oct 2, 2026, Wide</dd>'
    assert R.parse_rt_page(page)["year"] == 2026
