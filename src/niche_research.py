"""
Fresh, niche-specific research handed to the AI before it prices a market.

Without this, the AI sees only a market title and a price and falls back on
training data that can be months old. For the niches here the deciding facts
are public and change by the hour:

* Rotten Tomatoes: the film's current Tomatometer and review counts, read from
  its public Rotten Tomatoes page, plus recent review headlines.
* Trump mentions: recent headlines about the event and about Trump using the
  phrase, from Google News RSS.

Everything here is best-effort: any failure returns less context, never an
exception, so a flaky website can't stop the bot.
"""
from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_TIMEOUT = 12.0
_CACHE_TTL = 20 * 60
_cache: Dict[str, Tuple[float, Any]] = {}


def _cached(key: str):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    return None


def _store(key: str, value):
    _cache[key] = (time.time(), value)
    return value


async def _get(url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.text
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Google News
# ----------------------------------------------------------------------------

def parse_news_rss(xml_text: str, max_items: int = 6, max_age_days: float = 7.0) -> List[Dict[str, str]]:
    """Parse a Google News RSS feed into [{title, source, age}] newest first."""
    items: List[Tuple[datetime, Dict[str, str]]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    now = datetime.now(timezone.utc)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(pub_raw)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        age_h = (now - pub).total_seconds() / 3600
        if age_h > max_age_days * 24 or not title:
            continue
        # Google appends " - Source" to titles; drop it when we have the source.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        age = f"{int(age_h)}h ago" if age_h < 48 else f"{int(age_h // 24)}d ago"
        items.append((pub, {"title": html.unescape(title), "source": source, "age": age}))
    items.sort(key=lambda x: x[0], reverse=True)
    return [i[1] for i in items[:max_items]]


async def google_news(query: str, max_items: int = 6, max_age_days: float = 7.0) -> List[Dict[str, str]]:
    key = f"gn:{query}:{max_items}:{max_age_days}"
    hit = _cached(key)
    if hit is not None:
        return hit
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    text = await _get(url)
    return _store(key, parse_news_rss(text, max_items, max_age_days) if text else [])


def _fmt_news(label: str, items: List[Dict[str, str]]) -> str:
    if not items:
        return f"{label}: none found in the last week."
    lines = [f"{label}:"]
    for it in items:
        src = f" ({it['source']})" if it.get("source") else ""
        lines.append(f"- [{it['age']}] {it['title']}{src}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Rotten Tomatoes
# ----------------------------------------------------------------------------

_RULE_RE = re.compile(
    r"If\s+(?P<name>.+?)\s+has\s+a\s+Tomatometer\s+score\s+of\s+"
    r"(?P<op>above|below|at least|at most|greater than|less than)\s+(?P<num>\d{1,3})"
    r"(?:\s+on\s+(?P<date>[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}))?",
    re.IGNORECASE,
)


def parse_rt_market(market: Dict[str, Any]) -> Dict[str, Any]:
    """Pull title, threshold and resolution date out of an RT market."""
    rules = market.get("rules_primary") or ""
    out: Dict[str, Any] = {}
    m = _RULE_RE.search(rules)
    if m:
        out = {
            "name": m.group("name").strip(),
            "op": m.group("op").lower(),
            "threshold": int(m.group("num")),
            "date": m.group("date"),
        }
    if not out.get("name"):
        title = market.get("title") or ""
        name = re.sub(r"\s*(Rotten Tomatoes|RT)\s*score\??\s*$", "", title, flags=re.IGNORECASE).strip()
        if name:
            out["name"] = name
    return out


def rt_slug_candidates(name: str, year: Optional[int] = None) -> List[str]:
    """Likely Rotten Tomatoes URL slugs for a title, most likely first."""
    base = name.lower()
    base = base.replace("&", "and")
    base = re.sub(r"[’'`]", "", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    year = year or datetime.now().year
    cands = [base, f"{base}_{year}", f"{base}_{year - 1}", f"{base}_{year + 1}"]
    if base.startswith("the_"):
        cands += [base[4:], f"{base[4:]}_{year}"]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _first_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def parse_rt_page(page: str) -> Dict[str, Any]:
    """Extract critics-score data from a Rotten Tomatoes title page."""
    data: Dict[str, Any] = {}
    title = re.search(r"<title>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
    if title:
        data["page_title"] = html.unescape(re.sub(r"\s+", " ", title.group(1))).strip()

    idx = page.find('"criticsScore"')
    if idx != -1:
        window = page[idx: idx + 1500]
        data["score"] = _first_int(r'"score"\s*:\s*"?(\d{1,3})"?', window)
        data["reviews"] = _first_int(r'"reviewCount"\s*:\s*(\d+)', window)
        data["liked"] = _first_int(r'"likedCount"\s*:\s*(\d+)', window)
        data["not_liked"] = _first_int(r'"notLikedCount"\s*:\s*(\d+)', window)
    if data.get("score") is None:
        data["score"] = (
            _first_int(r'tomatometerscore="(\d{1,3})"', page)
            or _first_int(r'"tomatometerScore"\s*:\s*\{[^}]*?"score"\s*:\s*"?(\d{1,3})', page)
            or _first_int(r"(\d{1,3})%\s*</[^>]+>\s*(?:<[^>]+>\s*)*Tomatometer", page)
        )
    if data.get("reviews") is None:
        data["reviews"] = _first_int(r"(\d[\d,]*)\s+Reviews", page.replace(",", ""))
    liked, not_liked = data.get("liked"), data.get("not_liked")
    if liked is not None and not_liked is not None and liked + not_liked > 0:
        data["reviews"] = data.get("reviews") or liked + not_liked
        data["exact_pct"] = round(100 * liked / (liked + not_liked), 1)
    rel = re.search(r"Release Date \(Theaters\)[^0-9A-Za-z]*(?:<[^>]+>\s*)*([A-Z][a-z]{2} \d{1,2}, \d{4})", page)
    if rel:
        data["release"] = rel.group(1)
    data["year"] = _release_year(page, data.get("release"))
    return data


def _release_year(page: str, release: Optional[str] = None) -> Optional[int]:
    """Best guess at the title's release year (used to reject same-name older films)."""
    if release:
        m = re.search(r"(\d{4})$", release)
        if m:
            return int(m.group(1))
    for pat in (
        r'"releaseYear"\s*:\s*"?(\d{4})',
        r'releaseyear="(\d{4})"',
        r"Release Date \((?:Theaters|Streaming)\)[\s\S]{0,300}?\b((?:19|20)\d{2})\b",
        r"<title>[^<]*\(((?:19|20)\d{2})\)",
        r'"dateCreated"\s*:\s*"((?:19|20)\d{2})',
    ):
        y = _first_int(pat, page)
        if y:
            return y
    return None


async def rotten_tomatoes_score(name: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Scores for the title, preferring the page whose release year matches.

    Plain slugs often belong to an older film of the same name (/m/digger is
    a 1993 film; the 2026 one is /m/digger_2026), so a page is only accepted
    outright when its release year is within a year of the market's year.
    """
    year = year or datetime.now().year
    key = f"rt:{name.lower()}:{year}"
    hit = _cached(key)
    if hit is not None:
        return hit or None
    fallback = None
    for slug in rt_slug_candidates(name, year):
        for kind in ("m", "tv"):
            url = f"https://www.rottentomatoes.com/{kind}/{slug}"
            page = await _get(url)
            if not page:
                continue
            data = parse_rt_page(page)
            data["url"] = url
            y = data.get("year")
            if y is not None and abs(y - year) <= 1:
                return _store(key, data)
            if y is None and fallback is None:
                fallback = data  # unknown year: keep as a last resort
            break  # found a page for this slug; try the next slug variant
    if fallback:
        fallback["year_unverified"] = True
        return _store(key, fallback)
    _store(key, {})
    return None


async def _rt_context(market: Dict[str, Any]) -> str:
    info = parse_rt_market(market)
    name = info.get("name")
    if not name:
        return ""
    lines = []
    if info.get("threshold") is not None:
        when = f" on {info['date']}" if info.get("date") else ""
        lines.append(f"Resolves YES if the Tomatometer is {info['op']} {info['threshold']}%{when}.")
    year = None
    if info.get("date"):
        m = re.search(r"(\d{4})$", info["date"])
        year = int(m.group(1)) if m else None
    rt = await rotten_tomatoes_score(name, year)
    if rt:
        parts = []
        if rt.get("reviews") == 0 or (rt.get("score") is None and not rt.get("reviews")):
            parts.append("no critic reviews yet (score not posted; likely still under review embargo)")
        else:
            if rt.get("score") is not None:
                parts.append(f"Tomatometer {rt['score']}%")
            if rt.get("reviews"):
                parts.append(f"{rt['reviews']} reviews")
            if rt.get("liked") is not None and rt.get("not_liked") is not None:
                parts.append(f"{rt['liked']} fresh / {rt['not_liked']} rotten (exact {rt.get('exact_pct')}%)")
        if rt.get("release"):
            parts.append(f"theatrical release {rt['release']}")
        lines.append(f"Rotten Tomatoes right now ({rt.get('url')}): " + "; ".join(parts) + ".")
        if rt.get("page_title"):
            yr = f", release year {rt['year']}" if rt.get("year") else ""
            lines.append(f"(Page: {rt['page_title']}{yr} — check it is the right title.)")
        if rt.get("year_unverified"):
            lines.append("WARNING: could not confirm this page is the right year's title; treat with caution.")
    else:
        lines.append(f"Could not find a current Rotten Tomatoes page for '{name}'.")
    news = await google_news(f'"{name}" review OR "Rotten Tomatoes"', max_items=6, max_age_days=10)
    lines.append(_fmt_news("Recent headlines", news))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Trump mentions
# ----------------------------------------------------------------------------

def parse_mention_market(market: Dict[str, Any]) -> Dict[str, str]:
    title = market.get("title") or ""
    phrase = (market.get("yes_sub_title") or market.get("subtitle") or "").strip()
    event = ""
    m = re.search(r"\bsay\s+(?:during|at|in|on)\s+(.+?)\??$", title, re.IGNORECASE)
    if m:
        event = m.group(1).strip()
    return {"phrase": phrase, "event": event}


async def _mention_context(market: Dict[str, Any]) -> str:
    info = parse_mention_market(market)
    lines = []
    if info["phrase"]:
        lines.append(f"Phrase to be said: \"{info['phrase']}\" (see rules for exact wording/variants).")
    if info["event"]:
        # Drop filler so the search matches news phrasing.
        ev = re.sub(r"\b(remarks at|originally scheduled for|the)\b", " ", info["event"], flags=re.IGNORECASE)
        ev = re.sub(r"\s+", " ", ev).strip()
        news = await google_news(f"Trump {ev}", max_items=6, max_age_days=10)
        lines.append(_fmt_news(f"News about the event ({info['event']})", news))
    if info["phrase"]:
        variants = [v.strip() for v in info["phrase"].split("/") if v.strip()][:2]
        q = " OR ".join(f'"{v}"' for v in variants)
        news = await google_news(f"Trump ({q})", max_items=6, max_age_days=5)
        lines.append(_fmt_news("Recent headlines with Trump and the phrase", news))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

async def build_research_context(niche_name: Optional[str], market: Dict[str, Any]) -> str:
    """Research block for the AI prompt. Empty string if nothing useful."""
    parts: List[str] = [f"Current date/time (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"]
    rules = (market.get("rules_primary") or "").strip()
    if rules:
        parts.append(f"Resolution rules: {rules}")
    close = market.get("close_time")
    if close:
        parts.append(f"Market closes: {close}")
    try:
        if niche_name == "rotten_tomatoes":
            extra = await _rt_context(market)
        elif niche_name == "trump_mentions":
            extra = await _mention_context(market)
        else:
            extra = ""
    except Exception as e:  # never let research break trading
        extra = f"(Research step failed: {e})"
    if extra:
        parts.append(extra)
    return "\n".join(parts)
