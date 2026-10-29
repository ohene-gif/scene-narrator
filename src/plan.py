"""Work out what the user wants: photo or video, how long, which narrator style, and what to search.

plan_request() asks Gemini; guess_plan() is the offline keyword fallback used when Gemini is unavailable.
"""
import json
from typing import Any, cast

from google.genai import types

from src.gemini import generate
from src.models import Plan

STYLES = ["documentary", "sports commentator", "travel host", "news anchor", "movie trailer", "storyteller"]
DURATIONS = (30, 45, 60)
PLAN_TIMEOUT_S = 12  # a tiny JSON reply; if Gemini is this slow, the keyword guess is better than waiting
PHOTO_WORDS = {"photo", "photos", "picture", "pictures", "image", "images", "pic", "still"}
STYLE_WORDS = {  # first style whose keywords appear in the request wins; otherwise documentary
    "sports commentator": {"football", "soccer", "goal", "basketball", "tennis", "match", "race", "game", "player",
                           "boxing", "cricket", "surfing", "skateboarding", "olympics"},
    "travel host": {"city", "street", "tower", "beach", "island", "paris", "tokyo", "london", "hotel", "market",
                    "travel", "village", "temple", "skyline"},
    "news anchor": {"protest", "traffic", "storm", "flood", "fire", "election", "breaking"},
    "movie trailer": {"rocket", "explosion", "space", "launch", "chase", "volcano", "battle"},
}

PROMPT = """You turn a user's request into a media search plan. Reply with JSON only, with these keys:
- "kind": "video" if they want motion/action or say video/clip, "photo" if they say photo/picture/image or it is a still scene. Default "video".
- "duration_s": 30, 45 or 60. Use what they ask for ("a minute" = 60), else 30. For photos use 30.
- "style": one of {styles}. Pick what fits the scene (wildlife = documentary, a match = sports commentator, a city = travel host).
- "search_terms": ONE string of 2-4 plain keywords for a stock media site, e.g. "lion zebra chase".
- "fallback_terms": ONE string of 1-2 broader keywords in case nothing is found, e.g. "lion".

User request: {query}"""


def _words(query: str) -> set[str]:
    return set(query.lower().replace(",", " ").replace(".", " ").split())


def _text(value: Any) -> str:
    """Gemini sometimes returns a list of keyword options instead of one string; use the first option."""
    if isinstance(value, list):
        items = cast(list[Any], value)
        value = items[0] if items else ""
    return str(value or "").strip()


def _nearest_duration(value: Any) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return DURATIONS[0]
    return min(DURATIONS, key=lambda d: abs(d - seconds))


def _guess_style(query: str) -> str:
    words = _words(query)
    return next((style for style, keys in STYLE_WORDS.items() if words & keys), STYLES[0])


def _parse(text: str) -> dict[str, Any]:
    """Parse Gemini's JSON reply; anything malformed becomes {} so every field falls back to its default."""
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}


def _to_plan(raw: dict[str, Any], query: str) -> Plan:
    """Validate every field, falling back to a sensible default for anything missing or malformed."""
    search_terms = _text(raw.get("search_terms")) or query
    style = _text(raw.get("style")).lower()
    return Plan(
        kind="photo" if _text(raw.get("kind")).lower().startswith("photo") else "video",
        duration_s=_nearest_duration(raw.get("duration_s")),
        style=style if style in STYLES else _guess_style(query),
        search_terms=search_terms,
        fallback_terms=_text(raw.get("fallback_terms")) or search_terms.split()[0],
    )


def guess_plan(query: str) -> Plan:
    """Offline fallback: read kind, length and style from plain keywords; search with the request itself."""
    words = _words(query)
    return _to_plan({"kind": "photo" if words & PHOTO_WORDS else "video",
                     "duration_s": 60 if "minute" in words else 45 if "45" in words else 30}, query)


def plan_request(query: str) -> Plan:
    """Ask Gemini to plan the request. Raises RuntimeError if Gemini is unavailable."""
    text = generate(
        PROMPT.format(styles=", ".join(STYLES), query=query),
        types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
        timeout_s=PLAN_TIMEOUT_S,
    )
    return _to_plan(_parse(text), query)


if __name__ == "__main__":
    import sys

    print(plan_request(sys.argv[1] if len(sys.argv) > 1 else "show me a one minute video of a lion chasing a zebra"))
