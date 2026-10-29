"""Search the internet (Wikipedia) for real facts about the scene, so the narration is accurate.

Gemini's own Google Search grounding needs a paid plan (free keys get "429 quota exceeded"),
so we use Wikipedia's free public API instead: no key, fast, and reliable.
"""
from typing import Any

import requests

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "SceneNarrator/1.0 (hackathon project; https://github.com/ohene-gif/scene-narrator)"}
TIMEOUT_S = 10
MAX_ARTICLES = 2
MAX_CHARS = 1200


def _search_titles(topic: str) -> list[str]:
    params: dict[str, str | int] = {"action": "query", "list": "search", "srsearch": topic,
                                    "srlimit": MAX_ARTICLES, "format": "json"}
    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=TIMEOUT_S)
    r.raise_for_status()
    hits: list[dict[str, Any]] = r.json().get("query", {}).get("search", [])
    return [str(h["title"]) for h in hits]


def _summary(title: str) -> str:
    r = requests.get(SUMMARY_URL.format(title=title.replace(" ", "_")), headers=HEADERS, timeout=TIMEOUT_S)
    r.raise_for_status()
    return str(r.json().get("extract") or "")


def web_facts(topic: str) -> str:
    """Return short Wikipedia summaries about the topic, or "" if nothing is found or the web is unreachable.
    Narration still works without facts, so failures here never stop the app."""
    try:
        summaries = [s for s in (_summary(t) for t in _search_titles(topic)) if s]
    except requests.RequestException:
        return ""
    return "\n".join(summaries)[:MAX_CHARS]


if __name__ == "__main__":
    import sys

    print(web_facts(sys.argv[1] if len(sys.argv) > 1 else "lion hunting zebra"))
