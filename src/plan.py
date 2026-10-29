"""Ask Gemini what the user wants: photo or video, how long, which narrator style, and what to search."""


def plan_request(query: str) -> dict:
    """Return {kind: "photo"|"video", duration_s: 30|45|60, style, search_terms}."""
    raise NotImplementedError
