import json

import pytest

from src import plan


def run_plan(monkeypatch: pytest.MonkeyPatch, reply: str, query: str = "a lion chasing a zebra") -> plan.Plan:
    monkeypatch.setattr(plan, "generate", lambda *_a, **_k: reply)  # pyright: ignore[reportUnknownLambdaType]
    return plan.plan_request(query)


def test_valid_reply_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = json.dumps({"kind": "video", "duration_s": 60, "style": "documentary",
                        "search_terms": "lion zebra chase", "fallback_terms": "lion"})
    assert run_plan(monkeypatch, reply) == {"kind": "video", "duration_s": 60, "style": "documentary",
                                            "search_terms": "lion zebra chase", "fallback_terms": "lion"}


def test_keyword_lists_become_one_search_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seen live: Gemini returned search_terms as a list of options."""
    reply = json.dumps({"search_terms": ["busy city street", "night neon lights"], "fallback_terms": ["city"]})
    result = run_plan(monkeypatch, reply)
    assert (result["search_terms"], result["fallback_terms"]) == ("busy city street", "city")


@pytest.mark.parametrize(("given", "expected"), [(50, 45), (10, 30), (90, 60), ("45", 45), ("soon", 30), (None, 30)])
def test_duration_snaps_to_supported_lengths(monkeypatch: pytest.MonkeyPatch, given: object, expected: int) -> None:
    assert run_plan(monkeypatch, json.dumps({"duration_s": given}))["duration_s"] == expected


@pytest.mark.parametrize(("given", "expected"), [("photo", "photo"), ("Photos", "photo"), ("video", "video"),
                                                 ("animation", "video"), (None, "video")])
def test_kind_is_normalised(monkeypatch: pytest.MonkeyPatch, given: object, expected: str) -> None:
    assert run_plan(monkeypatch, json.dumps({"kind": given}))["kind"] == expected


@pytest.mark.parametrize(("given", "expected"), [("Sports Commentator", "sports commentator"),
                                                 ("pirate", "documentary"), (None, "documentary")])
def test_unknown_style_falls_back(monkeypatch: pytest.MonkeyPatch, given: object, expected: str) -> None:
    assert run_plan(monkeypatch, json.dumps({"style": given}))["style"] == expected


@pytest.mark.parametrize("reply", ["not json at all", "[1, 2, 3]", "", "{\"kind\": "])
def test_malformed_reply_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    result = run_plan(monkeypatch, reply, query="sunset beach")
    assert result == {"kind": "video", "duration_s": 30, "style": "travel host",
                      "search_terms": "sunset beach", "fallback_terms": "sunset"}


def test_gemini_unavailable_is_raised_for_the_pipeline_to_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_a: object, **_k: object) -> str:
        raise RuntimeError("Gemini is unavailable right now")

    monkeypatch.setattr(plan, "generate", down)
    with pytest.raises(RuntimeError):
        plan.plan_request("a lion")


@pytest.mark.parametrize(("query", "kind", "duration", "style"), [
    ("a photo of the Eiffel Tower at night", "photo", 30, "travel host"),
    ("show me a picture, of a cat", "photo", 30, "documentary"),
    ("a one minute video of a lion", "video", 60, "documentary"),
    ("a football player scoring a goal", "video", 30, "sports commentator"),
    ("a rocket launching into space", "video", 30, "movie trailer"),
])
def test_keyword_guess(query: str, kind: str, duration: int, style: str) -> None:
    result = plan.guess_plan(query)
    assert (result["kind"], result["duration_s"], result["style"], result["search_terms"]) == \
        (kind, duration, style, query)
