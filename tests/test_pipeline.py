"""Pipeline behaviour that matters for the demo: library first, cached narration, and working without Gemini."""
from pathlib import Path

import pytest

from src import describe, narrate, pipeline, pixabay, store, web
from src import plan as planner
from src.models import Kind, Media
from tests.conftest import make_media

NO_OVERRIDES = pipeline.Overrides()
REAL_PLAN_REQUEST = planner.plan_request
REAL_NARRATE = narrate.narrate


def gemini_down(*_a: object, **_k: object) -> str:
    raise RuntimeError("Gemini is unavailable right now")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, temp_store: Path, tmp_path: Path) -> dict[str, int]:
    """Fake Pixabay/Wikipedia/Gemini/gTTS and count how often each is used."""
    counts = {"plan": 0, "pixabay": 0, "describe": 0, "narrate": 0}
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")

    def fake_plan(query: str) -> planner.Plan:
        counts["plan"] += 1
        return planner.Plan(kind="video", duration_s=30, style="documentary",
                            search_terms="lion zebra chase", fallback_terms="lion")

    def fake_search(terms: str, kind: Kind = "video", **_: object) -> list[Media]:
        counts["pixabay"] += 1
        return [make_media("pixabay-video-1", kind, local_path=str(clip), tags="lion, zebra, hunt")]

    def fake_describe(*_a: object) -> str:
        counts["describe"] += 1
        return "0:00-0:10 A lioness chases a zebra across the savanna."

    def fake_narrate(*_a: object) -> str:
        counts["narrate"] += 1
        return f"Narration number {counts['narrate']}."

    monkeypatch.setattr(planner, "plan_request", fake_plan)
    monkeypatch.setattr(pixabay, "search_media", fake_search)
    monkeypatch.setattr(describe, "describe_media", fake_describe)
    monkeypatch.setattr(web, "web_facts", lambda _t: "Lions live in prides.")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(narrate, "narrate", fake_narrate)
    monkeypatch.setattr(narrate, "speak", lambda _t: str(tmp_path / "voice.mp3"))  # pyright: ignore[reportUnknownLambdaType]
    return counts


def test_first_request_fetches_then_repeat_is_fully_cached(calls: dict[str, int]) -> None:
    first = pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    second = pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    assert first and second
    assert not first.from_library and second.from_library
    assert second.script == first.script  # narration reused, not rewritten
    assert calls == {"plan": 1, "pixabay": 1, "describe": 1, "narrate": 1}  # repeat made no Gemini calls


def test_new_narrator_style_writes_new_narration_once(calls: dict[str, int]) -> None:
    pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    sports = pipeline.run("a lion chasing a zebra", pipeline.Overrides(style="sports commentator"), print)
    again = pipeline.run("a lion chasing a zebra", pipeline.Overrides(style="sports commentator"), print)
    assert sports and again and sports.script == again.script
    assert calls["narrate"] == 2


def test_seeded_scene_plays_with_gemini_completely_down(monkeypatch: pytest.MonkeyPatch,
                                                        calls: dict[str, int]) -> None:
    pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)  # seed while Gemini works
    # Now Gemini goes down: put back the real planner and narrator, whose Gemini calls fail.
    monkeypatch.setattr(planner, "plan_request", REAL_PLAN_REQUEST)
    monkeypatch.setattr(narrate, "narrate", REAL_NARRATE)
    monkeypatch.setattr(planner, "generate", gemini_down)
    monkeypatch.setattr(narrate, "generate", gemini_down)

    result = pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    assert result and result.from_library
    assert result.script == "Narration number 1."

    # A new wording has no saved plan, so the keyword guess is used and still finds the seeded scene.
    reworded = pipeline.run("A lion chasing zebras", NO_OVERRIDES, print)
    assert reworded and reworded.from_library and reworded.script == "Narration number 1."


def test_guessed_plans_are_not_saved(monkeypatch: pytest.MonkeyPatch, calls: dict[str, int]) -> None:
    monkeypatch.setattr(planner, "plan_request", gemini_down)
    pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    assert store.get_plan("a lion chasing a zebra") is None  # Gemini gets a chance next time


def test_saved_plan_lookup_ignores_case_and_spacing(calls: dict[str, int]) -> None:
    pipeline.run("a lion chasing a zebra", NO_OVERRIDES, print)
    assert store.get_plan("  A Lion   chasing a ZEBRA ") == store.get_plan("a lion chasing a zebra")
