from pathlib import Path

import pytest

from src import narrate
from src.models import Match
from tests.conftest import make_record


@pytest.fixture
def prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    def fake_generate(prompt: str) -> str:
        sent.append(prompt)
        return "narration"

    monkeypatch.setattr(narrate, "generate", fake_generate)
    return sent


def match(kind: str, facts: str = "") -> Match:
    return Match(id="m", description="0:00-0:05 a lion walks", distance=0.1,
                 metadata=make_record("m", "photo" if kind == "photo" else "video", facts=facts))


@pytest.mark.parametrize(("seconds", "words"), [(30, 75), (22, 55), (60, 150)])
def test_word_budget_matches_clip_length(prompts: list[str], seconds: int, words: int) -> None:
    narrate.narrate("lions", match("video"), "documentary", seconds)
    assert f"About {words} words" in prompts[0]


def test_video_follows_timestamps_photo_does_not(prompts: list[str]) -> None:
    narrate.narrate("lions", match("video"), "documentary", 30)
    narrate.narrate("lions", match("photo"), "travel host", 25)
    assert "Follow the timestamps" in prompts[0]
    assert "Follow the timestamps" not in prompts[1] and "travel host" in prompts[1]


def test_web_facts_are_included(prompts: list[str]) -> None:
    narrate.narrate("lions", match("video", facts="Lions sleep up to 20 hours a day."), "documentary", 30)
    assert "Lions sleep up to 20 hours a day." in prompts[0]


def test_speak_caches_identical_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: list[str] = []

    class FakeTTS:
        def __init__(self, text: str) -> None:
            created.append(text)

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"mp3")

    monkeypatch.setattr(narrate, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(narrate, "gTTS", FakeTTS)
    first, second = narrate.speak("hello there"), narrate.speak("hello there")
    assert first == second and Path(first).exists()
    assert created == ["hello there"]
