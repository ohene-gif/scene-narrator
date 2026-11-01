"""Provider fallback: Groq takes over only when every Gemini model is unavailable."""
import base64
from pathlib import Path
from typing import Any

import pytest
import requests

from src import ai, describe, gemini, groq_backup

GEMINI_DOWN = RuntimeError("Gemini is unavailable right now")


def gemini_down(*_a: object, **_k: object) -> str:
    raise GEMINI_DOWN


class FakeResponse:
    def __init__(self, payload: dict[str, Any], content: bytes = b"") -> None:
        self.payload, self.content = payload, content

    ok = True

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]:
        return self.payload


@pytest.fixture
def groq_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Fake Groq's HTTP API with a key set; returns the request bodies it received."""
    bodies: list[dict[str, Any]] = []

    def post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> FakeResponse:
        assert url == groq_backup.URL and headers["Authorization"] == "Bearer gsk_test"
        bodies.append(json)
        return FakeResponse({"choices": [{"message": {"content": "<think>hmm</think> from groq "}}]})

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(groq_backup.requests, "post", post)
    return bodies


# --- ai.generate (planning and narration) ---

def test_gemini_answer_is_used_and_groq_never_called(monkeypatch: pytest.MonkeyPatch,
                                                     groq_calls: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(gemini, "generate", lambda *_a, **_k: "from gemini")  # pyright: ignore[reportUnknownLambdaType]
    assert ai.generate("hi") == "from gemini"
    assert groq_calls == []


def test_groq_takes_over_when_gemini_is_down(monkeypatch: pytest.MonkeyPatch,
                                             groq_calls: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(gemini, "generate", gemini_down)
    assert ai.generate("plan this", json_mode=True) == "from groq"  # visible reasoning stripped
    [body] = groq_calls
    assert body["model"] == groq_backup.TEXT_MODEL
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "user", "content": "plan this"}]


def test_without_groq_key_the_gemini_error_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "generate", gemini_down)
    with pytest.raises(RuntimeError) as info:
        ai.generate("hi")
    assert info.value is GEMINI_DOWN


def test_both_providers_failing_reports_both(monkeypatch: pytest.MonkeyPatch) -> None:
    def groq_down(*_a: object, **_k: object) -> FakeResponse:
        raise requests.ConnectionError("groq unreachable")

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(gemini, "generate", gemini_down)
    monkeypatch.setattr(groq_backup.requests, "post", groq_down)
    with pytest.raises(RuntimeError, match="Gemini is unavailable.*Groq backup failed"):
        ai.generate("hi")


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {"content": "<think>x</think>"}}]}])
def test_bad_groq_replies_raise_runtime_error(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(groq_backup.requests, "post", lambda *_a, **_k: FakeResponse(payload))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    with pytest.raises(RuntimeError):
        groq_backup.generate("hi")


# --- describe (Groq can't watch video, so it gets a still image) ---

def test_describe_photo_falls_back_to_groq_with_the_photo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                          groq_calls: list[dict[str, Any]]) -> None:
    photo = tmp_path / "pic.jpg"
    photo.write_bytes(b"jpeg-bytes")
    monkeypatch.setattr(describe, "generate", gemini_down)
    assert describe.describe_media(str(photo), "photo", "tower, night") == "from groq"
    [body] = groq_calls
    text, image = body["messages"][0]["content"]
    assert body["model"] == groq_backup.VISION_MODEL
    assert "tower, night" in text["text"]
    assert image["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()


def test_describe_video_falls_back_to_groq_with_the_thumbnail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                              groq_calls: list[dict[str, Any]]) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    fetched: list[str] = []

    def get(url: str, timeout: float) -> FakeResponse:
        fetched.append(url)
        return FakeResponse({}, content=b"thumb-bytes")

    monkeypatch.setattr(describe, "generate", gemini_down)
    monkeypatch.setattr(describe.requests, "get", get)
    assert describe.describe_media(str(video), "video", "lion", "https://cdn/thumb.jpg") == "from groq"
    assert fetched == ["https://cdn/thumb.jpg"]
    assert "one frame from a short video" in groq_calls[0]["messages"][0]["content"][0]["text"]


def test_describe_video_without_thumbnail_reports_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                      groq_calls: list[dict[str, Any]]) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(describe, "generate", gemini_down)
    with pytest.raises(RuntimeError, match="no thumbnail"):
        describe.describe_media(str(video), "video")
    assert groq_calls == []
