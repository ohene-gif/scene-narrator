"""Drive the real Streamlit app headless with the external services faked out."""
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import describe, narrate, pixabay, plan, store, web
from src.models import Kind, Media, Plan
from tests.conftest import make_media

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fake_plan(kind: Kind = "video") -> Plan:
    return Plan(kind=kind, duration_s=30, style="documentary", search_terms="lion savanna", fallback_terms="lion")


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    video, photo, audio = tmp_path / "clip.mp4", tmp_path / "pic.jpg", tmp_path / "voice.mp3"
    video.write_bytes(b"\x00" * 64)
    Image.new("RGB", (8, 8)).save(photo)
    audio.write_bytes(b"\x00" * 64)
    return {"video": video, "photo": photo, "audio": audio}


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch, temp_store: Path, files: dict[str, Path]) -> dict[str, list[str]]:
    """Fake every outside call and record which ones happened."""
    calls: dict[str, list[str]] = {"pixabay": [], "describe": [], "web": []}

    def fake_search(terms: str, kind: Kind = "video", **_: object) -> list[Media]:
        calls["pixabay"].append(terms)
        if terms == "nothing":
            return []
        return [make_media("pixabay-video-1", kind, local_path=str(files[kind]), tags="lion, wildlife")]

    def fake_describe(path: str, kind: Kind, tags: str = "") -> str:
        calls["describe"].append(path)
        return "0:00-0:10 A lion walks through tall golden grass."

    def fake_web(topic: str) -> str:
        calls["web"].append(topic)
        return "Lions live in prides."

    monkeypatch.setattr(plan, "plan_request", lambda _q: fake_plan())  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(pixabay, "search_media", fake_search)
    monkeypatch.setattr(describe, "describe_media", fake_describe)
    monkeypatch.setattr(web, "web_facts", fake_web)
    monkeypatch.setattr(narrate, "narrate", lambda *_a: "Here, on the endless savanna, a lion walks.")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(narrate, "speak", lambda _t: str(files["audio"]))  # pyright: ignore[reportUnknownLambdaType]
    return calls


def ask(query: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input(query).run()
    assert not at.exception, at.exception
    return at


def test_new_scene_is_fetched_described_and_saved(services: dict[str, list[str]]) -> None:
    at = ask("a lion in the wild")
    assert "New scene" in at.info[0].value
    assert services == {"pixabay": ["lion savanna"], "describe": [services["describe"][0]],
                        "web": ["lion savanna"]}
    assert store.has_media("pixabay-video-1")
    assert store.get_media("pixabay-video-1")["metadata"]["facts"] == "Lions live in prides."
    assert any("savanna" in m.value for m in at.markdown)


def test_second_request_comes_from_library(services: dict[str, list[str]]) -> None:
    ask("a lion in the wild")
    at = ask("lions on the savanna")
    assert "Found in your library" in at.success[0].value
    assert len(services["pixabay"]) == 1 and len(services["describe"]) == 1  # nothing fetched the second time


def test_library_item_with_deleted_file_is_refetched(services: dict[str, list[str]], files: dict[str, Path]) -> None:
    ask("a lion in the wild")
    files["video"].unlink()
    files["video"].write_bytes(b"\x00")  # fake re-download; the library entry is reused, not described again
    ask("a lion in the wild")
    assert len(services["describe"]) == 1


def test_fallback_terms_used_when_first_search_is_empty(monkeypatch: pytest.MonkeyPatch,
                                                        services: dict[str, list[str]]) -> None:
    monkeypatch.setattr(plan, "plan_request", lambda _q: {**fake_plan(), "search_terms": "nothing"})  # pyright: ignore[reportUnknownLambdaType]
    ask("an obscure lion thing")
    assert services["pixabay"] == ["nothing", "lion"]


def test_nothing_found_shows_friendly_warning(monkeypatch: pytest.MonkeyPatch,
                                              services: dict[str, list[str]]) -> None:
    monkeypatch.setattr(plan, "plan_request",  # pyright: ignore[reportUnknownLambdaType]
                        lambda _q: {**fake_plan(), "search_terms": "nothing", "fallback_terms": "nothing"})
    at = ask("xqzv blorptastic")
    assert "Couldn't find any media" in at.warning[0].value
    assert services["pixabay"] == ["nothing"]  # identical terms are not searched twice


def test_service_failure_is_shown_not_crashed(monkeypatch: pytest.MonkeyPatch, services: dict[str, list[str]]) -> None:
    def broken(*_a: object, **_k: object) -> list[Media]:
        raise RuntimeError("PIXABAY_API_KEY is missing from .env")

    monkeypatch.setattr(pixabay, "search_media", broken)
    at = ask("a lion")
    assert "PIXABAY_API_KEY" in at.error[0].value


def test_gemini_down_while_planning_still_works(monkeypatch: pytest.MonkeyPatch,
                                                services: dict[str, list[str]]) -> None:
    def down(_q: str) -> Plan:
        raise RuntimeError("Gemini is unavailable right now")

    monkeypatch.setattr(plan, "plan_request", down)
    at = ask("a lion in the wild")
    assert not at.error and "New scene" in at.info[0].value
    assert services["pixabay"] == ["a lion in the wild"]  # searched with the request itself


def test_sidebar_photo_override(services: dict[str, list[str]]) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Photo").run()
    at.text_input[0].input("a lion running").run()
    assert not at.exception, at.exception
    assert store.get_media("pixabay-video-1")["metadata"]["kind"] == "photo"
    assert len(at.image) == 1
