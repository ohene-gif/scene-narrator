from pathlib import Path
from typing import Any

import pytest

from src import pixabay


def video_hit(hit_id: int, duration: int) -> dict[str, Any]:
    return {"id": hit_id, "duration": duration, "pageURL": f"https://pixabay.com/videos/id-{hit_id}/",
            "tags": "lion, wildlife", "user": "someone",
            "videos": {"large": {"url": "L"}, "medium": {"url": "M"}, "small": {"url": "S", "thumbnail": "T"}}}


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]:
        return self.payload


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, Path]]:
    """Fake the API key and downloads; returns the list of (url, path) downloads requested."""
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setenv("PIXABAY_API_KEY", "test-key")
    monkeypatch.setattr(pixabay, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(pixabay, "_download", lambda url, path: downloads.append((url, path)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    return downloads


def test_videos_in_length_window_come_first_and_keep_relevance_order() -> None:
    hits = [video_hit(1, 8), video_hit(2, 40), video_hit(3, 25), video_hit(4, 90)]
    ordered = pixabay._by_duration(hits, 20, 45)  # pyright: ignore[reportPrivateUsage]
    assert [h["id"] for h in ordered] == [2, 3, 1, 4]


def test_no_window_keeps_original_order() -> None:
    hits = [video_hit(1, 8), video_hit(2, 90)]
    assert pixabay._by_duration(hits, None, None) == hits  # pyright: ignore[reportPrivateUsage]


def test_prefers_small_rendition_then_falls_back() -> None:
    assert pixabay._pick_video_file({"small": {"url": "S"}, "large": {"url": "L"}})["url"] == "S"  # pyright: ignore[reportPrivateUsage]
    assert pixabay._pick_video_file({"small": {"url": ""}, "large": {"url": "L"}})["url"] == "L"  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError):
        pixabay._pick_video_file({})  # pyright: ignore[reportPrivateUsage]


def test_search_videos_returns_typed_media(monkeypatch: pytest.MonkeyPatch, offline: list[tuple[str, Path]]) -> None:
    seen: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        seen.update(url=url, params=params)
        return FakeResponse({"hits": [video_hit(7, 5), video_hit(8, 30)]})

    monkeypatch.setattr(pixabay.requests, "get", fake_get)
    [media] = pixabay.search_media("lion", "video", limit=1, min_dur=20, max_dur=40)

    assert seen["url"].endswith("/videos/") and seen["params"]["per_page"] == pixabay.VIDEO_CANDIDATES
    assert media == {"id": "pixabay-video-8", "url": "https://pixabay.com/videos/id-8/",
                     "local_path": str(pixabay.MEDIA_DIR / "video_8.mp4"), "kind": "video",
                     "tags": "lion, wildlife", "author": "someone", "duration": 30, "thumbnail": "T"}
    assert offline == [("S", pixabay.MEDIA_DIR / "video_8.mp4")]


def test_search_photos(monkeypatch: pytest.MonkeyPatch, offline: list[tuple[str, Path]]) -> None:
    hit = {"id": 5, "pageURL": "https://pixabay.com/photos/5", "largeImageURL": "BIG", "tags": "tower", "user": "ana"}
    monkeypatch.setattr(pixabay.requests, "get", lambda *_a, **_k: FakeResponse({"hits": [hit]}))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    [media] = pixabay.search_media("eiffel tower", "photo", limit=1)
    assert media["kind"] == "photo" and media["duration"] == 0 and media["author"] == "ana"
    assert offline == [("BIG", pixabay.MEDIA_DIR / "photo_5.jpg")]


def test_no_results_returns_empty(monkeypatch: pytest.MonkeyPatch, offline: list[tuple[str, Path]]) -> None:
    monkeypatch.setattr(pixabay.requests, "get", lambda *_a, **_k: FakeResponse({"hits": []}))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    assert pixabay.search_media("xqzv blorptastic", "video") == []
    assert offline == []


def test_missing_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PIXABAY_API_KEY"):
        pixabay.search_media("lion")
