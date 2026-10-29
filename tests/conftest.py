"""Shared fixtures. Every test runs offline: Gemini, Pixabay and the real library are never touched."""
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import store
from src.models import Kind, Media, MediaRecord


def make_media(media_id: str, kind: Kind = "video", local_path: str = "", tags: str = "") -> Media:
    """Exactly the shape pixabay.search_media returns."""
    return Media(id=media_id, url=f"https://pixabay.com/{media_id}", local_path=local_path, kind=kind,
                 tags=tags, author="tester", duration=30 if kind == "video" else 0, thumbnail="")


def make_record(media_id: str, kind: Kind = "video", local_path: str = "", tags: str = "",
                facts: str = "", query: str = "") -> MediaRecord:
    return MediaRecord(**make_media(media_id, kind, local_path, tags), facts=facts, query=query)


@pytest.fixture
def temp_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the store at an empty throwaway database for the duration of one test."""
    monkeypatch.setattr(store, "DB_DIR", tmp_path / "db")
    monkeypatch.setattr(store, "_collection", None)
    monkeypatch.setattr(store, "_plans", None)
    yield tmp_path
    store._collection = None  # pyright: ignore[reportPrivateUsage]
    store._plans = None  # pyright: ignore[reportPrivateUsage]
