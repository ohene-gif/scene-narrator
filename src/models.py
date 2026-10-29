"""Typed shapes passed between the pipeline steps, so every module agrees on field names."""
from typing import Literal, TypedDict

Kind = Literal["photo", "video"]


class Plan(TypedDict):
    """What the user asked for, as decided by Gemini (and optionally overridden in the UI)."""
    kind: Kind
    duration_s: int
    style: str
    search_terms: str
    fallback_terms: str


class Media(TypedDict):
    """One downloaded photo or video. Photos have duration 0 and an empty thumbnail."""
    id: str
    url: str
    local_path: str
    kind: Kind
    tags: str
    author: str
    duration: int
    thumbnail: str


class MediaRecord(Media):
    """What is saved in the library alongside the description."""
    facts: str
    query: str


class Match(TypedDict):
    """A library item returned by a search. Lower distance means a closer match."""
    id: str
    description: str
    metadata: MediaRecord
    distance: float
