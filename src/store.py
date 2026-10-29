"""ChromaDB vector store: save media descriptions and search them by meaning."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.models import Kind, Match, MediaRecord, Plan

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
DESCRIPTION_KEY = "description"

_collection: Collection | None = None
_plans: Collection | None = None


def _get_collection() -> Collection:
    """Open the media collection once and reuse it. Cosine distance: 0 = identical, 2 = opposite."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_or_create_collection("media", metadata={"hnsw:space": "cosine"})
    return _collection


def _get_plans() -> Collection:
    """Saved plans, looked up by exact request text."""
    global _plans
    if _plans is None:
        _plans = chromadb.PersistentClient(path=str(DB_DIR)).get_or_create_collection("plans")
    return _plans


def _record(metadata: Mapping[str, Any] | None) -> MediaRecord:
    """Rebuild a MediaRecord from Chroma's loose metadata mapping, with safe defaults for missing fields."""
    m = metadata or {}

    def text(key: str) -> str:
        return str(m.get(key) or "")

    return MediaRecord(
        id=text("id"), url=text("url"), local_path=text("local_path"),
        kind="photo" if m.get("kind") == "photo" else "video",
        tags=text("tags"), author=text("author"), duration=int(m.get("duration") or 0),
        thumbnail=text("thumbnail"), facts=text("facts"), query=text("query"),
    )


def _metadata(fields: Mapping[str, object]) -> dict[str, str | int]:
    """Chroma metadata must be flat scalars; all our TypedDict fields are str or int."""
    return {k: v for k, v in fields.items() if isinstance(v, (str, int))}


def _search_text(description: str, record: MediaRecord) -> str:
    """What gets embedded. Adding the original request and tags measurably separates related from unrelated
    queries (related <= ~0.6, unrelated >= ~0.8 cosine distance) compared with the description alone."""
    return f"Request: {record['query']}. Tags: {record['tags']}. {description}"


def add_media(media_id: str, description: str, record: MediaRecord) -> None:
    """Save (or overwrite) one media item. The plain description is kept in metadata for narration."""
    metadata = _metadata(record)
    metadata[DESCRIPTION_KEY] = description
    _get_collection().upsert(ids=[media_id], documents=[_search_text(description, record)], metadatas=[metadata])


def _match(media_id: str, document: str, metadata: Mapping[str, Any] | None, distance: float) -> Match:
    description = str((metadata or {}).get(DESCRIPTION_KEY) or document)
    return Match(id=media_id, description=description, metadata=_record(metadata), distance=distance)


def search(query: str, k: int = 3, kind: Kind | None = None) -> list[Match]:
    """Return up to k closest items, best first, optionally only photos or only videos."""
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, total), where={"kind": kind} if kind else None)
    ids = res["ids"][0]
    docs = (res["documents"] or [[]])[0]
    metas = (res["metadatas"] or [[]])[0]
    dists = (res["distances"] or [[]])[0]
    return [_match(i, d, m, dist) for i, d, m, dist in zip(ids, docs, metas, dists)]


def has_media(media_id: str) -> bool:
    """True if this media item is already stored, so we can skip describing it again."""
    return bool(_get_collection().get(ids=[media_id])["ids"])


def get_media(media_id: str) -> Match:
    """Load one stored item in the same shape search() returns, with distance 0. Raises KeyError if missing."""
    got = _get_collection().get(ids=[media_id], include=["documents", "metadatas"])
    if not got["ids"] or not got["documents"] or not got["metadatas"]:
        raise KeyError(media_id)
    return _match(media_id, got["documents"][0], got["metadatas"][0], 0.0)


def _narration_key(style: str) -> str:
    return f"narration:{style}"


def get_narration(media_id: str, style: str) -> str | None:
    """Narration saved earlier for this item in this style, so repeat requests need no Gemini call."""
    metas = _get_collection().get(ids=[media_id], include=["metadatas"])["metadatas"]
    value = (metas[0] or {}).get(_narration_key(style)) if metas else None
    return str(value) if value else None


def save_narration(media_id: str, style: str, text: str) -> None:
    col = _get_collection()
    metas = col.get(ids=[media_id], include=["metadatas"])["metadatas"]
    if not metas:
        raise KeyError(media_id)
    col.update(ids=[media_id], metadatas=[{**(metas[0] or {}), _narration_key(style): text}])


def _plan_id(query: str) -> str:
    return " ".join(query.lower().split())


def get_plan(query: str) -> Plan | None:
    """Plan saved earlier for this exact request (case and spacing ignored), so repeats need no Gemini call."""
    got = _get_plans().get(ids=[_plan_id(query)], include=["metadatas"])
    metas = got["metadatas"]
    if not metas or not metas[0]:
        return None
    m = metas[0]
    return Plan(kind="photo" if m.get("kind") == "photo" else "video", duration_s=int(str(m.get("duration_s") or 30)),
                style=str(m.get("style") or ""), search_terms=str(m.get("search_terms") or ""),
                fallback_terms=str(m.get("fallback_terms") or ""))


def save_plan(query: str, plan: Plan) -> None:
    _get_plans().upsert(ids=[_plan_id(query)], documents=[query], metadatas=[_metadata(plan)])


def count() -> int:
    return _get_collection().count()


if __name__ == "__main__":
    base = MediaRecord(id="", url="", local_path="", kind="video", tags="", author="", duration=0, thumbnail="",
                       facts="", query="")
    add_media("test-video-1", "A lioness sprints across dry savanna grass chasing a zebra.",
              {**base, "id": "test-video-1", "duration": 32})
    add_media("test-photo-1", "Sunset over white houses and blue domes in Santorini.",
              {**base, "id": "test-photo-1", "kind": "photo"})
    cases: list[tuple[str, Kind | None]] = [("lion hunting zebra", None), ("greek island at dusk", "photo"),
                                           ("lion", "photo")]
    for q, kind in cases:
        print(f"\n{q!r} kind={kind}")
        for m in search(q, k=2, kind=kind):
            print(f"  {m['distance']:.3f}  {m['id']}  {m['description'][:60]}")
