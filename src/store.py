"""ChromaDB vector store: save media descriptions and search them by meaning."""
from pathlib import Path

import chromadb

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_collection = None


def _get_collection():
    """Open the collection once and reuse it. Cosine distance: 0 = identical, 2 = opposite."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_or_create_collection("media", metadata={"hnsw:space": "cosine"})
    return _collection


def _clean(metadata: dict) -> dict:
    """Chroma only accepts str/int/float/bool values, so drop None and stringify the rest."""
    return {k: v if isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in metadata.items() if v is not None}


def add_media(media_id: str, description: str, metadata: dict) -> None:
    """Save (or overwrite) one media item. metadata should include "kind": "photo"|"video"."""
    _get_collection().upsert(ids=[media_id], documents=[description], metadatas=[_clean(metadata)])


def search(query: str, k: int = 3, kind: str | None = None) -> list[dict]:
    """Return the k best matches as {id, description, metadata, distance}, optionally only photos or videos."""
    col = _get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
        where={"kind": kind} if kind else None,
    )
    return [
        {"id": i, "description": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def has_media(media_id: str) -> bool:
    """True if this media item is already stored, so we can skip describing it again."""
    return bool(_get_collection().get(ids=[media_id])["ids"])


if __name__ == "__main__":
    add_media("test-video-1", "A lioness sprints across dry savanna grass chasing a zebra.",
              {"kind": "video", "duration": 32, "local_path": "media/test1.mp4"})
    add_media("test-photo-1", "Sunset over white houses and blue domes in Santorini.",
              {"kind": "photo", "local_path": "media/test2.jpg"})
    for q, kind in [("lion hunting zebra", None), ("greek island at dusk", "photo"), ("lion", "photo")]:
        print(f"\n{q!r} kind={kind}")
        for m in search(q, k=2, kind=kind):
            print(f"  {m['distance']:.3f}  {m['id']}  {m['description'][:60]}")
