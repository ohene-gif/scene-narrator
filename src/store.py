"""ChromaDB vector store: save media descriptions and search them by meaning."""


def add_media(media_id: str, description: str, metadata: dict) -> None:
    raise NotImplementedError


def search(query: str, k: int = 3, kind: str | None = None) -> list[dict]:
    """Return the k best matches as {id, description, metadata, distance}, optionally only photos or videos."""
    raise NotImplementedError
