"""ChromaDB vector store: save media descriptions and search them by meaning."""


def add_media(media_id: str, description: str, metadata: dict) -> None:
    raise NotImplementedError


def search(query: str, k: int = 3) -> list[dict]:
    """Return the k best matches as {id, description, metadata, distance}."""
    raise NotImplementedError
