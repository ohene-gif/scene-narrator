"""Fetch free photos/videos from the Pexels API and download them to media/."""


def search_pexels(query: str, kind: str = "videos", limit: int = 3,
                  min_dur: int | None = None, max_dur: int | None = None) -> list[dict]:
    """Return a list of {id, url, local_path, kind} for downloaded media."""
    raise NotImplementedError
