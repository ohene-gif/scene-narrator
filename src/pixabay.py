"""Fetch free photos/videos from the Pixabay API and download them to media/."""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
API = {"photo": "https://pixabay.com/api/", "video": "https://pixabay.com/api/videos/"}


def _kind(kind: str) -> str:
    """Accept "photo"/"photos"/"video"/"videos" and return the singular form."""
    kind = kind.lower().rstrip("s")
    if kind not in API:
        raise ValueError(f"kind must be photo or video, got {kind!r}")
    return kind


def _pick_video_file(videos: dict) -> dict:
    """Pick a mid-size rendition (~960px) so Gemini uploads stay fast."""
    for size in ("small", "medium", "tiny", "large"):
        if videos.get(size, {}).get("url"):
            return videos[size]
    raise ValueError("no downloadable video file")


def _by_duration(hits: list[dict], min_dur: int | None, max_dur: int | None) -> list[dict]:
    """Pixabay can't filter by length, so do it here. Clips in range come first, then the closest ones."""
    lo, hi = min_dur or 0, max_dur or 10**6

    def gap(h):
        d = h.get("duration") or 0
        return 0 if lo <= d <= hi else min(abs(d - lo), abs(d - hi))

    return sorted(hits, key=gap)


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def search_media(query: str, kind: str = "videos", limit: int = 3,
                 min_dur: int | None = None, max_dur: int | None = None) -> list[dict]:
    """Return a list of {id, url, local_path, kind, tags, ...} for downloaded media."""
    kind = _kind(kind)
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        raise RuntimeError("PIXABAY_API_KEY is missing from .env")

    # Pixabay needs per_page >= 3; ask for more videos so the duration filter has room to choose.
    per_page = max(3, limit) if kind == "photo" else 30
    params = {"key": key, "q": query[:100], "per_page": per_page, "safesearch": "true"}
    if kind == "photo":
        params["image_type"] = "photo"

    r = requests.get(API[kind], params=params, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    if kind == "video":
        hits = _by_duration(hits, min_dur, max_dur)

    results = []
    for hit in hits[:limit]:
        if kind == "photo":
            src = hit["largeImageURL"]
            local_path = MEDIA_DIR / f"photo_{hit['id']}.jpg"
            extra = {"author": hit.get("user") or ""}
        else:
            file = _pick_video_file(hit["videos"])
            src = file["url"]
            local_path = MEDIA_DIR / f"video_{hit['id']}.mp4"
            extra = {"duration": hit.get("duration") or 0, "thumbnail": file.get("thumbnail") or ""}
        _download(src, local_path)
        results.append({
            "id": f"pixabay-{kind}-{hit['id']}",
            "url": hit["pageURL"],
            "local_path": str(local_path),
            "kind": kind,
            "tags": hit.get("tags") or "",
            **extra,
        })
    return results


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "lion"
    for m in search_media(q, "photo", limit=2):
        print(m)
    for m in search_media(q, "video", limit=2, min_dur=20, max_dur=60):
        print(m)
