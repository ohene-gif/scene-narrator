"""Fetch free photos/videos from the Pixabay API and download them to media/."""
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.models import Kind, Media

load_dotenv()

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
API: dict[Kind, str] = {"photo": "https://pixabay.com/api/", "video": "https://pixabay.com/api/videos/"}
VIDEO_SIZES = ("small", "medium", "tiny", "large")  # ~960px first, so Gemini uploads stay fast
VIDEO_CANDIDATES = 30  # Pixabay can't filter by length, so fetch extra and pick by duration here

Hit = dict[str, Any]


def _pick_video_file(videos: dict[str, Any]) -> dict[str, Any]:
    for size in VIDEO_SIZES:
        rendition: dict[str, Any] = videos.get(size) or {}
        if rendition.get("url"):
            return rendition
    raise ValueError("no downloadable video file")


def _by_duration(hits: list[Hit], min_dur: int | None, max_dur: int | None) -> list[Hit]:
    """Clips inside [min_dur, max_dur] first (keeping Pixabay's relevance order), then the closest lengths."""
    lo, hi = min_dur or 0, max_dur or 10**6

    def gap(hit: Hit) -> int:
        d = int(hit.get("duration") or 0)
        return 0 if lo <= d <= hi else min(abs(d - lo), abs(d - hi))

    return sorted(hits, key=gap)


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")  # never leave a half-written file under the real name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.replace(path)


def _to_media(hit: Hit, kind: Kind) -> Media:
    if kind == "photo":
        src, local_path, duration, thumbnail = hit["largeImageURL"], MEDIA_DIR / f"photo_{hit['id']}.jpg", 0, ""
    else:
        file = _pick_video_file(hit["videos"])
        src, local_path = file["url"], MEDIA_DIR / f"video_{hit['id']}.mp4"
        duration, thumbnail = int(hit.get("duration") or 0), str(file.get("thumbnail") or "")
    _download(src, local_path)
    return Media(
        id=f"pixabay-{kind}-{hit['id']}",
        url=str(hit["pageURL"]),
        local_path=str(local_path),
        kind=kind,
        tags=str(hit.get("tags") or ""),
        author=str(hit.get("user") or ""),
        duration=duration,
        thumbnail=thumbnail,
    )


def search_media(query: str, kind: Kind = "video", limit: int = 3,
                 min_dur: int | None = None, max_dur: int | None = None) -> list[Media]:
    """Search Pixabay, download the best `limit` results to media/, and return them."""
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        raise RuntimeError("PIXABAY_API_KEY is missing from .env")

    per_page = max(3, limit) if kind == "photo" else VIDEO_CANDIDATES  # Pixabay requires per_page >= 3
    params: dict[str, str | int] = {"key": key, "q": query[:100], "per_page": per_page, "safesearch": "true"}
    if kind == "photo":
        params["image_type"] = "photo"

    r = requests.get(API[kind], params=params, timeout=30)
    r.raise_for_status()
    hits: list[Hit] = r.json().get("hits", [])
    if kind == "video":
        hits = _by_duration(hits, min_dur, max_dur)
    return [_to_media(hit, kind) for hit in hits[:limit]]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "lion"
    for m in search_media(q, "photo", limit=2):
        print(m)
    for m in search_media(q, "video", limit=2, min_dur=20, max_dur=60):
        print(m)
