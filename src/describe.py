"""Send a photo/video to Gemini and get back a detailed, timestamped description (Groq + a still image as backup)."""
import time
from pathlib import Path

import requests
from google.genai import types

from src import groq_backup
from src.gemini import client, generate
from src.models import Kind

INLINE_LIMIT = 18 * 1024 * 1024  # Gemini accepts inline requests up to ~20 MB; bigger files go through the Files API
DESCRIBE_TIMEOUT_S = 60  # watching a video takes longer than a text-only call
MIME: dict[Kind, str] = {"photo": "image/jpeg", "video": "video/mp4"}

PROMPT: dict[Kind, str] = {
    "photo": "Describe this photo in detail for a search index: subjects, actions, setting, lighting, mood, colours. "
             "Write 4-6 sentences. Plain text, no lists.",
    "video": "Describe this video in detail for a search index. Give a timestamped breakdown like "
             "'0:00-0:05 ...' covering subjects, actions, camera movement and setting, then one sentence on the "
             "overall mood. Plain text.",
}
BACKUP_PROMPT: dict[Kind, str] = {
    "photo": PROMPT["photo"],
    "video": "This is one frame from a short video. Describe the scene in detail for a search index: subjects, likely "
             "action, setting, lighting, mood. Write 4-6 sentences. Plain text, no timestamps.",
}


def _media_part(path: Path, kind: Kind) -> types.Part | types.File:
    if path.stat().st_size <= INLINE_LIMIT:
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=MIME[kind])
    uploaded = client().files.upload(file=str(path))
    while uploaded.state == types.FileState.PROCESSING and uploaded.name:
        time.sleep(2)
        uploaded = client().files.get(name=uploaded.name)
    return uploaded


def _still_image(local_path: str, kind: Kind, thumbnail_url: str) -> bytes:
    """What the image-only backup can look at: the photo itself, or the video's thumbnail frame."""
    if kind == "photo":
        return Path(local_path).read_bytes()
    if not thumbnail_url:
        raise RuntimeError("no thumbnail available for the backup describer")
    r = requests.get(thumbnail_url, timeout=15)
    r.raise_for_status()
    return r.content


def describe_media(local_path: str, kind: Kind, tags: str = "", thumbnail_url: str = "") -> str:
    """Gemini watches the actual photo/video. If Gemini is unavailable, Groq describes a still image instead."""
    hint = f"\nThe uploader tagged it: {tags}." if tags else ""
    try:
        return generate([_media_part(Path(local_path), kind), PROMPT[kind] + hint], timeout_s=DESCRIBE_TIMEOUT_S)
    except RuntimeError as gemini_error:
        if not groq_backup.available():
            raise
        try:
            image = _still_image(local_path, kind, thumbnail_url)
        except (OSError, requests.RequestException, RuntimeError) as e:
            raise RuntimeError(f"{gemini_error} | backup could not get an image: {e}") from e
        return groq_backup.generate(BACKUP_PROMPT[kind] + hint, image_jpeg=image, timeout_s=DESCRIBE_TIMEOUT_S)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    print(describe_media(path, "video" if path.endswith(".mp4") else "photo"))
