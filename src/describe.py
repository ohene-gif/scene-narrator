"""Send a photo/video to Gemini and get back a detailed, timestamped description."""
import time
from pathlib import Path

from google.genai import types

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


def _media_part(path: Path, kind: Kind) -> types.Part | types.File:
    if path.stat().st_size <= INLINE_LIMIT:
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=MIME[kind])
    uploaded = client().files.upload(file=str(path))
    while uploaded.state == types.FileState.PROCESSING and uploaded.name:
        time.sleep(2)
        uploaded = client().files.get(name=uploaded.name)
    return uploaded


def describe_media(local_path: str, kind: Kind, tags: str = "") -> str:
    prompt = PROMPT[kind] + (f"\nThe uploader tagged it: {tags}." if tags else "")
    return generate([_media_part(Path(local_path), kind), prompt], timeout_s=DESCRIBE_TIMEOUT_S)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    print(describe_media(path, "video" if path.endswith(".mp4") else "photo"))
