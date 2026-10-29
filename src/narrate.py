"""Have Gemini narrate a library match in a TV style, then voice it with gTTS."""
import hashlib
from pathlib import Path

from gtts import gTTS  # pyright: ignore[reportMissingTypeStubs]

from src.gemini import generate
from src.models import Match

AUDIO_DIR = Path(__file__).resolve().parent.parent / "media" / "narration"
WORDS_PER_SECOND = 2.5

PROMPT = """You are a {style} narrating this {kind} for a viewer who asked for: "{query}".
Write narration meant to be read aloud while it plays. About {words} words. No stage directions, no headings,
no emojis, just the spoken words. {pacing}

What is on screen:
{description}

Real facts you may weave in (only if they fit):
{facts}"""


def narrate(query: str, match: Match, style: str, duration_s: int) -> str:
    """Write narration in the given style, sized to about 2.5 words per second of media."""
    meta = match["metadata"]
    pacing = "Follow the timestamps so the words match what is on screen." if meta["kind"] == "video" else ""
    return generate(PROMPT.format(
        style=style, kind=meta["kind"], query=query, words=int(duration_s * WORDS_PER_SECOND), pacing=pacing,
        description=match["description"], facts=meta["facts"] or "none",
    ))


def speak(text: str) -> str:
    """Turn narration into an mp3 with gTTS; return the file path. Same text reuses the cached file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{hashlib.md5(text.encode()).hexdigest()[:12]}.mp3"
    if not path.exists():
        gTTS(text).save(str(path))  # pyright: ignore[reportUnknownMemberType]
    return str(path)
