"""Have the AI (Gemini, or Groq as backup) narrate a library match in a TV style, then voice it with gTTS."""
import hashlib
import re
from pathlib import Path

from gtts import gTTS  # pyright: ignore[reportMissingTypeStubs]

from src.ai import generate
from src.models import Match

AUDIO_DIR = Path(__file__).resolve().parent.parent / "media" / "narration"
WORDS_PER_SECOND = 2.5
# Clip timestamps like "0:00", "(0:05-0:10)" or "0:06 - 0:15:" copied from the description. Single-digit minutes
# only (clips are short), so a real clock time such as "10:30" is left alone.
TIMESTAMP = re.compile(r"\(?\b\d:\d{2}(?:\s*[-\u2013]\s*\d:\d{2})?\)?:?\s*")

# The narration is saved per item and style and replayed for other requests, so it must describe the scene only:
# no mention of what the viewer asked for.
PROMPT = """You are a {style} narrating this {kind}.
Write narration meant to be read aloud while it plays. About {words} words. No stage directions, no headings,
no emojis, no timestamps, just the spoken words. Describe only what is shown; never mention a request. {pacing}

What is on screen:
{description}

Background facts from Wikipedia. Use one only if it clearly fits, as general context; never present a fact as
something visible on screen (do not invent places, names, ages or details the description above does not show):
{facts}"""


def narrate(match: Match, style: str, duration_s: int) -> str:
    """Write narration in the given style, sized to about 2.5 words per second of media."""
    meta = match["metadata"]
    pacing = "Follow the action in the same order as the description." if meta["kind"] == "video" else ""
    text = generate(PROMPT.format(
        style=style, kind=meta["kind"], words=int(duration_s * WORDS_PER_SECOND), pacing=pacing,
        description=match["description"], facts=meta["facts"] or "none",
    ))
    return TIMESTAMP.sub("", text).strip()


def speak(text: str) -> str:
    """Turn narration into an mp3 with gTTS; return the file path. Same text reuses the cached file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{hashlib.md5(text.encode()).hexdigest()[:12]}.mp3"
    if not path.exists():
        gTTS(text).save(str(path))  # pyright: ignore[reportUnknownMemberType]
    return str(path)
