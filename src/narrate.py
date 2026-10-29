"""Pick a narrator style for the scene and have Gemini narrate the retrieved media."""


def narrate(query: str, matches: list[dict], style: str, duration_s: int) -> str:
    """Write narration in the given style, sized to about 2.5 words per second of media."""
    raise NotImplementedError


def speak(text: str) -> str:
    """Turn narration into an mp3 with gTTS; return the file path."""
    raise NotImplementedError
