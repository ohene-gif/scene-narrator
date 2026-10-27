"""Pick a narrator style for the scene and have Gemini narrate the retrieved media."""


def narrate(query: str, matches: list[dict]) -> str:
    raise NotImplementedError


def speak(text: str) -> str:
    """Turn narration into an mp3 with gTTS; return the file path."""
    raise NotImplementedError
