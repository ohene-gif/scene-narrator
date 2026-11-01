"""Text generation with provider fallback: Gemini first, then Groq if every Gemini model is unavailable."""
from google.genai import types

from src import gemini, groq_backup


def generate(prompt: str, *, json_mode: bool = False, timeout_s: float = gemini.DEFAULT_TIMEOUT_S) -> str:
    """Return the model's text. Raises RuntimeError only if Gemini and the Groq backup both fail."""
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2) if json_mode else None
    try:
        return gemini.generate(prompt, config, timeout_s=timeout_s)
    except RuntimeError as gemini_error:
        if not groq_backup.available():
            raise
        try:
            return groq_backup.generate(prompt, json_mode=json_mode, timeout_s=timeout_s)
        except RuntimeError as backup_error:
            raise RuntimeError(f"{gemini_error} | {backup_error}") from backup_error
