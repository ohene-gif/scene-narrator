"""Backup AI provider (Groq) used only when every Gemini model is busy or out of quota.

Groq has a free tier but cannot watch videos, so describe.py sends it the photo or the video's thumbnail instead.
Uses Groq's OpenAI-compatible HTTP API through requests, so no extra package is needed.
"""
import base64
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.groq.com/openai/v1/chat/completions"
TEXT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # models vary by account: GET /openai/v1/models
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
# Free tier allows ~1000 output tokens per minute per model, and Groq reserves the full max up front.
# Our longest reply (60 s of narration, ~150 words) is ~250 tokens, leaving room for any model reasoning.
MAX_TOKENS = 800
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)  # some models prefix their answer with visible reasoning


def available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def generate(prompt: str, *, image_jpeg: bytes | None = None, json_mode: bool = False,
             timeout_s: float = 30) -> str:
    """Ask Groq and return the text. Raises RuntimeError on any failure, like gemini.generate."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    content: str | list[dict[str, Any]] = prompt
    if image_jpeg is not None:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]
    body: dict[str, Any] = {
        "model": VISION_MODEL if image_jpeg is not None else TEXT_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": MAX_TOKENS,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=timeout_s)
        if not r.ok:  # Groq explains the problem (e.g. model not available to this key) in the body
            raise RuntimeError(f"Groq backup failed: {r.status_code} {r.text[:300]}")
        text = str(r.json()["choices"][0]["message"]["content"] or "")
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Groq backup failed: {e}") from e
    text = THINKING.sub("", text).strip()
    if not text:
        raise RuntimeError("Groq backup returned no text")
    return text
