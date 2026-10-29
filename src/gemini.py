"""Shared Gemini client with a hard per-call deadline and model fallback, so one busy model doesn't break the app."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as DeadlineExceeded

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Tried in order; a busy, slow or rate-limited model falls through to the next one.
# Ordered by what answered fastest in live testing (2026-10-29); override with GEMINI_MODEL in .env.
DEFAULT_MODELS = "gemini-3.6-flash,gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite"
MODELS = [m.strip() for m in os.getenv("GEMINI_MODEL", DEFAULT_MODELS).split(",") if m.strip()]

RETRYABLE = {429, 500, 503, 504}
DEFAULT_TIMEOUT_S = 30
RATE_LIMIT_PAUSE_S = 1.0  # brief pause after a 429 before trying the next model

_client: genai.Client | None = None
# Calls run in worker threads so a hung connection can be abandoned on our side, not just on Google's.
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")


def client() -> genai.Client:
    global _client
    if _client is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        _client = genai.Client()
    return _client


def _call(model: str, contents: types.ContentListUnionDict, config: types.GenerateContentConfig | None,
          timeout_s: float) -> str:
    future = _pool.submit(client().models.generate_content, model=model, contents=contents, config=config)
    resp = future.result(timeout=timeout_s)
    if not resp.text:
        raise RuntimeError(f"{model} returned no text")
    return resp.text.strip()


def generate(contents: types.ContentListUnionDict, config: types.GenerateContentConfig | None = None,
             timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Call Gemini and return the text, moving on to the next model in MODELS on any temporary failure."""
    failures: list[str] = []
    for model in MODELS:
        try:
            return _call(model, contents, config, timeout_s)
        except errors.APIError as e:
            if e.code not in RETRYABLE and e.code != 404:  # 404 = model retired or not available to this key
                raise
            failures.append(f"{model}: {e.code} {e.status}")
            if e.code == 429:
                time.sleep(RATE_LIMIT_PAUSE_S)
        except DeadlineExceeded:
            failures.append(f"{model}: no answer within {timeout_s:.0f}s")
        except (httpx.TransportError, RuntimeError) as e:  # dropped connection, DNS failure, empty reply
            failures.append(f"{model}: {type(e).__name__} {e}")
    raise RuntimeError("Gemini is unavailable right now (all models busy or unreachable). Try again in a minute. "
                       f"Details: {'; '.join(failures)}")
