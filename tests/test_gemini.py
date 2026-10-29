import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from google.genai import errors

from src import gemini


@dataclass
class Reply:
    text: str | None


class Hang:
    """Marker: the fake model sleeps instead of answering, like a stuck connection."""


@dataclass
class FakeModels:
    """Each model name maps to what it does when called: raise, hang, or return text."""
    behaviour: dict[str, Exception | Hang | str | None]
    calls: list[str] = field(default_factory=lambda: [])

    def generate_content(self, model: str, contents: Any, config: Any) -> Reply:
        self.calls.append(model)
        outcome = self.behaviour[model]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Hang):
            time.sleep(1)
            return Reply("too late")
        return Reply(outcome)


@dataclass
class FakeClient:
    models: FakeModels


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "test", "status": "TEST"}})


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    models = FakeModels({})
    monkeypatch.setattr(gemini, "MODELS", ["first", "second"])
    monkeypatch.setattr(gemini, "_client", FakeClient(models))
    monkeypatch.setattr(gemini, "RATE_LIMIT_PAUSE_S", 0)
    return models


def test_first_model_answers(fake: FakeModels) -> None:
    fake.behaviour.update(first="  hello  ", second="unused")
    assert gemini.generate("hi") == "hello"
    assert fake.calls == ["first"]


@pytest.mark.parametrize("failure", [
    api_error(503), api_error(504), api_error(429), api_error(404),
    httpx.ReadTimeout("slow"), httpx.ReadError("connection reset"), httpx.ConnectError("dns failed"), None,
])
def test_falls_back_to_next_model(fake: FakeModels, failure: Exception | None) -> None:
    fake.behaviour.update(first=failure, second="from second")
    assert gemini.generate("hi") == "from second"
    assert fake.calls == ["first", "second"]


def test_hung_model_is_abandoned_after_deadline(fake: FakeModels) -> None:
    fake.behaviour.update(first=Hang(), second="from second")
    start = time.monotonic()
    assert gemini.generate("hi", timeout_s=0.2) == "from second"
    assert time.monotonic() - start < 0.9  # did not wait for the hung call to finish


def test_non_retryable_error_is_raised_immediately(fake: FakeModels) -> None:
    fake.behaviour.update(first=api_error(400), second="unused")
    with pytest.raises(errors.APIError):
        gemini.generate("hi")
    assert fake.calls == ["first"]


def test_all_models_failing_gives_clear_error(fake: FakeModels) -> None:
    fake.behaviour.update(first=api_error(503), second=httpx.ConnectError("dns failed"))
    with pytest.raises(RuntimeError, match="Gemini is unavailable right now") as info:
        gemini.generate("hi")
    assert "first: 503" in str(info.value) and "second: ConnectError" in str(info.value)
