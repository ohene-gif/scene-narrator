from typing import Any

import pytest
import requests

from src import web


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]:
        return self.payload


def fake_wikipedia(pages: dict[str, str]):  # pyright: ignore[reportUnknownParameterType]
    """Search returns the given titles in order; each summary returns its text."""
    def get(url: str, params: dict[str, Any] | None = None, **_: Any) -> FakeResponse:
        if url == web.SEARCH_URL:
            return FakeResponse({"query": {"search": [{"title": t} for t in pages]}})
        title = url.rsplit("/", 1)[-1].replace("_", " ")
        return FakeResponse({"extract": pages[title]})
    return get  # pyright: ignore[reportUnknownVariableType]


def test_returns_summaries_of_top_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web.requests, "get", fake_wikipedia({"Lion": "Lions live in prides.",
                                                             "Plains zebra": "Zebras have stripes."}))
    assert web.web_facts("lion zebra") == "Lions live in prides.\nZebras have stripes."


def test_long_text_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web.requests, "get", fake_wikipedia({"Lion": "x" * 5000}))
    assert len(web.web_facts("lion")) == web.MAX_CHARS


def test_no_articles_gives_empty_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web.requests, "get", fake_wikipedia({}))
    assert web.web_facts("xqzv blorptastic") == ""


def test_network_failure_gives_empty_facts_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*_a: Any, **_k: Any) -> FakeResponse:
        raise requests.ConnectionError("no internet")

    monkeypatch.setattr(web.requests, "get", offline)
    assert web.web_facts("lion") == ""
