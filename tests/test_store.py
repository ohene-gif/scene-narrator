from pathlib import Path

import pytest

from src import store
from tests.conftest import make_record

GOOD_MATCH = 0.65  # mirrors app.GOOD_MATCH (app.py runs Streamlit on import, so it isn't imported here)
LION = "A lioness sprints across dry savanna grass chasing a zebra."
SANTORINI = "Sunset over white houses and blue domes in Santorini."


def save_examples() -> None:
    store.add_media("v1", LION, make_record("v1", tags="lion, zebra, hunt, wildlife", query="a lion chasing a zebra"))
    store.add_media("p1", SANTORINI, make_record("p1", "photo", tags="santorini, greece, sunset",
                                                  query="greek island sunset"))


def test_empty_library_returns_nothing(temp_store: Path) -> None:
    assert store.count() == 0
    assert store.search("anything") == []


@pytest.mark.parametrize("query", ["lion zebra chase", "lions hunting on the plains", "predator chasing prey africa"])
def test_related_wording_is_a_good_match(temp_store: Path, query: str) -> None:
    save_examples()
    best = store.search(query)[0]
    assert best["id"] == "v1" and best["distance"] < GOOD_MATCH


@pytest.mark.parametrize("query", ["eiffel tower night", "football goal", "cooking pasta", "snowy mountains"])
def test_unrelated_wording_is_not_a_good_match(temp_store: Path, query: str) -> None:
    save_examples()
    assert all(m["distance"] > GOOD_MATCH for m in store.search(query))


def test_results_carry_plain_description_and_metadata(temp_store: Path) -> None:
    save_examples()
    best = store.search("lion zebra chase")[0]
    assert best["description"] == LION  # not the enriched search text
    assert best["metadata"]["kind"] == "video" and best["metadata"]["duration"] == 30


def test_kind_filter(temp_store: Path) -> None:
    store.add_media("v1", "A lion walking.", make_record("v1", "video"))
    store.add_media("p1", "A lion portrait.", make_record("p1", "photo"))
    assert [m["id"] for m in store.search("lion", kind="photo")] == ["p1"]
    assert [m["id"] for m in store.search("lion", kind="video")] == ["v1"]


def test_upsert_overwrites_instead_of_duplicating(temp_store: Path) -> None:
    store.add_media("v1", "old description", make_record("v1"))
    store.add_media("v1", "new description", make_record("v1", facts="lions sleep 20 hours a day"))
    assert store.count() == 1
    got = store.get_media("v1")
    assert got["description"] == "new description"
    assert got["metadata"]["facts"] == "lions sleep 20 hours a day"


def test_has_media_and_missing_get_raises(temp_store: Path) -> None:
    store.add_media("v1", "A lion.", make_record("v1"))
    assert store.has_media("v1")
    assert not store.has_media("missing")
    with pytest.raises(KeyError):
        store.get_media("missing")


def test_narration_is_saved_per_style_without_touching_the_item(temp_store: Path) -> None:
    store.add_media("v1", LION, make_record("v1", facts="Lions live in prides."))
    assert store.get_narration("v1", "documentary") is None
    store.save_narration("v1", "documentary", "On the savanna...")
    store.save_narration("v1", "sports commentator", "AND HE'S OFF!")
    assert store.get_narration("v1", "documentary") == "On the savanna..."
    assert store.get_narration("v1", "sports commentator") == "AND HE'S OFF!"
    assert store.get_media("v1")["metadata"]["facts"] == "Lions live in prides."
    assert store.get_media("v1")["description"] == LION
    with pytest.raises(KeyError):
        store.save_narration("missing", "documentary", "x")


def test_record_fills_defaults_for_missing_fields() -> None:
    record = store._record({"kind": "photo", "duration": None})  # pyright: ignore[reportPrivateUsage]
    assert record["kind"] == "photo" and record["duration"] == 0 and record["facts"] == ""
    assert store._record(None)["kind"] == "video"  # pyright: ignore[reportPrivateUsage]
