"""The Scene Narrator pipeline, independent of any UI (used by app.py and scripts/seed.py).

plan the request -> check the library -> if nothing matches, fetch from Pixabay, describe with Gemini,
add facts from Wikipedia, and save -> narrate (cached per item and style) -> voice-over (cached per text).
"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src import describe, narrate, pixabay, store, web
from src import plan as planner
from src.models import Kind, Match, Media, MediaRecord, Plan

GOOD_MATCH = 0.65  # cosine distance; below this a saved item counts as a match (calibrated: related <= ~0.6)
PHOTO_SECONDS = 25  # narration length for a still photo
LENGTH_WINDOW = 15  # accept videos within +/- this many seconds of the requested length

Report = Callable[[str], None]


@dataclass(frozen=True)
class Overrides:
    kind: Kind | None = None
    duration_s: int | None = None
    style: str | None = None


@dataclass(frozen=True)
class Result:
    plan: Plan
    match: Match
    from_library: bool
    script: str
    audio_path: str


def apply_overrides(plan: Plan, o: Overrides) -> Plan:
    return Plan(
        kind=o.kind or plan["kind"],
        duration_s=o.duration_s or plan["duration_s"],
        style=o.style or plan["style"],
        search_terms=plan["search_terms"],
        fallback_terms=plan["fallback_terms"],
    )


def plan_for(query: str, report: Report) -> Plan:
    """Saved plan for this exact request, else ask Gemini (and save it), else the offline keyword guess."""
    saved = store.get_plan(query)
    if saved:
        return saved
    report("Understanding your request…")
    try:
        plan = planner.plan_request(query)
    except RuntimeError:
        report("Gemini is busy, reading the request from keywords…")
        return planner.guess_plan(query)
    store.save_plan(query, plan)
    return plan


def from_library(plan: Plan) -> Match | None:
    """Best saved item that matches well and whose media file is still on disk."""
    for m in store.search(plan["search_terms"], k=3, kind=plan["kind"]):
        if m["distance"] < GOOD_MATCH and Path(m["metadata"]["local_path"]).exists():
            return m
    return None


def fetch_and_save(query: str, plan: Plan, report: Report) -> Match | None:
    """Not in the library: fetch from Pixabay, describe with Gemini, add web facts, and save for next time."""
    d = plan["duration_s"]
    is_video = plan["kind"] == "video"
    min_dur, max_dur = (max(5, d - LENGTH_WINDOW), d + LENGTH_WINDOW) if is_video else (None, None)

    found: list[Media] = []
    for terms in dict.fromkeys((plan["search_terms"], plan["fallback_terms"])):  # dedupe, keep order
        report(f"Searching Pixabay for “{terms}”…")
        found = pixabay.search_media(terms, plan["kind"], limit=1, min_dur=min_dur, max_dur=max_dur)
        if found:
            break
    if not found:
        return None

    media = found[0]
    if not store.has_media(media["id"]):
        report("Looking up facts on Wikipedia…")
        facts = web.web_facts(plan["search_terms"])
        report(f"Gemini is watching the {media['kind']}…")
        description = describe.describe_media(media["local_path"], media["kind"], media["tags"])
        store.add_media(media["id"], description, MediaRecord(**media, facts=facts, query=query))
    return store.get_media(media["id"])


def narration_for(query: str, match: Match, plan: Plan, report: Report) -> str:
    """Reuse the saved narration for this item and style if there is one; otherwise write and save it."""
    cached = store.get_narration(match["id"], plan["style"])
    if cached:
        return cached
    meta = match["metadata"]
    seconds = (meta["duration"] or plan["duration_s"]) if meta["kind"] == "video" else PHOTO_SECONDS
    report(f"Writing {plan['style']} narration…")
    script = narrate.narrate(query, match, plan["style"], seconds)
    store.save_narration(match["id"], plan["style"], script)
    return script


def run(query: str, overrides: Overrides, report: Report) -> Result | None:
    """Run the whole pipeline for one request. Returns None if no media could be found anywhere."""
    plan = apply_overrides(plan_for(query, report), overrides)

    report("Checking your library…")
    match = from_library(plan)
    cached = match is not None
    if match is None:
        match = fetch_and_save(query, plan, report)
    if match is None:
        return None

    script = narration_for(query, match, plan, report)
    report("Recording voice-over…")
    return Result(plan, match, cached, script, narrate.speak(script))
