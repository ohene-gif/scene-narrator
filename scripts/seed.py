"""Pre-load the demo scenes so they play instantly, with no Gemini calls, during the demo.

Run once before presenting (re-running is safe; finished scenes are skipped quickly):
    python -m scripts.seed
Each scene is fetched, described, narrated and voiced, then saved to the library.
Gemini can be busy, so failed scenes are retried a few times with a pause in between.
"""
import time

from src import pipeline

DEMO_QUERIES = [
    "a lion chasing a zebra",
    "a football player scoring a goal",
    "a photo of the Eiffel Tower at night",
    "waves crashing on a beach at sunset",
    "a busy city street at night",
    "a rocket launching into space",
]
ATTEMPTS = 3
RETRY_PAUSE_S = 45  # the free Gemini tier allows only a few requests per minute
SCENE_PAUSE_S = 10


def seed(query: str) -> bool:
    for attempt in range(1, ATTEMPTS + 1):
        start = time.monotonic()
        try:
            result = pipeline.run(query, pipeline.Overrides(), lambda step: print(f"    {step}"))
        except Exception as e:  # keep going with the other scenes; report at the end
            print(f"  attempt {attempt} failed: {e}")
            if attempt < ATTEMPTS:
                time.sleep(RETRY_PAUSE_S)
            continue
        if result is None:
            print("  no media found on Pixabay for this wording; try rephrasing it in DEMO_QUERIES")
            return False
        source = "already in library" if result.from_library else "fetched and saved"
        print(f"  OK ({source}, {time.monotonic() - start:.0f}s): {result.plan['style']}, "
              f"{result.match['metadata']['kind']}, {len(result.script.split())} words")
        return True
    return False


def main() -> None:
    failed: list[str] = []
    for query in DEMO_QUERIES:
        print(f"\n> {query}")
        if not seed(query):
            failed.append(query)
        time.sleep(SCENE_PAUSE_S)
    print(f"\n{len(DEMO_QUERIES) - len(failed)}/{len(DEMO_QUERIES)} demo scenes ready.")
    if failed:
        print("Not ready (run again later, Gemini may be busy):", *failed, sep="\n  - ")


if __name__ == "__main__":
    main()
