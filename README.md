# Scene Narrator

Describe any scene. The app finds a matching photo or short video, then narrates it like a TV show.

## How it works

1. **Plan**: Gemini reads the request and decides photo or video, length (30/45/60 s), narrator style, and search keywords.
2. **Check the library first**: the request is matched against a local vector database (ChromaDB) of media descriptions. A good match is shown instantly.
3. **Otherwise, go to the internet**:
   - new media is fetched from the free Pixabay API (with broader keywords if the first search finds nothing),
   - Gemini watches the photo/video and writes a detailed, timestamped description,
   - real facts about the scene are looked up on Wikipedia,
   - everything is saved to the library, so the next similar request is instant.
4. **Narrate**: Gemini writes narration sized to the clip (about 2.5 words per second) in the chosen style, and gTTS reads it aloud next to the media.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then paste your keys into .env
streamlit run app.py
```

Free keys:
- Gemini: https://aistudio.google.com (use a personal Google account)
- Pixabay: https://pixabay.com/api/docs/ (key shown once logged in)

## Before a demo

Pre-load the demo scenes so they play instantly with **no Gemini calls** (media, description, narration,
voice-over and plan are all saved in the local library):

```bash
python -m scripts.seed
```

Edit `DEMO_QUERIES` in `scripts/seed.py` to change the scenes. Re-running is safe. The free Gemini tier allows only
a few requests per minute, so seed well before presenting. The library (`chroma_db/`, `media/`) is local and not in
git, so run this on the machine you present from.

If Gemini is busy or down, the app still reads the request from keywords and plays any saved scene.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # offline: Gemini, Pixabay and Wikipedia are faked
basedpyright app.py src tests scripts   # strict type check
```

## Project layout

| File | What it does |
|---|---|
| `app.py` | Streamlit UI only |
| `src/models.py` | Typed data shapes shared by all modules |
| `src/gemini.py` | Gemini client with retry and model fallback |
| `src/plan.py` | Gemini decides photo/video, length, and narrator style |
| `src/pixabay.py` | Fetch photos/videos from Pixabay |
| `src/describe.py` | Gemini describes photos/videos |
| `src/web.py` | Wikipedia facts about the scene |
| `src/pipeline.py` | The full pipeline (library first, then the internet), shared by the app and seed script |
| `scripts/seed.py` | Pre-loads demo scenes |
| `src/store.py` | ChromaDB storage + search |
| `src/narrate.py` | Narration + text-to-speech |

Each module in `src/` can be run on its own for testing, e.g. `python -m src.pixabay "lion"`.
