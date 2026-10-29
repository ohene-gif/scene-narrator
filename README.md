# Scene Narrator

Describe any scene. The app finds matching photos and videos, then narrates them like a TV show.

## How it works

1. **Search**: your question is matched against a local vector database (ChromaDB) of media descriptions.
2. **Fetch**: if nothing matches well, new media is pulled from the free Pexels API.
3. **Describe**: Gemini watches each new photo/video and writes a detailed, timestamped description, which is embedded and stored, if it doesn't find it, it searches the internet.
4. **Narrate**: Gemini narrates the best matches in a style that fits the scene (documentary, sports commentator, travel host), and gTTS reads it aloud.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then paste your keys into .env
streamlit run app.py
```

Free keys:
- Gemini: https://aistudio.google.com
- Pexels: https://www.pexels.com/api

## Project layout

| File | What it does |
|---|---|
| `app.py` | Streamlit UI |
| `src/plan.py` | Gemini decides photo/video, length, and narrator style |
| `src/pexels.py` | Fetch media from Pexels |
| `src/describe.py` | Gemini describes photos/videos |
| `src/store.py` | ChromaDB storage + search |
| `src/narrate.py` | Narration + text-to-speech |
