"""Streamlit UI: type a scene, see a matching photo/video, hear it narrated like a TV show.

All the work happens in src/pipeline.py; this file only collects input and shows the result.
"""
import streamlit as st

from src import pipeline, store
from src.plan import STYLES

AUTO = "Auto"


def sidebar() -> pipeline.Overrides:
    with st.sidebar:
        st.header("Options")
        kind = st.radio("Media", [AUTO, "Video", "Photo"], horizontal=True)
        length = st.select_slider("Video length (s)", [AUTO, "30", "45", "60"], value=AUTO)
        style = st.selectbox("Narrator", [AUTO, *STYLES])
        st.divider()
        st.metric("Items in your library", store.count())
    return pipeline.Overrides(
        kind=None if kind == AUTO else ("photo" if kind == "Photo" else "video"),
        duration_s=None if length == AUTO else int(length),
        style=None if style == AUTO else style,
    )


def render(result: pipeline.Result) -> None:
    meta = result.match["metadata"]
    if result.from_library:
        st.success("Found in your library: instant, no new download.")
    else:
        st.info("New scene: fetched from Pixabay, described by Gemini, enriched with Wikipedia facts, "
                "and saved to your library for next time.")

    left, right = st.columns([3, 2])
    with left:
        if meta["kind"] == "video":
            st.video(meta["local_path"], autoplay=True, muted=True)
        else:
            st.image(meta["local_path"], width="stretch")
        st.caption(f"Source: [Pixabay]({meta['url']}) · by {meta['author'] or 'Pixabay contributor'}")
    with right:
        st.subheader(f"🎙️ {result.plan['style'].title()}")
        st.audio(result.audio_path, autoplay=True)
        st.write(result.script)

    with st.expander("Behind the scenes"):
        st.json({"plan": result.plan, "match_id": result.match["id"],
                 "match_distance": round(result.match["distance"], 3)})
        st.markdown("**What Gemini saw**")
        st.write(result.match["description"])
        if meta["facts"]:
            st.markdown("**Facts from Wikipedia**")
            st.write(meta["facts"])


def main() -> None:
    st.set_page_config(page_title="Scene Narrator", page_icon="🎬", layout="wide")
    st.title("Scene Narrator")
    st.caption("Describe any scene. We find it, and narrate it like a TV show.")
    overrides = sidebar()

    query = st.text_input("What do you want to see?", placeholder="a one minute video of a lion chasing a zebra")
    if not query:
        return

    with st.status("Working…") as status:
        try:
            result = pipeline.run(query, overrides, lambda label: status.update(label=label))
        except Exception as e:  # show any API/network failure in the UI instead of a stack trace
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            return
        if result is None:
            status.update(label="Nothing found", state="error")
            st.warning("Couldn't find any media for that. Try different words.")
            return
        status.update(label="Ready", state="complete", expanded=False)
    render(result)


main()
