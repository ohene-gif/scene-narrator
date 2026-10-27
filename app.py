"""Streamlit UI: type a scene, see matching photos/videos, hear the narration."""
import streamlit as st

st.title("Scene Narrator")
st.caption("Describe any scene. We find it, and narrate it like a TV show.")

query = st.text_input("What do you want to see?", placeholder="a lion chasing a zebra")
if query:
    st.info("Search + narration coming soon.")
