"""Local, read-only data-exploration dashboard.

Launch from the repo root:
    streamlit run frontend/app.py

Reads existing outputs only (results/<spacing dir>/, configs/, data/). It never
writes to data/ or results/ and never starts training -- see common.py.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ACF data explorer", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Dataset overview", default=True),
    st.Page("pages/viewer.py", title="Volume viewer"),
    st.Page("pages/training.py", title="Training progress"),
]
st.navigation(pages).run()
