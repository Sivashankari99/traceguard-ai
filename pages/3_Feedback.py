"""
Feedback page.

Collects suggestions, bug reports, and feature requests into the same
CSV log that the main page's inline Helpful / Not Helpful buttons write
to -- over time this becomes a real evaluation dataset, not just a
comment box nobody reads.
"""

import sys
from pathlib import Path

import streamlit as st

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.feedback_store import append_feedback

st.set_page_config(page_title="Feedback - TraceGuard AI", page_icon="💬", layout="wide")

st.markdown("# 💬 Help Improve TraceGuard AI")
st.caption(
    "Suggestions, bugs, feature requests -- and every 👍 / 👎 left on an "
    "answer on the main page -- all land here."
)

st.info(
    "📘 Your feedback is recorded privately -- it is **not shown on this "
    "page or anywhere else in the app**."
)

with st.form("feedback_form", clear_on_submit=True):
    fb_type = st.selectbox(
        "What kind of feedback is this?",
        ["Suggestion", "Bug", "Feature request", "Confusing answer", "Other"],
    )
    message = st.text_area(
        "Tell us more", height=120,
        placeholder="What happened, or what would help?",
    )
    related_query = st.text_input(
        "Related query (optional)",
        placeholder="Paste the query this is about, if any",
    )
    submitted = st.form_submit_button("Submit Feedback", type="primary")

    if submitted:
        if not message.strip():
            st.warning("Please enter a message before submitting.")
        else:
            append_feedback(
                repo_root,
                source="feedback_page",
                type_=fb_type.lower().replace(" ", "_"),
                query=related_query.strip(),
                workflow="",
                rating_reason="",
                message=message.strip(),
            )
            st.success("Thanks -- your feedback was recorded!")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8a95a8; font-size:0.85em; padding: 10px 0;">
        Created as part of <b>LLM Zoomcamp</b> by <b>DataTalks.Club</b>.<br/>
        Built using <b>OpenAI</b> · <b>Sentence Transformers</b> · <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
