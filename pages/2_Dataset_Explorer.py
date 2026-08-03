"""
Dataset Explorer page.

Lets a visitor see exactly what TraceGuard AI actually knows about --
the real synthetic artifact set -- rather than wondering what universe
it's drawing from. Also doubles as a "search before you query" flow:
pick an artifact here, and one click sends a ready-made question to the
main page instead of the visitor having to remember or retype an ID.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

st.set_page_config(page_title="Dataset Explorer - TraceGuard AI", page_icon="📄", layout="wide")

st.markdown("# 📄 Dataset Explorer")
st.caption("Browse the synthetic engineering artifacts TraceGuard AI actually knows about.")

st.info(
    "📘 **About this demo** — This application is built on approximately "
    "**5,000 synthetic engineering artifacts**. The data is fictional and "
    "intended to demonstrate AI workflows, retrieval, traceability, and "
    "planning. It is **not connected to a live engineering repository**."
)


@st.cache_data
def load_artifacts():
    return pd.read_csv(repo_root / "data" / "artifacts.csv")


df = load_artifacts()

# ----------------------------------------------------------------------
# Dataset Summary
# ----------------------------------------------------------------------
st.markdown("### Dataset Summary")
type_counts = df["Type"].value_counts()

summary_cols = st.columns(min(len(type_counts) + 1, 6))
summary_cols[0].metric("Total Artifacts", len(df))
remaining_cols = summary_cols[1:]
# Wrap onto additional rows of columns if there are more types than
# fit in one row alongside the "Total" metric.
type_items = list(type_counts.items())
i = 0
first_row_capacity = len(remaining_cols)
for col, (artifact_type, count) in zip(remaining_cols, type_items[:first_row_capacity]):
    col.metric(artifact_type, int(count))
    i += 1

if i < len(type_items):
    extra_cols = st.columns(min(len(type_items) - i, 6))
    for col, (artifact_type, count) in zip(extra_cols, type_items[i:]):
        col.metric(artifact_type, int(count))

st.markdown("---")

# ----------------------------------------------------------------------
# Search & Filter
# ----------------------------------------------------------------------
st.markdown("### Search & Filter")

f1, f2 = st.columns([2, 1])
with f1:
    search_text = st.text_input(
        "Search by ID, summary, or text",
        placeholder="e.g. battery thermal, CR-00123, braking",
    )
with f2:
    type_filter = st.multiselect("Artifact Type", sorted(df["Type"].unique()))

filtered = df.copy()
if type_filter:
    filtered = filtered[filtered["Type"].isin(type_filter)]
if search_text.strip():
    needle = search_text.strip().lower()
    mask = (
        filtered["ID"].str.lower().str.contains(needle, na=False)
        | filtered["Summary"].str.lower().str.contains(needle, na=False)
        | filtered["Text"].str.lower().str.contains(needle, na=False)
    )
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered)} of {len(df)} artifacts.")
st.dataframe(
    filtered[["ID", "Type", "Summary", "State", "Project"]],
    use_container_width=True,
    height=380,
)

st.markdown("---")

# ----------------------------------------------------------------------
# Search dataset before querying -- one click sends a real question,
# with a real ID, straight to the main page.
# ----------------------------------------------------------------------
st.markdown("### 🔍 Search dataset before querying")
st.caption(
    "Pick an artifact below to send a ready-made question to the main "
    "TraceGuard page -- no need to remember or retype the exact ID."
)

if filtered.empty:
    st.warning("No artifacts match your current filters.")
else:
    options = filtered["ID"].tolist()
    picked_id = st.selectbox("Choose an artifact ID from the filtered results", options)
    picked_row = filtered[filtered["ID"] == picked_id].iloc[0]
    st.write(f"**{picked_id}** ({picked_row['Type']}) — {picked_row['Summary']}")

    b1, b2, b3 = st.columns(3)

    def _send_to_main(query_text):
        st.session_state["query_input"] = query_text
        if hasattr(st, "switch_page"):
            st.switch_page("app.py")
        else:
            st.success(
                "Query ready -- open the main TraceGuard AI page from the "
                "sidebar to run it."
            )

    if b1.button("💬 Explain this artifact", use_container_width=True):
        _send_to_main(f"Can you help me understand {picked_id}?")
    if b2.button("🔗 Trace this artifact", use_container_width=True):
        _send_to_main(f"Show traceability for {picked_id}")
    if b3.button("📋 Baseline impact", use_container_width=True):
        _send_to_main(f"Baseline impact of {picked_id}")

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
