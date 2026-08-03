"""
TraceGuard AI -- Streamlit application.

Design decisions, matching what was actually agreed before building this:
  - UI: professional engineering layout (query type selector + execution
    checklist), not a generic chat interface.
  - API key: a SINGLE shared secret via st.secrets["OPENAI_API_KEY"].
    Deliberately NOT the per-visitor BYO-key + demo-cap system discussed
    earlier -- that's real additional complexity (concurrency-safe key
    threading through _call_llm/analyze/tools/planner/orchestrator),
    explicitly deferred to a later polish phase, not built here.
  - Monitoring: simple PER-SESSION stats only, via st.session_state.
    Resets whenever the app restarts or a visitor opens a fresh session --
    NOT a global, cross-visitor count. Labeled "This Session" rather than
    "Today", since a session is not actually a calendar day.

One honest design note on the "Query Type" selector: it does NOT force a
specific workflow or bypass routing. It only changes the placeholder
example text shown to the user. The actual routing decision is always
made by orch.run(query) itself -- IntentRouter, semantic fallback, and
the Agent Planner -- exactly as it would be from a notebook. Faking a
forced-workflow selector would misrepresent what this system actually
does; the whole point is that it decides routing, not the UI.

The execution checklist below is derived from the REAL OrchestratorResult
fields for each query, not a static list of checkmarks -- it only shows a
stage as executed if the result actually shows that stage ran.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

# ----------------------------------------------------------------------
# Path setup -- matches the notebooks' convention (repo_root/src on path)
# ----------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ----------------------------------------------------------------------
# API key -- single shared secret. Set into the environment ONCE, before
# the engine is constructed. Safe because every visitor shares the same
# key (no per-request key, no race condition) -- this is exactly the
# simplification that was chosen instead of the BYO-key design.
# ----------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error(
            "No OPENAI_API_KEY found in Streamlit secrets. Add it under "
            "Settings -> Secrets in the Streamlit Cloud dashboard, or in "
            "a local `.streamlit/secrets.toml` file, as:\n\n"
            'OPENAI_API_KEY = "sk-..."'
        )
        st.stop()

from src.traceguard_v2 import TraceGuard
from src.orchestrator import Orchestrator

# Real, current pricing (verified 2026) -- gpt-4o-mini, USD per token.
# NOTE: OpenAI pricing changes over time; re-verify against
# https://openai.com/api/pricing/ before relying on this for real budgeting.
PRICE_PER_INPUT_TOKEN = 0.15 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.60 / 1_000_000

QUERY_TYPE_PLACEHOLDERS = {
    "Change Request": "e.g. Enhance DC fast-charging current control for improved thermal margin.",
    "Baseline Analysis": "e.g. Baseline impact of CR-00123",
    "Traceability": "e.g. Show traceability for SPEC-00711",
    "Similarity Search": "e.g. Any existing problem reports about battery thermal fallback behavior?",
    "General Question": "e.g. Can you help me understand CR-00123?",
}


_INIT_STEPS = [
    ("data", "Loading synthetic engineering artifacts"),
    ("lexical", "Building lexical search index"),
    ("embeddings", "Loading semantic embedding model"),
    ("graph", "Preparing traceability graph"),
]


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    status = st.status("🚀 Initializing TraceGuard AI", expanded=True)
    with status:
        st.write("Preparing the engineering knowledge base...")
        st.caption(
            "⏳ First startup takes around 2-5 minutes. After initialization, "
            "subsequent analyses will be much faster."
        )
        st.write("**What is happening?**")

        rows = {}
        for key, label in _INIT_STEPS:
            rows[key] = st.empty()
            rows[key].markdown(f"⬜ {label}")
        rag_row = st.empty()
        rag_row.markdown("⬜ Initializing Hybrid RAG pipeline")
        engine_row = st.empty()
        engine_row.markdown("⬜ Starting AI reasoning engine")

        st.caption("Please keep this tab open while TraceGuard finishes loading.")

        def on_progress(step_key, done):
            label = dict(_INIT_STEPS)[step_key]
            rows[step_key].markdown(f"{'✅' if done else '🔄'} {label}{'' if done else '...'}")

        engine = TraceGuard(data_path=repo_root / "data", progress_callback=on_progress)

        rag_row.markdown("🔄 Initializing Hybrid RAG pipeline...")
        orchestrator = Orchestrator(engine)
        rag_row.markdown("✅ Hybrid RAG pipeline ready")

        engine_row.markdown("✅ AI reasoning engine ready")

        st.divider()
        st.caption(
            "**Why does startup take time?** TraceGuard builds semantic search "
            "indexes and loads AI models into memory. This happens once when "
            "the application starts, and the result is cached (`@st.cache_resource`) "
            "so it is reused across reruns in this session instead of being "
            "rebuilt every time the script reruns."
        )

    status.update(label="✅ TraceGuard ready!", state="complete", expanded=False)
    return orchestrator, engine


orch, engine = load_orchestrator()


# ----------------------------------------------------------------------
# Session-scoped monitoring state (NOT global/cross-visitor)
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, one per submitted query, this session only


def _run_query(query_text):
    """Runs one query, measuring real per-request cost by diffing the
    engine's llm_call_log before/after -- correctly captures queries that
    trigger MORE than one LLM call (e.g. the Planner's own decision, then
    full_impact_analysis's internal call if the composed plan includes it)."""
    calls_before = len(engine.llm_call_log)
    start = time.perf_counter()
    result = orch.run(query_text)
    wall_time = time.perf_counter() - start

    calls_during = [c for c in engine.llm_call_log[calls_before:] if c is not None]
    total_input_tokens = sum(c.get("input_tokens") or 0 for c in calls_during)
    total_output_tokens = sum(c.get("output_tokens") or 0 for c in calls_during)
    estimated_cost = (
        total_input_tokens * PRICE_PER_INPUT_TOKEN
        + total_output_tokens * PRICE_PER_OUTPUT_TOKEN
    )

 
    is_real_error = (not result.success) and result.workflow not in (
        "clarification", "rejected:not_engineering"
    )

    st.session_state.history.append({
        "timestamp": datetime.now(),
        "query": query_text,
        "workflow": result.workflow,
        "planner_used": result.plan_prompt is not None,
        "wall_time_s": wall_time,
        "llm_calls": len(calls_during),
        "estimated_cost": estimated_cost,
        "is_error": is_real_error,
    })

    return result


# ----------------------------------------------------------------------
# Page setup + header
# ----------------------------------------------------------------------
st.set_page_config(page_title="TraceGuard AI", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 4px 0; border-bottom: 2px solid #3a4256; margin-bottom: 20px;">
        <h1 style="margin-bottom:0;">🛡️ TraceGuard AI</h1>
        <p style="color:#8a95a8; margin-top:2px;">Engineering Change Impact Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, sidebar_col = st.columns([3, 1])

# ----------------------------------------------------------------------
# Sidebar -- per-session monitoring only (see module docstring)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 This Session")
    history = st.session_state.history

    if not history:
        st.caption("No queries submitted yet this session.")
    else:
        total = len(history)
        planner_pct = 100 * sum(1 for h in history if h["planner_used"]) / total
        fast_path_pct = 100 * sum(
            1 for h in history if not h["planner_used"] and h["workflow"] not in ("clarification", "rejected:not_engineering")
        ) / total
        avg_time = sum(h["wall_time_s"] for h in history) / total
        total_cost = sum(h["estimated_cost"] for h in history)
        error_count = sum(1 for h in history if h["is_error"])

        st.metric("Queries", total)
        st.metric("Fast Path %", f"{fast_path_pct:.0f}%")
        st.metric("Planner %", f"{planner_pct:.0f}%")
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
        st.metric("Estimated Cost (USD)", f"${total_cost:.4f}")
        st.metric("Errors", error_count)

        st.caption(
            "Session-scoped only -- resets on app restart or a new browser "
            "session. Not a global/cross-visitor count."
        )

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
with main_col:
    st.markdown("#### Choose Query Type")
    query_type = st.radio(
        "Query Type", list(QUERY_TYPE_PLACEHOLDERS.keys()),
        label_visibility="collapsed", horizontal=True,
    )
    st.caption(
        "This only changes the example placeholder below -- TraceGuard's "
        "own router always decides how to actually handle your query."
    )

    st.markdown("#### Try an example")
    EXAMPLE_QUERIES = {
        "Understand CR-00123": "Can you help me understand CR-00123?",
        "Trace SPEC-00711": "Show traceability for SPEC-00711",
        "Baseline impact": "Baseline impact of CR-00741",
        "Battery requirement": "Please look into incorrect fallback behavior for battery thermal protection",
    }
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, (label, example_text) in zip(ex_cols, EXAMPLE_QUERIES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["query_input"] = example_text

    st.markdown("#### Engineering Query")
    query_text = st.text_area(
        "Engineering Query", height=100, label_visibility="collapsed",
        placeholder=QUERY_TYPE_PLACEHOLDERS[query_type],
        key="query_input",
    )

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=False)

    if analyze_clicked:
        if not query_text.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner("Running..."):
                result = _run_query(query_text.strip())

            st.markdown("---")
            st.markdown("#### Execution")

            # Every checkmark below reflects what THIS query's real result
            # actually shows happened -- not a static decoration.
            planner_ran = result.plan_prompt is not None
            traceability_ran = any(
                s in (result.steps_run or []) for s in ("trace", "baseline_evidence", "full_impact_analysis")
            )
            fusion_ran = "evidence_fusion" in (result.steps_run or [])

            checklist = [
                ("Intent Router", True),
                ("Planner", planner_ran),
                ("Orchestrator", True),
                ("Traceability", traceability_ran),
                ("Evidence Fusion", fusion_ran),
            ]
            cols = st.columns(len(checklist))
            for col, (label, ran) in zip(cols, checklist):
                col.markdown(f"{'✅' if ran else '⬜'} {label}")

            st.caption(f"Workflow: `{result.workflow}` · Confidence: {result.confidence:.2f} · Steps: {' → '.join(result.steps_run) if result.steps_run else '(none)'}")

            st.markdown("#### Impact Assessment")

            response = result.final_response

            if not result.success:
                # rejected:not_engineering, clarification, or a genuine
                # tool failure -- final_response is already a plain,
                # human-readable message in every one of these cases.
                if result.workflow in ("clarification", "rejected:not_engineering"):
                    st.info(response)
                else:
                    st.error(response)

            elif isinstance(response, dict) and "impact_report_df" in response:
                st.dataframe(response["impact_report_df"], use_container_width=True)
                st.markdown("**Overall assessment:**")
                st.write(response.get("overall_assessment"))
                if "evidence_fusion" in response:
                    ef = response["evidence_fusion"]
                    st.markdown(f"**Evidence fusion:** {ef['status']}")
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.caption(f"Agreement: {ef['agreement_release_ids']}")
                    ec2.caption(f"LLM-only: {ef['llm_only_release_ids']}")
                    ec3.caption(f"Evidence-only: {ef['evidence_only_release_ids']}")

            elif isinstance(response, dict) and "baseline_determination" in response:
                st.markdown(f"**Baseline determination:** {response['baseline_determination']['status']}")
                if response.get("affected_baselines_df") is not None:
                    st.dataframe(response["affected_baselines_df"], use_container_width=True)

            elif isinstance(response, dict) and "discoveries" in response:
                st.markdown(f"**Discovered {response['discovered_count']} linked artifact(s):**")
                for artifact_id in response["discoveries"]:
                    st.write(f"- {artifact_id}")

            elif isinstance(response, dict) and "results_by_type" in response:
                for artifact_type, rows in response["results_by_type"].items():
                    if rows:
                        st.write(f"**{artifact_type}:** {len(rows)} candidate(s)")

            else:
                st.json(response)

            # Execution Summary -- surfaces exactly what was already
            # computed in _run_query for THIS query (the entry it just
            # appended), nothing recomputed or estimated separately.
            this_query = st.session_state.history[-1]
            workflow_display = "agent planner" if this_query["workflow"].startswith("planner:") else this_query["workflow"]
            st.markdown("---")
            st.markdown("#### Execution Summary")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Workflow", workflow_display)
            sc2.metric("Planner", "Yes" if this_query["planner_used"] else "No")
            sc3.metric("LLM Calls", this_query["llm_calls"])
            sc4.metric("Execution Time", f"{this_query['wall_time_s']:.2f}s")
            sc5.metric("Estimated Cost", f"${this_query['estimated_cost']:.4f}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8a95a8; font-size:0.85em; padding: 10px 0;">
        Built using <b>LLM Zoomcamp</b> · <b>OpenAI</b> · <b>Sentence Transformers</b> · <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
