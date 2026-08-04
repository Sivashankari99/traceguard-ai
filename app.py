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
  - Feedback: a single shared CSV log (src/feedback_store.py) written to
    by both the inline Helpful/Not Helpful buttons below and the
    dedicated Feedback page. Streamlit Cloud's filesystem is ephemeral --
    this is a rolling log meant to be periodically exported, not a
    durable database. Documented honestly on the Feedback page itself.

One honest design note on the "Query Type" selector: it does NOT force a
specific workflow or bypass routing. It only changes the placeholder
example text shown to the user. The actual routing decision is always
made by orch.run(query) itself -- IntentRouter, semantic fallback, and
the Agent Planner -- exactly as it would be from a notebook. Faking a
forced-workflow selector would misrepresent what this system actually
does; the whole point is that it decides routing, not the UI.

The execution/workflow visualization below is derived from the REAL
OrchestratorResult fields for each query, not a static list of
checkmarks -- it only shows a stage as executed if the result actually
shows that stage ran. Same principle for the Planner Reasoning expander:
it only appears when a plan actually ran, and shows the Planner's real
reasoning text, not decorative filler.
"""

import streamlit as st


st.set_page_config(page_title="TraceGuard AI", page_icon="🛡️", layout="wide")

_boot = st.empty()
_boot.markdown("🛡️ **Starting TraceGuard AI...** loading application code, please wait a moment.")

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Path setup -- matches the notebooks' convention (repo_root/src on path)
# ----------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        _boot.empty()
        st.error(
            "No OPENAI_API_KEY found in Streamlit secrets. Add it under "
            "Settings -> Secrets in the Streamlit Cloud dashboard, or in "
            "a local `.streamlit/secrets.toml` file, as:\n\n"
            'OPENAI_API_KEY = "sk-..."'
        )
        st.stop()

from src.traceguard_v2 import TraceGuard
from src.orchestrator import Orchestrator
from src.feedback_store import append_rating
from src import engine_status

_boot.empty()  # clear the boot message now that the slow imports are done

# Real, current pricing (verified 2026) -- gpt-4o-mini, USD per token.
# NOTE: OpenAI pricing changes over time; re-verify against
# https://openai.com/api/pricing/ before relying on this for real budgeting.
PRICE_PER_INPUT_TOKEN = 0.15 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.60 / 1_000_000

QUERY_TYPE_PLACEHOLDERS = {
    "Change Request": "e.g. Enhance DC fast-charging current control for improved thermal margin.",
    "Baseline Analysis": "e.g. Baseline impact of CR-00123 -- or ask about any release.",
    "Traceability": "e.g. Show traceability for SPEC-00711, or trace REQ-00066.",
    "Similarity Search": "e.g. Find similar problem reports about battery thermal fallback behavior.",
    "General Question": "e.g. Can you help me understand CR-00123? or What requirements are related to battery charging?",
}

# All artifact IDs below are real IDs in the synthetic dataset -- these
# aren't just illustrative text, clicking them returns real results.
EXAMPLE_QUERIES = {
    "Understand CR-00123": "Can you help me understand CR-00123?",
    "Explain SPEC-00711": "Can you help me understand SPEC-00711?",
    "Trace REQ-00066": "Show traceability for REQ-00066",
    "Baseline impact of CR-00123": "Baseline impact of CR-00123",
    "Explain braking requirements": "Explain braking requirements",
    "Related to battery charging": "What requirements are related to battery charging?",
    "Similar battery requirements": "Find similar existing requirements about battery charging behavior",
    "Similar problem reports": "Find similar existing problem reports about battery thermal fallback behavior",
}

# tool name -> human label, used by the workflow visualization further down.
_STEP_LABELS = {
    "lookup": "Looked up the artifact",
    "retrieve": "Retrieved similar artifacts",
    "trace": "Expanded traceability",
    "assess_impact": "Assessed impact (LLM)",
    "validate": "Validated grounding",
    "determine_baseline": "Determined baseline (LLM)",
    "baseline_evidence": "Checked baseline evidence (traceability)",
    "evidence_fusion": "Compared evidence signals",
    "full_impact_analysis": "Ran full impact analysis",
}

_WORKFLOW_TITLES = {
    "direct_lookup": "Artifact Lookup",
    "similarity_check": "Similarity Search",
    "traceability_trace": "Traceability Trace",
    "baseline_check": "Baseline Check (traceability-only)",
    "full_impact_analysis": "Full Impact Analysis",
    "clarification": "Needs Clarification",
    "rejected:not_engineering": "Outside Engineering Domain",
}

# tool/stage name -> label shown on the LIVE per-query progress status
# (item 7). Built from the same real orchestrator.run() callback that
# drives steps_run -- a stage only ever appears because it genuinely ran.
_QUERY_STAGE_LABELS = {
    "route": "🔍 Understanding your question",
    "planner": "🧭 Composing a plan (Agent Planner)",
    "lookup": "📄 Looking up the artifact",
    "retrieve": "📄 Searching engineering artifacts",
    "trace": "🔗 Expanding traceability",
    "assess_impact": "🤖 Assessing impact (LLM)",
    "validate": "🤖 Validating grounding",
    "determine_baseline": "🤖 Determining baseline (LLM)",
    "baseline_evidence": "🔗 Checking baseline evidence",
    "evidence_fusion": "⚖️ Comparing evidence signals",
    "full_impact_analysis": "🤖 Running full impact analysis",
    "final": "🤖 Generating grounded response",
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
        st.info(
            "💡 While you wait, you can open **About** or **Dataset Explorer** "
            "from the sidebar -- neither needs the engine, so they load instantly."
        )

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
            "so it is reused across reruns and across visitors instead of being "
            "rebuilt every time the script reruns."
        )

    status.update(label="✅ TraceGuard ready!", state="complete", expanded=False)
    engine_status.mark_ready()
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
    full_impact_analysis's internal call if the composed plan includes it).

    The status box below shows REAL stage-by-stage progress via
    Orchestrator.run()'s step_callback -- a stage only ever appears
    because it genuinely ran, same principle as the init loading screen
    and the workflow visualization further down."""
    calls_before = len(engine.llm_call_log)
    start = time.perf_counter()

    status = st.status("Running your query...", expanded=True)
    rows = {}
    with status:
        def on_stage(key, done):
            label = _QUERY_STAGE_LABELS.get(key, key.replace("_", " ").title())
            if key not in rows:
                rows[key] = st.empty()
            rows[key].markdown(f"{'✅' if done else '🔄'} {label}{'' if done else '...'}")

        result = orch.run(query_text, step_callback=on_stage)

    wall_time = time.perf_counter() - start

    if result.success:
        status.update(label="✅ Analysis complete", state="complete", expanded=False)
    elif result.workflow in ("clarification", "rejected:not_engineering"):
        status.update(label="ℹ️ Query answered", state="complete", expanded=False)
    else:
        status.update(label="⚠️ A step failed", state="error", expanded=False)

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
        "entity_id": result.entity_id,
        "planner_used": result.plan_prompt is not None,
        "wall_time_s": wall_time,
        "llm_calls": len(calls_during),
        "estimated_cost": estimated_cost,
        "is_error": is_real_error,
    })

    return result


def _executive_summary_line(result):
    response = result.final_response
    if not result.success:
        if result.workflow == "rejected:not_engineering":
            return "🚫 **Declined** -- this wasn't recognized as an engineering-artifact question."
        if result.workflow == "clarification":
            return "❓ **Needs clarification** before TraceGuard can proceed."
        return "⚠️ **A step in the pipeline failed.**"

    if isinstance(response, dict) and "impact_report_df" in response:
        n = len(response["impact_report_df"])
        ef = response.get("evidence_fusion")
        if ef:
            n_releases = (
                len(ef.get("agreement_release_ids", []))
                + len(ef.get("llm_only_release_ids", []))
                + len(ef.get("evidence_only_release_ids", []))
            )
            return (
                f"✅ **Full impact analysis complete** -- {n} artifact(s) assessed, "
                f"{n_releases} release(s) flagged for review."
            )
        return f"✅ **Full impact analysis complete** -- {n} artifact(s) assessed."

    if isinstance(response, dict) and "baseline_determination" in response:
        status = (response["baseline_determination"] or {}).get("status", "Unknown")
        return f"✅ **Baseline check complete** -- status: {status}."

    if isinstance(response, dict) and "discoveries" in response and "discovered_count" in response:
        return f"✅ **Traceability trace complete** -- {response['discovered_count']} linked artifact(s) found."

    if isinstance(response, dict) and "results_by_type" in response:
        total = sum(len(v) for v in response["results_by_type"].values())
        return f"✅ **Similarity search complete** -- {total} candidate artifact(s) found."

    if isinstance(response, dict) and "ID" in response and "Summary" in response:
        return f"✅ **Found {response['ID']}** ({response.get('Type', 'artifact')})."

    return "✅ **Query completed.**"


def _render_workflow_steps(result):
    """The sequential, ChatGPT-style progress line (item 8) -- built
    entirely from result.steps_run, never a fixed decorative list."""
    lines = ["✅ Understood query"]
    if result.workflow == "rejected:not_engineering":
        lines.append("⛔ Declined -- outside engineering domain")
    elif result.workflow == "clarification":
        lines.append("❓ Needs clarification")
    else:
        for step in (result.steps_run or []):
            lines.append(f"✅ {_STEP_LABELS.get(step, step)}")
        lines.append("✅ Generated grounded answer" if result.success else "⛔ Step failed")
    st.markdown("  →  ".join(lines))


def _step_data_map(result):
    """Rebuilds a {tool_name: data} lookup from the real trace_log --
    this is the same information the orchestrator's internal `context`
    dict held while running, so it works identically for every
    workflow (including baseline_check, which runs `trace` internally
    without surfacing it in final_response) and for planner-composed
    plans alike."""
    return {r.tool: r.data for r in (result.trace_log or []) if r.ok}


def _retrieved_artifact_ids(result):
    """Pulls the real set of retrieved/candidate artifact IDs out of
    whichever tools actually ran for this query."""
    step_data = _step_data_map(result)
    if "full_impact_analysis" in step_data:
        df = step_data["full_impact_analysis"].get("selected_candidates_df")
        if df is not None and not df.empty and "ID" in df.columns:
            return df["ID"].astype(str).tolist()
    if "retrieve" in step_data:
        ids = []
        for rows in step_data["retrieve"].get("results_by_type", {}).values():
            ids.extend(str(r.get("ID")) for r in rows if r.get("ID"))
        return ids
    return []


def _traceability_discoveries(result):
    """Pulls the real discoveries dict out of whichever tool produced
    it (`trace` for the single-artifact workflows, `full_impact_analysis`
    for the free-text pipeline), regardless of what final_response
    ended up wrapping."""
    step_data = _step_data_map(result)
    if "trace" in step_data:
        return step_data["trace"].get("discoveries", {})
    if "full_impact_analysis" in step_data:
        return step_data["full_impact_analysis"].get("discoveries", {})
    return {}


def _render_id_chips(ids, empty_message="None found."):
    if not ids:
        st.caption(empty_message)
        return
    shown, extra = ids[:30], max(0, len(ids) - 30)
    st.markdown(" &nbsp; ".join(f"`{i}`" for i in shown), unsafe_allow_html=True)
    if extra:
        st.caption(f"+ {extra} more")


def _render_answer_body(result):
    """The main 'Answer' section -- dispatches on the real shape of
    final_response, covering every workflow's actual output (including
    a plain lookup record, which previously fell through to raw JSON)."""
    response = result.final_response

    if not result.success:
        if result.workflow in ("clarification", "rejected:not_engineering"):
            st.info(response)
        else:
            st.error(response)
        return

    if isinstance(response, dict) and "impact_report_df" in response:
        report_df = response["impact_report_df"]
        summary_cols = [c for c in ["artifact_id", "artifact_type", "impact_level", "confidence", "traceability_status"] if c in report_df.columns]
        st.dataframe(report_df[summary_cols], width="stretch")

        st.markdown("**Reasoning per artifact:**")
        for _, row in report_df.iterrows():
            level = row.get("impact_level", "")
            with st.expander(f"{row.get('artifact_id', '')} -- {row.get('artifact_type', '')} ({level})"):
                st.write(row.get("reason", "_No reason provided._"))
                st.caption(
                    f"Candidate category: {row.get('candidate_category', '-')} · "
                    f"Traceability: {row.get('traceability_status', '-')} · "
                    f"Confidence: {row.get('confidence', '-')}"
                )

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
            st.dataframe(response["affected_baselines_df"], width="stretch")

    elif isinstance(response, dict) and "discoveries" in response and "discovered_count" in response:
        st.markdown(f"Discovered **{response['discovered_count']}** linked artifact(s). See *Traceability* below for details.")

    elif isinstance(response, dict) and "results_by_type" in response:
        for artifact_type, rows in response["results_by_type"].items():
            if rows:
                st.write(f"**{artifact_type}:** {len(rows)} candidate(s)")

    elif isinstance(response, dict) and "ID" in response and ("Summary" in response or "Text" in response):
        st.markdown(f"### {response['ID']} -- {response.get('Type', '')}")
        summary = str(response.get("Summary") or "").strip()
        text = str(response.get("Text") or "").strip()
        main_content = summary or text or "_No description available for this artifact._"
        st.write(main_content)
        secondary = text if main_content == summary else (summary if main_content == text else None)
        if secondary:
            with st.expander("Additional detail"):
                st.write(secondary)
        meta_cols = st.columns(3)
        meta_cols[0].caption(f"**State:** {response.get('State', '-')}")
        meta_cols[1].caption(f"**Project:** {response.get('Project', '-')}")
        meta_cols[2].caption(f"**Document ID:** {response.get('Document_ID', '-')}")

    else:
        st.json(response)


def _render_traceability(discoveries):
    if not discoveries:
        st.caption("No linked artifacts were discovered for this query.")
        return
    for artifact_id, paths in list(discoveries.items())[:30]:
        hops = min((p.get("distance", 0) for p in paths), default=None)
        relationships = sorted({e["relationship"] for p in paths for e in p.get("edges", [])})
        rel_text = ", ".join(relationships) if relationships else "linked"
        hop_text = f"{hops} hop(s)" if hops is not None else ""
        st.write(f"- `{artifact_id}` -- {rel_text} ({hop_text})")
    if len(discoveries) > 30:
        st.caption(f"+ {len(discoveries) - 30} more linked artifact(s)")


def _render_planner_reasoning(result):
    if result.plan_prompt is None:
        return  # no planner run for this query -- nothing to show
    with st.expander("🧠 Planner Reasoning -- why was this workflow selected?", expanded=False):
        st.markdown(result.plan_reasoning or "_No reasoning text was returned._")
        st.caption(
            "Planner-composed steps: "
            + (" → ".join(result.steps_run) if result.steps_run else "(none)")
        )
        with st.expander("Show raw planner debug info (prompt + raw LLM response)"):
            st.markdown("**Prompt sent to the LLM:**")
            st.code(result.plan_prompt or "", language="text")
            st.markdown("**Raw parsed LLM response:**")
            st.json(result.plan_raw_response or {})


def _suggested_followups(result, current_query):
    """Contextual follow-up chips (item 12) -- built from the real
    entity_id this query resolved to, not a fixed generic list."""
    suggestions = []
    eid = result.entity_id
    if eid:
        suggestions.append((f"🔗 Trace {eid}", f"Show traceability for {eid}"))
        suggestions.append((f"📋 Baseline impact of {eid}", f"Baseline impact of {eid}"))
        suggestions.append((f"💬 Explain {eid}", f"Can you help me understand {eid}?"))
    suggestions.append(("🔍 Similar problem reports", "Find similar existing problem reports about battery thermal fallback behavior"))
    suggestions.append(("📐 Related requirements", "What requirements are related to battery charging?"))

    seen, final = set(), []
    for label, text in suggestions:
        if text == current_query or text in seen:
            continue
        seen.add(text)
        final.append((label, text))
    return final[:4]


def _section_card(icon, title, accent_color):
    """A colored, icon-labeled header + bordered card body, used to
    visually separate the result page into distinct sections (item 2)
    instead of one long undifferentiated block of output. Returns the
    container to render section content inside."""
    st.markdown(
        f"""
        <div style="border-left: 4px solid {accent_color}; padding: 2px 0 2px 12px; margin: 20px 0 6px 0;">
            <span style="font-size:1.05em; font-weight:600;">{icon} {title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.container(border=True)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 4px 0; border-bottom: 2px solid #3a4256; margin-bottom: 20px;">
        <h1 style="margin-bottom:0;">🛡️ TraceGuard AI</h1>
        <p style="color:#8a95a8; margin-top:2px;">Engineering Change Impact Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='font-size:1.1em; font-weight:500; color:#c9d1d9; margin:4px 0 18px 0;'>"
    "Ask engineering questions in natural language, and TraceGuard AI finds "
    "related artifacts, expands traceability, and explains the engineering impact."
    "</p>",
    unsafe_allow_html=True,
)

_artifact_count = len(engine.artifacts_df)
_type_count = engine.artifacts_df["Type"].nunique()
stat1, stat2, stat3 = st.columns(3)
stat1.metric("Artifacts", f"{_artifact_count:,}")
stat2.metric("Artifact Types", _type_count)
with stat3:
    st.caption("Approach")
    st.markdown("**Hybrid RAG + Traceability**")

st.markdown("#### TraceGuard AI can help you")
st.markdown(
    """
- ✅ Understand a Change Request
- ✅ Find similar Requirements
- ✅ Explore Traceability
- ✅ Estimate Baseline Impact
"""
)

st.info(
    "📘 **About this demo** -- This application is built on approximately "
    "**5,000 synthetic engineering artifacts**, created for educational "
    "purposes. It is fictional and **not connected to a live engineering "
    "repository.**"
)
st.page_link("pages/2_Dataset_Explorer.py", label="📄 Browse the dataset →")

if not st.session_state.history:
    st.markdown(
        "👋 **New here?** Start with one of the example questions below, "
        "or browse the Dataset Explorer to see what TraceGuard already knows."
    )

main_col, sidebar_col = st.columns([3, 1])

# ----------------------------------------------------------------------
# Sidebar -- per-session monitoring, query history, and page navigation
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

    st.markdown("---")
    st.markdown("### 🕘 Recent Queries")
    if not history:
        st.caption("Your query history will appear here.")
    else:
        for h in reversed(history[-8:]):
            title = _WORKFLOW_TITLES.get(h["workflow"], h["workflow"])
            with st.expander(h["query"][:60] + ("..." if len(h["query"]) > 60 else "")):
                st.caption(f"{h['timestamp'].strftime('%H:%M:%S')} · {title}")
                if st.button("↺ Ask again", key=f"rerun_{h['timestamp'].isoformat()}"):
                    st.session_state["query_input"] = h["query"]

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
with main_col:
    st.markdown("#### 💬 What can I ask?")
    with st.expander("See example questions", expanded=False):
        st.markdown(
            """
- Understand CR-00123
- Explain SPEC-00711
- Trace REQ-00066
- Find similar battery requirements
- Show baseline impact of CR-00123
- Explain braking requirements
- What requirements are related to battery charging?
- Find similar problem reports
"""
        )

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
    ex_items = list(EXAMPLE_QUERIES.items())
    for row_start in range(0, len(ex_items), 4):
        row_items = ex_items[row_start:row_start + 4]
        ex_cols = st.columns(len(row_items))
        for col, (label, example_text) in zip(ex_cols, row_items):
            if col.button(label, width="stretch", key=f"ex_{label}"):
                st.session_state["query_input"] = example_text

    st.markdown("#### Your Question")
    query_text = st.text_area(
        "Your Question", height=100, label_visibility="collapsed",
        placeholder=QUERY_TYPE_PLACEHOLDERS[query_type],
        key="query_input",
    )

    auto_analyze = st.session_state.pop("auto_analyze", False)
    analyze_clicked = st.button("Analyze", type="primary", width="content") or auto_analyze

    if analyze_clicked:
        if not query_text.strip():
            st.warning("Enter a query first.")
        else:
            result = _run_query(query_text.strip())
            # Persisted in session_state rather than a local variable --
            # see the render block below for why this matters.
            st.session_state["last_result"] = result
            st.session_state["last_query_text"] = query_text.strip()


    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        result_query_text = st.session_state.get("last_query_text", "")
        response = result.final_response

        # --- Executive Summary -----------------------------------
        with _section_card("📋", "Executive Summary", "#3b82f6"):
            st.markdown(_executive_summary_line(result))
            _render_workflow_steps(result)
            st.caption(
                f"Workflow: `{result.workflow}` · Confidence: {result.confidence:.2f} · "
                f"Steps: {' → '.join(result.steps_run) if result.steps_run else '(none)'}"
            )

        # --- Answer ------------------------------------------------
        with _section_card("💬", "Answer", "#22c55e"):
            _render_answer_body(result)

        # --- Retrieved Artifacts ------------------------------------
        retrieved_ids = _retrieved_artifact_ids(result)
        if retrieved_ids:
            with _section_card("📦", "Retrieved Artifacts", "#a855f7"):
                _render_id_chips(retrieved_ids)

        # --- Traceability -------------------------------------------
        discoveries = _traceability_discoveries(result)
        if discoveries:
            with _section_card("🔗", "Traceability", "#f97316"):
                _render_traceability(discoveries)

        # --- Planner Reasoning ---------------------------------------
        if result.plan_prompt is not None:
            with _section_card("🧠", "Planner Reasoning", "#14b8a6"):
                _render_planner_reasoning(result)

        # --- Metrics ---------------------------------------------------
        this_query = st.session_state.history[-1]
        workflow_display = _WORKFLOW_TITLES.get(
            result.workflow,
            "Agent Planner" if this_query["workflow"].startswith("planner:") else this_query["workflow"],
        )
        with _section_card("📊", "Metrics", "#64748b"):
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Workflow", workflow_display)
            sc2.metric("Planner", "Yes" if this_query["planner_used"] else "No")
            sc3.metric("LLM Calls", this_query["llm_calls"])
            sc4.metric("Execution Time", f"{this_query['wall_time_s']:.2f}s")
            sc5.metric("Estimated Cost", f"${this_query['estimated_cost']:.4f}")

        # --- Suggested follow-up queries --------------------------------
        followups = _suggested_followups(result, result_query_text)
        if followups:
            st.markdown("#### Suggested Follow-ups")
            fc = st.columns(len(followups))
            for col, (label, followup_text) in zip(fc, followups):
                if col.button(label, width="stretch", key=f"followup_{label}"):
                    st.session_state["query_input"] = followup_text

        # --- Helpful / Not Helpful ---------------------------------------
        st.markdown("---")
        st.markdown("**Was this helpful?**")
        fb_key = f"fb_{len(st.session_state.history)}"
        hc1, hc2, hc3 = st.columns([1, 1, 4])
        if hc1.button("👍 Helpful", key=f"{fb_key}_up"):
            append_rating(
                repo_root, query=result_query_text, workflow=result.workflow,
                type_="helpful",
            )
            st.success("Thanks for the feedback!")
        if hc2.button("👎 Not Helpful", key=f"{fb_key}_down"):
            st.session_state[f"{fb_key}_show_reason"] = True
        if st.session_state.get(f"{fb_key}_show_reason"):
            reason = st.selectbox(
                "What went wrong?",
                ["Wrong answer", "Didn't understand", "Missing data", "Slow", "Other"],
                key=f"{fb_key}_reason",
            )
            detail_label = (
                "What went wrong? (required for \"Other\")" if reason == "Other"
                else "Add more detail (optional)"
            )
            detail = st.text_area(detail_label, key=f"{fb_key}_detail", height=80)
            if st.button("Submit", key=f"{fb_key}_reason_submit"):
                if reason == "Other" and not detail.strip():
                    st.warning('Please add a bit of detail -- "Other" alone doesn\'t say what went wrong.')
                else:
                    append_rating(
                        repo_root, query=result_query_text, workflow=result.workflow,
                        type_="not_helpful", rating_reason=reason, message=detail.strip(),
                    )
                    st.success("Thanks -- this helps us improve TraceGuard.")
                    st.session_state[f"{fb_key}_show_reason"] = False

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
