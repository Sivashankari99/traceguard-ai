import streamlit as st

st.set_page_config(page_title="About - TraceGuard AI", page_icon="🛡️", layout="wide")

st.markdown("# 🛡️ About TraceGuard AI")
st.caption("Agentic, Traceability-Aware Engineering Change Impact Analysis")

tab_how, tab_arch, tab_data, tab_limits = st.tabs(
    ["How It Works", "Architecture", "Dataset", "Limitations"]
)

with tab_how:
    st.markdown(
        """
Given a proposed engineering change — or a question about a known artifact,
release, or baseline — TraceGuard routes the request through the cheapest
mechanism capable of answering it correctly, rather than always running the
full pipeline:

1. **Fixed keyword rules** — instant, no model call, for the most common
   query shapes.
2. **Semantic fallback** — embedding similarity against real domain
   exemplars, for phrasing the keyword rules miss.
3. **Agent Planner** — an LLM composes a fresh tool sequence for anything
   the first two layers can't confidently route, or explicitly declines the
   request if it's outside the engineering domain entirely.
4. **Full impact analysis** — hybrid retrieval, traceability expansion, LLM
   assessment, and grounding validation, only when genuinely needed.

The objective is to assist engineers by providing structured, explainable
impact analysis while keeping the final engineering decision with the
human reviewer.
"""
    )

with tab_arch:
    st.markdown("### Request Pipeline")
    st.code(
        """
User Query
     |
Entity Extraction
     |
Lexical Intent Rules --- high confidence? --- yes --> Run fixed workflow
     |
    no
     |
Semantic Fallback --- high confidence? --- yes --> Run fixed workflow
     |
    no
     |
Agent Planner
     |
  +--+------------------+
  |                     |
plan          clarification / not_engineering
  |                     |
Orchestrator      Ask user / Decline
executes steps    (zero tool calls)
""",
        language="text",
    )
    st.markdown("### Traceability Model")
    st.markdown(
        """
| Relationship | Meaning |
|---|---|
| `Release --Covers--> CR/PR` | Release membership |
| `Release --Spawns--> Release` | Release hierarchy |
| `ALM Requirement / Specification / Input --Spawns--> CR/PR` | Upstream originator |
| `CR/PR --Spawns--> CR/PR / Task` | Downstream follow-up |
| `ALM Test Case / Test Suite --Validates--> ALM Requirement / Specification` | Verification coverage |

Verified empirically against the real dataset, not assumed.
"""
    )
    st.markdown(
        "Full architecture diagram and detailed write-up: "
        "[project README](https://github.com/Sivashankari99/traceguard-ai)"
    )

with tab_data:
    st.markdown(
        """
This project uses **entirely synthetic automotive engineering data**,
created specifically for educational, experimentation, and portfolio
purposes.

No proprietary, confidential, employer-specific, customer-specific, or
real-world organizational engineering data is used anywhere in this
project. Artifact types include Change Requests, Problem Reports, ALM
Requirements, Specifications, Test Cases, Test Suites, Inputs, Tasks, and
Releases.
"""
    )

with tab_limits:
    st.markdown(
        """
Documented honestly rather than hidden:

- **Semantic fallback rarely fires in practice.** Its confidence threshold
  was calibrated against real data; genuinely relevant queries only ever
  scored 0.31–0.37 against workflow exemplars, so at the calibrated
  threshold this layer mostly defers to the Agent Planner.
- **The Planner's relevance check is not perfect.** Most irrelevant queries
  are correctly declined at no cost, but some plausible-sounding-but-
  irrelevant phrasings can still slip through and trigger a real, paid
  impact analysis.
- **Exemplar/workflow boundaries are still somewhat fuzzy** for
  near-identical traceability-flavored phrasings.
- **Retrieval evaluation** (Recall@K, Precision@K) is not yet implemented.
- **This demo uses a single shared API key** with no per-visitor cost cap
  yet — a known, accepted tradeoff for this stage, not a solved problem.

AI-generated impact assessments are intended to support human engineering
analysis and should **not** be considered authoritative engineering,
safety, configuration management, release, quality, or compliance
decisions. All results require appropriate human engineering review.
"""
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#8a95a8; font-size:0.85em; padding: 10px 0;">
        Built using <b>LLM Zoomcamp</b> · <b>OpenAI</b> · <b>Sentence Transformers</b> · <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
