# TraceGuard AI

### Agentic AI for Engineering Change Impact Analysis

TraceGuard AI is an AI-assisted engineering change impact analysis system that explores how **Retrieval-Augmented Generation (RAG), semantic search, engineering traceability, LLM reasoning, and lightweight agentic orchestration** can support impact analysis in automotive engineering environments.

Given a proposed engineering change — or a question about a known artifact, release, or baseline — TraceGuard routes the request through the cheapest mechanism capable of answering it correctly: fixed keyword rules, semantic matching, a composed agent plan, or (only when genuinely needed) a full retrieval + traceability + LLM assessment pipeline.

The objective is to assist engineers by providing structured, explainable impact analysis while keeping the final engineering decision with the human reviewer.

---

## 🚀 Live Demo

**[traceguard-ai.streamlit.app](https://traceguard-ai.streamlit.app/)**

The app is deployed on Streamlit Community Cloud. First load may take a few minutes while the engine builds its embeddings and indexes (see *Known Limitations* for cost-exposure notes on public deployments).

---

## 📚 Table of Contents

- [Live Demo](#-live-demo)
- [Project Goals](#project-goals)
- [Current Capabilities](#current-capabilities)
- [Architecture Overview](#architecture-overview)
- [Request Pipeline](#request-pipeline)
- [Traceability Model](#traceability-model)
- [Workflows](#workflows)
- [Agent Planner](#agent-planner)
- [Project Structure](#project-structure)
- [Example Usage](#example-usage)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Monitoring & Feedback](#monitoring--feedback)
- [Current Status](#current-status)
- [Known Limitations](#known-limitations)
- [Planned Development](#planned-development)
- [Design Principle](#design-principle)
- [Disclaimer](#disclaimer)

---

<a id="architecture-overview"></a>

## 🗺️ Architecture Overview

![TraceGuard AI Architecture](./docs/architecture.png)

---

<a id="project-goals"></a>

## 🚀 Project Goals

- Identify potentially impacted engineering artifacts from a proposed change.
- Discover related Change Requests and Problem Reports.
- Identify potentially affected requirements, specifications, tests, and tasks.
- Use existing traceability relationships as additional engineering evidence, independent of textual similarity.
- Identify potential release and baseline impacts — from two independent signals (LLM judgment and raw traceability), compared rather than silently merged.
- Route each request through the least expensive mechanism that can answer it correctly, rather than always running the full pipeline.
- Handle requests that don't fit any fixed workflow by composing a plan from atomic tools, instead of defaulting to "I don't understand."
- Reduce the effort required to manually explore large engineering artifact repositories.
- Support human engineering and configuration management review rather than replace it.

---

<a id="current-capabilities"></a>

## 🧠 Current Capabilities

- **Lexical retrieval** using MinSearch.
- **Semantic retrieval** using Sentence Transformers, fused with lexical results via Reciprocal Rank Fusion (RRF) — not a manual blend of two differently-scaled raw scores.
- **Artifact-type-aware retrieval** across all engineering artifact categories, with full per-type semantic scoring preserved (not discarded after Top-K selection) so downstream evidence checks can look up any artifact's real similarity, not just retained candidates.
- **Typed, direction-aware traceability expansion** — Release membership, upstream originator, downstream follow-up, and test-validation links are each distinct, verified relationship types (see *Traceability Model* below), not a single undifferentiated graph walk.
- **Rank-based traceability seeding** — which retrieved candidates earn a traceability expansion is decided by fused rank position, not a fixed similarity number (an earlier fixed-threshold approach was tested against real queries and found unreliable — see *Known Limitations*).
- **Dual, independent baseline/release determination** — one signal from LLM judgment, one from raw traceability data alone (zero LLM cost) — compared via evidence fusion rather than merged silently.
- **A 3-layer request pipeline**: fixed lexical rules → semantic fallback → an LLM-based Agent Planner that composes a fresh tool sequence for anything the first two layers can't confidently route (see *Agentic Orchestration* below).
- **LLM-assisted engineering impact assessment**, grounded strictly in supplied evidence — instructed not to invent artifact IDs, relationships, or traceability paths.
- **A genuine three-way relevance decision** (not just accept/reject): the system can build a plan, ask a real clarifying question, or explicitly decline a request as outside the engineering domain — each with a distinct, honest response, not one generic fallback message.
- **Grounding validation** to detect unsupported artifact or relationship claims in LLM output.
- **Structured impact reports** containing impact level, confidence, traceability status, candidate category, and reasoning per artifact.

---

<a id="request-pipeline"></a>

## 🧭 Request Pipeline

```text
                        User Query
                             │
                             ▼
                    Entity Extraction
               (artifact ID, if present)
                             │
                             ▼
                  Lexical Intent Rules
              (keyword + entity patterns)
                             │
                  High confidence? ──Yes──▶ Run the matching
                             │              fixed workflow
                             No
                             ▼
                   Semantic Fallback
          (embedding similarity vs. workflow
                exemplars, real domain text)
                             │
                  High confidence? ──Yes──▶ Run the matching
                             │              fixed workflow
                             No
                             ▼
                     Agent Planner
        (LLM composes a fresh tool sequence,
           or explicitly declines to)
                             │
              ┌──────────────┼──────────────┐
              ▼               ▼              ▼
        Valid plan      Needs a real     Not an engineering
              │          clarifying         request at all
              ▼            question             │
        Orchestrator          │                 ▼
        executes steps        ▼            Declined, no
              │           Ask the           tool call runs
              ▼            user
        Structured result
```

Every layer after entity extraction is a genuine decision point, not a rubber stamp — a query only reaches the Planner if both cheaper layers were honestly uncertain, and the Planner itself can refuse rather than force a plan onto irrelevant input.

---

<a id="traceability-model"></a>

## 🔗 Traceability Model

Verified empirically against the actual dataset (not assumed) after an earlier version of the data generator had two relationships backwards:

| Relationship | Meaning |
|---|---|
| `Release --Covers--> CR/PR` | Release membership |
| `Release --Spawns--> Release` | Release hierarchy (not traversed from a CR/PR seed) |
| `ALM Requirement / Specification / Input --Spawns--> CR/PR` | Upstream originator of a change |
| `CR/PR --Spawns--> CR/PR / Task` | Downstream follow-up work |
| `ALM Test Case / Test Suite --Validates--> ALM Requirement / Specification` | Verification coverage |

Traceability expansion from a known artifact follows these rules:

1. **Release membership is recorded once, for the root artifact only** — traversal never steps sideways from a Release into sibling CR/PRs it also covers. (An earlier version of this logic treated Release as a hub and could pull in ~90% of all releases in the dataset from a single seed; this is now structurally prevented, not just tuned down.)
2. **Upstream spawner** — may be an ALM originator or a parent CR/PR in a follow-up chain.
3. **Validation** — from any discovered Requirement/Specification, to the Test Case/Suite that verifies it.
4. **Downstream children** — recursed, but rule 1 never repeats for a descendant.

---

<a id="workflows"></a>

## ⚙️ Workflows

| Workflow | Steps | LLM cost |
|---|---|---|
| `direct_lookup` | `lookup` | None |
| `similarity_check` | `retrieve` | None |
| `traceability_trace` | `lookup → trace` | None |
| `baseline_check` | `lookup → trace → baseline_evidence` | **None** — genuinely zero-LLM release/baseline determination |
| `full_impact_analysis` | `full_impact_analysis → baseline_evidence → evidence_fusion` | Yes |

`baseline_check` and `full_impact_analysis` both answer "is a release/baseline affected," from two independent evidence sources that are never silently merged:

- **`baseline_evidence`** — traceability-only (`Covers` / `baselines.csv` membership), no LLM call needed.
- **LLM-based determination** — inside `full_impact_analysis`, filtered by the LLM's own High/Medium impact judgments.
- **`evidence_fusion`** — compares the two, reporting agreement, LLM-only, and evidence-only findings separately. Evidence-only findings are further ranked by real query-similarity (using the full, non-Top-K-limited similarity map, not just retained candidates) so a release backed by one strongly-matched artifact correctly outranks one backed by several weakly-matched ones.

---

<a id="agent-planner"></a>

## 🤖 Agent Planner

When neither lexical rules nor semantic matching confidently routes a request, an LLM-based Planner composes a fresh tool sequence directly from the underlying tools — it has no built-in concept of the 5 fixed workflows above, only of what each tool needs and produces (read from each tool's own docstring, not a separately-maintained description that could drift out of sync).

The Planner makes a genuine three-way decision on every query:

- **`plan`** — builds an ordered tool sequence (e.g. `lookup → trace` for "explain this artifact," a pattern with no matching fixed workflow).
- **`clarification`** — the request is a real engineering question but too vague to act on; the Planner's own clarifying question is shown to the user, not a generic error.
- **`not_engineering`** — the request is outside the domain entirely (small talk, unrelated topics); declined with zero tool calls, rather than forcing a plan onto irrelevant input.

Every proposed plan is mechanically validated before execution — unknown tool names, plans longer than 7 steps (only 9 tools exist; longer means looping), and consecutive duplicate steps are all rejected before anything runs. The full **prompt → raw LLM response → validated plan → executed steps** chain is preserved for debugging, not just the final outcome.

---

<a id="project-structure"></a>

## 📁 Project Structure

```text
traceguard-ai/
│
├── data/
│   ├── artifacts.csv
│   ├── baselines.csv
│   ├── evaluation_ground_truth.csv
│   └── evaluation_new_crs.csv
│
├── notebooks/
│   ├── 01-data-generation.ipynb
│   ├── 02-traceguard-simple-runner.ipynb
│   └── 04-planner-evaluation-runner.ipynb
│
├── src/
│   ├── traceguard_v2.py       # Core engine: retrieval, traceability, LLM assessment
│   ├── tools.py               # 9 atomic tools, each wrapping engine logic behind a context-based call
│   ├── workflows.py           # The 5 fixed workflows (data, not code)
│   ├── intent_router.py       # Layer 1: lexical rules + semantic fallback
│   ├── planner.py             # Layer 3: Agent Planner + plan validation
│   ├── orchestrator.py        # Layer 2: executes a workflow's or plan's steps
│   ├── feedback_store.py      # Google Sheets-backed feedback/ratings logging
│   ├── engine_status.py       # Server-wide "has the engine finished loading" flag
│   ├── config/
│   │   └── intents.py         # Keyword lists, exemplar phrases, confidence thresholds
│   └── test_intent_router.py  # 22-case routing regression suite
│
├── pages/
│   ├── 1_About.py             # Architecture, dataset, limitations
│   ├── 2_Dataset_Explorer.py  # Searchable dataset browser + search-before-query flow
│   └── 3_Feedback.py          # Suggestion / bug / feature-request form
│
├── app.py                     # Main Streamlit entry point
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### Notable files

- **`01-data-generation.ipynb`** — generates the synthetic dataset, including the corrected relationship-generation logic (see *Traceability Model*).
- **`02-traceguard-simple-runner.ipynb`** — minimal interface: initialize once, run a query through the full request pipeline, see results.
- **`04-planner-evaluation-runner.ipynb`** — 20 real queries testing the routing pipeline end to end, including deliberately keyword-free phrasings designed to reach the Planner, and deliberately irrelevant queries testing the `not_engineering` decision.

---

<a id="example-usage"></a>

## 🧪 Example Usage

```python
query = "Change braking system axle brake requirements and functionality."

result = orch.run(query)

print(result.workflow, result.confidence, result.success)

if "impact_report_df" in result.final_response:
    display(result.final_response["impact_report_df"])
    print(result.final_response["overall_assessment"])
    print(result.final_response["evidence_fusion"]["status"])
```

Every request — regardless of whether it's a known-ID lookup, a free-text impact analysis, or something no fixed workflow anticipated — goes through the same single `orch.run(query)` call.

---

<a id="dataset"></a>

## 📊 Dataset

This project uses **entirely synthetic automotive engineering data**, created specifically for educational, experimentation, and portfolio purposes. No proprietary, confidential, employer-specific, customer-specific, or real-world organizational engineering data is used in this project.

- `artifacts.csv` — ~5,000 synthetic engineering artifacts across 9 types (Requirements, Specifications, Test Cases, Test Suites, Inputs, Change Requests, Problem Reports, Tasks, Releases).
- `baselines.csv` — release/baseline membership records.
- `evaluation_ground_truth.csv` + `evaluation_new_crs.csv` — 100 paired (proposed-change description → known-affected-artifact-IDs) records, generated for retrieval/impact evaluation. Not yet consumed by an automated evaluation script (see *Known Limitations*).

---

<a id="evaluation"></a>

## 📏 Evaluation

`04-planner-evaluation-runner.ipynb` runs 20 real queries against the live routing pipeline — a mix of known-workflow controls, deliberately keyword-free phrasings designed to test semantic fallback and the Planner, and deliberately irrelevant queries testing the `not_engineering` decision. Results are compared against a stated hypothesis per query, with a human-reviewed judgment on plan quality rather than a fully automated pass/fail (whether a composed plan is the *right* plan is a judgment call an automated check can't fully make).

**Retrieval evaluation (Hit Rate@K, MRR across lexical-only / semantic-only / hybrid RRF) is not yet implemented**, despite a ready-made 100-query ground-truth set already existing in `data/`. This is the most concrete, actionable gap in the project's evaluation story — see *Known Limitations* and *Planned Development*.

---

<a id="monitoring--feedback"></a>

## 📈 Monitoring & Feedback

- **User feedback is collected**: a 👍/👎 rating (with an optional reason and free-text detail for 👎) on every answer on the main page, plus a general suggestion/bug/feature-request form on the Feedback page.
- Both are written to a private Google Sheet via a service account, split across two tabs (**Ratings** and **Feedback**) rather than one mixed log.
- Feedback is intentionally **not displayed anywhere in the app** — the Sheet is the sole, private record.
- **No aggregated dashboard yet.** Per-query metrics (workflow, response time, LLM call count, estimated cost) are shown live for the current answer, and session-scoped summary stats appear in the sidebar, but there is no cross-visitor, historical view (e.g. cost/response-time trends over time, ratings breakdown by workflow). See *Planned Development*.

---

<a id="current-status"></a>

## 🚧 Current Status

```text
✓ Synthetic engineering knowledge base (corrected relationship model)
✓ Lexical + semantic hybrid retrieval (RRF fusion)
✓ Typed, direction-aware traceability expansion
✓ Rank-based traceability seeding
✓ 5 fixed workflows, including a genuinely zero-LLM baseline check
✓ Dual baseline/release determination (LLM + traceability-only) with evidence fusion
✓ Semantic fallback routing, calibrated against real query data
✓ Agent Planner: composes novel tool sequences for unanticipated requests
✓ Three-way Planner decision (plan / clarification / not_engineering)
✓ Mechanical plan validation (unknown tools, length, duplicate steps)
✓ Grounding validation
✓ Planner evaluation notebook (20 real queries, human-reviewed)
✓ Public deployment on Streamlit Community Cloud
✓ User feedback collection (ratings + free-form), Google Sheets-backed
✗ Retrieval evaluation (Hit Rate@K / MRR) -- ground truth exists, not yet run
✗ Aggregated monitoring dashboard
✗ Containerization (Dockerfile / docker-compose)
```

---

<a id="known-limitations"></a>

## ⚠️ Known Limitations

Documented honestly rather than hidden, consistent with this project's design principle below:

- **Semantic fallback rarely fires in practice.** Its confidence threshold was calibrated against real data to 0.90, after finding that genuinely relevant queries only ever scored 0.31–0.37 against workflow exemplars. At 0.90, semantic fallback mostly defers to the Agent Planner rather than actively routing — an intentional tradeoff (the Planner reasons more reliably), but it means this layer is closer to dormant than active today.
- **The Planner's relevance check is not perfect.** Most irrelevant queries (e.g. greetings) are correctly declined with `not_engineering` at essentially no cost, but some plausible-sounding-but-irrelevant phrasings (e.g. a calendar/meeting question) can still slip through and trigger a full, paid LLM-based impact analysis. This is an accepted, monitored limitation — the live deployment currently runs on a single shared API key rather than a per-visitor budget, so treat cost exposure accordingly.
- **Exemplar/workflow boundaries are still somewhat fuzzy** for near-identical traceability-flavored phrasings — two very similarly-worded requests can occasionally land on different (both individually reasonable) workflows.
- **Retrieval evaluation is not yet implemented.** A 100-query ground-truth set (`evaluation_ground_truth.csv` + `evaluation_new_crs.csv`) already exists specifically for this purpose, but Hit Rate@K/MRR have not yet been computed across lexical-only, semantic-only, and hybrid RRF to confirm hybrid is actually the strongest approach rather than assumed to be.
- **No containerization.** The app runs directly via `streamlit run app.py` / Streamlit Community Cloud's own dependency install; no Dockerfile or docker-compose setup exists yet for running it in an isolated, portable container.
- **No aggregated monitoring dashboard.** Feedback and ratings are collected and stored, but there is no visualization of trends over time or across visitors yet — see *Monitoring & Feedback*.

---

<a id="planned-development"></a>

## 🛣️ Planned Development

- Retrieval evaluation and calibration (Hit Rate@K, MRR, candidate coverage) using the existing ground-truth set
- A monitoring dashboard (Streamlit-native, reading the existing Google Sheets feedback/ratings data) covering response time, cost, workflow breakdown, and rating trends
- Containerization (Dockerfile at minimum; docker-compose if a local dependency, e.g. a database, is added later)
- Tightening the Planner's relevance check against the residual gap noted above
- Usage/cost monitoring and a rupee-budget-based demo limiter for the live deployment, calibrated against real token usage rather than an arbitrary query cap
- Richer artifact information in final impact reports
- Plan Memory: reusing previously-composed plans for semantically similar goals, with a structural fit-check before reuse (deferred pending further design work, not yet built)

---

<a id="design-principle"></a>

## 💡 Design Principle

> **AI should assist engineering judgment, not replace it.**

Retrieval identifies potentially relevant evidence. Traceability provides engineering relationship context, verified against real data rather than assumed. The LLM helps interpret that evidence, and — through the Agent Planner — helps route requests that don't fit a fixed pattern. The final decision remains with the engineer, and every known limitation above is documented rather than glossed over, for the same reason.

---

<a id="disclaimer"></a>

## ⚠️ Disclaimer

TraceGuard AI is an **educational and portfolio project**. AI-generated impact assessments are intended to support human engineering analysis and experimentation with AI-assisted engineering workflows. Outputs should **not** be considered authoritative engineering, safety, configuration management, release, quality, or compliance decisions. All results require appropriate human engineering review.
