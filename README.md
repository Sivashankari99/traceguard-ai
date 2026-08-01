# TraceGuard AI

### Hybrid RAG + Traceability-Aware Engineering Change Impact Analysis

TraceGuard AI is an AI-assisted engineering change impact analysis prototype that explores how **Retrieval-Augmented Generation (RAG), hybrid retrieval, engineering traceability, and Large Language Models (LLMs)** can support impact analysis in automotive engineering environments.

Given a proposed engineering change, TraceGuard searches a synthetic engineering knowledge base to identify potentially affected artifacts such as Change Requests, Problem Reports, requirements, specifications, test cases, tasks, releases, and other lifecycle artifacts.

Rather than relying only on keyword matching, TraceGuard combines **lexical retrieval, semantic retrieval, and existing traceability relationships** before providing grounded engineering evidence to an LLM for impact assessment.

Version 2 replaces ad-hoc evidence merging with **validated Reciprocal Rank Fusion (RRF)**, based on retrieval experiments carried out in the project's Module 3 vector-search notebook.

The objective is not to automate engineering decisions, but to provide engineers with a structured and explainable candidate impact analysis for human review.

---

## 🚀 Project Goals

TraceGuard AI explores how AI-assisted retrieval and reasoning can help:

- Identify potentially impacted engineering artifacts from a proposed change.
- Discover related Change Requests and Problem Reports.
- Identify potentially affected requirements, specifications, tests, and tasks.
- Use existing traceability relationships as additional engineering evidence.
- Identify potential release and baseline impacts.
- Generate structured and explainable AI-assisted impact assessments.
- Reduce the effort required to manually explore large engineering artifact repositories.
- Support human engineering and configuration management review rather than replace it.

---

## 🧠 Current Capabilities

The current TraceGuard prototype (Version 2) implements a **Hybrid RAG + RRF-Fused + Traceability-Aware Impact Analysis** pipeline.

It currently supports:

- **Lexical retrieval** using MinSearch, ranked over the full artifact population per type.
- **Semantic retrieval** using Sentence Transformers (MiniLM), ranked over the full artifact population per type.
- **Hybrid candidate discovery via Reciprocal Rank Fusion (RRF)** — lexical and semantic rankings are fused by rank position, not by merging raw scores from two differently calibrated spaces.
- **Top-K candidate selection performed after fusion**, not before — so no candidate is chosen based on either signal in isolation.
- **Artifact-type-aware retrieval** across different engineering artifact categories.
- **Traceability-aware expansion** using existing artifact relationships.
- **Candidate ranking** driven by the fused RRF signal plus a traceability bonus, while preserving underlying lexical/semantic evidence for review.
- **Retrieval diagnostics** reporting lexical/semantic population size, hybrid-retained counts, traceability seed/discovery counts, and final candidate counts sent to the LLM.
- **LLM-assisted engineering impact assessment** using retrieved context.
- **Input relevance classification** to reject queries outside the engineering knowledge domain.
- **Grounding validation** to detect unsupported artifact or traceability claims.
- **Release and baseline impact determination** using available traceability evidence.
- **Structured impact reports** containing impact level, confidence, traceability status, and reasoning.

The retrieval pipeline's embedding model and RRF fusion constant (`rrf_k`) are configurable at initialization, allowing different embedding models or fusion tuning to be evaluated without changing the overall engineering impact analysis workflow.

---

## 🔍 How TraceGuard Works

```text
Incoming Engineering Change
            │
            ▼
 ┌──────────────────────────┐
 │      Lexical Search      │
 │       (MinSearch,        │
 │    full population)      │
 └────────────┬─────────────┘
              │
 ┌────────────┴─────────────┐
 │      Semantic Search     │
 │  (MiniLM, cosine sim.,   │
 │    full population)      │
 └────────────┬─────────────┘
              │
              ▼
     Reciprocal Rank Fusion
        (RRF, per type)
              │
              ▼
      Top-K Candidates
      (selected AFTER
          fusion)
              │
              ▼
      Traceability-Aware
          Expansion
              │
              ▼
      Candidate Ranking
   (fused rank + traceability
          bonus)
              │
              ▼
      Retrieval Diagnostics
      (counts at each stage)
              │
              ▼
       Candidate Context
              │
              ▼
        LLM Relevance
            Check
              │
        ┌─────┴─────┐
        │           │
   Irrelevant    Relevant
        │           │
        ▼           ▼
       Stop       LLM Impact
                  Assessment
                      │
                      ▼
               Grounding
                Validation
                      │
                      ▼
             Release/Baseline
              Determination
                      │
                      ▼
              Structured Human
                Review Report
```

---

## 🔎 Hybrid Retrieval

TraceGuard uses multiple complementary evidence sources rather than relying on a single retrieval technique — and, in Version 2, fuses the two retrieval signals in a way that has been validated rather than assumed.

### 1. Lexical Retrieval

Lexical retrieval is performed using **MinSearch**, ranked over every artifact of a given type.

It identifies engineering artifacts containing words and terminology related to the incoming change.

This approach is particularly useful when the proposed change uses terminology that closely matches existing engineering artifacts.

---

### 2. Semantic Retrieval

TraceGuard also uses **Sentence Transformers (all-MiniLM-L6-v2)** to generate normalized embeddings for engineering artifact text and incoming change descriptions, ranked by cosine similarity over every artifact of a given type.

Semantic similarity allows TraceGuard to identify conceptually related artifacts even when the wording is different.

For example, an incoming change may contain incomplete descriptions, alternate terminology, or spelling mistakes while still expressing an engineering concept represented in the knowledge base.

This complements lexical search by providing meaning-based retrieval.

---

### 3. Hybrid Fusion via Reciprocal Rank Fusion (RRF)

Earlier versions of TraceGuard tracked lexical and semantic evidence independently and merged them by taking the best available score per artifact. Version 2 replaces this with **Reciprocal Rank Fusion**, validated against the project's Module 3 vector-search experiments:

- Lexical and semantic retrieval each produce a **complete ranking** of every artifact in a type — not just a shortlist.
- RRF combines those two full rankings using **rank position**, not raw score, since TF-IDF-style lexical scores and cosine similarity scores live in different, non-comparable scales.
- **Top-K candidates are selected only after fusion**, so no artifact is chosen purely because it ranked well on one signal alone — it has to rank well in the fused ordering.

This is the single most significant retrieval change in Version 2 and is the basis for everything downstream.

---

### 4. Traceability-Aware Discovery

Textual similarity alone does not represent engineering traceability.

TraceGuard therefore preserves existing engineering relationships as a separate evidence source, applied **after** the fused Top-K candidates are selected.

Relevant artifacts can be expanded through available traceability relationships to discover connected:

- Change Requests
- Problem Reports
- Requirements
- Specifications
- Test artifacts
- Tasks
- Releases
- Other lifecycle artifacts

Similarity and traceability are intentionally treated as **different evidence signals**.

A highly similar artifact does not automatically prove engineering impact, while an existing traceability relationship provides additional evidence that should be considered during review.

---

## 🗂️ Artifact-Type-Aware Retrieval

Engineering repositories contain artifact types with very different characteristics and dataset sizes.

TraceGuard therefore performs candidate retrieval, RRF fusion, and Top-K selection **independently per artifact type**, before traceability expansion combines results across types.

The synthetic dataset currently contains artifact categories such as:

- Change Requests
- Problem Reports
- ALM Inputs
- ALM Requirements
- ALM Specifications
- ALM Test Suites
- ALM Test Cases
- Tasks
- Releases

Candidate retention (Top-K per type, post-fusion) is configurable by artifact type rather than assuming that the same retrieval configuration is appropriate for every category.

---

## 🔗 Evidence Aggregation

An artifact reaching the LLM context can be discovered through more than one mechanism:

- Retained in the fused RRF Top-K (with its underlying lexical and semantic scores preserved for display).
- Connected through existing traceability from a seed Change Request or Problem Report.
- Discovered through multiple traceability paths.

TraceGuard preserves these evidence sources instead of discarding them when candidate results are merged, allowing downstream impact assessment to distinguish between **similarity evidence** and **traceability evidence**.

---

## 📊 Retrieval Diagnostics

Version 2 introduces explicit retrieval diagnostics returned alongside every `analyze()` result, so the retrieval pipeline's behavior is visible rather than opaque:

- Lexical population size and semantic population size, per artifact type.
- Hybrid-retained (post-RRF Top-K) count, per artifact type.
- Number of traceability seeds selected from the fused candidates.
- Number of artifacts newly discovered through traceability expansion.
- Number of "unlinked relevant" candidates (similarity-discovered but not traceability-linked).
- Final candidate pool size and number of candidates actually sent to the LLM context.

This makes it possible to reason about retrieval behavior and tune per-type Top-K settings without re-running the full pipeline blind.

---

## 🤖 LLM-Assisted Impact Assessment

After candidate discovery, fusion, Top-K selection, and traceability expansion, selected candidate artifacts are supplied to an LLM for impact assessment.

The LLM is instructed to:

- Use only the supplied candidate artifacts.
- Avoid inventing artifact IDs.
- Avoid inventing engineering relationships.
- Avoid inventing traceability paths.
- Distinguish similarity from traceability.
- Assess potential engineering impact.
- Communicate uncertainty.
- Provide reasoning for identified candidates.
- Classify the relevance of the incoming query before performing impact analysis.

The LLM therefore acts as an **assessment layer over retrieved engineering evidence**, rather than independently searching or inventing engineering artifacts.

---

## 🚫 Input Relevance Checking

Retrieval systems will normally return the closest available results even when a query is unrelated to the dataset.

TraceGuard therefore includes an LLM-based domain relevance check.

Before performing impact assessment, the model determines whether the proposed change is meaningfully related to the engineering domain represented by the available artifacts.

For example, an unrelated query such as:

```text
Can I join class in July?
```

is classified as:

```text
Input relevance: Irrelevant
```

and no artifact impact assessment is produced.

At the same time, noisy but engineering-related inputs can still proceed through impact analysis.

This helps prevent the system from forcing engineering interpretations onto unrelated user inputs.

---

## 🛡️ Grounding Validation

LLM-generated engineering assessments should remain grounded in retrieved evidence.

TraceGuard therefore performs post-assessment grounding validation.

The validation checks whether:

- Returned artifact IDs exist in the supplied candidate context.
- Claimed traceability relationships are supported by discovered evidence.
- Referenced traceability paths were actually available to the model.
- Linked claims correspond to relationships represented in the source data.

This provides an additional safeguard against unsupported LLM-generated engineering claims.

---

## 📦 Release and Baseline Impact

TraceGuard also explores whether identified High or Medium impact Change Requests or Problem Reports can be connected to releases through explicit engineering relationships.

Where sufficient evidence exists, the system can identify potentially affected release or baseline information.

If the available evidence is insufficient, the result remains:

```text
Undetermined
```

rather than inferring a release or baseline impact without supporting evidence.

---

## 📊 Impact Assessment Output

The current impact report provides information such as:

- Artifact ID
- Artifact type
- Potential impact level
- Candidate category
- Traceability status
- Confidence
- Reason for potential impact

Example conceptual output:

```text
Artifact ID    Artifact Type       Impact     Traceability    Confidence
-----------------------------------------------------------------------
INP-00006      ALM Input           High       Linked          High
SPEC-00642     ALM Specification   High       Linked          High
PR-00226       Problem Report      Medium     Linked          Medium
TC-01489       ALM Test Case       High       Linked          High
```

Future versions will enrich the report with additional artifact metadata, summaries, and more detailed traceability explanations.

---

## 📁 Project Structure

```text
traceguard-ai/
│
├── data/
│   ├── artifacts.csv
│   ├── baselines.csv
│   └── evaluation_ground_truth.csv
|   └── evaluation_new_crs.csv
│
├── notebooks/
│   ├── 01-data-generation.ipynb
│   ├── 02-basic-rag.ipynb
│   ├── 02-traceguard-simple-runner.ipynb
│   └── 03-vector-search.ipynb
│
├── src/
│   ├── __init__.py
│   ├── traceguard_v1.py
│   └── traceguard_v2.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### `01-data-generation.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/01-data-generation.ipynb

Generates the synthetic automotive engineering dataset used by TraceGuard.

### `02-basic-rag.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/02-basic-rag.ipynb

Contains the original Hybrid RAG and traceability-aware impact analysis pipeline (Version 1).

The notebook is intentionally retained so that the complete pipeline can be executed step-by-step for:

- Learning
- Experimentation
- Debugging
- Retrieval inspection
- Historical comparison against Version 2

### `03-vector-search.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/03-vector-search.ipynb

Module 3 vector-search experimentation notebook. Evaluates lexical, brute-force semantic, and RRF-fused hybrid retrieval against a dedicated evaluation ground truth using Recall@K and Precision@K, per artifact type. The validated Hybrid RRF result from this notebook is what Version 2 carries into `traceguard_v2.py`.

### `src/traceguard_v1.py` - https://github.com/Sivashankari99/traceguard-ai/blob/main/src/traceguard_v1.py

The original reusable TraceGuard implementation: independent lexical/semantic evidence tracking merged by best-available score, plus traceability, LLM assessment, and validation.

### `src/traceguard_v2.py` - https://github.com/Sivashankari99/traceguard-ai/blob/main/src/traceguard_v2.py

The current reusable TraceGuard implementation. Retrieval is replaced with the validated Hybrid RRF pipeline (lexical + semantic full-population ranking, fused by RRF, Top-K selected after fusion) and retrieval diagnostics are added. Traceability expansion, LLM prompting, grounding validation, and baseline determination are unchanged from Version 1.

### `02-traceguard-simple-runner.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/02-traceguard-simple-runner.ipynb

Provides a simplified interface for running TraceGuard.

Instead of executing the complete implementation notebook cell-by-cell, a user can initialize TraceGuard, enter a proposed engineering change, and execute the analysis.

---

## 🧪 Example Usage

A proposed engineering change can be submitted to TraceGuard using:

```python
from src.traceguard_v2 import TraceGuard

traceguard = TraceGuard(data_path="data")

query = """
Change braking system axle brake requirements and functionality.
""".strip()

result = traceguard.analyze(query)

display(result["impact_report_df"])

print("\nOverall assessment:")
print(result["overall_assessment"])

print("\nRetrieval diagnostics:")
print(result["retrieval_diagnostics"])
```

TraceGuard then performs:

```text
Lexical Retrieval
        +
Semantic Retrieval
        ↓
Reciprocal Rank Fusion
        ↓
Top-K Candidate Selection
        ↓
Traceability Discovery
        ↓
Candidate Ranking
        ↓
LLM Relevance Check
        ↓
Impact Assessment
        ↓
Grounding Validation
        ↓
Release/Baseline Analysis
```

before returning the structured result.

---

## 📊 Dataset

This project uses **entirely synthetic automotive engineering data** created specifically for educational, experimentation, and portfolio purposes.

The dataset represents engineering lifecycle artifacts and relationships needed to experiment with RAG-based engineering impact analysis.

No proprietary, confidential, employer-specific, customer-specific, or real-world organizational engineering data is used in this project.

---

## 📏 Evaluation

An evaluation dataset containing known incoming changes and expected affected-artifact mappings is maintained separately from the primary retrieval dataset.

This dataset was used in `03-vector-search.ipynb` to compare lexical, brute-force semantic, and RRF-fused hybrid retrieval strategies using:

- Retrieval Recall
- Precision
- Recall@K
- Precision@K
- Candidate-pool recall/precision (runtime-like, per-type Top-K unioned)
- Retrieval behavior by artifact type

The Hybrid RRF strategy carried into `traceguard_v2.py` was selected from these measured results rather than assumed in advance.

Future evaluation work will extend this to embedding-model comparisons and similarity-threshold tuning.

---

## 🕘 Version History

**Version 1**
- Initial Hybrid RAG implementation
- Independent lexical and semantic retrieval
- Traceability-aware expansion
- LLM-assisted impact assessment

**Version 2 (Current)**
- Reciprocal Rank Fusion (RRF)
- Top-K selection after fusion
- Retrieval diagnostics
- Improved retrieval architecture
- Preserved traceability and grounding validation

---

## 🚧 Current Status

**Work in Progress — Version 2**

The current milestone implements:

### Hybrid RRF-Fused Retrieval + Traceability-Aware Engineering Change Impact Analysis Pipeline

Implemented capabilities include:

```text
✓ Synthetic engineering knowledge base
✓ Lexical retrieval (full population ranking)
✓ Semantic retrieval (full population ranking, MiniLM)
✓ Reciprocal Rank Fusion (RRF) of lexical + semantic rankings
✓ Top-K candidate selection performed after fusion
✓ Artifact-type-aware candidate retrieval
✓ Retrieval diagnostics per stage
✓ Traceability-aware discovery
✓ Candidate ranking (fused rank + traceability bonus)
✓ LLM-assisted impact assessment
✓ Input relevance classification
✓ Grounding validation
✓ Release/baseline determination
✓ Reusable Python implementation (v1 retained, v2 current)
✓ Simplified notebook runner
```

---

## 🛣️ Planned Development

TraceGuard will continue to evolve alongside further AI Engineering concepts.

Planned areas include:

- **AI Orchestration** — coordinating retrieval, traceability expansion, and LLM assessment as explicit, observable pipeline stages rather than a single monolithic call.
- **Retrieval Evaluation module** — formalizing the Recall@K / Precision@K comparisons from `03-vector-search.ipynb` into a repeatable evaluation harness, run against every retrieval change.
- Top-K optimization by artifact type, informed by retrieval diagnostics.
- Similarity-threshold experimentation.
- Precision and recall analysis across embedding models.
- Richer artifact information in final impact reports.
- Improved traceability explanations.
- Retrieval and LLM monitoring.
- Token and cost monitoring.
- Interactive user interface.
- Additional configuration management and compliance use cases.

The objective is to evolve the project incrementally while keeping each stage understandable, testable, and explainable.

---

## 💡 Design Principle

TraceGuard is intentionally designed around the principle that:

> **AI should assist engineering judgment, not replace it.**

Retrieval identifies potentially relevant evidence.

Traceability provides engineering relationship context.

The LLM helps interpret that evidence.

The final decision remains with the engineer.

---

## ⚠️ Disclaimer

TraceGuard AI is an **educational and portfolio project**.

AI-generated impact assessments are intended to support human engineering analysis and experimentation with AI-assisted engineering workflows.

Outputs should **not** be considered authoritative engineering, safety, configuration management, release, quality, or compliance decisions.

All results require appropriate human engineering review.
