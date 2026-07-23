# TraceGuard AI

### Hybrid RAG + Traceability-Aware Engineering Change Impact Analysis

TraceGuard AI is an AI-assisted engineering change impact analysis prototype that explores how **Retrieval-Augmented Generation (RAG), semantic search, engineering traceability, and Large Language Models (LLMs)** can support impact analysis in automotive engineering environments.

Given a proposed engineering change, TraceGuard searches a synthetic engineering knowledge base to identify potentially affected artifacts such as Change Requests, Problem Reports, requirements, specifications, test cases, tasks, releases, and other lifecycle artifacts.

Rather than relying only on keyword matching, TraceGuard combines **lexical retrieval, semantic retrieval, and existing traceability relationships** before providing grounded engineering evidence to an LLM for impact assessment.

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

The current TraceGuard prototype implements a **Hybrid RAG + Traceability-Aware Impact Analysis** pipeline.

It currently supports:

- **Lexical retrieval** using MinSearch.
- **Semantic retrieval** using Sentence Transformers.
- **Hybrid candidate discovery** combining lexical and semantic evidence.
- **Artifact-type-aware retrieval** across different engineering artifact categories.
- **Traceability-aware expansion** using existing artifact relationships.
- **Candidate ranking** while preserving retrieval and traceability evidence.
- **LLM-assisted engineering impact assessment** using retrieved context.
- **Input relevance classification** to reject queries outside the engineering knowledge domain.
- **Grounding validation** to detect unsupported artifact or traceability claims.
- **Release and baseline impact determination** using available traceability evidence.
- **Structured impact reports** containing impact level, confidence, traceability status, and reasoning.

---

## 🔍 How TraceGuard Works

```text
Incoming Engineering Change
            │
            ▼
 ┌─────────────────────────┐
 │     Hybrid Retrieval    │
 │                         │
 │   Lexical Retrieval     │
 │      (MinSearch)        │
 │           +             │
 │   Semantic Retrieval    │
 │ (Sentence Transformers) │
 └────────────┬────────────┘
              │
              ▼
      Candidate Artifacts
              │
              ▼
      Traceability-Aware
          Expansion
              │
              ▼
       Evidence Merging
         and Ranking
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

TraceGuard uses multiple complementary evidence sources rather than relying on a single retrieval technique.

### 1. Lexical Retrieval

Lexical retrieval is performed using **MinSearch**.

It identifies engineering artifacts containing words and terminology related to the incoming change.

This approach is particularly useful when the proposed change uses terminology that closely matches existing engineering artifacts.

---

### 2. Semantic Retrieval

TraceGuard also uses **Sentence Transformers** to generate embeddings for engineering artifact text and incoming change descriptions.

Semantic similarity allows TraceGuard to identify conceptually related artifacts even when the wording is different.

For example, an incoming change may contain incomplete descriptions, alternate terminology, or spelling mistakes while still expressing an engineering concept represented in the knowledge base.

This complements lexical search by providing meaning-based retrieval.

---

### 3. Traceability-Aware Discovery

Textual similarity alone does not represent engineering traceability.

TraceGuard therefore preserves existing engineering relationships as a separate evidence source.

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

TraceGuard therefore performs candidate retrieval independently by artifact type before combining the results.

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

Candidate retention is configurable by artifact type rather than assuming that the same retrieval configuration is appropriate for every category.

Complete semantic scoring can also be retained for evaluation before Top-K candidate selection is applied.

---

## 🔗 Evidence Aggregation

An artifact may be discovered through more than one mechanism.

For example, an artifact may be:

- Retrieved lexically.
- Retrieved semantically.
- Connected through existing traceability.
- Discovered through multiple traceability paths.

TraceGuard preserves these evidence sources instead of discarding them when candidate results are merged.

This allows downstream impact assessment to distinguish between:

**Similarity evidence** and **traceability evidence**.

---

## 🤖 LLM-Assisted Impact Assessment

After candidate discovery and evidence aggregation, selected candidate artifacts are supplied to an LLM for impact assessment.

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
│   └── 02-traceguard-simple-runner.ipynb
│
├── src/
│   ├── __init__.py
│   └── traceguard.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### `01-data-generation.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/01-data-generation.ipynb

Generates the synthetic automotive engineering dataset used by TraceGuard.

### `02-basic-rag.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/02-basic-rag.ipynb

Contains the detailed implementation of the Hybrid RAG and traceability-aware impact analysis pipeline.

The notebook is intentionally retained so that the complete pipeline can be executed step-by-step for:

- Learning
- Experimentation
- Debugging
- Retrieval inspection
- Future evaluation work

### `src/traceguard.py` - https://github.com/Sivashankari99/traceguard-ai/blob/main/src/traceguard.py

Contains the reusable TraceGuard implementation with the core retrieval, traceability, LLM assessment, and validation functionality.

### `02-traceguard-simple-runner.ipynb` - https://github.com/Sivashankari99/traceguard-ai/blob/main/notebooks/02-traceguard-simple-runner.ipynb

Provides a simplified interface for running TraceGuard.

Instead of executing the complete implementation notebook cell-by-cell, a user can initialize TraceGuard, enter a proposed engineering change, and execute the analysis.

---

## 🧪 Example Usage

A proposed engineering change can be submitted to TraceGuard using:

```python
query = """
Change braking system axle brake requirements and functionality.
""".strip()

result = traceguard.analyze(query)

display(result["impact_report_df"])

print("\nOverall assessment:")
print(result["overall_assessment"])
```

TraceGuard then performs:

```text
Retrieval
   ↓
Semantic Comparison
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

This enables future evaluation of whether the retrieval pipeline successfully discovers artifacts that are expected to be impacted.

Evaluation and retrieval calibration are intentionally treated as a separate development stage from the initial RAG implementation.

Future evaluation work will explore metrics such as:

- Retrieval Recall
- Precision
- Recall@K
- Precision@K
- Candidate coverage
- Retrieval behavior by artifact type
- Semantic similarity distributions
- Appropriate Top-K configuration
- Potential similarity thresholds

The objective will be to calibrate retrieval based on measured performance rather than arbitrary thresholds.

---

## 🚧 Current Status

**Work in Progress**

The current milestone implements the core:

### Hybrid RAG + Traceability-Aware Engineering Change Impact Analysis Pipeline

Implemented capabilities include:

```text
✓ Synthetic engineering knowledge base
✓ Lexical retrieval
✓ Semantic retrieval
✓ Sentence Transformer embeddings
✓ Artifact-type-aware candidate retrieval
✓ Hybrid evidence aggregation
✓ Traceability-aware discovery
✓ Candidate ranking
✓ LLM-assisted impact assessment
✓ Input relevance classification
✓ Grounding validation
✓ Release/baseline determination
✓ Reusable Python implementation
✓ Simplified notebook runner
```

---

## 🛣️ Planned Development

TraceGuard will continue to evolve alongside further AI Engineering concepts.

Planned areas include:

- Retrieval evaluation and calibration
- Top-K optimization by artifact type
- Similarity-threshold experimentation
- Precision and recall analysis
- Richer artifact information in final impact reports
- Improved traceability explanations
- Retrieval and LLM monitoring
- Token and cost monitoring
- Improved orchestration
- Interactive user interface
- Additional configuration management and compliance use cases

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
