"""
Tool wrappers around TraceGuard V2.

Every tool takes the single accumulated `context` dict directly (the
same dict the orchestrator builds up as steps run) and extracts what it
needs internally -- including "smart" detection when a value could
legitimately come from more than one upstream source (e.g.
baseline_evidence works whether the previous step was `trace` or
`full_impact_analysis`).

Design decision: no formal OR-group dependency solver. With only 9
tools, that would be over-engineering -- a few if/elif checks inside the
tool itself is simpler, easier to read, and just as correct. Each tool
raises a clear ToolResult(ok=False, error=...) if nothing it recognizes
is present in context, so a broken plan fails loudly at the point it's
actually run, not silently.

No new retrieval, traceability, or LLM logic lives here -- everything
still defers to the V2 engine (self.engine).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ToolResult:
    """Uniform wrapper so the orchestrator can log/inspect every tool call
    the same way, regardless of which tool ran."""
    tool: str
    ok: bool
    data: object
    error: str = None


class TraceGuardTools:
    """One instance per loaded TraceGuard engine. Every method takes the
    shared `context` dict and returns a ToolResult. The orchestrator
    calls these by name -- it never needs to know which keys a given
    tool reads from context."""

    def __init__(self, engine):
        self.engine = engine

    # ------------------------------------------------------------------
    # Tool: lookup
    # Answers: does this artifact exist, and what is it?
    # ------------------------------------------------------------------
    def lookup(self, context):
        """Retrieve metadata for a known artifact ID. No search, no traceability."""
        artifact_id = context.get("entity_id")
        if not artifact_id:
            return ToolResult(tool="lookup", ok=False, data=None,
                               error="lookup needs entity_id in context.")
        artifact_id = str(artifact_id).strip()
        row = self.engine.artifact_lookup.get(artifact_id)
        if row is None:
            return ToolResult(
                tool="lookup", ok=False, data=None,
                error=f"Artifact ID '{artifact_id}' not found.",
            )
        record = {col: row.get(col, "") for col in self.engine.DOCUMENT_COLUMNS}
        return ToolResult(tool="lookup", ok=True, data=record)

    # ------------------------------------------------------------------
    # Tool: retrieve
    # Answers: what's similar to this free-text input?
    # Hybrid RRF retrieval across every artifact type. No traceability,
    # no LLM call.
    # ------------------------------------------------------------------
    def retrieve(self, context):
        """Hybrid RRF retrieval (lexical + semantic) across all artifact types for free-text input. No traceability, no LLM call."""
        query_text = str(context.get("query_text", "")).strip()
        if not query_text:
            return ToolResult(tool="retrieve", ok=False, data=None,
                               error="retrieve needs query_text in context.")

        query_embedding = self.engine._embed_query(query_text)
        results_by_type = {}
        diagnostics = []
        for artifact_type in self.engine.retrieval_types:
            rows, diag = self.engine._hybrid_retrieve(
                query_text, artifact_type, query_embedding
            )
            results_by_type[artifact_type] = rows
            diagnostics.append(diag)

        return ToolResult(tool="retrieve", ok=True, data={
            "results_by_type": results_by_type,
            "diagnostics": diagnostics,
        })

    # ------------------------------------------------------------------
    # Tool: trace
    # Answers: what does this already-known artifact link to?
    # ------------------------------------------------------------------
    def trace(self, context):
        """Typed traceability expansion from a known artifact ID (release membership, upstream spawner, validated test cases, downstream children)."""
        artifact_id = context.get("entity_id")
        if not artifact_id:
            return ToolResult(tool="trace", ok=False, data=None,
                               error="trace needs entity_id in context.")
        artifact_id = str(artifact_id).strip()
        if artifact_id not in self.engine.known_ids:
            return ToolResult(
                tool="trace", ok=False, data=None,
                error=f"Artifact ID '{artifact_id}' not found; run lookup first.",
            )

        discoveries = self.engine._expand_traceability({artifact_id})
        return ToolResult(tool="trace", ok=True, data={
            "seed_id": artifact_id,
            "discoveries": discoveries,
            "discovered_count": len(discoveries),
        })

    # ------------------------------------------------------------------
    # Tool: assess_impact
    # Answers: what does the LLM think about this evidence?
    # SMART: builds context_records from whichever upstream step
    # produced usable evidence -- trace's raw discoveries, or an
    # explicit context_records list already sitting in context.
    # ------------------------------------------------------------------
    def assess_impact(self, context):
        """LLM impact judgment over evidence already gathered by lookup/trace/retrieve."""
        query_text = str(context.get("query_text", "")).strip()
        if not query_text:
            return ToolResult(tool="assess_impact", ok=False, data=None,
                               error="assess_impact needs query_text in context.")

        context_records = context.get("context_records")
        if context_records is None:
            discoveries = None
            if "trace" in context:
                discoveries = context["trace"].get("discoveries")
            elif "full_impact_analysis" in context:
                discoveries = context["full_impact_analysis"].get("discoveries")

            if not discoveries:
                return ToolResult(
                    tool="assess_impact", ok=False, data=None,
                    error=(
                        "assess_impact needs context_records, or a prior "
                        "'trace'/'full_impact_analysis' step with discoveries; "
                        "none was found in context."
                    ),
                )
            context_records = self.engine._build_context_records_from_discoveries(discoveries)

        prompt = self.engine._build_prompt(query_text, context_records)
        try:
            assessment = self.engine._call_llm(prompt)
        except Exception as exc:
            return ToolResult(tool="assess_impact", ok=False, data=None, error=repr(exc))
        return ToolResult(tool="assess_impact", ok=True, data={
            "assessment": assessment,
            "prompt": prompt,
        })

    # ------------------------------------------------------------------
    # Tool: validate
    # Answers: is the LLM's assessment actually grounded in the context
    # it was given?
    # ------------------------------------------------------------------
    def validate(self, context):
        """Checks whether an LLM assessment is grounded in the evidence it was given."""
        assess_result = context.get("assess_impact")
        if not assess_result:
            return ToolResult(tool="validate", ok=False, data=None,
                               error="validate needs a prior 'assess_impact' step in context.")
        assessment = assess_result.get("assessment")
        context_records = context.get("context_records", [])
        result = self.engine._validate_grounding(assessment, context_records)
        return ToolResult(tool="validate", ok=bool(result.get("passed", False)), data=result)

    # ------------------------------------------------------------------
    # Tool: determine_baseline
    # Answers: is a release/baseline affected, per the LLM's own
    # impact_level judgments?
    # LEGACY / currently unused by any of the 5 built-in workflows --
    # full_impact_analysis computes the equivalent internally via
    # analyze(). Available for planner composition.
    # ------------------------------------------------------------------
    def determine_baseline(self, context):
        """LLM-judged release/baseline determination, based on assess_impact's output."""
        assess_result = context.get("assess_impact")
        assessment = (
            assess_result.get("assessment") if assess_result
            else {"assessments": []}
        )
        affected_df, determination = self.engine._determine_baselines(assessment)
        return ToolResult(tool="determine_baseline", ok=True, data={
            "affected_baselines_df": affected_df,
            "baseline_determination": determination,
        })

    # ------------------------------------------------------------------
    # Tool: baseline_evidence
    # Answers: is a release/baseline affected, using ONLY traceability
    # data -- no LLM call, no impact_level needed.
    # SMART: works whether the previous step was `trace` (single-artifact
    # workflows) or `full_impact_analysis` (its own internal multi-seed
    # traceability).
    # ------------------------------------------------------------------
    def baseline_evidence(self, context):
        """Traceability-only release/baseline determination -- no LLM call needed."""
        if "trace" in context:
            trace_result = context["trace"]
            seed_ids = {trace_result["seed_id"]}
            discoveries = trace_result["discoveries"]
        elif "full_impact_analysis" in context:
            fia = context["full_impact_analysis"]
            seed_ids = fia.get("seed_ids", set())
            discoveries = fia.get("discoveries", {})
        else:
            return ToolResult(
                tool="baseline_evidence", ok=False, data=None,
                error="baseline_evidence needs a prior 'trace' or 'full_impact_analysis' step in context.",
            )

        affected_df, determination = self.engine._determine_baselines_from_traceability(
            seed_ids, discoveries
        )
        return ToolResult(tool="baseline_evidence", ok=True, data={
            "affected_baselines_df": affected_df,
            "baseline_determination": determination,
        })

    # ------------------------------------------------------------------
    # Tool: evidence_fusion
    # Answers: do the LLM-judged baseline result and the traceability-only
    # baseline result agree? Disagreement is a signal for human review,
    # not something to silently resolve in code.
    # Only meaningful after full_impact_analysis (the LLM-based baseline
    # judgment lives inside its result) AND baseline_evidence have both run.
    # ------------------------------------------------------------------
    def evidence_fusion(self, context):
        """Compares an LLM-judged baseline result against a traceability-only one; flags disagreement."""
        fia = context.get("full_impact_analysis")
        evidence_result = context.get("baseline_evidence")
        if not fia or not evidence_result:
            return ToolResult(
                tool="evidence_fusion", ok=False, data=None,
                error="evidence_fusion needs both 'full_impact_analysis' and 'baseline_evidence' to have already run.",
            )

        llm_determination = fia.get("baseline_determination") or {"affected_release_ids": []}
        llm_affected_df = fia.get("affected_baselines_df")
        evidence_determination = evidence_result["baseline_determination"]
        evidence_affected_df = evidence_result["affected_baselines_df"]
        similarity_lookup = fia.get("full_semantic_similarity", {}) or {}

        llm_release_ids = set((llm_determination or {}).get("affected_release_ids", []))
        evidence_release_ids = set((evidence_determination or {}).get("affected_release_ids", []))

        agreement = sorted(llm_release_ids & evidence_release_ids)
        llm_only = sorted(llm_release_ids - evidence_release_ids)
        evidence_only = sorted(evidence_release_ids - llm_release_ids)

        if llm_only or evidence_only:
            status = "Disagreement — review recommended"
        elif agreement:
            status = "Agreement — both signals confirm impact"
        else:
            status = "No release impact detected by either signal"

        # Strength-of-evidence detail for releases NOT confirmed by the
        # LLM: supporting_cr_pr_count/affected_alm_count are STRUCTURAL
        # (real graph edges); best_supporting_similarity is TOPICAL (max
        # similarity to the actual query among supporting seeds). No new
        # embeddings here -- similarity_lookup reuses what retrieval
        # already computed.
        evidence_only_detail = []
        if evidence_only and evidence_affected_df is not None and not evidence_affected_df.empty:
            detail_df = evidence_affected_df[evidence_affected_df["Release_ID"].isin(evidence_only)]
            for _, row in detail_df.iterrows():
                supporting_ids = [
                    s.strip() for s in str(row["Supporting_CR_PR_IDs"]).split(",") if s.strip()
                ]
                sims = [
                    similarity_lookup[sid] for sid in supporting_ids
                    if sid in similarity_lookup and pd.notna(similarity_lookup[sid])
                ]
                best_similarity = max(sims) if sims else None
                evidence_only_detail.append({
                    "release_id": row["Release_ID"],
                    "supporting_cr_pr_count": int(row["Supporting_CR_PR_Count"]),
                    "affected_alm_count": int(row["Affected_ALM_Count"]),
                    "best_supporting_similarity": round(float(best_similarity), 4) if best_similarity is not None else None,
                })

            evidence_only_detail.sort(
                key=lambda d: (
                    d["best_supporting_similarity"] if d["best_supporting_similarity"] is not None else -1.0,
                    d["supporting_cr_pr_count"],
                ),
                reverse=True,
            )

        return ToolResult(tool="evidence_fusion", ok=True, data={
            "status": status,
            "agreement_release_ids": agreement,
            "llm_only_release_ids": llm_only,
            "evidence_only_release_ids": evidence_only,
            "evidence_only_detail": evidence_only_detail,
            "llm_affected_baselines_df": llm_affected_df,
            "evidence_affected_baselines_df": evidence_affected_df,
        })

    # ------------------------------------------------------------------
    # Tool: full_impact_analysis
    # The complete, unchanged V2 pipeline, exposed as a single tool.
    # ------------------------------------------------------------------
    def full_impact_analysis(self, context):
        """The complete retrieval + traceability + LLM assessment pipeline in one call. Equivalent to retrieve+trace+assess_impact+validate+determine_baseline chained manually -- prefer this single tool over recreating that sequence."""
        query_text = str(context.get("query_text", "")).strip()
        if not query_text:
            return ToolResult(tool="full_impact_analysis", ok=False, data=None,
                               error="full_impact_analysis needs query_text in context.")
        call_llm = context.get("call_llm", True)
        result = self.engine.analyze(query_text, call_llm=call_llm)
        return ToolResult(tool="full_impact_analysis", ok=True, data=result)
