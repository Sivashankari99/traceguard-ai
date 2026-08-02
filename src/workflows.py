"""
Workflow registry.

Each predefined workflow is DATA — an ordered list of tool names — not a
new code path. The orchestrator looks up a workflow by name and executes
its `steps` in order against TraceGuardTools. Adding a new workflow later
means adding an entry here, not writing new pipeline logic.

`requires_entity` tells the orchestrator whether this workflow needs a
valid artifact ID pulled out of the query before it can run at all. If
entity extraction fails for one of these, the orchestrator should fall
back to the clarification path rather than guessing.
"""

WORKFLOWS = {
    "direct_lookup": {
        "description": "Fetch a known artifact's metadata directly. No retrieval, no LLM.",
        "steps": ["lookup"],
        "requires_entity": True,
    },
    "similarity_check": {
        "description": "Find artifacts similar to free-text input. Retrieval only, no LLM call.",
        "steps": ["retrieve"],
        "requires_entity": False,
    },
    "traceability_trace": {
        "description": "Show traceability links for a known artifact. Lookup verifies the ID first.",
        "steps": ["lookup", "trace"],
        "requires_entity": True,
    },
    "baseline_check": {
        "description": (
            "Check release/baseline impact for a known artifact, using ONLY "
            "traceability evidence (Covers / baselines.csv membership). "
            "No LLM call — genuinely zero-LLM, per the resolved dual-answer design."
        ),
        "steps": ["lookup", "trace", "baseline_evidence"],
        "requires_entity": True,
    },
    "full_impact_analysis": {
        "description": (
            "Complete V2 pipeline: hybrid RRF retrieval, traceability "
            "expansion, LLM impact assessment, grounding validation, "
            "LLM-based baseline determination — plus independent "
            "traceability-only baseline evidence, fused and compared "
            "for agreement/disagreement."
        ),
        "steps": ["full_impact_analysis", "baseline_evidence", "evidence_fusion"],
        "requires_entity": False,
    },
}
