"""
Intent routing configuration.

Everything that defines HOW a query maps to an intent lives here — keyword
lists, the artifact-ID pattern, and the confidence threshold. IntentRouter
(intent_router.py) only contains matching LOGIC; it should never need to
change just because a new automotive term, abbreviation, or synonym needs
to be added to a keyword set.

This is the file to grow over time (20-30+ keywords, synonyms, automotive
terminology) without touching routing behavior itself.
"""

import re

ARTIFACT_ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-\d{3,6})\b")

TRACE_KEYWORDS = {
    "trace", "traceability", "linked", "links", "related to",
}

BASELINE_KEYWORDS = {
    "baseline", "release", "affected release",
}

SIMILARITY_KEYWORDS = {
    "similar", "duplicate", "already raised", "existing",
}

IMPACT_KEYWORDS = {
    "change", "impact", "modify", "update", "requirement change",
}

# Confidence below this routes to clarification instead of a workflow,
# regardless of which rule matched.
CONFIDENCE_THRESHOLD = 0.65

# --- Semantic fallback (used only when lexical keyword matching misses) ---
SEMANTIC_CONFIDENCE_THRESHOLD = 0.90

WORKFLOW_EXEMPLARS = {

    "direct_lookup": [
        "Can you help me understand this change request",
        "Show me the details of this problem report",
        "What does this specification cover",
        "Give me the summary and current status of this artifact",
    ],
    "similarity_check": [
        "Has a similar diagnostic response issue been reported before in battery temperature processing",
        "Are there existing problem reports about compressor control diagnostic recovery under transient faults",
        "Find prior change requests related to torque sensing behavior for steering",
        "Check if a similar defect in ADAS false detection warning logic already exists",
    ],
    "traceability_trace": [
        "Show what this change request links to",
        "What upstream requirement or specification led to this problem report",
        "Trace the origin of this defect back to its requirement",
        "Which test cases validate the requirement behind this change",
    ],
    "baseline_check": [
        "Is the current release baseline affected by this defect",
        "Which release does this change request belong to",
        "Assess this problem report's impact against the release baseline",
        "Does this change affect any baselined specifications or test cases",
    ],
    "full_impact_analysis": [
        "Refine torque limitation handling to improve fault detection and recovery behavior",
        "Adjust diagnostic logic for warning behavior to reduce false detections",
        "Enhance network wakeup and sleep management to support improved diagnostic monitoring",
        "Extend DC fast charging handling to cover an additional operating scenario",
        "Introduce configurable thresholds for damper control to support variant-specific tuning",
        "Align steering behavior for torque sensing with updated program requirements",
    ],
}
