"""
IntentRouter — Layer 1 / Decision Layer.

    User Query -> Entity Extraction -> Intent Detection -> Confidence Check
                                                                   │
                                                    ┌──────────────┴──────────────┐
                                                    ▼                             ▼
                                                  High                          Low
                                                    │                             │
                                                    ▼                             ▼
                                            Workflow Selection            Clarification

Named IntentRouter (not just Router) because later stages of this project
will likely add sibling routers with different jobs — e.g. a ToolRouter
deciding which tool implementation to call, or a ModelRouter deciding
which LLM/embedding model to use for a given request. Being explicit here
keeps those distinct once they exist.

Intent detection is now TWO signals, tried in order — mirroring the same
lexical-then-semantic hybrid philosophy used for artifact retrieval, just
applied one layer up at routing instead:

  1. LEXICAL (keyword rules, config/intents.py) — free, instant, tried
     first. If it clears CONFIDENCE_THRESHOLD, semantic is never invoked.
  2. SEMANTIC (embedding similarity against exemplar phrases per
     workflow) — only tried when lexical confidence is too low. Reuses
     whatever embedding model TraceGuard already loaded; no separate
     model or API call needed.

This keeps IntentRouter's dependency on the engine OPTIONAL: constructed
with no embedding_model, it behaves exactly as the pure lexical version
did (test_intent_router.py's 22 cases all still pass unmodified, with
zero data/model loading). The semantic fallback only activates once an
embedding model is actually supplied — e.g. by Orchestrator, which passes
engine.embedding_model through.

All keyword lists, exemplar phrases, the artifact-ID pattern, and both
confidence thresholds live in config/intents.py, not here — this file is
matching LOGIC only. Growing either signal's phrase list should never
require touching this file.
"""

import numpy as np
from dataclasses import dataclass

from src.workflows import WORKFLOWS
from src.config.intents import (
    ARTIFACT_ID_PATTERN,
    TRACE_KEYWORDS,
    BASELINE_KEYWORDS,
    SIMILARITY_KEYWORDS,
    IMPACT_KEYWORDS,
    CONFIDENCE_THRESHOLD,
    SEMANTIC_CONFIDENCE_THRESHOLD,
    WORKFLOW_EXEMPLARS,
)


@dataclass
class RoutingDecision:
    intent: str          # workflow name, or "clarification"
    entity_id: str
    confidence: float
    reason: str          # which rule/signal fired / why clarification was needed


class IntentRouter:
    """Layer 1 only: entity extraction, intent detection, confidence gate."""

    def __init__(self, embedding_model=None):
        # Optional on purpose — see module docstring. None here means
        # "lexical only", matching the router's original behavior.
        self.embedding_model = embedding_model
        self._exemplar_embeddings = None  # built lazily, only if ever needed

    def extract_entity(self, query_text):
        """Rule-based entity extraction: first artifact-ID-shaped token,
        e.g. CR-00123, REQ-00456, SPEC-00642."""
        match = ARTIFACT_ID_PATTERN.search(query_text)
        return match.group(1) if match else None

    def detect_intent(self, query_text, entity_id):
        """Rule-based (lexical) intent detection with an explicit
        confidence score per rule. Tried FIRST, before any semantic
        fallback — it's free and instant.
        """
        text = query_text.lower()
        word_count = len(text.split())

        if entity_id and any(k in text for k in BASELINE_KEYWORDS):
            return "baseline_check", 0.90, "entity + baseline keyword"

        if entity_id and any(k in text for k in TRACE_KEYWORDS):
            return "traceability_trace", 0.90, "entity + traceability keyword"

        if entity_id and word_count <= 6:
            return "direct_lookup", 0.80, "entity + short query, no other keyword"

        if any(k in text for k in SIMILARITY_KEYWORDS):
            return "similarity_check", 0.75, "similarity keyword"

        if any(k in text for k in IMPACT_KEYWORDS):
            return "full_impact_analysis", 0.70, "impact/change keyword"

        # No lexical rule matched with confidence. This is NOT "assume
        # full analysis" — it's genuine uncertainty, scored low so the
        # caller knows to try semantic fallback (or clarify) instead.
        return "full_impact_analysis", 0.30, "no lexical rule matched"

    def _get_exemplar_embeddings(self):
        """Embed every workflow's exemplar phrases once, lazily, and
        cache the result. Only ever called if a semantic fallback is
        actually attempted."""
        if self._exemplar_embeddings is not None:
            return self._exemplar_embeddings

        flat_phrases, flat_workflows = [], []
        for workflow_name, phrases in WORKFLOW_EXEMPLARS.items():
            for phrase in phrases:
                flat_phrases.append(phrase)
                flat_workflows.append(workflow_name)

        vectors = self.embedding_model.encode(
            flat_phrases, normalize_embeddings=True, show_progress_bar=False
        )
        self._exemplar_embeddings = (np.asarray(vectors), flat_phrases, flat_workflows)
        return self._exemplar_embeddings

    def detect_intent_semantic(self, query_text):
        """Semantic fallback: embed the query, compare against every
        exemplar phrase via cosine similarity (normalized embeddings, so
        dot product == cosine similarity), route to the workflow whose
        best-matching exemplar is closest.

        Only called when lexical confidence was too low AND an
        embedding_model was supplied to this router.
        """
        vectors, flat_phrases, flat_workflows = self._get_exemplar_embeddings()
        query_vector = self.embedding_model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )[0]

        similarities = vectors @ query_vector
        best_idx = int(np.argmax(similarities))
        best_workflow = flat_workflows[best_idx]
        best_phrase = flat_phrases[best_idx]
        best_score = float(similarities[best_idx])

        return best_workflow, best_score, f"semantic fallback, closest exemplar: {best_phrase!r}"

    def route(self, query_text):
        """Returns a RoutingDecision. Tries lexical first; falls back to
        semantic only if lexical confidence is too low AND an
        embedding_model was supplied. intent is a real workflow name only
        when the winning signal clears ITS OWN threshold; otherwise
        intent is 'clarification' and the caller should ask, not execute.

        IMPORTANT: lexical_cleared/semantic_cleared are tracked as
        separate booleans, NOT inferred from comparing a blended
        confidence number against a single threshold. An earlier version
        did `confidence = max(lexical_confidence, semantic_confidence)`
        and then checked that blended value against
        SEMANTIC_CONFIDENCE_THRESHOLD — which silently broke the moment
        SEMANTIC_CONFIDENCE_THRESHOLD was calibrated below the lexical
        fallback's hardcoded 0.30 "no rule matched" constant: max(0.30,
        weak_semantic) always came out >= 0.30, so genuinely irrelevant
        queries stopped reaching clarification even though neither
        signal actually cleared its own bar. Comparing two
        differently-scaled confidence numbers via max() is unsound
        regardless of what either threshold is set to.
        """
        entity_id = self.extract_entity(query_text)
        intent, confidence, reason = self.detect_intent(query_text, entity_id)
        lexical_cleared = confidence >= CONFIDENCE_THRESHOLD
        semantic_cleared = False

        if not lexical_cleared and self.embedding_model is not None:
            semantic_intent, semantic_confidence, semantic_reason = (
                self.detect_intent_semantic(query_text)
            )
            semantic_cleared = semantic_confidence >= SEMANTIC_CONFIDENCE_THRESHOLD
            if semantic_cleared:
                intent, confidence, reason = semantic_intent, semantic_confidence, semantic_reason
            else:
                reason = (
                    f"lexical: {reason}; semantic fallback also low "
                    f"({semantic_confidence:.2f}, {semantic_reason})"
                )
                # Kept for DISPLAY only — this max() must never be
                # compared against a threshold again; the routing
                # decision below is driven solely by the two booleans.
                confidence = max(confidence, semantic_confidence)

        if not lexical_cleared and not semantic_cleared:
            return RoutingDecision(
                intent="clarification",
                entity_id=entity_id,
                confidence=confidence,
                reason=f"Confidence {confidence:.2f} below threshold ({reason}).",
            )

        workflow = WORKFLOWS[intent]
        if workflow["requires_entity"] and not entity_id:
            return RoutingDecision(
                intent="clarification",
                entity_id=None,
                confidence=confidence,
                reason=f"Workflow '{intent}' needs an artifact ID but none was found.",
            )

        return RoutingDecision(
            intent=intent, entity_id=entity_id, confidence=confidence, reason=reason,
        )
