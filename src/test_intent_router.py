"""Quick routing smoke test — IntentRouter only, no engine/data needed.
Not a formal eval; just catches obvious misrouting before wiring tools."""

from src.intent_router import IntentRouter

router = IntentRouter()

test_cases = [
    # (query, expected_intent)
    ("What is REQ-00123?", "direct_lookup"),
    ("Show me SPEC-00642", "direct_lookup"),
    ("CR-00456 details", "direct_lookup"),
    ("Give me info on TC-01489", "direct_lookup"),
    ("PR-00226?", "direct_lookup"),

    ("Is there anything similar to overheating in the battery pack?", "similarity_check"),
    ("Has a duplicate of this brake failure already been raised?", "similarity_check"),
    ("Any existing artifacts about sensor calibration drift?", "similarity_check"),
    ("Similar issues to axle vibration at high speed", "similarity_check"),
    ("Already raised complaints about charging port corrosion?", "similarity_check"),

    ("Change braking system axle brake requirements and functionality.", "full_impact_analysis"),
    ("We need to modify the fuel cell cooling requirement.", "full_impact_analysis"),
    ("Impact of updating the eCompressor firmware version.", "full_impact_analysis"),
    ("Requirement change for battery thermal management.", "full_impact_analysis"),
    ("What is the impact of changing the axle torque spec?", "full_impact_analysis"),

    ("Show traceability for CR-00123", "traceability_trace"),
    ("What links to REQ-00456?", "traceability_trace"),
    ("Is release R-045 affected by CR-00789?", "baseline_check"),
    ("Baseline impact of PR-00111", "baseline_check"),

    # Ambiguous / should trigger clarification
    ("hey", "clarification"),
    ("can I join class in July", "clarification"),
    ("hmm not sure what I need", "clarification"),
]

passed, failed = 0, 0
for query, expected in test_cases:
    decision = router.route(query)
    status = "PASS" if decision.intent == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"[{status}] '{query}'")
    print(f"       -> intent={decision.intent!r} (expected {expected!r}), "
          f"confidence={decision.confidence:.2f}, entity={decision.entity_id}, reason={decision.reason}")

print(f"\n{passed} passed, {failed} failed out of {len(test_cases)}")
