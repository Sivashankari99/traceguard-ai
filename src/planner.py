"""
Agent Planner.

Given a user goal, generates an execution plan directly from the 9
available tools -- NOT restricted to the 5 predefined workflows in
workflows.py. Unlike IntentRouter, the Planner does not know workflows
exist; it reasons from each tool's own docstring.

Design decision: ExecutionPlan.steps is a plain list of tool-name
strings -- the EXACT same datatype as WORKFLOWS[intent]["steps"]. The
orchestrator's execution loop never needs to know or care whether its
step list came from the fixed workflow registry or from a freshly-
composed plan.

Also by design: no formal tool-schema/InputSpec dependency solver here.
With only 9 tools, each tool already does its own smart internal
extraction (see tools.py) and fails loudly with a clear error if
something it needs isn't in context. This Planner does a few cheap,
mechanical sanity checks on the proposed plan -- not a full pre-
execution fit-check.
"""

import inspect
from dataclasses import dataclass, field

from src.tools import TraceGuardTools

MAX_PLAN_LENGTH = 7  # only 9 tools exist; a plan longer than this is almost certainly wrong


def _build_tool_descriptions():
    """Generated from TraceGuardTools' actual docstrings, not a
    separately-maintained dict -- so the Planner's prompt can never
    silently drift out of sync with what the tools actually do. Each
    tool method's docstring is the single source of truth."""
    descriptions = {}
    for name, method in inspect.getmembers(TraceGuardTools, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(method)
        descriptions[name] = doc.strip() if doc else "(no description available)"
    return descriptions


TOOL_DESCRIPTIONS = _build_tool_descriptions()


@dataclass
class ExecutionPlan:
    goal: str
    status: str           # "plan" | "clarification" | "not_engineering" -- the Planner's actual decision
    steps: list          # list[str] -- tool names, SAME shape as WORKFLOWS[intent]["steps"]
    reasoning: str
    valid: bool
    validation_errors: list = field(default_factory=list)
    prompt: str = None           # logging only: the exact prompt sent to the LLM
    raw_response: dict = None    # logging only: the LLM's parsed JSON, before validation


class Planner:
    """Composes a tool sequence for a goal the router couldn't
    confidently match to a known workflow. engine is only used for its
    _call_llm method -- the same LLM-call infrastructure assess_impact
    already reuses, so no new LLM plumbing is introduced here.
    """

    def __init__(self, engine):
        self.engine = engine

    def _build_tool_catalog_text(self):
        return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())

    def _build_prompt(self, goal, query_text, entity_id=None):
        catalog = self._build_tool_catalog_text()
        entity_line = (
            f'A known artifact ID was found in the query: "{entity_id}".'
            if entity_id else
            "No specific artifact ID was found in the query -- this is a free-text request."
        )
        return f"""
You are an execution planner for an engineering traceability system.

GOAL (inferred from the user's request): {goal}

USER'S ORIGINAL QUERY: {query_text}

{entity_line}

AVAILABLE TOOLS (use ONLY these, by exact name):
{catalog}

Build the SHORTEST ordered sequence of tools that answers the goal.

Rules:
0. First decide whether this query can be answered by these tools at all:
   - If the query is NOT a genuine engineering request about a Change
     Request, Problem Report, Requirement, Specification, Test Case,
     Task, or Release -- e.g. small talk, greetings, or anything
     unrelated to this domain -- respond with status "not_engineering"
     and an empty steps list. Do NOT force a plan onto irrelevant input.
   - If the query IS a genuine engineering request, but has NO
     discernible subject or specific ask at all (e.g. "can you help me"
     or "check on the status of things" with nothing further), respond
     with status "clarification", an empty steps list, and put the
     actual clarifying question you'd ask the user in "reasoning".
   - IMPORTANT: a specific, clearly-described engineering change or
     problem -- even with NO artifact ID cited at all -- is NOT vague.
     That is exactly what full_impact_analysis is for; a missing ID is
     expected and normal for this case, not a reason to ask for
     clarification. Only use "clarification" when the request itself has
     no discernible engineering subject, not merely because it lacks an ID.
   - Otherwise, respond with status "plan" and build the steps as below.
1. Only use tool names from the list above, spelled exactly as shown.
2. Prefer the fewest steps that genuinely answer the goal. Do not add
   assess_impact, validate, evidence_fusion, or determine_baseline
   unless the goal specifically needs LLM judgment or release/baseline
   determination.
2a. If a known artifact ID is present AND the user's request is to
   explain, understand, describe, or inspect that artifact (rather than
   assess an engineering change or ask about a release/baseline),
   prefer lookup -> trace as the complete answer -- it is almost always
   sufficient, and no LLM judgment step is needed for this pattern.
3. IMPORTANT: if the goal is equivalent to the full existing
   retrieve -> trace -> assess_impact -> validate -> determine_baseline
   pipeline (i.e. a genuine end-to-end impact analysis of free-text
   input, not anchored to one known artifact), use the single tool
   full_impact_analysis instead of recreating that sequence step by
   step. Do not propose both a manual chain AND full_impact_analysis --
   pick one.
4. Each tool depends on earlier ones producing what it needs (e.g. trace
   needs a known artifact ID; assess_impact needs prior evidence from
   trace or retrieve). Order your steps so each one has what it needs
   from the artifact ID given above or from an earlier step.
5. Never repeat the same tool twice in a row. Keep the plan under
   {MAX_PLAN_LENGTH} steps -- there are only 9 tools available, so a
   longer plan means you are going in circles.
6. Return ONLY valid JSON, no markdown fences, no prose outside the
   JSON, in exactly this shape:

{{
  "status": "plan",
  "steps": ["lookup", "trace"],
  "reasoning": "One or two sentences on why this sequence answers the goal, OR (if status is clarification) the actual question to ask the user, OR (if status is not_engineering) a brief note on why."
}}

status must be exactly one of: "plan", "clarification", "not_engineering".
When status is "clarification" or "not_engineering", steps MUST be [].
""".strip()

    def create_plan(self, goal, query_text, entity_id=None):
        prompt = self._build_prompt(goal, query_text, entity_id)
        raw = self.engine._call_llm(prompt)

        status = raw.get("status", "plan")
        reasoning = raw.get("reasoning", "")

        if status in ("not_engineering", "clarification"):
            return ExecutionPlan(
                goal=goal, status=status, steps=[], reasoning=reasoning,
                valid=False,
                validation_errors=[
                    "Query is not a genuine engineering request." if status == "not_engineering"
                    else "Planner needs clarification from the user."
                ],
                prompt=prompt, raw_response=raw,
            )

        steps = [str(s) for s in raw.get("steps", [])]

        errors = self._validate_plan(steps)
        return ExecutionPlan(
            goal=goal, status="plan", steps=steps, reasoning=reasoning,
            valid=(len(errors) == 0), validation_errors=errors,
            prompt=prompt, raw_response=raw,
        )

    def _validate_plan(self, steps):
        """Cheap, mechanical checks -- not a full dependency solver.
        Each check here was added in response to a specific real
        failure mode a planner-composed plan could hit."""
        errors = []

        if not steps:
            errors.append("Plan has no steps.")
            return errors

        if len(steps) > MAX_PLAN_LENGTH:
            errors.append(
                f"Plan has {len(steps)} steps, exceeding the {MAX_PLAN_LENGTH}-step "
                f"cap for a 9-tool system -- almost certainly a runaway/looping plan."
            )

        for step in steps:
            if step not in TOOL_DESCRIPTIONS:
                errors.append(f"Unknown tool: {step!r} -- not a real tool name.")

        for i in range(1, len(steps)):
            if steps[i] == steps[i - 1]:
                errors.append(f"Duplicate consecutive step: '{steps[i]}' appears twice in a row.")

        return errors
