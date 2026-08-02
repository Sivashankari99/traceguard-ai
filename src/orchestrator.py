"""
Orchestrator — Layer 2 / Execution Layer, now with a Planner fallback.

    IntentRouter
         │
    High confidence?
      │        │
     YES       NO
      │        │
      ▼        ▼
  Workflow   Planner.create_plan()
      │        │
      │        ▼
      │   ExecutionPlan.steps
      │        │
      └───┬────┘
          ▼
    Orchestrator.run()
          ▼
      Execute steps

The key design choice (per user decision): WORKFLOWS[intent]["steps"]
and Planner.create_plan(...).steps are the SAME datatype -- a plain
list of tool-name strings. The execution loop below never needs to know
or care where its step list came from; a workflow-provided list and a
planner-provided list are executed identically.

This is also why the tool map below is now trivial: every tool in
tools.py takes the single `context` dict directly and does its own
smart extraction internally (see tools.py's docstring) -- there are no
adapter lambdas here anymore, because there's nothing left for an
adapter to do.
"""

import time
from dataclasses import dataclass, field

from src.tools import TraceGuardTools, ToolResult
from src.workflows import WORKFLOWS
from src.intent_router import IntentRouter
from src.planner import Planner


@dataclass
class OrchestratorResult:
    workflow: str
    entity_id: str
    confidence: float
    reason: str
    steps_run: list
    final_response: object
    success: bool
    tool_call_count: int
    execution_time_seconds: float
    trace_log: list = field(default_factory=list)
    plan_reasoning: str = None      # only set when a Planner-composed plan ran
    plan_prompt: str = None         # logging: exact prompt sent to the LLM for planning
    plan_raw_response: dict = None  # logging: the LLM's raw parsed JSON, before validation


class Orchestrator:
    def __init__(self, engine):
        self.tools = TraceGuardTools(engine)
        # Reuses the embedding model TraceGuard already loaded for
        # retrieval — no separate model/API call for semantic fallback.
        self.router = IntentRouter(embedding_model=engine.embedding_model)
        # Reuses engine._call_llm — same LLM-call infrastructure
        # assess_impact already uses, no new LLM plumbing.
        self.planner = Planner(engine)
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self):
        """Every tool takes `context` directly now, so this is just a
        name -> bound method lookup. No adapters needed."""
        return {
            "lookup": self.tools.lookup,
            "retrieve": self.tools.retrieve,
            "trace": self.tools.trace,
            "assess_impact": self.tools.assess_impact,
            "validate": self.tools.validate,
            "determine_baseline": self.tools.determine_baseline,
            "baseline_evidence": self.tools.baseline_evidence,
            "evidence_fusion": self.tools.evidence_fusion,
            "full_impact_analysis": self.tools.full_impact_analysis,
        }

    def _call_tool(self, step, context):
        tool_fn = self._tool_map.get(step)
        if tool_fn is None:
            raise ValueError(f"Unknown tool: {step!r}")
        return tool_fn(context)

    def _build_final_response(self, steps, context):
        """final_response is normally just the last step's output — fine
        for every workflow except one that ends with full_impact_analysis
        followed by evidence_fusion. Returning evidence_fusion's
        comparison dict alone would silently drop the rich analyze()
        result (impact_report_df, overall_assessment, validation) that
        ran before it. So specifically when BOTH of those steps ran
        (whether via the fixed full_impact_analysis workflow or a
        planner-composed plan that happens to include them): merge
        evidence_fusion's result INTO the analyze() dict as an added
        key, rather than replacing it. Checked by what's actually in
        context, not by the router's intent label, since a planner
        plan can include these same two steps too.
        """
        if "full_impact_analysis" in context and "evidence_fusion" in context:
            merged = dict(context["full_impact_analysis"])
            merged["evidence_fusion"] = context["evidence_fusion"]
            return merged
        if steps:
            return context.get(steps[-1])
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, query_text):
        start = time.perf_counter()
        decision = self.router.route(query_text)

        if decision.intent != "clarification":
            # Known workflow, high confidence -- fast path, unchanged.
            steps = WORKFLOWS[decision.intent]["steps"]
            intent_label = decision.intent
            plan_reasoning = None
            plan_prompt = None
            plan_raw_response = None
        else:
            # Low confidence on BOTH lexical and semantic signals. Rather
            # than give up immediately, let the Planner attempt to
            # compose a fresh sequence from the raw tools -- this is the
            # fallback for a real request that just doesn't fit any of
            # the 5 fixed workflows (e.g. "Can you help me understand
            # CR-00123?", which is genuinely lookup -> trace, not any
            # named workflow).
            goal = query_text  # the goal IS the query for now; no separate goal-inference step yet
            plan = self.planner.create_plan(goal, query_text, entity_id=decision.entity_id)

            if not plan.valid or not plan.steps:
                return OrchestratorResult(
                    workflow="clarification",
                    entity_id=decision.entity_id,
                    confidence=decision.confidence,
                    reason=(
                        f"{decision.reason} Planner also could not build a "
                        f"valid plan: {'; '.join(plan.validation_errors) or 'no steps proposed'}."
                    ),
                    steps_run=[],
                    final_response=(
                        "I'm not confident enough to route that automatically, and "
                        "couldn't compose a fallback plan either. Could you rephrase, "
                        "or provide a specific artifact ID (e.g. CR-00123)?"
                    ),
                    success=False,
                    tool_call_count=0,
                    execution_time_seconds=time.perf_counter() - start,
                    plan_reasoning=plan.reasoning,
                    plan_prompt=plan.prompt,
                    plan_raw_response=plan.raw_response,
                )

            steps = plan.steps
            intent_label = f"planner:{plan.goal}"
            plan_reasoning = plan.reasoning
            plan_prompt = plan.prompt
            plan_raw_response = plan.raw_response

        trace_log = []
        context = {"query_text": query_text, "entity_id": decision.entity_id}

        for step in steps:
            result: ToolResult = self._call_tool(step, context)
            trace_log.append(result)
            if not result.ok:
                return OrchestratorResult(
                    workflow=intent_label,
                    entity_id=decision.entity_id,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    steps_run=[r.tool for r in trace_log],
                    final_response=f"Step '{step}' failed: {result.error}",
                    success=False,
                    tool_call_count=len(trace_log),
                    execution_time_seconds=time.perf_counter() - start,
                    trace_log=trace_log,
                    plan_reasoning=plan_reasoning,
                    plan_prompt=plan_prompt,
                    plan_raw_response=plan_raw_response,
                )
            context[step] = result.data

        return OrchestratorResult(
            workflow=intent_label,
            entity_id=decision.entity_id,
            confidence=decision.confidence,
            reason=decision.reason,
            steps_run=[r.tool for r in trace_log],
            final_response=self._build_final_response(steps, context),
            success=True,
            tool_call_count=len(trace_log),
            execution_time_seconds=time.perf_counter() - start,
            trace_log=trace_log,
            plan_reasoning=plan_reasoning,
            plan_prompt=plan_prompt,
            plan_raw_response=plan_raw_response,
        )
