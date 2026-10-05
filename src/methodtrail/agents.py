"""LLM roles with narrow responsibilities and typed outputs."""

from __future__ import annotations

import json
from typing import Any

from .llm import StructuredLLM
from .schemas import (
    AssessmentArtifact,
    CandidateProposalArtifact,
    CandidateSelectionArtifact,
    ChangeRequestArtifact,
    CodePlanArtifact,
    ExecutionDecisionArtifact,
    HypothesisArtifact,
    ImplementationReviewArtifact,
    RecoveryArtifact,
    ResearchState,
    RunArtifact,
    TaskContract,
)

SYSTEM = """You are one role in MethodTrail, an autonomous experiment system.
Use only facts in the supplied context. Return one JSON object matching the requested schema.
Do not claim a run happened unless it is present in the context. Keep code changes within the task contract."""


# LLM context is a working set, not a database dump.  The state object still
# keeps the complete graph and artifact history on disk, but prompts receive a
# bounded, role-neutral summary. Saved v6 prompts exceeded 200k characters.
# Packing must retain graph identity and structured experimental evidence
# while dropping large payloads rather than removing decision information.
_DEFAULT_CONTEXT_LIMIT = 80_000


class SourceContextTooLarge(ValueError):
    """Source cannot fit without hiding code needed for an exact patch."""


def _short(value: Any, limit: int) -> Any:
    """Return a stable, readable prefix for prompt text."""

    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: max(0, limit - 32)] + f" ...[truncated {len(value) - limit} chars]"


def _compact_log(value: Any, limit: int = 7000) -> Any:
    """Keep both the beginning and traceback/metric tail of process logs."""

    if not isinstance(value, str) or len(value) <= limit:
        return value
    head = max(1200, limit // 3)
    tail = limit - head - 80
    return value[:head] + f"\n...[truncated {len(value) - limit} chars]...\n" + value[-tail:]


def _compact_method(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"value": _short(value, 240)}
    return {
        key: _short(value.get(key), 360)
        for key in ("family", "components", "changed_factors", "target_scope")
        if value.get(key) not in (None, "", [], {})
    }


def _compact_node(value: Any, *, include_change: bool = False) -> dict[str, Any]:
    """Keep graph identity, evidence and edge meaning while dropping blobs."""

    if not isinstance(value, dict):
        return {"value": _short(value, 360)}
    result: dict[str, Any] = {}
    for key in (
        "node_id",
        "card_id",
        "variant_id",
        "node_type",
        "origin",
        "origin_reason",
        "status",
        "iteration",
        "title",
        "relation",
        "edge_type",
        "context_role",
        "distance",
        "traversal",
        "parent_variant_id",
        "metric",
        "wall_seconds",
        "priority",
        "relation_warning",
        "replan_reason",
        "candidate_adjustments",
        "evidence_parent_ids",
        "decision",
        "applicable_conditions",
        "method_family",
        "changed_factors",
        "method_components",
        "retrieval_source",
        "source_project_id",
        "read_only",
        "relevance",
    ):
        if key in value and value[key] not in (None, "", [], {}):
            result[key] = _short(value[key], 360)
    method = value.get("method") or value.get("method_descriptor")
    if method:
        result["method"] = _compact_method(method)
    for key in (
        "question",
        "research_question",
        "conclusion",
        "change_logic",
        "rationale",
        "reason",
        "evidence_summary",
    ):
        if value.get(key):
            result[key] = _short(value[key], 520)
    for key in ("measured_facts", "evidence"):
        if value.get(key):
            values = value[key] if isinstance(value[key], list) else [value[key]]
            result[key] = [_short(item, 320) for item in values[:6]]
    if include_change and value.get("change_request"):
        change = value["change_request"]
        if isinstance(change, dict):
            result["change_request"] = {
                key: _short(change.get(key), 420)
                for key in (
                    "title",
                    "mutation_class",
                    "relation",
                    "research_question",
                    "parent_variant_id",
                    "required_invariants",
                    "repeat_reason",
                    "rationale",
                )
                if change.get(key) not in (None, "", [], {})
            }
            if change.get("method"):
                result["change_request"]["method"] = _compact_method(change["method"])
    if value.get("edge_trace"):
        result["edge_trace"] = [
            _compact_node(edge, include_change=False)
            if isinstance(edge, dict)
            else _short(edge, 300)
            for edge in value["edge_trace"][:8]
        ]
    if isinstance(value.get("memory_graph_trace"), dict):
        trace = value["memory_graph_trace"]
        result["memory_graph_trace"] = {
            key: trace[key]
            for key in ("card_id", "score", "distance", "path")
            if key in trace
        }
        result["memory_graph_trace"]["edges"] = [
            {
                key: _short(edge[key], 520)
                for key in ("source", "target", "relation", "edge_type", "reason", "direction", "target_change")
                if key in edge
            }
            for edge in trace.get("edges", [])[:4]
        ]
    return result


def _sample_method_pool(state: dict[str, Any], limit: int = 36) -> list[dict[str, Any]]:
    """Retain the active lineage and pending alternatives before recent history."""

    pool = state.get("method_pool") or []
    anchor_ids = {state.get("incumbent_variant_id")}
    anchor_ids.update(row.get("node_id") for row in state.get("graph_context", []) if isinstance(row, dict))
    anchors = [row for row in pool if row.get("node_id") in anchor_ids and row.get("node_type") == "outcome"]
    pending = [row for row in pool if row.get("node_type") == "proposal" and row.get("status") in {"proposed", "ranked", "selected"}]
    chosen = []
    seen = set()
    for row in [*anchors, *pending, *reversed(pool)]:
        node_id = row.get("node_id")
        if node_id in seen:
            continue
        seen.add(node_id)
        chosen.append(_compact_node(row))
        if len(chosen) >= limit:
            break
    return chosen


def _compact_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"value": _short(value, 360)}
    candidate = value.get("candidate") if isinstance(value.get("candidate"), dict) else value
    result = {
        key: _short(value.get(key), 360)
        for key in ("index", "priority")
        if key in value
    }
    signal = value.get("graph_signal")
    if isinstance(signal, dict):
        result["graph_signal"] = {
            key: _short(signal.get(key), 360)
            for key in (
                "feasible",
                "family_status",
                "novelty",
                "unproductive_family",
                "required_seconds",
                "estimated_seconds",
                "failure_risk",
                "composition_penalty",
                "reason",
                "method_family",
                "family_outcome_count",
                "family_unproductive_count",
                "semantic_outcome_count",
                "semantic_unproductive_count",
                "semantic_novelty",
                "uncertainty",
                "semantic_positive_count",
                "promising_region",
                "plateau_rounds",
                "plateau_pressure",
                "novel_changed_factors",
                "exact_measured",
                "factor_stuck",
                "recent_non_improving",
                "gain_multiplier",
                "information_multiplier",
            )
            if signal.get(key) not in (None, "", [], {})
        }
    for key in (
        "title",
        "relation",
        "mutation_class",
        "research_question",
        "expected_gain",
        "information_gain",
        "estimated_seconds",
        "failure_risk",
        "parent_variant_id",
        "evidence_parent_ids",
        "repeat_reason",
    ):
        if candidate.get(key) not in (None, "", [], {}):
            result[key] = _short(candidate[key], 420)
    if candidate.get("method"):
        result["method"] = _compact_method(candidate["method"])
    # Rationale is useful for selection, but full prose and evidence duplicates
    # the hypothesis and graph context.  Keep a short decision-facing excerpt.
    if candidate.get("rationale"):
        result["rationale"] = _short(candidate["rationale"], 180)
    return result


def _compact_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"value": _short(value, 1200)}
    result = {
        key: _short(value.get(key), 1600)
        for key in ("summary", "expected_test", "affected_interfaces", "invariant_checks")
        if value.get(key) not in (None, "", [], {})
    }
    edits = value.get("edits") or []
    result["edits"] = []
    for edit in edits[:8]:
        if not isinstance(edit, dict):
            result["edits"].append(_short(edit, 1000))
            continue
        result["edits"].append(
            {
                key: _short(edit.get(key), 9000 if key in {"old_text", "new_text"} else 420)
                for key in ("operation", "path", "symbol", "purpose", "old_text", "new_text")
                if edit.get(key) is not None
            }
        )
    return result


def _compact_run(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"value": _short(value, 1200)}
    result = {
        key: _compact_log(value.get(key), 7000)
        if "stdout" in key or "stderr" in key
        else _short(value.get(key), 700)
        for key in (
            "command",
            "return_code",
            "timed_out",
            "wall_seconds",
            "metric",
            "metrics",
            "metric_constraints_passed",
            "stdout",
            "stderr",
            "evaluation_stdout",
            "evaluation_stderr",
            "output_files",
            "termination_reason",
        )
        if value.get(key) not in (None, "", [], {})
    }
    checkpoints = value.get("execution_checkpoints") or []
    if checkpoints:
        result["execution_checkpoints"] = [
            {
                key: _compact_log(item.get(key), 1400)
                if key in {"stdout_tail", "stderr_tail"}
                else _short(item.get(key), 500)
                for key in (
                    "stage",
                    "elapsed_seconds",
                    "process_running",
                    "output_bytes",
                    "output_growth",
                    "metric_file_exists",
                    "monitor_action",
                    "monitor_reason",
                    "stdout_tail",
                    "stderr_tail",
                    "monitor_error",
                    "final",
                )
                if item.get(key) not in (None, "", [], {})
            }
            for item in checkpoints[-8:]
            if isinstance(item, dict)
        ]
    return result


def _compact_state(value: Any) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    result = {
        key: _short(state[key], 900)
        for key in (
            "task_id",
            "iteration",
            "remaining_seconds",
            "best_metric",
            "incumbent_metric",
            "incumbent_variant_id",
            "plateau_rounds",
            "research_round",
            "repair_step",
        )
        if key in state
    }
    for key, cap in (
        ("recent_facts", 8),
        ("unresolved_questions", 6),
        ("path_hints", 10),
    ):
        if key in state:
            values = state[key] if isinstance(state[key], list) else [state[key]]
            result[key] = [
                _short(item, 700) if isinstance(item, str) else _compact_node(item)
                for item in values[-cap:]
            ]
    result["graph_context"] = [
        _compact_node(item)
        for item in (state.get("graph_context") or [])[:14]
    ]
    result["method_pool"] = _sample_method_pool(state)
    result["memory_context"] = [
        _compact_node(item, include_change=False)
        for item in (state.get("memory_context") or [])[:8]
    ]
    result["prior_evidence_context"] = [
        _compact_node(item, include_change=False)
        for item in (state.get("prior_evidence_context") or [])[:6]
    ]
    result["portfolio_context"] = [
        {
            key: _short(item.get(key), 360)
            for key in (
                "variant_id",
                "metric",
                "wall_seconds",
                "information_gain",
                "failure_risk",
                "signature",
            )
            if item.get(key) is not None
        }
        for item in (state.get("portfolio_context") or [])[:8]
        if isinstance(item, dict)
    ]
    profile = state.get("memory_graph_context")
    if isinstance(profile, dict):
        result["memory_graph_context"] = {
            key: (
                [_compact_node(row) for row in profile[key][:12]]
                if isinstance(profile[key], list)
                else _short(profile[key], 900)
            )
            for key in profile
            if key in {"directed", "nodes", "edges", "families", "summary", "recent", "edge_types", "card_count", "decisions", "method_families", "linked_card_count"}
        }
    return result


def _compact_payload(key: str, value: Any) -> Any:
    if isinstance(value, TaskContract):
        return value.llm_context()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if key == "state":
        return _compact_state(value)
    if key in {"plan", "code_plan"}:
        return _compact_plan(value)
    if key in {"run", "recovery", "implementation_review"}:
        if key == "run":
            return _compact_run(value)
        if isinstance(value, dict):
            return {
                name: _short(item, 1800) if isinstance(item, str) else item
                for name, item in value.items()
                if name in {
                    "failure_class",
                    "decision",
                    "replan_required",
                    "diagnosis",
                    "repair_directions",
                    "candidate_adjustments",
                    "preserve_question",
                    "return_to_reflection_reason",
                    "action",
                    "passed",
                    "scope_ok",
                    "invariants_ok",
                    "summary",
                    "checks",
                    "issues",
                }
            }
    if key in {"actual_diff", "diff"}:
        return value
    if key in {"ranked_candidates", "candidates"} and isinstance(value, list):
        return [_compact_candidate(item) for item in value[:40]]
    if key in {"repository_context", "repo_context", "parent_context"} and isinstance(value, list):
        compact = []
        for item in value:
            if isinstance(item, dict):
                compact.append(
                    {
                        name: item.get(name) if name == "content" else _short(item.get(name), 420)
                        for name in ("path", "content", "language", "retrieval_reason", "complete")
                        if item.get(name) is not None
                    }
                )
            else:
                compact.append(item)
        return compact
    if key in {"known_failures", "issues", "checks"} and isinstance(value, list):
        return [_short(item, 900) if isinstance(item, str) else _compact_node(item) for item in value[:12]]
    if isinstance(value, list):
        return [_short(item, 900) if isinstance(item, str) else item for item in value[:30]]
    if isinstance(value, dict):
        return {
            name: _short(item, 1200) if isinstance(item, str) else item
            for name, item in list(value.items())[:60]
        }
    return _short(value, 1200)


def _shrink_strings(value: Any, limit: int = 2600) -> Any:
    if isinstance(value, str):
        return _short(value, limit)
    if isinstance(value, list):
        return [_shrink_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _shrink_strings(item, limit) for key, item in value.items()}
    return value


class ReflectionAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def refine(
        self, contract: TaskContract, state: ResearchState, trigger: str | None = None
    ) -> HypothesisArtifact:
        if state.iteration <= 1 and state.incumbent_metric is None:
            instruction = """This is the first research round. Do a strong initial design pass rather than a tiny local tweak. Read the task contract, data description, metric, execution limits, and any supplied history. Form a coherent first predictor plan that covers the most important representation, model, training, composition, and output decisions. Treat distinct representations or model families as potentially complementary until evidence separates them. When useful components may complement each other, consider a composition challenger and component-level controls; select the direction from task facts and evidence, without requiring a particular candidate type. At the same time, state the independent evidence that would tell us which parts deserve deeper work. The result must still be testable in one run: write one clear initial hypothesis, its complete comparison plan, adoption conditions, and fallback conditions. When the output schema already works, spend the initial design effort on substantive predictive choices rather than formatting-only or probability-postprocessing tweaks."""
        else:
            instruction = """Refine the current observation into one research question. State what evidence would answer it,
how to compare alternatives, and what conditions lead to adoption or fallback. Later rounds should make changes more
evidence-driven and attributable than the initial design. Preserve an unmeasured alternative when it answers a concrete
question, but do not interrupt a supported local line merely because another family has not been tried. Once the
independent evaluator and execution checks have
verified the interface and output invariants, further formatting, warning-reporting, or numerically equivalent hygiene
changes belong to technical maintenance unless recorded evidence identifies an unresolved uncertainty that could change
the research conclusion. Prefer a substantive predictive or validation question over rechecking already satisfied
output invariants. After repeated non-improving experiments that share changed
factors, inspect whether the evidence supports a local continuation, a measured
backtrack, or a new direction; do not treat any one of these as mandatory."""
        plateau_rounds = int(getattr(state, "plateau_rounds", 0) or 0)
        if plateau_rounds >= 2:
            instruction += f"""

The archive has not produced a new best primary metric for {plateau_rounds} completed rounds. Treat this as evidence
to inspect the promising path, its unresolved question, and nearby alternatives more carefully. You may continue a
promising local line, backtrack, or open a new direction; choose the balance from the evidence rather than following a
mandatory novelty rule. A new label alone is not evidence, and a weaker exploratory result remains an archived
stepping stone. Keep the best checkpoint available and state what observation would make you stay with or leave the
current line."""
        if state.prior_evidence_context:
            instruction += """

The context may contain prior_evidence_context from another completed project.
Treat it as read-only comparative evidence, not as current-project graph state:
do not reuse its IDs as code parents. Use it only to avoid forgetting a
measured direction and decide whether the current question should test it again
under the current task protocol."""
        user = _context(
            instruction,
            contract=contract,
            state=state,
            trigger=trigger,
        )
        return self.llm.complete(SYSTEM, user, HypothesisArtifact)


class ChooseAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def propose(
        self,
        contract: TaskContract,
        state: ResearchState,
        hypothesis: HypothesisArtifact,
        coverage_feedback: str | None = None,
        constraint_feedback: str | None = None,
    ) -> CandidateProposalArtifact:
        initial = state.iteration <= 1 and state.incumbent_metric is None
        phase_instruction = (
            """This is the initial design round. Explore broadly enough to produce a coherent strong starting
predictor and several genuinely different, executable directions around it. Cover distinct method families when the
contract supports them, such as representation/features, model family, training objective, calibration, or output
handling. Include an integrated first candidate plus component-level candidates that can later be tested independently.
If the candidate set contains multiple useful families or components, consider
an explicit composition or fusion challenger, but do not assume that composition
is always the right direction. Do not treat useful families as mutually
exclusive without evidence. Return only as many candidates as add information;
two to four is common, but one is acceptable when no other executable direction
adds information. Do not pad the list with duplicates
or cosmetic rewrites when the graph cannot support that breadth. Give substantive
representation, model, objective, training, and composition alternatives
priority over formatting-only changes when the output schema already works."""
            if initial
            else
            """This is a refinement round. Use the graph evidence to focus on a small set of high-value, attributable
 changes. Consider an unmeasured family, an explicit backtrack, or a controlled
change when it answers a visible question; otherwise deepen a recent supported improvement. Keep an executable alternative in the frontier when
the evidence leaves a distinct family or factor untested. Merge existing pending candidates with new
proposals and keep the comparison set small, usually two to four distinct executable choices; one is acceptable when
no other choice adds information. Preserve a local refinement when it tests a concrete unresolved cause. A
local calibration improvement does not make all other representation families
ineligible."""
        )
        plateau_rounds = int(getattr(state, "plateau_rounds", 0) or 0)
        if plateau_rounds >= 2:
            phase_instruction += f"""

The best metric has not improved for {plateau_rounds} rounds. Use the archive to decide whether the current line still
has an unresolved, promising comparison or whether a new direction or backtrack is justified. Do not create a new
family name only to satisfy the plateau signal, and do not assume that a local refinement is useless. Keep the batch
small and executable; the controller will retain useful unselected candidates for later consideration."""
        if coverage_feedback:
            phase_instruction += f"\n\nProgrammatic coverage review found a gap:\n{coverage_feedback}\nRevise the candidate batch to close this gap while keeping every candidate executable and tied to the hypothesis."
        if constraint_feedback:
            phase_instruction += (
                "\n\nThe previous selected candidate was returned for contract re-planning. "
                "Keep the same research question and intended comparison. "
                "Treat the review as a hard design constraint: do not recreate "
                "the rejected invariant. Revise the candidate's method map, "
                "settings, and declared checks only as needed so the experiment "
                "is executable and each check is observable at the right stage; "
                "or choose another candidate that tests the same question. "
                "Explain the revision in its rationale. Feedback:\n"
                + constraint_feedback
            )
        user = _context(
            f"""{phase_instruction}

Expand the project method graph with a small set of executable method nodes. Return only
methods that add a distinct implementation, configuration, composition, or
recovery possibility. Usually two to four relevant choices are enough; one is
acceptable when no other executable choice adds information; do not pad the
batch merely to reach a count, and do not repeat a method already present in
the supplied graph context. Each candidate must be tied to the current
hypothesis and one graph relation. Separate configuration-only changes from
composition, implementation, and recovery changes. For every candidate fill
method.family, method.components, method.changed_factors, and method.target_scope
so the program can compare it with related nodes. Give conservative estimates
grounded in the supplied history. The program will attach each node, compute a
path priority, and keep unselected nodes in the method pool. The current task
may start with an empty method graph, so derive the first methods from the task
and data contract rather than assuming a supplied predictor. Treat the method
graph as a coverage map: when a family already has repeated non-improving
outcomes and no nearby positive evidence, consider an unmeasured family or
state a genuinely new factor that answers a visible unresolved question. Do
not spend a proposal slot on a
reworded version of a measured method. If you intentionally repeat a measured
method, fill repeat_reason with the new seed, fold, protocol, or unresolved
cause; without that reason the controller will leave the old result in history
instead of executing it again. Use the memory-graph summary when
deciding whether a direction is already sufficiently explored. The graph is
directed: an edge points from an earlier method or candidate parent to the
newer branch. If a poor result should be retested from an older measured
version, set parent_variant_id to that existing outcome node ID from
graph_context, where traversal is marked backtrack_ancestor; otherwise leave
it null. This is a deliberate directed backtrack and must not point to a
proposal node or invent an ID. Any prior_evidence_context is read-only evidence
from another project: it can suggest a direction, but its IDs cannot be used
as parent_variant_id or evidence_parent_ids in this project. You own the semantic relation field: declare
ablate only when the research question intentionally removes a named
component; declare deepen, combine, or explore when that is the experiment's
meaning. The graph may flag a mismatch between the relation and component map,
but it will not silently relabel your experiment. When a proposal extends,
contrasts with, or combines measured outcomes, set evidence_parent_ids to every
relevant existing outcome ID. Leave it empty only when the proposal is genuinely
independent or no measured outcome exists yet. These are evidence links, not code
parents, and must not be proposal IDs. When the contract uses an independent evaluator, make
required_invariants checkable before or after execution at the correct stage:
solution code must write valid predictions, public validation comparisons can
be printed by that code, and the experiment controller compares the private
primary metric after the evaluator runs. Never require solution code to report
or compare an unavailable private score. The program priority is a decision aid,
not an instruction to follow one path: you may select a lower-ranked feasible
candidate, retain a promising family, defer a weak candidate, or stop adding new
proposals when the existing frontier already answers the question. Explain the
evidence for that choice in the returned reason.""",
            contract=contract,
            state=state,
            hypothesis=hypothesis,
        )
        return self.llm.complete(SYSTEM, user, CandidateProposalArtifact)

    def select(
        self,
        contract: TaskContract,
        state: ResearchState,
        hypothesis: HypothesisArtifact,
        ranked_candidates: list[dict[str, Any]],
    ) -> CandidateSelectionArtifact:
        user = _context(
            """Select one feasible candidate index after reading the program-computed path priorities. The index is the
original candidate index in each supplied row; do not use the row's position after sorting. Rows marked infeasible
are not selectable. Explain how the method graph's family status, prior evidence, expected gain, information value,
cost, and risk fit the current question and remaining budget. Consider an
unmeasured family when its value is comparable to a repeatedly unproductive
family, but keep a promising supported path eligible when the evidence justifies
continuing it. Compare a repeatability experiment with other candidates using the specific
measured uncertainty it can resolve, its comparison protocol, and an explicit closure
condition. Account for dataset and sampling differences when interpreting validation
and independent-evaluation scores. When the comparison has answered its question,
close that uncertainty and return to the available frontier. Do not select an index outside the
supplied list. If prior_evidence_context is present, use it as read-only evidence
from another project; do not select a historical ID as a current parent.""",
            contract=contract,
            state=state,
            hypothesis=hypothesis,
            ranked_candidates=ranked_candidates,
        )
        return self.llm.complete(SYSTEM, user, CandidateSelectionArtifact)


class CodingAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def write_plan(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        repo_context: list[dict[str, str]],
        entrypoint_exists: bool = False,
        known_failures: list[dict[str, Any]] | None = None,
    ) -> CodePlanArtifact:
        initial = not entrypoint_exists
        edit_scope = (
            """For the initial candidate, a complete solution implementation is allowed and preferred when there is no
usable incumbent. Build the strongest coherent runnable predictor you can from the task contract and supplied data,
while keeping the entrypoint and evaluation interface intact."""
            if initial
            else
            """For a measured incumbent, keep the edit local and attributable to the selected question; do not rebuild
unrelated parts of the predictor."""
        )
        user = _context(
            f"""{edit_scope}

Produce an executable edit plan for the selected experiment. The entrypoint_exists flag is the authoritative state
of the current workspace: create the entrypoint only when it is absent. For a new file use a create edit with its full
contents. For an existing file prefer replace with an exact, unique local snippet; use replace_symbol with a top-level
function/class name when one complete symbol must change. Use rewrite only when the declared solution entrypoint
genuinely needs a new complete implementation and this is the initial root candidate. Child candidates and all
recovery steps must use replace or replace_symbol; the workspace rejects whole-file rewrites for them. Never use create for an existing file and never append a
second program below an existing main guard. Do not edit protected or data files. Include the required solution
entrypoint and keep the change focused on the research question. Before editing,
reconcile protocol details stated in the ChangeRequest, including fold count,
seed, n-gram range, calibration, and metric conventions. If two declarations
conflict, do not silently choose one: surface the exact conflict in the plan
and invariant_checks so the candidate can be replanned. Treat every
required_invariant in the ChangeRequest as an acceptance check: explain in the
    plan how the edit satisfies it, and list a concrete check for each one in
    invariant_checks. For an independently evaluated task, implement only the
    code-side portion of a metric invariant; the controller checks the private
    metric after execution. Do not invent or print a private score. Do not
silently change preprocessing, model family, validation, calibration, or
inference components that are outside the declared change. State a test that
should pass.
If the task has no separate configuration file, a configuration experiment may
change a small source-level constant; keep that edit local and explicit.""",
            contract=contract,
            change=change,
            repository_context=repo_context,
            entrypoint_exists=entrypoint_exists,
            known_failures=known_failures or [],
        )
        return self.llm.complete(SYSTEM, user, CodePlanArtifact)

    def repair(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        run: RunArtifact,
        recovery: RecoveryArtifact,
        repo_context: list[dict[str, str]],
        known_failures: list[dict[str, Any]] | None = None,
        implementation_review: ImplementationReviewArtifact | None = None,
        parent_context: list[dict[str, str]] | None = None,
    ) -> CodePlanArtifact:
        user = _context(
            """Repair the implementation that just failed. Read the recorded stderr, diagnosis, and current
repository files. Use exact local replace edits or replace_symbol for one top-level function/class. Never use rewrite
in a repair plan: the workspace rejects whole-file rewrites for recovery steps. Use create only when the file is absent.
Never append a second implementation or duplicate a top-level function. The workspace validates syntax and duplicate
definitions before execution.
Preserve the current research question unless the diagnosis says it cannot be tested. Do not edit protected or data
files and do not invent results. Compare the current stderr with the known failure records before proposing an edit.
        Never repeat an edit that already failed with the same signature. If the same failure remains after a repair,
inspect the latest repository context and propose a different minimal edit that addresses the still-open issue. Do not
abandon merely because an earlier patch failed; only return an abandon action when the diagnosis establishes that the
task contract, available dependencies, or execution environment makes this candidate technically impossible. The revised
files will be executed immediately without human intervention. If an
implementation_review object is present, treat its failed checks and issues as
an acceptance checklist: fix each named issue with the smallest local edit,
preserve every other parent component, and do not rewrite the whole entrypoint
just to silence the review. Put one concrete check per review issue in
         invariant_checks. The immutable parent_context is the source of truth
         for the version selected by the research agent. Compare it with the
         current repository before editing; do not reconstruct an earlier
         parent from a summary or stale diff. Keep earlier valid patches and
         make the smallest edit against the latest source.""",
            contract=contract,
            change=change,
            run=run,
            recovery=recovery,
            implementation_review=implementation_review,
            parent_context=parent_context or [],
            repository_context=repo_context,
            known_failures=known_failures or [],
        )
        return self.llm.complete(SYSTEM, user, CodePlanArtifact)

    def review_implementation(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        plan: CodePlanArtifact,
        diff: str,
        repo_context: list[dict[str, str]],
        known_failures: list[dict[str, Any]] | None = None,
        initial_candidate: bool = False,
        parent_context: list[dict[str, str]] | None = None,
    ) -> ImplementationReviewArtifact:
        """Compatibility wrapper for callers of the earlier Coding API."""

        return ImplementationReviewAgent(self.llm).review(
            contract,
            change,
            plan,
            diff,
            repo_context,
            known_failures=known_failures,
            initial_candidate=initial_candidate,
            parent_context=parent_context,
        )


class ImplementationReviewAgent:
    """Independently checks a realized patch before it can consume run budget.

    The Coding Agent writes and explains a patch. This role sees the applied
    diff and source context and can reject a patch that silently changed the
    tested factor. The deterministic ``Verifier`` remains the interface and
    dependency check after this semantic gate.
    """

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def review(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        plan: CodePlanArtifact,
        diff: str,
        repo_context: list[dict[str, str]],
        known_failures: list[dict[str, Any]] | None = None,
        initial_candidate: bool = False,
        parent_context: list[dict[str, str]] | None = None,
    ) -> ImplementationReviewArtifact:
        # Lightweight test doubles and custom providers can opt out. They still
        # go through WorkspaceManager and Verifier; production providers opt in
        # through the capability flag.
        if not getattr(self.llm, "supports_implementation_review", False):
            return ImplementationReviewArtifact(
                passed=True,
                summary="implementation review disabled for this local provider",
                checks=["static workspace verification remains enabled"],
            )
        user = _context(
            """Act as an independent implementation gate for one selected experiment. Read the selected research
question, the declared required invariants, the Coding Agent plan, the cumulative diff from the original parent, and the current source
context. Check that the code tests the requested factor, keeps the task interface and output schema intact, and does
not silently substitute a fallback or data leak. If initial_candidate is true, there is no usable parent predictor:
the complete first implementation and its model/vectorizer choices are in scope, so do not reject them merely because
they were not enumerated as separate mutation factors. Still reject missing invariants, validation leakage, invalid
outputs, or code that does not implement the selected method. If initial_candidate is false, preserve every unrequested
parent component and reject a diff that changes more factors than the ChangeRequest declares. Removing a component is
not automatically an ablation: when relation is ablate, the removed component must be named and all retained
components and the comparison protocol must stay fixed; when relation is deepen, combine, or explore, a redesign may
remove or replace a component only when the ChangeRequest's method map, changed_factors, rationale, and invariants
explicitly describe that new method. Treat relation_warning as a prompt to resolve the ambiguity, not as permission to
silently rewrite the research question. This review is not metric assessment and must not
propose a new research question. If the independent evaluator owns the primary
metric, verify that the solution emits valid predictions and implements the
public comparison. Do not demand that solution code know, print, or compare
the private score before evaluation, even when a declared invariant combines
code-side and evaluator-side checks. Leave the evaluator-side comparison to
the controller. If the selected candidate's own required invariants or method
map contradict each other, require unavailable data, or require solution code
to see a private evaluator result, source edits cannot repair that contract.
In that case set passed=false, decision="replan", replan_required=true, and
list exact candidate_adjustments: which declaration is impossible, what can
be checked instead, and which research question and intended comparison must
be kept. Do not silently change the research question or relabel an ablation.
Do not use replan for an ordinary code bug: set passed=false,
decision="repair", and give concrete source-level issues. A replan returns to
candidate selection in the same research round; neither outcome counts as
measured evidence. During technical repair, earlier patches remain in the current source and cumulative diff:
judge the complete resulting implementation, not whether the latest small patch repeats every earlier change.
Do not add requirements beyond the selected contract and required invariants. Set decision="pass" only when passed=true.
The parent_context is immutable evidence of the code version this candidate was
supposed to extend. List any removed parent component explicitly and check that
the ChangeRequest relation and changed_factors authorize that removal.""",
            contract=contract,
            change=change,
            plan=plan,
            actual_diff=diff,
            repository_context=repo_context,
            known_failures=known_failures or [],
            initial_candidate=initial_candidate,
            parent_context=parent_context or [],
        )
        return self.llm.complete(SYSTEM, user, ImplementationReviewArtifact)


class ExecutionMonitorAgent:
    """Decides whether a long-running process is healthy enough to continue."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def decide(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        checkpoint: dict[str, Any],
    ) -> ExecutionDecisionArtifact:
        user = _context(
            """Inspect this live execution checkpoint. Continue by default: elapsed time alone is never a reason to
terminate a valid experiment. Terminate only when the process is clearly stuck, repeatedly emitting the same fatal
error, violating the task contract, or producing an unrecoverable resource/format failure. Normal sparse logging,
long model training, and delayed metric files are healthy possibilities. If termination is requested, state the exact
observable evidence so Repair Agent can act on it. Do not change the research question or score the experiment.""",
            contract=contract,
            state=state,
            change=change,
            checkpoint=checkpoint,
        )
        return self.llm.complete(SYSTEM, user, ExecutionDecisionArtifact)


class EvidenceAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def investigate(
        self,
        contract: TaskContract,
        state: ResearchState,
        run: RunArtifact,
        assessment: AssessmentArtifact,
    ) -> HypothesisArtifact:
        user = _context(
            """Results are incomplete or inconsistent. Turn the conflict into one narrower question.
Request an ablation, matched control, segmented validation, or repeated run that can distinguish plausible causes.""",
            contract=contract,
            state=state,
            run=run,
            assessment=assessment,
        )
        return self.llm.complete(SYSTEM, user, HypothesisArtifact)


class RecoveryAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def diagnose(
        self,
        contract: TaskContract,
        state: ResearchState,
        run: RunArtifact,
        known_failures: list[dict[str, Any]] | None = None,
        implementation_review: ImplementationReviewArtifact | None = None,
    ) -> RecoveryArtifact:
        user = _context(
            """Classify only the technical failure. Keep the research question and
the selected method target unchanged. Decide whether to continue repairing the
same candidate or switch to a nearby implementation that tests the same question.
When the implementation review explicitly marks replan_required, preserve that
decision and return action="replan_candidate"; do not ask Coding Agent to patch
an impossible contract.
Use abandon_candidate only when the diagnosis establishes that the task contract,
available dependencies, permissions, or execution environment make this candidate
technically impossible; repeated review failures or a bad patch are reasons to
request another repair, not evidence of impossibility. Return one concrete next
repair action and never claim that a code or environment failure disproves the
research hypothesis. If
implementation_review is present, treat its structured issues and failed
checks as authoritative constraints. Explain how the next repair addresses
each issue, preserve the original research question, and avoid changing
unmentioned components.""",
            contract=contract,
            state=state,
            run=run,
            implementation_review=implementation_review,
            known_failures=known_failures or [],
        )
        return self.llm.complete(SYSTEM, user, RecoveryArtifact)


class AssessAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def assess(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        run: RunArtifact,
    ) -> AssessmentArtifact:
        user = _context(
            """Assess a successfully measured run. Compare it with the project-wide
incumbent and the relevant parent, while also considering information value and
future path potential. Adopt only when the measured primary metric improves the
incumbent and all declared constraints pass. A worse but informative or
promising result should remain in the method graph as evidence or deferred
work; it must not replace the incumbent. Choose 'evidence' when the next step
should isolate uncertainty, 'defer' for a promising but currently inferior
result, and 'stop' only when no useful executable path remains. Set
question_status='resolved' when the declared comparison is answered, or
'inconclusive' when repeating the same protocol would not add evidence. Use
'open' or 'frontier' only when the next question requires a materially
different experiment. A resolved or inconclusive question must not request a
near-duplicate Evidence Agent follow-up.""",
            contract=contract,
            state=state,
            change=change,
            run=run,
        )
        return self.llm.complete(SYSTEM, user, AssessmentArtifact)


class MemoryAgent:
    """Creates a concise, retrievable conclusion. It never manufactures run facts."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def summarize(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        run: RunArtifact,
        assessment: AssessmentArtifact,
    ) -> AssessmentArtifact:
        user = _context(
            """Write a reusable conclusion for the experiment graph. Preserve the measured fact, explain
where it applies, and identify uncertainty that should remain visible. Separate measured facts from the reusable
conclusion. Keep the decision unchanged.""",
            contract=contract,
            change=change,
            run=run,
            assessment=assessment,
        )
        return self.llm.complete(SYSTEM, user, AssessmentArtifact)


def _context(
    instruction: str, *, context_limit: int = _DEFAULT_CONTEXT_LIMIT, **payload: Any
) -> str:
    """Build a bounded prompt without losing the graph's directed evidence.

    The complete state remains available in the artifact store.  Prompt
    context is sampled by role: graph rows keep IDs, parent/edge meaning,
    method family and metric; repository rows keep only the files relevant to
    the edit; candidate rows keep every original index so selection remains
    valid.  The final shrink is a safety valve for unusually large source
    files. Editable source and cumulative diffs are never shortened: an
    incomplete file is not usable evidence for an exact local patch.
    """

    packed = {
        key: _compact_payload(key, value)
        for key, value in payload.items()
        if value is not None
    }
    protected = {
        key: packed.pop(key)
        for key in ("repository_context", "repo_context", "actual_diff", "diff")
        if key in packed
    }
    protected_size = len(json.dumps(protected, ensure_ascii=False, separators=(",", ":")))
    if protected_size > context_limit:
        raise SourceContextTooLarge(
            f"Complete source and diff need {protected_size} characters; context limit is {context_limit}. "
            "Increase the coding context budget or select fewer editable files; source was not truncated."
        )
    prose_limit = max(0, context_limit - protected_size)
    serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > prose_limit:
        # Prefer preserving all candidate indices and the latest state while
        # making prose and source excerpts progressively smaller.
        packed = _shrink_strings(packed, limit=1400)
        serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > prose_limit:
        for key in (
            "method_pool",
            "graph_context",
            "memory_context",
            "prior_evidence_context",
            "portfolio_context",
        ):
            if isinstance(packed.get("state"), dict) and isinstance(packed["state"].get(key), list):
                packed["state"][key] = packed["state"][key][:8]
        serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > prose_limit:
        # This should be rare after the per-field caps.  Keep the instruction,
        # contract, state scalars, and a clearly marked compact remainder.
        compact: dict[str, Any] = {}
        for key in (
            "contract",
            "state",
            "change",
            "hypothesis",
            "plan",
            "run",
            "recovery",
            "ranked_candidates",
            "candidates",
        ):
            if key in packed:
                value = packed[key]
                compact[key] = value
        compact["context_notice"] = (
            "Older graph details were compacted after reaching the prompt limit; "
            "use the supplied IDs and current repository files for the next decision."
        )
        packed = _shrink_strings(compact, 900)
    packed.update(protected)
    serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if protected and len(serialized) > context_limit:
        raise SourceContextTooLarge(
            "Complete source, diff and required task context exceed the coding context budget; "
            "source was not truncated."
        )
    return f"{instruction}\n\nContext:\n{serialized}"


# Names used by the design document.  These thin subclasses preserve the old
# role APIs while making the state-machine responsibilities explicit.
class QuestionAgent(ReflectionAgent):
    """Forms a new question or refines one after valid evidence."""


class MethodGraphAgent(ChooseAgent):
    """Expands the project method pool and selects a graph path."""


class RepairAgent(RecoveryAgent):
    """Diagnoses technical failures without changing the research question."""


class AssessmentMemoryAgent:
    """The valid-run reviewer and memory writer share one evidence contract."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.assess_agent = AssessAgent(llm)
        self.memory_agent = MemoryAgent(llm)

    def assess(self, *args: Any, **kwargs: Any) -> AssessmentArtifact:
        return self.assess_agent.assess(*args, **kwargs)

    def summarize(self, *args: Any, **kwargs: Any) -> AssessmentArtifact:
        return self.memory_agent.summarize(*args, **kwargs)
