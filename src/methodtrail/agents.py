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
# bounded, role-neutral summary.  This is especially important for proposal
# and selection calls: sending every node's full ``change_request`` made the
# v6 prompts grow past 200k characters and caused the research budget to be
# spent serialising and rereading old evidence.
_DEFAULT_CONTEXT_LIMIT = 80_000


def _short(value: Any, limit: int) -> Any:
    """Return a stable, readable prefix for prompt text."""

    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: max(0, limit - 32)] + f" ...[truncated {len(value) - limit} chars]"


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
        "variant_id",
        "node_type",
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
    return result


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
    return {
        key: _short(value.get(key), 7000 if "stdout" in key or "stderr" in key else 700)
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
        )
        if value.get(key) not in (None, "", [], {})
    }


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
        for item in (state.get("graph_context") or [])[-14:]
    ]
    result["method_pool"] = [
        _compact_node(item, include_change=False)
        for item in (state.get("method_pool") or [])[-36:]
    ]
    result["memory_context"] = [
        _compact_node(item, include_change=False)
        for item in (state.get("memory_context") or [])[:8]
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
            if key in {"nodes", "edges", "families", "summary", "recent"}
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
        return _short(value, 18_000)
    if key in {"ranked_candidates", "candidates"} and isinstance(value, list):
        return [_compact_candidate(item) for item in value[:40]]
    if key in {"repository_context", "repo_context"} and isinstance(value, list):
        compact = []
        for item in value[:8]:
            if isinstance(item, dict):
                compact.append(
                    {
                        name: _short(item.get(name), 9000 if name == "content" else 420)
                        for name in ("path", "content", "language", "retrieval_reason")
                        if item.get(name) is not None
                    }
                )
            else:
                compact.append(_short(item, 9000))
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
            instruction = """This is the first research round. Do a strong initial design pass rather than a tiny local tweak. Read the task contract, data description, metric, execution limits, and any supplied history. Form a coherent first predictor plan that covers the most important representation, model, training, composition, and output decisions. Treat distinct representations or model families as potentially complementary until evidence separates them. When two or more useful components are available, require a composition challenger as well as component-level controls. At the same time, state the independent evidence that would tell us which parts deserve deeper work. The result must still be testable in one run: write one clear initial hypothesis, its complete comparison plan, adoption conditions, and fallback conditions. When the output schema already works, spend the initial design effort on substantive predictive choices rather than formatting-only or probability-postprocessing tweaks."""
        else:
            instruction = """Refine the current observation into one research question. State what evidence would answer it,
how to compare alternatives, and what conditions lead to adoption or fallback. Later rounds should make changes more
evidence-driven and attributable than the initial design. Do not let a local calibration or implementation gain erase
an unmeasured complementary representation or composition. After two consecutive non-improving experiments that share
the same changed factors, prefer a backtrack to a measured ancestor or an orthogonal method family."""
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
If the candidate set contains multiple useful families or components, include at
least one explicit composition or fusion challenger; do not treat those families
as mutually exclusive without evidence. Do not force a fixed number of candidates
or pad the list; stop when the important design space is represented. Give
substantive representation, model, objective, training, and composition
alternatives priority over formatting-only changes when the output schema already
works."""
            if initial
            else
            """This is a refinement round. Use the graph evidence to focus on a small set of high-value, attributable
changes. Prefer an unmeasured family, an explicit backtrack, or a controlled
composition that answers a visible question. Keep at least one orthogonal or
composition challenger in the frontier when the graph contains complementary
unmeasured factors. A local calibration improvement does not make all other
representation families ineligible."""
        )
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

Expand the project method graph with one or more new, executable method nodes.
There is no fixed candidate count. Return only methods that add a distinct
implementation, configuration, composition, or recovery possibility. Stop when
the current question has enough relevant alternatives, and explain why in
discovery_complete/discovery_reason. Do not repeat a method already present in
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
outcomes, prefer an unmeasured family or state a genuinely new factor that
answers a visible unresolved question. Do not spend a proposal slot on a
reworded version of a measured method. Use the memory-graph summary when
deciding whether a direction is already sufficiently explored. The graph is
directed: an edge points from an earlier method or candidate parent to the
newer branch. If a poor result should be retested from an older measured
version, set parent_variant_id to that existing outcome node ID from
graph_context, where traversal is marked backtrack_ancestor; otherwise leave
it null. This is a deliberate directed backtrack and must not point to a
proposal node or invent an ID. You own the semantic relation field: declare
ablate only when the research question intentionally removes a named
component; declare deepen, combine, or explore when that is the experiment's
meaning. The graph may flag a mismatch between the relation and component map,
but it will not silently relabel your experiment. You may also set
evidence_parent_ids to existing measured outcome IDs whose conclusions
informed this proposal. These are evidence links, not code parents, and must
not be proposal IDs. When the contract uses an independent evaluator, make
required_invariants checkable before or after execution at the correct stage:
solution code must write valid predictions, public validation comparisons can
be printed by that code, and the experiment controller compares the private
primary metric after the evaluator runs. Never require solution code to report
or compare an unavailable private score.""",
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
cost, and risk fit the current question and remaining budget. Prefer an unmeasured family when its value is
comparable to a repeatedly unproductive family. After two non-improving
outcomes in one family, select a feasible orthogonal branch unless there is a
specific unresolved contradiction that another replication would answer.
Changing only validation folds or mixture weights for the same method is not
new information when repeated runs agree and the primary metric does not
improve. Do not select an index outside the supplied list.""",
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
entrypoint and keep the change focused on the research question. Treat every
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
invariant_checks.""",
            contract=contract,
            change=change,
            run=run,
            recovery=recovery,
            implementation_review=implementation_review,
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
question, the declared required invariants, the Coding Agent plan, the actual applied diff, and the current source
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
measured evidence. Set decision="pass" only when passed=true.""",
            contract=contract,
            change=change,
            plan=plan,
            actual_diff=diff,
            repository_context=repo_context,
            known_failures=known_failures or [],
            initial_candidate=initial_candidate,
        )
        return self.llm.complete(SYSTEM, user, ImplementationReviewArtifact)


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
result, and 'stop' only when no useful executable path remains.""",
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
    files, and still emits valid JSON rather than cutting a serialized object
    in the middle.
    """

    packed = {
        key: _compact_payload(key, value)
        for key, value in payload.items()
        if value is not None
    }
    serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > context_limit:
        # Prefer preserving all candidate indices and the latest state while
        # making prose and source excerpts progressively smaller.
        packed = _shrink_strings(packed, limit=1400)
        serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > context_limit:
        for key in ("method_pool", "graph_context", "memory_context", "portfolio_context"):
            if isinstance(packed.get("state"), dict) and isinstance(packed["state"].get(key), list):
                packed["state"][key] = packed["state"][key][-8:]
        serialized = json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > context_limit:
        # This should be rare after the per-field caps.  Keep the instruction,
        # contract, state scalars, and a clearly marked compact remainder.
        compact: dict[str, Any] = {}
        for key in (
            "contract",
            "state",
            "change",
            "hypothesis",
            "plan",
            "actual_diff",
            "repository_context",
            "repo_context",
            "run",
            "recovery",
            "ranked_candidates",
            "candidates",
        ):
            if key in packed:
                value = packed[key]
                if key in {"repository_context", "repo_context"} and isinstance(value, list):
                    value = [
                        {
                            "path": item.get("path", ""),
                            "content": _short(item.get("content", ""), 1800),
                        }
                        for item in value[:8]
                        if isinstance(item, dict)
                    ]
                compact[key] = value
        compact["context_notice"] = (
            "Older graph/source details were compacted after reaching the prompt limit; "
            "use the supplied IDs and current repository files for the next decision."
        )
        serialized = json.dumps(_shrink_strings(compact, 900), ensure_ascii=False, separators=(",", ":"))
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
