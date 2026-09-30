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
not be proposal IDs.""",
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
comparable to a repeatedly unproductive family. Do not select an index outside the supplied list.""",
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
    invariant_checks. Do not
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
propose a new research question. Return passed=false with concrete issues so the technical repair loop can revise the
same candidate. A review failure is technical feedback and does not consume a research round.""",
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


def _context(instruction: str, **payload: Any) -> str:
    packed = {
        key: (
            value.llm_context()
            if isinstance(value, TaskContract)
            else value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
        )
        for key, value in payload.items()
        if value is not None
    }
    return (
        f"{instruction}\n\nContext:\n{json.dumps(packed, ensure_ascii=False, indent=2)}"
    )


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
