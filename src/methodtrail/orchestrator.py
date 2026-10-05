"""The serial research process that coordinates the main agent and four return paths."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agents import (
    AssessmentMemoryAgent,
    CodingAgent,
    EvidenceAgent,
    ExecutionMonitorAgent,
    ImplementationReviewAgent,
    MethodGraphAgent,
    QuestionAgent,
    RepairAgent,
)
from .artifacts import ArtifactStore
from .dashboard import write_dashboard
from .execution import Executor, Verifier
from .llm import (
    LLMDeadlineExceeded,
    LLMOutputValidationError,
    LLMRequestTimeout,
    StructuredLLM,
)
from .memory import BugMemory, BugRecord, ExperimentMemory, MemoryCard
from .path_graph import ExperimentPathGraph, candidate_coverage_gap
from .portfolio import PortfolioManager, VariantProfile
from .project import CandidateRecord, ProjectManager, ProjectRecord, SessionRecord
from .repository import RepoMap
from .schemas import (
    AssessmentArtifact,
    CandidatePath,
    ChangeRequestArtifact,
    HypothesisArtifact,
    ImplementationReviewArtifact,
    MutationClass,
    PathNode,
    RecoveryArtifact,
    ResearchState,
    RunArtifact,
    TaskContract,
    ValueWeights,
)
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)
MIN_COMPARISON_CANDIDATES = 4
MAX_AGENT_CANDIDATES = 18


def _selection_view(
    ranked_candidates: list[dict[str, Any]],
    limit: int = MAX_AGENT_CANDIDATES,
) -> list[dict[str, Any]]:
    """Give the selector a compact, decision-useful view of the frontier.

    The graph keeps every pending proposal, but sending an ever-growing list
    to the LLM makes it harder to reason about the live question.  Preserve a
    high-priority core and a few evidence-based alternatives so the LLM can
    still override the program rank when a lower-ranked branch is promising.
    This is a context bound, not a forced search policy.
    """

    if len(ranked_candidates) <= limit:
        return ranked_candidates
    selected: dict[int, dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        index = row.get("index")
        if isinstance(index, int):
            selected.setdefault(index, row)

    for row in ranked_candidates[: max(8, limit // 2)]:
        add(row)
    perspectives = [
        lambda row: float(row.get("graph_signal", {}).get("semantic_novelty", 0.0)),
        lambda row: float(row.get("candidate", {}).get("expected_gain", 0.0)),
        lambda row: float(row.get("candidate", {}).get("information_gain", 0.0)),
        lambda row: 1.0
        if row.get("candidate", {}).get("parent_variant_id")
        else 0.0,
        lambda row: 1.0
        if row.get("candidate", {}).get("relation") in {"deepen", "recover"}
        else 0.0,
    ]
    for score in perspectives:
        for row in sorted(ranked_candidates, key=score, reverse=True):
            if row.get("graph_signal", {}).get("feasible"):
                add(row)
                break
    for row in ranked_candidates:
        if len(selected) >= limit:
            break
        add(row)
    chosen = set(selected)
    return [row for row in ranked_candidates if row.get("index") in chosen]


@dataclass
class IterationResult:
    iteration: int
    workspace: Path
    change: ChangeRequestArtifact
    run: RunArtifact | None
    assessment: AssessmentArtifact | None
    next_hypothesis: str | None
    artifact_ids: dict[str, str]
    hypothesis: HypothesisArtifact | None = None
    resume_mode: str = "research"
    completed_research: bool = False


class BudgetExhausted(RuntimeError):
    """Raised before editing when no executable candidate fits the tail budget."""


class NoExecutableCandidate(RuntimeError):
    """Raised when discovery only repeats candidates that were already rejected."""


def _metric_deteriorated(
    previous: float | None,
    current: float | None,
    maximize_metric: bool,
) -> bool:
    """Return whether the current result is strictly worse than the prior one."""

    if previous is None or current is None:
        return False
    if math.isclose(previous, current, rel_tol=1e-12, abs_tol=1e-12):
        return False
    # A maximize metric deteriorates when it decreases; a minimize metric
    # deteriorates when it increases. Equality is not deterioration.
    return current < previous if maximize_metric else current > previous


def _failure_signature(run: RunArtifact) -> str:
    """Identify the root error, ignoring changing traceback paths and lines."""

    lines = [line.strip() for line in (run.stderr or run.stdout or "").splitlines()]
    cause = next(
        (
            line
            for line in reversed(lines)
            if re.match(r"^[\w.]+(?:Error|Exception):", line)
            or line.startswith(("edit could not be applied:", "verification failed:"))
        ),
        next((line for line in reversed(lines) if line), "unknown technical failure"),
    )
    return re.sub(r"\s+", " ", cause).lower()[:500]


def _candidate_contract_issues(
    change: ChangeRequestArtifact, *, initial_candidate: bool
) -> list[str]:
    """Catch only obvious contract errors before creating a worktree.

    Semantic questions such as whether a calibration comparison leaks labels
    still belong to the implementation review. This small preflight prevents
    an avoidable code-plan/execution loop for malformed declarations without
    introducing another LLM role.
    """

    issues: list[str] = []
    if not change.research_question.strip():
        issues.append("research_question must state the experiment being tested")
    if not change.rationale.strip():
        issues.append("rationale must explain the intended change")
    if change.relation == "ablate" and not change.method.changed_factors:
        issues.append(
            "relation=ablate requires changed_factors to name the component being removed"
        )
    if (
        not initial_candidate
        and change.method.family == "unspecified"
        and not change.method.components
        and not change.method.changed_factors
    ):
        issues.append(
            "a child candidate must declare a method family, component, or changed factor"
        )
    invariants = [item.strip() for item in change.required_invariants if item.strip()]
    if len({item.casefold() for item in invariants}) != len(invariants):
        issues.append("required_invariants contains a duplicate declaration")
    for invariant in invariants:
        lower = invariant.casefold()
        fold_values = set(re.findall(r"\b(\d+)\s*[- ]?fold\b", lower))
        if len(fold_values) > 1 and not ("inner" in lower and "outer" in lower):
            issues.append(
                "one required_invariant declares multiple fold counts; choose one protocol or label inner/outer folds"
            )
        seed_values = set(re.findall(r"\bseed(?:\s*\+\s*1)?\s*(?:=|:)\s*(\d+)\b", lower))
        if len(seed_values) > 1:
            issues.append(
                "one required_invariant declares multiple seeds; state the exact seed protocol"
            )
    return issues


class MethodTrail:
    """Runs one serial experiment at a time and writes every fact into artifacts."""

    def __init__(
        self,
        project_root: str | Path,
        llm: StructuredLLM,
        session_id: str | None = None,
        project_id: str | None = None,
        prior_project_id: str | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.state_root = self.root / ".methodtrail"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.state_root / "artifacts.sqlite")
        self.graph = ExperimentPathGraph(self.state_root / "experiment_graph.json")
        self.portfolio = PortfolioManager(self.state_root / "portfolio.json")
        self.workspaces = WorkspaceManager(self.state_root / "runs")
        self.projects = ProjectManager(self.state_root)
        self.experiment_memory = ExperimentMemory(self.state_root / "memory")
        self.bug_memory = BugMemory(self.state_root / "memory")
        self._research_budget_seconds: int | None = None
        self.session_id = session_id
        self.project_id = project_id
        self.prior_project_id = prior_project_id
        self.project: ProjectRecord | None = None
        self.session: SessionRecord | None = None
        self.verifier = Verifier()
        self.executor = Executor()
        self.question_agent = QuestionAgent(llm)
        self.method_graph_agent = MethodGraphAgent(llm)
        # Compatibility attributes for callers of the earlier API.
        self.reflection = self.question_agent
        self.choose = self.method_graph_agent
        self.coding = CodingAgent(llm)
        self.implementation_reviewer = ImplementationReviewAgent(llm)
        self.evidence = EvidenceAgent(llm)
        self.execution_monitor = ExecutionMonitorAgent(llm)
        self.recovery = RepairAgent(llm)
        self.assessment_memory = AssessmentMemoryAgent(llm)
        self.assess = self.assessment_memory.assess_agent
        self.memory_agent = self.assessment_memory.memory_agent

    def _prior_evidence_context(self, contract: TaskContract) -> list[dict[str, Any]]:
        """Read a compact, immutable evidence summary from another project.

        Prior projects are deliberately not loaded into the active graph or
        repository. Their measured cards only help a fresh run avoid forgetting
        a previously successful direction; parent and evidence IDs remain
        local to the current project.
        """

        if not self.prior_project_id or self.project is None:
            return []
        if self.prior_project_id == self.project.project_id:
            return []
        try:
            prior = self.projects.load_project(self.prior_project_id)
        except FileNotFoundError:
            return []
        if prior.task_id != contract.task_id:
            return []
        cards_path = self.projects.project_path(prior) / "memory" / "cards.jsonl"
        if not cards_path.exists():
            return []
        cards: list[MemoryCard] = []
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                card = MemoryCard.model_validate_json(line)
            except ValueError:
                continue
            if card.task_id == contract.task_id and card.metric is not None:
                cards.append(card)
        if not cards:
            return []
        ordered = sorted(
            cards,
            key=lambda card: (
                card.metric if card.metric is not None else float("inf"),
                -(card.iteration or 0),
            )
            if not contract.maximize_metric
            else (
                -(card.metric if card.metric is not None else float("-inf")),
                -(card.iteration or 0),
            ),
        )
        chosen: list[MemoryCard] = []
        seen_families: set[str] = set()
        for card in ordered:
            family = card.method_family or "unspecified"
            if family not in seen_families or len(chosen) < 2:
                chosen.append(card)
                seen_families.add(family)
            if len(chosen) >= 6:
                break
        return [
            {
                **card.model_dump(mode="json"),
                "retrieval_source": "prior_project",
                "source_project_id": prior.project_id,
                "read_only": True,
            }
            for card in chosen
        ]

    def _ensure_project_session(self, contract: TaskContract) -> None:
        if self.project is None:
            self.project = self.projects.ensure_project(contract, self.project_id)
            if self.project.incumbent_metric is None and self.project.incumbent_variant_id:
                incumbent = self.projects.get_candidate(
                    self.project, self.project.incumbent_variant_id
                )
                if incumbent is not None:
                    self.project.incumbent_metric = incumbent.metric
            project_path = self.projects.project_path(self.project)
            self.graph = ExperimentPathGraph(project_path / "memory" / "experiment_graph.json")
            self.portfolio = PortfolioManager(project_path / "memory" / "portfolio.json")
            self.experiment_memory = ExperimentMemory(project_path / "memory")
            self.bug_memory = BugMemory(project_path / "memory")
        if self.session is None:
            self.session = self.projects.start_session(self.project, self.session_id)
            self.session_id = self.session.session_id
            # Publish a usable monitor as soon as the session exists. The
            # first LLM call may take several minutes before an iteration
            # result is available, but the dashboard should be reachable
            # during that interval as well.
            try:
                dashboard_path = (
                    self.projects.project_path(self.project)
                    / "sessions"
                    / self.session.session_id
                    / "dashboard.html"
                )
                self.export_dashboard(
                    dashboard_path,
                    project_id=self.project.project_id,
                    session_id=self.session.session_id,
                )
            except Exception as exc:  # noqa: BLE001 - monitoring must not stop research.
                logger.debug("initial dashboard export failed: %s", exc)

    @staticmethod
    def _budget_reserve(contract: TaskContract, remaining_seconds: int) -> int:
        """Reserve time for final output and one last recovery decision.

        The reserve grows with the run up to five minutes and never drops below
        the contract's explicit tail allowance.  This keeps a late candidate
        from starting when its score could not be recorded in time.
        """

        if remaining_seconds <= 0:
            return 0
        dynamic = max(60, min(300, remaining_seconds // 10))
        return max(contract.finalization_reserve_seconds, dynamic)

    def run_iteration(
        self,
        contract: TaskContract,
        remaining_seconds: int,
        parent_variant_id: str | None = None,
        trigger: str | None = None,
        hypothesis_override: HypothesisArtifact | None = None,
        constraint_feedback: str | None = None,
    ) -> IterationResult:
        iteration_started = time.monotonic()
        self._ensure_project_session(contract)
        assert self.project is not None and self.session is not None
        code_parent_variant_id = parent_variant_id
        if code_parent_variant_id is None:
            code_parent_variant_id = self.project.incumbent_variant_id
        graph_parent_variant_id = (
            code_parent_variant_id
            or self.graph.best_outcome_id(contract.maximize_metric)
        )
        state = self._state(contract, remaining_seconds, graph_parent_variant_id)
        state_id = self.store.put("research_state", state)
        self.projects.append_event(
            self.project,
            self.session.session_id,
            "research_state",
            {"artifact_id": state_id, "iteration": state.iteration},
        )
        # A technical repair resumes the exact same research question.  The
        # question agent is called only for a genuinely new research round or
        # after valid evidence changes the question.
        hypothesis = hypothesis_override or self.reflection.refine(contract, state, trigger)
        hypothesis_id = self.store.put("hypothesis", hypothesis, [state_id])
        proposals = self.choose.propose(
            contract,
            state,
            hypothesis,
            constraint_feedback=constraint_feedback,
        )
        coverage_gap = candidate_coverage_gap(
            proposals.candidates,
            initial=state.iteration <= 1 and state.incumbent_metric is None,
        )
        # Only a genuinely narrow initial batch gets one revision. Once a
        # measured frontier exists, the LLM decides whether to extend,
        # revisit, or replace the current line; automatically generating a
        # second batch is expensive and over-specifies exploration.
        if coverage_gap and state.iteration <= 1:
            # One bounded revision pass prevents a narrow initial proposal set
            # from becoming the entire search space.  It keeps the same
            # hypothesis and does not invent a task-specific predictor.
            proposals = self.choose.propose(
                contract,
                state,
                hypothesis,
                coverage_feedback=coverage_gap,
                constraint_feedback=constraint_feedback,
            )
        attached = self.graph.attach_proposals(
            graph_parent_variant_id, proposals.candidates, state.iteration
        )
        # Existing unexecuted nodes remain in the same comparison set as new
        # proposals.  This is the project method pool: the LLM can revisit an
        # earlier direction when new evidence changes its priority.
        existing_ids = {proposal_id for proposal_id, _ in attached}
        for proposal_id, change in self.graph.frontier_changes(limit=50):
            if proposal_id not in existing_ids:
                attached.append((proposal_id, change))
        if not attached:
            raise NoExecutableCandidate(
                "method discovery produced no new or revisitable executable node"
            )
        proposal_node_ids = [proposal_node_id for proposal_node_id, _ in attached]
        proposals = proposals.model_copy(
            update={"candidates": [change for _, change in attached]}
        )
        proposal_id = self.store.put("candidate_proposals", proposals, [hypothesis_id])
        candidates = [
            CandidatePath(
                variant_id=proposal_node_ids[index],
                title=change.title,
                relation=change.relation,
                expected_gain=change.expected_gain,
                information_gain=change.information_gain,
                estimated_seconds=change.estimated_seconds,
                failure_risk=change.failure_risk,
                evidence=[hypothesis.observation, hypothesis.hypothesis],
                method=change.method,
            )
            for index, change in enumerate(proposals.candidates)
        ]
        decision_remaining = max(
            0,
            int(remaining_seconds - (time.monotonic() - iteration_started)),
        )
        # Candidate count and exploration phase are not fixed task parameters.
        # The method graph supplies the relevant alternatives; this formula
        # provides a transparent baseline for comparing their measured value.
        remaining_fraction = (
            decision_remaining
            / max(self._research_budget_seconds or decision_remaining, 1)
        )
        weights = ValueWeights.for_stage(
            remaining_fraction, has_incumbent=self.project.incumbent_metric is not None
        )
        reserve_seconds = self._budget_reserve(contract, decision_remaining)
        ranked = self.graph.rank(
            candidates,
            decision_remaining,
            weights,
            maximize_metric=contract.maximize_metric,
            reserve_seconds=reserve_seconds,
            plateau_rounds=state.plateau_rounds,
        )
        candidate_indices = {candidate.variant_id: index for index, candidate in enumerate(candidates)}
        for candidate_path, priority in ranked:
            self.graph.set_proposal_priority(candidate_path.variant_id, priority)
        ranked_payload = [
            {
                "index": candidate_indices[candidate.variant_id],
                "priority": priority,
                "graph_signal": self.graph.selection_signal(
                    candidate,
                    decision_remaining,
                    reserve_seconds=reserve_seconds,
                    maximize_metric=contract.maximize_metric,
                    plateau_rounds=state.plateau_rounds,
                ),
                "candidate": candidate.model_dump(mode="json"),
            }
            for candidate, priority in ranked
        ]
        feasible_ranked = [item for item in ranked_payload if item["graph_signal"]["feasible"]]
        if not feasible_ranked:
            raise BudgetExhausted(
                "no candidate fits the remaining time after the finalization reserve"
            )
        selection_view = _selection_view(feasible_ranked)
        priority_id = self.store.put(
            "candidate_priorities",
            {
                "iteration": state.iteration,
                "parent_variant_id": graph_parent_variant_id,
                "comparison_floor": MIN_COMPARISON_CANDIDATES,
                "comparison_count": len(ranked_payload),
                "comparison_floor_met": len(ranked_payload) >= MIN_COMPARISON_CANDIDATES,
                "agent_selection_count": len(selection_view),
                "agent_selection_limit": MAX_AGENT_CANDIDATES,
                "plateau_rounds": state.plateau_rounds,
                "exploration_pressure": min(1.0, state.plateau_rounds / 3.0),
                "weights": weights.model_dump(mode="json"),
                "ranked": ranked_payload,
            },
            [proposal_id],
        )
        selection = self.choose.select(contract, state, hypothesis, selection_view)
        selected_row = next(
            (row for row in feasible_ranked if row["index"] == selection.selected_index),
            None,
        )
        if selected_row is None:
            # The LLM receives original candidate indices, but an invalid or
            # infeasible choice must never reach the editor or executor.
            raise RuntimeError(
                f"Choose selected an unavailable candidate index {selection.selected_index}"
            )
        if selection.selected_index >= len(proposals.candidates):
            raise RuntimeError(
                f"Choose selected invalid candidate index {selection.selected_index}"
            )
        requested_parent = proposals.candidates[selection.selected_index].parent_variant_id
        # A proposal may explicitly walk back along a directed graph path to a
        # measured ancestor. Only outcome nodes are valid code bases; an
        # unknown or proposal ID falls back to the active incumbent anchor.
        selected_parent = graph_parent_variant_id
        if (
            requested_parent
            and requested_parent in self.graph.graph
            and self.graph.graph.nodes[requested_parent].get("node_type") == "outcome"
        ):
            selected_parent = requested_parent
        change = proposals.candidates[selection.selected_index].model_copy(
            update={"parent_variant_id": selected_parent}
        )
        proposal_node_id = proposal_node_ids[selection.selected_index]
        if selected_parent != graph_parent_variant_id:
            # Keep the candidate's incoming edge consistent with the actual
            # code parent chosen after directed backtracking.
            if self.graph.graph.has_edge(graph_parent_variant_id, proposal_node_id):
                self.graph.graph.remove_edge(graph_parent_variant_id, proposal_node_id)
            self.graph.add_candidate_relation(
                selected_parent, proposal_node_id, change.relation
            )
        self.graph.mark_proposal_selected(proposal_node_id)
        selection_id = self.store.put(
            "candidate_selection",
            selection,
            [proposal_id, priority_id],
        )
        change_id = self.store.put("change_request", change, [selection_id])

        preflight_issues = _candidate_contract_issues(
            change,
            initial_candidate=not bool(change.parent_variant_id or code_parent_variant_id),
        )
        if preflight_issues:
            feedback = (
                "The selected candidate failed the deterministic preflight. "
                "Keep the research question and intended comparison unchanged, "
                "then revise only these declarations before editing:\n- "
                + "\n- ".join(preflight_issues)
            )
            self.graph.mark_proposal_replan_rejected(
                proposal_node_id,
                feedback,
                preflight_issues,
            )
            preflight_id = self.store.put(
                "candidate_contract_preflight",
                {"passed": False, "issues": preflight_issues},
                [change_id],
            )
            result = IterationResult(
                iteration=state.iteration,
                workspace=Path(self.project.repository),
                change=change,
                run=None,
                assessment=None,
                next_hypothesis=feedback,
                artifact_ids={
                    "state": state_id,
                    "hypothesis": hypothesis_id,
                    "change": change_id,
                    "preflight": preflight_id,
                },
                hypothesis=hypothesis,
                resume_mode="replan",
                completed_research=False,
            )
            self._write_handoff(result)
            return result

        candidate = self.projects.create_candidate(
            self.project,
            self.session,
            contract,
            change.parent_variant_id or code_parent_variant_id,
        )
        workspace = Path(candidate.workspace)
        repo_context = RepoMap(workspace).retrieve(
            f"{change.research_question} {change.title}",
            full_paths=set(contract.editable_paths or [contract.solution_entrypoint]),
        )
        plan = self.coding.write_plan(
            contract,
            change,
            repo_context,
            entrypoint_exists=(workspace / contract.solution_entrypoint).exists(),
            known_failures=self.bug_memory.guidance(contract.task_id),
        )
        # A configuration experiment may still need to edit a source entrypoint
        # when the task has no separate config file. Preserve the research label,
        # but let the workspace safety checks see the actual implementation scope.
        if (
            change.mutation_class == MutationClass.CONFIGURATION
            and any(edit.path.endswith(".py") for edit in plan.edits)
        ):
            change = change.model_copy(
                update={"mutation_class": MutationClass.IMPLEMENTATION}
            )
        plan_id = self.store.put("code_plan", plan, [change_id, priority_id])
        candidate.source_files = sorted({edit.path for edit in plan.edits})
        self.projects.record_candidate(self.project, candidate, status="prepared")

        run, run_id, repair_ids, last_recovery = self._execute_with_repairs(
            contract,
            state,
            change,
            candidate,
            workspace,
            plan,
            plan_id,
            max(1, int(remaining_seconds - (time.monotonic() - iteration_started))),
        )
        if last_recovery and last_recovery.action == "replan_candidate":
            # Keep the selected proposal auditable, but do not turn an
            # unsatisfiable candidate contract into measured failure evidence.
            self.graph.mark_proposal_replan_rejected(
                proposal_node_id,
                last_recovery.diagnosis,
                last_recovery.repair_directions,
            )
            self.projects.record_candidate(self.project, candidate, status="deferred")
            result = IterationResult(
                iteration=state.iteration,
                workspace=workspace,
                change=change,
                run=run,
                assessment=None,
                next_hypothesis=last_recovery.return_to_reflection_reason,
                artifact_ids={
                    "state": state_id,
                    "hypothesis": hypothesis_id,
                    "change": change_id,
                    "plan": plan_id,
                    "run": run_id,
                    "repair": repair_ids,
                    "variant": candidate.variant_id,
                },
                hypothesis=hypothesis,
                resume_mode="replan",
                completed_research=False,
            )
            self._write_handoff(result)
            return result
        if run.return_code != 0 or run.timed_out or run.metric is None:
            self.projects.record_candidate(
                self.project, candidate, status="failed", metric=run.metric
            )
            recovery = last_recovery or self.recovery.diagnose(contract, state, run)
            recovery_id = self.store.put("recovery", recovery, [run_id])
            self._record_failed_outcome(
                contract, change, candidate, run, proposal_node_id
            )
            result = IterationResult(
                iteration=state.iteration,
                workspace=workspace,
                change=change,
                run=run,
                assessment=None,
                next_hypothesis=(
                    recovery.return_to_reflection_reason if recovery else None
                ),
                artifact_ids={
                    "state": state_id,
                    "hypothesis": hypothesis_id,
                    "change": change_id,
                    "plan": plan_id,
                    "run": run_id,
                    "recovery": recovery_id,
                    "repair": repair_ids,
                    "variant": candidate.variant_id,
                },
                hypothesis=hypothesis,
                resume_mode="technical",
                completed_research=False,
            )
            self._write_handoff(result)
            return result

        try:
            assessment = self.assess.assess(contract, state, change, run)
        except Exception as exc:
            self._preserve_unassessed_run(
                contract,
                state,
                change,
                candidate,
                workspace,
                run,
                run_id,
                proposal_node_id,
                repair_ids,
                {"state": state_id, "hypothesis": hypothesis_id, "change": change_id, "plan": plan_id},
                exc,
            )
            raise
        if (
            assessment.decision == "adopt"
            and run.metric_constraints_passed is False
        ):
            assessment = assessment.model_copy(
                update={
                    "decision": "defer",
                    "reason": (
                        f"The primary metric was measured, but declared metric constraints failed: "
                        f"{contract.metric_constraints}."
                    ),
                }
            )
        improves_incumbent = (
            run.metric_constraints_passed is not False
            and self._metric_improves_incumbent(contract, run.metric)
        )
        if improves_incumbent and assessment.decision != "adopt":
            # A valid primary-metric improvement becomes the working incumbent
            # immediately. The next round can still investigate its uncertainty;
            # evidence collection must not leave the code pointer on an older
            # candidate.
            assessment = assessment.model_copy(
                update={
                    "decision": "adopt",
                    "reason": (
                        f"Primary metric improved the incumbent to {run.metric}; "
                        "the version is promoted while any declared uncertainty remains available for follow-up."
                    ),
                }
            )
        if assessment.decision == "adopt" and not improves_incumbent:
            assessment = assessment.model_copy(
                update={
                    "decision": "defer",
                    "reason": (
                        "The measured candidate did not improve the project-wide "
                        f"incumbent ({self.project.incumbent_metric}); it remains "
                        "available in the method graph for further evidence."
                    ),
                }
            )
        assessment_id = self.store.put("assessment", assessment, [run_id])
        try:
            result = self._complete_assessment(
                contract,
                state,
                change,
                run,
                assessment,
                workspace,
                candidate,
                plan,
                proposal_node_id,
                {
                    "state": state_id,
                    "hypothesis": hypothesis_id,
                    "change": change_id,
                    "plan": plan_id,
                    "run": run_id,
                    "assessment": assessment_id,
                    "repair": repair_ids,
                },
            )
        except LLMDeadlineExceeded as exc:
            self._preserve_unassessed_run(
                contract,
                state,
                change,
                candidate,
                workspace,
                run,
                run_id,
                proposal_node_id,
                repair_ids,
                {"state": state_id, "hypothesis": hypothesis_id, "change": change_id, "plan": plan_id},
                exc,
            )
            self.projects.append_event(
                self.project,
                self.session.session_id,
                "budget_interruption",
                {
                    "iteration": state.iteration,
                    "variant_id": candidate.variant_id,
                    "metric": run.metric,
                    "reason": str(exc),
                },
            )
            raise
        except Exception as exc:
            self._preserve_unassessed_run(
                contract,
                state,
                change,
                candidate,
                workspace,
                run,
                run_id,
                proposal_node_id,
                repair_ids,
                {"state": state_id, "hypothesis": hypothesis_id, "change": change_id, "plan": plan_id},
                exc,
            )
            raise
        self._write_handoff(result)
        return result

    def _preserve_unassessed_run(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        candidate: CandidateRecord,
        workspace: Path,
        run: RunArtifact,
        run_id: str,
        proposal_node_id: str,
        repair_ids: list[str],
        artifact_ids: dict[str, str],
        error: Exception,
    ) -> None:
        assert self.project is not None and self.session is not None
        if run.return_code != 0 or run.timed_out or run.metric is None:
            return
        already_adopted = self.project.incumbent_variant_id == candidate.variant_id
        improves = (
            run.metric_constraints_passed is not False
            and self._metric_improves_incumbent(contract, run.metric)
        )
        adopted = already_adopted or improves
        summary = (
            f"Independent evaluator measured {contract.metric_name}={run.metric}. "
            "The assessment agent did not finish; no causal conclusion was inferred."
        )
        assessment = AssessmentArtifact(
            decision="adopt" if adopted else "defer",
            reason=summary,
            reusable_conclusion=summary,
            applicable_conditions=[
                "The independent evaluator completed successfully.",
                "Agent assessment was unavailable after measurement.",
            ],
            next_question=change.research_question,
        )
        fallback_id = self.store.put("assessment_fallback", assessment, [run_id])
        if candidate.variant_id not in self.graph.graph:
            self.graph.add_node(
                PathNode(
                    variant_id=candidate.variant_id,
                    iteration=state.iteration,
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    relation=change.relation,
                    relation_warning=self.graph.relation_warning_for(change.parent_variant_id, change),
                    mutation_class=change.mutation_class,
                    question=change.research_question,
                    change_logic=change.rationale,
                    evidence_summary=summary,
                    applicable_conditions=assessment.applicable_conditions,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    failure_risk=change.failure_risk,
                    status="adopted" if adopted else "deferred",
                    method=change.method,
                )
            )
        # A later assessment or memory call may fail after the outcome node
        # was already written. Its status must still match the measured
        # incumbent decision recovered here.
        self.graph.graph.nodes[candidate.variant_id]["status"] = (
            "adopted" if adopted else "deferred"
        )
        self.graph.record_outcome(
            proposal_node_id,
            candidate.variant_id,
            "adopted" if adopted else "deferred",
        )
        if not any(
            node.get("variant_id") == candidate.variant_id
            for node in self.experiment_memory.graph.nodes.values()
        ):
            card = self.experiment_memory.add(
                MemoryCard(
                    task_id=contract.task_id,
                    session_id=self.session.session_id,
                    variant_id=candidate.variant_id,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    question=change.research_question,
                    conclusion=summary,
                    measured_facts=[f"{contract.metric_name}={run.metric}"],
                    evidence=["independent evaluator output"],
                    applicable_conditions=assessment.applicable_conditions,
                    relation=change.relation,
                    decision=assessment.decision,
                    metric=run.metric,
                    method_family=change.method.family,
                    changed_factors=change.method.changed_factors,
                    iteration=state.iteration,
                    title=change.title,
                    mutation_class=change.mutation_class.value,
                    change_logic=change.rationale,
                    method_components=change.method.components,
                )
            )
            artifact_ids["memory"] = self.store.put("memory_fallback", card, [fallback_id])
        if candidate.variant_id not in self.portfolio.profiles:
            self.portfolio.register(
                VariantProfile(
                    variant_id=candidate.variant_id,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    reliability=1.0,
                    information_gain=change.information_gain,
                    failure_risk=change.failure_risk,
                    signature=f"{change.mutation_class}:{change.title}",
                )
            )
        if improves:
            self.projects.adopt_candidate(
                self.project,
                candidate,
                candidate.source_files,
                f"methodtrail: independently measured {change.title}",
                run.metric,
            )
        elif not already_adopted:
            self.projects.record_candidate(
                self.project,
                candidate,
                status="constraint_failed" if run.metric_constraints_passed is False else "deferred",
                metric=run.metric,
            )
        self._write_handoff(
            IterationResult(
                iteration=state.iteration,
                workspace=workspace,
                change=change,
                run=run,
                assessment=assessment,
                next_hypothesis=assessment.next_question,
                artifact_ids={**artifact_ids, "run": run_id, "assessment": fallback_id, "repair": repair_ids, "variant": candidate.variant_id},
                completed_research=True,
            )
        )
        self.projects.update_session(
            self.project,
            self.session,
            status="interrupted",
            current_variant_id=self.project.incumbent_variant_id,
            consecutive_deteriorating=0,
            last_metric=run.metric,
        )
        self.projects.append_event(
            self.project,
            self.session.session_id,
            "assessment_interrupted",
            {
                "iteration": state.iteration,
                "variant_id": candidate.variant_id,
                "metric": run.metric,
                "reason": f"{type(error).__name__}: {str(error)[:240]}",
            },
        )

    def _execute_with_repairs(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        candidate: CandidateRecord,
        workspace: Path,
        plan: Any,
        plan_id: str,
        remaining_seconds: int,
    ) -> tuple[RunArtifact, str, list[str], Any | None]:
        """Apply, verify, execute and repair one candidate without research-count drift."""

        repair_ids: list[str] = []
        repair_step = 0
        last_recovery = None
        current_plan = plan
        current_plan_id = plan_id
        current_change = change
        started = time.monotonic()
        last_review: ImplementationReviewArtifact | None = None
        seen_patch_signatures: set[str] = set()
        parent_sources = self.workspaces.source_snapshot(workspace, contract)
        parent_context = [
            {
                "path": path,
                "content": content,
                "language": Path(path).suffix.lstrip(".") or "text",
                "complete": "true",
                "retrieval_reason": "immutable source snapshot captured before the first edit",
            }
            for path, content in sorted(parent_sources.items())
        ]
        changed_paths: set[str] = set()

        def monitor(checkpoint: dict[str, Any]) -> Any:
            """Ask the monitor role at checkpoints; a failed monitor is non-fatal."""

            try:
                return self.execution_monitor.decide(
                    contract, state, current_change, checkpoint
                )
            except Exception as exc:  # noqa: BLE001 - never kill healthy work
                checkpoint["monitor_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                return {"action": "continue", "reason": "monitor unavailable; continue"}

        while True:
            if time.monotonic() - started >= max(1, remaining_seconds):
                run = RunArtifact(
                    command=contract.run_command,
                    return_code=-3,
                    timed_out=True,
                    wall_seconds=time.monotonic() - started,
                    stdout="",
                    stderr="technical repair budget exhausted before a model run",
                )
                run_id = self.store.put("run", run, [current_plan_id])
                self._record_bug(
                    contract, current_change, candidate, run, "budget", repair_step
                )
                return run, run_id, repair_ids, last_recovery

            try:
                diff = self.workspaces.apply_plan(
                    workspace, current_plan, contract, current_change
                )
            except (PermissionError, ValueError) as exc:
                last_review = None
                run = RunArtifact(
                    command=contract.run_command,
                    return_code=-2,
                    timed_out=False,
                    wall_seconds=0.0,
                    stdout="",
                    stderr=f"edit could not be applied: {exc}",
                )
                run_id = self.store.put("run", run, [current_plan_id])
                self._record_bug(
                    contract, current_change, candidate, run, "edit", repair_step
                )
            else:
                changed_paths.update(edit.path for edit in current_plan.edits)
                cumulative_diff = self.workspaces.cumulative_diff(
                    workspace, parent_sources, contract, changed_paths
                )
                implementation_id = self.store.put(
                    "implementation",
                    {
                        "workspace": str(workspace),
                        "variant_id": candidate.variant_id,
                        "diff": diff,
                        "cumulative_diff": cumulative_diff,
                        "entrypoint": contract.solution_entrypoint,
                        "repair_step": repair_step,
                    },
                    [current_plan_id],
                )
                patch_signature = hashlib.sha256(cumulative_diff.encode("utf-8")).hexdigest()
                if patch_signature in seen_patch_signatures:
                    # A byte-identical repair cannot provide new semantic
                    # evidence. Keep a structured failure for Recovery/Coding
                    # and avoid paying for a duplicate LLM review call.
                    review = ImplementationReviewArtifact(
                        passed=False,
                        scope_ok=False,
                        invariants_ok=False,
                        summary="repair made no progress",
                        checks=["the applied diff repeats an earlier patch"],
                        issues=[
                            (
                                "Produce a different minimal edit after rereading the latest workspace; "
                                f"repeated diff signature is {patch_signature[:12]}."
                            )
                        ],
                    )
                else:
                    seen_patch_signatures.add(patch_signature)
                    review = self.implementation_reviewer.review(
                        contract,
                        change,
                        current_plan,
                        cumulative_diff,
                        RepoMap(workspace).retrieve(
                            f"{current_change.research_question} {current_change.title}",
                            full_paths=set(contract.editable_paths or [contract.solution_entrypoint]),
                        ),
                        known_failures=self.bug_memory.guidance(contract.task_id, limit=12),
                        initial_candidate=candidate.parent_variant_id is None,
                        parent_context=parent_context,
                    )
                review_id = self.store.put(
                    "implementation_review" if repair_step == 0 else "repair_implementation_review",
                    review,
                    [implementation_id],
                )
                if review.replan_required or review.decision == "replan":
                    adjustments = review.candidate_adjustments or review.issues
                    recovery = RecoveryArtifact(
                        failure_class="candidate_contract",
                        diagnosis=review.summary,
                        repair_directions=list(adjustments),
                        preserve_question=True,
                        return_to_reflection_reason=(
                            "Revise the selected candidate contract before editing: "
                            + "; ".join(adjustments[:4])
                        ),
                        action="replan_candidate",
                    )
                    recovery_id = self.store.put("recovery", recovery, [review_id])
                    repair_ids.append(recovery_id)
                    run = RunArtifact(
                        command=contract.run_command,
                        return_code=-8,
                        timed_out=False,
                        wall_seconds=0.0,
                        stdout="",
                        stderr=(
                            "candidate contract requires re-planning: "
                            + ("; ".join(adjustments) or review.summary)
                        ),
                    )
                    run_id = self.store.put("run", run, [recovery_id])
                    self._record_bug(
                        contract,
                        current_change,
                        candidate,
                        run,
                        "candidate_contract",
                        repair_step,
                        recovery=recovery,
                    )
                    return run, run_id, repair_ids, recovery
                if not review.passed:
                    last_review = review
                    detail = "\n".join(review.issues) or review.summary
                    run = RunArtifact(
                        command=contract.run_command,
                        return_code=-7,
                        timed_out=False,
                        wall_seconds=0.0,
                        stdout="",
                        stderr=f"implementation review failed: {detail}",
                    )
                    run_id = self.store.put("run", run, [review_id])
                    self._record_bug(
                        contract, current_change, candidate, run, "implementation_review", repair_step
                    )
                else:
                    last_review = None
                    verification = self.verifier.verify(workspace, contract)
                    verify_id = self.store.put(
                        "verification" if repair_step == 0 else "repair_verification",
                        verification,
                        [review_id],
                    )
                    if verification.passed:
                        run = self.executor.run(
                            workspace,
                            contract,
                            deadline_monotonic=time.monotonic() + max(1, remaining_seconds),
                            progress_callback=monitor,
                        )
                        run_id = self.store.put(
                            "run" if repair_step == 0 else "repair_run",
                            run,
                            [verify_id],
                        )
                        if run.return_code == 0 and not run.timed_out and run.metric is not None:
                            if repair_step:
                                self._record_bug(
                                    contract,
                                    current_change,
                                    candidate,
                                    run,
                                    "repair_resolved",
                                    repair_step,
                                    resolved=True,
                                )
                            return run, run_id, repair_ids, last_recovery
                        self._record_bug(
                            contract, current_change, candidate, run, "execution", repair_step
                        )
                    else:
                        detail = "\n".join(
                            check["detail"]
                            for check in verification.checks
                            if not check["passed"]
                        )
                        run = RunArtifact(
                            command=contract.run_command,
                            return_code=-2,
                            timed_out=False,
                            wall_seconds=0.0,
                            stdout="",
                            stderr=f"verification failed: {detail}",
                        )
                        run_id = self.store.put("run", run, [verify_id])
                        self._record_bug(
                            contract, current_change, candidate, run, "verification", repair_step
                        )

            # Keep repairing the same selected candidate until the Recovery
            # Agent explicitly declares it impossible, the technical repair
            # safety cap is reached, or the shared time budget expires. A
            # repeated review message is feedback for the next repair, not an
            # automatic reason to switch research methods.
            if repair_step >= contract.max_repair_steps:
                return run, run_id, repair_ids, last_recovery
            if time.monotonic() - started >= max(1, remaining_seconds):
                return run, run_id, repair_ids, last_recovery
            repair_state = state.model_copy(update={"repair_step": repair_step + 1})
            known_failures = self.bug_memory.guidance(contract.task_id, limit=12)
            recovery = self.recovery.diagnose(
                contract,
                repair_state,
                run,
                known_failures=known_failures,
                implementation_review=last_review,
            )
            last_recovery = recovery
            recovery_id = self.store.put("recovery", recovery, [run_id])
            repair_ids.append(recovery_id)
            if recovery.action == "abandon_candidate":
                return run, run_id, repair_ids, last_recovery
            repair_step += 1
            repair_change = change.model_copy(
                update={
                    "title": f"repair {repair_step}: {change.title}",
                    "mutation_class": MutationClass.RECOVERY,
                    "relation": "recover",
                    "rationale": recovery.diagnosis,
                }
            )
            repair_input = run
            while True:
                try:
                    repair_plan = self.coding.repair(
                        contract,
                        repair_change,
                        repair_input,
                        recovery,
                        RepoMap(workspace).retrieve(
                            f"{change.research_question} {recovery.diagnosis}",
                            full_paths=set(contract.editable_paths or [contract.solution_entrypoint]),
                        ),
                        known_failures=known_failures,
                        implementation_review=last_review,
                        parent_context=parent_context,
                    )
                    break
                except LLMOutputValidationError as exc:
                    # No patch was applied. Correct the response using fresh
                    # source and the original outstanding review/run issues;
                    # never reapply the last plan or consume a research round.
                    invalid_run = RunArtifact(
                        command=contract.run_command,
                        return_code=-9,
                        timed_out=False,
                        wall_seconds=0.0,
                        stdout="",
                        stderr=f"repair plan validation failed: {exc}",
                    )
                    invalid_id = self.store.put("repair_plan_error", invalid_run, [recovery_id])
                    repair_ids.append(invalid_id)
                    self._record_bug(
                        contract, change, candidate, invalid_run, "planning", repair_step
                    )
                    if (
                        repair_step >= contract.max_repair_steps
                        or time.monotonic() - started >= max(1, remaining_seconds)
                    ):
                        return invalid_run, invalid_id, repair_ids, last_recovery
                    repair_step += 1
                    repair_input = run.model_copy(update={
                        "stderr": invalid_run.stderr + "\nOutstanding original failure:\n" + run.stderr,
                    })
                    known_failures = self.bug_memory.guidance(contract.task_id, limit=12)
            current_plan = repair_plan
            current_change = repair_change
            current_plan_id = self.store.put("repair_plan", repair_plan, [recovery_id])
            repair_ids.append(current_plan_id)
            candidate.source_files = sorted(
                set(candidate.source_files).union(edit.path for edit in repair_plan.edits)
            )
            self.projects.record_candidate(self.project, candidate, status="prepared")

    def run_research(
        self,
        contract: TaskContract,
        total_seconds: int,
        max_iterations: int = 20,
        parent_variant_id: str | None = None,
    ) -> list[IterationResult]:
        """Run serial iterations until a stopping decision, budget exhaustion, or iteration limit."""

        self._research_budget_seconds = total_seconds
        set_deadline = getattr(self.question_agent.llm, "set_deadline", None)
        if callable(set_deadline):
            set_deadline(time.monotonic() + total_seconds)
        remaining_seconds = total_seconds
        trigger: str | None = None
        constraint_feedback: str | None = None
        pending_hypothesis: HypothesisArtifact | None = None
        current_parent = parent_variant_id
        results: list[IterationResult] = []
        completed_rounds = 0
        consecutive_replans = 0
        consecutive_llm_timeouts = 0
        plateau_rounds = (
            self.session.consecutive_non_improving if self.session is not None else 0
        )
        consecutive_deteriorating = (
            self.session.consecutive_deteriorating if self.session is not None else 0
        )
        previous_metric = self.session.last_metric if self.session is not None else None
        while completed_rounds < max_iterations:
            if remaining_seconds <= self._budget_reserve(contract, remaining_seconds):
                if self.project is not None and self.session is not None:
                    self.projects.append_event(
                        self.project,
                        self.session.session_id,
                        "budget_stop",
                        {
                            "remaining_seconds": remaining_seconds,
                            "reserve_seconds": self._budget_reserve(
                                contract, remaining_seconds
                            ),
                            "reason": "remaining time reserved for final output and recovery",
                        },
                    )
                break
            iteration_started = time.monotonic()
            best_before = self.project.incumbent_metric if self.project is not None else None
            try:
                result = self.run_iteration(
                    contract,
                    remaining_seconds=remaining_seconds,
                    parent_variant_id=current_parent,
                    trigger=trigger,
                    hypothesis_override=pending_hypothesis,
                    constraint_feedback=constraint_feedback,
                )
            except (BudgetExhausted, LLMDeadlineExceeded) as exc:
                elapsed = max(1, int(time.monotonic() - iteration_started))
                remaining_seconds = max(0, remaining_seconds - elapsed)
                if self.project is not None and self.session is not None:
                    self.projects.append_event(
                        self.project,
                        self.session.session_id,
                        "budget_stop",
                        {
                            "remaining_seconds": remaining_seconds,
                            "reserve_seconds": self._budget_reserve(
                                contract, remaining_seconds
                            ),
                            "reason": str(exc),
                        },
                    )
                break
            except LLMRequestTimeout as exc:
                elapsed = max(1, int(time.monotonic() - iteration_started))
                remaining_seconds = max(0, remaining_seconds - elapsed)
                consecutive_llm_timeouts += 1
                if self.project is not None and self.session is not None:
                    self.projects.append_event(
                        self.project,
                        self.session.session_id,
                        "llm_request_timeout",
                        {
                            "iteration": completed_rounds + 1,
                            "elapsed_seconds": elapsed,
                            "remaining_seconds": remaining_seconds,
                            "consecutive_timeouts": consecutive_llm_timeouts,
                            "reason": str(exc),
                        },
                    )
                # A provider stall must not consume a research round or trap
                # the controller in the same request forever. Give one fresh
                # attempt a chance to continue with the same evidence, then
                # stop cleanly if the provider remains unavailable.
                if (
                    consecutive_llm_timeouts >= 2
                    or remaining_seconds <= self._budget_reserve(contract, remaining_seconds)
                ):
                    if self.project is not None and self.session is not None:
                        self.projects.append_event(
                            self.project,
                            self.session.session_id,
                            "research_stop",
                            {
                                "reason": "repeated LLM request timeouts",
                                "completed_rounds": completed_rounds,
                            },
                        )
                    break
                trigger = (
                    "The previous provider request timed out before returning a typed artifact. "
                    "Keep the research question and use the existing graph evidence; retry with a "
                    "compact, executable decision."
                )
                constraint_feedback = None
                pending_hypothesis = None
                continue
            except NoExecutableCandidate as exc:
                if self.project is not None and self.session is not None:
                    self.projects.append_event(
                        self.project,
                        self.session.session_id,
                        "research_stop",
                        {
                            "reason": str(exc),
                            "completed_rounds": completed_rounds,
                            "consecutive_deteriorating": consecutive_deteriorating,
                        },
                    )
                break
            except Exception as exc:
                # Preserve a truthful resumable state on unexpected planning,
                # schema, API or controller failures. Never leave a dead
                # process labelled active in the monitor.
                if callable(set_deadline):
                    set_deadline(None)
                if self.project is not None and self.session is not None:
                    self.projects.append_event(
                        self.project, self.session.session_id, "research_interrupted",
                        {"reason": f"{type(exc).__name__}: {str(exc)[:1800]}"},
                    )
                    self.projects.update_session(self.project, self.session, status="interrupted")
                    try:
                        self.export_dashboard(
                            self.projects.project_path(self.project) / "sessions"
                            / self.session.session_id / "dashboard.html",
                            project_id=self.project.project_id, session_id=self.session.session_id,
                        )
                    except Exception as dashboard_error:  # noqa: BLE001 - preserve the original failure.
                        logger.debug("interrupted dashboard export failed: %s", dashboard_error)
                raise
            results.append(result)
            consecutive_llm_timeouts = 0
            # The budget covers the complete research turn, including LLM
            # planning, code editing and execution—not only the Python process.
            elapsed = max(1, int(time.monotonic() - iteration_started))
            remaining_seconds -= elapsed
            if result.resume_mode == "replan":
                consecutive_replans += 1
                if consecutive_replans >= contract.max_replan_steps:
                    if self.project is not None and self.session is not None:
                        self.projects.append_event(
                            self.project,
                            self.session.session_id,
                            "research_stop",
                            {
                                "reason": (
                                    "candidate contracts remained infeasible after "
                                    f"{consecutive_replans} consecutive replans"
                                ),
                                "completed_rounds": completed_rounds,
                                "consecutive_replans": consecutive_replans,
                            },
                        )
                    break
            if result.completed_research:
                consecutive_replans = 0
                completed_rounds += 1
                measured = result.run.metric if result.run else None
                if measured is not None:
                    if best_before is None or (
                        measured > best_before
                        if contract.maximize_metric
                        else measured < best_before
                    ):
                        plateau_rounds = 0
                    else:
                        plateau_rounds += 1
                    deteriorated = _metric_deteriorated(
                        previous_metric, measured, contract.maximize_metric
                    )
                    consecutive_deteriorating = (
                        consecutive_deteriorating + 1 if deteriorated else 0
                    )
                    previous_metric = measured
                if self.project is not None and self.session is not None:
                    self.projects.update_session(
                        self.project,
                        self.session,
                        consecutive_non_improving=plateau_rounds,
                        consecutive_deteriorating=consecutive_deteriorating,
                        last_metric=measured,
                    )
            if (
                result.assessment
                and result.assessment.decision == "stop"
                and completed_rounds >= contract.minimum_iterations
            ):
                break
            trigger = result.next_hypothesis
            constraint_feedback = (
                result.next_hypothesis if result.resume_mode == "replan" else None
            )
            pending_hypothesis = (
                result.hypothesis
                if result.resume_mode in {"technical", "replan"}
                else None
            )
            if result.assessment and result.assessment.decision == "adopt":
                current_parent = result.artifact_ids.get("variant", current_parent)
        if self.project is not None and self.session is not None:
            self._finalize_best_valid_candidate(contract)
            self.projects.update_session(
                self.project,
                self.session,
                status="completed",
                handoff=(
                    "# Research session complete\n\n"
                    f"Current accepted variant: {self.project.incumbent_variant_id or 'baseline'}\n"
                    f"Next question: {self.session.next_question or 'none'}\n"
                ),
            )
        if callable(set_deadline):
            set_deadline(None)
        # A deadline can interrupt a turn before _write_handoff receives a
        # result. Keep the dashboard consistent with the persisted session.
        try:
            assert self.project is not None and self.session is not None
            dashboard_path = (
                self.projects.project_path(self.project)
                / "sessions"
                / self.session.session_id
                / "dashboard.html"
            )
            self.export_dashboard(
                dashboard_path,
                project_id=self.project.project_id,
                session_id=self.session.session_id,
            )
        except Exception as exc:  # noqa: BLE001 - dashboard export must not stop research.
            logger.debug("final dashboard export failed: %s", exc)
        return results

    def _metric_improves_incumbent(
        self, contract: TaskContract, metric: float | None
    ) -> bool:
        """Keep adoption monotonic without blocking exploration of weaker nodes."""

        if metric is None or self.project is None or self.project.incumbent_metric is None:
            return metric is not None
        if math.isclose(metric, self.project.incumbent_metric, rel_tol=1e-12, abs_tol=1e-12):
            return False
        if contract.maximize_metric:
            return metric > self.project.incumbent_metric
        return metric < self.project.incumbent_metric

    def _finalize_best_valid_candidate(self, contract: TaskContract) -> None:
        """Persist the strongest valid candidate if exploration produced no adoption.

        An Evidence Agent may deliberately keep asking for more confirmation until
        the iteration budget ends.  The run still needs a reproducible output,
        so this deterministic fallback chooses the best independently measured
        candidate from the completed session.  It does not reinterpret failed
        executions as evidence or overwrite an already adopted incumbent.
        """

        assert self.project is not None and self.session is not None
        if self.project.incumbent_variant_id is not None:
            return
        viable = [
            candidate
            for candidate in self.projects.session_candidates(
                self.project, self.session.session_id
            )
            if candidate.metric is not None
            and candidate.status in {"needs_evidence", "deferred", "promising"}
            and (Path(candidate.workspace) / contract.solution_entrypoint).exists()
        ]
        if not viable:
            return
        key = lambda candidate: candidate.metric if candidate.metric is not None else float("-inf")
        best = (max if contract.maximize_metric else min)(viable, key=key)
        files = best.source_files or [contract.solution_entrypoint]
        self.projects.adopt_candidate(
            self.project,
            best,
            files,
            "methodtrail: best valid candidate at research budget end",
            best.metric,
        )
        self.projects.append_event(
            self.project,
            self.session.session_id,
            "best_valid_candidate_selected",
            {"variant_id": best.variant_id, "metric": best.metric, "reason": "iteration budget ended without an adopted variant"},
        )

    def _complete_assessment(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        run: RunArtifact,
        assessment: AssessmentArtifact,
        workspace: Path,
        candidate: CandidateRecord,
        plan: Any,
        proposal_node_id: str,
        artifact_ids: dict[str, str],
    ) -> IterationResult:
        if assessment.decision == "evidence":
            memory = self.memory_agent.summarize(contract, change, run, assessment)
            memory.decision = assessment.decision
            memory_id = self.store.put("memory", memory, [artifact_ids["assessment"]])
            artifact_ids["memory"] = memory_id
            variant_id = candidate.variant_id
            artifact_ids["variant"] = variant_id
            self.graph.add_node(
                PathNode(
                    variant_id=variant_id,
                    iteration=state.iteration,
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    relation=change.relation,
                    relation_warning=self.graph.relation_warning_for(change.parent_variant_id, change),
                    mutation_class=change.mutation_class,
                    question=change.research_question,
                    evidence_summary=memory.reusable_conclusion or assessment.reason,
                    applicable_conditions=memory.applicable_conditions,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    failure_risk=change.failure_risk,
                    status="needs_evidence",
                    method=change.method,
                )
            )
            self.graph.record_outcome(proposal_node_id, variant_id, "needs_evidence")
            assert self.project is not None and self.session is not None
            self.experiment_memory.add(
                MemoryCard(
                    task_id=contract.task_id,
                    session_id=self.session.session_id,
                    variant_id=variant_id,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    question=change.research_question,
                    conclusion=memory.reusable_conclusion or assessment.reason,
                    measured_facts=[
                        f"metric={run.metric}",
                        f"wall_seconds={run.wall_seconds:.3f}",
                        f"return_code={run.return_code}",
                    ],
                    evidence=[assessment.reason],
                    applicable_conditions=memory.applicable_conditions,
                    relation=change.relation,
                    decision=assessment.decision,
                    metric=run.metric,
                    method_family=change.method.family,
                    changed_factors=change.method.changed_factors,
                    iteration=state.iteration,
                    title=change.title,
                    mutation_class=change.mutation_class.value,
                    change_logic=change.rationale,
                    method_components=change.method.components,
                    tags=[change.title, change.mutation_class.value, change.relation, *memory.applicable_conditions],
                )
            )
            if assessment.question_status in {"resolved", "inconclusive"}:
                # The current comparison is closed.  Do not automatically ask
                # Evidence Agent to restate the same unresolved question; the
                # next research round starts from the graph frontier.
                next_hypothesis = None
                artifact_ids["evidence_closure"] = self.store.put(
                    "evidence_closure",
                    {
                        "status": assessment.question_status,
                        "reason": assessment.reason,
                    },
                    [artifact_ids["assessment"]],
                )
            else:
                next_hypothesis = self.evidence.investigate(
                    contract, state, run, assessment
                )
                evidence_id = self.store.put(
                    "evidence_question", next_hypothesis, [artifact_ids["assessment"]]
                )
                artifact_ids["evidence"] = evidence_id
            self.projects.record_candidate(
                self.project, candidate, status="needs_evidence", metric=run.metric
            )
            return IterationResult(
                state.iteration,
                workspace,
                change,
                run,
                assessment,
                next_hypothesis.hypothesis if next_hypothesis else None,
                artifact_ids,
                completed_research=True,
            )

        if assessment.decision in {"adopt", "defer"}:
            memory = self.memory_agent.summarize(contract, change, run, assessment)
            memory.decision = assessment.decision
            memory_id = self.store.put("memory", memory, [artifact_ids["assessment"]])
            artifact_ids["memory"] = memory_id
            variant_id = candidate.variant_id
            artifact_ids["variant"] = variant_id
            node_status = (
                "promising"
                if assessment.decision == "defer"
                and (change.information_gain > 0 or bool(assessment.next_question))
                else "deferred"
            )
            self.graph.add_node(
                PathNode(
                    variant_id=variant_id,
                    iteration=state.iteration,
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    relation=change.relation,
                    relation_warning=self.graph.relation_warning_for(change.parent_variant_id, change),
                    mutation_class=change.mutation_class,
                    question=change.research_question,
                    evidence_summary=memory.reusable_conclusion or assessment.reason,
                    applicable_conditions=memory.applicable_conditions,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    failure_risk=change.failure_risk,
                    status="adopted" if assessment.decision == "adopt" else node_status,
                    method=change.method,
                )
            )
            self.graph.record_outcome(
                proposal_node_id,
                variant_id,
                "adopted" if assessment.decision == "adopt" else node_status,
            )
            self.portfolio.register(
                VariantProfile(
                    variant_id=variant_id,
                    metric=run.metric or 0.0,
                    wall_seconds=run.wall_seconds,
                    reliability=1.0,
                    information_gain=change.information_gain,
                    failure_risk=change.failure_risk,
                    signature=f"{change.mutation_class}:{change.title}",
                )
            )
            assert self.project is not None and self.session is not None
            self.experiment_memory.add(
                MemoryCard(
                    task_id=contract.task_id,
                    session_id=self.session.session_id,
                    variant_id=variant_id,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    question=change.research_question,
                    conclusion=memory.reusable_conclusion or assessment.reason,
                    measured_facts=[
                        f"metric={run.metric}",
                        f"wall_seconds={run.wall_seconds:.3f}",
                        f"return_code={run.return_code}",
                    ],
                    evidence=[assessment.reason, f"metric={run.metric}"],
                    applicable_conditions=memory.applicable_conditions,
                    relation=change.relation,
                    decision=assessment.decision,
                    metric=run.metric,
                    method_family=change.method.family,
                    changed_factors=change.method.changed_factors,
                    iteration=state.iteration,
                    title=change.title,
                    mutation_class=change.mutation_class.value,
                    change_logic=change.rationale,
                    method_components=change.method.components,
                    tags=[
                        change.title,
                        change.mutation_class.value,
                        change.relation,
                        *memory.applicable_conditions,
                    ],
                )
            )
            if assessment.decision == "adopt":
                self.projects.adopt_candidate(
                    self.project,
                    candidate,
                    candidate.source_files,
                    f"methodtrail: {change.title}",
                    run.metric,
                )
            else:
                self.projects.record_candidate(
                    self.project, candidate, status=node_status, metric=run.metric
                )
        elif assessment.decision == "stop":
            memory = self.memory_agent.summarize(contract, change, run, assessment)
            memory_id = self.store.put("memory", memory, [artifact_ids["assessment"]])
            artifact_ids["memory"] = memory_id
            variant_id = candidate.variant_id
            artifact_ids["variant"] = variant_id
            self.graph.add_node(
                PathNode(
                    variant_id=variant_id,
                    iteration=state.iteration,
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    relation=change.relation,
                    relation_warning=self.graph.relation_warning_for(change.parent_variant_id, change),
                    mutation_class=change.mutation_class,
                    question=change.research_question,
                    evidence_summary=memory.reusable_conclusion or assessment.reason,
                    applicable_conditions=memory.applicable_conditions,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    failure_risk=change.failure_risk,
                    status="deferred",
                    method=change.method,
                )
            )
            self.graph.record_outcome(proposal_node_id, variant_id, "stopped")
            assert self.project is not None and self.session is not None
            self.experiment_memory.add(
                MemoryCard(
                    task_id=contract.task_id,
                    session_id=self.session.session_id,
                    variant_id=variant_id,
                    parent_variant_id=change.parent_variant_id,
                    evidence_parent_ids=change.evidence_parent_ids,
                    question=change.research_question,
                    conclusion=memory.reusable_conclusion or assessment.reason,
                    measured_facts=[
                        f"metric={run.metric}",
                        f"wall_seconds={run.wall_seconds:.3f}",
                        f"return_code={run.return_code}",
                    ],
                    evidence=[assessment.reason],
                    applicable_conditions=memory.applicable_conditions,
                    relation=change.relation,
                    decision="stop",
                    metric=run.metric,
                    method_family=change.method.family,
                    changed_factors=change.method.changed_factors,
                    iteration=state.iteration,
                    title=change.title,
                    mutation_class=change.mutation_class.value,
                    change_logic=change.rationale,
                    method_components=change.method.components,
                    tags=[change.title, change.mutation_class.value, "stopped"],
                )
            )
            self.projects.record_candidate(
                self.project, candidate, status="stopped", metric=run.metric
            )
        return IterationResult(
            state.iteration,
            workspace,
            change,
            run,
            assessment,
            assessment.next_question,
            artifact_ids,
            completed_research=True,
        )

    def _record_failed_outcome(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        candidate: CandidateRecord,
        run: RunArtifact,
        proposal_node_id: str,
    ) -> None:
        """Keep technical failures available to later recovery choices."""

        summary = (run.stderr or run.stdout or "execution failed").strip()[:600]
        self.graph.add_node(
            PathNode(
                variant_id=candidate.variant_id,
                title=change.title,
                parent_variant_id=change.parent_variant_id,
                evidence_parent_ids=change.evidence_parent_ids,
                relation=change.relation,
                relation_warning=self.graph.relation_warning_for(change.parent_variant_id, change),
                mutation_class=change.mutation_class,
                question=change.research_question,
                evidence_summary=summary,
                applicable_conditions=["technical failure"],
                metric=run.metric,
                wall_seconds=run.wall_seconds,
                failure_risk=1.0,
                status="failed",
                method=change.method,
            )
        )
        self.graph.record_outcome(proposal_node_id, candidate.variant_id, "failed")

    def _record_bug(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        candidate: CandidateRecord,
        run: RunArtifact,
        phase: str,
        repair_step: int,
        recovery: Any | None = None,
        resolved: bool = False,
    ) -> None:
        """Record implementation trouble separately from measured evidence."""

        if self.session is None:
            return
        summary = (run.stderr or run.stdout or "technical failure").strip()[:1200]
        self.bug_memory.add(
            BugRecord(
                task_id=contract.task_id,
                session_id=self.session.session_id,
                variant_id=candidate.variant_id,
                phase=phase,
                failure_class=(recovery.failure_class if recovery else phase),
                message=summary,
                diagnosis=recovery.diagnosis if recovery else "",
                repair_directions=(recovery.repair_directions if recovery else []),
                action=(recovery.action if recovery else "continue_repair"),
                repair_step=repair_step,
                resolved=resolved,
                question=change.research_question,
                related_method=change.title,
            )
        )

    def _write_handoff(self, result: IterationResult) -> None:
        """Leave a compact restart point without replaying the full LLM chat."""

        assert self.project is not None and self.session is not None
        decision = (
            result.assessment.decision
            if result.assessment
            else "replan" if result.resume_mode == "replan" else "recovery"
        )
        metric = result.run.metric if result.run else None
        active_variant = (
            result.artifact_ids.get("variant")
            if decision == "adopt"
            else self.project.incumbent_variant_id
        )
        handoff = "\n".join(
            [
                "# Session handoff",
                "",
                f"- Iteration: {result.iteration}",
                f"- Decision: {decision}",
                f"- Current accepted variant: {active_variant or 'baseline'}",
                f"- Latest metric: {metric if metric is not None else 'not measured'}",
                f"- Next question: {result.next_hypothesis or 'none'}",
                f"- Candidate workspace: {result.workspace}",
            ]
        )
        self.projects.update_session(
            self.project,
            self.session,
            current_variant_id=active_variant,
            next_question=result.next_hypothesis,
            handoff=handoff + "\n",
        )
        self.projects.append_event(
            self.project,
            self.session.session_id,
            "iteration_finished" if result.completed_research else "technical_attempt",
            {
                "iteration": result.iteration,
                "completed_research": result.completed_research,
                "decision": decision,
                "metric": metric,
                "variant_id": result.artifact_ids.get("variant"),
                "next_question": result.next_hypothesis,
                "artifact_ids": result.artifact_ids,
            },
        )
        # Keep an offline page current while a long run is in progress.  A
        # dashboard failure must never interrupt the research process.
        try:
            dashboard_path = (
                self.projects.project_path(self.project)
                / "sessions"
                / self.session.session_id
                / "dashboard.html"
            )
            self.export_dashboard(
                dashboard_path,
                project_id=self.project.project_id,
                session_id=self.session.session_id,
            )
        except Exception as exc:  # noqa: BLE001 - dashboard export must not stop research.
            logger.debug("dashboard export failed: %s", exc)

    def _state(
        self,
        contract: TaskContract,
        remaining_seconds: int,
        parent_variant_id: str | None,
    ) -> ResearchState:
        assert self.project is not None and self.session is not None
        candidates = self.projects.session_candidates(
            self.project, self.session.session_id
        )
        recent_candidates = candidates[-8:]
        facts = []
        best_metric = self.project.incumbent_metric
        for candidate in recent_candidates:
            metric = candidate.metric
            if metric is not None:
                if best_metric is None:
                    best_metric = metric
                elif contract.maximize_metric:
                    best_metric = max(best_metric, metric)
                else:
                    best_metric = min(best_metric, metric)
                facts.append(
                    f"variant={candidate.variant_id}, metric={metric}, status={candidate.status}"
                )
            elif candidate.status == "failed":
                facts.append(f"failed candidate: {candidate.variant_id}")
        query = " ".join(
            [contract.description, *facts, self.session.next_question or ""]
        )
        completed_rounds = sum(
            1
            for event in self.projects.events(self.project, self.session.session_id, limit=100000)
            if event["kind"] == "iteration_finished"
            and event["payload"].get("completed_research", True)
        )
        return ResearchState(
            task_id=contract.task_id,
            iteration=completed_rounds + 1,
            remaining_seconds=remaining_seconds,
            best_metric=best_metric,
            incumbent_metric=self.project.incumbent_metric,
            incumbent_variant_id=self.project.incumbent_variant_id,
            plateau_rounds=self.session.consecutive_non_improving,
            research_round=completed_rounds + 1,
            repair_step=0,
            recent_facts=facts,
            unresolved_questions=[
                event["payload"].get("next_question", "")
                for event in self.projects.events(
                    self.project, self.session.session_id, limit=5
                )
                if event["kind"] == "iteration_finished"
                and event["payload"].get("next_question")
            ],
            graph_context=self.graph.context_for(parent_variant_id, query),
            path_hints=self.graph.expansion_hints(parent_variant_id),
            # Portfolio.json remains available for reports and compatibility,
            # but its multi-objective Pareto summary is not a separate decision
            # signal. The graph and cards already carry the evidence the agents
            # need, so keep this prompt channel empty to reduce noise.
            portfolio_context=[],
            memory_context=self.experiment_memory.search(
                contract.task_id,
                query,
                seed_variant_ids=[parent_variant_id] if parent_variant_id else None,
            ),
            prior_evidence_context=self._prior_evidence_context(contract),
            method_pool=self.graph.method_pool(limit=60),
            memory_graph_context=self.experiment_memory.graph_profile(contract.task_id),
        )

    def export_trajectory(self, target: str | Path) -> Path:
        """Exports typed artifacts for later retrieval, preference construction, or offline learning."""

        target_path = Path(target)
        payload: dict[str, Any] = {
            "artifacts": self.store.recent(limit=100000),
            "graph": self.graph.latest_nodes(limit=100000),
        }
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target_path

    def export_dashboard(
        self,
        target: str | Path,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> Path:
        """Write a self-contained offline view of method and memory graphs."""

        project = self.project
        if project is None:
            project = self.projects.load_project(project_id or self.project_id or "")
        selected_session = session_id or self.session_id
        if not selected_session:
            raise ValueError("session_id is required when exporting a dashboard")
        session = self.projects.load_session(project, selected_session)
        project_path = self.projects.project_path(project)
        graph = ExperimentPathGraph(project_path / "memory" / "experiment_graph.json")
        memory = ExperimentMemory(project_path / "memory")
        payload = {
            "project": asdict(project),
            "session": asdict(session),
            "events": self.projects.events(project, selected_session, limit=100000),
            "method_graph": graph.export_payload(),
            "memory_graph": {
                "directed": True,
                "nodes": list(memory.graph.nodes.values()),
                "edges": list(memory.graph.edges),
            },
            "artifacts": self.store.recent(limit=500),
        }
        return write_dashboard(target, payload)
