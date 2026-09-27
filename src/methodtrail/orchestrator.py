"""The serial research process that coordinates the main agent and four return paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents import (
    AssessAgent,
    ChooseAgent,
    CodingAgent,
    EvidenceAgent,
    MemoryAgent,
    RecoveryAgent,
    ReflectionAgent,
)
from .artifacts import ArtifactStore
from .execution import Executor, Verifier
from .llm import StructuredLLM
from .memory import ExperimentMemory, MemoryCard
from .path_graph import ExperimentPathGraph
from .portfolio import PortfolioManager, VariantProfile
from .project import CandidateRecord, ProjectManager, ProjectRecord, SessionRecord
from .repository import RepoMap
from .schemas import (
    AssessmentArtifact,
    CandidatePath,
    ChangeRequestArtifact,
    MutationClass,
    PathNode,
    ResearchState,
    RunArtifact,
    TaskContract,
    ValueWeights,
)
from .workspace import WorkspaceManager


@dataclass
class IterationResult:
    iteration: int
    workspace: Path
    change: ChangeRequestArtifact
    run: RunArtifact | None
    assessment: AssessmentArtifact | None
    next_hypothesis: str | None
    artifact_ids: dict[str, str]


class MethodTrail:
    """Runs one serial experiment at a time and writes every fact into artifacts."""

    def __init__(
        self,
        project_root: str | Path,
        llm: StructuredLLM,
        session_id: str | None = None,
        project_id: str | None = None,
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
        self.session_id = session_id
        self.project_id = project_id
        self.project: ProjectRecord | None = None
        self.session: SessionRecord | None = None
        self.verifier = Verifier()
        self.executor = Executor()
        self.reflection = ReflectionAgent(llm)
        self.choose = ChooseAgent(llm)
        self.coding = CodingAgent(llm)
        self.evidence = EvidenceAgent(llm)
        self.recovery = RecoveryAgent(llm)
        self.assess = AssessAgent(llm)
        self.memory_agent = MemoryAgent(llm)

    def _ensure_project_session(self, contract: TaskContract) -> None:
        if self.project is None:
            self.project = self.projects.ensure_project(contract, self.project_id)
            project_path = self.projects.project_path(self.project)
            self.graph = ExperimentPathGraph(project_path / "memory" / "experiment_graph.json")
            self.portfolio = PortfolioManager(project_path / "memory" / "portfolio.json")
            self.experiment_memory = ExperimentMemory(project_path / "memory")
        if self.session is None:
            self.session = self.projects.start_session(self.project, self.session_id)
            self.session_id = self.session.session_id

    def run_iteration(
        self,
        contract: TaskContract,
        remaining_seconds: int,
        parent_variant_id: str | None = None,
        trigger: str | None = None,
    ) -> IterationResult:
        self._ensure_project_session(contract)
        assert self.project is not None and self.session is not None
        if parent_variant_id is None:
            parent_variant_id = self.project.incumbent_variant_id
        state = self._state(contract, remaining_seconds, parent_variant_id)
        state_id = self.store.put("research_state", state)
        self.projects.append_event(
            self.project,
            self.session.session_id,
            "research_state",
            {"artifact_id": state_id, "iteration": state.iteration},
        )
        hypothesis = self.reflection.refine(contract, state, trigger)
        hypothesis_id = self.store.put("hypothesis", hypothesis, [state_id])
        proposals = self.choose.propose(contract, state, hypothesis)
        attached = self.graph.attach_proposals(
            parent_variant_id, proposals.candidates, state.iteration
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
        ranked = self.graph.rank(
            candidates,
            remaining_seconds,
            ValueWeights(),
            maximize_metric=contract.maximize_metric,
        )
        candidate_indices = {candidate.variant_id: index for index, candidate in enumerate(candidates)}
        for candidate_path, priority in ranked:
            self.graph.set_proposal_priority(candidate_path.variant_id, priority)
        ranked_payload = [
            {
                "index": candidate_indices[candidate.variant_id],
                "priority": priority,
                "candidate": candidate.model_dump(mode="json"),
            }
            for candidate, priority in ranked
        ]
        priority_id = self.store.put(
            "candidate_priorities", ranked_payload, [proposal_id]
        )
        selection = self.choose.select(contract, state, hypothesis, ranked_payload)
        if selection.selected_index >= len(proposals.candidates):
            raise RuntimeError(
                f"Choose selected invalid candidate index {selection.selected_index}"
            )
        change = proposals.candidates[selection.selected_index].model_copy(
            update={"parent_variant_id": parent_variant_id}
        )
        proposal_node_id = proposal_node_ids[selection.selected_index]
        self.graph.mark_proposal_selected(proposal_node_id)
        selection_id = self.store.put(
            "candidate_selection",
            selection,
            [proposal_id, priority_id],
        )
        change_id = self.store.put("change_request", change, [selection_id])

        candidate = self.projects.create_candidate(
            self.project,
            self.session,
            contract,
            parent_variant_id,
        )
        workspace = Path(candidate.workspace)
        repo_context = RepoMap(workspace).retrieve(
            f"{change.research_question} {change.title}"
        )
        plan = self.coding.write_plan(contract, change, repo_context)
        plan_id = self.store.put("code_plan", plan, [change_id, priority_id])
        candidate.source_files = sorted({edit.path for edit in plan.edits})
        self.projects.record_candidate(self.project, candidate, status="prepared")

        try:
            diff = self.workspaces.apply_plan(workspace, plan, contract, change)
        except (PermissionError, ValueError) as exc:
            self.projects.record_candidate(self.project, candidate, status="failed")
            result = self._recover_from_preflight(
                contract, state, change, workspace, str(exc), [plan_id]
            )
            assert result.run is not None
            self._record_failed_outcome(
                contract, change, candidate, result.run, proposal_node_id
            )
            self._write_handoff(result)
            return result
        implementation_id = self.store.put(
            "implementation",
            {
                "workspace": str(workspace),
                "variant_id": candidate.variant_id,
                "diff": diff,
                "entrypoint": contract.solution_entrypoint,
            },
            [plan_id],
        )

        verification = self.verifier.verify(workspace, contract)
        verify_id = self.store.put("verification", verification, [implementation_id])
        if not verification.passed:
            detail = "\n".join(
                check["detail"] for check in verification.checks if not check["passed"]
            )
            self.projects.record_candidate(self.project, candidate, status="failed")
            result = self._recover_from_preflight(
                contract, state, change, workspace, detail, [verify_id]
            )
            assert result.run is not None
            self._record_failed_outcome(
                contract, change, candidate, result.run, proposal_node_id
            )
            self._write_handoff(result)
            return result

        run = self.executor.run(workspace, contract)
        run_id = self.store.put("run", run, [verify_id])
        repair_ids: list[str] = []
        repair_attempt = 0
        while (
            (run.return_code != 0 or run.timed_out or run.metric is None)
            and repair_attempt < contract.max_repair_attempts
        ):
            repair_attempt += 1
            recovery = self.recovery.diagnose(contract, state, run)
            recovery_id = self.store.put("recovery", recovery, [run_id])
            repair_ids.append(recovery_id)
            repair_change = change.model_copy(
                update={
                    "title": f"repair {repair_attempt}: {change.title}",
                    "mutation_class": MutationClass.RECOVERY,
                    "relation": "recover",
                    "rationale": recovery.diagnosis,
                }
            )
            repair_plan = self.coding.repair(
                contract,
                repair_change,
                run,
                recovery,
                RepoMap(workspace).retrieve(
                    f"{change.research_question} {recovery.diagnosis}"
                ),
            )
            repair_plan_id = self.store.put("repair_plan", repair_plan, [recovery_id])
            repair_ids.append(repair_plan_id)
            candidate.source_files = sorted(
                set(candidate.source_files).union(edit.path for edit in repair_plan.edits)
            )
            self.projects.record_candidate(self.project, candidate, status="prepared")
            try:
                self.workspaces.apply_plan(workspace, repair_plan, contract, repair_change)
            except (PermissionError, ValueError) as exc:
                run = RunArtifact(
                    command=contract.run_command,
                    return_code=-2,
                    timed_out=False,
                    wall_seconds=0.0,
                    stdout="",
                    stderr=f"repair attempt {repair_attempt} could not be applied: {exc}",
                )
                run_id = self.store.put("run", run, [repair_plan_id])
                continue
            verification = self.verifier.verify(workspace, contract)
            verify_id = self.store.put("repair_verification", verification, [repair_plan_id])
            if not verification.passed:
                detail = "\n".join(
                    check["detail"] for check in verification.checks if not check["passed"]
                )
                run = RunArtifact(
                    command=contract.run_command,
                    return_code=-2,
                    timed_out=False,
                    wall_seconds=0.0,
                    stdout="",
                    stderr=f"repair attempt {repair_attempt} failed verification: {detail}",
                )
                run_id = self.store.put("run", run, [verify_id])
                continue
            run = self.executor.run(workspace, contract)
            run_id = self.store.put("repair_run", run, [verify_id])
        if run.return_code != 0 or run.timed_out or run.metric is None:
            self.projects.record_candidate(
                self.project, candidate, status="failed", metric=run.metric
            )
            recovery = self.recovery.diagnose(contract, state, run)
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
                next_hypothesis=recovery.return_to_reflection_reason,
                artifact_ids={
                    "state": state_id,
                    "hypothesis": hypothesis_id,
                    "change": change_id,
                    "plan": plan_id,
                    "run": run_id,
                    "recovery": recovery_id,
                    "repair": repair_ids,
                },
            )
            self._write_handoff(result)
            return result

        assessment = self.assess.assess(contract, state, change, run)
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
        assessment_id = self.store.put("assessment", assessment, [run_id])
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
        self._write_handoff(result)
        return result

    def run_research(
        self,
        contract: TaskContract,
        total_seconds: int,
        max_iterations: int,
        parent_variant_id: str | None = None,
    ) -> list[IterationResult]:
        """Run serial iterations until a stopping decision, budget exhaustion, or iteration limit."""

        remaining_seconds = total_seconds
        trigger: str | None = None
        current_parent = parent_variant_id
        results: list[IterationResult] = []
        for _ in range(max_iterations):
            if remaining_seconds <= 0:
                break
            result = self.run_iteration(
                contract,
                remaining_seconds=remaining_seconds,
                parent_variant_id=current_parent,
                trigger=trigger,
            )
            results.append(result)
            elapsed = max(1, int(result.run.wall_seconds)) if result.run else 1
            remaining_seconds -= elapsed
            if (
                result.assessment
                and result.assessment.decision == "stop"
                and len(results) >= contract.minimum_iterations
            ):
                break
            trigger = result.next_hypothesis
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
        return results

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
            and candidate.status in {"needs_evidence", "deferred"}
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

    def _recover_from_preflight(
        self,
        contract: TaskContract,
        state: ResearchState,
        change: ChangeRequestArtifact,
        workspace: Path,
        detail: str,
        parent_ids: list[str],
    ) -> IterationResult:
        run = RunArtifact(
            command=contract.run_command,
            return_code=-2,
            timed_out=False,
            wall_seconds=0.0,
            stdout="",
            stderr=detail,
        )
        recovery = self.recovery.diagnose(contract, state, run)
        recovery_id = self.store.put("recovery", recovery, parent_ids)
        return IterationResult(
            iteration=state.iteration,
            workspace=workspace,
            change=change,
            run=run,
            assessment=None,
            next_hypothesis=recovery.return_to_reflection_reason,
            artifact_ids={"recovery": recovery_id},
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
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    relation=change.relation,
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
                    tags=[change.title, change.mutation_class.value, change.relation, *memory.applicable_conditions],
                )
            )
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
                next_hypothesis.hypothesis,
                artifact_ids,
            )

        if assessment.decision in {"adopt", "defer"}:
            memory = self.memory_agent.summarize(contract, change, run, assessment)
            memory.decision = assessment.decision
            memory_id = self.store.put("memory", memory, [artifact_ids["assessment"]])
            artifact_ids["memory"] = memory_id
            variant_id = candidate.variant_id
            artifact_ids["variant"] = variant_id
            self.graph.add_node(
                PathNode(
                    variant_id=variant_id,
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    relation=change.relation,
                    mutation_class=change.mutation_class,
                    question=change.research_question,
                    evidence_summary=memory.reusable_conclusion or assessment.reason,
                    applicable_conditions=memory.applicable_conditions,
                    metric=run.metric,
                    wall_seconds=run.wall_seconds,
                    failure_risk=change.failure_risk,
                    status="adopted" if assessment.decision == "adopt" else "deferred",
                    method=change.method,
                )
            )
            self.graph.record_outcome(
                proposal_node_id,
                variant_id,
                "adopted" if assessment.decision == "adopt" else "deferred",
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
                    self.project, candidate, status="deferred", metric=run.metric
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
                    title=change.title,
                    parent_variant_id=change.parent_variant_id,
                    relation=change.relation,
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

        summary = (run.stderr or run.stdout or "execution failed").strip()
        summary = summary[:600]
        self.graph.add_node(
            PathNode(
                variant_id=candidate.variant_id,
                title=change.title,
                parent_variant_id=change.parent_variant_id,
                relation=change.relation,
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
        if self.project is not None and self.session is not None:
            self.experiment_memory.add(
                MemoryCard(
                    task_id=contract.task_id,
                    session_id=self.session.session_id,
                    variant_id=candidate.variant_id,
                    parent_variant_id=change.parent_variant_id,
                    question=change.research_question,
                    conclusion="The implementation could not complete this experiment.",
                    measured_facts=[
                        f"return_code={run.return_code}",
                        f"wall_seconds={run.wall_seconds:.3f}",
                    ],
                    evidence=[summary],
                    applicable_conditions=["technical failure"],
                    relation=change.relation,
                    decision="recovery",
                    metric=run.metric,
                    tags=[change.title, change.mutation_class.value, "failure"],
                )
            )

    def _write_handoff(self, result: IterationResult) -> None:
        """Leave a compact restart point without replaying the full LLM chat."""

        assert self.project is not None and self.session is not None
        decision = result.assessment.decision if result.assessment else "recovery"
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
            "iteration_finished",
            {
                "iteration": result.iteration,
                "decision": decision,
                "metric": metric,
                "variant_id": result.artifact_ids.get("variant"),
                "next_question": result.next_hypothesis,
                "artifact_ids": result.artifact_ids,
            },
        )

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
        recent_candidates = candidates[-5:]
        facts = []
        best_metric = None
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
        return ResearchState(
            task_id=contract.task_id,
            iteration=len(candidates) + 1,
            remaining_seconds=remaining_seconds,
            best_metric=best_metric,
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
            portfolio_context=[
                profile.model_dump(mode="json") for profile in self.portfolio.pareto()
            ],
            memory_context=self.experiment_memory.search(contract.task_id, query),
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
