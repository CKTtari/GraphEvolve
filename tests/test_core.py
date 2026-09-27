from __future__ import annotations

import sys
from pathlib import Path

from methodtrail.artifacts import ArtifactStore
from methodtrail.execution import Executor, Verifier
from methodtrail.memory import ExperimentMemory, MemoryCard
from methodtrail.orchestrator import MethodTrail
from methodtrail.path_graph import ExperimentPathGraph
from methodtrail.project import ProjectManager
from methodtrail.schemas import (
    AssessmentArtifact,
    CandidatePath,
    CandidateProposalArtifact,
    CandidateSelectionArtifact,
    ChangeRequestArtifact,
    CodePlanArtifact,
    FileEdit,
    HypothesisArtifact,
    MethodDescriptor,
    MutationClass,
    PathNode,
    RecoveryArtifact,
    TaskContract,
    ValueWeights,
)
from methodtrail.workspace import WorkspaceManager


def test_artifact_store_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts.sqlite")
    artifact_id = store.put("fact", {"metric": 0.42})
    assert store.get(artifact_id)["payload"] == {"metric": 0.42}


def test_path_priority_prefers_information_with_same_cost(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    candidates = [
        CandidatePath(
            variant_id="a",
            title="low",
            relation="explore",
            expected_gain=0.01,
            information_gain=0.01,
            estimated_seconds=30,
            failure_risk=0.1,
        ),
        CandidatePath(
            variant_id="b",
            title="high",
            relation="explore",
            expected_gain=0.01,
            information_gain=0.08,
            estimated_seconds=30,
            failure_risk=0.1,
        ),
    ]
    ranked = graph.rank(candidates, 300, ValueWeights())
    assert ranked[0][0].variant_id == "b"


def test_path_graph_keeps_directed_history_and_matched_branches(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    graph.add_node(
        PathNode(
            variant_id="root",
            title="linear baseline",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="build baseline",
            evidence_summary="baseline runs",
            status="adopted",
            method=MethodDescriptor(family="linear", components={"features": "raw"}),
        )
    )
    graph.add_node(
        PathNode(
            variant_id="active",
            title="regularized linear model",
            parent_variant_id="root",
            relation="deepen",
            mutation_class=MutationClass.CONFIGURATION,
            question="does regularization help?",
            evidence_summary="stable gain",
            status="adopted",
            method=MethodDescriptor(
                family="linear", components={"features": "raw", "l2": "1.0"}, changed_factors=["l2"]
            ),
        )
    )
    graph.add_node(
        PathNode(
            variant_id="sibling",
            title="feature ablation",
            parent_variant_id="root",
            relation="ablate",
            mutation_class=MutationClass.CONFIGURATION,
            question="are raw features necessary?",
            evidence_summary="feature removal hurt",
            status="deferred",
            method=MethodDescriptor(family="linear", components={"features": "none"}),
        )
    )
    context = graph.context_for("active", "regularization features")
    roles = {(row["node_id"], row["context_role"]) for row in context}
    assert ("root", "path_history") in roles
    assert ("sibling", "matched_alternative") in roles


def test_path_graph_attaches_and_normalizes_candidate_branches(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    graph.add_node(
        PathNode(
            variant_id="parent",
            title="tree model",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="baseline",
            evidence_summary="valid",
            status="adopted",
            method=MethodDescriptor(family="tree", components={"depth": "4"}),
        )
    )
    proposed = ChangeRequestArtifact(
        title="increase depth",
        mutation_class=MutationClass.CONFIGURATION,
        relation="explore",
        research_question="does a deeper tree help?",
        rationale="one controlled change",
        method=MethodDescriptor(
            family="tree", components={"depth": "6"}, changed_factors=["depth"]
        ),
    )
    attached = graph.attach_proposals("parent", [proposed], iteration=2)
    proposal_id, normalized = attached[0]
    assert normalized.relation == "deepen"
    assert graph.graph.edges["parent", proposal_id]["relation"] == "deepen"


def test_path_graph_returns_unselected_branch_as_frontier(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    graph.add_node(
        PathNode(
            variant_id="parent",
            title="parent",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="parent",
            evidence_summary="parent",
            status="adopted",
        )
    )
    attached = graph.attach_proposals(
        "parent",
        [
            ChangeRequestArtifact(
                title="branch one",
                mutation_class=MutationClass.IMPLEMENTATION,
                relation="explore",
                research_question="branch one",
                rationale="one",
            ),
            ChangeRequestArtifact(
                title="branch two",
                mutation_class=MutationClass.IMPLEMENTATION,
                relation="explore",
                research_question="branch two",
                rationale="two",
            ),
        ],
        iteration=1,
    )
    graph.set_proposal_priority(attached[0][0], 0.9)
    graph.set_proposal_priority(attached[1][0], 0.5)
    graph.mark_proposal_selected(attached[0][0])
    graph.record_outcome(attached[0][0], "active", "adopted")
    graph.add_node(
        PathNode(
            variant_id="active",
            title="active",
            parent_variant_id="parent",
            relation="explore",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="active",
            evidence_summary="active",
            status="adopted",
        )
    )
    context = graph.context_for("active", "branch")
    assert any(row["node_id"] == attached[1][0] for row in context)


def test_path_graph_blends_llm_estimate_with_measured_history(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    graph.add_node(
        PathNode(
            variant_id="base",
            title="base",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="base",
            evidence_summary="base",
            metric=0.50,
            status="adopted",
            method=MethodDescriptor(family="tree", components={"depth": "4"}),
        )
    )
    graph.add_node(
        PathNode(
            variant_id="observed",
            title="deepen",
            parent_variant_id="base",
            relation="deepen",
            mutation_class=MutationClass.CONFIGURATION,
            question="deepen",
            evidence_summary="measured",
            metric=0.70,
            wall_seconds=20,
            status="adopted",
            method=MethodDescriptor(family="tree", components={"depth": "6"}, changed_factors=["depth"]),
        )
    )
    candidate = CandidatePath(
        variant_id="new",
        title="new deepen",
        relation="deepen",
        expected_gain=0.01,
        information_gain=0.1,
        estimated_seconds=100,
        failure_risk=0.2,
        method=MethodDescriptor(family="tree", components={"depth": "8"}, changed_factors=["depth"]),
    )
    calibrated = graph.calibrate(candidate)
    assert calibrated.expected_gain > candidate.expected_gain
    assert calibrated.estimated_seconds < candidate.estimated_seconds


def test_workspace_verifier_and_executor(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    contract = TaskContract(
        task_id="toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        allow_self_reported_metric=True,
    )
    plan = CodePlanArtifact(
        summary="write a runnable toy solution",
        expected_test="writes metric and predictions",
        edits=[
            FileEdit(
                operation="create",
                path="solution.py",
                purpose="toy solution",
                new_text=(
                    "import json\n"
                    "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n"
                    "json.dump({'score': 0.75}, open('metrics.json', 'w', encoding='utf-8'))\n"
                ),
            )
        ],
    )
    change = ChangeRequestArtifact(
        title="toy",
        mutation_class=MutationClass.IMPLEMENTATION,
        research_question="can it run?",
        allowed_files=["solution.py"],
        rationale="test",
    )
    workspace = WorkspaceManager(tmp_path / "runs").create(template)
    WorkspaceManager(tmp_path / "runs").apply_plan(workspace, plan, contract, change)
    assert Verifier().verify(workspace, contract).passed
    run = Executor().run(workspace, contract)
    assert run.return_code == 0
    assert run.metric == 0.75
    assert run.output_files == ["predictions.csv"]


def test_workspace_applies_only_a_unique_local_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        "def score(x):\n    return x + 1\n\nprint(score(2))\n", encoding="utf-8"
    )
    contract = TaskContract(
        task_id="patch-toy",
        description="toy",
        workspace_template=str(workspace),
        allowed_data_paths=[],
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
        editable_paths=["solution.py"],
    )
    change = ChangeRequestArtifact(
        title="local replacement",
        mutation_class=MutationClass.IMPLEMENTATION,
        research_question="change one return value",
        allowed_files=["solution.py"],
        rationale="test precise edit",
    )
    plan = CodePlanArtifact(
        summary="change one local statement",
        expected_test="score changes",
        edits=[
            FileEdit(
                operation="replace",
                path="solution.py",
                old_text="    return x + 1\n",
                new_text="    return x + 2\n",
                purpose="adjust score",
            )
        ],
    )
    WorkspaceManager(tmp_path / "runs").apply_plan(workspace, plan, contract, change)
    assert "return x + 2" in (workspace / "solution.py").read_text(encoding="utf-8")


def test_workspace_rejects_data_file_edits(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    contract = TaskContract(
        task_id="data-protection-toy",
        description="toy",
        workspace_template=str(workspace),
        allowed_data_paths=["train.csv"],
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
    )
    change = ChangeRequestArtifact(
        title="bad data edit",
        mutation_class=MutationClass.IMPLEMENTATION,
        research_question="should be rejected",
        allowed_files=["train.csv"],
        rationale="test data protection",
    )
    plan = CodePlanArtifact(
        summary="bad plan",
        expected_test="never runs",
        edits=[
            FileEdit(
                operation="replace",
                path="train.csv",
                old_text="x,y\n1,0\n",
                new_text="x,y\n1,1\n",
                purpose="invalid data modification",
            )
        ],
    )
    try:
        WorkspaceManager(tmp_path / "runs").apply_plan(workspace, plan, contract, change)
    except PermissionError as exc:
        assert "task data cannot be edited" in str(exc)
    else:
        raise AssertionError("data edit should have been rejected")


class FakeLLM:
    """Provides typed role outputs to test orchestration without an external API."""

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is HypothesisArtifact:
            return HypothesisArtifact(
                observation="no previous result",
                hypothesis="a minimal implementation can produce a valid score",
                evidence_needed=["metric file"],
                comparison_plan="run the solution",
                adoption_condition="score is present",
                fallback_condition="inspect failure",
            )
        if response_model is CandidateProposalArtifact:
            return CandidateProposalArtifact(
                candidates=[
                    ChangeRequestArtifact(
                        title="write toy solution",
                        mutation_class=MutationClass.IMPLEMENTATION,
                        research_question="can a generated program complete the task?",
                        allowed_files=["solution.py"],
                        expected_gain=0.1,
                        information_gain=0.5,
                        estimated_seconds=10,
                        failure_risk=0.1,
                        rationale="establishes a runnable baseline",
                    )
                ]
            )
        if response_model is CandidateSelectionArtifact:
            return CandidateSelectionArtifact(
                selected_index=0,
                reason="the only candidate establishes a runnable baseline",
            )
        if response_model is CodePlanArtifact:
            return CodePlanArtifact(
                summary="write a valid solution",
                expected_test="metric and prediction file exist",
                edits=[
                    FileEdit(
                        operation="create",
                        path="solution.py",
                        purpose="generated toy solution",
                        new_text=(
                            "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n"
                        ),
                    )
                ],
            )
        if response_model is AssessmentArtifact:
            return AssessmentArtifact(
                decision="adopt",
                reason="the measured result is valid",
                reusable_conclusion="minimal implementation produced a valid result",
                applicable_conditions=["toy workspace"],
                next_question="try a stronger feature",
            )
        raise AssertionError(f"unexpected response type: {response_model}")


class RepairingFakeLLM(FakeLLM):
    """First implementation fails; the coding role repairs it from the failure record."""

    def __init__(self) -> None:
        self.code_plan_calls = 0

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is RecoveryArtifact:
            return RecoveryArtifact(
                failure_class="runtime",
                diagnosis="the solution raises before writing predictions",
                repair_directions=["replace the failing body with prediction output"],
                preserve_question=True,
                return_to_reflection_reason="repair the current implementation",
            )
        if response_model is CodePlanArtifact:
            self.code_plan_calls += 1
            content = (
                "raise RuntimeError('intentional first-pass failure')\n"
                if self.code_plan_calls == 1
                else "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n"
            )
            return CodePlanArtifact(
                summary="first attempt or autonomous repair",
                expected_test="prediction file exists",
                edits=[
                    FileEdit(
                        operation="create" if self.code_plan_calls == 1 else "replace",
                        path="solution.py",
                        purpose="solution",
                        old_text=None if self.code_plan_calls == 1 else "raise RuntimeError('intentional first-pass failure')\n",
                        new_text=content,
                    )
                ],
            )
        return super().complete(system, user, response_model)


def test_orchestrator_runs_a_full_adopted_iteration(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="orchestration-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        protected_paths=["evaluate.py", "metrics.json"],
    )
    trail = MethodTrail(tmp_path / "project", FakeLLM())
    result = trail.run_iteration(contract, remaining_seconds=120)
    assert result.assessment is not None
    assert result.assessment.decision == "adopt"
    assert result.run is not None and result.run.metric == 0.8
    assert any(node["status"] == "adopted" for node in trail.graph.latest_nodes())


def test_orchestrator_repairs_a_failed_generated_program(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="repair-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        max_repair_attempts=1,
        protected_paths=["evaluate.py", "metrics.json"],
    )
    trail = MethodTrail(tmp_path / "project", RepairingFakeLLM())
    result = trail.run_iteration(contract, remaining_seconds=120)
    assert result.assessment is not None and result.assessment.decision == "adopt"
    assert result.run is not None and result.run.metric == 0.8
    assert trail.coding.llm.code_plan_calls == 2


def test_external_evaluator_overwrites_model_reported_metric(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        "import json\n"
        "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n"
        "json.dump({'score': 0.1}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    (workspace / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.9}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="evaluated-toy",
        description="toy",
        workspace_template=str(workspace),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        protected_paths=["evaluate.py", "metrics.json"],
    )
    assert Verifier().verify(workspace, contract).passed
    result = Executor().run(workspace, contract)
    assert result.return_code == 0
    assert result.evaluation_return_code == 0
    assert result.metric == 0.9


def test_executor_collects_auxiliary_metrics_and_constraints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n",
        encoding="utf-8",
    )
    (workspace / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.9, 'stability': 0.72}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="multi-metric-toy",
        description="toy",
        workspace_template=str(workspace),
        allowed_data_paths=[],
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        auxiliary_metric_names=["stability"],
        metric_constraints={"stability": 0.8},
        required_outputs=["predictions.csv"],
        protected_paths=["evaluate.py", "metrics.json"],
    )
    result = Executor().run(workspace, contract)
    assert result.metrics == {"score": 0.9, "stability": 0.72}
    assert result.metric == 0.9
    assert result.metric_constraints_passed is False


def test_private_evaluator_keeps_labels_out_of_public_workspace(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    (public / "solution.py").write_text(
        "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n",
        encoding="utf-8",
    )
    (private / "labels.txt").write_text("1\n", encoding="utf-8")
    (private / "evaluate.py").write_text(
        "import json\n"
        "assert open('labels.txt', encoding='utf-8').read().strip() == '1'\n"
        "assert open('predictions.csv', encoding='utf-8').read().strip().endswith('1')\n"
        "json.dump({'score': 0.9}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="private-eval-toy",
        description="toy",
        workspace_template=str(public),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        private_evaluator_dir=str(private),
        private_evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        protected_paths=["metrics.json"],
    )
    public_context = contract.llm_context()
    assert "private_evaluator_dir" not in public_context
    assert "private_evaluation_command" not in public_context
    assert str(private) not in str(public_context)
    result = Executor().run(public, contract)
    assert result.return_code == 0
    assert result.metric == 0.9
    assert not (public / "labels.txt").exists()
    assert not (public / "metrics.json").exists()


def test_project_candidate_can_be_adopted_and_reused(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "train.csv").write_text("feature,target\n1,0\n", encoding="utf-8")
    (template / "evaluate.py").write_text("print('ok')\n", encoding="utf-8")
    contract = TaskContract(
        task_id="versioned-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=["train.csv"],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
    )
    manager = ProjectManager(tmp_path / "state")
    project = manager.ensure_project(contract, project_id="toy-primary")
    sibling = manager.ensure_project(contract, project_id="toy-alternative")
    assert project.repository != sibling.repository
    session = manager.start_session(project)
    candidate = manager.create_candidate(project, session, contract, None)
    workspace = Path(candidate.workspace)
    assert (workspace / "train.csv").exists()
    (workspace / "solution.py").write_text("print('first')\n", encoding="utf-8")
    adopted = manager.adopt_candidate(
        project, candidate, ["solution.py"], "first solution", 0.8
    )
    assert adopted.status == "adopted"
    assert adopted.commit
    child = manager.create_candidate(project, session, contract, adopted.variant_id)
    assert "first" in (Path(child.workspace) / "solution.py").read_text(
        encoding="utf-8"
    )
    manager.set_incumbent(project, adopted.variant_id)
    assert manager.load_project("toy-primary").incumbent_commit == adopted.commit
    manager.update_session(project, session, status="paused", next_question="try a child")
    resumed = manager.start_session(project, session.session_id)
    assert resumed.status == "active"
    assert resumed.next_question == "try a child"


def test_experiment_memory_returns_relevant_cards(tmp_path: Path) -> None:
    memory = ExperimentMemory(tmp_path / "memory")
    memory.add(
        MemoryCard(
            task_id="task",
            session_id="session",
            variant_id="v1",
            question="does categorical one hot encoding improve accuracy",
            conclusion="one hot encoding improved the validation result",
            relation="deepen",
            decision="adopt",
            tags=["categorical", "encoding"],
        )
    )
    memory.add(
        MemoryCard(
            task_id="task",
            session_id="session",
            variant_id="v2",
            question="does a large neural network fit in memory",
            conclusion="the run exceeded the memory budget",
            relation="recover",
            decision="defer",
            tags=["neural", "memory"],
        )
    )
    results = memory.search("task", "categorical encoding experiment")
    assert results and results[0]["variant_id"] == "v1"
