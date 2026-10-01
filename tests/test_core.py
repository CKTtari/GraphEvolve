from __future__ import annotations

import sys
from pathlib import Path

from methodtrail.agents import _context
from methodtrail.artifacts import ArtifactStore
from methodtrail.execution import Executor, Verifier
from methodtrail.llm import LLMDeadlineExceeded
from methodtrail.memory import BugMemory, ExperimentMemory, MemoryCard
from methodtrail.orchestrator import MethodTrail
from methodtrail.path_graph import ExperimentPathGraph, candidate_coverage_gap
from methodtrail.project import ProjectManager
from methodtrail.repository import RepoMap
from methodtrail.schemas import (
    AssessmentArtifact,
    CandidatePath,
    CandidateProposalArtifact,
    CandidateSelectionArtifact,
    ChangeRequestArtifact,
    CodePlanArtifact,
    FileEdit,
    HypothesisArtifact,
    ImplementationReviewArtifact,
    MethodDescriptor,
    MutationClass,
    PathNode,
    RecoveryArtifact,
    TaskContract,
    ValueWeights,
)
from methodtrail.workspace import WorkspaceManager


def test_agent_context_is_bounded_but_keeps_directed_graph_identity() -> None:
    state = {
        "task_id": "demo",
        "iteration": 9,
        "remaining_seconds": 300,
        "incumbent_variant_id": "outcome-8",
        "graph_context": [
            {
                "node_id": f"node-{index}",
                "node_type": "outcome",
                "title": f"method {index}",
                "parent_variant_id": f"node-{index - 1}",
                "relation": "deepen",
                "metric": index / 100,
                "change_logic": "x" * 2000,
                "edge_trace": [{"relation": "lineage", "reason": "edge"}],
            }
            for index in range(30)
        ],
        "method_pool": [
            {
                "node_id": f"proposal-{index}",
                "node_type": "proposal",
                "title": f"proposal {index}",
                "change_logic": "y" * 3000,
            }
            for index in range(100)
        ],
        "memory_context": [],
    }
    rendered = _context("select one candidate", context_limit=12_000, state=state)
    assert len(rendered) <= 12_500
    assert '"node_id":"node-29"' in rendered
    assert '"parent_variant_id":"node-28"' in rendered
    assert "truncated" in rendered


def test_candidate_context_preserves_original_indices() -> None:
    candidates = [
        {
            "index": index,
            "priority": 1.0 - index / 100,
            "graph_signal": {"feasible": True, "required_seconds": 20},
            "candidate": {
                "title": f"candidate {index}",
                "relation": "combine",
                "method": {"family": "fusion", "changed_factors": ["features"]},
                "rationale": "z" * 5000,
            },
        }
        for index in range(30)
    ]
    rendered = _context("choose", context_limit=20_000, ranked_candidates=candidates)
    for index in range(30):
        assert f'"index":{index}' in rendered


def test_repo_map_reads_workspace_under_parent_runs_directory(tmp_path: Path) -> None:
    """A candidate workspace may itself be nested below a directory named runs."""

    workspace = tmp_path / "runs" / "project" / "wt" / "candidate"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text(
        "def predict():\n    return 1\n", encoding="utf-8"
    )
    repo = RepoMap(workspace)
    context = repo.retrieve("repair solution", full_paths={"solution.py"})
    assert context and context[0]["path"] == "solution.py"
    assert "def predict" in context[0]["content"]


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


def test_path_graph_marks_candidates_that_cannot_fit_tail_budget(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    candidate = CandidatePath(
        variant_id="late",
        title="late candidate",
        relation="explore",
        expected_gain=0.1,
        information_gain=0.5,
        estimated_seconds=20,
        failure_risk=0.1,
    )
    signal = graph.selection_signal(candidate, remaining_seconds=25, reserve_seconds=10)
    assert signal["feasible"] is False
    assert signal["required_seconds"] == 30
    assert graph.rank([candidate], 25, ValueWeights(), reserve_seconds=10)[0][1] == float("-inf")


def test_initial_candidate_coverage_requires_a_composition_challenger() -> None:
    word = ChangeRequestArtifact(
        title="word representation",
        mutation_class=MutationClass.IMPLEMENTATION,
        research_question="compare representations",
        rationale="word features",
        method=MethodDescriptor(family="word", components={"features": "word ngrams"}),
    )
    character = word.model_copy(
        update={
            "title": "character representation",
            "method": MethodDescriptor(
                family="character", components={"features": "character ngrams"}
            ),
        }
    )
    gap = candidate_coverage_gap([word, character], initial=True)
    assert gap and "composition" in gap
    combined = word.model_copy(
        update={
            "title": "combined representations",
            "relation": "combine",
            "mutation_class": MutationClass.COMPOSITION,
            "method": MethodDescriptor(
                family="word_character_fusion",
                components={"word": "word ngrams", "character": "character ngrams"},
                changed_factors=["feature fusion"],
            ),
        }
    )
    assert candidate_coverage_gap([word, character, combined], initial=True) is None
    same_family_components = [
        word.model_copy(
            update={
                "method": MethodDescriptor(
                    family="tfidf",
                    components={"view": "word"},
                )
            }
        ),
        word.model_copy(
            update={
                "method": MethodDescriptor(
                    family="tfidf",
                    components={"view": "character"},
                )
            }
        ),
    ]
    assert candidate_coverage_gap(same_family_components, initial=True)


def test_composition_candidate_is_not_penalized_before_positive_history(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    candidate = CandidatePath(
        variant_id="fusion",
        title="combined features",
        relation="combine",
        expected_gain=0.05,
        information_gain=0.8,
        estimated_seconds=30,
        failure_risk=0.2,
        method=MethodDescriptor(
            family="feature_fusion",
            components={"a": "one", "b": "two"},
            changed_factors=["feature fusion"],
        ),
    )
    signal = graph.selection_signal(candidate, 300)
    assert signal["composition_penalty"] is False


def test_late_candidate_coverage_breaks_a_repeating_non_improving_factor() -> None:
    local = ChangeRequestArtifact(
        title="local calibration tweak",
        mutation_class=MutationClass.CONFIGURATION,
        research_question="improve probabilities",
        rationale="adjust one local factor",
        method=MethodDescriptor(
            family="calibration", changed_factors=["temperature"]
        ),
    )
    gap = candidate_coverage_gap(
        [local],
        initial=False,
        recent_outcomes=[
            {
                "improved": False,
                "method": {"family": "calibration", "changed_factors": ["temperature"]},
            },
            {
                "improved": False,
                "method": {"family": "calibration", "changed_factors": ["temperature"]},
            },
        ],
    )
    assert gap and "orthogonal" in gap


def test_repeated_composition_does_not_mask_coverage_gap() -> None:
    repeated = ChangeRequestArtifact(
        title="another blend-weight repeat",
        mutation_class=MutationClass.COMPOSITION,
        relation="combine",
        research_question="does another blend weight help?",
        rationale="repeat the same composition family",
        method=MethodDescriptor(
            family="blend", changed_factors=["composition_weight"]
        ),
    )
    recent = [
        {
            "improved": False,
            "method": {
                "family": "blend",
                "changed_factors": ["composition_weight"],
            },
        }
        for _ in range(2)
    ]
    assert candidate_coverage_gap(
        [repeated], initial=False, recent_outcomes=recent
    )


def test_search_policy_changes_value_weights() -> None:
    breadth = ValueWeights.for_search_policy("breadth", 1, 3)
    balanced_late = ValueWeights.for_search_policy("balanced", 5, 3)
    depth = ValueWeights.for_search_policy("depth", 1, 3)
    assert breadth.beta > breadth.alpha
    assert balanced_late.alpha > balanced_late.beta
    assert depth.alpha > depth.beta


def test_portfolio_respects_minimization_metric(tmp_path: Path) -> None:
    from methodtrail.portfolio import PortfolioManager, VariantProfile

    portfolio = PortfolioManager(tmp_path / "portfolio.json")
    low = VariantProfile(
        variant_id="low",
        metric=0.2,
        wall_seconds=10,
        reliability=1.0,
        information_gain=0.5,
        failure_risk=0.1,
        signature="low",
    )
    high = low.model_copy(update={"variant_id": "high", "metric": 0.8})
    portfolio.profiles = {"low": low, "high": high}
    assert [item.variant_id for item in portfolio.pareto(maximize_metric=False)] == ["low"]


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
    ancestor = next(row for row in context if row["node_id"] == "root")
    assert ancestor["traversal"] == "backtrack_ancestor"
    assert ancestor["edge_trace"][0]["target"] == "active"
    assert ancestor["edge_trace"][0]["relation"] == "deepen"


def test_path_graph_preserves_agent_relation_and_flags_component_mismatch(tmp_path: Path) -> None:
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
    assert normalized.relation == "explore"
    edge = graph.graph.edges["parent", proposal_id]
    assert edge["relation"] == "explore"
    assert edge["edge_type"] == "candidate"
    assert edge["target_change"]["change_logic"] == "one controlled change"

    standalone = ChangeRequestArtifact(
        title="word-only control",
        mutation_class=MutationClass.IMPLEMENTATION,
        relation="explore",
        research_question="what is the word-only control?",
        rationale="remove the character view for an attributable ablation",
        method=MethodDescriptor(
            family="word", components={"features": "word"}, changed_factors=["word view"]
        ),
    )
    graph.graph.nodes["parent"]["method"] = {
        "family": "fusion",
        "components": {"word": "word", "character": "character"},
    }
    normalized_standalone = graph.attach_proposals(
        "parent", [standalone], iteration=3
    )[0][1]
    assert normalized_standalone.relation == "explore"
    standalone_id = next(
        node_id
        for node_id, data in graph.graph.nodes(data=True)
        if data.get("title") == "word-only control"
    )
    assert graph.graph.nodes[standalone_id]["relation_warning"]

    explicit_ablation = standalone.model_copy(
        update={
            "relation": "ablate",
            "method": standalone.method.model_copy(
                update={"changed_factors": ["word view", "declared ablation"]}
            ),
        }
    )
    explicit_id, explicit = graph.attach_proposals(
        "parent", [explicit_ablation], iteration=4
    )[0]
    assert explicit.relation == "ablate"
    assert graph.graph.nodes[explicit_id]["relation"] == "ablate"


def test_path_graph_deduplicates_methods_and_links_related_nodes(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    first = ChangeRequestArtifact(
        title="first depth change",
        mutation_class=MutationClass.CONFIGURATION,
        relation="explore",
        research_question="does depth help?",
        rationale="compare depth",
        method=MethodDescriptor(
            family="tree", components={"depth": "6"}, changed_factors=["depth"]
        ),
    )
    attached = graph.attach_proposals(None, [first], iteration=1)
    assert len(attached) == 1
    proposal_id = attached[0][0]
    root_edges = [
        data
        for source, target, data in graph.graph.edges(data=True)
        if target == proposal_id and data.get("edge_type") == "root"
    ]
    assert len(root_edges) == 1
    assert root_edges[0]["relation"] == "explore"
    assert graph.attach_proposals(None, [first], iteration=2) == []
    graph.add_node(
        PathNode(
            variant_id="measured",
            iteration=1,
            title="measured depth change",
            relation="deepen",
            mutation_class=MutationClass.CONFIGURATION,
            question="does depth help?",
            evidence_summary="measured",
            status="adopted",
            method=first.method,
        )
    )
    related = [
        data
        for _, _, data in graph.graph.edges(data=True)
        if data.get("edge_type") == "related"
    ]
    assert related


def test_path_graph_records_multiple_evidence_parents_separately_from_code_parent(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    for node_id in ("evidence-a", "evidence-b"):
        graph.add_node(
            PathNode(
                variant_id=node_id,
                title=node_id,
                mutation_class=MutationClass.IMPLEMENTATION,
                question="prior result",
                evidence_summary="measured",
                status="adopted",
                method=MethodDescriptor(family=node_id, components={"x": node_id}),
            )
        )
    change = ChangeRequestArtifact(
        title="multi-source proposal",
        mutation_class=MutationClass.COMPOSITION,
        relation="combine",
        research_question="combine both findings",
        evidence_parent_ids=["evidence-a", "evidence-b"],
        rationale="use both measured results",
        method=MethodDescriptor(family="combined", components={"x": "both"}),
    )
    attached = graph.attach_proposals(None, [change], iteration=3)
    proposal_id = attached[0][0]
    evidence_edges = [
        data
        for source, target, data in graph.graph.edges(data=True)
        if target == proposal_id and data.get("edge_type") == "evidence"
    ]
    assert {data["relation"] for data in evidence_edges} == {"informed_by"}
    assert len(evidence_edges) == 2
    assert graph.graph.nodes[proposal_id]["change_request"]["evidence_parent_ids"] == [
        "evidence-a",
        "evidence-b",
    ]


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


def test_path_graph_prefers_unmeasured_family_after_repeated_regressions(tmp_path: Path) -> None:
    graph = ExperimentPathGraph(tmp_path / "graph.json")
    graph.add_node(
        PathNode(
            variant_id="base",
            title="base",
            mutation_class=MutationClass.IMPLEMENTATION,
            question="base",
            evidence_summary="base",
            metric=0.3,
            status="adopted",
            method=MethodDescriptor(family="linear", components={"x": "base"}),
        )
    )
    for index, metric in enumerate([0.4, 0.5], 1):
        graph.add_node(
            PathNode(
                variant_id=f"bad-{index}",
                title=f"bad {index}",
                parent_variant_id="base",
                relation="deepen",
                mutation_class=MutationClass.CONFIGURATION,
                question="linear change",
                evidence_summary="regression",
                metric=metric,
                status="needs_evidence",
                method=MethodDescriptor(
                    family="linear", components={"x": str(index)}, changed_factors=["x"]
                ),
            )
        )
    repeated = CandidatePath(
        variant_id="repeat",
        title="another linear change",
        relation="explore",
        expected_gain=0.05,
        information_gain=0.5,
        estimated_seconds=30,
        failure_risk=0.1,
        method=MethodDescriptor(family="linear", components={"x": "3"}, changed_factors=["x"]),
    )
    fresh = repeated.model_copy(
        update={
            "variant_id": "fresh",
            "title": "new tree family",
            "method": MethodDescriptor(family="tree", components={"depth": "4"}, changed_factors=["depth"]),
        }
    )
    ranked = graph.rank([repeated, fresh], 300, ValueWeights())
    assert ranked[0][0].variant_id == "fresh"


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


def test_workspace_rejects_full_rewrite_for_child_or_repair_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('parent')\n", encoding="utf-8")
    contract = TaskContract(
        task_id="rewrite-guard-toy",
        description="toy",
        workspace_template=str(workspace),
        allowed_data_paths=[],
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
        editable_paths=["solution.py"],
    )
    change = ChangeRequestArtifact(
        title="child rewrite",
        mutation_class=MutationClass.IMPLEMENTATION,
        parent_variant_id="parent-variant",
        research_question="change one local behavior",
        allowed_files=["solution.py"],
        rationale="test the child rewrite guard",
    )
    plan = CodePlanArtifact(
        summary="invalid whole-file rewrite",
        expected_test="never runs",
        edits=[
            FileEdit(
                operation="rewrite",
                path="solution.py",
                new_text="print('rewritten')\n",
                purpose="should be rejected",
            )
        ],
    )
    try:
        WorkspaceManager(tmp_path / "runs").apply_plan(workspace, plan, contract, change)
    except ValueError as exc:
        assert "local patch" in str(exc)
    else:
        raise AssertionError("child full rewrite should have been rejected")


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


class DeadlineDuringMemoryFakeLLM(FakeLLM):
    """Simulates the shared budget expiring during post-run memory writing."""

    def __init__(self) -> None:
        self.assessment_calls = 0

    def set_deadline(self, deadline: float | None) -> None:
        self.deadline = deadline

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is AssessmentArtifact:
            self.assessment_calls += 1
            if self.assessment_calls == 2:
                raise LLMDeadlineExceeded("research budget exhausted during memory write")
        return super().complete(system, user, response_model)


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


class SemanticRetryFakeLLM(FakeLLM):
    """Keeps repairing through more than the legacy three-review threshold."""

    supports_implementation_review = True

    def __init__(self) -> None:
        self.code_plan_calls = 0
        self.review_calls = 0
        self.seen_markers: list[int] = []

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is ImplementationReviewArtifact:
            self.review_calls += 1
            passed = self.review_calls >= 5
            return ImplementationReviewArtifact(
                passed=passed,
                summary="review passed" if passed else "repairable review issue",
                checks=["the current source is inspected"],
                issues=[] if passed else ["fix the current implementation locally"],
            )
        if response_model is RecoveryArtifact:
            return RecoveryArtifact(
                failure_class="implementation_review",
                diagnosis="the current source has one repairable issue",
                repair_directions=["apply the next local marker update"],
                preserve_question=True,
                return_to_reflection_reason="continue repairing the same candidate",
            )
        if response_model is CodePlanArtifact:
            self.code_plan_calls += 1
            if self.code_plan_calls > 1:
                expected_marker = self.code_plan_calls - 2
                assert f"# marker {expected_marker}" in user
                self.seen_markers.append(expected_marker)
            if self.code_plan_calls == 1:
                edit = FileEdit(
                    operation="create",
                    path="solution.py",
                    purpose="initial valid toy implementation",
                    new_text=(
                        "# marker 0\n"
                        "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n"
                    ),
                )
            else:
                old_marker = self.code_plan_calls - 2
                new_marker = self.code_plan_calls - 1
                edit = FileEdit(
                    operation="replace",
                    path="solution.py",
                    old_text=f"# marker {old_marker}\n",
                    purpose="apply the next local repair",
                    new_text=f"# marker {new_marker}\n",
                )
            return CodePlanArtifact(
                summary="initial plan or local repair",
                expected_test="prediction file exists",
                edits=[edit],
            )
        return super().complete(system, user, response_model)


class ReplanReviewFakeLLM(FakeLLM):
    """The review identifies a candidate contract that source cannot satisfy."""

    supports_implementation_review = True

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is ImplementationReviewArtifact:
            return ImplementationReviewArtifact(
                passed=False,
                decision="replan",
                replan_required=True,
                summary="the candidate asks source code to report a private score",
                issues=["move the private-score comparison to the controller"],
                candidate_adjustments=[
                    "remove the private-score output invariant",
                    "retain only prediction validity and public validation checks",
                ],
            )
        return super().complete(system, user, response_model)


class ReplanThenPassFakeLLM(FakeLLM):
    supports_implementation_review = True

    def __init__(self) -> None:
        self.proposal_calls = 0
        self.review_calls = 0

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is CandidateProposalArtifact:
            self.proposal_calls += 1
            if self.proposal_calls == 2:
                assert "remove the private-score output invariant" in user
            required_invariant = (
                "solution code reports the private score"
                if self.proposal_calls == 1
                else "solution code writes valid predictions"
            )
            return CandidateProposalArtifact(
                candidates=[
                    ChangeRequestArtifact(
                        title="write toy solution",
                        mutation_class=MutationClass.IMPLEMENTATION,
                        research_question="can a generated program complete the task?",
                        allowed_files=["solution.py"],
                        required_invariants=[required_invariant],
                        expected_gain=0.1,
                        information_gain=0.5,
                        estimated_seconds=10,
                        failure_risk=0.1,
                        rationale="establishes a runnable baseline",
                        method=MethodDescriptor(
                            family="toy",
                            components={
                                "predictor": (
                                    "constant with private-score reporting"
                                    if self.proposal_calls == 1
                                    else "constant with prediction-only output"
                                )
                            },
                            changed_factors=["implementation"],
                        ),
                    )
                ]
            )
        if response_model is ImplementationReviewArtifact:
            self.review_calls += 1
            if self.review_calls == 1:
                return ImplementationReviewArtifact(
                    passed=False,
                    decision="replan",
                    replan_required=True,
                    summary="private score is unavailable to solution code",
                    candidate_adjustments=["remove the private-score output invariant"],
                )
            return ImplementationReviewArtifact(
                passed=True, decision="pass", summary="candidate is executable"
            )
        return super().complete(system, user, response_model)


class AssessmentFailureFakeLLM(FakeLLM):
    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is AssessmentArtifact:
            raise RuntimeError("assessment provider is unavailable")
        return super().complete(system, user, response_model)


class PreflightRepairingFakeLLM(FakeLLM):
    """The first plan targets an existing file with create; repair must recover it."""

    def __init__(self) -> None:
        self.code_plan_calls = 0

    def complete(self, system: str, user: str, response_model):  # type: ignore[no-untyped-def]
        if response_model is RecoveryArtifact:
            return RecoveryArtifact(
                failure_class="edit",
                diagnosis="the requested file operation does not match the current workspace",
                repair_directions=["replace the existing entrypoint locally"],
                preserve_question=True,
                return_to_reflection_reason="repair the current implementation",
            )
        if response_model is CodePlanArtifact:
            self.code_plan_calls += 1
            if self.code_plan_calls == 1:
                edit = FileEdit(
                    operation="create",
                    path="solution.py",
                    purpose="intentionally exercises preflight repair",
                    new_text="open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n",
                )
            else:
                old = "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n0\\n')\n"
                edit = FileEdit(
                    operation="replace",
                    path="solution.py",
                    old_text=old,
                    purpose="repair the existing entrypoint",
                    new_text="open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n1\\n')\n",
                )
            return CodePlanArtifact(
                summary="initial edit or preflight repair",
                expected_test="prediction file exists",
                edits=[edit],
            )
        return super().complete(system, user, response_model)


def test_preflight_repair_does_not_consume_research_round(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "solution.py").write_text(
        "open('predictions.csv', 'w', encoding='utf-8').write('prediction\\n0\\n')\n",
        encoding="utf-8",
    )
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="preflight-repair-toy",
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
    llm = PreflightRepairingFakeLLM()
    trail = MethodTrail(tmp_path / "project", llm)
    results = trail.run_research(contract, total_seconds=120, max_iterations=1)
    assert len(results) == 1
    assert results[0].completed_research
    assert results[0].run is not None and results[0].run.metric == 0.8
    assert llm.code_plan_calls == 2
    bug_paths = list((tmp_path / "project").rglob("bugs.jsonl"))
    assert bug_paths
    assert len(BugMemory(bug_paths[0].parent).read()) >= 1
    card_paths = list((tmp_path / "project").rglob("cards.jsonl"))
    assert card_paths and card_paths[0].read_text(encoding="utf-8").strip()


def test_semantic_review_retries_until_recovery_succeeds(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="semantic-retry-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        solution_entrypoint="solution.py",
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        timeout_seconds=30,
        max_semantic_review_retries=3,
        protected_paths=["evaluate.py", "metrics.json"],
    )
    llm = SemanticRetryFakeLLM()
    trail = MethodTrail(tmp_path / "project", llm)
    result = trail.run_iteration(contract, remaining_seconds=120)
    assert result.completed_research
    assert result.run is not None and result.run.metric == 0.8
    assert llm.review_calls == 5
    assert llm.code_plan_calls == 5
    assert llm.seen_markers == [0, 1, 2, 3]


def test_review_replans_unsatisfiable_candidate_without_repair_loop(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="replan-review-toy",
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
    result = MethodTrail(tmp_path / "project", ReplanReviewFakeLLM()).run_iteration(
        contract, remaining_seconds=120
    )
    assert not result.completed_research
    assert result.assessment is None
    assert result.resume_mode == "replan"
    assert result.run is not None and result.run.return_code == -8


def test_replanned_candidate_runs_in_same_research_round(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="replanned-candidate-toy",
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
    llm = ReplanThenPassFakeLLM()
    trail = MethodTrail(tmp_path / "project", llm)
    results = trail.run_research(contract, total_seconds=120, max_iterations=1)
    assert len(results) == 2
    assert [result.completed_research for result in results] == [False, True]
    assert [result.iteration for result in results] == [1, 1]
    assert results[-1].run is not None and results[-1].run.metric == 0.8
    assert llm.proposal_calls == llm.review_calls == 2
    assert sorted(
        node.get("status")
        for _, node in trail.graph.graph.nodes(data=True)
        if node.get("node_type") == "proposal"
    ) == ["adopted_outcome", "replan_rejected"]
    revision_edges = [
        (source, target)
        for source, target, edge in trail.graph.graph.edges(data=True)
        if edge.get("edge_type") == "replan"
    ]
    assert len(revision_edges) == 1
    assert trail.graph.graph.nodes[revision_edges[0][0]]["status"] == "replan_rejected"
    assert trail.graph.graph.nodes[revision_edges[0][1]]["status"] == "adopted_outcome"


def test_repeated_unsatisfiable_candidate_stops_without_loop(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    contract = TaskContract(
        task_id="replan-repeat-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        run_command=[sys.executable, "solution.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
    )
    trail = MethodTrail(tmp_path / "project", ReplanReviewFakeLLM())
    results = trail.run_research(contract, total_seconds=120, max_iterations=1)
    assert len(results) == 1
    assert not results[0].completed_research
    assert trail.project is not None and trail.session is not None
    stop_events = [
        event
        for event in trail.projects.events(trail.project, trail.session.session_id)
        if event["kind"] == "research_stop"
    ]
    assert len(stop_events) == 1
    assert "no new or revisitable executable node" in stop_events[0]["payload"]["reason"]


def test_valid_metric_survives_assessment_provider_failure(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="assessment-outage-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        run_command=[sys.executable, "solution.py"],
        evaluation_command=[sys.executable, "evaluate.py"],
        metric_name="score",
        required_outputs=["predictions.csv"],
        protected_paths=["evaluate.py", "metrics.json"],
    )
    trail = MethodTrail(tmp_path / "project", AssessmentFailureFakeLLM())
    try:
        trail.run_iteration(contract, remaining_seconds=120)
    except RuntimeError as error:
        assert "assessment provider" in str(error)
    else:
        raise AssertionError("assessment provider failure should remain visible")
    assert trail.project is not None and trail.session is not None
    assert trail.project.incumbent_metric == 0.8
    assert trail.session.status == "interrupted"
    assert any(
        node.get("metric") == 0.8
        for _, node in trail.graph.graph.nodes(data=True)
        if node.get("node_type") == "outcome"
    )
    assert any(
        node.get("metric") == 0.8
        for node in trail.experiment_memory.graph.nodes.values()
    )


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


def test_budget_deadline_stops_cleanly_after_valid_run(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "evaluate.py").write_text(
        "import json\njson.dump({'score': 0.8}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    contract = TaskContract(
        task_id="budget-stop-toy",
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
    trail = MethodTrail(tmp_path / "project", DeadlineDuringMemoryFakeLLM())
    results = trail.run_research(contract, total_seconds=120, max_iterations=20)

    assert results == []
    assert trail.session is not None and trail.session.status == "completed"
    assert trail.project is not None and trail.project.incumbent_metric == 0.8
    candidates = trail.projects.session_candidates(trail.project, trail.session.session_id)
    assert candidates and candidates[0].metric == 0.8
    events = trail.projects.events(trail.project, trail.session.session_id, limit=100)
    assert any(event["kind"] == "budget_interruption" for event in events)
    assert any(event["kind"] == "budget_stop" for event in events)
    dashboard = (
        trail.projects.project_path(trail.project)
        / "sessions"
        / trail.session.session_id
        / "dashboard.html"
    )
    assert dashboard.exists()


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
    assert "independent evaluator" in public_context["metric_boundary"]
    assert "unavailable to solution code" in public_context["metric_boundary"]
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


def test_memory_graph_links_related_cards_and_exposes_profile(tmp_path: Path) -> None:
    memory = ExperimentMemory(tmp_path / "memory")
    memory.add(
        MemoryCard(
            task_id="task",
            session_id="session",
            variant_id="v1",
            question="word features",
            conclusion="word reference",
            relation="explore",
            decision="evidence",
            tags=["text"],
            method_family="sparse",
            changed_factors=["word"],
        )
    )
    memory.add(
        MemoryCard(
            task_id="task",
            session_id="session",
            variant_id="v2",
            parent_variant_id="v1",
            question="character features",
            conclusion="combination helps",
            relation="combine",
            decision="adopt",
            tags=["text"],
            method_family="sparse",
            changed_factors=["character"],
        )
    )
    profile = memory.graph_profile("task")
    assert profile["card_count"] == 2
    assert profile["linked_card_count"] >= 1
    edges = memory.graph.edges
    v1_card = next(card_id for card_id, node in memory.graph.nodes.items() if node["variant_id"] == "v1")
    v2_card = next(card_id for card_id, node in memory.graph.nodes.items() if node["variant_id"] == "v2")
    assert any(
        edge["source"] == v1_card and edge["target"] == v2_card and edge["relation"] == "follows"
        for edge in edges
    )
    assert all(edge["source"] != edge["target"] for edge in edges)
    results = memory.search("task", "word reference")
    assert results and results[0]["variant_id"] == "v1"
    connected = [row for row in results if row.get("retrieval_source") == "memory_graph"]
    assert connected
    assert connected[0]["memory_graph_trace"]["edges"][0]["relation"] == "follows"


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
