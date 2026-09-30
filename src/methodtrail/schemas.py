"""Typed artifacts exchanged by the research agent, coding agent, and executor."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class MutationClass(StrEnum):
    """The allowed scope of one experiment's change."""

    CONFIGURATION = "configuration"
    COMPOSITION = "composition"
    IMPLEMENTATION = "implementation"
    RECOVERY = "recovery"


class MethodDescriptor(BaseModel):
    """A compact description of what an experiment changes.

    It lets the path graph compare methods by their declared ingredients instead
    of treating every experiment title as an unrelated text string.
    """

    family: str = "unspecified"
    components: dict[str, str] = Field(default_factory=dict)
    changed_factors: list[str] = Field(default_factory=list)
    target_scope: str = "global"


class TaskContract(BaseModel):
    """What a task adapter may provide without supplying a predictor."""

    task_id: str
    description: str
    workspace_template: str
    allowed_data_paths: list[str]
    solution_entrypoint: str = "solution.py"
    smoke_command: list[str] = Field(default_factory=list)
    test_command: list[str] = Field(default_factory=list)
    run_command: list[str]
    evaluation_command: list[str] = Field(default_factory=list)
    private_evaluator_dir: str | None = None
    private_evaluation_command: list[str] = Field(default_factory=list)
    execution_isolation: Literal["process", "container"] = "process"
    metric_file: str = "metrics.json"
    required_outputs: list[str] = Field(default_factory=list)
    metric_name: str
    auxiliary_metric_names: list[str] = Field(default_factory=list)
    metric_constraints: dict[str, float] = Field(default_factory=dict)
    maximize_metric: bool = True
    allow_self_reported_metric: bool = False
    timeout_seconds: int = Field(default=1800, ge=1)
    # Keep a small tail of the research budget for final output, trajectory
    # writing and a last repair decision.  A candidate must fit before this
    # reserve is consumed; it is not counted as a research round.
    finalization_reserve_seconds: int = Field(default=60, ge=0)
    # Repair is an internal technical loop.  It must not be confused with the
    # number of research rounds.  The large safety cap prevents a broken
    # adapter from spinning forever; RecoveryAgent can stop much earlier.
    max_repair_steps: int = Field(default=100, ge=1, le=1000)
    # Kept for contract compatibility. Semantic review failures no longer
    # auto-abandon a candidate; RecoveryAgent must explicitly choose that
    # action. The hard repair-step and shared-time limits remain safety guards.
    max_semantic_review_retries: int = Field(default=3, ge=1, le=10)
    # Kept only so older task contracts can still be loaded.  Runtime code does
    # not use it as the research-round budget.
    max_repair_attempts: int | None = Field(default=None, ge=0, le=1000)
    minimum_iterations: int = Field(default=1, ge=1)
    # Deprecated compatibility fields.  Current runtime selection is derived
    # from the method graph and measured evidence; these values are ignored.
    search_policy: Literal["breadth", "balanced", "depth"] = "balanced"
    exploration_rounds: int = Field(default=0, ge=0, le=50)
    allowed_dependencies: list[str] = Field(default_factory=list)
    editable_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=lambda: ["task_contract.json"])

    @field_validator("run_command")
    @classmethod
    def command_is_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("run_command must contain at least one command token")
        return value

    def llm_context(self) -> dict[str, Any]:
        """Task facts the research roles may see; never disclose private paths."""

        return self.model_dump(
            exclude={
                "private_evaluator_dir",
                "private_evaluation_command",
                "max_repair_steps",
                "max_semantic_review_retries",
                "max_repair_attempts",
                "search_policy",
                "exploration_rounds",
            }
        )


class ResearchState(BaseModel):
    task_id: str
    iteration: int
    remaining_seconds: int
    best_metric: float | None = None
    incumbent_metric: float | None = None
    incumbent_variant_id: str | None = None
    research_round: int = 0
    repair_step: int = 0
    recent_facts: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    path_hints: list[dict[str, Any]] = Field(default_factory=list)
    portfolio_context: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: list[dict[str, Any]] = Field(default_factory=list)
    method_pool: list[dict[str, Any]] = Field(default_factory=list)
    memory_graph_context: dict[str, Any] = Field(default_factory=dict)


class HypothesisArtifact(BaseModel):
    observation: str
    hypothesis: str
    evidence_needed: list[str]
    comparison_plan: str
    adoption_condition: str
    fallback_condition: str


class ChangeRequestArtifact(BaseModel):
    title: str
    mutation_class: MutationClass
    relation: Literal["deepen", "ablate", "combine", "explore", "recover"] = "explore"
    relation_warning: str | None = None
    research_question: str
    parent_variant_id: str | None = None
    # The code parent says which workspace/version is edited.  These optional
    # IDs say which measured outcomes informed the research decision; there
    # may be more than one and they do not imply code inheritance.
    evidence_parent_ids: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    required_invariants: list[str] = Field(default_factory=list)
    expected_gain: float = Field(default=0.0, ge=0.0)
    information_gain: float = Field(default=0.0, ge=0.0)
    estimated_seconds: int = Field(default=60, ge=1)
    failure_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str
    method: MethodDescriptor = Field(default_factory=MethodDescriptor)


class CandidateProposalArtifact(BaseModel):
    """A dynamically sized batch of executable method nodes.

    The system deliberately does not prescribe a fixed number of candidates.
    The LLM can stop discovery when the method graph has enough relevant
    alternatives for the current question.
    """

    candidates: list[ChangeRequestArtifact] = Field(min_length=1)
    discovery_complete: bool = False
    discovery_reason: str = ""


class CandidateSelectionArtifact(BaseModel):
    selected_index: int = Field(ge=0)
    reason: str


class FileEdit(BaseModel):
    """One constrained source edit proposed by the Coding Agent.

    Existing files default to one exact, unique snippet replacement.  A symbol
    replacement is available when a whole function/class must change, and an
    explicit rewrite is reserved for replacing the declared solution entrypoint.
    """

    operation: Literal["create", "replace", "replace_symbol", "rewrite"]
    path: str
    old_text: str | None = None
    symbol: str | None = None
    new_text: str
    purpose: str

    @model_validator(mode="after")
    def validate_edit(self) -> FileEdit:
        if self.operation == "create" and (self.old_text is not None or self.symbol):
            raise ValueError("create edits must not include old_text or symbol")
        if self.operation == "replace" and (not self.old_text or self.symbol):
            raise ValueError("replace edits require old_text and no symbol")
        if self.operation == "replace_symbol" and (not self.symbol or self.old_text):
            raise ValueError("replace_symbol edits require symbol and no old_text")
        if self.operation == "rewrite" and (self.old_text or self.symbol):
            raise ValueError("rewrite edits replace the complete entrypoint")
        return self


class CodePlanArtifact(BaseModel):
    summary: str
    affected_interfaces: list[str] = Field(default_factory=list)
    invariant_checks: list[str] = Field(default_factory=list)
    expected_test: str
    edits: list[FileEdit] = Field(min_length=1)


class ImplementationReviewArtifact(BaseModel):
    """A pre-execution review of the realized code change.

    The review is deliberately separate from the research assessment.  It
    checks whether the edit actually implements the selected question and
    preserves the declared comparison target; it does not judge the metric.
    """

    passed: bool
    scope_ok: bool = True
    invariants_ok: bool = True
    summary: str
    checks: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    passed: bool
    checks: list[dict[str, Any]]


class RunArtifact(BaseModel):
    command: list[str]
    return_code: int
    timed_out: bool
    wall_seconds: float
    stdout: str
    stderr: str
    evaluation_return_code: int | None = None
    evaluation_stdout: str = ""
    evaluation_stderr: str = ""
    smoke_return_code: int | None = None
    test_return_code: int | None = None
    metric: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_constraints_passed: bool | None = None
    output_files: list[str] = Field(default_factory=list)


class AssessmentArtifact(BaseModel):
    decision: Literal["adopt", "evidence", "recovery", "defer", "stop"]
    reason: str
    reusable_conclusion: str | None = None
    applicable_conditions: list[str] = Field(default_factory=list)
    next_question: str | None = None


class RecoveryArtifact(BaseModel):
    failure_class: str
    diagnosis: str
    repair_directions: list[str]
    preserve_question: bool
    return_to_reflection_reason: str
    action: Literal["continue_repair", "switch_implementation", "abandon_candidate"] = "continue_repair"


class PathNode(BaseModel):
    variant_id: str
    iteration: int | None = None
    title: str
    parent_variant_id: str | None = None
    evidence_parent_ids: list[str] = Field(default_factory=list)
    relation: Literal["deepen", "ablate", "combine", "explore", "recover"] = "explore"
    relation_warning: str | None = None
    mutation_class: MutationClass
    question: str
    change_logic: str = ""
    evidence_summary: str
    applicable_conditions: list[str] = Field(default_factory=list)
    metric: float | None = None
    wall_seconds: float | None = None
    failure_risk: float = 0.0
    status: Literal[
        "adopted",
        "promising",
        "deferred",
        "needs_evidence",
        "failed",
        "abandoned",
    ]
    method: MethodDescriptor = Field(default_factory=MethodDescriptor)


class CandidatePath(BaseModel):
    variant_id: str
    title: str
    relation: Literal["deepen", "ablate", "combine", "explore", "recover"]
    expected_gain: float = Field(ge=0.0)
    information_gain: float = Field(ge=0.0)
    estimated_seconds: int = Field(ge=1)
    failure_risk: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    method: MethodDescriptor = Field(default_factory=MethodDescriptor)


class ValueWeights(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=1.0, ge=0.0)
    delta: float = Field(default=1.0, ge=0.0)

    @classmethod
    def for_stage(cls, remaining_fraction: float, has_incumbent: bool) -> ValueWeights:
        """Set transparent search priorities from experiment stage, not task names.

        Early rounds value information; once a reliable incumbent exists, direct
        improvement matters more; near the deadline, cost and failure risk rise.
        The interpolation is deterministic and recorded with every ranking.
        """
        fraction = min(1.0, max(0.0, remaining_fraction))
        if not has_incumbent:
            return cls(alpha=0.8, beta=1.2, gamma=0.9, delta=1.0)
        if fraction < 0.25:
            return cls(alpha=1.0, beta=0.8, gamma=1.3, delta=1.3)
        if fraction > 0.5:
            return cls(alpha=1.2, beta=1.0, gamma=1.0, delta=1.1)
        return cls(alpha=1.0, beta=1.0, gamma=1.1, delta=1.15)

    @classmethod
    def for_search_policy(
        cls, policy: str, iteration: int, exploration_rounds: int
    ) -> ValueWeights:
        """Choose how strongly the ranker values information versus direct gain.

        ``breadth`` spends early iterations distinguishing alternatives,
        ``depth`` favors improving a supported direction, and ``balanced``
        moves from the former to the latter after the configured exploration
        rounds.  These are planning weights, not learned model parameters.
        """

        if policy == "breadth":
            return cls(alpha=0.65, beta=1.55, gamma=0.85, delta=0.95)
        if policy == "depth":
            return cls(alpha=1.45, beta=0.75, gamma=1.05, delta=1.10)
        if iteration <= max(0, exploration_rounds):
            return cls(alpha=0.85, beta=1.25, gamma=0.90, delta=1.00)
        return cls(alpha=1.25, beta=0.95, gamma=1.00, delta=1.10)
