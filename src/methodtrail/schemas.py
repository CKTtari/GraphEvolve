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
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    minimum_iterations: int = Field(default=1, ge=1)
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
            exclude={"private_evaluator_dir", "private_evaluation_command"}
        )


class ResearchState(BaseModel):
    task_id: str
    iteration: int
    remaining_seconds: int
    best_metric: float | None = None
    recent_facts: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    path_hints: list[dict[str, Any]] = Field(default_factory=list)
    portfolio_context: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: list[dict[str, Any]] = Field(default_factory=list)


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
    research_question: str
    parent_variant_id: str | None = None
    allowed_files: list[str] = Field(default_factory=list)
    required_invariants: list[str] = Field(default_factory=list)
    expected_gain: float = Field(default=0.0, ge=0.0)
    information_gain: float = Field(default=0.0, ge=0.0)
    estimated_seconds: int = Field(default=60, ge=1)
    failure_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str
    method: MethodDescriptor = Field(default_factory=MethodDescriptor)


class CandidateProposalArtifact(BaseModel):
    """Several executable alternatives before the program applies path-value ranking."""

    candidates: list[ChangeRequestArtifact] = Field(min_length=1, max_length=5)


class CandidateSelectionArtifact(BaseModel):
    selected_index: int = Field(ge=0)
    reason: str


class FileEdit(BaseModel):
    """One constrained source edit proposed by the Coding Agent.

    Existing files can only be changed by replacing one exact, unique snippet.
    Full contents are reserved for creating a genuinely new file.
    """

    operation: Literal["create", "replace"]
    path: str
    old_text: str | None = None
    new_text: str
    purpose: str

    @model_validator(mode="after")
    def validate_edit(self) -> FileEdit:
        if self.operation == "create" and self.old_text is not None:
            raise ValueError("create edits must not include old_text")
        if self.operation == "replace" and not self.old_text:
            raise ValueError("replace edits require a non-empty old_text")
        return self


class CodePlanArtifact(BaseModel):
    summary: str
    affected_interfaces: list[str] = Field(default_factory=list)
    expected_test: str
    edits: list[FileEdit] = Field(min_length=1)


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


class PathNode(BaseModel):
    variant_id: str
    title: str
    parent_variant_id: str | None = None
    relation: Literal["deepen", "ablate", "combine", "explore", "recover"] = "explore"
    mutation_class: MutationClass
    question: str
    evidence_summary: str
    applicable_conditions: list[str] = Field(default_factory=list)
    metric: float | None = None
    wall_seconds: float | None = None
    failure_risk: float = 0.0
    status: Literal["adopted", "deferred", "needs_evidence", "failed"]
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
