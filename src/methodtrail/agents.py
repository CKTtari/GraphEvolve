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
        user = _context(
            """Refine the current observation into one research question. State what evidence would answer it,
how to compare alternatives, and what conditions lead to adoption or fallback.""",
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
    ) -> CandidateProposalArtifact:
        user = _context(
            f"""Propose exactly {contract.proposal_count} distinct next experiments. Each must be executable and tied to the current
hypothesis and one graph-expansion hint. Separate configuration-only changes from composition, implementation,
and recovery changes. For every candidate fill method.family, method.components, method.changed_factors, and
method.target_scope so the program can compare it with the active path. Give realistic expected gain, information
gain, runtime and failure-risk estimates. The program will attach these as proposed graph branches and rank them
before one is selected. Do not propose a disconnected change when an existing path can be deepened, ablated,
combined, or repaired. Follow the task's search_policy: breadth keeps distinct
families and controls alive; depth prefers a supported direction's composition
and tuning; balanced explores first and then deepens the strongest path.""",
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
            """Select one candidate index after reading the program-computed path priorities. The index is the
original candidate index in each supplied row; do not use the row's position after sorting. Explain why its
expected gain, information value, cost, and risk fit the current evidence and remaining budget. Do not select an
index outside the supplied list.""",
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
    ) -> CodePlanArtifact:
        user = _context(
            """Produce a minimal, executable edit plan for the selected experiment. For a new file use a create edit
with its full contents. For an existing file use a replace edit: old_text must be an exact, unique local snippet and
new_text is its replacement. Do not return an entire existing file. Do not edit protected or data files. Include the
required solution entrypoint and keep the change focused on the research question. State a test that should pass.
If the task has no separate configuration file, a configuration experiment may
change a small source-level constant; keep that edit local and explicit.""",
            contract=contract,
            change=change,
            repository_context=repo_context,
        )
        return self.llm.complete(SYSTEM, user, CodePlanArtifact)

    def repair(
        self,
        contract: TaskContract,
        change: ChangeRequestArtifact,
        run: RunArtifact,
        recovery: RecoveryArtifact,
        repo_context: list[dict[str, str]],
    ) -> CodePlanArtifact:
        user = _context(
            """Repair the implementation that just failed. Read the recorded stderr, diagnosis, and current
repository files. Return only minimal create or exact replace edits. For replace, old_text must be copied exactly
from the supplied file context and must identify one local fragment. Use create only when the file is absent; use
replace when it already exists. Do not rewrite an entire existing file.
Preserve the current research question unless the diagnosis says it cannot be tested. Do not edit protected or data
files and do not invent results. The revised files will be executed immediately without human intervention.""",
            contract=contract,
            change=change,
            run=run,
            recovery=recovery,
            repository_context=repo_context,
        )
        return self.llm.complete(SYSTEM, user, CodePlanArtifact)


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
        self, contract: TaskContract, state: ResearchState, run: RunArtifact
    ) -> RecoveryArtifact:
        user = _context(
            """Classify the technical failure. Suggest parameter repair, lower-cost implementation,
neighbor-path replacement, or a reason the research question must be revised. Do not pretend that a failed run
is evidence against the hypothesis.""",
            contract=contract,
            state=state,
            run=run,
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
            """Assess the measured run. Adopt only when the available evidence supports it and all declared
metric constraints pass. Use the auxiliary metrics to judge stability, cost, and output quality when present.
Choose 'evidence'
when the next step should isolate uncertainty, 'recovery' only for technical failure, 'defer' for a useful but
unadopted result, and 'stop' when remaining budget cannot support a useful next run.""",
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
