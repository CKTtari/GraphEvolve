"""Directed experiment-path memory and value-aware candidate choice."""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import networkx as nx

from .schemas import (
    CandidatePath,
    ChangeRequestArtifact,
    MethodDescriptor,
    PathNode,
    ValueWeights,
)

ROOT_NODE_ID = "__project_root__"


class ExperimentPathGraph:
    """Stores experiment evidence and the logical paths between method variants."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        if self.path.exists():
            self._load()
            self._enrich_edges()
            self._prune_related_edges()
        self._ensure_root_edges()

    def add_node(self, node: PathNode) -> None:
        data = node.model_dump(mode="json")
        data["node_type"] = "outcome"
        data["method_signature"] = _descriptor_signature(
            node.method,
            node.mutation_class.value,
            node.relation,
            node.question,
            node.title,
        )
        self.graph.add_node(node.variant_id, **data)
        if node.parent_variant_id:
            warning = node.relation_warning
            self.graph.add_edge(
                node.parent_variant_id,
                node.variant_id,
                relation=node.relation,
                status=node.status,
                edge_type="lineage",
                label=f"{node.relation}: {node.title}",
                reason=(
                    f"第{node.iteration or '?'}轮沿父版本继续；本轮修改："
                    f"{node.change_logic or node.title}"
                    + (f"；关系校验提示：{warning}" if warning else "")
                ),
                target_change=self._change_payload(node.variant_id),
            )
        else:
            self._ensure_root_node()
            self.graph.add_edge(
                ROOT_NODE_ID,
                node.variant_id,
                relation=node.relation,
                status=node.status,
                edge_type="root",
                label=f"root: {node.relation}",
                reason="该结果没有代码父版本，连接到项目根节点；具体研究关系由节点 relation 表示。",
                target_change=self._change_payload(node.variant_id),
            )
        for evidence_id in node.evidence_parent_ids:
            if evidence_id not in self.graph or self.graph.nodes[evidence_id].get("node_type") != "outcome":
                continue
            self.graph.add_edge(
                evidence_id,
                node.variant_id,
                relation="informed_by",
                status=node.status,
                edge_type="evidence",
                label="informed_by",
                reason="本版本的研究决策显式参考该已测量版本；代码父版本由 lineage 边表示。",
                target_change=self._change_payload(node.variant_id),
            )
        self._link_node_to_related(node.variant_id)
        self._save()

    def add_candidate_relation(
        self, parent_id: str, candidate_id: str, relation: str
    ) -> None:
        """Backward-compatible edge writer for callers that already have IDs."""
        self.graph.add_edge(
            parent_id,
            candidate_id,
            relation=relation,
            status="proposed",
            edge_type="candidate",
            label=relation,
            reason="从当前父版本扩展出的候选分支。",
            target_change=self._change_payload(candidate_id),
        )
        self._save()

    def attach_proposals(
        self,
        parent_variant_id: str | None,
        changes: list[ChangeRequestArtifact],
        iteration: int,
    ) -> list[tuple[str, ChangeRequestArtifact]]:
        """Put candidates into the graph before selection.

        The LLM describes a method with a small component map and owns the
        semantic relation. The graph records every candidate as a proposed
        branch, including branches not selected today, and adds a visible
        consistency warning when the component map disagrees with that label.
        """

        parent = self.graph.nodes.get(parent_variant_id, {}) if parent_variant_id else {}
        if not parent_variant_id:
            self._ensure_root_node()
        attached: list[tuple[str, ChangeRequestArtifact]] = []
        for index, original in enumerate(changes):
            relation = self._infer_relation(parent.get("method"), original)
            relation_warning = self._relation_warning(
                parent.get("method"), original.model_copy(update={"relation": relation})
            )
            change = original.model_copy(
                update={"relation": relation, "relation_warning": relation_warning}
            )
            proposal_id = f"proposal-{iteration}-{index}-{uuid.uuid4().hex[:8]}"
            signature = method_signature(change)
            rejected_sources = [
                node_id
                for node_id, data in self.graph.nodes(data=True)
                if data.get("node_type") == "proposal"
                and data.get("status") == "replan_rejected"
                and data.get("iteration") == iteration
                and (
                    data.get("method_signature") == signature
                    or data.get("question") == change.research_question
                )
            ]
            # A method node is global to the project.  Do not create a second
            # executable node for an identical method; the existing node stays
            # available as history and can be revisited when new evidence
            # changes its priority.
            if any(
                data.get("method_signature") == signature
                and data.get("node_type") in {"proposal", "outcome"}
                and (
                    data.get("status") != "replan_rejected"
                    or data.get("change_request", {}).get("required_invariants")
                    == change.required_invariants
                )
                for _, data in self.graph.nodes(data=True)
            ):
                continue
            self.graph.add_node(
                proposal_id,
                    node_type="proposal",
                    variant_id=proposal_id,
                    iteration=iteration,
                title=change.title,
                relation=relation,
                relation_warning=relation_warning,
                mutation_class=change.mutation_class.value,
                question=change.research_question,
                method=change.method.model_dump(mode="json"),
                expected_gain=change.expected_gain,
                information_gain=change.information_gain,
                estimated_seconds=change.estimated_seconds,
                failure_risk=change.failure_risk,
                status="proposed",
                method_signature=signature,
                change_request=change.model_dump(mode="json"),
            )
            if parent_variant_id:
                self.graph.add_edge(
                    parent_variant_id,
                    proposal_id,
                    relation=relation,
                    status="proposed",
                    edge_type="candidate",
                    label=f"{relation}: {change.title}",
                    reason=(
                        f"候选分支：{change.rationale}"
                        + (f"；关系校验提示：{relation_warning}" if relation_warning else "")
                    ),
                    target_change={
                        "iteration": iteration,
                        "title": change.title,
                        "mutation_class": change.mutation_class.value,
                        "change_logic": change.rationale,
                        "changed_factors": change.method.changed_factors,
                        "method_components": change.method.components,
                        "evidence_parent_ids": change.evidence_parent_ids,
                        "relation_warning": relation_warning,
                    },
                )
            else:
                self.graph.add_edge(
                    ROOT_NODE_ID,
                    proposal_id,
                    relation=relation,
                    status="proposed",
                    edge_type="root",
                    label=f"root: {relation}",
                    reason="首轮候选没有代码父版本，连接到项目根节点；候选之间的差异由各自 method 和 relation 表示。",
                    target_change=self._change_payload(proposal_id),
                )
            attached.append((proposal_id, change))
            for rejected_id in rejected_sources:
                rejected = self.graph.nodes[rejected_id]
                self.graph.add_edge(
                    rejected_id,
                    proposal_id,
                    relation="revises_candidate",
                    status="proposed",
                    edge_type="replan",
                    label="revises_candidate",
                    reason=str(rejected.get("replan_reason") or "候选约束经过审查后修订。"),
                    target_change=self._change_payload(proposal_id),
                )
            for evidence_id in change.evidence_parent_ids:
                if evidence_id not in self.graph:
                    continue
                evidence_data = self.graph.nodes[evidence_id]
                if evidence_data.get("node_type") != "outcome":
                    continue
                self.graph.add_edge(
                    evidence_id,
                    proposal_id,
                    relation="informed_by",
                    status="proposed",
                    edge_type="evidence",
                    label="informed_by",
                    reason="候选研究决策显式参考该已测量版本；它不是代码父版本。",
                    target_change=self._change_payload(proposal_id),
                )
        # Keep alternatives related even when they were proposed in different
        # rounds.  These links make the method pool a navigable graph: a new
        # candidate can be reached from a measured method through shared
        # factors, a common family, or an explicit alternative relation.
        self._link_related_proposals(attached)
        self._save()
        return attached

    def _link_related_proposals(
        self, attached: list[tuple[str, ChangeRequestArtifact]]
    ) -> None:
        entries = [
            (node_id, data)
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") in {"proposal", "outcome"}
        ]
        attached_ids = {node_id for node_id, _ in attached}
        for node_id, change in attached:
            factors = set(change.method.changed_factors)
            family = change.method.family
            ranked: list[tuple[float, str, str]] = []
            for other_id, other in entries:
                if other_id == node_id or other_id in attached_ids:
                    continue
                other_method = other.get("method") or {}
                other_factors = set(other_method.get("changed_factors", []))
                shared = factors.intersection(other_factors)
                same_family = family != "unspecified" and family == other_method.get("family")
                if not shared and not same_family:
                    continue
                relation = "same_family" if same_family else "shared_factor"
                weight = 0.75 if same_family else 0.45
                if shared:
                    weight = min(1.0, weight + 0.1 * len(shared))
                ranked.append((weight, other_id, relation))
            # A proposal records a few useful alternatives, not a complete
            # family clique.  The edge always points from existing evidence to
            # the newly proposed branch.
            for weight, other_id, relation in sorted(
                ranked, key=lambda item: item[0], reverse=True
            )[:2]:
                self._add_relation_edge(other_id, node_id, relation, weight)

    def _link_node_to_related(self, node_id: str) -> None:
        """Connect a newly measured node to older methods it can explain."""

        current = self.graph.nodes.get(node_id, {})
        current_method = current.get("method") or {}
        factors = set(current_method.get("changed_factors", []))
        family = current_method.get("family", "unspecified")
        ranked: list[tuple[float, str, str]] = []
        for other_id, other in list(self.graph.nodes(data=True)):
            if other_id == node_id or other.get("node_type") not in {"proposal", "outcome"}:
                continue
            other_method = other.get("method") or {}
            other_factors = set(other_method.get("changed_factors", []))
            shared = factors.intersection(other_factors)
            same_family = family != "unspecified" and family == other_method.get("family")
            if not shared and not same_family:
                continue
            relation = "same_family" if same_family else "shared_factor"
            weight = 0.75 if same_family else 0.45
            if shared:
                weight = min(1.0, weight + 0.1 * len(shared))
            ranked.append((weight, other_id, relation))
        # Keep the measured graph readable and make provenance meaningful:
        # related evidence flows from an older node into this new outcome.
        for weight, other_id, relation in sorted(
            ranked, key=lambda item: item[0], reverse=True
        )[:2]:
            self._add_relation_edge(other_id, node_id, relation, weight)

    def _add_relation_edge(
        self, source: str, target: str, relation: str, weight: float
    ) -> None:
        if source == target:
            return
        if self.graph.has_edge(source, target):
            edge = self.graph.edges[source, target]
            if edge.get("edge_type") in {"lineage", "candidate", "execution", "replan"}:
                return
            edge["relation"] = relation
            edge["weight"] = max(float(edge.get("weight", 0.0)), round(weight, 4))
            edge["edge_type"] = "related"
            edge.setdefault("label", relation)
            edge.setdefault("reason", self._relation_reason(relation))
            edge.setdefault("target_change", self._change_payload(target))
            return
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            status="proposed",
            weight=round(weight, 4),
            edge_type="related",
            label=relation,
            reason=self._relation_reason(relation),
            target_change=self._change_payload(target),
        )

    @staticmethod
    def _relation_reason(relation: str) -> str:
        return {
            "same_family": "同一方法族的前序证据，供当前分支比较。",
            "shared_factor": "共享改动因素的前序证据，供当前分支归因。",
        }.get(relation, "方法图中的相关证据关系。")

    def _change_payload(self, node_id: str) -> dict[str, Any]:
        data = self.graph.nodes.get(node_id, {})
        raw = data.get("change_request") or {}
        method = data.get("method") or raw.get("method") or {}
        return {
            "iteration": data.get("iteration", raw.get("iteration")),
            "title": data.get("title", raw.get("title", "")),
            "mutation_class": data.get("mutation_class", raw.get("mutation_class", "")),
            "change_logic": data.get("change_logic", raw.get("rationale", "")),
            "changed_factors": method.get("changed_factors", []),
            "method_components": method.get("components", {}),
            "evidence_parent_ids": raw.get("evidence_parent_ids", data.get("evidence_parent_ids", [])),
            "relation_warning": data.get("relation_warning", raw.get("relation_warning")),
        }

    def method_pool(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the project-wide method pool, including measured failures."""

        rows = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") not in {"proposal", "outcome"}:
                continue
            row = dict(data)
            row["node_id"] = node_id
            rows.append(row)
        return rows[-limit:]

    def recent_outcome_summary(
        self, limit: int = 3, maximize_metric: bool = True
    ) -> list[dict[str, Any]]:
        """Return ordered outcome metadata for generic exploration checks.

        The summary deliberately contains no task-specific feature names. It
        only says which declared factors were recently tried and whether each
        result improved its directed parent when that comparison is available.
        """

        if limit <= 0:
            return []
        rows = [
            (node_id, data)
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") == "outcome" and data.get("metric") is not None
        ]
        rows.sort(
            key=lambda item: (
                int(item[1].get("iteration") or 0),
                str(item[0]),
            )
        )
        summary: list[dict[str, Any]] = []
        for node_id, data in rows[-limit:]:
            parent_id = data.get("parent_variant_id")
            parent = self.graph.nodes.get(parent_id, {}) if parent_id else {}
            parent_metric = parent.get("metric")
            improved: bool | None = None
            if parent_metric is not None:
                delta = float(data["metric"]) - float(parent_metric)
                improved = delta > 0 if maximize_metric else delta < 0
            summary.append(
                {
                    "node_id": node_id,
                    "iteration": data.get("iteration"),
                    "method": data.get("method") or {},
                    "metric": data.get("metric"),
                    "parent_metric": parent_metric,
                    "improved": improved,
                }
            )
        return summary

    def candidate_frontier(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return proposed/deferred nodes that remain eligible for revisiting."""

        eligible = {
            "proposed",
            "ranked",
            "deferred_outcome",
            "needs_evidence_outcome",
            "promising_outcome",
        }
        return [
            row
            for row in self.method_pool(limit=limit)
            if row.get("node_type") == "proposal" and row.get("status") in eligible
        ]

    def frontier_changes(self, limit: int = 50) -> list[tuple[str, ChangeRequestArtifact]]:
        """Rehydrate unexecuted method nodes for a later path comparison."""

        result: list[tuple[str, ChangeRequestArtifact]] = []
        for row in self.candidate_frontier(limit=limit):
            raw_change = row.get("change_request")
            if not raw_change:
                continue
            parent_ids = [
                parent_id
                for parent_id in self.graph.predecessors(str(row["node_id"]))
                if self.graph.edges[parent_id, str(row["node_id"])].get("edge_type")
                in {"candidate", "lineage"}
            ]
            parent_id = parent_ids[0] if parent_ids else None
            change = ChangeRequestArtifact.model_validate(raw_change).model_copy(
                update={"parent_variant_id": parent_id}
            )
            result.append((str(row["node_id"]), change))
        return result

    def set_proposal_priority(self, proposal_id: str, priority: float) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["priority"] = priority
            self.graph.nodes[proposal_id]["status"] = "ranked"
            self._save()

    def mark_proposal_selected(self, proposal_id: str) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["status"] = "selected"
            self._save()

    def mark_proposal_replan_rejected(
        self, proposal_id: str, reason: str, adjustments: list[str]
    ) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["status"] = "replan_rejected"
            self.graph.nodes[proposal_id]["replan_reason"] = reason
            self.graph.nodes[proposal_id]["candidate_adjustments"] = adjustments
            self._save()

    def record_outcome(self, proposal_id: str, variant_id: str, status: str) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["status"] = f"{status}_outcome"
            if variant_id in self.graph:
                self.graph.add_edge(
                    proposal_id,
                    variant_id,
                    relation="executed_as",
                    status=status,
                    edge_type="execution",
                    label="executed_as",
                    reason="候选方法已执行，连接到本轮实际结果节点。",
                    target_change=self._change_payload(variant_id),
                )
            self._save()

    def context_for(
        self, variant_id: str | None, query: str, limit: int = 14
    ) -> list[dict[str, Any]]:
        """Return a compact, directed evidence pack for the current question.

        Ancestors explain how the active method was reached.  Siblings provide
        matched alternatives from the same parent.  The frontier exposes prior
        proposed or deferred branches that can be revisited.  This deliberately
        avoids treating a two-hop undirected neighbourhood as one flat history.
        """

        if not variant_id or variant_id not in self.graph:
            return self._latest_context(query, limit)

        rows: list[tuple[float, dict[str, Any]]] = []
        reverse = self.graph.reverse(copy=False)
        for node_id, distance in nx.single_source_shortest_path_length(
            reverse, variant_id, cutoff=4
        ).items():
            if node_id == variant_id or self.graph.nodes[node_id].get("node_type") in {"proposal", "root"}:
                continue
            try:
                trace_path = nx.shortest_path(self.graph, node_id, variant_id)
            except nx.NetworkXNoPath:
                trace_path = [node_id, variant_id]
            rows.append(
                (
                    self._relevance(node_id, query, distance) + 3.0,
                    self._row(
                        node_id,
                        "path_history",
                        distance,
                        edge_trace=self._edge_trace(trace_path),
                        traversal="backtrack_ancestor",
                    ),
                )
            )

        # Read explicitly related neighbours as well as strict ancestors and
        # siblings.  This is the part that lets a decision move between two
        # logically connected methods without pretending they form one linear
        # parent-child chain.
        for neighbour_id in set(self.graph.predecessors(variant_id)).union(
            self.graph.successors(variant_id)
        ):
            edge = self.graph.get_edge_data(variant_id, neighbour_id) or self.graph.get_edge_data(neighbour_id, variant_id) or {}
            if edge.get("edge_type") != "related":
                continue
            rows.append(
                (
                    self._relevance(neighbour_id, query, 1) + 1.0,
                    self._row(
                        neighbour_id,
                        "related_method",
                        1,
                        edge_trace=self._edge_trace_between(variant_id, neighbour_id),
                    ),
                )
            )

        parents = [
            node_id
            for node_id in self.graph.predecessors(variant_id)
            if self.graph.edges[node_id, variant_id].get("edge_type") == "lineage"
        ]
        for parent_id in parents:
            for sibling_id in self.graph.successors(parent_id):
                if sibling_id == variant_id:
                    continue
                edge = self.graph.edges[parent_id, sibling_id]
                if edge.get("edge_type") not in {"lineage", "candidate"}:
                    continue
                rows.append(
                    (
                        self._relevance(sibling_id, query, 1) + 2.0,
                        self._row(
                            sibling_id,
                            "matched_alternative",
                            1,
                            edge_trace=self._edge_trace_between(parent_id, sibling_id),
                        ),
                    )
                )

        # Revisit candidate branches hanging from the whole active lineage, not
        # only the latest parent. This keeps an older deferred branch reachable
        # after several adopted descendants have been explored.
        lineage = [variant_id]
        lineage.extend(
            node_id
            for node_id, distance in nx.single_source_shortest_path_length(
                reverse, variant_id, cutoff=4
            ).items()
            if node_id != variant_id and self.graph.nodes[node_id].get("node_type") not in {"proposal", "root"}
        )
        for anchor_distance, anchor_id in enumerate(lineage):
            for child_id in self.graph.successors(anchor_id):
                edge = self.graph.edges[anchor_id, child_id]
                if edge.get("edge_type") != "candidate":
                    continue
                status = self.graph.nodes[child_id].get("status", "")
                if status not in {"proposed", "ranked", "selected", "deferred_outcome", "needs_evidence_outcome"}:
                    continue
                rows.append(
                    (
                        self._relevance(child_id, query, anchor_distance + 1) + 1.5,
                        self._row(
                            child_id,
                            "candidate_frontier",
                            anchor_distance + 1,
                            edge_trace=self._edge_trace_between(anchor_id, child_id),
                        ),
                    )
                )

        return self._deduplicate(rows, limit)

    def expansion_hints(self, variant_id: str | None) -> list[dict[str, str]]:
        """Expose the graph's available next moves before the LLM proposes code."""

        if not variant_id or variant_id not in self.graph:
            return [
                {
                    "relation": "explore",
                    "reason": "No validated method exists yet; establish a runnable reference path.",
                }
            ]
        node = self.graph.nodes[variant_id]
        hints = [
            {
                "relation": "deepen",
                "reason": "Extend one supported component while keeping the rest comparable.",
            },
            {
                "relation": "ablate",
                "reason": "Remove or hold one component fixed to identify the source of a result.",
            },
            {
                "relation": "combine",
                "reason": "Combine complementary components or representation families as an explicit challenger; independent evidence is useful but not a prerequisite for measuring the composition.",
            },
            {
                "relation": "explore",
                "reason": "Open one distinct branch when current evidence cannot answer the question.",
            },
        ]
        if node.get("status") in {"failed", "deferred", "needs_evidence"}:
            hints.insert(
                0,
                {
                    "relation": "recover",
                    "reason": "Repair a recorded technical obstacle or use a neighboring implementation.",
                },
            )
        return hints

    def neighborhood(self, variant_id: str | None, radius: int = 2) -> list[dict[str, Any]]:
        """Compatibility wrapper for callers that do not yet provide a query."""

        return self.context_for(variant_id, "", limit=max(1, radius * 6))

    def latest_nodes(self, limit: int = 12) -> list[dict[str, Any]]:
        values = list(self.graph.nodes(data=True))[-limit:]
        return [dict(data) for _, data in values]

    def export_payload(self) -> dict[str, Any]:
        """Return nodes and all typed edges for offline monitoring/export."""

        data = nx.node_link_data(self.graph, edges="edges")
        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
        }

    def best_outcome_id(self, maximize_metric: bool = True) -> str | None:
        """Return the best measured outcome as a memory anchor.

        The code parent and the graph-memory anchor are intentionally separate:
        an unadopted result may still be the most useful place to continue
        evidence search, while its code must not silently become the incumbent.
        """

        measured = [
            (node_id, data)
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") == "outcome" and data.get("metric") is not None
        ]
        if not measured:
            return None
        key = lambda item: float(item[1]["metric"])
        return (max if maximize_metric else min)(measured, key=key)[0]

    @staticmethod
    def priority(
        candidate: CandidatePath, remaining_seconds: int, weights: ValueWeights
    ) -> float:
        value = (
            weights.alpha * candidate.expected_gain
            + weights.beta * candidate.information_gain
        )
        time_ratio = candidate.estimated_seconds / max(remaining_seconds, 1)
        denominator = (
            1.0 + weights.gamma * time_ratio + weights.delta * candidate.failure_risk
        )
        return value / denominator

    def rank(
        self,
        candidates: Iterable[CandidatePath],
        remaining_seconds: int,
        weights: ValueWeights,
        maximize_metric: bool = True,
        reserve_seconds: int = 0,
    ) -> list[tuple[CandidatePath, float]]:
        scored = []
        for candidate in candidates:
            calibrated = self.calibrate(candidate, maximize_metric)
            signal = self.selection_signal(
                calibrated,
                remaining_seconds,
                reserve_seconds=reserve_seconds,
                maximize_metric=maximize_metric,
            )
            adjusted = calibrated.model_copy(
                update={
                    "expected_gain": max(
                        0.0,
                        calibrated.expected_gain * signal["gain_multiplier"],
                    ),
                    "information_gain": max(
                        0.0,
                        calibrated.information_gain
                        * signal["information_multiplier"],
                    ),
                }
            )
            priority = (
                self.priority(adjusted, remaining_seconds, weights)
                if signal["feasible"]
                else float("-inf")
            )
            scored.append((calibrated, priority))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)

    def selection_signal(
        self,
        candidate: CandidatePath,
        remaining_seconds: int,
        reserve_seconds: int = 0,
        maximize_metric: bool = True,
    ) -> dict[str, Any]:
        """Explain how graph evidence changes a candidate's priority.

        The graph contributes concrete coverage controls. A family with no
        measured outcome receives an information bonus, while a family with
        several non-improving outcomes receives a repeat penalty. A candidate
        that only repeats the most recent changed factors is penalized after
        two non-improving outcomes; an explicit composition remains eligible
        as an information-seeking challenger. These controls never delete a
        branch. Time feasibility is a hard condition, not a soft score.
        """

        family = candidate.method.family
        outcomes = [
            data
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == "outcome"
            and data.get("method", {}).get("family", "unspecified") == family
            and data.get("metric") is not None
        ]
        all_outcomes = [
            data
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == "outcome" and data.get("metric") is not None
        ]
        ordered_outcomes = sorted(
            all_outcomes,
            key=lambda data: (
                int(data.get("iteration") or 0),
                str(data.get("variant_id") or ""),
            ),
        )
        unproductive = 0
        historical_factors: set[str] = set()
        for data in outcomes:
            historical_factors.update(data.get("method", {}).get("changed_factors", []))
            parent_id = data.get("parent_variant_id")
            parent_metric = (
                self.graph.nodes[parent_id].get("metric")
                if parent_id in self.graph.nodes
                else None
            )
            if parent_metric is None:
                continue
            improvement = float(data["metric"]) - float(parent_metric)
            if not maximize_metric:
                improvement = -improvement
            if improvement <= 0:
                unproductive += 1
        candidate_factors = set(candidate.method.changed_factors)
        novel_factors = sorted(candidate_factors - historical_factors)
        saturated = len(outcomes) >= 2 and unproductive >= 2
        initial_phase = not all_outcomes
        positive_families: set[str] = set()
        positive_factors: set[str] = set()
        for data in all_outcomes:
            parent_id = data.get("parent_variant_id")
            parent_metric = (
                self.graph.nodes[parent_id].get("metric")
                if parent_id in self.graph.nodes
                else None
            )
            if parent_metric is None:
                continue
            improvement = float(data["metric"]) - float(parent_metric)
            if not maximize_metric:
                improvement = -improvement
            if improvement > 0:
                positive_families.add(
                    data.get("method", {}).get("family", "unspecified")
                )
                positive_factors.update(data.get("method", {}).get("changed_factors", []))
        is_composition = _looks_like_composition(candidate)
        composition_supported = bool(
            positive_families.intersection({candidate.method.family})
            or positive_factors.intersection(set(candidate.method.changed_factors))
            or is_composition
        )
        recent = ordered_outcomes[-3:]
        recent_factor_counts: Counter[str] = Counter()
        recent_non_improving = 0
        for data in recent:
            method = data.get("method") or {}
            parent_id = data.get("parent_variant_id")
            parent_metric = (
                self.graph.nodes[parent_id].get("metric")
                if parent_id in self.graph.nodes
                else None
            )
            if parent_metric is not None and data.get("metric") is not None:
                improvement = float(data["metric"]) - float(parent_metric)
                if not maximize_metric:
                    improvement = -improvement
                if improvement <= 0:
                    recent_non_improving += 1
                    recent_factor_counts.update(method.get("changed_factors", []))
        factor_stuck = bool(candidate_factors) and recent_non_improving >= 2 and all(
            recent_factor_counts[factor] >= 2 for factor in candidate_factors
        )
        composition_gap = is_composition and not any(
            _looks_like_method_metadata(data) for data in all_outcomes
        )
        feasible = candidate.estimated_seconds + max(0, reserve_seconds) <= max(
            0, remaining_seconds
        )
        exact_measured = any(
            data.get("method_signature") == _candidate_signature(candidate)
            for _, data in self.graph.nodes(data=True)
            if data.get("node_type") == "outcome"
        )
        return {
            "feasible": feasible,
            "required_seconds": candidate.estimated_seconds + max(0, reserve_seconds),
            "remaining_seconds": remaining_seconds,
            "reserve_seconds": reserve_seconds,
            "method_family": family,
            "family_outcome_count": len(outcomes),
            "family_unproductive_count": unproductive,
            "novel_changed_factors": novel_factors,
            "exact_measured": exact_measured,
            "initial_phase": initial_phase,
            "composition_supported": composition_supported,
            "composition_penalty": False,
            "composition_gap": composition_gap,
            "factor_stuck": factor_stuck,
            "recent_non_improving": recent_non_improving,
            "family_status": (
                "unmeasured"
                if not outcomes
                else "saturated"
                if saturated and not novel_factors
                else "active"
            ),
            "gain_multiplier": (
                0.45
                if factor_stuck and not is_composition
                else 0.55
                if saturated and not novel_factors
                else 1.0
            ),
            "information_multiplier": (
                1.45
                if initial_phase and not outcomes
                else 1.25
                if not outcomes
                else 1.35
                if is_composition and composition_gap
                else 1.25
                if factor_stuck
                else 0.45
                if saturated and not novel_factors
                else 1.0
            ),
        }

    def calibrate(
        self, candidate: CandidatePath, maximize_metric: bool = True
    ) -> CandidatePath:
        """Blend the LLM estimate with measured outcomes from related paths.

        A single historical run should not override the LLM's prior. Confidence
        grows with comparable outcomes and is capped, so the graph supplies a
        correction rather than silently becoming a brittle hand-tuned model.
        """

        samples: list[dict[str, Any]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "outcome" or data.get("metric") is None:
                continue
            if data.get("relation") != candidate.relation:
                continue
            method = data.get("method") or {}
            family = method.get("family", "unspecified")
            if candidate.method.family != "unspecified" and family != candidate.method.family:
                continue
            changed = set(candidate.method.changed_factors)
            historical_changed = set(method.get("changed_factors", []))
            if changed and historical_changed and not changed.intersection(historical_changed):
                continue
            parent_id = data.get("parent_variant_id")
            parent_metric = (
                self.graph.nodes[parent_id].get("metric")
                if parent_id in self.graph.nodes
                else None
            )
            gain = None
            if parent_metric is not None:
                gain = float(data["metric"]) - float(parent_metric)
                if not maximize_metric:
                    gain = -gain
            samples.append({"gain": gain, "failed": data.get("status") == "failed", "seconds": data.get("wall_seconds")})
        if not samples:
            return candidate

        usable_gains = [float(item["gain"]) for item in samples if item["gain"] is not None]
        confidence = min(0.75, len(samples) / 4.0)
        expected_gain = (
            sum(usable_gains) / len(usable_gains)
            if usable_gains
            else candidate.expected_gain
        )
        failure_rate = sum(bool(item["failed"]) for item in samples) / len(samples)
        durations = [float(item["seconds"]) for item in samples if item["seconds"] is not None]
        measured_seconds = sum(durations) / len(durations) if durations else candidate.estimated_seconds
        return candidate.model_copy(
            update={
                "expected_gain": max(0.0, (1.0 - confidence) * candidate.expected_gain + confidence * max(0.0, expected_gain)),
                "failure_risk": min(1.0, (1.0 - confidence) * candidate.failure_risk + confidence * failure_rate),
                "estimated_seconds": max(1, round((1.0 - confidence) * candidate.estimated_seconds + confidence * measured_seconds)),
            }
        )

    def _infer_relation(
        self, parent_method: dict[str, Any] | None, change: ChangeRequestArtifact
    ) -> str:
        # The relation is part of the Agent's research declaration.  Structural
        # inference is intentionally advisory only; silently changing explore
        # into ablate/deepen changes the question that the Agent selected.
        return change.relation

    @staticmethod
    def _relation_warning(
        parent_method: dict[str, Any] | None, change: ChangeRequestArtifact
    ) -> str | None:
        """Report a semantic mismatch without rewriting the Agent's label."""

        if not parent_method:
            return None
        parent = MethodDescriptor.model_validate(parent_method)
        changed = set(change.method.changed_factors)
        if not changed:
            changed = {
                key
                for key, value in change.method.components.items()
                if parent.components.get(key) != value
            }
        values = {str(change.method.components.get(key, "")).lower() for key in changed}
        removed = set(parent.components).difference(change.method.components)
        looks_like_ablation = bool(removed) or any(
            value in {"none", "off", "removed", "disabled"} for value in values
        )
        if looks_like_ablation and change.relation != "ablate":
            return "组件图看起来移除了父版本因素，但候选未声明 ablate；请确认这是有意消融还是完整方法演进。"
        if change.relation == "ablate" and not looks_like_ablation:
            return "候选声明为 ablate，但组件图没有显示被移除或关闭的父版本因素；请明确消融对象。"
        return None

    def relation_warning_for(
        self, parent_variant_id: str | None, change: ChangeRequestArtifact
    ) -> str | None:
        parent = self.graph.nodes.get(parent_variant_id, {}) if parent_variant_id else {}
        return self._relation_warning(parent.get("method"), change)

    def _latest_context(self, query: str, limit: int) -> list[dict[str, Any]]:
        rows = [
            (self._relevance(node_id, query, 0), self._row(node_id, "recent_history", 0))
            for node_id in self.graph.nodes
            if self.graph.nodes[node_id].get("node_type") != "proposal"
        ]
        return self._deduplicate(rows, limit)

    def _relevance(self, node_id: str, query: str, distance: int) -> float:
        data = self.graph.nodes[node_id]
        haystack = json.dumps(data, ensure_ascii=False).lower()
        overlap = sum(token in haystack for token in _tokens(query))
        status_bonus = 0.5 if data.get("status") in {"adopted", "adopted_outcome", "promising_outcome", "deferred_outcome"} else 0.0
        return float(overlap) + status_bonus - 0.25 * distance

    def _edge_trace_between(self, source: str, target: str) -> list[dict[str, Any]]:
        try:
            path = nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(self.graph, target, source)
            except nx.NetworkXNoPath:
                return []
        return self._edge_trace(path)

    def _edge_trace(self, path: list[str]) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for source, target in pairwise(path):
            edge = self.graph.get_edge_data(source, target) or {}
            trace.append(
                {
                    "source": source,
                    "target": target,
                    "relation": edge.get("relation"),
                    "edge_type": edge.get("edge_type"),
                    "label": edge.get("label"),
                    "reason": edge.get("reason", ""),
                    "target_change": edge.get("target_change"),
                }
            )
        return trace

    def _row(
        self,
        node_id: str,
        role: str,
        distance: int,
        edge_trace: list[dict[str, Any]] | None = None,
        traversal: str | None = None,
    ) -> dict[str, Any]:
        data = dict(self.graph.nodes[node_id])
        return {
            "context_role": role,
            "distance": distance,
            "node_id": node_id,
            "edge_trace": edge_trace or [],
            "traversal": traversal or role,
            **data,
        }

    @staticmethod
    def _deduplicate(rows: list[tuple[float, dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
        chosen: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, row in rows:
            node_id = str(row["node_id"])
            if node_id not in chosen or score > chosen[node_id][0]:
                chosen[node_id] = (score, row)
        return [row for _, row in sorted(chosen.values(), key=lambda item: item[0], reverse=True)[:limit]]

    def _save(self) -> None:
        data = nx.node_link_data(self.graph, edges="edges")
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.graph = nx.node_link_graph(data, edges="edges", directed=True)

    def _enrich_edges(self) -> None:
        """Upgrade old graph records with explicit direction semantics."""

        changed = False
        for source, target, edge in self.graph.edges(data=True):
            relation = str(edge.get("relation", edge.get("label", "related")))
            if "edge_type" not in edge:
                edge["edge_type"] = (
                    "lineage"
                    if relation in {"deepen", "ablate", "combine", "explore", "recover"}
                    else "related"
                )
                changed = True
            if "label" not in edge:
                edge["label"] = relation
                changed = True
            if "reason" not in edge:
                edge["reason"] = {
                    "same_family": "同一方法族的前序证据，供当前分支比较。",
                    "shared_factor": "共享改动因素的前序证据，供当前分支归因。",
                    "informed_by": "当前方法的研究决策显式参考该已测量版本。",
                    "executed_as": "候选方法已执行，连接到本轮实际结果节点。",
                }.get(relation, "沿有向方法关系追踪到该节点。")
                changed = True
            if "target_change" not in edge:
                edge["target_change"] = self._change_payload(str(target))
                changed = True
        if changed:
            self._save()

    def _prune_related_edges(self) -> None:
        """Trim dense legacy similarity cliques to a few useful incoming links."""

        incoming: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for source, target, edge in self.graph.edges(data=True):
            if edge.get("edge_type") == "related":
                incoming.setdefault(str(target), []).append((str(source), str(target), dict(edge)))
        remove: list[tuple[str, str]] = []
        for target, candidates in incoming.items():
            candidates.sort(key=lambda item: float(item[2].get("weight", 0.0)), reverse=True)
            remove.extend((source, target) for source, _, _ in candidates[2:])
        if remove:
            self.graph.remove_edges_from(remove)
            self._save()

    def _ensure_root_node(self) -> None:
        if ROOT_NODE_ID in self.graph:
            return
        self.graph.add_node(
            ROOT_NODE_ID,
            node_type="root",
            variant_id=ROOT_NODE_ID,
            title="project root",
            relation="explore",
            mutation_class="implementation",
            question="",
            evidence_summary="项目的初始方法空间；没有代码父版本的候选从这里扩展。",
            status="root",
            method=MethodDescriptor().model_dump(mode="json"),
        )

    def _ensure_root_edges(self) -> None:
        """Give root-level nodes an explicit, visible provenance edge.

        Older graph files were allowed to create first-round proposals without
        a parent, which left them visually and semantically orphaned.  A root
        edge records that absence of a code parent without inventing one.
        """

        targets = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") in {"proposal", "outcome"}
        ]
        if not targets:
            return
        self._ensure_root_node()
        changed = False
        for node_id in targets:
            has_code_parent = any(
                self.graph.edges[source, node_id].get("edge_type")
                in {"candidate", "lineage", "root"}
                for source in self.graph.predecessors(node_id)
            )
            if has_code_parent or self.graph.has_edge(ROOT_NODE_ID, node_id):
                continue
            data = self.graph.nodes[node_id]
            relation = str(data.get("relation", "explore"))
            self.graph.add_edge(
                ROOT_NODE_ID,
                node_id,
                relation=relation,
                status=data.get("status", "proposed"),
                edge_type="root",
                label=f"root: {relation}",
                reason="历史节点没有代码父版本，补充项目根节点关系以保留其初始来源。",
                target_change=self._change_payload(node_id),
            )
            changed = True
        if changed:
            self._save()

def method_signature(change: ChangeRequestArtifact) -> str:
    """Stable, task-local identity for a proposed method.

    Titles and prose are intentionally excluded.  Two methods that change the
    same family/components/factors represent one method node even when the LLM
    describes them with different wording.
    """

    return _descriptor_signature(
        change.method,
        change.mutation_class.value,
        change.relation,
        change.research_question,
        change.title,
    )


def candidate_coverage_gap(
    changes: Iterable[ChangeRequestArtifact],
    *,
    initial: bool,
    recent_outcomes: Iterable[dict[str, Any]] | None = None,
) -> str | None:
    """Return a missing-search-direction warning before ranking candidates.

    This is intentionally task agnostic. It does not name a predictor or a
    feature type; it notices either an initial batch with multiple distinct
    families but no composition candidate, or a later batch that repeats recent
    factors after non-improving outcomes. The orchestrator gives the same
    research question one bounded revision pass instead of silently accepting a
    narrow candidate batch.
    """

    items = list(changes)
    if initial:
        families = {
            change.method.family
            for change in items
            if change.method.family and change.method.family != "unspecified"
        }
        component_profiles = {
            json.dumps(change.method.components, ensure_ascii=False, sort_keys=True)
            for change in items
            if change.method.components
        }
        if len(families) < 2 and len(component_profiles) < 2:
            return None
        if any(_looks_like_composition(change) for change in items):
            return None
        return (
            "The initial batch covers multiple method families/components ("
            + ", ".join(sorted(families))
            + ") but contains no explicit composition/fusion challenger. Add one executable candidate that combines complementary components, while retaining independent controls."
        )

    recent = list(recent_outcomes or [])
    non_improving = [row for row in recent if row.get("improved") is False]
    if len(non_improving) < 2:
        return None
    recent_factors = {
        factor
        for row in recent
        for factor in (row.get("method") or {}).get("changed_factors", [])
    }
    recent_families = {
        (row.get("method") or {}).get("family", "unspecified") for row in recent
    }
    has_orthogonal = any(
        set(change.method.changed_factors) - recent_factors
        or (
            change.method.family not in recent_families
            and change.method.family != "unspecified"
        )
        for change in items
    )
    # A composition already measured in the recent family is another local
    # retest, not a new search direction. Otherwise repeated blend-weight and
    # fold checks can satisfy the coverage guard indefinitely.
    has_new_composition = any(
        _looks_like_composition(change)
        and change.method.family not in recent_families
        for change in items
    )
    if has_orthogonal or has_new_composition:
        return None
    return (
        "The last measured candidates contain at least two non-improving outcomes and the current batch repeats their method families and changed factors. Add one executable orthogonal-family or composition challenger, while retaining at most one controlled local refinement."
    )


def _looks_like_composition(change: ChangeRequestArtifact | CandidatePath) -> bool:
    return _looks_like_method_metadata(
        {
            "relation": change.relation,
            "mutation_class": getattr(change, "mutation_class", ""),
            "method": change.method.model_dump(mode="json"),
        }
    )


def _looks_like_method_metadata(data: dict[str, Any]) -> bool:
    relation = str(data.get("relation", "")).lower()
    mutation = str(data.get("mutation_class", "")).lower()
    try:
        method = MethodDescriptor.model_validate(data.get("method") or {})
    except (TypeError, ValueError):
        # Older graph files may contain partial method metadata. Such a row
        # should not make candidate ranking fail; relation/mutation text can
        # still provide a conservative composition signal.
        method = MethodDescriptor()
    text = " ".join(
        [
            relation,
            mutation,
            method.family,
            *method.changed_factors,
            *method.components.keys(),
            *method.components.values(),
        ]
    ).lower()
    return relation == "combine" or mutation == "composition" or any(
        token in text
        for token in (
            "combine",
            "composition",
            "fusion",
            "ensemble",
            "concat",
            "stack",
            "blend",
            "multi-view",
            "multi_view",
            "feature union",
            "feature_union",
        )
    )


def _candidate_signature(candidate: CandidatePath) -> str:
    """Build the same stable identity for a ranked candidate as for a change."""

    payload = {
        "family": candidate.method.family,
        "components": candidate.method.components,
        "changed_factors": sorted(candidate.method.changed_factors),
        "target_scope": candidate.method.target_scope,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _descriptor_signature(
    method: MethodDescriptor,
    mutation_class: str,
    relation: str,
    question: str,
    title: str = "",
) -> str:
    # A method node represents the concrete change, not the wording of the
    # question that motivated it.  Relation and question remain on the node
    # and edge for interpretation, while identity stays stable across rounds.
    payload = {
        "family": method.family,
        "components": method.components,
        "changed_factors": sorted(method.changed_factors),
        "target_scope": method.target_scope,
    }
    # Legacy callers may omit MethodDescriptor details.  In that case the
    # title is the only available identity; fully described methods remain
    # stable when the LLM rephrases their question or title.
    if (
        method.family == "unspecified"
        and not method.components
        and not method.changed_factors
        and method.target_scope == "global"
    ):
        payload["fallback_title"] = title.strip().lower()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tokens(text: str) -> list[str]:
    latin = [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
        if len(token) > 1
    ]
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return latin + chinese
