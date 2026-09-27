"""Directed experiment-path memory and value-aware candidate choice."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
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


class ExperimentPathGraph:
    """Stores experiment evidence and the logical paths between method variants."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        if self.path.exists():
            self._load()

    def add_node(self, node: PathNode) -> None:
        data = node.model_dump(mode="json")
        data["node_type"] = "outcome"
        self.graph.add_node(node.variant_id, **data)
        if node.parent_variant_id:
            self.graph.add_edge(
                node.parent_variant_id,
                node.variant_id,
                relation=node.relation,
                status=node.status,
                edge_type="lineage",
            )
        self._save()

    def add_candidate_relation(
        self, parent_id: str, candidate_id: str, relation: str
    ) -> None:
        """Backward-compatible edge writer for callers that already have IDs."""
        self.graph.add_edge(
            parent_id, candidate_id, relation=relation, status="proposed"
        )
        self._save()

    def attach_proposals(
        self,
        parent_variant_id: str | None,
        changes: list[ChangeRequestArtifact],
        iteration: int,
    ) -> list[tuple[str, ChangeRequestArtifact]]:
        """Put candidates into the graph before selection.

        The LLM describes a method with a small component map.  The graph then
        derives a relation against the parent where possible and records every
        candidate as a proposed branch, including branches not selected today.
        """

        parent = self.graph.nodes.get(parent_variant_id, {}) if parent_variant_id else {}
        attached: list[tuple[str, ChangeRequestArtifact]] = []
        for index, original in enumerate(changes):
            relation = self._infer_relation(parent.get("method"), original)
            change = original.model_copy(update={"relation": relation})
            proposal_id = f"proposal-{iteration}-{index}-{uuid.uuid4().hex[:8]}"
            self.graph.add_node(
                proposal_id,
                node_type="proposal",
                variant_id=proposal_id,
                title=change.title,
                relation=relation,
                mutation_class=change.mutation_class.value,
                question=change.research_question,
                method=change.method.model_dump(mode="json"),
                expected_gain=change.expected_gain,
                information_gain=change.information_gain,
                estimated_seconds=change.estimated_seconds,
                failure_risk=change.failure_risk,
                status="proposed",
            )
            if parent_variant_id:
                self.graph.add_edge(
                    parent_variant_id,
                    proposal_id,
                    relation=relation,
                    status="proposed",
                    edge_type="candidate",
                )
            attached.append((proposal_id, change))
        self._save()
        return attached

    def set_proposal_priority(self, proposal_id: str, priority: float) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["priority"] = priority
            self.graph.nodes[proposal_id]["status"] = "ranked"
            self._save()

    def mark_proposal_selected(self, proposal_id: str) -> None:
        if proposal_id in self.graph:
            self.graph.nodes[proposal_id]["status"] = "selected"
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
            if node_id == variant_id or self.graph.nodes[node_id].get("node_type") == "proposal":
                continue
            rows.append((self._relevance(node_id, query, distance) + 3.0, self._row(node_id, "path_history", distance)))

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
                rows.append((self._relevance(sibling_id, query, 1) + 2.0, self._row(sibling_id, "matched_alternative", 1)))

        # Revisit candidate branches hanging from the whole active lineage, not
        # only the latest parent. This keeps an older deferred branch reachable
        # after several adopted descendants have been explored.
        lineage = [variant_id]
        lineage.extend(
            node_id
            for node_id, distance in nx.single_source_shortest_path_length(
                reverse, variant_id, cutoff=4
            ).items()
            if node_id != variant_id and self.graph.nodes[node_id].get("node_type") != "proposal"
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
                        self._row(child_id, "candidate_frontier", anchor_distance + 1),
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
                "reason": "Combine the active method with a complementary, previously evidenced component.",
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
    ) -> list[tuple[CandidatePath, float]]:
        scored = []
        for candidate in candidates:
            calibrated = self.calibrate(candidate, maximize_metric)
            scored.append((calibrated, self.priority(calibrated, remaining_seconds, weights)))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)

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
        if change.mutation_class.value == "recovery":
            return "recover"
        if not parent_method:
            return "explore"
        parent = MethodDescriptor.model_validate(parent_method)
        changed = set(change.method.changed_factors)
        if not changed:
            changed = {
                key
                for key, value in change.method.components.items()
                if parent.components.get(key) != value
            }
        values = {str(change.method.components.get(key, "")).lower() for key in changed}
        if any(value in {"none", "off", "removed", "disabled"} for value in values):
            return "ablate"
        if change.relation == "combine" or len(changed) > 1 and "fusion" in changed:
            return "combine"
        if parent.family == change.method.family and changed:
            return "deepen"
        return change.relation if change.relation in {"ablate", "combine", "explore"} else "explore"

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
        status_bonus = 0.5 if data.get("status") in {"adopted", "adopted_outcome", "deferred_outcome"} else 0.0
        return float(overlap) + status_bonus - 0.25 * distance

    def _row(self, node_id: str, role: str, distance: int) -> dict[str, Any]:
        data = dict(self.graph.nodes[node_id])
        return {
            "context_role": role,
            "distance": distance,
            "node_id": node_id,
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


def _tokens(text: str) -> list[str]:
    latin = [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
        if len(token) > 1
    ]
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return latin + chinese
