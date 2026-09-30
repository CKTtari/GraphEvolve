"""File-backed reusable experiment memory.

Cards are append-only JSONL so they remain inspectable and portable. Retrieval is
transparent token matching plus project-local recency; a database can index this
file later without becoming the source of truth.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MemoryCard(BaseModel):
    card_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str
    session_id: str
    variant_id: str
    parent_variant_id: str | None = None
    evidence_parent_ids: list[str] = Field(default_factory=list)
    question: str
    conclusion: str
    measured_facts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    relation: str
    decision: str
    metric: float | None = None
    tags: list[str] = Field(default_factory=list)
    method_family: str = "unspecified"
    changed_factors: list[str] = Field(default_factory=list)
    iteration: int | None = None
    title: str = ""
    mutation_class: str = ""
    change_logic: str = ""
    method_components: dict[str, Any] = Field(default_factory=dict)


class MemoryGraph:
    """A small graph over conclusions, separate from the method-path graph.

    The method graph answers *which method can be tried next*.  This graph
    answers *which past conclusions belong together*.  It links cards through
    parent variants, method families, changed factors and shared tags so a
    question retrieves connected evidence instead of only matching recent
    words in a flat log.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        if self.path.exists():
            self._load()

    def add(self, card: MemoryCard, *, persist: bool = True) -> None:
        node = {
            "card_id": card.card_id,
            "task_id": card.task_id,
            "variant_id": card.variant_id,
            "parent_variant_id": card.parent_variant_id,
            "evidence_parent_ids": card.evidence_parent_ids,
            "method_family": card.method_family,
            "changed_factors": sorted(set(card.changed_factors)),
            "relation": card.relation,
            "decision": card.decision,
            "metric": card.metric,
            "tags": card.tags,
            "iteration": card.iteration,
            "title": card.title or (card.tags[0] if card.tags else ""),
            "question": card.question,
            "conclusion": card.conclusion,
            "mutation_class": card.mutation_class,
            "change_logic": card.change_logic,
            "method_components": card.method_components,
        }
        self.nodes[card.card_id] = node
        existing = [
            value
            for key, value in self.nodes.items()
            if key != card.card_id and value.get("task_id") == card.task_id
        ]
        parent_linked = False
        for other in existing:
            if card.parent_variant_id and other.get("variant_id") == card.parent_variant_id:
                self._edge(
                    other,
                    node,
                    relation="follows",
                    edge_type="lineage",
                    weight=1.0,
                    reason="当前实验从父版本继续，并执行本轮记录的修改。",
                )
                parent_linked = True
                break

        # Preserve every explicit source that the research Agent cited.  These
        # directed links describe intellectual provenance; they are separate
        # from the single code-parent lineage link above.
        for evidence_id in card.evidence_parent_ids:
            for other in existing:
                if other.get("variant_id") != evidence_id:
                    continue
                self._edge(
                    other,
                    node,
                    relation="informed_by",
                    edge_type="evidence",
                    weight=1.0,
                    reason="本轮经验记录显式参考该已测量版本。",
                )
                break

        # Keep semantic evidence links sparse and directed.  They supplement a
        # lineage edge; they do not connect every pair of cards in a family.
        ranked: list[tuple[float, dict[str, Any], str, list[str]]] = []
        for other in existing:
            if parent_linked and other.get("variant_id") == card.parent_variant_id:
                continue
            shared_factors = sorted(
                set(card.changed_factors).intersection(other.get("changed_factors", []))
            )
            same_family = (
                card.method_family != "unspecified"
                and card.method_family == other.get("method_family")
            )
            if same_family:
                ranked.append((0.8 + 0.1 * len(shared_factors), other, "same_family", shared_factors))
            elif shared_factors:
                ranked.append((0.4 + 0.1 * len(shared_factors), other, "shared_factor", shared_factors))
        # At most two older evidence neighbours are shown for a new card.  The
        # newest card remains the target, so every relation has a clear arrow.
        ranked.sort(key=lambda item: item[0], reverse=True)
        for weight, other, relation, shared_factors in ranked[:2]:
            if any(
                edge.get("source") == other["card_id"]
                and edge.get("target") == card.card_id
                for edge in self.edges
            ):
                continue
            self._edge(
                other,
                node,
                relation=relation,
                edge_type="evidence",
                weight=min(1.0, weight),
                reason=(
                    "同一方法族的前序证据。"
                    if relation == "same_family"
                    else "共享改动因素的前序证据。"
                ),
                shared_factors=shared_factors,
            )
        if persist:
            self._save()

    def rebuild(self, cards: list[MemoryCard]) -> None:
        """Rebuild derived graph edges without deleting append-only memory cards."""

        self.nodes = {}
        self.edges = []
        for card in cards:
            self.add(card, persist=False)
        self._save()

    def profile(self, task_id: str) -> dict[str, Any]:
        cards = [node for node in self.nodes.values() if node.get("task_id") == task_id]
        decisions = Counter(str(node.get("decision", "unknown")) for node in cards)
        families = Counter(str(node.get("method_family", "unspecified")) for node in cards)
        return {
            "card_count": len(cards),
            "decisions": dict(decisions),
            "method_families": dict(families),
            "linked_card_count": sum(
                1
                for edge in self.edges
                if self.nodes.get(edge.get("source"), {}).get("task_id") == task_id
            ),
        }

    def related_ids(self, seed_ids: list[str], task_id: str, limit: int = 8) -> list[str]:
        """Return nearby cards reached through directed memory relations."""

        return [item["card_id"] for item in self.related_paths(seed_ids, task_id, limit)]

    def related_paths(
        self,
        seed_ids: list[str],
        task_id: str,
        limit: int = 8,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Trace directed ancestors first, then descendants, from lexical seeds."""

        incoming: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        outgoing: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if self.nodes.get(source, {}).get("task_id") != task_id:
                continue
            incoming[target].append(edge)
            outgoing[source].append(edge)
        queue: list[tuple[str, int, list[str], list[dict[str, Any]]]] = [
            (str(seed), 0, [str(seed)], []) for seed in seed_ids if str(seed) in self.nodes
        ]
        visited = set(seed_ids)
        found: dict[str, dict[str, Any]] = {}
        while queue:
            current, depth, path, traversed = queue.pop(0)
            if depth >= max_depth:
                continue
            # An incoming edge is a provenance path to an earlier conclusion;
            # outgoing edges expose later evidence branches.  Both remain
            # directed and are reported to the agent with their edge labels.
            choices: list[tuple[dict[str, Any], str]] = [
                (edge, "ancestor") for edge in incoming.get(current, [])
            ] + [(edge, "descendant") for edge in outgoing.get(current, [])]
            for edge, direction in choices:
                next_id = (
                    str(edge.get("source"))
                    if direction == "ancestor"
                    else str(edge.get("target"))
                )
                if next_id in visited or self.nodes.get(next_id, {}).get("task_id") != task_id:
                    continue
                visited.add(next_id)
                next_path = path + [next_id]
                next_edges = traversed + [
                    {
                        "relation": edge.get("relation"),
                        "edge_type": edge.get("edge_type"),
                        "reason": edge.get("reason", ""),
                        "target_change": edge.get("target_change", {}),
                        "direction": direction,
                    }
                ]
                score = float(edge.get("weight", 0.0)) * (0.72 ** depth)
                found[next_id] = {
                    "card_id": next_id,
                    "score": score,
                    "distance": depth + 1,
                    "path": next_path,
                    "edges": next_edges,
                }
                queue.append((next_id, depth + 1, next_path, next_edges))
        return sorted(found.values(), key=lambda item: item["score"], reverse=True)[:limit]

    def _edge(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
        *,
        relation: str,
        edge_type: str,
        weight: float,
        reason: str,
        shared_factors: list[str] | None = None,
    ) -> None:
        target_change = {
            "iteration": target.get("iteration"),
            "title": target.get("title", ""),
            "mutation_class": target.get("mutation_class", ""),
            "change_logic": target.get("change_logic", ""),
            "changed_factors": target.get("changed_factors", []),
            "method_components": target.get("method_components", {}),
        }
        edge = {
            "source": source["card_id"],
            "target": target["card_id"],
            "relation": relation,
            "edge_type": edge_type,
            "label": relation,
            "reason": reason,
            "shared_factors": shared_factors or [],
            "target_change": target_change,
            "weight": round(weight, 4),
        }
        if edge not in self.edges:
            self.edges.append(edge)

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"nodes": list(self.nodes.values()), "edges": self.edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {str(node["card_id"]): node for node in payload.get("nodes", [])}
        self.edges = list(payload.get("edges", []))


class ExperimentMemory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "cards.jsonl"
        self.graph = MemoryGraph(self.root / "memory_graph.json")
        # Projects created before the memory graph existed can be upgraded
        # without changing their append-only cards.
        if self.path.exists():
            # Edges are derived data.  Rebuilding them keeps old projects
            # compatible and removes dense legacy all-pairs links while
            # preserving every append-only card.
            cards = self._read()
            cards = self._enrich_legacy_cards(cards)
            self.graph.rebuild(cards)

    def add(self, card: MemoryCard) -> MemoryCard:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(card.model_dump_json() + "\n")
        self.graph.add(card)
        return card

    def search(self, task_id: str, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Retrieve lexical matches plus connected conclusions from the memory graph."""

        tokens = set(_tokens(query))
        cards = [card for card in self._read() if card.task_id == task_id]
        scored: list[tuple[float, int, MemoryCard, str]] = []
        by_id = {card.card_id: card for card in cards}
        seed_ids: list[str] = []
        for position, card in enumerate(reversed(cards)):
            searchable = " ".join(
                [
                    card.question,
                    card.conclusion,
                    *card.measured_facts,
                    *card.evidence,
                    *card.applicable_conditions,
                    *card.tags,
                ]
            ).lower()
            overlap = sum(token in searchable for token in tokens)
            if overlap:
                seed_ids.append(card.card_id)
                # Earlier tuple component is relevance; the second keeps newer
                # equally relevant cards ahead without hiding the score.
                scored.append((float(overlap), -position, card, "lexical"))
        related_paths = self.graph.related_paths(seed_ids, task_id, limit=max(limit, 8))
        related_meta = {item["card_id"]: item for item in related_paths}
        for item in related_paths:
            card_id = item["card_id"]
            card = by_id.get(card_id)
            if card is not None and card.card_id not in seed_ids:
                scored.append((0.35 * float(item["score"]), 0, card, "memory_graph"))
        ranked = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
        chosen: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, _, card, source in ranked:
            if card.card_id in seen:
                continue
            seen.add(card.card_id)
            row = {"relevance": score, "retrieval_source": source, **card.model_dump(mode="json")}
            if source == "memory_graph":
                row["memory_graph_trace"] = related_meta.get(card.card_id, {})
            chosen.append(row)
            if len(chosen) >= limit:
                break
        return chosen

    def graph_profile(self, task_id: str) -> dict[str, Any]:
        return self.graph.profile(task_id)

    def _read(self) -> list[MemoryCard]:
        if not self.path.exists():
            return []
        cards = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cards.append(MemoryCard.model_validate_json(line))
        return cards

    def _enrich_legacy_cards(self, cards: list[MemoryCard]) -> list[MemoryCard]:
        """Backfill graph-facing change metadata for cards written by older runs.

        Cards remain append-only evidence records.  This migration only adds
        fields that were already persisted on the corresponding method outcome
        node, so the dashboard can explain an old edge in the same way as a
        newly written card.
        """

        graph_path = self.root / "experiment_graph.json"
        if not graph_path.exists():
            return cards
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cards
        outcomes = {
            str(node.get("variant_id", node.get("id", ""))): node
            for node in payload.get("nodes", [])
            if node.get("node_type") == "outcome"
        }
        changed = False
        enriched: list[MemoryCard] = []
        for card in cards:
            node = outcomes.get(card.variant_id, {})
            change = node.get("change_request") or {}
            method = node.get("method") or change.get("method") or {}
            updates: dict[str, Any] = {}
            if card.iteration is None and node.get("iteration") is not None:
                updates["iteration"] = node["iteration"]
            if not card.title:
                updates["title"] = str(node.get("title") or (card.tags[0] if card.tags else ""))
            if not card.mutation_class:
                updates["mutation_class"] = str(node.get("mutation_class") or change.get("mutation_class") or "")
            if not card.change_logic:
                title = str(node.get("title") or (card.tags[0] if card.tags else "本轮方法修改"))
                factors = list(card.changed_factors or method.get("changed_factors", []))
                logic = str(change.get("rationale") or node.get("rationale") or "")
                if not logic:
                    logic = f"围绕“{title}”进行受控迭代；改动因素：{', '.join(factors) or '见方法组件'}。"
                updates["change_logic"] = logic
            if not card.method_components and method:
                updates["method_components"] = method.get("components") or {}
            if updates:
                enriched.append(card.model_copy(update=updates))
                changed = True
            else:
                enriched.append(card)
        if changed:
            # Keep the same card order and IDs.  This is a schema migration,
            # not a new experiment write.
            self.path.write_text(
                "".join(card.model_dump_json() + "\n" for card in enriched),
                encoding="utf-8",
            )
        return enriched


class BugRecord(BaseModel):
    """Technical failure history kept out of research conclusions."""

    bug_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str
    session_id: str
    variant_id: str
    phase: str
    failure_class: str
    message: str
    diagnosis: str = ""
    repair_directions: list[str] = Field(default_factory=list)
    action: str = "continue_repair"
    repair_step: int = 0
    resolved: bool = False
    question: str = ""
    related_method: str = ""


class BugMemory:
    """Append-only repair log; it is never supplied as experiment evidence."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "bugs.jsonl"

    def add(self, record: BugRecord) -> BugRecord:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record

    def read(self) -> list[BugRecord]:
        if not self.path.exists():
            return []
        return [
            BugRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def guidance(self, task_id: str, limit: int = 8) -> list[dict[str, str]]:
        """Return concise, distinct technical errors for the next code edit."""

        seen: set[str] = set()
        result: list[dict[str, str]] = []
        for record in reversed(self.read()):
            if record.task_id != task_id or record.resolved:
                continue
            message = record.message.strip()
            lines = [line.strip() for line in message.splitlines() if line.strip()]
            error = next(
                (
                    line
                    for line in reversed(lines)
                    if re.search(r"(?:Error|Exception|failed|could not be applied):", line)
                ),
                lines[-1] if lines else record.failure_class,
            )
            error = error[:300]
            key = error.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append({"error": error, "phase": record.phase})
            if len(result) >= limit:
                break
        return result


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
        if len(token) > 1
    ]
