"""File-backed reusable experiment memory.

Cards are append-only JSONL so they remain inspectable and portable. Retrieval is
transparent token matching plus project-local recency; a database can index this
file later without becoming the source of truth.
"""

from __future__ import annotations

import re
import uuid
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
    question: str
    conclusion: str
    measured_facts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(default_factory=list)
    relation: str
    decision: str
    metric: float | None = None
    tags: list[str] = Field(default_factory=list)


class ExperimentMemory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "cards.jsonl"

    def add(self, card: MemoryCard) -> MemoryCard:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(card.model_dump_json() + "\n")
        return card

    def search(self, task_id: str, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Return only conclusions that have lexical evidence of relevance."""

        tokens = set(_tokens(query))
        cards = [card for card in self._read() if card.task_id == task_id]
        scored: list[tuple[int, int, MemoryCard]] = []
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
                # Earlier tuple component is relevance; the second keeps newer
                # equally relevant cards ahead without hiding the score.
                scored.append((overlap, -position, card))
        return [
            {"relevance": score, **card.model_dump(mode="json")}
            for score, _, card in sorted(scored, reverse=True)[:limit]
        ]

    def _read(self) -> list[MemoryCard]:
        if not self.path.exists():
            return []
        cards = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cards.append(MemoryCard.model_validate_json(line))
        return cards


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
        if len(token) > 1
    ]
