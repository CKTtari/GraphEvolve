"""SQLite-backed artifact history for replay, retrieval, and later learning."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ArtifactStore:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    parent_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_kind_time
                    ON artifacts(kind, created_at DESC);
                """
            )

    def put(
        self,
        kind: str,
        payload: BaseModel | dict[str, Any],
        parent_ids: list[str] | None = None,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        raw = (
            payload.model_dump(mode="json")
            if isinstance(payload, BaseModel)
            else payload
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    kind,
                    json.dumps(raw, ensure_ascii=False),
                    json.dumps(parent_ids or []),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return artifact_id

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def recent(self, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts"
        params: tuple[Any, ...] = ()
        if kind:
            query += " WHERE kind = ?"
            params = (kind,)
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, (*params, limit)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "payload": json.loads(row["payload"]),
            "parent_ids": json.loads(row["parent_ids"]),
            "created_at": row["created_at"],
        }
