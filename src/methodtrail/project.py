"""Project, session, and Git-worktree lifecycle management.

The source of truth is deliberately file- and Git-based:
project metadata, session handoffs, JSONL trajectories, candidate worktrees, and
Git commits can all be inspected without a database service.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import TaskContract


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ProjectRecord:
    project_id: str
    task_id: str
    repository: str
    template: str
    baseline_commit: str
    incumbent_commit: str
    incumbent_variant_id: str | None
    created_at: str
    updated_at: str
    incumbent_metric: float | None = None


@dataclass
class SessionRecord:
    session_id: str
    project_id: str
    task_id: str
    status: str
    created_at: str
    updated_at: str
    current_variant_id: str | None = None
    next_question: str | None = None
    # Number of completed research rounds that did not beat the best
    # checkpoint. It raises exploration pressure; it is not a stop counter.
    consecutive_non_improving: int = 0
    # Direction-aware diagnostic for the trajectory; retained for dashboards
    # and reports, but it no longer stops research by itself.
    consecutive_deteriorating: int = 0
    last_metric: float | None = None


@dataclass
class CandidateRecord:
    variant_id: str
    session_id: str
    parent_variant_id: str | None
    parent_commit: str
    branch: str
    workspace: str
    status: str
    created_at: str
    updated_at: str
    commit: str | None = None
    metric: float | None = None
    source_files: list[str] = field(default_factory=list)


class ProjectManager:
    """Creates one Git-backed project and explicit resumable research sessions."""

    def __init__(self, state_root: str | Path) -> None:
        self.state_root = Path(state_root)
        self.projects_root = self.state_root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def ensure_project(
        self, contract: TaskContract, project_id: str | None = None
    ) -> ProjectRecord:
        """Seed a Git repository from the task template once per task project."""

        identifier = project_id or contract.task_id
        project_dir = self.projects_root / identifier
        metadata_path = project_dir / "project.json"
        if metadata_path.exists():
            record = ProjectRecord(**json.loads(metadata_path.read_text(encoding="utf-8")))
            if record.task_id != contract.task_id:
                raise ValueError("project metadata task_id does not match task contract")
            return record

        project_dir.mkdir(parents=True, exist_ok=True)
        repository = project_dir / "repository"
        template = Path(contract.workspace_template).resolve()
        if not template.is_dir():
            raise FileNotFoundError(f"workspace template does not exist: {template}")

        shutil.copytree(template, repository, dirs_exist_ok=True)
        for data_path in contract.allowed_data_paths:
            target = repository / data_path
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

        ignore_lines = [
            ".methodtrail.diff",
            "predictions.csv",
            "metrics.json",
            "__pycache__/",
            ".pytest_cache/",
            *contract.allowed_data_paths,
        ]
        (repository / ".gitignore").write_text(
            "\n".join(ignore_lines) + "\n", encoding="utf-8"
        )
        self._git(repository, "init")
        self._git(repository, "config", "user.email", "methodtrail@local")
        self._git(repository, "config", "user.name", "MethodTrail")
        self._git(repository, "add", "--all")
        self._git(repository, "commit", "-m", "methodtrail: task baseline")
        baseline = self._git(repository, "rev-parse", "HEAD").strip()
        now = _now()
        record = ProjectRecord(
            project_id=identifier,
            task_id=contract.task_id,
            repository=str(repository),
            template=str(template),
            baseline_commit=baseline,
            incumbent_commit=baseline,
            incumbent_variant_id=None,
            created_at=now,
            updated_at=now,
            incumbent_metric=None,
        )
        self._write_json(metadata_path, asdict(record))
        return record

    def load_project(self, project_id: str) -> ProjectRecord:
        path = self.projects_root / project_id / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown MethodTrail project: {project_id}")
        return ProjectRecord(**json.loads(path.read_text(encoding="utf-8")))

    def load_session(self, project: ProjectRecord, session_id: str) -> SessionRecord:
        path = self._session_dir(project, session_id) / "session.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown session: {session_id}")
        return SessionRecord(**json.loads(path.read_text(encoding="utf-8")))

    def project_path(self, project: ProjectRecord) -> Path:
        return self._project_dir(project)

    def start_session(
        self, project: ProjectRecord, session_id: str | None = None
    ) -> SessionRecord:
        session_dir = self._session_dir(project, session_id or uuid.uuid4().hex)
        metadata_path = session_dir / "session.json"
        if metadata_path.exists():
            record = SessionRecord(**json.loads(metadata_path.read_text(encoding="utf-8")))
            if record.project_id != project.project_id:
                raise ValueError("session belongs to a different project")
            record.status = "active"
            record.updated_at = _now()
            self._write_json(metadata_path, asdict(record))
            return record

        session_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        record = SessionRecord(
            session_id=session_dir.name,
            project_id=project.project_id,
            task_id=project.task_id,
            status="active",
            created_at=now,
            updated_at=now,
        )
        self._write_json(metadata_path, asdict(record))
        (session_dir / "trajectory.jsonl").touch()
        return record

    def create_candidate(
        self,
        project: ProjectRecord,
        session: SessionRecord,
        contract: TaskContract,
        parent_variant_id: str | None,
    ) -> CandidateRecord:
        """Create a candidate branch/worktree and hydrate task data into it."""

        parent = self.get_candidate(project, parent_variant_id) if parent_variant_id else None
        parent_commit = parent.commit if parent and parent.commit else project.incumbent_commit
        variant_id = uuid.uuid4().hex
        branch = f"methodtrail/{session.session_id}/{variant_id[:10]}"
        # Worktrees need a deliberately short on-disk path on Windows. The full
        # session and variant identifiers remain in candidate metadata.
        workspace = self._project_dir(project) / "wt" / variant_id[:12]
        workspace.parent.mkdir(parents=True, exist_ok=True)
        self._git(
            Path(project.repository),
            "worktree",
            "add",
            "-b",
            branch,
            str(workspace),
            parent_commit,
        )
        # Each candidate has its own branch. The project baseline moves only when
        # an accepted candidate is recorded below.
        self._hydrate_inputs(Path(project.template), workspace, contract.allowed_data_paths)
        now = _now()
        record = CandidateRecord(
            variant_id=variant_id,
            session_id=session.session_id,
            parent_variant_id=parent_variant_id,
            parent_commit=parent_commit,
            branch=branch,
            workspace=str(workspace),
            status="prepared",
            created_at=now,
            updated_at=now,
        )
        self._write_candidate(project, record)
        self.append_event(
            project,
            session.session_id,
            "candidate_created",
            {"variant_id": variant_id, "parent_commit": parent_commit},
        )
        return record

    def record_candidate(
        self,
        project: ProjectRecord,
        candidate: CandidateRecord,
        *,
        status: str,
        metric: float | None = None,
    ) -> CandidateRecord:
        candidate.status = status
        candidate.metric = metric
        candidate.updated_at = _now()
        self._write_candidate(project, candidate)
        return candidate

    def adopt_candidate(
        self,
        project: ProjectRecord,
        candidate: CandidateRecord,
        changed_files: list[str],
        message: str,
        metric: float | None,
    ) -> CandidateRecord:
        workspace = Path(candidate.workspace)
        if changed_files:
            self._git(workspace, "add", "--", *changed_files)
            # A model can occasionally propose a no-op edit. In that case the
            # parent commit remains the candidate revision without a false commit.
            if self._git(workspace, "status", "--porcelain").strip():
                self._git(workspace, "commit", "-m", message)
        candidate.commit = self._git(workspace, "rev-parse", "HEAD").strip()
        self.record_candidate(project, candidate, status="adopted", metric=metric)
        project.incumbent_commit = candidate.commit
        project.incumbent_variant_id = candidate.variant_id
        project.incumbent_metric = metric
        project.updated_at = _now()
        self._write_json(self._project_dir(project) / "project.json", asdict(project))
        return candidate

    def get_candidate(
        self, project: ProjectRecord, variant_id: str | None
    ) -> CandidateRecord | None:
        if not variant_id:
            return None
        path = self._project_dir(project) / "candidates" / f"{variant_id}.json"
        return CandidateRecord(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None

    def session_candidates(
        self, project: ProjectRecord, session_id: str
    ) -> list[CandidateRecord]:
        directory = self._project_dir(project) / "candidates"
        if not directory.exists():
            return []
        candidates = [
            CandidateRecord(**json.loads(path.read_text(encoding="utf-8")))
            for path in directory.glob("*.json")
        ]
        return sorted(
            (item for item in candidates if item.session_id == session_id),
            key=lambda item: item.created_at,
        )

    def events(
        self, project: ProjectRecord, session_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        path = self._session_dir(project, session_id) / "trajectory.jsonl"
        if not path.exists():
            return []
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[-limit:]

    def set_incumbent(self, project: ProjectRecord, variant_id: str) -> ProjectRecord:
        """Roll back or advance the project pointer to a previously adopted variant."""

        candidate = self.get_candidate(project, variant_id)
        if not candidate or candidate.status != "adopted" or not candidate.commit:
            raise ValueError("incumbent can only point to an adopted candidate")
        project.incumbent_variant_id = candidate.variant_id
        project.incumbent_commit = candidate.commit
        project.incumbent_metric = candidate.metric
        project.updated_at = _now()
        self._write_json(self._project_dir(project) / "project.json", asdict(project))
        return project

    def update_session(
        self,
        project: ProjectRecord,
        session: SessionRecord,
        *,
        status: str | None = None,
        current_variant_id: str | None = None,
        next_question: str | None = None,
        consecutive_non_improving: int | None = None,
        consecutive_deteriorating: int | None = None,
        last_metric: float | None = None,
        handoff: str | None = None,
    ) -> SessionRecord:
        if status is not None:
            session.status = status
        if current_variant_id is not None:
            session.current_variant_id = current_variant_id
        if next_question is not None:
            session.next_question = next_question
        if consecutive_non_improving is not None:
            session.consecutive_non_improving = consecutive_non_improving
        if consecutive_deteriorating is not None:
            session.consecutive_deteriorating = consecutive_deteriorating
        if last_metric is not None:
            session.last_metric = last_metric
        session.updated_at = _now()
        session_dir = self._session_dir(project, session.session_id)
        self._write_json(session_dir / "session.json", asdict(session))
        if handoff is not None:
            (session_dir / "handoff.md").write_text(handoff, encoding="utf-8")
        return session

    def append_event(
        self,
        project: ProjectRecord,
        session_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        event = {"at": _now(), "kind": kind, "payload": payload}
        path = self._session_dir(project, session_id) / "trajectory.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _git(directory: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=directory,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout

    @staticmethod
    def _hydrate_inputs(template: Path, workspace: Path, allowed_paths: list[str]) -> None:
        for relative in allowed_paths:
            source = template / relative
            target = workspace / relative
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            elif source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def _project_dir(self, project: ProjectRecord) -> Path:
        preferred = self.projects_root / project.project_id
        # Projects created before explicit project IDs used task_id as their
        # directory name; retain access to those local histories.
        legacy = self.projects_root / project.task_id
        return preferred if preferred.exists() or not legacy.exists() else legacy

    def _session_dir(self, project: ProjectRecord, session_id: str) -> Path:
        return self._project_dir(project) / "sessions" / session_id

    def _write_candidate(self, project: ProjectRecord, candidate: CandidateRecord) -> None:
        path = self._project_dir(project) / "candidates" / f"{candidate.variant_id}.json"
        self._write_json(path, asdict(candidate))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
