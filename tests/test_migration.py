"""Regression coverage for reopening recorded experiments after the name change."""

from pathlib import Path

from graphevolve import GraphEvolve, TaskContract
from graphevolve.project import ProjectManager, resolve_state_root


def test_new_projects_do_not_reuse_a_legacy_namespace(tmp_path: Path) -> None:
    legacy = tmp_path / ".methodtrail"
    legacy.mkdir()
    assert resolve_state_root(tmp_path, "new-project") == tmp_path / ".graphevolve"
    template = tmp_path / "template"
    template.mkdir()
    contract = TaskContract(
        task_id="new-project",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        run_command=["python", "solution.py"],
        metric_name="score",
    )
    # Without an explicit ID, the contract determines the new project namespace.
    experiment = GraphEvolve(tmp_path, None)
    experiment._ensure_project_session(contract)
    assert experiment.state_root == tmp_path / ".graphevolve"
    assert not (legacy / "projects" / contract.task_id).exists()


def test_legacy_session_is_reopened_without_moving_its_records(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    contract = TaskContract(
        task_id="migration-toy",
        description="toy",
        workspace_template=str(template),
        allowed_data_paths=[],
        run_command=["python", "solution.py"],
        metric_name="score",
    )
    state = tmp_path / "state"
    legacy = state / ".methodtrail"
    manager = ProjectManager(legacy)
    project = manager.ensure_project(contract, "recorded-project")
    session = manager.start_session(project)
    project_record = manager.project_path(project) / "project.json"
    recorded = project_record.read_bytes()
    # A new store may coexist; lookup must follow the requested project.
    (state / ".graphevolve").mkdir()

    experiment = GraphEvolve(
        state, None, project_id=project.project_id, session_id=session.session_id,
    )
    experiment._ensure_project_session(contract)

    assert experiment.state_root == legacy
    assert experiment.project is not None
    assert experiment.session is not None
    assert experiment.project.repository == project.repository
    assert experiment.session.session_id == session.session_id
    assert project_record.read_bytes() == recorded
    assert not (state / ".graphevolve" / "projects" / project.project_id).exists()
    assert (manager.project_path(project) / "sessions" / session.session_id / "dashboard.html").exists()
