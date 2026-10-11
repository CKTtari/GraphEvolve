"""Benchmark adapters expose task contracts, never ready-made predictors."""

from __future__ import annotations

from pathlib import Path

from .schemas import TaskContract


class MLEBenchAdapter:
    """Creates a contract for a prepared MLE-bench task directory."""

    def contract(
        self,
        task_id: str,
        task_root: str | Path,
        metric_name: str,
        run_command: list[str],
        evaluation_command: list[str],
        timeout_seconds: int,
    ) -> TaskContract:
        root = Path(task_root).resolve()
        return TaskContract(
            task_id=f"mle-bench:{task_id}",
            description="Write an ML solution from the supplied competition files and produce the required prediction file.",
            workspace_template=str(root),
            allowed_data_paths=["."],
            run_command=run_command,
            evaluation_command=evaluation_command,
            metric_name=metric_name,
            timeout_seconds=timeout_seconds,
            required_outputs=["submission.csv"],
            allowed_dependencies=["numpy", "pandas", "scikit-learn", "torch"],
        )


class SWEBenchAdapter:
    """Creates a code-repair contract for a checked-out SWE-bench repository instance."""

    def contract(
        self,
        instance_id: str,
        repository_root: str | Path,
        test_command: list[str],
        timeout_seconds: int = 1800,
    ) -> TaskContract:
        root = Path(repository_root).resolve()
        return TaskContract(
            task_id=f"swe-bench:{instance_id}",
            description="Resolve the supplied repository issue by editing code and passing the task test command.",
            workspace_template=str(root),
            allowed_data_paths=[],
            solution_entrypoint="",  # Repository tasks are verified by their test command.
            run_command=test_command,
            metric_file="metrics.json",
            metric_name="resolved",
            maximize_metric=True,
            timeout_seconds=timeout_seconds,
            required_outputs=[],
        )


class TerminalBenchAdapter:
    """Creates a task contract for a prepared Terminal-Bench workspace."""

    def contract(
        self,
        task_id: str,
        workspace_root: str | Path,
        verification_command: list[str],
        timeout_seconds: int = 1800,
    ) -> TaskContract:
        root = Path(workspace_root).resolve()
        return TaskContract(
            task_id=f"terminal-bench:{task_id}",
            description="Complete the terminal task and satisfy its verification command.",
            workspace_template=str(root),
            allowed_data_paths=[],
            solution_entrypoint="",
            run_command=verification_command,
            metric_file="metrics.json",
            metric_name="resolved",
            maximize_metric=True,
            timeout_seconds=timeout_seconds,
        )
