"""Programmatic verification and command execution; LLMs never self-grade code."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .schemas import RunArtifact, TaskContract, VerificationResult


class Verifier:
    def verify(self, workspace: Path, contract: TaskContract) -> VerificationResult:
        checks: list[dict] = []
        if contract.solution_entrypoint:
            entrypoint = workspace / contract.solution_entrypoint
            if not entrypoint.exists():
                checks.append(
                    {
                        "name": "entrypoint",
                        "passed": False,
                        "detail": f"missing {contract.solution_entrypoint}",
                    }
                )
                return VerificationResult(passed=False, checks=checks)
            try:
                py_compile.compile(str(entrypoint), doraise=True)
                checks.append(
                    {
                        "name": "python_compile",
                        "passed": True,
                        "detail": contract.solution_entrypoint,
                    }
                )
            except py_compile.PyCompileError as exc:
                checks.append(
                    {"name": "python_compile", "passed": False, "detail": str(exc)}
                )
            checks.extend(self._dependency_checks(entrypoint, contract))
        else:
            checks.append(
                {
                    "name": "repository_task",
                    "passed": True,
                    "detail": "verified by task test command",
                }
            )

        if (
            contract.metric_name != "resolved"
            and not contract.evaluation_command
            and not contract.private_evaluation_command
            and not contract.allow_self_reported_metric
        ):
            checks.append(
                {
                    "name": "independent_evaluation",
                    "passed": False,
                    "detail": "a benchmark task needs public or private evaluation command, or allow_self_reported_metric=true",
                }
            )

        for output in contract.required_outputs:
            parent = (workspace / output).parent
            checks.append(
                {
                    "name": f"output_parent:{output}",
                    "passed": parent.exists(),
                    "detail": str(parent),
                }
            )
        return VerificationResult(
            passed=all(check["passed"] for check in checks), checks=checks
        )

    @staticmethod
    def _dependency_checks(entrypoint: Path, contract: TaskContract) -> list[dict]:
        if not contract.allowed_dependencies:
            return []
        aliases = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
        allowed = {
            aliases.get(dependency.lower(), dependency.lower())
            .replace("-", "_")
            .split(".")[0]
            for dependency in contract.allowed_dependencies
        }
        allowed.update(sys.stdlib_module_names)
        imports: set[str] = set()
        try:
            tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
        except SyntaxError:
            return []
        disallowed = sorted(
            module
            for module in imports
            if module not in allowed and module != "__future__"
        )
        if not disallowed:
            return [
                {
                    "name": "allowed_dependencies",
                    "passed": True,
                    "detail": "imports permitted",
                }
            ]
        return [
            {
                "name": "allowed_dependencies",
                "passed": False,
                "detail": f"disallowed imports: {', '.join(disallowed)}",
            }
        ]


class Executor:
    def run(
        self,
        workspace: Path,
        contract: TaskContract,
        *,
        deadline_monotonic: float | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> RunArtifact:
        """Run a candidate with progress checks instead of a hidden 5-minute cap.

        The old runner used ``timeout_seconds`` for every subprocess.  That made
        a task contract accidentally decide that all experiments, including
        long training jobs, must finish in five minutes.  The new runner has no
        implicit per-command timeout.  It polls output at the contract's check
        interval, lets an optional monitor callback request termination, and
        respects only the shared research deadline or an explicit hard cap.
        """
        started = time.perf_counter()
        environment = {**os.environ, "METHODTRAIL_METRIC_FILE": contract.metric_file}
        check_interval = contract.execution_check_interval_seconds
        hard_timeout = contract.execution_hard_timeout_seconds
        smoke_return_code: int | None = None
        test_return_code: int | None = None
        all_checkpoints: list[dict[str, Any]] = []
        termination_reason: str | None = None
        if contract.smoke_command:
            smoke = self._run_monitored_command(
                contract.smoke_command,
                workspace,
                environment,
                check_interval_seconds=check_interval,
                hard_timeout_seconds=hard_timeout,
                deadline_monotonic=deadline_monotonic,
                progress_callback=progress_callback,
                stage="smoke",
            )
            smoke_return_code, smoke_timed_out, smoke_stdout, smoke_stderr, checkpoints, termination_reason = smoke
            all_checkpoints.extend(checkpoints)
            if smoke_return_code != 0 or smoke_timed_out:
                return RunArtifact(
                    command=contract.run_command,
                    return_code=-5,
                    timed_out=smoke_timed_out,
                    wall_seconds=time.perf_counter() - started,
                    stdout=self._bounded_text(smoke_stdout),
                    stderr=self._bounded_text(f"smoke command failed: {smoke_stderr}"),
                    smoke_return_code=smoke_return_code,
                    termination_reason=termination_reason,
                    execution_checkpoints=all_checkpoints,
                )
        if contract.test_command:
            test = self._run_monitored_command(
                contract.test_command,
                workspace,
                environment,
                check_interval_seconds=check_interval,
                hard_timeout_seconds=hard_timeout,
                deadline_monotonic=deadline_monotonic,
                progress_callback=progress_callback,
                stage="test",
            )
            test_return_code, test_timed_out, test_stdout, test_stderr, checkpoints, termination_reason = test
            all_checkpoints.extend(checkpoints)
            if test_return_code != 0 or test_timed_out:
                return RunArtifact(
                    command=contract.run_command,
                    return_code=-6,
                    timed_out=test_timed_out,
                    wall_seconds=time.perf_counter() - started,
                    stdout=self._bounded_text(test_stdout),
                    stderr=self._bounded_text(f"test command failed: {test_stderr}"),
                    smoke_return_code=smoke_return_code,
                    test_return_code=test_return_code,
                    termination_reason=termination_reason,
                    execution_checkpoints=all_checkpoints,
                )
        run_result = self._run_monitored_command(
            contract.run_command,
            workspace,
            environment,
            check_interval_seconds=check_interval,
            hard_timeout_seconds=hard_timeout,
            deadline_monotonic=deadline_monotonic,
            progress_callback=progress_callback,
            output_paths=[*contract.required_outputs, contract.metric_file],
            stage="run",
        )
        return_code, timed_out, stdout, stderr, checkpoints, termination_reason = run_result
        all_checkpoints.extend(checkpoints)
        outputs = [
            path for path in contract.required_outputs if (workspace / path).exists()
        ]
        if return_code == 0 and len(outputs) != len(contract.required_outputs):
            missing = sorted(set(contract.required_outputs) - set(outputs))
            return_code = -3
            stderr = f"{stderr}\nmissing required outputs: {', '.join(missing)}".strip()

        evaluation_return_code: int | None = None
        evaluation_stdout = ""
        evaluation_stderr = ""
        if return_code == 0 and contract.private_evaluator_dir:
            (
                evaluation_return_code,
                evaluation_timed_out,
                evaluation_stdout,
                evaluation_stderr,
                metric_path,
            ) = self._run_private_evaluation(
                workspace,
                contract,
                environment,
                deadline_monotonic=deadline_monotonic,
                progress_callback=progress_callback,
            )
            if evaluation_timed_out:
                timed_out = True
                termination_reason = termination_reason or "private evaluator execution deadline reached"
            if evaluation_return_code != 0:
                return_code = -4
                stderr = f"{stderr}\nprivate evaluation failed: {evaluation_stderr}".strip()
        elif return_code == 0 and contract.evaluation_command:
            evaluation_result = self._run_monitored_command(
                contract.evaluation_command,
                workspace,
                environment,
                check_interval_seconds=check_interval,
                hard_timeout_seconds=hard_timeout,
                deadline_monotonic=deadline_monotonic,
                progress_callback=progress_callback,
                output_paths=[contract.metric_file],
                stage="evaluation",
            )
            evaluation_return_code, evaluation_timed_out, evaluation_stdout, evaluation_stderr, checkpoints, evaluation_termination = evaluation_result
            all_checkpoints.extend(checkpoints)
            termination_reason = evaluation_termination or termination_reason
            if evaluation_timed_out:
                timed_out = True
            if evaluation_return_code != 0:
                return_code = -4
                stderr = f"{stderr}\nevaluation failed: {evaluation_stderr}".strip()
            metric_path = workspace / contract.metric_file
        else:
            metric_path = workspace / contract.metric_file
        metrics = self._read_metrics(metric_path)
        metric = metrics.get(contract.metric_name)
        metric_constraints_passed = all(
            key in metrics and metrics[key] >= minimum
            for key, minimum in contract.metric_constraints.items()
        )
        if metric is None and contract.metric_name == "resolved":
            metric = 1.0 if return_code == 0 and not timed_out else 0.0
        return RunArtifact(
            command=contract.run_command,
            return_code=return_code,
            timed_out=timed_out,
            wall_seconds=time.perf_counter() - started,
            stdout=self._bounded_text(stdout),
            stderr=self._bounded_text(stderr),
            evaluation_return_code=evaluation_return_code,
            evaluation_stdout=self._bounded_text(evaluation_stdout),
            evaluation_stderr=self._bounded_text(evaluation_stderr),
            smoke_return_code=smoke_return_code,
            test_return_code=test_return_code,
            metric=metric,
            metrics=metrics,
            metric_constraints_passed=metric_constraints_passed,
            output_files=outputs,
            termination_reason=termination_reason,
            execution_checkpoints=all_checkpoints,
        )

    def _run_private_evaluation(
        self,
        workspace: Path,
        contract: TaskContract,
        environment: dict[str, str],
        *,
        deadline_monotonic: float | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[int, bool, str, str, Path]:
        """Evaluate outputs outside the generated-code worktree.

        The public run has already ended when this method creates the private
        directory. Thus task labels and evaluator code are not present while the
        Coding Agent's program executes. `container` additionally requires a
        configured container runner; the current local runner fails closed when
        one is requested but unavailable.
        """

        private_source = Path(contract.private_evaluator_dir or "").resolve()
        if not private_source.is_dir():
            return -1, False, "", f"private evaluator directory missing: {private_source}", workspace / contract.metric_file
        if private_source == workspace or private_source in workspace.parents:
            return -1, False, "", "private evaluator directory must not be inside public workspace", workspace / contract.metric_file
        if contract.execution_isolation == "container":
            # No partial fallback: claiming container isolation without a live
            # container engine would make the private-label guarantee misleading.
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                text=True,
                capture_output=True,
                check=False,
            )
            if probe.returncode != 0:
                return -1, False, "", "container isolation requested but Docker engine is unavailable", workspace / contract.metric_file
            return -1, False, "", "container evaluation is not yet configured with a task image", workspace / contract.metric_file

        evaluation_command = (
            contract.private_evaluation_command or contract.evaluation_command
        )
        if not evaluation_command:
            return -1, False, "", "private evaluator requires private_evaluation_command", workspace / contract.metric_file
        private_root = Path(tempfile.mkdtemp(prefix="methodtrail-private-eval-"))
        try:
            shutil.copytree(private_source, private_root, dirs_exist_ok=True)
            for relative in contract.required_outputs:
                source = workspace / relative
                target = private_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            private_environment = {
                **environment,
                "METHODTRAIL_OUTPUT_DIR": str(private_root),
                "METHODTRAIL_METRIC_FILE": contract.metric_file,
            }
            code, timed_out, stdout, stderr, _checkpoints, termination_reason = self._run_monitored_command(
                evaluation_command,
                private_root,
                private_environment,
                check_interval_seconds=contract.execution_check_interval_seconds,
                hard_timeout_seconds=contract.execution_hard_timeout_seconds,
                deadline_monotonic=deadline_monotonic,
                progress_callback=progress_callback,
                output_paths=[contract.metric_file],
                stage="private_evaluation",
            )
            # Copy only the metric back. Labels and private evaluator sources never
            # enter the candidate workspace.
            metric_source = private_root / contract.metric_file
            metric_copy = workspace / ".methodtrail.private.metrics.json"
            if metric_source.exists():
                shutil.copy2(metric_source, metric_copy)
            if termination_reason and not stderr:
                stderr = termination_reason
            return code, timed_out, stdout, stderr, metric_copy
        finally:
            # The metric is copied before cleanup; private labels are ephemeral.
            with contextlib.suppress(OSError):
                shutil.rmtree(private_root)

    @staticmethod
    def _run_command(
        command: list[str],
        workspace: Path,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> tuple[int, bool, str, str]:
        """Compatibility helper retaining the old explicit timeout API."""
        code, timed_out, stdout, stderr, _checkpoints, _reason = Executor._run_monitored_command(
            command,
            workspace,
            environment,
            check_interval_seconds=max(1, min(timeout_seconds, 300)),
            hard_timeout_seconds=timeout_seconds,
            deadline_monotonic=None,
            progress_callback=None,
            stage="command",
        )
        return code, timed_out, stdout, stderr

    @staticmethod
    def _bounded_text(value: str, limit: int = 12000) -> str:
        if len(value) <= limit:
            return value
        head = max(1000, limit // 3)
        tail = limit - head - 80
        return value[:head] + f"\n...[truncated {len(value) - limit} chars]...\n" + value[-tail:]

    @classmethod
    def _run_monitored_command(
        cls,
        command: list[str],
        workspace: Path,
        environment: dict[str, str],
        *,
        check_interval_seconds: int,
        hard_timeout_seconds: int | None,
        deadline_monotonic: float | None,
        progress_callback: Callable[[dict[str, Any]], Any] | None,
        output_paths: list[str] | None = None,
        stage: str,
    ) -> tuple[int, bool, str, str, list[dict[str, Any]], str | None]:
        checkpoints: list[dict[str, Any]] = []
        command_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="methodtrail-exec-") as temp_dir:
            stdout_path = Path(temp_dir) / "stdout.log"
            stderr_path = Path(temp_dir) / "stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    env=environment,
                    text=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    stdin=subprocess.DEVNULL,
                )
                termination_reason: str | None = None
                timed_out = False
                last_check = command_started
                last_bytes: dict[str, int] = {}

                def snapshot() -> dict[str, Any]:
                    output_bytes: dict[str, int] = {}
                    for relative in output_paths or []:
                        path = workspace / relative
                        try:
                            output_bytes[relative] = path.stat().st_size
                        except OSError:
                            output_bytes[relative] = 0
                    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                    current = {
                        "stage": stage,
                        "elapsed_seconds": round(time.monotonic() - command_started, 3),
                        "process_running": process.poll() is None,
                        "stdout_tail": cls._bounded_text(stdout_text, 5000),
                        "stderr_tail": cls._bounded_text(stderr_text, 5000),
                        "output_bytes": output_bytes,
                        "output_growth": {
                            key: output_bytes.get(key, 0) - last_bytes.get(key, 0)
                            for key in output_bytes
                        },
                        "metric_file_exists": bool(
                            output_paths and any(path.endswith("metrics.json") and (workspace / path).exists() for path in output_paths)
                        ),
                    }
                    last_bytes.clear()
                    last_bytes.update(output_bytes)
                    return current

                while process.poll() is None:
                    now = time.monotonic()
                    hard_deadline = (
                        command_started + hard_timeout_seconds
                        if hard_timeout_seconds is not None
                        else None
                    )
                    if deadline_monotonic is not None and now >= deadline_monotonic:
                        timed_out = True
                        termination_reason = "research budget deadline reached"
                    elif hard_deadline is not None and now >= hard_deadline:
                        timed_out = True
                        termination_reason = "explicit execution hard timeout reached"
                    if termination_reason:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    wait_for = max(0.05, min(float(check_interval_seconds), 1.0))
                    # Poll in one-second slices so the outer research deadline
                    # is honored promptly without busy-spinning.
                    if process.poll() is None:
                        try:
                            process.wait(timeout=wait_for)
                        except subprocess.TimeoutExpired:
                            pass
                    now = time.monotonic()
                    if now - last_check >= check_interval_seconds and process.poll() is None:
                        checkpoint = snapshot()
                        checkpoints.append(checkpoint)
                        last_check = now
                        if progress_callback is not None:
                            try:
                                decision = progress_callback(checkpoint)
                                action = getattr(decision, "action", None)
                                if action is None and isinstance(decision, dict):
                                    action = decision.get("action")
                                checkpoint["monitor_action"] = action or "continue"
                                checkpoint["monitor_reason"] = (
                                    getattr(decision, "reason", None)
                                    or (decision.get("reason") if isinstance(decision, dict) else None)
                                    or "no reason supplied"
                                )
                                if action in {"terminate", "stop", "abandon"}:
                                    termination_reason = (
                                        getattr(decision, "reason", None)
                                        or (decision.get("reason") if isinstance(decision, dict) else None)
                                        or "execution monitor requested termination"
                                    )
                                    process.terminate()
                                    try:
                                        process.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                                    break
                            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                                # Monitor failure must not kill healthy work.
                                checkpoint["monitor_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                final = snapshot()
                final["process_running"] = False
                final["final"] = True
                checkpoints.append(final)
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                return process.returncode or 0, timed_out, stdout, stderr, checkpoints, termination_reason

    @staticmethod
    def _read_metric(path: Path, metric_name: str) -> float | None:
        return Executor._read_metrics(path).get(metric_name)

    @staticmethod
    def _read_metrics(path: Path) -> dict[str, float]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            metrics: dict[str, float] = {}
            for key, raw in value.items():
                if isinstance(raw, bool):
                    continue
                try:
                    metrics[str(key)] = float(raw)
                except (TypeError, ValueError):
                    continue
            return metrics
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
