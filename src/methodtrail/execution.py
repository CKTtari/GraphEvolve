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
from pathlib import Path

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
    def run(self, workspace: Path, contract: TaskContract) -> RunArtifact:
        started = time.perf_counter()
        environment = {**os.environ, "METHODTRAIL_METRIC_FILE": contract.metric_file}
        smoke_return_code: int | None = None
        test_return_code: int | None = None
        if contract.smoke_command:
            smoke_return_code, smoke_timed_out, smoke_stdout, smoke_stderr = (
                self._run_command(
                    contract.smoke_command,
                    workspace,
                    environment,
                    min(contract.timeout_seconds, 120),
                )
            )
            if smoke_return_code != 0 or smoke_timed_out:
                return RunArtifact(
                    command=contract.run_command,
                    return_code=-5,
                    timed_out=smoke_timed_out,
                    wall_seconds=time.perf_counter() - started,
                    stdout=smoke_stdout[-12000:],
                    stderr=f"smoke command failed: {smoke_stderr}"[-12000:],
                    smoke_return_code=smoke_return_code,
                )
        if contract.test_command:
            test_return_code, test_timed_out, test_stdout, test_stderr = (
                self._run_command(
                    contract.test_command,
                    workspace,
                    environment,
                    min(contract.timeout_seconds, 300),
                )
            )
            if test_return_code != 0 or test_timed_out:
                return RunArtifact(
                    command=contract.run_command,
                    return_code=-6,
                    timed_out=test_timed_out,
                    wall_seconds=time.perf_counter() - started,
                    stdout=test_stdout[-12000:],
                    stderr=f"test command failed: {test_stderr}"[-12000:],
                    smoke_return_code=smoke_return_code,
                    test_return_code=test_return_code,
                )
        return_code, timed_out, stdout, stderr = self._run_command(
            contract.run_command,
            workspace,
            environment,
            contract.timeout_seconds,
        )
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
            ) = self._run_private_evaluation(workspace, contract, environment)
            if evaluation_timed_out:
                timed_out = True
            if evaluation_return_code != 0:
                return_code = -4
                stderr = f"{stderr}\nprivate evaluation failed: {evaluation_stderr}".strip()
        elif return_code == 0 and contract.evaluation_command:
            (
                evaluation_return_code,
                evaluation_timed_out,
                evaluation_stdout,
                evaluation_stderr,
            ) = self._run_command(
                contract.evaluation_command,
                workspace,
                environment,
                contract.timeout_seconds,
            )
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
            stdout=stdout[-12000:],
            stderr=stderr[-12000:],
            evaluation_return_code=evaluation_return_code,
            evaluation_stdout=evaluation_stdout[-12000:],
            evaluation_stderr=evaluation_stderr[-12000:],
            smoke_return_code=smoke_return_code,
            test_return_code=test_return_code,
            metric=metric,
            metrics=metrics,
            metric_constraints_passed=metric_constraints_passed,
            output_files=outputs,
        )

    def _run_private_evaluation(
        self,
        workspace: Path,
        contract: TaskContract,
        environment: dict[str, str],
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
            code, timed_out, stdout, stderr = self._run_command(
                evaluation_command,
                private_root,
                private_environment,
                contract.timeout_seconds,
            )
            # Copy only the metric back. Labels and private evaluator sources never
            # enter the candidate workspace.
            metric_source = private_root / contract.metric_file
            metric_copy = workspace / ".methodtrail.private.metrics.json"
            if metric_source.exists():
                shutil.copy2(metric_source, metric_copy)
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
        try:
            result = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            return result.returncode, False, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            return -1, True, exc.stdout or "", exc.stderr or ""

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
