"""Creates isolated workspaces and applies Coding Agent file writes safely."""

from __future__ import annotations

import ast
import difflib
import shutil
import uuid
from fnmatch import fnmatch
from pathlib import Path

from .schemas import ChangeRequestArtifact, CodePlanArtifact, TaskContract


class WorkspaceManager:
    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create(self, template: str | Path) -> Path:
        template_path = Path(template).resolve()
        if not template_path.is_dir():
            raise FileNotFoundError(
                f"workspace_template is not a directory: {template_path}"
            )
        run_path = self.runs_root / uuid.uuid4().hex
        shutil.copytree(
            template_path,
            run_path,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "runs"),
        )
        return run_path

    def apply_plan(
        self,
        workspace: Path,
        plan: CodePlanArtifact,
        contract: TaskContract,
        change: ChangeRequestArtifact,
    ) -> str:
        diffs: list[str] = []
        original: dict[Path, str] = {}
        staged: dict[Path, str] = {}
        for edit in plan.edits:
            target = self._safe_target(workspace, edit.path, contract)
            if change.allowed_files and edit.path not in change.allowed_files:
                raise PermissionError(
                    f"edit not allowed by ChangeRequest: {edit.path}"
                )
            if contract.editable_paths and not any(
                fnmatch(edit.path, pattern) for pattern in contract.editable_paths
            ):
                raise PermissionError(f"edit outside contract editable_paths: {edit.path}")
            if (
                change.mutation_class.value == "configuration"
                and target.suffix == ".py"
            ):
                raise PermissionError(
                    "configuration ChangeRequest cannot edit Python source; request implementation evolution instead"
                )
            if target not in original:
                original[target] = target.read_text(encoding="utf-8") if target.exists() else ""
                staged[target] = original[target]
            old = staged[target]
            if edit.operation == "create":
                if target.exists() or (target in staged and staged[target] != original[target]):
                    raise ValueError(
                        f"create edit requires a new file, but {edit.path} already exists"
                    )
                new = edit.new_text
            elif edit.operation == "replace":
                if not target.exists() and not staged[target]:
                    raise ValueError(
                        f"replace edit requires an existing file: {edit.path}"
                    )
                assert edit.old_text is not None
                matches = old.count(edit.old_text)
                if matches != 1:
                    raise ValueError(
                        f"replace edit must match exactly once in {edit.path}; found {matches} matches"
                    )
                if len(old) > 400 and edit.old_text == old:
                    raise ValueError(
                        f"replace edit rewrites all of {edit.path}; return local edits instead"
                    )
                new = old.replace(edit.old_text, edit.new_text, 1)
            elif edit.operation == "replace_symbol":
                if not target.exists() and not staged[target]:
                    raise ValueError(f"replace_symbol edit requires an existing file: {edit.path}")
                new = self._replace_symbol(old, edit.symbol or "", edit.new_text, edit.path)
            else:  # rewrite
                if edit.path != contract.solution_entrypoint:
                    raise PermissionError("rewrite edits are limited to the solution entrypoint")
                if change.parent_variant_id is not None or change.mutation_class.value == "recovery":
                    raise ValueError(
                        "rewrite is not allowed for a child or repair candidate; "
                        "use a unique replace or replace_symbol edit for a local patch"
                    )
                if not target.exists() and not staged[target]:
                    raise ValueError(f"rewrite edit requires an existing file: {edit.path}")
                new = edit.new_text
            staged[target] = new

        for target, new in staged.items():
            if target.suffix == ".py":
                self._validate_python(new, target, contract)
        for target, new in staged.items():
            old = original[target]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new, encoding="utf-8")
            rel = target.relative_to(workspace).as_posix()
            diffs.extend(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
        diff = "".join(diffs)
        (workspace / ".methodtrail.diff").write_text(diff, encoding="utf-8")
        return diff

    @staticmethod
    def _replace_symbol(source: str, symbol: str, replacement: str, path: str) -> str:
        """Replace one top-level class/function without asking for the whole file."""
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise ValueError(f"cannot edit symbol in invalid Python source: {exc}") from exc
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ]
        if len(matches) != 1:
            raise ValueError(f"symbol edit must find exactly one top-level {symbol!r}; found {len(matches)}")
        node = matches[0]
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
        end = node.end_lineno or node.lineno
        lines = source.splitlines(keepends=True)
        replacement = replacement.rstrip("\n") + "\n"
        return "".join(lines[:start] + [replacement] + lines[end:])

    @staticmethod
    def _validate_python(source: str, target: Path, contract: TaskContract) -> None:
        try:
            tree = ast.parse(source, filename=str(target))
        except SyntaxError as exc:
            raise ValueError(f"Python syntax error in {target.name}: {exc}") from exc
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate top-level definitions in {target.name}: {', '.join(duplicates)}")
        if target.name == Path(contract.solution_entrypoint).name:
            main_guards = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ]
            if len(main_guards) > 1:
                raise ValueError(f"multiple __main__ entry guards in {target.name}")

    @staticmethod
    def _safe_target(
        workspace: Path, relative_path: str, contract: TaskContract
    ) -> Path:
        target = (workspace / relative_path).resolve()
        if workspace not in target.parents and target != workspace:
            raise ValueError(f"write escapes workspace: {relative_path}")
        if relative_path in contract.protected_paths:
            raise PermissionError(
                f"protected task file cannot be edited: {relative_path}"
            )
        for data_path in contract.allowed_data_paths:
            data_root = Path(data_path)
            candidate = Path(relative_path)
            if candidate == data_root or data_root in candidate.parents:
                raise PermissionError(f"task data cannot be edited: {relative_path}")
        return target
