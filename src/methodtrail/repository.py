"""Small, deterministic repository map used to assemble Coding Agent context."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "runs",
}


@dataclass(frozen=True)
class SourceFile:
    path: str
    symbols: tuple[str, ...]
    imports: tuple[str, ...]


class RepoMap:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.files = self._scan()

    def _scan(self) -> list[SourceFile]:
        entries: list[SourceFile] = []
        for path in self.root.rglob("*.py"):
            # Check only path components below the repository root.  Candidate
            # workspaces commonly live under a parent directory named ``runs``
            # (for example ``.../.methodtrail/.../runs/.../wt/...``); looking at
            # ``path.parts`` would incorrectly discard the entire workspace and
            # leave Coding/Repair agents with an empty repository context.
            relative = path.relative_to(self.root)
            if any(part in IGNORED_PARTS for part in relative.parts):
                continue
            entries.append(self._parse(path))
        return entries

    def _parse(self, path: Path) -> SourceFile:
        relative = str(path.relative_to(self.root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return SourceFile(relative, (), ())
        symbols = tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return SourceFile(relative, symbols, tuple(imports))

    def summary(self) -> str:
        lines = []
        for entry in self.files:
            parts = [entry.path]
            if entry.symbols:
                parts.append(f"symbols={','.join(entry.symbols[:12])}")
            if entry.imports:
                parts.append(f"imports={','.join(entry.imports[:8])}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def retrieve(
        self,
        query: str,
        limit: int = 6,
        max_chars_each: int = 5000,
        full_paths: set[str] | None = None,
        full_file_limit: int | None = None,
    ) -> list[dict[str, str]]:
        tokens = {
            token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query)
        }
        scored: list[tuple[int, SourceFile]] = []
        for entry in self.files:
            text = " ".join((entry.path, *entry.symbols, *entry.imports)).lower()
            score = sum(token in text for token in tokens)
            if score:
                scored.append((score, entry))
        if not scored:
            scored = [(0, entry) for entry in self.files]
        full_paths = full_paths or set()
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        # Editable files are the source of truth for a patch.  Always include
        # them even when their symbol names do not match the natural-language
        # query or when another file receives a higher lexical score.
        required = [
            entry
            for entry in self.files
            if any(fnmatch(entry.path, pattern) for pattern in full_paths)
        ]
        required_paths = {entry.path for entry in required}
        ordered = required + [
            entry for _, entry in ranked if entry.path not in required_paths
        ]
        result = []
        for entry in ordered[: max(limit, len(required))]:
            full_path = self.root / entry.path
            content = full_path.read_text(encoding="utf-8", errors="replace")
            include_full = any(fnmatch(entry.path, pattern) for pattern in full_paths)
            # Exact patches require the complete editable file. A former
            # size cutoff silently returned its prefix precisely when the
            # implementation grew large enough to need careful repairs.
            if include_full:
                excerpt = content
            else:
                excerpt = content[:max_chars_each]
            result.append(
                {
                    "path": entry.path,
                    "content": excerpt,
                    "symbols": ", ".join(entry.symbols),
                    "complete": str(include_full or len(content) <= max_chars_each).lower(),
                }
            )
        return result
