"""Small, deterministic repository map used to assemble Coding Agent context."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
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
            if any(part in IGNORED_PARTS for part in path.parts):
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
        self, query: str, limit: int = 6, max_chars_each: int = 5000
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
        result = []
        for _, entry in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]:
            full_path = self.root / entry.path
            result.append(
                {
                    "path": entry.path,
                    "content": full_path.read_text(encoding="utf-8", errors="replace")[
                        :max_chars_each
                    ],
                    "symbols": ", ".join(entry.symbols),
                }
            )
        return result
