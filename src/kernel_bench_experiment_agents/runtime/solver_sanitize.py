"""Sanitize solver-facing text while preserving raw archive artifacts.

Measured commands keep durable archive files useful for maintainers, but anything
returned to the solver or mirrored into the live workspace should not expose host
filesystem layout. This module is the shared backstop for those solver-facing
surfaces.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.runtime.gpu_pool import discover_cuda_home


@dataclass(frozen=True)
class PathReplacement:
    path: Path | None
    replacement: str


def _resolved(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve(strict=False)


def _common_root(paths: tuple[Path | None, ...]) -> Path | None:
    candidates = [path.resolve(strict=False) for path in paths if path is not None]
    if len(candidates) < 2:
        return None
    try:
        root = Path(os.path.commonpath([str(path) for path in candidates]))
    except ValueError:
        return None
    if str(root) in {"", "/"}:
        return None
    return root


def _optional_env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return _resolved(raw)


def _cuda_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("CUDA_HOME", "").strip(),
        os.environ.get("CUDA_PATH", "").strip(),
        discover_cuda_home() or "",
    ):
        if raw:
            roots.append(Path(raw).expanduser().resolve(strict=False))
    return tuple(roots)


def _default_replacements(
    *,
    workspace: Path | None,
    problem_archive_root: Path | None,
    extra_roots: tuple[Path | None, ...],
) -> list[PathReplacement]:
    cwd = Path.cwd().resolve(strict=False)
    kernelbench_root = _optional_env_path("KERNELBENCH_ROOT")
    data_root = _optional_env_path("DATA_ROOT")
    common_root = _common_root(
        tuple(path for path in (workspace, problem_archive_root, data_root, *extra_roots))
    )
    python_roots = tuple(
        Path(value).expanduser().resolve(strict=False)
        for value in (sys.prefix, sys.base_prefix)
        if value
    )

    replacements: list[PathReplacement] = [
        PathReplacement(workspace, "."),
        PathReplacement(problem_archive_root, "archive/current_problem"),
        PathReplacement(cwd, "<harness>"),
        PathReplacement(kernelbench_root, "<kernelbench>"),
        *(PathReplacement(root, "<cuda>") for root in _cuda_roots()),
        *(PathReplacement(root, "<python>") for root in python_roots),
        PathReplacement(Path.home().resolve(strict=False), "<home>"),
        PathReplacement(data_root, "<harness-data>"),
        PathReplacement(common_root, "<harness-data>"),
    ]
    return replacements


def _ordered_replacements(
    *,
    workspace: Path | None,
    problem_archive_root: Path | None,
    extra_paths: tuple[PathReplacement, ...],
    extra_roots: tuple[Path | None, ...],
) -> list[PathReplacement]:
    replacements = [
        *extra_paths,
        *_default_replacements(
            workspace=workspace,
            problem_archive_root=problem_archive_root,
            extra_roots=extra_roots,
        ),
    ]
    deduped: dict[str, PathReplacement] = {}
    for replacement in replacements:
        path = _resolved(replacement.path)
        if path is None:
            continue
        raw = str(path)
        if not raw or raw == "/":
            continue
        deduped.setdefault(raw, PathReplacement(path, replacement.replacement))
    return sorted(deduped.values(), key=lambda item: len(str(item.path)), reverse=True)


def sanitize_solver_text(
    text: str,
    *,
    workspace: Path | None = None,
    problem_archive_root: Path | None = None,
    extra_paths: tuple[PathReplacement, ...] = (),
    extra_roots: tuple[Path | None, ...] = (),
) -> str:
    """Replace host-specific paths in text with stable solver-facing labels."""
    if not text:
        return text
    sanitized = text
    for item in _ordered_replacements(
        workspace=_resolved(workspace),
        problem_archive_root=_resolved(problem_archive_root),
        extra_paths=extra_paths,
        extra_roots=extra_roots,
    ):
        if item.path is not None:
            sanitized = sanitized.replace(str(item.path), item.replacement)
    return sanitized


def sanitize_solver_value(
    value: Any,
    *,
    workspace: Path | None = None,
    problem_archive_root: Path | None = None,
    extra_paths: tuple[PathReplacement, ...] = (),
    extra_roots: tuple[Path | None, ...] = (),
) -> Any:
    """Recursively sanitize strings in JSON-like values."""
    if isinstance(value, str):
        return sanitize_solver_text(
            value,
            workspace=workspace,
            problem_archive_root=problem_archive_root,
            extra_paths=extra_paths,
            extra_roots=extra_roots,
        )
    if isinstance(value, dict):
        return {
            key: sanitize_solver_value(
                nested,
                workspace=workspace,
                problem_archive_root=problem_archive_root,
                extra_paths=extra_paths,
                extra_roots=extra_roots,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_solver_value(
                nested,
                workspace=workspace,
                problem_archive_root=problem_archive_root,
                extra_paths=extra_paths,
                extra_roots=extra_roots,
            )
            for nested in value
        ]
    return value
