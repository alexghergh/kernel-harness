from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir() -> Path:
    return ensure_dir(experiment_root() / "runs")


def artifacts_dir() -> Path:
    return ensure_dir(experiment_root() / "artifacts")


def build_dir() -> Path:
    return ensure_dir(experiment_root() / "build")


def runtime_dir() -> Path:
    return ensure_dir(experiment_root() / ".runtime")


def gpu_lock_dir() -> Path:
    return ensure_dir(runtime_dir() / "gpu_locks")


def artifact_lock_dir() -> Path:
    return ensure_dir(runtime_dir() / "artifact_locks")


def _lock_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _validate_component(value: str, *, label: str) -> str:
    if not value:
        raise RuntimeError(f"{label} must not be empty")
    if Path(value).name != value:
        raise RuntimeError(
            f"{label} must be a single path component without separators: {value!r}"
        )
    if value in {".", ".."}:
        raise RuntimeError(f"{label} is not allowed: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError(
            f"{label} may contain only ASCII letters, digits, dot, underscore, and hyphen"
        )
    return value


def validate_run_name(run_name: str) -> str:
    return _validate_component(run_name, label="run_name")


def validate_sample_key(sample_key: str) -> str:
    return _validate_component(sample_key, label="sample_key")


def artifact_lock_path(run_name: str, level: int, problem_id: int) -> Path:
    run_name = validate_run_name(run_name)
    return artifact_lock_dir() / (
        f"{_lock_slug(run_name)}_level_{level}_problem_{problem_id}.lock"
    )


def workspace_root(explicit: str | None = None) -> Path:
    candidate = explicit or os.environ.get("KBE_WORKSPACE_ROOT")
    if candidate:
        return ensure_dir(Path(candidate).expanduser().resolve())
    return ensure_dir(runtime_dir() / "workspaces")


def workspace_dir(
    run_name: str,
    level: int,
    problem_id: int,
    explicit_root: str | None = None,
) -> Path:
    run_name = validate_run_name(run_name)
    return ensure_dir(
        workspace_root(explicit_root)
        / run_name
        / f"level_{level}"
        / f"problem_{problem_id}"
    )


def run_dir(run_name: str) -> Path:
    run_name = validate_run_name(run_name)
    return ensure_dir(runs_dir() / run_name)


def artifact_problem_dir(run_name: str, level: int, problem_id: int) -> Path:
    run_name = validate_run_name(run_name)
    return ensure_dir(
        artifacts_dir() / run_name / f"level_{level}" / f"problem_{problem_id}"
    )


def artifact_codex_dir(run_name: str, level: int, problem_id: int) -> Path:
    return ensure_dir(artifact_problem_dir(run_name, level, problem_id) / "codex")


def build_problem_dir(
    run_name: str,
    level: int,
    problem_id: int,
    sample_key: str,
) -> Path:
    run_name = validate_run_name(run_name)
    sample_key = validate_sample_key(sample_key)
    return ensure_dir(
        build_dir() / run_name / f"level_{level}" / f"problem_{problem_id}" / sample_key
    )


def kernelbench_root(explicit: str | None = None) -> Path:
    candidate = explicit or os.environ.get("KERNELBENCH_ROOT")
    if not candidate:
        raise RuntimeError(
            "KERNELBENCH_ROOT is not set. Point it at the official KernelBench checkout."
        )

    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"KERNELBENCH_ROOT does not exist: {path}")
    return path


def next_sample_id(run_name: str, level: int, problem_id: int) -> int:
    pattern = re.compile(
        rf"^level_{level}_problem_{problem_id}_sample_(\d+)_kernel\.py$"
    )
    max_sample = -1
    for child in run_dir(run_name).iterdir():
        match = pattern.match(child.name)
        if match:
            max_sample = max(max_sample, int(match.group(1)))
    return max_sample + 1


def official_kernel_path(run_name: str, level: int, problem_id: int, sample_id: int) -> Path:
    return run_dir(run_name) / (
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )


def official_prompt_path(run_name: str, level: int, problem_id: int, sample_id: int) -> Path:
    return run_dir(run_name) / (
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_prompt.txt"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    # callers must hold the per-problem artifact lease before appending here
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP)
