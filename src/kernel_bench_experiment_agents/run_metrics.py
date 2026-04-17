"""Summarize measured candidate attempts against the archived eager and compile baselines.

Goal status, completion payloads, and run summaries all use these helpers when they need best-result comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import as_float

EAGER_BASELINE_FILENAME = "baseline_time_torch.json"
COMPILE_BASELINE_FILENAME = "baseline_time_torch_compile_inductor_default.json"


def candidate_runtime(result: dict[str, Any]) -> float | None:
    runtime = as_float(result.get("runtime"))
    if runtime is not None:
        return runtime

    runtime_stats = result.get("runtime_stats")
    if isinstance(runtime_stats, dict):
        for key in ("mean", "mean_runtime_ms", "runtime_ms"):
            value = as_float(runtime_stats.get(key))
            if value is not None:
                return value

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for key in ("runtime_ms", "mean_runtime_ms"):
            value = as_float(metadata.get(key))
            if value is not None:
                return value
    return None


def timings_dir_for_hardware(*, kernelbench_root: str, timings_dir: str | None, hardware_name: str) -> Path:
    if timings_dir:
        path = Path(timings_dir).expanduser().resolve()
    else:
        path = (
            Path(kernelbench_root).expanduser().resolve()
            / "results"
            / "timing"
            / hardware_name
        )
    if not path.exists():
        raise RuntimeError(
            f"KernelBench timings directory does not exist: {path}. "
            "Set KERNELBENCH_TIMINGS_DIR explicitly or choose a hardware name that matches an existing timings subdirectory."
        )
    return path


def baseline_file_paths(*, kernelbench_root: str, timings_dir: str | None, hardware_name: str) -> tuple[Path, Path]:
    root = timings_dir_for_hardware(
        kernelbench_root=kernelbench_root,
        timings_dir=timings_dir,
        hardware_name=hardware_name,
    )
    eager_path = root / EAGER_BASELINE_FILENAME
    compile_path = root / COMPILE_BASELINE_FILENAME
    missing = [str(path) for path in (eager_path, compile_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing expected KernelBench baseline file(s): "
            + ", ".join(missing)
        )
    return eager_path, compile_path


def load_baseline_file(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_mean_for_problem(
    *,
    baseline: dict[str, Any] | None,
    level: int,
    problem_name: str | None,
) -> float | None:
    if baseline is None or not problem_name:
        return None
    level_entry = baseline.get(f"level{level}")
    if not isinstance(level_entry, dict):
        return None
    problem_entry = level_entry.get(problem_name)
    if isinstance(problem_entry, dict):
        for key in ("mean", "runtime", "runtime_ms"):
            value = as_float(problem_entry.get(key))
            if value is not None:
                return value
    return None


def baseline_payload_for_problem(
    *,
    level: int,
    problem_id: int,
    problem_name: str,
    eager_baseline_file: str | Path,
    compile_baseline_file: str | Path,
) -> dict[str, Any]:
    eager_payload = load_baseline_file(eager_baseline_file)
    compile_payload = load_baseline_file(compile_baseline_file)
    eager_runtime_ms = baseline_mean_for_problem(
        baseline=eager_payload,
        level=level,
        problem_name=problem_name,
    )
    compile_runtime_ms = baseline_mean_for_problem(
        baseline=compile_payload,
        level=level,
        problem_name=problem_name,
    )
    if eager_runtime_ms is None:
        raise RuntimeError(
            f"Problem {problem_name!r} was not found in eager baseline file {eager_baseline_file}"
        )
    if compile_runtime_ms is None:
        raise RuntimeError(
            f"Problem {problem_name!r} was not found in compile baseline file {compile_baseline_file}"
        )
    return {
        "level": level,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "eager": {
            "runtime_ms": eager_runtime_ms,
        },
        "compile": {
            "runtime_ms": compile_runtime_ms,
        },
    }


def payload_counts_toward_progress(payload: dict[str, Any]) -> bool:
    value = payload.get("counts_toward_progress")
    if value is None:
        return True
    return bool(value)


def best_correct_payload(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_payload: dict[str, Any] | None = None
    best_runtime: float | None = None
    for payload in entries:
        if not payload_counts_toward_progress(payload):
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        if not result.get("correctness"):
            continue
        runtime = candidate_runtime(result)
        if runtime is None:
            continue
        if best_runtime is None or runtime < best_runtime:
            best_runtime = runtime
            best_payload = payload
    return best_payload


def sum_numeric_field(entries: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in entries:
        value = as_float(payload.get(key))
        if value is not None:
            total += value
    return total
