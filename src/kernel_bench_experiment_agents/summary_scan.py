"""Scan archived problem directories and collect the raw inputs needed for run-level summaries.

The reporting layer depends on these helpers to traverse the archive without touching live workspace state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import as_float
from .run_metrics import candidate_runtime
from .workspace_paths import read_json_file


def _load_samples(problem_dir: Path) -> list[dict[str, Any]]:
    attempts_dir = problem_dir / "attempts"
    samples: list[dict[str, Any]] = []
    for manifest_path in sorted(
        attempts_dir.glob("sample_*.json"),
        key=lambda path: int(path.stem.split("_", 1)[1]),
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        samples.append(
            {
                "sample_id": payload.get("sample_id"),
                "status": payload.get("status"),
                "compiled": bool(result.get("compiled")),
                "correct": bool(result.get("correctness")),
                "runtime_ms": candidate_runtime(result),
            }
        )
    return samples


def _load_completion(problem_dir: Path) -> dict[str, Any] | None:
    completion_path = problem_dir / "agent" / "completion.json"
    if completion_path.exists():
        return read_json_file(completion_path)
    return None


def _baseline_means(contract_problem: dict[str, Any]) -> tuple[float | None, float | None]:
    eager_mean = None
    compile_mean = None
    if isinstance(contract_problem, dict):
        eager_mean = as_float(contract_problem.get("baseline_runtime_ms", {}).get("eager"))
    if compile_mean is None and isinstance(contract_problem, dict):
        compile_mean = as_float(contract_problem.get("baseline_runtime_ms", {}).get("compile"))
    return eager_mean, compile_mean


def build_problem_row(*, problem_dir: Path, level: int, problem_id: int) -> dict[str, Any] | None:
    samples = _load_samples(problem_dir)
    completion_payload = _load_completion(problem_dir)
    if not samples and completion_payload is None:
        return None

    best_correct_runtime = min(
        (
            sample["runtime_ms"]
            for sample in samples
            if sample["correct"] and sample["runtime_ms"] is not None
        ),
        default=None,
    )

    contract_problem_path = problem_dir / "contract" / "problem.json"
    contract_problem = read_json_file(contract_problem_path) if contract_problem_path.exists() else {}
    problem_name = contract_problem.get("problem_name") if isinstance(contract_problem, dict) else None
    eager_mean, compile_mean = _baseline_means(contract_problem)

    row_token_usage = completion_payload.get("token_usage") if isinstance(completion_payload, dict) else None
    audit_payload = completion_payload.get("audit") if isinstance(completion_payload, dict) else None
    row_trace_counts = completion_payload.get("trace_counts") if isinstance(completion_payload, dict) else None
    row_cost_usd = (
        as_float(completion_payload.get("cost_usd"))
        if isinstance(completion_payload, dict)
        else None
    )
    audit_valid = (
        bool(audit_payload.get("valid"))
        if isinstance(audit_payload, dict) and "valid" in audit_payload
        else True
    )
    effective_correct_samples = (
        sum(1 for sample in samples if sample["correct"])
        if audit_valid
        else 0
    )
    effective_best_correct_runtime = best_correct_runtime if audit_valid else None
    return {
        "level": level,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "num_samples": len(samples),
        "compiled_samples": sum(1 for sample in samples if sample["compiled"]),
        "correct_samples": sum(1 for sample in samples if sample["correct"]),
        "effective_correct_samples": effective_correct_samples,
        "best_correct_runtime_ms": effective_best_correct_runtime,
        "raw_best_correct_runtime_ms": best_correct_runtime,
        "raw_beats_eager": (
            best_correct_runtime is not None
            and eager_mean is not None
            and best_correct_runtime < eager_mean
        ),
        "raw_beats_compile": (
            best_correct_runtime is not None
            and compile_mean is not None
            and best_correct_runtime < compile_mean
        ),
        "raw_beats_both": (
            best_correct_runtime is not None
            and eager_mean is not None
            and compile_mean is not None
            and best_correct_runtime < eager_mean
            and best_correct_runtime < compile_mean
        ),
        "eager_baseline_ms": eager_mean,
        "compile_baseline_ms": compile_mean,
        "beats_eager": (
            effective_best_correct_runtime is not None
            and eager_mean is not None
            and effective_best_correct_runtime < eager_mean
        ),
        "beats_compile": (
            effective_best_correct_runtime is not None
            and compile_mean is not None
            and effective_best_correct_runtime < compile_mean
        ),
        "solver_state": (
            completion_payload.get("solver_state")
            if completion_payload is not None
            else None
        ),
        "terminal_state": (
            completion_payload.get("terminal_state")
            if completion_payload is not None
            else None
        ),
        "measured_outcome": (
            completion_payload.get("measured_outcome")
            if completion_payload is not None
            else None
        ),
        "completion_success": (
            bool(completion_payload.get("success"))
            if completion_payload is not None
            else None
        ),
        "tool": (
            completion_payload.get("tool")
            if completion_payload is not None
            else None
        ),
        "audit_valid": audit_valid,
        "audit": audit_payload,
        "cost_usd": row_cost_usd,
        "token_usage": row_token_usage,
        "trace_counts": row_trace_counts,
        "samples": samples,
    }


def collect_problem_rows(*, run_root: Path, selected_levels: set[int], selected_problem_ids: set[int]) -> list[dict[str, Any]]:
    problem_rows: list[dict[str, Any]] = []
    for level_dir in sorted(run_root.glob("level_*")):
        try:
            level = int(level_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if selected_levels and level not in selected_levels:
            continue

        for problem_dir in sorted(level_dir.glob("problem_*")):
            try:
                problem_id = int(problem_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if selected_problem_ids and problem_id not in selected_problem_ids:
                continue
            row = build_problem_row(problem_dir=problem_dir, level=level, problem_id=problem_id)
            if row is not None:
                problem_rows.append(row)
    return problem_rows
