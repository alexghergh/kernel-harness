from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import as_float
from .kernelbench import load_problem
from .run_metrics import baseline_mean_for_problem, candidate_runtime
from .workspace_paths import read_json_file


@dataclass(frozen=True)
class SummaryScanConfig:
    dataset_src: str
    kernelbench_root: str | None
    eager_baseline: dict[str, Any] | None
    compile_baseline: dict[str, Any] | None


def _load_samples(problem_dir: Path) -> list[dict[str, Any]]:
    attempts_dir = problem_dir / "attempts"
    history_path = attempts_dir / "history.jsonl"
    samples: list[dict[str, Any]] = []
    if not history_path.exists():
        return samples
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
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


def _problem_name_for_row(
    *,
    level: int,
    problem_id: int,
    contract_problem: dict[str, Any],
    config: SummaryScanConfig,
) -> str | None:
    problem_name = contract_problem.get("problem_name") if isinstance(contract_problem, dict) else None
    if problem_name:
        return problem_name
    if config.eager_baseline is None and config.compile_baseline is None:
        return None
    problem = load_problem(
        level=level,
        problem_id=problem_id,
        dataset_src=config.dataset_src,
        explicit_kernelbench_root=config.kernelbench_root,
    )
    return problem.name


def _baseline_means(
    *,
    level: int,
    problem_name: str | None,
    contract_baseline: dict[str, Any],
    config: SummaryScanConfig,
) -> tuple[float | None, float | None]:
    eager_mean = None
    compile_mean = None
    if isinstance(contract_baseline, dict):
        eager_mean = as_float(contract_baseline.get("eager", {}).get("runtime_ms"))
        compile_mean = as_float(contract_baseline.get("compile", {}).get("runtime_ms"))
    if eager_mean is None and config.eager_baseline is not None and problem_name is not None:
        eager_mean = baseline_mean_for_problem(
            baseline=config.eager_baseline,
            level=level,
            problem_name=problem_name,
        )
    if compile_mean is None and config.compile_baseline is not None and problem_name is not None:
        compile_mean = baseline_mean_for_problem(
            baseline=config.compile_baseline,
            level=level,
            problem_name=problem_name,
        )
    return eager_mean, compile_mean


def build_problem_row(
    *,
    problem_dir: Path,
    level: int,
    problem_id: int,
    config: SummaryScanConfig,
) -> dict[str, Any] | None:
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
    contract_baseline_path = problem_dir / "contract" / "baseline.json"
    contract_problem = read_json_file(contract_problem_path) if contract_problem_path.exists() else {}
    contract_baseline = read_json_file(contract_baseline_path) if contract_baseline_path.exists() else {}
    problem_name = _problem_name_for_row(
        level=level,
        problem_id=problem_id,
        contract_problem=contract_problem,
        config=config,
    )
    eager_mean, compile_mean = _baseline_means(
        level=level,
        problem_name=problem_name,
        contract_baseline=contract_baseline,
        config=config,
    )

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


def collect_problem_rows(
    *,
    run_root: Path,
    selected_levels: set[int],
    selected_problem_ids: set[int],
    config: SummaryScanConfig,
) -> list[dict[str, Any]]:
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
            row = build_problem_row(
                problem_dir=problem_dir,
                level=level,
                problem_id=problem_id,
                config=config,
            )
            if row is not None:
                problem_rows.append(row)
    return problem_rows
