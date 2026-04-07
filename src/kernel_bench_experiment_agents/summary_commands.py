from __future__ import annotations

import argparse
import json
import math
from typing import Any

from .common import as_float, emit_json
from .kernelbench import load_problem
from .project import artifacts_dir, write_json
from .run_metrics import baseline_mean_for_problem, candidate_runtime, load_baseline_file
from .workspace_paths import read_json_file


def parse_pass_k_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("pass@k values must be positive integers")
        values.append(value)
    return sorted(set(values))


def pass_at_k_estimate(n: int, c: int, k: int) -> float | None:
    if n <= 0 or k <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    numerator = math.comb(n - c, k)
    denominator = math.comb(n, k)
    return 1.0 - (numerator / denominator)


def command_summarize_run(args: argparse.Namespace) -> None:
    pass_k_values = parse_pass_k_list(args.pass_k)
    eager_baseline = load_baseline_file(args.eager_baseline_file)
    compile_baseline = load_baseline_file(args.compile_baseline_file)

    run_root = artifacts_dir() / args.run_name
    if not run_root.exists():
        raise SystemExit(f"No run artifacts found at {run_root}")

    selected_levels = set(args.level)
    selected_problem_ids = set(args.problem_id)

    total_samples = 0
    compiled_samples = 0
    correct_samples = 0
    token_usage_totals = {
        "turns_completed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "problems_with_usage": 0,
    }
    cost_usd_totals = {
        "total_usd": 0.0,
        "problems_with_cost": 0,
    }
    trace_count_totals = {
        "command_executions": 0,
        "file_change_events": 0,
        "wrapper_commands": 0,
        "gpu_wrapper_commands": 0,
        "problem_info_calls": 0,
        "hardware_info_calls": 0,
        "run_candidate_calls": 0,
        "profile_ncu_calls": 0,
        "goal_status_calls": 0,
        "best_result_calls": 0,
        "complete_problem_calls": 0,
        "spawn_agent_calls": 0,
        "wait_calls": 0,
        "web_search_calls": 0,
        "subagents_spawned": 0,
        "problems_with_trace_counts": 0,
    }
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

            attempts_dir = problem_dir / "attempts"
            history_path = attempts_dir / "history.jsonl"
            samples: list[dict[str, Any]] = []
            if history_path.exists():
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

            completion_payload = None
            completion_path = problem_dir / "agent" / "completion.json"
            if completion_path.exists():
                completion_payload = read_json_file(completion_path)

            if not samples and completion_payload is None:
                continue

            total_samples += len(samples)
            compiled_samples += sum(1 for sample in samples if sample["compiled"])
            correct_samples += sum(1 for sample in samples if sample["correct"])

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
            problem_name = contract_problem.get("problem_name") if isinstance(contract_problem, dict) else None

            if not problem_name and eager_baseline is not None and compile_baseline is not None:
                problem = load_problem(
                    level=level,
                    problem_id=problem_id,
                    dataset_src=args.dataset_src,
                    explicit_kernelbench_root=args.kernelbench_root,
                )
                problem_name = problem.name

            eager_mean = None
            compile_mean = None
            if isinstance(contract_baseline, dict):
                eager_mean = as_float(contract_baseline.get("eager", {}).get("runtime_ms"))
                compile_mean = as_float(contract_baseline.get("compile", {}).get("runtime_ms"))
            if eager_mean is None and eager_baseline is not None and problem_name is not None:
                eager_mean = baseline_mean_for_problem(
                    baseline=eager_baseline,
                    level=level,
                    problem_name=problem_name,
                )
            if compile_mean is None and compile_baseline is not None and problem_name is not None:
                compile_mean = baseline_mean_for_problem(
                    baseline=compile_baseline,
                    level=level,
                    problem_name=problem_name,
                )

            row_token_usage = (
                completion_payload.get("token_usage")
                if isinstance(completion_payload, dict)
                else None
            )
            audit_payload = (
                completion_payload.get("audit")
                if isinstance(completion_payload, dict)
                else None
            )
            row_trace_counts = (
                completion_payload.get("trace_counts")
                if isinstance(completion_payload, dict)
                else None
            )
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
            if isinstance(row_token_usage, dict):
                token_usage_totals["problems_with_usage"] += 1
                for key in (
                    "turns_completed",
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_creation_input_tokens",
                    "uncached_input_tokens",
                    "output_tokens",
                ):
                    token_usage_totals[key] += int(
                        as_float(row_token_usage.get(key)) or 0
                    )
            if isinstance(row_trace_counts, dict):
                trace_count_totals["problems_with_trace_counts"] += 1
                for key in (
                    "command_executions",
                    "file_change_events",
                    "wrapper_commands",
                    "gpu_wrapper_commands",
                    "problem_info_calls",
                    "hardware_info_calls",
                    "run_candidate_calls",
                    "profile_ncu_calls",
                    "goal_status_calls",
                    "best_result_calls",
                    "complete_problem_calls",
                    "spawn_agent_calls",
                    "wait_calls",
                    "web_search_calls",
                    "subagents_spawned",
                ):
                    trace_count_totals[key] += int(
                        as_float(row_trace_counts.get(key)) or 0
                    )
            if row_cost_usd is not None:
                cost_usd_totals["problems_with_cost"] += 1
                cost_usd_totals["total_usd"] += float(row_cost_usd)
            problem_rows.append(
                {
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
            )

    total_problems = len(problem_rows)
    audit_invalid_problems = sum(1 for row in problem_rows if not row["audit_valid"])
    problems_with_compiled = sum(1 for row in problem_rows if row["compiled_samples"] > 0)
    problems_with_correct = sum(1 for row in problem_rows if row["effective_correct_samples"] > 0)

    eager_comparable = [
        row for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["eager_baseline_ms"] is not None
    ]
    compile_comparable = [
        row for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["compile_baseline_ms"] is not None
    ]
    terminal_states: dict[str, int] = {}
    solver_states: dict[str, int] = {}
    measured_outcomes: dict[str, int] = {}
    for row in problem_rows:
        terminal_state = row.get("terminal_state")
        if terminal_state:
            terminal_states[terminal_state] = terminal_states.get(terminal_state, 0) + 1
        solver_state = row.get("solver_state")
        if solver_state:
            solver_states[solver_state] = solver_states.get(solver_state, 0) + 1
        measured_outcome = row.get("measured_outcome")
        if measured_outcome:
            measured_outcomes[measured_outcome] = measured_outcomes.get(measured_outcome, 0) + 1

    pass_at_k: dict[str, Any] = {}
    for k in pass_k_values:
        estimates = []
        eligible = 0
        for row in problem_rows:
            estimate = pass_at_k_estimate(
                row["num_samples"],
                row["effective_correct_samples"],
                k,
            )
            if estimate is None:
                continue
            eligible += 1
            estimates.append(estimate)
        pass_at_k[str(k)] = {
            "eligible_problems": eligible,
            "average": (sum(estimates) / eligible) if eligible else None,
        }

    payload = {
        "run_name": args.run_name,
        "levels_filter": sorted(selected_levels),
        "problem_ids_filter": sorted(selected_problem_ids),
        "total_problems": total_problems,
        "audit_invalid_problems": audit_invalid_problems,
        "total_samples": total_samples,
        "compiled_samples": compiled_samples,
        "correct_samples": correct_samples,
        "effective_correct_samples": sum(
            row["effective_correct_samples"] for row in problem_rows
        ),
        "compiled_sample_rate": (
            compiled_samples / total_samples if total_samples else None
        ),
        "correct_sample_rate": (
            correct_samples / total_samples if total_samples else None
        ),
        "effective_correct_sample_rate": (
            sum(row["effective_correct_samples"] for row in problem_rows) / total_samples
            if total_samples
            else None
        ),
        "problem_compile_hit_rate": (
            problems_with_compiled / total_problems if total_problems else None
        ),
        "problem_correct_hit_rate": (
            problems_with_correct / total_problems if total_problems else None
        ),
        "terminal_states": terminal_states,
        "solver_states": solver_states,
        "measured_outcomes": measured_outcomes,
        "cost_usd": {
            "total_usd": cost_usd_totals["total_usd"],
            "problems_with_cost": cost_usd_totals["problems_with_cost"],
            "average_per_problem_usd": (
                cost_usd_totals["total_usd"] / cost_usd_totals["problems_with_cost"]
                if cost_usd_totals["problems_with_cost"]
                else None
            ),
        },
        "token_usage": token_usage_totals,
        "trace_counts": trace_count_totals,
        "beats_eager": {
            "eligible_problems": len(eager_comparable),
            "count": sum(1 for row in eager_comparable if row["beats_eager"]),
            "rate": (
                sum(1 for row in eager_comparable if row["beats_eager"]) / len(eager_comparable)
                if eager_comparable
                else None
            ),
        },
        "beats_compile": {
            "eligible_problems": len(compile_comparable),
            "count": sum(1 for row in compile_comparable if row["beats_compile"]),
            "rate": (
                sum(1 for row in compile_comparable if row["beats_compile"]) / len(compile_comparable)
                if compile_comparable
                else None
            ),
        },
        "pass_at_k": pass_at_k,
        "problems": [
            {
                key: value
                for key, value in row.items()
                if key != "samples"
            }
            for row in problem_rows
        ],
    }
    write_json(run_root / "run_summary.json", payload)
    emit_json(payload)
