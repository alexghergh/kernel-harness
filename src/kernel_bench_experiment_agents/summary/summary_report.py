"""Assemble the final per-run summary payload from scanned archived problem results.

This module sits above summary scanning and math helpers to produce the durable run-level report written into archive/.
"""

from __future__ import annotations

from typing import Any

from kernel_bench_experiment_agents.runtime.common import as_float
from kernel_bench_experiment_agents.summary.summary_math import pass_at_k_estimate

TOKEN_USAGE_KEYS = (
    "turns_completed",
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
)

TRACE_COUNT_KEYS = (
    "command_executions",
    "file_change_events",
    "wrapper_commands",
    "gpu_wrapper_commands",
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
)


def _token_usage_totals(problem_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "turns_completed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "problems_with_usage": 0,
    }
    for row in problem_rows:
        row_token_usage = row.get("token_usage")
        if not isinstance(row_token_usage, dict):
            continue
        totals["problems_with_usage"] += 1
        for key in TOKEN_USAGE_KEYS:
            totals[key] += int(as_float(row_token_usage.get(key)) or 0)
    return totals


def _trace_count_totals(problem_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in TRACE_COUNT_KEYS}
    totals["problems_with_trace_counts"] = 0
    for row in problem_rows:
        row_trace_counts = row.get("trace_counts")
        if not isinstance(row_trace_counts, dict):
            continue
        totals["problems_with_trace_counts"] += 1
        for key in TRACE_COUNT_KEYS:
            totals[key] += int(as_float(row_trace_counts.get(key)) or 0)
    return totals


def _cost_totals(problem_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_usd = 0.0
    problems_with_cost = 0
    for row in problem_rows:
        cost_usd = as_float(row.get("cost_usd"))
        if cost_usd is None:
            continue
        problems_with_cost += 1
        total_usd += float(cost_usd)
    return {
        "total_usd": total_usd,
        "problems_with_cost": problems_with_cost,
        "average_per_problem_usd": (
            total_usd / problems_with_cost if problems_with_cost else None
        ),
    }


def _state_counts(problem_rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in problem_rows:
        value = row.get(key)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _pass_at_k_summary(problem_rows: list[dict[str, Any]], pass_k_values: list[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
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
        payload[str(k)] = {
            "eligible_problems": eligible,
            "average": (sum(estimates) / eligible) if eligible else None,
        }
    return payload


def build_run_summary_payload(
    *,
    run_name: str,
    selected_levels: set[int],
    selected_problem_ids: set[int],
    pass_k_values: list[int],
    problem_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_problems = len(problem_rows)
    total_samples = sum(row["num_samples"] for row in problem_rows)
    compiled_samples = sum(row["compiled_samples"] for row in problem_rows)
    correct_samples = sum(row["correct_samples"] for row in problem_rows)
    effective_correct_samples = sum(row["effective_correct_samples"] for row in problem_rows)
    audit_invalid_problems = sum(1 for row in problem_rows if not row["audit_valid"])
    problems_with_compiled = sum(1 for row in problem_rows if row["compiled_samples"] > 0)
    problems_with_correct = sum(1 for row in problem_rows if row["effective_correct_samples"] > 0)

    eager_comparable = [
        row
        for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["eager_baseline_ms"] is not None
    ]
    compile_comparable = [
        row
        for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["compile_baseline_ms"] is not None
    ]

    return {
        "run_name": run_name,
        "levels_filter": sorted(selected_levels),
        "problem_ids_filter": sorted(selected_problem_ids),
        "total_problems": total_problems,
        "audit_invalid_problems": audit_invalid_problems,
        "total_samples": total_samples,
        "compiled_samples": compiled_samples,
        "correct_samples": correct_samples,
        "effective_correct_samples": effective_correct_samples,
        "compiled_sample_rate": (compiled_samples / total_samples if total_samples else None),
        "correct_sample_rate": (correct_samples / total_samples if total_samples else None),
        "effective_correct_sample_rate": (
            effective_correct_samples / total_samples if total_samples else None
        ),
        "problem_compile_hit_rate": (
            problems_with_compiled / total_problems if total_problems else None
        ),
        "problem_correct_hit_rate": (
            problems_with_correct / total_problems if total_problems else None
        ),
        "terminal_states": _state_counts(problem_rows, "terminal_state"),
        "solver_states": _state_counts(problem_rows, "solver_state"),
        "measured_outcomes": _state_counts(problem_rows, "measured_outcome"),
        "cost_usd": _cost_totals(problem_rows),
        "token_usage": _token_usage_totals(problem_rows),
        "trace_counts": _trace_count_totals(problem_rows),
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
        "pass_at_k": _pass_at_k_summary(problem_rows, pass_k_values),
        "problems": [
            {key: value for key, value in row.items() if key != "samples"}
            for row in problem_rows
        ],
    }
