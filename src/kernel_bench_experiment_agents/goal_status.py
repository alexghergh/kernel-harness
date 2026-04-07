from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive_layout import (
    goal_status_archive_path,
    history_path,
    history_entries,
    profile_entries,
    profile_index_path,
    trace_events_path,
)
from .common import as_float
from .project import now_iso, write_json, write_text
from .run_metrics import best_correct_payload, candidate_runtime, sum_numeric_field
from .trace_analysis import trace_counts, web_searches_from_ir
from .trace_ir import load_trace_event_entries, materialize_trace_ir
from .workspace_paths import (
    load_workspace_baseline,
    load_workspace_metadata,
    write_workspace_best_sample,
)


def _elapsed_minutes_since(started_at: Any) -> float | None:
    if not isinstance(started_at, str) or not started_at.strip():
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (now - started.astimezone(timezone.utc)).total_seconds() / 60.0)


def live_trace_counts_for_problem(
    run_name: str,
    level: int,
    problem_id: int,
    *,
    tool: str = "codex",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_events, raw_event_entries = load_trace_event_entries(
        trace_events_path(run_name, level, problem_id)
    )
    ir_events = materialize_trace_ir(raw_event_entries, tool=tool)
    return (
        trace_counts(ir_events, raw_events=raw_events, tool=tool),
        web_searches_from_ir(ir_events),
    )


def goal_status_snapshot(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    workspace: Path,
) -> dict[str, Any]:
    metadata = load_workspace_metadata(workspace)
    baseline = load_workspace_baseline(workspace)
    history_path_value = history_path(run_name, level, problem_id)
    profile_index = profile_index_path(run_name, level, problem_id)
    entries = history_entries(history_path_value)
    profiles = profile_entries(profile_index)
    best_payload = best_correct_payload(history_path_value)
    best_runtime_ms = None
    best_sample_id = None
    best_kernel_path = None
    if best_payload is not None:
        result = best_payload.get("result")
        if isinstance(result, dict):
            best_runtime_ms = candidate_runtime(result)
        best_sample_id = best_payload.get("sample_id")
        best_kernel_path = best_payload.get("official_kernel_path")

    eager_ms = as_float(baseline.get("eager", {}).get("runtime_ms"))
    compile_ms = as_float(baseline.get("compile", {}).get("runtime_ms"))
    beats_eager = best_runtime_ms is not None and eager_ms is not None and best_runtime_ms < eager_ms
    beats_compile = best_runtime_ms is not None and compile_ms is not None and best_runtime_ms < compile_ms

    num_attempts = len(entries)
    num_correct_attempts = sum(
        1
        for payload in entries
        if isinstance(payload.get("result"), dict)
        and bool(payload["result"].get("correctness"))
    )
    num_failed_attempts = sum(1 for payload in entries if payload.get("status") == "failed")
    timing_runs = sum(
        1
        for payload in entries
        if isinstance(payload.get("result"), dict)
        and candidate_runtime(payload["result"]) is not None
    )
    gpu_wait_minutes_total = (
        sum_numeric_field(entries, "gpu_wait_seconds")
        + sum_numeric_field(profiles, "gpu_wait_seconds")
    ) / 60.0
    started_at = metadata.get("created_at")
    wall_clock_elapsed_minutes = _elapsed_minutes_since(started_at)
    budget_minutes = as_float(metadata.get("time_budget_minutes"))
    counted_elapsed_minutes = None
    if budget_minutes is not None and wall_clock_elapsed_minutes is not None:
        counted_elapsed_minutes = max(0.0, wall_clock_elapsed_minutes - gpu_wait_minutes_total)
        remaining_minutes = max(0.0, budget_minutes - counted_elapsed_minutes)
    else:
        remaining_minutes = None

    tool = str(metadata.get("tool") or "codex")
    live_trace_counts, live_web_searches = live_trace_counts_for_problem(
        run_name,
        level,
        problem_id,
        tool=tool,
    )
    resolved = beats_eager and beats_compile
    recommended_actions = []
    if resolved:
        recommended_actions.append(
            "Re-check SPEC.md once, then end via ./bin/complete_problem.sh --state done --summary 'both baselines beaten'."
        )
    else:
        recommended_actions.extend(
            [
                "Keep iterating until both baselines are beaten or another truthful terminal state is justified.",
                "Re-read SPEC.md and HARDWARE.md before each major strategy change.",
                "Use ./bin/profile_ncu.sh when stuck; read profiles/latest.summary.txt first.",
                "Do not end with a plain assistant message. The only valid exit path is ./bin/complete_problem.sh.",
            ]
        )

    return {
        "generated_at": now_iso(),
        "started_at": started_at,
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "tool": tool,
        "problem_name": metadata.get("problem_name"),
        "time_budget_minutes": budget_minutes,
        "wall_clock_elapsed_minutes": wall_clock_elapsed_minutes,
        "elapsed_minutes": counted_elapsed_minutes,
        "gpu_wait_minutes_total": gpu_wait_minutes_total,
        "remaining_minutes": remaining_minutes,
        "status_mode": "resolved" if resolved else "unresolved",
        "solver_should_continue": not resolved,
        "num_attempts": num_attempts,
        "num_correct_attempts": num_correct_attempts,
        "num_failed_attempts": num_failed_attempts,
        "num_timing_runs": timing_runs,
        "num_profile_runs": len(profiles),
        "best_correct_sample_id": best_sample_id,
        "best_correct_runtime_ms": best_runtime_ms,
        "best_correct_kernel_path": best_kernel_path,
        "eager_baseline_ms": eager_ms,
        "compile_baseline_ms": compile_ms,
        "beats_eager": beats_eager,
        "beats_compile": beats_compile,
        "beats_both": resolved,
        "has_correct_solution": best_payload is not None,
        "history_path": str(history_path_value),
        "trace_counts": live_trace_counts,
        "web_searches": live_web_searches,
        "static_docs": ["AGENTS.md", "SPEC.md", "HARDWARE.md"],
        "live_docs": ["GOAL_STATUS.md", "goal_status.json"],
        "workspace_mirrors": {
            "samples_dir": "samples/",
            "best_sample": "samples/best_sample.py",
            "best_result": "samples/best_result.json",
            "profiles_dir": "profiles/",
            "latest_profile_summary": "profiles/latest.summary.txt",
            "latest_profile_details": "profiles/latest.details.txt",
            "latest_profile_raw_csv": "profiles/latest.raw.csv",
        },
        "recommended_actions": recommended_actions,
    }


def goal_status_markdown(snapshot: dict[str, Any]) -> str:
    best_runtime = snapshot.get("best_correct_runtime_ms")
    eager_baseline = snapshot.get("eager_baseline_ms")
    compile_baseline = snapshot.get("compile_baseline_ms")
    problem_name = snapshot.get("problem_name") or "unknown"
    wall_clock_elapsed_minutes = as_float(snapshot.get("wall_clock_elapsed_minutes"))
    elapsed_minutes = as_float(snapshot.get("elapsed_minutes"))
    gpu_wait_minutes_total = as_float(snapshot.get("gpu_wait_minutes_total"))
    remaining_minutes = as_float(snapshot.get("remaining_minutes"))
    time_budget_minutes = as_float(snapshot.get("time_budget_minutes"))
    substantial_budget_remains = (
        remaining_minutes is not None
        and time_budget_minutes is not None
        and remaining_minutes > max(60.0, time_budget_minutes * 0.25)
    )
    unresolved = not snapshot["beats_both"]
    if unresolved:
        heading = "# Goal Status: UNRESOLVED — keep working"
        standing_orders = [
            "- You MUST NOT stop, summarize, or hand back control. Keep working.",
            "- Do NOT ask the user for confirmation, approval, or whether to continue. Choose the next action yourself.",
            "- Re-read `SPEC.md` and `HARDWARE.md` before every major strategy change.",
            "- Timing and profiling are normal tools, not expensive last resorts. Use them even for small constant or layout changes.",
            "- Wrapper commands are authoritative. If one is slow, wait for it. Do NOT monitor it with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection.",
            "- If stuck: run `./bin/profile_ncu.sh`, read `HARDWARE.md`, search NVIDIA docs, make a new plan, and try a new branch without asking for approval.",
            "- The budget clock is wall time since workspace creation minus recorded GPU wait time. End through `./bin/complete_problem.sh` before remaining time reaches zero.",
            "- A plain assistant message is NEVER a valid way to end this run. The ONLY exit is `./bin/complete_problem.sh`.",
        ]
    else:
        heading = "# Goal Status: RESOLVED — both baselines beaten; complete with success"
        standing_orders = [
            "- Re-check `SPEC.md` once, then end through `./bin/complete_problem.sh --state done --summary 'both baselines beaten'.",
        ]

    if remaining_minutes is None:
        remaining_line = "unknown"
    elif substantial_budget_remains:
        remaining_line = (
            f"{remaining_minutes} (most of your budget remains — stopping now wastes it)"
        )
    else:
        remaining_line = str(remaining_minutes)

    if best_runtime is None:
        best_runtime_line = "none yet"
    else:
        best_runtime_line = (
            f"{best_runtime} ms (must be below {eager_baseline} ms and {compile_baseline} ms)"
        )

    profiler_line = str(snapshot["num_profile_runs"])
    if unresolved and int(snapshot.get("num_profile_runs") or 0) < 1:
        profiler_line += " — you cannot declare stalled until you profile at least once"
    lines = [
        heading,
        "",
        "Standing orders (active until both baselines are beaten):",
        "",
        *standing_orders,
        "",
        "## Current State",
        "",
        f"- problem: level {snapshot['level']} problem {snapshot['problem_id']} ({problem_name})",
        f"- best correct runtime: {best_runtime_line}",
        f"- beats eager ({eager_baseline} ms): {snapshot['beats_eager']}",
        f"- beats compile ({compile_baseline} ms): {snapshot['beats_compile']}",
        f"- beats both: {snapshot['beats_both']}",
        f"- attempts: {snapshot['num_attempts']} ({snapshot['num_correct_attempts']} correct, {snapshot['num_failed_attempts']} failed)",
        f"- timing calls: {snapshot['num_timing_runs']}",
        f"- profiler calls: {profiler_line}",
        f"- best correct sample: {snapshot.get('best_correct_sample_id')}",
        f"- wall-clock minutes since workspace creation: {wall_clock_elapsed_minutes}",
        f"- gpu wait minutes excluded from budget: {gpu_wait_minutes_total}",
        f"- elapsed minutes counted against budget: {elapsed_minutes}",
        f"- remaining minutes: {remaining_line}",
        "- static docs: `AGENTS.md`, `SPEC.md`, `HARDWARE.md`",
        "- live docs: `GOAL_STATUS.md`, `goal_status.json`",
        "- local sample mirrors: `samples/`, `samples/best_sample.py`, `samples/best_result.json`",
        "- latest profiler mirrors: `profiles/latest.summary.txt`, `profiles/latest.details.txt`, and `profiles/latest.raw.csv`",
        "",
        "Source of truth: measured run history plus the live solver trace. Refresh via `./bin/goal_status.sh` or `./bin/run_candidate.sh`.",
    ]
    return "\n".join(lines) + "\n"


def write_goal_status_files(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    workspace: Path,
) -> dict[str, Any]:
    snapshot = goal_status_snapshot(
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        workspace=workspace,
    )
    write_workspace_best_sample(
        workspace,
        best_correct_payload(history_path(run_name, level, problem_id)),
    )
    write_json(workspace / "goal_status.json", snapshot)
    write_text(workspace / "GOAL_STATUS.md", goal_status_markdown(snapshot))
    write_json(goal_status_archive_path(run_name, level, problem_id), snapshot)
    return snapshot
