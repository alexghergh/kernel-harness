"""Build and render the live per-problem goal status that the solver re-reads after each step.

This module bridges archived attempts, profiler outputs, and live trace counts back into workspace-facing status files.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive_layout import (
    goal_status_archive_path,
    mcp_trace_events_path,
    profile_manifest_entries,
    sample_manifest_entries,
    trace_events_path,
)
from .common import as_float
from .live_gpu_wait import active_live_gpu_wait_seconds
from .mcp.trace import load_mcp_ir_events
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
    ir_events.extend(
        load_mcp_ir_events(
            mcp_trace_events_path(run_name, level, problem_id),
            starting_line=len(raw_event_entries) + 1_000_000,
        )
    )
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
    """Collect the current archived measurements and trace-derived activity into one live snapshot."""
    metadata = load_workspace_metadata(workspace)
    baseline = load_workspace_baseline(workspace)
    entries = sample_manifest_entries(run_name, level, problem_id)
    profiles = profile_manifest_entries(run_name, level, problem_id)
    best_payload = best_correct_payload(entries)
    best_runtime_ms = None
    best_sample_id = None
    if best_payload is not None:
        result = best_payload.get("result")
        if isinstance(result, dict):
            best_runtime_ms = candidate_runtime(result)
        best_sample_id = best_payload.get("sample_id")

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
    num_incorrect_attempts = sum(
        1
        for payload in entries
        if isinstance(payload.get("result"), dict)
        and payload["result"].get("correctness") is False
    )
    num_execution_failed_attempts = sum(
        1 for payload in entries if payload.get("status") == "failed"
    )
    num_other_attempts = max(
        0,
        num_attempts - num_correct_attempts - num_incorrect_attempts - num_execution_failed_attempts,
    )
    timing_runs = sum(
        1
        for payload in entries
        if isinstance(payload.get("result"), dict)
        and candidate_runtime(payload["result"]) is not None
    )
    recorded_gpu_wait_minutes = (
        sum_numeric_field(entries, "gpu_wait_seconds")
        + sum_numeric_field(profiles, "gpu_wait_seconds")
    ) / 60.0
    live_gpu_wait_minutes = (
        active_live_gpu_wait_seconds(run_name, level, problem_id) / 60.0
    )
    gpu_wait_minutes_total = recorded_gpu_wait_minutes + live_gpu_wait_minutes
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
            "Re-check SPEC.md once, then end via the `complete_problem` MCP tool with a short success summary."
        )
    else:
        recommended_actions.extend(
            [
                "Keep iterating until both baselines are beaten or another truthful terminal state is justified.",
                "Re-read SPEC.md and HARDWARE.md before each major strategy change.",
                "Use the `profile_ncu` MCP tool when stuck; read `profiles/latest.summary.txt` first.",
                "Do not end with a plain assistant message. The only valid exit path is the `complete_problem` MCP tool.",
                "Never overlap MCP tool calls. Start a new harness tool call only after the previous one has fully returned.",
                "`run_candidate` and `profile_ncu` may take a while; wait for the tool result instead of treating them as hung.",
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
        "recorded_gpu_wait_minutes": recorded_gpu_wait_minutes,
        "live_gpu_wait_minutes": live_gpu_wait_minutes,
        "gpu_wait_minutes_total": gpu_wait_minutes_total,
        "remaining_minutes": remaining_minutes,
        "status_mode": "resolved" if resolved else "unresolved",
        "solver_should_continue": not resolved,
        "num_attempts": num_attempts,
        "num_correct_attempts": num_correct_attempts,
        "num_incorrect_attempts": num_incorrect_attempts,
        "num_execution_failed_attempts": num_execution_failed_attempts,
        "num_other_attempts": num_other_attempts,
        "num_timing_runs": timing_runs,
        "num_profile_runs": len(profiles),
        "best_correct_sample_id": best_sample_id,
        "best_correct_runtime_ms": best_runtime_ms,
        "eager_baseline_ms": eager_ms,
        "compile_baseline_ms": compile_ms,
        "beats_eager": beats_eager,
        "beats_compile": beats_compile,
        "beats_both": resolved,
        "has_correct_solution": best_payload is not None,
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
        },
        "recommended_actions": recommended_actions,
    }


def goal_status_markdown(snapshot: dict[str, Any]) -> str:
    """Render the live goal-status markdown that the solver re-reads during the loop."""
    best_runtime = snapshot.get("best_correct_runtime_ms")
    eager_baseline = snapshot.get("eager_baseline_ms")
    compile_baseline = snapshot.get("compile_baseline_ms")
    problem_name = snapshot.get("problem_name") or "unknown"
    wall_clock_elapsed_minutes = as_float(snapshot.get("wall_clock_elapsed_minutes"))
    elapsed_minutes = as_float(snapshot.get("elapsed_minutes"))
    recorded_gpu_wait_minutes = as_float(snapshot.get("recorded_gpu_wait_minutes"))
    live_gpu_wait_minutes = as_float(snapshot.get("live_gpu_wait_minutes"))
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
            "- Never overlap MCP tool calls. Start a new harness tool call only after the previous one has fully returned.",
            "- Harness MCP tools are authoritative. If one is slow, wait for it. Do NOT monitor it with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection.",
            "- If stuck: call `profile_ncu`, read `HARDWARE.md`, search NVIDIA docs, make a new plan, and try a new branch without asking for approval.",
            "- The budget clock is wall time since workspace creation minus recorded GPU wait time and any live GPU lease wait currently in progress. End through `complete_problem` before remaining time reaches zero.",
            "- A plain assistant message is NEVER a valid way to end this run. The ONLY exit is `complete_problem(summary=...)`.",
            "- `run_candidate` and `profile_ncu` may take a while; wait for the tool result instead of treating them as hung.",
        ]
    else:
        heading = "# Goal Status: RESOLVED — both baselines beaten; complete with success"
        standing_orders = [
            "- Re-check `SPEC.md` once, then end through `complete_problem(summary='both baselines beaten')`.",
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
    attempt_breakdown = (
        f"{snapshot['num_correct_attempts']} correct, "
        f"{snapshot['num_incorrect_attempts']} incorrect, "
        f"{snapshot['num_execution_failed_attempts']} execution-failed"
    )
    if snapshot.get("num_other_attempts"):
        attempt_breakdown += f", {snapshot['num_other_attempts']} other"
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
        f"- attempts: {snapshot['num_attempts']} ({attempt_breakdown})",
        f"- timing calls: {snapshot['num_timing_runs']}",
        f"- profiler calls: {profiler_line}",
        f"- best correct sample: {snapshot.get('best_correct_sample_id')}",
        f"- wall-clock minutes since workspace creation: {wall_clock_elapsed_minutes}",
        f"- completed gpu wait minutes excluded from budget: {recorded_gpu_wait_minutes}",
        f"- currently active gpu queue-wait minutes excluded from budget: {live_gpu_wait_minutes}",
        f"- total gpu wait minutes excluded from budget: {gpu_wait_minutes_total}",
        f"- elapsed minutes counted against budget: {elapsed_minutes}",
        f"- remaining minutes: {remaining_line}",
        "- static docs: `AGENTS.md`, `SPEC.md`, `HARDWARE.md`",
        "- live docs: `GOAL_STATUS.md`, `goal_status.json`",
        "- local sample mirrors: `samples/`, `samples/best_sample.py`, `samples/best_result.json`",
        "- latest profiler mirrors: `profiles/latest.summary.txt` and `profiles/latest.details.txt`",
        "",
        "Source of truth: measured run history plus the live solver trace. Refresh via the `goal_status` or `run_candidate` MCP tools.",
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
        best_correct_payload(sample_manifest_entries(run_name, level, problem_id)),
    )
    write_json(workspace / "goal_status.json", snapshot)
    write_text(workspace / "GOAL_STATUS.md", goal_status_markdown(snapshot))
    write_json(goal_status_archive_path(run_name, level, problem_id), snapshot)
    return snapshot
