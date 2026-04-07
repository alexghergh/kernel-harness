from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME
from .common import as_float
from .project import (
    archive_attempts_dir,
    archive_contract_dir,
    archive_profiles_dir,
    artifact_agent_dir,
    now_iso,
    workspace_dir,
    write_json,
    write_text,
)
from .trace_analysis import trace_counts, web_searches_from_ir
from .trace_ir import load_trace_event_entries, materialize_trace_ir


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


def serialize_exception(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def load_baseline_file(path: str | None) -> dict[str, Any] | None:
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


def workspace_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workspace_metadata(workspace: Path) -> dict[str, Any]:
    return read_json_file(workspace / "problem.json")


def load_workspace_baseline(workspace: Path) -> dict[str, Any]:
    return read_json_file(workspace / "baseline.json")


def archive_problem_contract_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_contract_dir(run_name, level, problem_id)


def archive_problem_attempts_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_attempts_dir(run_name, level, problem_id)


def archive_problem_profiles_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_profiles_dir(run_name, level, problem_id)


def history_path(run_name: str, level: int, problem_id: int) -> Path:
    return archive_problem_attempts_dir(run_name, level, problem_id) / "history.jsonl"


def sample_manifest_path(run_name: str, level: int, problem_id: int, sample_id: int) -> Path:
    return archive_problem_attempts_dir(run_name, level, problem_id) / f"sample_{sample_id}.json"


def goal_status_archive_path(run_name: str, level: int, problem_id: int) -> Path:
    return artifact_agent_dir(run_name, level, problem_id) / "goal_status.json"


def profile_index_path(run_name: str, level: int, problem_id: int) -> Path:
    return archive_problem_profiles_dir(run_name, level, problem_id) / "index.jsonl"


def trace_events_path(run_name: str, level: int, problem_id: int) -> Path:
    return artifact_agent_dir(run_name, level, problem_id) / "events.jsonl"


def history_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def profile_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


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


def write_workspace_sample_copy(
    workspace: Path,
    sample_id: int,
    candidate_src: str,
) -> None:
    write_text(
        workspace_samples_dir(workspace) / f"sample_{sample_id}.py",
        candidate_src,
    )


def write_workspace_best_sample(
    workspace: Path,
    payload: dict[str, Any] | None,
) -> None:
    best_sample_path = workspace_samples_dir(workspace) / "best_sample.py"
    best_result_path = workspace_samples_dir(workspace) / "best_result.json"
    if payload is None:
        if best_sample_path.exists():
            best_sample_path.unlink()
        if best_result_path.exists():
            best_result_path.unlink()
        return

    official_kernel = payload.get("official_kernel_path")
    if isinstance(official_kernel, str):
        official_kernel_path = Path(official_kernel)
        if official_kernel_path.exists():
            write_text(
                best_sample_path,
                official_kernel_path.read_text(encoding="utf-8"),
            )
        elif best_sample_path.exists():
            best_sample_path.unlink()
    elif best_sample_path.exists():
        best_sample_path.unlink()
    write_json(best_result_path, payload)


def latest_workspace_profile_paths(workspace: Path) -> dict[str, Path]:
    profiles_dir = workspace_profiles_dir(workspace)
    return {
        "details": profiles_dir / "latest.details.txt",
        "details_stderr": profiles_dir / "latest.details.stderr.txt",
        "raw_csv": profiles_dir / "latest.raw.csv",
        "raw_csv_stderr": profiles_dir / "latest.raw.stderr.txt",
        "summary": profiles_dir / "latest.summary.txt",
        "stdout": profiles_dir / "latest.stdout.txt",
        "stderr": profiles_dir / "latest.stderr.txt",
        "json": profiles_dir / "latest.json",
    }


def baseline_payload_for_problem(
    *,
    level: int,
    problem_id: int,
    problem_name: str,
    eager_baseline_file: str,
    compile_baseline_file: str,
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
            "source_file": eager_baseline_file,
        },
        "compile": {
            "runtime_ms": compile_runtime_ms,
            "source_file": compile_baseline_file,
        },
    }


def best_correct_payload(history_path_value: Path) -> dict[str, Any] | None:
    best_payload: dict[str, Any] | None = None
    best_runtime: float | None = None
    for payload in history_entries(history_path_value):
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
    gpu_wait_minutes_total = sum_numeric_field(entries, "gpu_wait_seconds") / 60.0
    elapsed_minutes_total = sum_numeric_field(entries, "artifact_reservation_wait_seconds")
    elapsed_minutes_total += sum_numeric_field(entries, "artifact_commit_wait_seconds")
    elapsed_minutes_total += sum_numeric_field(entries, "gpu_wait_seconds")
    elapsed_minutes_total += sum_numeric_field(profiles, "artifact_reservation_wait_seconds")
    elapsed_minutes_total += sum_numeric_field(profiles, "artifact_commit_wait_seconds")
    elapsed_minutes_total += sum_numeric_field(profiles, "gpu_wait_seconds")
    elapsed_minutes_total /= 60.0
    budget_minutes = as_float(metadata.get("time_budget_minutes"))
    counted_elapsed_minutes = None
    if budget_minutes is not None:
        counted_elapsed_minutes = max(0.0, elapsed_minutes_total - gpu_wait_minutes_total)
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

    return {
        "generated_at": now_iso(),
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "tool": tool,
        "problem_name": metadata.get("problem_name"),
        "time_budget_minutes": budget_minutes,
        "elapsed_minutes": counted_elapsed_minutes,
        "gpu_wait_minutes_total": gpu_wait_minutes_total,
        "remaining_minutes": remaining_minutes,
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
        "beats_both": beats_eager and beats_compile,
        "has_correct_solution": best_payload is not None,
        "history_path": str(history_path_value),
        "trace_counts": live_trace_counts,
        "web_searches": live_web_searches,
    }


def goal_status_markdown(snapshot: dict[str, Any]) -> str:
    best_runtime = snapshot.get("best_correct_runtime_ms")
    eager_baseline = snapshot.get("eager_baseline_ms")
    compile_baseline = snapshot.get("compile_baseline_ms")
    problem_name = snapshot.get("problem_name") or "unknown"
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
            "- Re-read `SPEC.md` and `HARDWARE.md` before every major strategy change.",
            "- Timing and profiling are normal tools, not expensive last resorts. Use them even for small constant or layout changes.",
            "- Wrapper commands are authoritative. If one is slow, wait for it. Do NOT monitor it with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection.",
            "- If stuck: run `./bin/profile_ncu.sh`, read `HARDWARE.md`, search NVIDIA docs, and try a new branch.",
            "- The time budget is enforced by the harness. End through `./bin/complete_problem.sh` before remaining time reaches zero.",
            "- A plain assistant message is NEVER a valid way to end this run. The ONLY exit is `./bin/complete_problem.sh`.",
        ]
    else:
        heading = "# Goal Status: RESOLVED — both baselines beaten; complete with success"
        standing_orders = [
            "- Re-check `SPEC.md` once, then end through `./bin/complete_problem.sh --state done --summary 'both baselines beaten'`.",
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
        f"- elapsed minutes counted against budget: {elapsed_minutes}",
        f"- gpu wait minutes excluded from budget: {gpu_wait_minutes_total}",
        f"- remaining minutes: {remaining_line}",
        "- local sample history: `samples/`",
        "- local best sample mirror: `samples/best_sample.py`",
        "- latest profiler files: `profiles/latest.summary.txt`, `profiles/latest.details.txt`, and `profiles/latest.raw.csv`",
        "",
        "Source of truth: measured from run history plus the live solver trace. Refresh via `./bin/goal_status.sh` or `./bin/run_candidate.sh`.",
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


def problem_workspace_paths(
    run_name: str,
    level: int,
    problem_id: int,
    workspace_root: str | None,
) -> dict[str, Path]:
    workspace = workspace_dir(run_name, level, problem_id, explicit_root=workspace_root)
    return {
        "workspace": workspace,
        "samples": workspace / "samples",
        "profiles": workspace / "profiles",
        "bin": workspace / "bin",
    }


def workspace_candidate_path(workspace: Path) -> Path:
    return workspace / CANDIDATE_FILENAME


def workspace_samples_dir(workspace: Path) -> Path:
    return workspace / "samples"


def workspace_profiles_dir(workspace: Path) -> Path:
    return workspace / "profiles"


def workspace_relpath(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def next_archive_profile_index(run_name: str, level: int, problem_id: int) -> int:
    profiles_dir = archive_problem_profiles_dir(run_name, level, problem_id)
    max_index = 0
    for child in profiles_dir.glob("profile_*.json"):
        match = re.fullmatch(r"profile_(\d+)\.json", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1
