from __future__ import annotations

import argparse
import csv
import io
import math
import json
import re
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urlparse

from .candidate_contract import CANDIDATE_FILENAME, candidate_template
from .candidate_validation import CandidateValidationError, validate_candidate_source
from .gpu_pool import lease_gpu_slot, lease_problem_artifacts
from .hardware_catalog import render_hardware_markdown, resolve_hardware_spec
from .kernelbench import evaluate_candidate, load_problem
from .project import (
    append_jsonl,
    artifact_agent_dir,
    artifact_problem_dir,
    experiment_root,
    kernelbench_root,
    make_executable,
    next_sample_id,
    now_iso,
    official_kernel_path,
    official_prompt_path,
    workspace_dir,
    write_json,
    write_text,
)

TOOL_CHOICES = ("codex", "claude")
_ALLOWED_WEB_SEARCH_HOSTS = ("docs.nvidia.com",)


def _default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-problem-workspace")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--level", type=int, required=True)
    prepare.add_argument("--problem-id", type=int, required=True)
    prepare.add_argument("--dataset-src", default="local")
    prepare.add_argument("--kernelbench-root", default=None)
    prepare.add_argument("--kernelbench-python", required=True)
    prepare.add_argument("--workspace-root", default=None)
    prepare.add_argument("--gpu-name", default="")
    prepare.add_argument("--num-gpus", type=int, default=1)
    prepare.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    prepare.add_argument("--model", default="gpt-5-codex")
    prepare.add_argument("--time-budget-minutes", type=int, default=720)
    prepare.add_argument("--eager-baseline-file", required=True)
    prepare.add_argument("--compile-baseline-file", required=True)

    problem_info = subparsers.add_parser("problem-info")
    problem_info.add_argument("--level", type=int, required=True)
    problem_info.add_argument("--problem-id", type=int, required=True)
    problem_info.add_argument("--dataset-src", default="local")
    problem_info.add_argument("--kernelbench-root", default=None)

    run = subparsers.add_parser("run-candidate")
    run.add_argument("--candidate", required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--level", type=int, required=True)
    run.add_argument("--problem-id", type=int, required=True)
    run.add_argument("--dataset-src", default="local")
    run.add_argument("--kernelbench-root", default=None)
    run.add_argument("--gpu-id", type=int, default=None)
    run.add_argument("--num-gpu-slots", type=int, default=1)
    run.add_argument("--timing-method", default=None)
    run.add_argument("--backend", default="cuda")
    run.add_argument("--precision", default="fp32")
    run.add_argument("--num-correct-trials", type=int, default=5)
    run.add_argument("--num-perf-trials", type=int, default=100)
    run.add_argument("--prompt-path", default=None)
    run.add_argument("--workspace", default=None)

    profile = subparsers.add_parser("profile-ncu")
    profile.add_argument("--candidate", required=True)
    profile.add_argument("--run-name", required=True)
    profile.add_argument("--level", type=int, required=True)
    profile.add_argument("--problem-id", type=int, required=True)
    profile.add_argument("--dataset-src", default="local")
    profile.add_argument("--kernelbench-root", default=None)
    profile.add_argument("--gpu-id", type=int, default=None)
    profile.add_argument("--num-gpu-slots", type=int, default=1)
    profile.add_argument("--sample-id", type=int, default=None)
    profile.add_argument("--ncu-set", default="full")
    profile.add_argument("--workspace", default=None)

    best = subparsers.add_parser("best-result")
    best.add_argument("--run-name", required=True)
    best.add_argument("--level", type=int, required=True)
    best.add_argument("--problem-id", type=int, required=True)

    goal = subparsers.add_parser("goal-status")
    goal.add_argument("--run-name", required=True)
    goal.add_argument("--level", type=int, required=True)
    goal.add_argument("--problem-id", type=int, required=True)
    goal.add_argument("--workspace", required=True)

    complete = subparsers.add_parser("complete-problem")
    complete.add_argument("--run-name", required=True)
    complete.add_argument("--level", type=int, required=True)
    complete.add_argument("--problem-id", type=int, required=True)
    complete.add_argument("--workspace", required=True)
    complete.add_argument(
        "--decision",
        required=True,
        choices=[
            "beats_both_baselines",
            "beats_eager_only",
            "beats_compile_only",
            "budget_exhausted",
            "stalled",
            "harness_failure",
            "failed_to_generate",
        ],
    )
    complete.add_argument("--summary", default="")
    complete.add_argument("--allow-overwrite", action="store_true")

    trace = subparsers.add_parser("materialize-agent-trace")
    trace.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    trace.add_argument("--events-path", required=True)
    trace.add_argument("--output-path", required=True)
    trace.add_argument("--completion-path", default=None)
    trace.add_argument("--final-message-path", default=None)
    trace.add_argument("--workspace", default=None)

    legacy_trace = subparsers.add_parser("materialize-codex-trace")
    legacy_trace.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    legacy_trace.add_argument("--events-path", required=True)
    legacy_trace.add_argument("--output-path", required=True)
    legacy_trace.add_argument("--completion-path", default=None)
    legacy_trace.add_argument("--final-message-path", default=None)
    legacy_trace.add_argument("--workspace", default=None)

    summary = subparsers.add_parser("summarize-run")
    summary.add_argument("--run-name", required=True)
    summary.add_argument("--level", type=int, action="append", default=[])
    summary.add_argument("--problem-id", type=int, action="append", default=[])
    summary.add_argument("--dataset-src", default="local")
    summary.add_argument("--kernelbench-root", default=None)
    summary.add_argument("--eager-baseline-file", default=None)
    summary.add_argument("--compile-baseline-file", default=None)
    summary.add_argument("--pass-k", default="1,5,10")

    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_tool_name(raw: Any) -> str:
    tool = str(raw or "codex").strip().lower()
    if tool not in TOOL_CHOICES:
        raise SystemExit(
            f"Unsupported tool {tool!r}. Expected one of: {', '.join(TOOL_CHOICES)}."
        )
    return tool


def _candidate_runtime(result: dict[str, Any]) -> float | None:
    runtime = _as_float(result.get("runtime"))
    if runtime is not None:
        return runtime

    runtime_stats = result.get("runtime_stats")
    if isinstance(runtime_stats, dict):
        for key in ("mean", "mean_runtime_ms", "runtime_ms"):
            value = _as_float(runtime_stats.get(key))
            if value is not None:
                return value

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for key in ("runtime_ms", "mean_runtime_ms"):
            value = _as_float(metadata.get(key))
            if value is not None:
                return value
    return None


def _serialize_exception(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _load_baseline_file(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _baseline_mean_for_problem(
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
            value = _as_float(problem_entry.get(key))
            if value is not None:
                return value
    return None


def _workspace_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_workspace_metadata(workspace: Path) -> dict[str, Any]:
    return _read_json(workspace / "problem.json")


def _load_workspace_baseline(workspace: Path) -> dict[str, Any]:
    return _read_json(workspace / "baseline.json")


def _history_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def _trace_events_path(run_name: str, level: int, problem_id: int) -> Path:
    return artifact_agent_dir(run_name, level, problem_id) / "events.jsonl"


def _load_trace_event_entries(
    events_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    raw_events: list[dict[str, Any]] = []
    raw_event_entries: list[tuple[int, dict[str, Any]]] = []
    if not events_path.exists():
        return raw_events, raw_event_entries

    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_events.append(payload)
        raw_event_entries.append((line_number, payload))
    return raw_events, raw_event_entries


def _collect_urls(payload: Any, *, urls: set[str]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_urls(value, urls=urls)
        return
    if isinstance(payload, list):
        for value in payload:
            _collect_urls(value, urls=urls)
        return
    if isinstance(payload, str):
        for match in re.findall(r"https?://[^\s\"'<>]+", payload):
            urls.add(match)


def _claude_content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("type") != "assistant":
        return []
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _claude_tool_use_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in _claude_content_blocks(payload)
        if block.get("type") == "tool_use"
    ]


def _claude_tool_name(block: dict[str, Any]) -> str:
    return str(block.get("name") or "").strip()


def _claude_tool_input(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("input")
    return value if isinstance(value, dict) else {}


def _claude_tool_command(block: dict[str, Any]) -> str | None:
    tool_input = _claude_tool_input(block)
    for key in ("command", "cmd", "shell_command"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _claude_tool_path(block: dict[str, Any]) -> str | None:
    tool_input = _claude_tool_input(block)
    for key in ("file_path", "path", "target_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _web_searches_from_entries(
    raw_event_entries: list[tuple[int, dict[str, Any]]],
    *,
    tool: str = "codex",
) -> list[dict[str, Any]]:
    tool = _normalize_tool_name(tool)
    if tool == "claude":
        web_searches: list[dict[str, Any]] = []
        for line_number, payload in raw_event_entries:
            for block in _claude_tool_use_blocks(payload):
                if _claude_tool_name(block).strip().lower() not in {
                    "websearch",
                    "web_search",
                }:
                    continue
                tool_input = _claude_tool_input(block)
                query = tool_input.get("query")
                raw_queries = tool_input.get("queries")
                queries = (
                    [str(value) for value in raw_queries if value]
                    if isinstance(raw_queries, list)
                    else ([str(query)] if query else [])
                )
                urls: set[str] = set()
                _collect_urls(block, urls=urls)
                domains = sorted(
                    {
                        parsed.hostname
                        for parsed in (urlparse(url) for url in urls)
                        if parsed.hostname
                    }
                )
                web_searches.append(
                    {
                        "line": line_number,
                        "query": str(query) if query else None,
                        "queries": queries,
                        "domains": domains,
                    }
                )
        return web_searches

    web_searches: list[dict[str, Any]] = []
    for line_number, payload in raw_event_entries:
        if payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "web_search":
            continue

        query = item.get("query")
        action = item.get("action")
        queries = None
        if isinstance(action, dict):
            raw_queries = action.get("queries")
            if isinstance(raw_queries, list):
                queries = [str(value) for value in raw_queries if value]
            if not query:
                query = action.get("query")

        urls: set[str] = set()
        _collect_urls(item, urls=urls)
        domains = sorted(
            {
                parsed.hostname
                for parsed in (urlparse(url) for url in urls)
                if parsed.hostname
            }
        )

        web_searches.append(
            {
                "line": line_number,
                "query": str(query) if query else None,
                "queries": queries or ([str(query)] if query else []),
                "domains": domains,
            }
        )
    return web_searches


def _live_trace_counts_for_problem(
    run_name: str,
    level: int,
    problem_id: int,
    *,
    tool: str = "codex",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, raw_event_entries = _load_trace_event_entries(
        _trace_events_path(run_name, level, problem_id)
    )
    return _trace_counts_from_entries(
        raw_event_entries,
        tool=tool,
    ), _web_searches_from_entries(raw_event_entries, tool=tool)


def _write_workspace_sample_copy(
    workspace: Path,
    sample_id: int,
    candidate_src: str,
) -> None:
    write_text(
        _workspace_samples_dir(workspace) / f"sample_{sample_id}.py",
        candidate_src,
    )


def _write_workspace_best_sample(
    workspace: Path,
    payload: dict[str, Any] | None,
) -> None:
    best_sample_path = _workspace_samples_dir(workspace) / "best_sample.py"
    best_result_path = _workspace_samples_dir(workspace) / "best_result.json"
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


def _latest_workspace_profile_paths(workspace: Path) -> dict[str, Path]:
    profiles_dir = _workspace_profiles_dir(workspace)
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


def _allowed_workspace_read_paths(workspace: Path) -> set[Path]:
    return {
        (workspace / "AGENTS.md").resolve(),
        (workspace / "SPEC.md").resolve(),
        (workspace / "HARDWARE.md").resolve(),
        (workspace / "GOAL_STATUS.md").resolve(),
        (workspace / "goal_status.json").resolve(),
        (workspace / "problem.json").resolve(),
        (workspace / "problem_reference.py").resolve(),
        (workspace / "baseline.json").resolve(),
        _workspace_candidate_path(workspace).resolve(),
    }


def _allowed_workspace_read_roots(workspace: Path) -> tuple[Path, ...]:
    return (
        _workspace_samples_dir(workspace).resolve(),
        _workspace_profiles_dir(workspace).resolve(),
    )


def _is_allowed_workspace_read(path: Path, workspace: Path) -> bool:
    resolved = path.resolve()
    if resolved in _allowed_workspace_read_paths(workspace):
        return True
    return any(
        _is_relative_to(resolved, root)
        for root in _allowed_workspace_read_roots(workspace)
    )


def _summarize_ncu_raw_csv(raw_csv_text: str) -> str:
    rows = list(csv.DictReader(io.StringIO(raw_csv_text)))
    if not rows:
        return (
            "NCU summary could not be generated because the raw CSV had no data rows.\n"
            "Read profiles/latest.details.txt for the full text report.\n"
        )

    def score(row: dict[str, str]) -> int:
        return sum(1 for value in row.values() if isinstance(value, str) and any(ch.isdigit() for ch in value))

    row = max(rows, key=score)

    def first_value(*keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    lines = [
        "# NCU Summary",
        "",
        "Prefer this file first. Read `profiles/latest.details.txt` only when you need the full report.",
        "",
    ]

    metric_groups = (
        (
            "Key performance metrics",
            (
                ("duration", "gpu__time_duration.sum"),
                ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                (
                    "compute+memory throughput",
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                ),
                ("registers per thread", "launch__registers_per_thread"),
                ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
            ),
        ),
        (
            "Memory and shared-memory indicators",
            (
                ("L1/TEX throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
                ("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_active"),
                ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                ("shared-memory conflict n-way", "derived__memory_l1_conflicts_shared_nway"),
                (
                    "shared-memory excessive wavefronts",
                    "derived__memory_l1_wavefronts_shared_excessive",
                ),
            ),
        ),
        (
            "Occupancy limiters",
            (
                ("block limit by registers", "launch__occupancy_limit_registers"),
                ("block limit by shared memory", "launch__occupancy_limit_shared_mem"),
                ("block limit by warps", "launch__occupancy_limit_warps"),
            ),
        ),
    )

    for title, metrics in metric_groups:
        lines.append(f"## {title}")
        wrote_any = False
        for label, key in metrics:
            value = first_value(key)
            if value is None:
                continue
            lines.append(f"- {label}: {value}")
            wrote_any = True
        if not wrote_any:
            lines.append("- no values found in the exported raw CSV")
        lines.append("")

    stall_entries: list[tuple[str, float, str]] = []
    for key, value in row.items():
        if "smsp__average_warps_issue_stalled_" not in key:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        numeric = _as_float(value)
        if numeric is None or numeric <= 0:
            continue
        stall_name = key.split("stalled_", 1)[1].split("_per_", 1)[0]
        stall_entries.append((stall_name, numeric, value.strip()))

    lines.append("## Top warp stalls")
    if stall_entries:
        for stall_name, _, raw_value in sorted(stall_entries, key=lambda item: -item[1])[:8]:
            lines.append(f"- {stall_name}: {raw_value}")
    else:
        lines.append("- no positive warp-stall metrics were found in the exported raw CSV")
    lines.append("")

    lines.append("## Next step")
    lines.append(
        "- Re-read `HARDWARE.md`, then use this summary plus `profiles/latest.details.txt` to pick the next branch."
    )
    lines.append("")
    return "\n".join(lines)


def _baseline_payload_for_problem(
    *,
    level: int,
    problem_id: int,
    problem_name: str | None,
    eager_baseline_file: str,
    compile_baseline_file: str,
) -> dict[str, Any]:
    eager_baseline = _load_baseline_file(eager_baseline_file)
    compile_baseline = _load_baseline_file(compile_baseline_file)
    eager_ms = _baseline_mean_for_problem(
        baseline=eager_baseline,
        level=level,
        problem_name=problem_name,
    )
    compile_ms = _baseline_mean_for_problem(
        baseline=compile_baseline,
        level=level,
        problem_name=problem_name,
    )
    if eager_ms is None:
        raise RuntimeError(
            f"Failed to resolve eager baseline for level={level} problem_id={problem_id} problem_name={problem_name!r}"
        )
    if compile_ms is None:
        raise RuntimeError(
            f"Failed to resolve torch.compile baseline for level={level} problem_id={problem_id} problem_name={problem_name!r}"
        )
    return {
        "level": level,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "eager": {
            "runtime_ms": eager_ms,
            "source_file": str(Path(eager_baseline_file).expanduser().resolve()),
        },
        "compile": {
            "runtime_ms": compile_ms,
            "source_file": str(Path(compile_baseline_file).expanduser().resolve()),
        },
    }


def _best_correct_payload(history_path: Path) -> dict[str, Any] | None:
    best_payload: dict[str, Any] | None = None
    best_ms: float | None = None
    for payload in _history_entries(history_path):
        result = payload.get("result", {})
        if not bool(result.get("correctness")):
            continue
        runtime_ms = _candidate_runtime(result)
        if runtime_ms is None:
            continue
        if best_ms is None or runtime_ms < best_ms:
            best_ms = runtime_ms
            best_payload = payload
    return best_payload


def _elapsed_minutes(created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        started = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    elapsed = (now - started).total_seconds() / 60.0
    return max(elapsed, 0.0)


def _sum_numeric_field(payloads: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in payloads:
        value = _as_float(payload.get(key))
        if value is None:
            continue
        total += value
    return max(total, 0.0)


def _profile_entries(ncu_dir: Path) -> list[dict[str, Any]]:
    if not ncu_dir.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(ncu_dir.glob("*.json")):
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return entries


def _goal_status_snapshot(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    workspace: Path,
) -> dict[str, Any]:
    metadata = _load_workspace_metadata(workspace)
    tool = _normalize_tool_name(metadata.get("tool"))
    baseline = _load_workspace_baseline(workspace)
    history_path = artifact_problem_dir(run_name, level, problem_id) / "history.jsonl"
    entries = _history_entries(history_path)
    best_payload = _best_correct_payload(history_path)
    live_trace_counts, live_web_searches = _live_trace_counts_for_problem(
        run_name,
        level,
        problem_id,
        tool=tool,
    )

    best_runtime_ms = None
    best_sample_id = None
    best_kernel_path = None
    if best_payload is not None:
        best_runtime_ms = _candidate_runtime(best_payload.get("result", {}))
        best_sample_id = best_payload.get("sample_id")
        best_kernel_path = best_payload.get("official_kernel_path")

    eager_ms = _as_float(baseline.get("eager", {}).get("runtime_ms"))
    compile_ms = _as_float(baseline.get("compile", {}).get("runtime_ms"))
    beats_eager = (
        best_runtime_ms is not None and eager_ms is not None and best_runtime_ms < eager_ms
    )
    beats_compile = (
        best_runtime_ms is not None
        and compile_ms is not None
        and best_runtime_ms < compile_ms
    )
    elapsed_minutes_wall_clock = _elapsed_minutes(metadata.get("created_at"))
    time_budget_minutes = _as_float(metadata.get("time_budget_minutes"))
    ncu_dir = artifact_problem_dir(run_name, level, problem_id) / "ncu"
    profile_payloads = _profile_entries(ncu_dir)
    gpu_wait_minutes_total = (
        _sum_numeric_field(entries, "gpu_wait_seconds")
        + _sum_numeric_field(profile_payloads, "gpu_wait_seconds")
    ) / 60.0
    elapsed_minutes = None
    if elapsed_minutes_wall_clock is not None:
        elapsed_minutes = max(elapsed_minutes_wall_clock - gpu_wait_minutes_total, 0.0)
    remaining_minutes = None
    if elapsed_minutes is not None and time_budget_minutes is not None:
        remaining_minutes = max(time_budget_minutes - elapsed_minutes, 0.0)
    profile_runs = sorted(ncu_dir.glob("*.json")) if ncu_dir.exists() else []

    succeeded_entries = [entry for entry in entries if entry.get("status") == "succeeded"]
    failed_entries = [entry for entry in entries if entry.get("status") == "failed"]
    correct_entries = [
        entry
        for entry in succeeded_entries
        if bool(entry.get("result", {}).get("correctness"))
    ]

    return {
        "generated_at": now_iso(),
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "tool": tool,
        "problem_name": metadata.get("problem_name"),
        "time_budget_minutes": time_budget_minutes,
        "elapsed_minutes_wall_clock": elapsed_minutes_wall_clock,
        "elapsed_minutes": elapsed_minutes,
        "gpu_wait_minutes_total": gpu_wait_minutes_total,
        "remaining_minutes": remaining_minutes,
        "num_timing_runs": len(entries),
        "num_profile_runs": max(
            len(profile_runs),
            int(_as_float(live_trace_counts.get("profile_ncu_calls")) or 0),
        ),
        "num_attempts": len(entries),
        "num_successful_attempts": len(succeeded_entries),
        "num_failed_attempts": len(failed_entries),
        "num_correct_attempts": len(correct_entries),
        "num_web_search_calls": int(
            _as_float(live_trace_counts.get("web_search_calls")) or 0
        ),
        "best_correct_sample_id": best_sample_id,
        "best_correct_runtime_ms": best_runtime_ms,
        "best_correct_kernel_path": best_kernel_path,
        "eager_baseline_ms": eager_ms,
        "compile_baseline_ms": compile_ms,
        "beats_eager": beats_eager,
        "beats_compile": beats_compile,
        "beats_both": beats_eager and beats_compile,
        "has_correct_solution": best_payload is not None,
        "history_path": str(history_path),
        "trace_counts": live_trace_counts,
        "web_searches": live_web_searches,
    }


def _goal_status_markdown(snapshot: dict[str, Any]) -> str:
    best_runtime = snapshot.get("best_correct_runtime_ms")
    eager_baseline = snapshot.get("eager_baseline_ms")
    compile_baseline = snapshot.get("compile_baseline_ms")
    problem_name = snapshot.get("problem_name") or "unknown"
    elapsed_minutes = _as_float(snapshot.get("elapsed_minutes"))
    gpu_wait_minutes_total = _as_float(snapshot.get("gpu_wait_minutes_total"))
    remaining_minutes = _as_float(snapshot.get("remaining_minutes"))
    time_budget_minutes = _as_float(snapshot.get("time_budget_minutes"))
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
            "- Re-check `SPEC.md` once, then end through `./bin/complete_problem.sh --decision beats_both_baselines`.",
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


def _write_goal_status_files(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    workspace: Path,
) -> dict[str, Any]:
    snapshot = _goal_status_snapshot(
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        workspace=workspace,
    )
    _write_workspace_best_sample(
        workspace,
        _best_correct_payload(
            artifact_problem_dir(run_name, level, problem_id) / "history.jsonl"
        ),
    )
    write_json(workspace / "goal_status.json", snapshot)
    write_text(workspace / "GOAL_STATUS.md", _goal_status_markdown(snapshot))
    write_json(
        artifact_agent_dir(run_name, level, problem_id) / "goal_status.json",
        snapshot,
    )
    return snapshot


def _workspace_spec_markdown(
    *,
    problem: Any,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    hardware_markdown_name: str,
) -> str:
    return dedent(
        f"""
        # Orders

        You MUST beat both baselines or exhaust the full budget trying. There is no middle ground.
        NEVER STOP EARLY. DO NOT pause to ask whether you should continue. Continue working until a stopping rule truthfully fires.

        ## Target

        - problem: `{metadata.get("problem_name") or "unknown"}` (level `{metadata["level"]}`, problem `{metadata["problem_id"]}`)
        - eager PyTorch baseline: `{baseline["eager"]["runtime_ms"]}` ms
        - `torch.compile` baseline: `{baseline["compile"]["runtime_ms"]}` ms
        - you succeed only when your best correct runtime is below BOTH numbers
        - optimize `problem_reference.py` by editing only `{CANDIDATE_FILENAME}`
        - the evaluated implementation must be raw custom CUDA/C++ extension code with minimal glue; vendor-library wrappers, Triton, and ATen compute helpers are forbidden
        - correctness and runtime are evaluated on the harness `fp32` path
        - internal mixed-precision math is allowed if the final outputs still pass that `fp32` correctness check

        ## Stopping Rules

        You are forbidden from stopping unless one of these is true:

        1. You beat both baselines -> `./bin/complete_problem.sh --decision beats_both_baselines`.
        2. Budget is nearly exhausted, you profiled at least once, and you tried a new branch informed by that evidence -> `budget_exhausted`.
        3. You profiled, consulted `HARDWARE.md` plus NVIDIA docs, tried a new informed branch, and progress is genuinely stalled -> `stalled`.
        4. The harness or environment is genuinely broken in a way that prevents truthful progress -> `harness_failure`.

        A plain assistant message is NEVER a valid exit. The ONLY exit is `./bin/complete_problem.sh`.
        The harness enforces the budget. You must end cleanly through `./bin/complete_problem.sh --decision budget_exhausted` before remaining time reaches zero.

        ## Loop

        LOOP UNTIL DONE:

        1. Edit `{CANDIDATE_FILENAME}`.
        2. Run `./bin/run_candidate.sh`.
        3. Read `GOAL_STATUS.md`. If both baselines are beaten, stop with success.
        4. For small constant or layout changes, just edit and run again. Timing and profiling are normal and cheap.
        5. Trust the wrappers. If a timing or profiling call is slow, wait for it; do NOT monitor it with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection.
        6. If stuck or uncertain, run `./bin/profile_ncu.sh`, read `HARDWARE.md`, search NVIDIA docs, and try a new branch.
        7. Repeat until a stopping rule fires.

        Re-read this file and `GOAL_STATUS.md` before every major decision.

        ## Budget

        - total budget: `{metadata["time_budget_minutes"]}` minutes
        - remaining budget: see `GOAL_STATUS.md`
        - recorded GPU lock wait time is excluded from the remaining budget
        - tens or hundreds of attempts are normal
        - a failed attempt is not a reason to stop
        - a series of failed attempts is not a reason to stop
        - there is no human confirmation step during this run
        - if you run out of ideas, think harder: re-read `HARDWARE.md`, re-read `profiles/latest.summary.txt`, consult `profiles/latest.details.txt` when needed, re-read `samples/`, revisit NVIDIA docs, combine near-misses, or try a more radical branch

        ## Reference

        - problem code: `problem_reference.py`
        - solution file: `{CANDIDATE_FILENAME}` (only file you edit for evaluation)
        - hardware limits and docs: `{hardware_markdown_name}`
        - constraints and forbidden shortcuts: `AGENTS.md`
        """
    ).strip() + "\n"


def _find_first_value(payload: Any, keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and value not in (None, "", [], {}):
                return value
        for value in payload.values():
            found = _find_first_value(value, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_first_value(value, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _collect_text_fragments(payload: Any, fragments: list[str], limit: int = 6) -> None:
    if len(fragments) >= limit:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if len(fragments) >= limit:
                return
            if key in {"text", "message", "summary", "content", "delta"} and isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    fragments.append(stripped)
                    continue
            _collect_text_fragments(value, fragments, limit=limit)
    elif isinstance(payload, list):
        for value in payload:
            if len(fragments) >= limit:
                return
            _collect_text_fragments(value, fragments, limit=limit)


def _extract_trace_line(
    payload: dict[str, Any],
    line_number: int,
    *,
    tool: str = "codex",
) -> dict[str, Any]:
    tool = _normalize_tool_name(tool)
    if tool == "claude":
        blocks = _claude_content_blocks(payload)
        command = None
        tool_name = None
        for block in _claude_tool_use_blocks(payload):
            tool_name = _claude_tool_name(block) or tool_name
            command = _claude_tool_command(block) or command
        fragments: list[str] = []
        _collect_text_fragments(payload, fragments)
        excerpt = " ".join(fragment.replace("\n", " ") for fragment in fragments).strip()
        if len(excerpt) > 400:
            excerpt = excerpt[:397] + "..."
        serialized = json.dumps(payload, sort_keys=True)
        event_type = str(payload.get("type") or "unknown")
        if payload.get("subtype"):
            event_type = f"{event_type}:{payload.get('subtype')}"
        role = None
        if payload.get("type") == "assistant":
            role = "assistant"
        elif payload.get("type") == "user":
            role = "user"
        return {
            "line": line_number,
            "event_type": event_type,
            "role": role,
            "tool_name": tool_name,
            "command": command[:400] if isinstance(command, str) else None,
            "text": excerpt or None,
            "sample_refs": sorted(set(re.findall(r"sample_(\d+)", serialized))),
            "tool_blocks": len(blocks),
        }

    event_type = (
        _find_first_value(payload, {"type", "event", "kind", "event_type"})
        or "unknown"
    )
    role = _find_first_value(payload, {"role", "sender", "author"})
    tool_name = _find_first_value(
        payload,
        {"tool_name", "recipient_name", "function_name", "command_name"},
    )
    command = _find_first_value(payload, {"command", "cmd", "shell_command"})
    fragments: list[str] = []
    _collect_text_fragments(payload, fragments)
    excerpt = " ".join(fragment.replace("\n", " ") for fragment in fragments).strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:397] + "..."
    serialized = json.dumps(payload, sort_keys=True)
    return {
        "line": line_number,
        "event_type": str(event_type),
        "role": str(role) if role is not None else None,
        "tool_name": str(tool_name) if tool_name is not None else None,
        "command": str(command)[:400] if isinstance(command, str) else None,
        "text": excerpt or None,
        "sample_refs": sorted(set(re.findall(r"sample_(\d+)", serialized))),
    }


def _trace_usage_summary(
    raw_events: list[dict[str, Any]],
    *,
    tool: str = "codex",
) -> dict[str, Any]:
    tool = _normalize_tool_name(tool)
    summary = {
        "turns_completed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
        "uncached_input_tokens": 0,
    }

    if tool == "claude":
        def _claude_usage_summary_from_result(payload: dict[str, Any]) -> dict[str, int] | None:
            turns_completed = int(_as_float(payload.get("num_turns")) or 0)

            model_usage = payload.get("modelUsage")
            if isinstance(model_usage, dict):
                model_usage_blocks = [
                    value for value in model_usage.values() if isinstance(value, dict)
                ]
                if model_usage_blocks:
                    direct_input_tokens = 0
                    cache_creation_input_tokens = 0
                    cache_read_input_tokens = 0
                    output_tokens = 0
                    for block in model_usage_blocks:
                        direct_input_tokens += int(_as_float(block.get("inputTokens")) or 0)
                        cache_creation_input_tokens += int(
                            _as_float(block.get("cacheCreationInputTokens")) or 0
                        )
                        cache_read_input_tokens += int(
                            _as_float(block.get("cacheReadInputTokens")) or 0
                        )
                        output_tokens += int(_as_float(block.get("outputTokens")) or 0)

                    uncached_input_tokens = (
                        direct_input_tokens + cache_creation_input_tokens
                    )
                    return {
                        "turns_completed": turns_completed,
                        "input_tokens": cache_read_input_tokens + uncached_input_tokens,
                        "cached_input_tokens": cache_read_input_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "output_tokens": output_tokens,
                        "uncached_input_tokens": uncached_input_tokens,
                    }

            usage = payload.get("usage")
            if not isinstance(usage, dict):
                return None

            direct_input_tokens = int(_as_float(usage.get("input_tokens")) or 0)
            cache_creation_input_tokens = int(
                _as_float(usage.get("cache_creation_input_tokens")) or 0
            )
            cache_read_input_tokens = int(
                _as_float(usage.get("cache_read_input_tokens")) or 0
            )
            output_tokens = int(_as_float(usage.get("output_tokens")) or 0)
            uncached_input_tokens = direct_input_tokens + cache_creation_input_tokens
            return {
                "turns_completed": turns_completed,
                "input_tokens": cache_read_input_tokens + uncached_input_tokens,
                "cached_input_tokens": cache_read_input_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "output_tokens": output_tokens,
                "uncached_input_tokens": uncached_input_tokens,
            }

        result_candidates: list[dict[str, int]] = []
        for payload in raw_events:
            if payload.get("type") != "result":
                continue
            usage_summary = _claude_usage_summary_from_result(payload)
            if usage_summary is not None:
                result_candidates.append(usage_summary)

        if result_candidates:
            max_turns_completed = max(
                candidate["turns_completed"] for candidate in result_candidates
            )
            summary = max(
                result_candidates,
                key=lambda candidate: (
                    candidate["input_tokens"]
                    + candidate["output_tokens"],
                    candidate["turns_completed"],
                ),
            )
            summary["turns_completed"] = max_turns_completed or summary["turns_completed"] or 1
            return summary

        seen_assistant_message_ids: set[str] = set()
        for payload in raw_events:
            if payload.get("type") != "assistant":
                continue
            message = payload.get("message")
            if not isinstance(message, dict):
                continue
            message_id = message.get("id")
            if isinstance(message_id, str) and message_id:
                if message_id in seen_assistant_message_ids:
                    continue
                seen_assistant_message_ids.add(message_id)
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            direct_input_tokens = int(_as_float(usage.get("input_tokens")) or 0)
            cache_creation_input_tokens = int(
                _as_float(usage.get("cache_creation_input_tokens")) or 0
            )
            cache_read_input_tokens = int(
                _as_float(usage.get("cache_read_input_tokens")) or 0
            )
            summary["turns_completed"] += 1
            summary["cached_input_tokens"] += cache_read_input_tokens
            summary["cache_creation_input_tokens"] += cache_creation_input_tokens
            summary["uncached_input_tokens"] += (
                direct_input_tokens + cache_creation_input_tokens
            )
            summary["output_tokens"] += int(_as_float(usage.get("output_tokens")) or 0)

        summary["input_tokens"] = (
            summary["cached_input_tokens"] + summary["uncached_input_tokens"]
        )
        return summary

    for payload in raw_events:
        if payload.get("type") != "turn.completed":
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        summary["turns_completed"] += 1
        summary["input_tokens"] += int(_as_float(usage.get("input_tokens")) or 0)
        summary["cached_input_tokens"] += int(
            _as_float(usage.get("cached_input_tokens")) or 0
        )
        summary["output_tokens"] += int(_as_float(usage.get("output_tokens")) or 0)

    summary["uncached_input_tokens"] = max(
        summary["input_tokens"] - summary["cached_input_tokens"],
        0,
    )
    return summary


def _trace_cost_usd(
    raw_events: list[dict[str, Any]],
    *,
    tool: str = "codex",
) -> float | None:
    tool = _normalize_tool_name(tool)
    if tool != "claude":
        return None

    max_cost = None
    for payload in raw_events:
        if payload.get("type") != "result":
            continue

        explicit_cost = _as_float(payload.get("total_cost_usd"))

        model_usage = payload.get("modelUsage")
        model_usage_cost = None
        if isinstance(model_usage, dict):
            usage_blocks = [
                value for value in model_usage.values() if isinstance(value, dict)
            ]
            if usage_blocks:
                model_usage_cost = sum(
                    float(_as_float(block.get("costUSD")) or 0.0)
                    for block in usage_blocks
                )

        cost = explicit_cost
        if cost is None and model_usage_cost is not None:
            cost = model_usage_cost
        if cost is None:
            continue

        max_cost = cost if max_cost is None else max(max_cost, cost)

    return max_cost


_ALLOWED_WORKSPACE_COMMAND_PREFIXES = (
    "./bin/problem_info.sh",
    "./bin/run_candidate.sh",
    "./bin/profile_ncu.sh",
    "./bin/goal_status.sh",
    "./bin/best_result.sh",
    "./bin/complete_problem.sh",
    "cat ",
    "cat\t",
    "cat\n",
    "sed ",
    "head ",
    "tail ",
    "ls",
    "pwd",
    "rg ",
    "grep ",
    "find ",
    "wc ",
    "nl ",
    "cp ",
)

_GPU_WRAPPER_PREFIXES = (
    "./bin/run_candidate.sh",
    "./bin/profile_ncu.sh",
)

_FORBIDDEN_INSPECTION_MARKERS = (
    ".ptx",
    ".cubin",
    "cuobjdump",
    "nvdisasm",
    "torchinductor",
    "torch_compile_debug",
    "triton",
    "inductor",
)

_FORBIDDEN_MONITORING_PREFIXES = (
    "ps",
    "pgrep",
    "top",
    "htop",
    "nvidia-smi",
    "strace",
)

_FORBIDDEN_MONITORING_MARKERS = (
    "/proc",
)

_WORKSPACE_WRAPPER_NAMES = {
    "./bin/problem_info.sh": "problem_info_calls",
    "./bin/run_candidate.sh": "run_candidate_calls",
    "./bin/profile_ncu.sh": "profile_ncu_calls",
    "./bin/goal_status.sh": "goal_status_calls",
    "./bin/best_result.sh": "best_result_calls",
    "./bin/complete_problem.sh": "complete_problem_calls",
}

_CLAUDE_ALLOWED_BASH_PREFIXES = tuple(_WORKSPACE_WRAPPER_NAMES)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _extract_shell_snippet(command: str) -> str:
    prefix = "/bin/bash -lc "
    if not command.startswith(prefix):
        return command.strip()
    snippet = command[len(prefix) :].strip()
    if len(snippet) >= 2 and snippet[0] in {"'", '"'} and snippet[-1] == snippet[0]:
        snippet = snippet[1:-1]
    return snippet.strip()


def _split_leading_cd(snippet: str) -> tuple[str, str] | None:
    match = re.match(
        r"^cd\s+(?P<path>'[^']*'|\"[^\"]*\"|[^;&]+?)\s*(?:&&|;)\s*(?P<rest>.+)$",
        snippet,
        flags=re.DOTALL,
    )
    if not match:
        return None
    raw_path = match.group("path").strip()
    if len(raw_path) >= 2 and raw_path[0] in {"'", '"'} and raw_path[-1] == raw_path[0]:
        raw_path = raw_path[1:-1]
    return raw_path, match.group("rest").strip()


def _normalize_workspace_snippet(
    snippet: str,
    workspace: Path,
) -> tuple[str | None, str | None]:
    stripped = snippet.strip()
    cd_parts = _split_leading_cd(stripped)
    if cd_parts is None:
        return stripped, None

    raw_target, rest = cd_parts
    target = Path(raw_target).expanduser()
    if not target.is_absolute():
        target = workspace / target
    try:
        resolved_target = target.resolve()
    except OSError:
        return None, "command execution left the problem workspace"
    if not _is_relative_to(resolved_target, workspace.resolve()):
        return None, "command execution left the problem workspace"
    return rest, None


def _empty_trace_counts() -> dict[str, Any]:
    return {
        "command_executions": 0,
        "file_change_events": 0,
        "wrapper_commands": 0,
        "gpu_wrapper_commands": 0,
        "problem_info_calls": 0,
        "run_candidate_calls": 0,
        "profile_ncu_calls": 0,
        "goal_status_calls": 0,
        "best_result_calls": 0,
        "complete_problem_calls": 0,
        "spawn_agent_calls": 0,
        "wait_calls": 0,
        "web_search_calls": 0,
        "subagents_spawned": 0,
    }


def _trace_counts_from_entries(
    raw_event_entries: list[tuple[int, dict[str, Any]]],
    *,
    tool: str = "codex",
) -> dict[str, Any]:
    tool = _normalize_tool_name(tool)
    counts = _empty_trace_counts()
    spawned_threads: set[str] = set()

    if tool == "claude":
        explicit_web_search_calls = 0
        for _, payload in raw_event_entries:
            for block in _claude_tool_use_blocks(payload):
                tool_name = _claude_tool_name(block).strip().lower()
                if tool_name == "bash":
                    command = _claude_tool_command(block)
                    if not isinstance(command, str):
                        continue
                    counts["command_executions"] += 1
                    snippet = command.strip()
                    cd_parts = _split_leading_cd(snippet)
                    effective_snippet = (cd_parts[1] if cd_parts else snippet).strip()
                    for prefix, key in _WORKSPACE_WRAPPER_NAMES.items():
                        if effective_snippet.startswith(prefix):
                            counts[key] += 1
                            counts["wrapper_commands"] += 1
                            if prefix in _GPU_WRAPPER_PREFIXES:
                                counts["gpu_wrapper_commands"] += 1
                            break
                    continue

                if tool_name in {"edit", "multiedit", "write"}:
                    counts["file_change_events"] += 1
                    continue

                if tool_name in {"websearch", "web_search"}:
                    explicit_web_search_calls += 1
                    continue

                if tool_name in {"task", "subagent", "agent"}:
                    counts["spawn_agent_calls"] += 1
                    counts["subagents_spawned"] += 1
                    continue

        usage_web_search_calls = 0
        for _, payload in raw_event_entries:
            if payload.get("type") != "result":
                continue
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                continue
            server_tool_use = usage.get("server_tool_use")
            if not isinstance(server_tool_use, dict):
                continue
            usage_web_search_calls = max(
                usage_web_search_calls,
                int(_as_float(server_tool_use.get("web_search_requests")) or 0),
            )

        counts["web_search_calls"] = max(
            explicit_web_search_calls,
            usage_web_search_calls,
        )

        return counts

    for _, payload in raw_event_entries:
        if payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "command_execution":
            command = item.get("command")
            if not isinstance(command, str):
                continue
            counts["command_executions"] += 1
            snippet = _extract_shell_snippet(command)
            cd_parts = _split_leading_cd(snippet)
            effective_snippet = (cd_parts[1] if cd_parts else snippet).strip()
            for prefix, key in _WORKSPACE_WRAPPER_NAMES.items():
                if effective_snippet.startswith(prefix):
                    counts[key] += 1
                    counts["wrapper_commands"] += 1
                    if prefix in _GPU_WRAPPER_PREFIXES:
                        counts["gpu_wrapper_commands"] += 1
                    break
            continue

        if item_type == "file_change":
            counts["file_change_events"] += 1
            continue

        if item_type == "web_search":
            counts["web_search_calls"] += 1
            continue

        if item_type == "collab_tool_call":
            tool_name = str(item.get("tool") or "").strip().lower()
            if tool_name == "spawn_agent":
                counts["spawn_agent_calls"] += 1
                receiver_ids = item.get("receiver_thread_ids")
                if isinstance(receiver_ids, list):
                    for receiver_id in receiver_ids:
                        if isinstance(receiver_id, str) and receiver_id:
                            spawned_threads.add(receiver_id)
            elif tool_name == "wait":
                counts["wait_calls"] += 1
            elif tool_name == "web_search":
                counts["web_search_calls"] += 1
            continue

        tool_name = _find_first_value(
            payload,
            {"tool", "tool_name", "recipient_name", "function_name", "command_name"},
        )
        if isinstance(tool_name, str) and "web_search" in tool_name.lower():
            counts["web_search_calls"] += 1

    counts["subagents_spawned"] = len(spawned_threads)
    return counts


def _audit_trace(
    *,
    raw_event_entries: list[tuple[int, dict[str, Any]]],
    workspace: Path,
    tool: str = "codex",
) -> dict[str, Any]:
    tool = _normalize_tool_name(tool)
    workspace = workspace.resolve()
    allowed_edit_paths = {
        (workspace / CANDIDATE_FILENAME).resolve(),
    }
    allowed_read_paths = _allowed_workspace_read_paths(workspace)
    violations: list[dict[str, Any]] = []
    trace_counts = _trace_counts_from_entries(raw_event_entries, tool=tool)
    web_searches = _web_searches_from_entries(raw_event_entries, tool=tool)

    if tool == "claude":
        for search in web_searches:
            domains = [
                str(domain).strip().lower()
                for domain in search.get("domains", [])
                if isinstance(domain, str) and domain.strip()
            ]
            disallowed_domains = [
                domain
                for domain in domains
                if not any(
                    domain == allowed or domain.endswith(f".{allowed}")
                    for allowed in _ALLOWED_WEB_SEARCH_HOSTS
                )
            ]
            if disallowed_domains:
                violations.append(
                    {
                        "line": search["line"],
                        "kind": "web_search_outside_allowed_domains",
                        "domains": disallowed_domains,
                        "message": "web search touched domains outside the allowed NVIDIA docs scope",
                    }
                )

    if tool == "claude":
        for line_number, payload in raw_event_entries:
            for block in _claude_tool_use_blocks(payload):
                tool_name = _claude_tool_name(block).strip().lower()
                if tool_name == "read":
                    raw_path = _claude_tool_path(block)
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        continue
                    read_path = Path(raw_path).expanduser()
                    if not read_path.is_absolute():
                        read_path = workspace / read_path
                    read_path = read_path.resolve()
                    if read_path in allowed_read_paths or _is_allowed_workspace_read(
                        read_path, workspace
                    ):
                        continue
                    violations.append(
                        {
                            "line": line_number,
                            "kind": "read_outside_allowed_set",
                            "path": raw_path,
                            "message": "file read touched a path outside the allowed workspace reading set",
                        }
                    )
                    continue

                if tool_name == "bash":
                    command = _claude_tool_command(block)
                    if not isinstance(command, str) or not command.strip():
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "empty_command",
                                "command": None,
                                "message": "command execution was empty",
                            }
                        )
                        continue

                    snippet, outside_workspace_message = _normalize_workspace_snippet(
                        command.strip(),
                        workspace,
                    )
                    if outside_workspace_message is not None:
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "command_outside_workspace",
                                "command": command[:400],
                                "message": outside_workspace_message,
                            }
                        )
                        continue

                    lowered_snippet = snippet.lower().strip()
                    if any(
                        lowered_snippet == prefix
                        or lowered_snippet.startswith(prefix + " ")
                        for prefix in _FORBIDDEN_MONITORING_PREFIXES
                    ) or any(
                        marker in lowered_snippet for marker in _FORBIDDEN_MONITORING_MARKERS
                    ):
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "forbidden_system_monitoring",
                                "command": command[:400],
                                "message": "system monitoring commands are forbidden; trust the local wrappers instead",
                            }
                        )
                        continue

                    if not snippet.startswith(_CLAUDE_ALLOWED_BASH_PREFIXES):
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "command_not_allowed",
                                "command": command[:400],
                                "message": "Claude Bash must be limited to the local ./bin/*.sh wrapper commands; use the Read tool for allowed files",
                            }
                        )
                        continue

                    forbidden_hit = False
                    for marker in _FORBIDDEN_INSPECTION_MARKERS:
                        if marker in lowered_snippet:
                            violations.append(
                                {
                                    "line": line_number,
                                    "kind": "forbidden_compiled_artifact_inspection",
                                    "command": command[:400],
                                    "message": "command inspected compiled PTX, Triton, Inductor, or similar generated artifacts",
                                }
                            )
                            forbidden_hit = True
                            break
                    if forbidden_hit:
                        continue

                    for match in re.findall(
                        r"(?<![A-Za-z0-9._-])(/[A-Za-z0-9._~/-]+)",
                        snippet,
                    ):
                        absolute = Path(match)
                        if not _is_relative_to(absolute, workspace):
                            violations.append(
                                {
                                    "line": line_number,
                                    "kind": "absolute_path_escape",
                                    "command": command[:400],
                                    "message": f"command referenced absolute path outside workspace: {match}",
                                }
                            )
                            break
                    continue

                if tool_name in {"edit", "multiedit", "write"}:
                    raw_path = _claude_tool_path(block)
                    if not isinstance(raw_path, str):
                        continue
                    changed_path = Path(raw_path).expanduser()
                    if not changed_path.is_absolute():
                        changed_path = workspace / changed_path
                    changed_path = changed_path.resolve()
                    if changed_path not in allowed_edit_paths:
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "file_change_outside_allowed_set",
                                "path": raw_path,
                                "message": "file change touched a path outside candidate_model_new.py",
                            }
                        )

        summary = (
            "trace stayed within the enforced workspace command and file-change contract"
            if not violations
            else violations[0]["message"]
        )
        return {
            "valid": not violations,
            "summary": summary,
            "num_violations": len(violations),
            "command_count": trace_counts["command_executions"],
            "file_change_count": trace_counts["file_change_events"],
            "wrapper_commands": trace_counts["wrapper_commands"],
            "gpu_wrapper_commands": trace_counts["gpu_wrapper_commands"],
            "trace_counts": trace_counts,
            "violations": violations,
        }

    for line_number, payload in raw_event_entries:
        if payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "command_execution":
            command = item.get("command")
            if not isinstance(command, str):
                continue
            snippet = _extract_shell_snippet(command)
            if not snippet:
                violations.append(
                    {
                        "line": line_number,
                        "kind": "empty_command",
                        "command": command[:400],
                        "message": "command execution was empty",
                    }
                )
                continue

            snippet, outside_workspace_message = _normalize_workspace_snippet(
                snippet,
                workspace,
            )
            if outside_workspace_message is not None:
                violations.append(
                    {
                        "line": line_number,
                        "kind": "command_outside_workspace",
                        "command": command[:400],
                        "message": outside_workspace_message,
                    }
                )
                continue

            lowered_snippet = snippet.lower().strip()
            if any(
                lowered_snippet == prefix
                or lowered_snippet.startswith(prefix + " ")
                for prefix in _FORBIDDEN_MONITORING_PREFIXES
            ) or any(marker in lowered_snippet for marker in _FORBIDDEN_MONITORING_MARKERS):
                violations.append(
                    {
                        "line": line_number,
                        "kind": "forbidden_system_monitoring",
                        "command": command[:400],
                        "message": "system monitoring commands are forbidden; trust the local wrappers instead",
                    }
                )
                continue

            if not snippet.startswith(_ALLOWED_WORKSPACE_COMMAND_PREFIXES):
                violations.append(
                    {
                        "line": line_number,
                        "kind": "command_not_allowed",
                        "command": command[:400],
                        "message": "command is outside the allowed workspace command set",
                    }
                )
                continue

            for marker in _FORBIDDEN_INSPECTION_MARKERS:
                if marker in lowered_snippet:
                    violations.append(
                        {
                            "line": line_number,
                            "kind": "forbidden_compiled_artifact_inspection",
                            "command": command[:400],
                            "message": "command inspected compiled PTX, Triton, Inductor, or similar generated artifacts",
                        }
                    )
                    break
            else:
                for match in re.findall(r"(?<![A-Za-z0-9._-])(/[A-Za-z0-9._~/-]+)", snippet):
                    absolute = Path(match)
                    if not _is_relative_to(absolute, workspace):
                        violations.append(
                            {
                                "line": line_number,
                                "kind": "absolute_path_escape",
                                "command": command[:400],
                                "message": f"command referenced absolute path outside workspace: {match}",
                            }
                        )
                        break

                continue

            continue

        elif item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, dict):
                    continue
                raw_path = change.get("path")
                if not isinstance(raw_path, str):
                    continue
                changed_path = Path(raw_path).resolve()
                if changed_path not in allowed_edit_paths:
                    violations.append(
                        {
                            "line": line_number,
                            "kind": "file_change_outside_allowed_set",
                            "path": raw_path,
                            "message": "file change touched a path outside candidate_model_new.py",
                        }
                    )

    summary = (
        "trace stayed within the enforced workspace command and file-change contract"
        if not violations
        else violations[0]["message"]
    )
    return {
        "valid": not violations,
        "summary": summary,
        "num_violations": len(violations),
        "command_count": trace_counts["command_executions"],
        "file_change_count": trace_counts["file_change_events"],
        "wrapper_commands": trace_counts["wrapper_commands"],
        "gpu_wrapper_commands": trace_counts["gpu_wrapper_commands"],
        "trace_counts": trace_counts,
        "violations": violations,
    }


def _apply_trace_audit_to_completion(
    completion_payload: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    trace_counts = (
        audit.get("trace_counts")
        if isinstance(audit.get("trace_counts"), dict)
        else _empty_trace_counts()
    )
    completion_payload["audit"] = audit
    completion_payload["trace_counts"] = trace_counts
    if audit.get("valid", True):
        return completion_payload

    if "reported_decision" not in completion_payload:
        completion_payload["reported_decision"] = completion_payload.get("decision")
    if "reported_summary" not in completion_payload:
        completion_payload["reported_summary"] = completion_payload.get("summary")
    if "reported_success" not in completion_payload:
        completion_payload["reported_success"] = completion_payload.get("success")

    completion_payload["decision"] = "harness_failure"
    completion_payload["success"] = False
    completion_payload["summary"] = f"invalidated by trace audit: {audit.get('summary')}"
    return completion_payload


def _annotate_completion_outcomes(
    completion_payload: dict[str, Any],
) -> dict[str, Any]:
    goal_status = completion_payload.get("goal_status")
    if isinstance(goal_status, dict):
        raw_best_runtime = _as_float(goal_status.get("best_correct_runtime_ms"))
        raw_beats_eager = bool(goal_status.get("beats_eager"))
        raw_beats_compile = bool(goal_status.get("beats_compile"))
        raw_beats_both = bool(goal_status.get("beats_both"))
    else:
        raw_best_runtime = None
        raw_beats_eager = False
        raw_beats_compile = False
        raw_beats_both = False

    completion_payload["raw_best_correct_runtime_ms"] = raw_best_runtime
    completion_payload["raw_beats_eager"] = raw_beats_eager
    completion_payload["raw_beats_compile"] = raw_beats_compile
    completion_payload["raw_beats_both"] = raw_beats_both
    completion_payload["outside_harness_success"] = raw_beats_both
    return completion_payload


def _substantial_budget_remaining(snapshot: dict[str, Any]) -> bool:
    remaining_minutes = _as_float(snapshot.get("remaining_minutes"))
    time_budget_minutes = _as_float(snapshot.get("time_budget_minutes"))
    if remaining_minutes is None or time_budget_minutes is None:
        return False
    return remaining_minutes > max(60.0, time_budget_minutes * 0.25)


def _apply_completion_policy(
    completion_payload: dict[str, Any],
) -> dict[str, Any]:
    trace_counts = completion_payload.get("trace_counts")
    goal_status = completion_payload.get("goal_status")
    if not isinstance(trace_counts, dict) or not isinstance(goal_status, dict):
        return completion_payload

    if completion_payload.get("decision") != "stalled":
        return completion_payload
    if not _substantial_budget_remaining(goal_status):
        return completion_payload

    profile_calls = int(_as_float(trace_counts.get("profile_ncu_calls")) or 0)
    if profile_calls >= 1:
        return completion_payload

    if "reported_decision" not in completion_payload:
        completion_payload["reported_decision"] = completion_payload.get("decision")
    if "reported_summary" not in completion_payload:
        completion_payload["reported_summary"] = completion_payload.get("summary")
    if "reported_success" not in completion_payload:
        completion_payload["reported_success"] = completion_payload.get("success")

    completion_payload["decision"] = "harness_failure"
    completion_payload["success"] = False
    completion_payload["summary"] = (
        "invalidated by completion policy: `stalled` is not allowed while substantial "
        "budget remains and no `./bin/profile_ncu.sh` call was recorded"
    )
    return completion_payload

def _parse_pass_k_list(raw: str) -> list[int]:
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


def _pass_at_k_estimate(n: int, c: int, k: int) -> float | None:
    if n <= 0 or k <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    numerator = math.comb(n - c, k)
    denominator = math.comb(n, k)
    return 1.0 - (numerator / denominator)


def _problem_workspace_paths(
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


def _workspace_candidate_path(workspace: Path) -> Path:
    return workspace / CANDIDATE_FILENAME


def _workspace_samples_dir(workspace: Path) -> Path:
    return workspace / "samples"


def _workspace_profiles_dir(workspace: Path) -> Path:
    return workspace / "profiles"


def _workspace_relpath(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _next_workspace_profile_index(workspace: Path) -> int:
    profiles_dir = _workspace_profiles_dir(workspace)
    max_index = 0
    for child in profiles_dir.glob("profile_*.json"):
        match = re.fullmatch(r"profile_(\d+)\.json", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _write_workspace_script(path: Path, content: str) -> None:
    write_text(path, content)
    make_executable(path)


def _run_subprocess_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def _generate_workspace_agents_md(args: argparse.Namespace) -> str:
    return dedent(
        f"""
        # Solver Instructions

        You are an autonomous optimizer for exactly one KernelBench problem.

        Assignment:

        - run name: `{args.run_name}`
        - level: `{args.level}`
        - problem id: `{args.problem_id}`
        - dataset source: `{args.dataset_src}`
        - available GPU slots for execution: `{args.num_gpus}`
        - reported GPU name: `{args.gpu_name or "not provided"}`
        - model budget: `{args.time_budget_minutes}` minutes

        Scope:

        - stay inside this workspace
        - read `SPEC.md` and `HARDWARE.md` first, then keep referring back to them so the goal does not drift
        - the project goal is narrow: test whether raw custom CUDA code, without vendor-library wrappers or ATen compute helpers, can beat optimized PyTorch baselines
        - the harness evaluates the problem in `fp32`; optimize for that judged path
        - do not inspect unrelated problems
        - do not inspect maintainer docs in the repository unless explicitly asked
        - do not inspect harness source code, wrapper scripts, or repository internals to reverse-engineer the evaluator
        - do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas
        - do not modify files outside this workspace
        - use `problem_reference.py` as the reference architecture
        - edit only `{CANDIDATE_FILENAME}` for the actual solution
        - in `{CANDIDATE_FILENAME}`, edit only the marked editable blocks and keep the fixed scaffold unchanged

        Local commands:

        - `./bin/problem_info.sh`
        - `./bin/run_candidate.sh`
        - `./bin/profile_ncu.sh`
        - `./bin/goal_status.sh`
        - `./bin/best_result.sh`
        - `./bin/complete_problem.sh --decision ... --summary "..."`

        Valid completion decisions:

        - `beats_both_baselines`
        - `beats_eager_only`
        - `beats_compile_only`
        - `budget_exhausted`
        - `stalled`
        - `harness_failure`

        `failed_to_generate` is reserved for the launcher if the agent exits without writing completion.

        Allowed reads:

        - `AGENTS.md`
        - `SPEC.md`
        - `HARDWARE.md`
        - `GOAL_STATUS.md`
        - `goal_status.json`
        - `problem.json`
        - `problem_reference.py`
        - `{CANDIDATE_FILENAME}`
        - `samples/`
        - `profiles/`
        - `baseline.json`

        Allowed edits:

        - `{CANDIDATE_FILENAME}`

        Web policy:

        - if live web search is enabled, use it only for NVIDIA docs
        - allowed domain is `docs.nvidia.com`
        - you do not need user confirmation to use the allowed NVIDIA docs search path
        - do not use shell networking for research or downloads
        - do not run `curl`, `wget`, package installers, browsers, or ad hoc Python networking code
        - do not use online papers, blogs, forums, or code repositories for solution ideas
        - if you need documentation, prefer the hosted web-search tool and keep it within the allowed NVIDIA domain

        Subagents:

        - use `runner` aggressively when timing output, failed-attempt logs, or branch exploration would pollute your context
        - use `profiler` when `ncu` output would pollute your context
        - if progress slows, use subagents to explore new branches instead of repeating the same local search loop
        - subagents must stay on this problem only

        Rules:

        - LOOP UNTIL DONE. NEVER STOP EARLY.
        - do NOT ask the user whether you should continue
        - every evaluated attempt must go through `./bin/run_candidate.sh`
        - there is no limit on wrapper use; even quick sanity checks and simple performance tests must go through the local wrappers
        - timing and profiling are routine tools, not expensive last resorts
        - profiling is allowed for small constant or layout changes, not just final candidates
        - large search loops are expected when needed; exploring many tile sizes, stage counts, vector widths, or block layouts is normal
        - wrapper commands are authoritative; if a wrapper is slow, wait for it and trust it
        - do not run `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, inspect `/proc`, or inspect build directories to monitor wrapper progress
        - all GPU work is serialized through the local wrappers; do not bypass them
        - do not run ad hoc Python, shell, or benchmarking commands for GPU experiments outside the local wrappers
        - do not use `python -c`, heredoc Python, or `git diff` as a substitute for reading files or testing ideas
        - do not read or import `kernel_bench_experiment_agents`, `kernelbench`, or other repository code to infer hidden evaluator details
        - do not inspect `bin/*.sh`; execute the local commands directly instead
        - after `./bin/profile_ncu.sh`, read `profiles/latest.summary.txt` first, then `profiles/latest.details.txt` or `profiles/latest.raw.csv` only if needed; do not inspect artifact-tree profiler outputs, binary report files, or parse profiler files with ad hoc shell/Python commands
        - do not ask the user what to do next during the run
        - do not stop, hand control back, or declare completion early
        - do not end the run with a plain assistant message; plain summaries are not terminal actions
        - you may continue autonomously for the full model budget in this workspace; long runs are expected and allowed
        - there is no human-in-the-loop confirmation step during this run; act autonomously inside the budget
        - continue autonomously until you either beat both baselines or terminate through `./bin/complete_problem.sh` with a truthful non-success decision
        - check `./bin/goal_status.sh` before deciding whether to stop
        - re-read `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` before any major strategy change and before any stop decision
        - measured status comes from `goal_status.json` and `GOAL_STATUS.md`, including the remaining time budget
        - the harness enforces the corrected remaining budget; do not run past it
        - if remaining time is close to zero and the target is still unresolved, call `./bin/complete_problem.sh --decision budget_exhausted` before the harness stops the run
        - do not call the search stalled while substantial budget remains unless you have already profiled a strong candidate, consulted `HARDWARE.md` and NVIDIA documentation, and then tried a new branch informed by that evidence
        - if you run out of ideas, think harder: re-read `HARDWARE.md`, re-read `profiles/latest.summary.txt`, consult `profiles/latest.details.txt` when needed, re-read prior samples in `samples/`, revisit NVIDIA docs, combine prior near-misses, or try a more radical branch
        - treat shell networking as forbidden even if the launcher allows it for cluster compatibility
        - every measured attempt is mirrored into `samples/sample_<id>.py`; use those workspace-local copies instead of inspecting `runs/`
        - `{CANDIDATE_FILENAME}` must define `ModelNew` and raw custom CUDA/C++ extension code only
        - internal mixed-precision math is allowed if the candidate still passes the harness `fp32` correctness checks
        - minimal extension glue is allowed, but ATen compute helpers and native ops are out of scope
        - do not redefine `Model`, `get_inputs`, or `get_init_inputs`
        - do not use `torch.compile`, Triton, environment-variable changes, torch backend flags, pure PyTorch matmul shortcuts, `register_buffer`, output-buffer reuse tricks, cuBLAS, cuBLASLt, CUTLASS, or ATen compute wrappers
        """
    ).strip() + "\n"


def _generate_initial_prompt(args: argparse.Namespace) -> str:
    return dedent(
        f"""
        Optimize exactly one KernelBench problem.

        - run name: {args.run_name}
        - level: {args.level}
        - problem id: {args.problem_id}
        - dataset source: {args.dataset_src}
        - time budget: {args.time_budget_minutes} minutes
        - GPU slots available for execution: {args.num_gpus}

        Start by reading:

        - `AGENTS.md`
        - `SPEC.md`
        - `HARDWARE.md`
        - `GOAL_STATUS.md`
        - `problem_reference.py`
        - `{CANDIDATE_FILENAME}`

        The experiment target is strict: raw custom CUDA code only, with minimal Python/C++ extension glue as needed, but no vendor-library wrappers such as cuBLAS, cuBLASLt, CUTLASS, Triton, or ATen compute helpers.
        The harness evaluates correctness and runtime on the `fp32` path. Internal mixed-precision math is allowed if the final outputs still pass that `fp32` correctness check.

        Remaining budget is tracked in `GOAL_STATUS.md`; you are allowed to keep working autonomously for hours until the budget is genuinely exhausted. Prefer using the `runner` and `profiler` subagents to keep your own context clean during long searches.
        There is no human confirmation step during this run. Keep acting autonomously inside the budget, and keep re-reading `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` so the goal, hardware limits, and current status stay explicit.
        NEVER STOP EARLY. DO NOT ask whether you should continue. Continue working until you either beat both baselines or truthfully terminate through `./bin/complete_problem.sh`.
        The harness enforces the corrected remaining budget. Do not let remaining time reach zero without first calling `./bin/complete_problem.sh --decision budget_exhausted` if the target is still unresolved.

        If you think you are stalled while substantial budget remains, first consult `HARDWARE.md`, then the linked NVIDIA docs, run `./bin/profile_ncu.sh` on a strong candidate, and try at least one new branch informed by that evidence. Only then may you consider `stalled`.

        Large micro-searches are expected when needed. You are allowed to spend tens or hundreds of timing runs exploring tile sizes, stage counts, vector widths, warp layouts, or launch shapes, as long as every measured attempt goes through the local wrappers. Timing and profiling are normal tools, not expensive last resorts.

        If you run out of ideas, think harder: re-read `HARDWARE.md`, re-read `profiles/latest.summary.txt`, consult `profiles/latest.details.txt` when needed, re-read prior samples in `samples/`, revisit NVIDIA docs, combine prior near-misses, or try a more radical branch.

        Only edit `{CANDIDATE_FILENAME}` for the solution, and only inside its marked editable blocks. Do not inspect harness internals. Do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas. Use the local `./bin/*.sh` commands exactly as provided, avoid shell-network commands entirely, do not run ad hoc GPU experiments outside the wrappers, do not use Python snippets or `git diff` for quick checks, do not run `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, inspect `/proc`, inspect build directories, or use ad hoc shell parsing commands to monitor wrapper progress or mine profiler outputs, and do not finish without `./bin/complete_problem.sh`. After profiling, read `profiles/latest.summary.txt` first, then `profiles/latest.details.txt` or `profiles/latest.raw.csv` only if needed. Revisit prior measured attempts through `samples/sample_<id>.py` or `samples/best_sample.py`, never through `runs/`. Do not end with a plain assistant summary. Valid solver-written completion decisions are `beats_both_baselines`, `beats_eager_only`, `beats_compile_only`, `budget_exhausted`, `stalled`, and `harness_failure`.
        """
    ).strip() + "\n"


def _workspace_wrapper_common(
    *,
    kernelbench_python: str,
    project_root: Path,
    kernelbench_root: str | None,
) -> str:
    kb_root_line = (
        f'KERNELBENCH_ROOT={shlex.quote(str(Path(kernelbench_root).resolve()))}\n'
        if kernelbench_root
        else 'KERNELBENCH_ROOT=""\n'
    )
    return dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        WORKSPACE="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
        PROJECT_ROOT={shlex.quote(str(project_root))}
        KERNELBENCH_PYTHON={shlex.quote(kernelbench_python)}
        {kb_root_line.rstrip()}
        PROJECT_PYTHONPATH="${{PROJECT_ROOT}}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"
        """
    ).lstrip()


def _generate_run_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    kernelbench_root: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=kernelbench_root,
    )
    kb_arg = (
        '  --kernelbench-root "${KERNELBENCH_ROOT}" \\\n'
        if kernelbench_root
        else ""
    )
    return common + dedent(
        f"""
        CANDIDATE="${{WORKSPACE}}/{CANDIDATE_FILENAME}"

        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli run-candidate \\
          --candidate "${{CANDIDATE}}" \\
          --run-name {shlex.quote(run_name)} \\
          --level {level} \\
          --problem-id {problem_id} \\
          --dataset-src {shlex.quote(dataset_src)} \\
          --workspace "${{WORKSPACE}}" \\
{kb_arg}          --num-gpu-slots {num_gpus} \\
          "$@"
        echo ">>> Re-read GOAL_STATUS.md and SPEC.md before your next decision."
        echo ">>> Timing and profiling are normal tools. Use ./bin/run_candidate.sh and ./bin/profile_ncu.sh freely."
        echo ">>> Trust the wrapper result. Do not monitor progress with ps, pgrep, nvidia-smi, strace, /proc, or build-tree inspection."
        """
    )


def _generate_profile_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    kernelbench_root: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=kernelbench_root,
    )
    kb_arg = (
        '  --kernelbench-root "${KERNELBENCH_ROOT}" \\\n'
        if kernelbench_root
        else ""
    )
    return common + dedent(
        f"""
        CANDIDATE="${{WORKSPACE}}/{CANDIDATE_FILENAME}"

        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli profile-ncu \\
          --candidate "${{CANDIDATE}}" \\
          --run-name {shlex.quote(run_name)} \\
          --level {level} \\
          --problem-id {problem_id} \\
          --dataset-src {shlex.quote(dataset_src)} \\
          --workspace "${{WORKSPACE}}" \\
{kb_arg}          --num-gpu-slots {num_gpus} \\
          "$@"
        echo ">>> Read profiles/latest.summary.txt first, then profiles/latest.details.txt if needed. Re-read HARDWARE.md and GOAL_STATUS.md."
        echo ">>> Trust the wrapper result. Do not monitor progress with ps, pgrep, nvidia-smi, strace, /proc, or build-tree inspection."
        """
    )


def _generate_info_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    kernelbench_root: str | None,
    level: int,
    problem_id: int,
    dataset_src: str,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=kernelbench_root,
    )
    kb_arg = (
        '  --kernelbench-root "${KERNELBENCH_ROOT}" \\\n'
        if kernelbench_root
        else ""
    )
    return common + dedent(
        f"""
        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli problem-info \\
          --level {level} \\
          --problem-id {problem_id} \\
          --dataset-src {shlex.quote(dataset_src)} \\
{kb_arg}          "$@"
        """
    )


def _generate_goal_status_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    run_name: str,
    level: int,
    problem_id: int,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=None,
    )
    return common + dedent(
        f"""
        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli goal-status \\
          --run-name {shlex.quote(run_name)} \\
          --level {level} \\
          --problem-id {problem_id} \\
          --workspace "${{WORKSPACE}}" \\
          "$@"
        """
    )


def _generate_best_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    run_name: str,
    level: int,
    problem_id: int,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=None,
    )
    return common + dedent(
        f"""
        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli best-result \\
          --run-name {shlex.quote(run_name)} \\
          --level {level} \\
          --problem-id {problem_id} \\
          "$@"
        """
    )


def _generate_complete_wrapper(
    *,
    kernelbench_python: str,
    project_root: Path,
    run_name: str,
    level: int,
    problem_id: int,
) -> str:
    common = _workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        project_root=project_root,
        kernelbench_root=None,
    )
    return common + dedent(
        f"""
        PYTHONPATH="${{PROJECT_PYTHONPATH}}" "${{KERNELBENCH_PYTHON}}" -m kernel_bench_experiment_agents.cli complete-problem \\
          --run-name {shlex.quote(run_name)} \\
          --level {level} \\
          --problem-id {problem_id} \\
          --workspace "${{WORKSPACE}}" \\
          "$@"
        """
    )


def command_prepare_problem_workspace(args: argparse.Namespace) -> None:
    resolved_kernelbench_root = str(kernelbench_root(args.kernelbench_root))
    try:
        hardware = resolve_hardware_spec(args.gpu_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=resolved_kernelbench_root,
    )
    paths = _problem_workspace_paths(
        args.run_name,
        args.level,
        args.problem_id,
        args.workspace_root,
    )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for stale_name in ("latest_candidate.py", "completion.json"):
        stale_path = paths["workspace"] / stale_name
        if stale_path.exists():
            stale_path.unlink()

    project_root = experiment_root()
    baseline = _baseline_payload_for_problem(
        level=args.level,
        problem_id=args.problem_id,
        problem_name=problem.name,
        eager_baseline_file=args.eager_baseline_file,
        compile_baseline_file=args.compile_baseline_file,
    )
    metadata = {
        "created_at": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "tool": _normalize_tool_name(args.tool),
        "dataset_src": args.dataset_src,
        "problem_name": problem.name,
        "problem_path": problem.path,
        "gpu_name": hardware.display_name,
        "gpu_architecture": hardware.architecture,
        "gpu_compute_capability": hardware.compute_capability,
        "num_gpus": args.num_gpus,
        "model": args.model,
        "time_budget_minutes": args.time_budget_minutes,
        "kernelbench_root": resolved_kernelbench_root,
        "kernelbench_python": args.kernelbench_python,
        "workspace": str(paths["workspace"]),
        "candidate_path": str(_workspace_candidate_path(paths["workspace"])),
        "eager_baseline_file": baseline["eager"]["source_file"],
        "compile_baseline_file": baseline["compile"]["source_file"],
    }
    write_json(paths["workspace"] / "problem.json", metadata)
    write_json(paths["workspace"] / "baseline.json", baseline)
    write_text(paths["workspace"] / "problem_reference.py", problem.code)
    write_text(_workspace_candidate_path(paths["workspace"]), candidate_template())
    write_text(paths["workspace"] / "HARDWARE.md", render_hardware_markdown(hardware))
    write_text(
        paths["workspace"] / "SPEC.md",
        _workspace_spec_markdown(
            problem=problem,
            metadata=metadata,
            baseline=baseline,
            hardware_markdown_name="HARDWARE.md",
        ),
    )
    write_text(paths["workspace"] / "AGENTS.md", _generate_workspace_agents_md(args))
    write_text(paths["workspace"] / "INITIAL_PROMPT.md", _generate_initial_prompt(args))

    _write_workspace_script(
        paths["bin"] / "run_candidate.sh",
        _generate_run_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            kernelbench_root=resolved_kernelbench_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
            num_gpus=args.num_gpus,
        ),
    )
    _write_workspace_script(
        paths["bin"] / "profile_ncu.sh",
        _generate_profile_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            kernelbench_root=resolved_kernelbench_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
            num_gpus=args.num_gpus,
        ),
    )
    _write_workspace_script(
        paths["bin"] / "problem_info.sh",
        _generate_info_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            kernelbench_root=resolved_kernelbench_root,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
        ),
    )
    _write_workspace_script(
        paths["bin"] / "goal_status.sh",
        _generate_goal_status_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
    )
    _write_workspace_script(
        paths["bin"] / "best_result.sh",
        _generate_best_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
    )
    _write_workspace_script(
        paths["bin"] / "complete_problem.sh",
        _generate_complete_wrapper(
            kernelbench_python=args.kernelbench_python,
            project_root=project_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
    )

    status_snapshot = _write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=paths["workspace"],
    )

    _emit(
        {
            "workspace": str(paths["workspace"]),
            "problem_json": str(paths["workspace"] / "problem.json"),
            "spec": str(paths["workspace"] / "SPEC.md"),
            "hardware": str(paths["workspace"] / "HARDWARE.md"),
            "prompt": str(paths["workspace"] / "INITIAL_PROMPT.md"),
            "candidate": str(_workspace_candidate_path(paths["workspace"])),
            "goal_status": str(paths["workspace"] / "goal_status.json"),
            "agent_artifact_dir": str(
                artifact_agent_dir(args.run_name, args.level, args.problem_id)
            ),
            "status_snapshot": status_snapshot,
        }
    )


def command_problem_info(args: argparse.Namespace) -> None:
    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=args.kernelbench_root,
    )
    _emit(
        {
            "level": problem.level,
            "problem_id": problem.problem_id,
            "dataset_src": problem.dataset_src,
            "name": problem.name,
            "path": problem.path,
            "code": problem.code,
        }
    )


def command_run_candidate(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).resolve()
    workspace = _workspace_path(args.workspace) if args.workspace else None
    artifact_dir = artifact_problem_dir(args.run_name, args.level, args.problem_id)
    lease_name = f"artifacts:{args.run_name}:level_{args.level}:problem_{args.problem_id}"
    sample_id: int | None = None
    kernel_path: Path | None = None
    prompt_path: Path | None = None
    payload: dict[str, Any] | None = None
    sample_json_path: Path | None = None
    history_path = artifact_dir / "history.jsonl"
    failure: Exception | None = None
    persist_failure: Exception | None = None
    status_refresh_failure: Exception | None = None

    try:
        with lease_problem_artifacts(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            lease_name=lease_name,
        ) as artifact_lease:
            sample_id = next_sample_id(args.run_name, args.level, args.problem_id)
            kernel_path = official_kernel_path(
                args.run_name,
                args.level,
                args.problem_id,
                sample_id,
            )
            sample_json_path = artifact_dir / f"sample_{sample_id}.json"
            if args.prompt_path:
                prompt_path = official_prompt_path(
                    args.run_name,
                    args.level,
                    args.problem_id,
                    sample_id,
                )

            payload = {
                "status": "started",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "run_name": args.run_name,
                "level": args.level,
                "problem_id": args.problem_id,
                "sample_id": sample_id,
                "candidate_path": str(candidate_path),
                "official_kernel_path": str(kernel_path),
                "official_prompt_path": str(prompt_path) if prompt_path else None,
                "backend": args.backend,
                "precision": args.precision,
                "artifact_reservation_wait_seconds": artifact_lease.wait_seconds,
                "artifact_commit_wait_seconds": None,
                "gpu_id": None,
                "gpu_wait_seconds": None,
                "result": {},
                "error": None,
            }

            if workspace is not None:
                expected_candidate_path = _workspace_candidate_path(workspace)
                if candidate_path != expected_candidate_path:
                    raise CandidateValidationError(
                        f"Only {CANDIDATE_FILENAME} may be evaluated from the problem workspace."
                    )

            candidate_src = candidate_path.read_text(encoding="utf-8")
            validate_candidate_source(candidate_src)
            write_text(kernel_path, candidate_src)
            if workspace is not None:
                _write_workspace_sample_copy(workspace, sample_id, candidate_src)
            if prompt_path is not None:
                write_text(prompt_path, Path(args.prompt_path).read_text(encoding="utf-8"))
            write_json(sample_json_path, payload)

        with lease_gpu_slot(
            num_slots=args.num_gpu_slots,
            requested_slot=args.gpu_id,
            lease_name=f"run:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
        ) as lease:
            payload["gpu_id"] = lease.slot_id
            payload["gpu_wait_seconds"] = lease.wait_seconds
            result = evaluate_candidate(
                candidate_src=candidate_src,
                level=args.level,
                problem_id=args.problem_id,
                dataset_src=args.dataset_src,
                run_name=args.run_name,
                sample_id=sample_id,
                gpu_id=lease.slot_id,
                timing_method=args.timing_method,
                backend=args.backend,
                precision=args.precision,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                explicit_kernelbench_root=args.kernelbench_root,
            )

        payload["status"] = "succeeded"
        payload["updated_at"] = now_iso()
        payload["result"] = result
    except Exception as exc:
        failure = exc
        if payload is None or sample_id is None or sample_json_path is None:
            raise
        payload["status"] = "failed"
        payload["updated_at"] = now_iso()
        payload["error"] = _serialize_exception(exc)
    finally:
        if payload is not None and sample_json_path is not None:
            try:
                with lease_problem_artifacts(
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    lease_name=lease_name,
                ) as artifact_lease:
                    payload["artifact_commit_wait_seconds"] = artifact_lease.wait_seconds
                    payload["updated_at"] = now_iso()
                    write_json(sample_json_path, payload)
                    append_jsonl(history_path, payload)
            except Exception as exc:
                persist_failure = exc
        if workspace is not None:
            try:
                _write_goal_status_files(
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    workspace=workspace,
                )
            except Exception as exc:
                status_refresh_failure = exc

    _emit(payload)
    if failure is not None:
        if persist_failure is not None:
            print(
                f"warning: artifact persistence also failed for sample {sample_id}: {persist_failure}",
                file=sys.stderr,
            )
        if status_refresh_failure is not None:
            print(
                f"warning: goal-status refresh also failed for sample {sample_id}: {status_refresh_failure}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"Candidate evaluation failed for sample {sample_id}: {failure}"
        ) from failure
    if persist_failure is not None:
        raise SystemExit(
            f"Artifact persistence failed for sample {sample_id}: {persist_failure}"
        ) from persist_failure
    if status_refresh_failure is not None:
        print(
            f"warning: failed to refresh goal status after sample {sample_id}: {status_refresh_failure}",
            file=sys.stderr,
        )


def command_profile_ncu(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).resolve()
    workspace: Path | None = None
    if args.workspace:
        workspace = _workspace_path(args.workspace)
        expected_candidate_path = _workspace_candidate_path(workspace)
        if candidate_path != expected_candidate_path:
            raise SystemExit(
                f"Only {CANDIDATE_FILENAME} may be profiled from the problem workspace."
            )
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)
    if args.sample_id is not None:
        sample_label = f"sample_{args.sample_id}"
    elif workspace is not None:
        sample_label = f"profile_{_next_workspace_profile_index(workspace)}"
    else:
        sample_label = "scratch"

    artifact_dir = artifact_problem_dir(args.run_name, args.level, args.problem_id) / "ncu"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = artifact_dir / (
        f"level_{args.level}_problem_{args.problem_id}_{sample_label}"
    )
    report_path = Path(str(report_prefix) + ".ncu-rep")
    stdout_path = report_prefix.with_suffix(".stdout.txt")
    stderr_path = report_prefix.with_suffix(".stderr.txt")
    details_path = report_prefix.with_suffix(".details.txt")
    details_stderr_path = report_prefix.with_suffix(".details.stderr.txt")
    raw_csv_path = report_prefix.with_suffix(".raw.csv")
    raw_csv_stderr_path = report_prefix.with_suffix(".raw.stderr.txt")
    summary_path = report_prefix.with_suffix(".summary.txt")

    with lease_gpu_slot(
        num_slots=args.num_gpu_slots,
        requested_slot=args.gpu_id,
        lease_name=f"profile:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
    ) as lease:
        command = [
            "ncu",
            "--set",
            args.ncu_set,
            "--force-overwrite",
            "--target-processes",
            "all",
            "--export",
            str(report_prefix),
            sys.executable,
            "-m",
            "kernel_bench_experiment_agents.ncu_runner",
            "--candidate",
            str(candidate_path),
            "--level",
            str(args.level),
            "--problem-id",
            str(args.problem_id),
            "--dataset-src",
            args.dataset_src,
            "--gpu-id",
            str(lease.slot_id),
            "--run-name",
            args.run_name,
            "--sample-label",
            sample_label,
        ]
        if args.kernelbench_root:
            command.extend(["--kernelbench-root", args.kernelbench_root])

        completed = _run_subprocess_capture(command)
    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)

    details_command = [
        "ncu",
        "--import",
        str(report_path),
        "--page",
        "details",
    ]
    details_completed = _run_subprocess_capture(details_command)
    write_text(details_path, details_completed.stdout)
    write_text(details_stderr_path, details_completed.stderr)

    raw_csv_command = [
        "ncu",
        "--import",
        str(report_path),
        "--page",
        "raw",
        "--csv",
    ]
    raw_csv_completed = _run_subprocess_capture(raw_csv_command)
    write_text(raw_csv_path, raw_csv_completed.stdout)
    write_text(raw_csv_stderr_path, raw_csv_completed.stderr)
    summary_text = _summarize_ncu_raw_csv(raw_csv_completed.stdout)
    write_text(summary_path, summary_text)

    payload = {
        "timestamp": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "sample_label": sample_label,
        "candidate_path": str(candidate_path),
        "ncu_report_prefix": str(report_prefix),
        "details_path": str(details_path),
        "details_stderr_path": str(details_stderr_path),
        "raw_csv_path": str(raw_csv_path),
        "raw_csv_stderr_path": str(raw_csv_stderr_path),
        "summary_path": str(summary_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": completed.returncode,
        "command": command,
        "details_command": details_command,
        "details_returncode": details_completed.returncode,
        "raw_csv_command": raw_csv_command,
        "raw_csv_returncode": raw_csv_completed.returncode,
        "gpu_id": lease.slot_id,
        "gpu_wait_seconds": lease.wait_seconds,
    }
    write_json(report_prefix.with_suffix(".json"), payload)

    emit_payload = dict(payload)
    if workspace is not None:
        profiles_dir = _workspace_profiles_dir(workspace)
        profile_base = profiles_dir / sample_label
        local_paths = {
            "details_path": profile_base.with_suffix(".details.txt"),
            "details_stderr_path": profile_base.with_suffix(".details.stderr.txt"),
            "raw_csv_path": profile_base.with_suffix(".raw.csv"),
            "raw_csv_stderr_path": profile_base.with_suffix(".raw.stderr.txt"),
            "summary_path": profile_base.with_suffix(".summary.txt"),
            "stdout_path": profile_base.with_suffix(".stdout.txt"),
            "stderr_path": profile_base.with_suffix(".stderr.txt"),
            "json_path": profile_base.with_suffix(".json"),
        }
        latest_paths = _latest_workspace_profile_paths(workspace)

        write_text(local_paths["details_path"], details_completed.stdout)
        write_text(local_paths["details_stderr_path"], details_completed.stderr)
        write_text(local_paths["raw_csv_path"], raw_csv_completed.stdout)
        write_text(local_paths["raw_csv_stderr_path"], raw_csv_completed.stderr)
        write_text(local_paths["summary_path"], summary_text)
        write_text(local_paths["stdout_path"], completed.stdout)
        write_text(local_paths["stderr_path"], completed.stderr)
        for key, latest_path in latest_paths.items():
            source_key = "json_path" if key == "json" else key
            source_path = local_paths[source_key]
            if source_key == "json_path":
                continue
            write_text(latest_path, source_path.read_text(encoding="utf-8"))

        emit_payload = {
            "timestamp": payload["timestamp"],
            "run_name": args.run_name,
            "level": args.level,
            "problem_id": args.problem_id,
            "sample_label": sample_label,
            "candidate_path": _workspace_relpath(candidate_path, workspace),
            "details_path": _workspace_relpath(latest_paths["details"], workspace),
            "raw_csv_path": _workspace_relpath(latest_paths["raw_csv"], workspace),
            "summary_path": _workspace_relpath(latest_paths["summary"], workspace),
            "stdout_path": _workspace_relpath(latest_paths["stdout"], workspace),
            "stderr_path": _workspace_relpath(latest_paths["stderr"], workspace),
            "profile_details_path": _workspace_relpath(local_paths["details_path"], workspace),
            "profile_raw_csv_path": _workspace_relpath(local_paths["raw_csv_path"], workspace),
            "profile_summary_path": _workspace_relpath(local_paths["summary_path"], workspace),
            "profile_stdout_path": _workspace_relpath(local_paths["stdout_path"], workspace),
            "profile_stderr_path": _workspace_relpath(local_paths["stderr_path"], workspace),
            "returncode": completed.returncode,
            "details_returncode": details_completed.returncode,
            "raw_csv_returncode": raw_csv_completed.returncode,
            "gpu_id": lease.slot_id,
            "gpu_wait_seconds": lease.wait_seconds,
        }
        write_json(local_paths["json_path"], emit_payload)
        write_json(latest_paths["json"], emit_payload)

    if completed.returncode != 0:
        raise SystemExit(
            "ncu profiling failed "
            f"(return code {completed.returncode}); see {stderr_path}"
        )
    if details_completed.returncode != 0:
        raise SystemExit(
            "ncu text summary export failed "
            f"(return code {details_completed.returncode}); see {details_stderr_path}"
        )
    if raw_csv_completed.returncode != 0:
        raise SystemExit(
            "ncu raw csv export failed "
            f"(return code {raw_csv_completed.returncode}); see {raw_csv_stderr_path}"
        )
    if not details_completed.stdout.strip():
        raise SystemExit(
            f"ncu details export produced no readable output; see {details_path}"
        )
    if not raw_csv_completed.stdout.strip():
        raise SystemExit(
            f"ncu raw csv export produced no readable output; see {raw_csv_path}"
        )
    _emit(emit_payload)


def command_best_result(args: argparse.Namespace) -> None:
    artifact_dir = artifact_problem_dir(args.run_name, args.level, args.problem_id)
    history_path = artifact_dir / "history.jsonl"
    if not history_path.exists():
        raise SystemExit(f"No history found at {history_path}")

    best_payload = _best_correct_payload(history_path)
    if best_payload is None:
        raise SystemExit("No correct runtime-bearing results were found in history.jsonl")
    _emit(best_payload)


def command_goal_status(args: argparse.Namespace) -> None:
    workspace = _workspace_path(args.workspace)
    snapshot = _write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=workspace,
    )
    _emit(snapshot)


def command_complete_problem(args: argparse.Namespace) -> None:
    workspace = _workspace_path(args.workspace)
    metadata = _load_workspace_metadata(workspace)
    tool = _normalize_tool_name(metadata.get("tool"))
    agent_dir = artifact_agent_dir(args.run_name, args.level, args.problem_id)
    completion_path = agent_dir / "completion.json"
    if completion_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"Completion already exists at {completion_path}. Use --allow-overwrite to replace it."
        )

    snapshot = _write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=workspace,
    )
    if args.decision == "beats_both_baselines" and not snapshot["beats_both"]:
        raise SystemExit(
            "Cannot record beats_both_baselines because goal_status does not show both baselines beaten."
        )
    if args.decision == "beats_eager_only" and (
        not snapshot["beats_eager"] or snapshot["beats_compile"]
    ):
        raise SystemExit(
            "Cannot record beats_eager_only because goal_status does not match that state."
        )
    if args.decision == "beats_compile_only" and (
        not snapshot["beats_compile"] or snapshot["beats_eager"]
    ):
        raise SystemExit(
            "Cannot record beats_compile_only because goal_status does not match that state."
        )
    if args.decision == "stalled" and _substantial_budget_remaining(snapshot):
        if int(snapshot.get("num_profile_runs") or 0) < 1:
            raise SystemExit(
                "Cannot record stalled while substantial budget remains and no profiler "
                "run has been recorded. Run ./bin/profile_ncu.sh on a strong candidate "
                "and try a new branch first."
            )
    payload = {
        "completed_at": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "tool": tool,
        "decision": args.decision,
        "success": args.decision == "beats_both_baselines",
        "summary": args.summary,
        "goal_status": snapshot,
    }
    payload = _annotate_completion_outcomes(payload)
    write_json(completion_path, payload)
    write_json(workspace / "completion.json", payload)
    _emit(payload)


def _write_final_message(
    *,
    output_path: Path,
    tool: str,
    raw_events: list[dict[str, Any]],
) -> None:
    tool = _normalize_tool_name(tool)
    final_text = None
    if tool == "claude":
        for payload in reversed(raw_events):
            if payload.get("type") != "assistant":
                continue
            message = payload.get("message")
            if not isinstance(message, dict):
                continue
            fragments = [
                str(block.get("text")).strip()
                for block in _claude_content_blocks(payload)
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            ]
            final_text = "\n\n".join(fragment for fragment in fragments if fragment)
            if final_text:
                break
    else:
        for payload in reversed(raw_events):
            fragments: list[str] = []
            _collect_text_fragments(payload, fragments)
            final_text = "\n\n".join(fragment for fragment in fragments if fragment)
            if final_text:
                break

    if final_text:
        write_text(output_path, final_text.strip() + "\n")


def command_materialize_agent_trace(args: argparse.Namespace) -> None:
    tool = _normalize_tool_name(args.tool)
    events_path = Path(args.events_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    events: list[dict[str, Any]] = []
    raw_events, raw_event_entries = _load_trace_event_entries(events_path)
    line_count = 0
    if events_path.exists():
        for line_count, line in enumerate(
            events_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                events.append(
                    {
                        "line": line_count,
                        "event_type": "non_json_output",
                        "role": None,
                        "tool_name": None,
                        "command": None,
                        "text": line[:400],
                        "sample_refs": [],
                    }
                )
                continue
            events.append(_extract_trace_line(payload, line_count, tool=tool))

    token_usage = _trace_usage_summary(raw_events, tool=tool)
    cost_usd = _trace_cost_usd(raw_events, tool=tool)
    trace_counts = _trace_counts_from_entries(raw_event_entries, tool=tool)
    web_searches = _web_searches_from_entries(raw_event_entries, tool=tool)
    audit = None
    if args.workspace:
        audit = _audit_trace(
            raw_event_entries=raw_event_entries,
            workspace=Path(args.workspace).expanduser().resolve(),
            tool=tool,
        )
    payload = {
        "tool": tool,
        "source_events_path": str(events_path),
        "generated_at": now_iso(),
        "num_events": len(events),
        "token_usage": token_usage,
        "cost_usd": cost_usd,
        "trace_counts": trace_counts,
        "web_searches": web_searches,
        "audit": audit,
        "events": events,
    }
    write_json(output_path, payload)
    if args.final_message_path:
        _write_final_message(
            output_path=Path(args.final_message_path).expanduser().resolve(),
            tool=tool,
            raw_events=raw_events,
        )
    if args.completion_path:
        completion_path = Path(args.completion_path).expanduser().resolve()
        if completion_path.exists():
            completion_payload = _read_json(completion_path)
            completion_payload["tool"] = tool
            completion_payload["token_usage"] = token_usage
            completion_payload["cost_usd"] = cost_usd
            completion_payload["trace_counts"] = trace_counts
            completion_payload["web_searches"] = web_searches
            if audit is not None:
                completion_payload = _apply_trace_audit_to_completion(
                    completion_payload,
                    audit,
                )
            completion_payload = _apply_completion_policy(completion_payload)
            completion_payload = _annotate_completion_outcomes(completion_payload)
            write_json(completion_path, completion_payload)
            if args.workspace:
                write_json(
                    Path(args.workspace).expanduser().resolve() / "completion.json",
                    completion_payload,
                )
    _emit(
        {
            "output_path": str(output_path),
            "num_events": len(events),
            "source_events_path": str(events_path),
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "audit": audit,
        }
    )


def command_summarize_run(args: argparse.Namespace) -> None:
    pass_k_values = _parse_pass_k_list(args.pass_k)
    eager_baseline = _load_baseline_file(args.eager_baseline_file)
    compile_baseline = _load_baseline_file(args.compile_baseline_file)

    run_root = experiment_root() / "artifacts" / args.run_name
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

            history_path = problem_dir / "history.jsonl"
            samples: list[dict[str, Any]] = []
            if history_path.exists():
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    result = payload.get("result", {})
                    sample = {
                        "sample_id": payload.get("sample_id"),
                        "status": payload.get("status"),
                        "compiled": bool(result.get("compiled")),
                        "correct": bool(result.get("correctness")),
                        "runtime_ms": _candidate_runtime(result),
                    }
                    samples.append(sample)

            completion_payload = None
            completion_path = problem_dir / "agent" / "completion.json"
            if completion_path.exists():
                completion_payload = _read_json(completion_path)

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

            problem = load_problem(
                level=level,
                problem_id=problem_id,
                dataset_src=args.dataset_src,
                explicit_kernelbench_root=args.kernelbench_root,
            )
            eager_mean = _baseline_mean_for_problem(
                baseline=eager_baseline,
                level=level,
                problem_name=problem.name,
            )
            compile_mean = _baseline_mean_for_problem(
                baseline=compile_baseline,
                level=level,
                problem_name=problem.name,
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
                _as_float(completion_payload.get("cost_usd"))
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
                        _as_float(row_token_usage.get(key)) or 0
                    )
            if isinstance(row_trace_counts, dict):
                trace_count_totals["problems_with_trace_counts"] += 1
                for key in (
                    "command_executions",
                    "file_change_events",
                    "wrapper_commands",
                    "gpu_wrapper_commands",
                    "problem_info_calls",
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
                        _as_float(row_trace_counts.get(key)) or 0
                    )
            if row_cost_usd is not None:
                cost_usd_totals["problems_with_cost"] += 1
                cost_usd_totals["total_usd"] += float(row_cost_usd)
            problem_rows.append(
                {
                    "level": level,
                    "problem_id": problem_id,
                    "problem_name": problem.name,
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
                    "completion_decision": (
                        completion_payload.get("decision")
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
    problems_with_compiled = sum(
        1 for row in problem_rows if row["compiled_samples"] > 0
    )
    problems_with_correct = sum(
        1 for row in problem_rows if row["effective_correct_samples"] > 0
    )

    eager_comparable = [
        row for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["eager_baseline_ms"] is not None
    ]
    compile_comparable = [
        row for row in problem_rows
        if row["best_correct_runtime_ms"] is not None and row["compile_baseline_ms"] is not None
    ]
    terminal_decisions: dict[str, int] = {}
    for row in problem_rows:
        decision = row.get("completion_decision")
        if not decision:
            continue
        terminal_decisions[decision] = terminal_decisions.get(decision, 0) + 1

    pass_at_k: dict[str, Any] = {}
    for k in pass_k_values:
        estimates = []
        eligible = 0
        for row in problem_rows:
            estimate = _pass_at_k_estimate(
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
        "terminal_decisions": terminal_decisions,
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
    _emit(payload)


def main() -> None:
    parser = _default_parser()
    args = parser.parse_args()

    handlers = {
        "prepare-problem-workspace": command_prepare_problem_workspace,
        "problem-info": command_problem_info,
        "run-candidate": command_run_candidate,
        "profile-ncu": command_profile_ncu,
        "best-result": command_best_result,
        "goal-status": command_goal_status,
        "complete-problem": command_complete_problem,
        "materialize-agent-trace": command_materialize_agent_trace,
        "materialize-codex-trace": command_materialize_agent_trace,
        "summarize-run": command_summarize_run,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
