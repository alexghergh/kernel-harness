"""Implement the solver-visible MCP tool handlers on top of existing harness commands."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from typing import Any

from kernel_bench_experiment_agents.kernelbench.commands.run_candidate import command_run_candidate
from kernel_bench_experiment_agents.kernelbench.candidate.contract import CANDIDATE_FILENAME
from kernel_bench_experiment_agents.agent_contract.policy import FIXED_WORKSPACE_RESOURCE_PATHS, MCP_TOOL_SPECS
from kernel_bench_experiment_agents.kernelbench.commands.profile import command_profile_ncu
from kernel_bench_experiment_agents.runtime.project import write_text
from kernel_bench_experiment_agents.kernelbench.metrics import blocked_run_reason
from kernel_bench_experiment_agents.kernelbench.commands.status import command_best_result, command_complete_problem, command_goal_status
from kernel_bench_experiment_agents.workspace.paths import load_workspace_metadata, workspace_candidate_path
from . import SERVER_NAME
from .context import ServerContext
from .filesystem import (
    RESOURCE_LIST_DIRS,
    allowed_directory,
    assert_allowed_edit,
    assert_allowed_read,
    resolve_workspace_path,
    safe_relative,
)
from .trace import append_mcp_event



def text_result(
    text: str,
    *,
    structured: dict[str, Any] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    payload = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if structured is not None:
        payload["structuredContent"] = structured
    return payload



def append_trace_event(
    ctx: ServerContext,
    *,
    kind: str,
    tool_name: str,
    command: str | None = None,
    path: str | None = None,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    append_mcp_event(
        ctx.events_path,
        {
            "tool": ctx.client_tool,
            "kind": kind,
            "tool_name": f"mcp__{SERVER_NAME}__{tool_name}",
            "command": command,
            "path": path,
            "text": text,
            "metadata": metadata or {},
        },
    )



def invoke_command(handler: Any, namespace: argparse.Namespace) -> dict[str, Any]:
    """Run a CLI-style handler and recover its JSON payload even on SystemExit.

    The run/profile/status handlers emit their JSON payload first and then may raise SystemExit to
    signal failure. Preserve that payload so the MCP client still sees structured failure details
    instead of only a generic exception string.
    """
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            handler(namespace)
    except SystemExit:
        output = buffer.getvalue().strip()
        if output:
            return json.loads(output)
        raise
    output = buffer.getvalue().strip()
    if not output:
        return {}
    return json.loads(output)



def handle_workspace_overview(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    metadata = load_workspace_metadata(ctx.workspace)
    append_trace_event(
        ctx,
        kind="tool_query",
        tool_name="workspace_overview",
        metadata={"problem_id": ctx.problem_id, "level": ctx.level},
    )
    overview = {
        "assignment": {
            "run_name": ctx.run_name,
            "level": ctx.level,
            "problem_id": ctx.problem_id,
            "problem_name": metadata.get("problem_name"),
            "dataset_src": ctx.dataset_src,
            "precision": metadata.get("precision") or ctx.precision,
            "time_budget_minutes": metadata.get("time_budget_minutes"),
            "gpu_name": metadata.get("gpu_name"),
            "model": metadata.get("model"),
        },
        "resources": list(FIXED_WORKSPACE_RESOURCE_PATHS),
        "history_dirs": [f"{directory}/" for directory in RESOURCE_LIST_DIRS],
        "mcp_tools": [spec.name for spec in MCP_TOOL_SPECS if spec.name != "workspace_overview"],
        "helper_agents": ["runner", "profiler"],
    }
    text = (
        f"Problem {ctx.level}/{ctx.problem_id} ({overview['assignment']['problem_name'] or 'unknown'}). "
        "Act as the planner-manager for this problem. Read the fixed workspace resources first: AGENTS.md, INITIAL_PROMPT.md, SPEC.md, HARDWARE.md, and GOAL_STATUS.md. "
        "For past attempts or profiler outputs, use `list_workspace_dir` only on `samples` or `profiles`, then `read_workspace_file` on those listed files. "
        "Use only the kernelbench MCP tools for candidate edits, measured runs, profiling, status refreshes, best-result lookup, and completion. "
        "WHEN you want a measured evaluation, spawn `runner` if available; WHEN you want Nsight Compute work, spawn `profiler` if available. Use direct MCP run/profile calls yourself only when helper spawning is unavailable."
    )
    return text_result(text, structured=overview)



def handle_list_workspace_dir(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "samples")
    path = resolve_workspace_path(ctx, raw_path)
    if not allowed_directory(path, ctx.workspace):
        allowed = ", ".join(repr(entry) for entry in RESOURCE_LIST_DIRS)
        raise RuntimeError(f"directory listing is limited to {allowed}")
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"directory does not exist: {raw_path}")
    entries = []
    for child in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        entries.append(
            {
                "name": child.name,
                "path": safe_relative(child, ctx.workspace),
                "is_dir": child.is_dir(),
            }
        )
    relative_path = safe_relative(path, ctx.workspace)
    append_trace_event(
        ctx,
        kind="file_read",
        tool_name="list_workspace_dir",
        path=relative_path,
        metadata={"listed": len(entries)},
    )
    return text_result(
        json.dumps(entries, indent=2, sort_keys=True),
        structured={"path": relative_path, "entries": entries},
    )



def handle_read_workspace_file(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise RuntimeError("path is required")
    path = resolve_workspace_path(ctx, raw_path)
    assert_allowed_read(ctx, path)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"file does not exist: {raw_path}")
    text = path.read_text(encoding="utf-8")
    relative_path = safe_relative(path, ctx.workspace)
    append_trace_event(
        ctx,
        kind="file_read",
        tool_name="read_workspace_file",
        path=relative_path,
        metadata={"bytes": len(text.encode("utf-8"))},
    )
    return text_result(text, structured={"path": relative_path, "text": text})



def handle_write_candidate(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, str):
        raise RuntimeError("content must be a string")
    candidate_path = workspace_candidate_path(ctx.workspace)
    assert_allowed_edit(ctx, candidate_path)
    write_text(candidate_path, content)
    append_trace_event(
        ctx,
        kind="file_change",
        tool_name="write_candidate",
        path=safe_relative(candidate_path, ctx.workspace),
        metadata={"bytes": len(content.encode("utf-8"))},
    )
    byte_count = len(content.encode("utf-8"))
    return text_result(
        f"Wrote {CANDIDATE_FILENAME} ({byte_count} bytes).",
        structured={"path": CANDIDATE_FILENAME, "bytes": byte_count},
    )



def handle_run_candidate(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = invoke_command(
        command_run_candidate,
        argparse.Namespace(
            candidate=str(workspace_candidate_path(ctx.workspace)),
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
            dataset_src=ctx.dataset_src,
            kernelbench_root=ctx.kernelbench_root,
            gpu_id=None,
            num_gpu_slots=ctx.num_gpu_slots,
            timing_method=None,
            backend="cuda",
            precision=ctx.precision,
            num_correct_trials=5,
            num_perf_trials=100,
            prompt_path=None,
            workspace=str(ctx.workspace),
        ),
    )
    append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="run_candidate",
        command="./bin/run_candidate.sh",
        metadata={"status": payload.get("status"), "sample_id": payload.get("sample_id")},
    )
    notice = blocked_run_reason(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if notice:
        text = f"This run is not counted toward progress because {notice}.\n\n{text}"
    return text_result(text, structured=payload)



def handle_profile_ncu(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = invoke_command(
        command_profile_ncu,
        argparse.Namespace(
            candidate=str(workspace_candidate_path(ctx.workspace)),
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
            dataset_src=ctx.dataset_src,
            kernelbench_root=ctx.kernelbench_root,
            gpu_id=None,
            num_gpu_slots=ctx.num_gpu_slots,
            sample_id=None,
            ncu_set="full",
            precision=ctx.precision,
            workspace=str(ctx.workspace),
        ),
    )
    append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="profile_ncu",
        command="./bin/profile_ncu.sh",
        metadata={"status": payload.get("status"), "profile_id": payload.get("profile_id")},
    )
    return text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)



def handle_goal_status(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = invoke_command(
        command_goal_status,
        argparse.Namespace(
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
            workspace=str(ctx.workspace),
        ),
    )
    goal_status_text = (ctx.workspace / "GOAL_STATUS.md").read_text(encoding="utf-8")
    append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="goal_status",
        command="./bin/goal_status.sh",
        metadata={"status_mode": payload.get("status_mode")},
    )
    return text_result(goal_status_text, structured={"snapshot": payload, "markdown": goal_status_text})



def handle_best_result(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = invoke_command(
        command_best_result,
        argparse.Namespace(
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
        ),
    )
    append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="best_result",
        command="./bin/best_result.sh",
        metadata={"sample_id": payload.get("sample_id")},
    )
    return text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)



def handle_complete_problem(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    summary = arguments.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("summary is required")
    payload = invoke_command(
        command_complete_problem,
        argparse.Namespace(
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
            workspace=str(ctx.workspace),
            summary=summary,
            allow_overwrite=False,
        ),
    )
    append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="complete_problem",
        command="./bin/complete_problem.sh",
        metadata={"summary": summary},
        text=summary,
    )
    return text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)


TOOL_HANDLERS: dict[str, Any] = {
    "workspace_overview": handle_workspace_overview,
    "list_workspace_dir": handle_list_workspace_dir,
    "read_workspace_file": handle_read_workspace_file,
    "write_candidate": handle_write_candidate,
    "run_candidate": handle_run_candidate,
    "profile_ncu": handle_profile_ncu,
    "goal_status": handle_goal_status,
    "best_result": handle_best_result,
    "complete_problem": handle_complete_problem,
}
