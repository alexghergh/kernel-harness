"""Expose the solver-visible harness surface as a small stdio MCP server.

The launcher starts one server per problem session and injects the assigned workspace/run metadata
through environment variables. The model never needs direct local filesystem access; instead it
reads workspace files, writes the candidate, and invokes measured harness commands through these
MCP tools.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME
from .candidate_commands import command_run_candidate
from .common import normalize_tool_name
from .mcp_trace import append_mcp_event
from .policy_model import workspace_edit_surface, workspace_read_surface
from .profile_commands import command_profile_ncu
from .project import archive_contract_dir, write_text
from .status_commands import (
    command_best_result,
    command_complete_problem,
    command_goal_status,
)
from .workspace_paths import (
    load_workspace_metadata,
    validate_workspace_assignment,
    workspace_candidate_path,
)

SERVER_NAME = "kernelbench"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL_VERSION = "2025-03-26"


@dataclass(frozen=True)
class ServerContext:
    workspace: Path
    run_name: str
    level: int
    problem_id: int
    dataset_src: str
    kernelbench_root: str | None
    num_gpu_slots: int
    precision: str
    client_tool: str
    events_path: Path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None = None


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="workspace_overview",
        description="Return the assigned problem metadata, key readable files, and the allowed harness MCP tools.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={"readOnlyHint": True},
    ),
    ToolSpec(
        name="list_workspace_dir",
        description="List a safe workspace directory. Use '.' for the workspace root, or a subdirectory such as 'samples' or 'profiles'.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
            },
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
    ),
    ToolSpec(
        name="read_workspace_file",
        description="Read one allowed workspace file as UTF-8 text.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
    ),
    ToolSpec(
        name="write_candidate",
        description=f"Overwrite {CANDIDATE_FILENAME} with new source text.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": False,
        },
    ),
    ToolSpec(
        name="run_candidate",
        description="Run the current candidate through the measured evaluation harness.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    ),
    ToolSpec(
        name="profile_ncu",
        description="Profile the current candidate with Nsight Compute and archive the result.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    ),
    ToolSpec(
        name="goal_status",
        description="Refresh and return the live goal-status snapshot plus GOAL_STATUS.md text.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={"readOnlyHint": True},
    ),
    ToolSpec(
        name="best_result",
        description="Return the current best measured correct result, if any.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={"readOnlyHint": True},
    ),
    ToolSpec(
        name="complete_problem",
        description="Record solver completion with a required summary.",
        input_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": False,
        },
    ),
)


RESOURCE_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "SPEC.md",
    "HARDWARE.md",
    "GOAL_STATUS.md",
    "goal_status.json",
    "problem.json",
    "workspace_contract.json",
    "problem_reference.py",
    CANDIDATE_FILENAME,
)
RESOURCE_URI_PREFIX = "kb://workspace/"




def _env(name: str, *, required: bool = True, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(f"Missing required environment variable for MCP server: {name}")
    return value


def _load_archive_provenance(*, run_name: str, level: int, problem_id: int) -> dict[str, Any]:
    provenance_path = archive_contract_dir(run_name, level, problem_id) / "provenance.json"
    if not provenance_path.exists():
        return {}
    return json.loads(provenance_path.read_text(encoding="utf-8"))



def load_context() -> ServerContext:
    workspace = Path(_env("KBH_WORKSPACE") or "").expanduser().resolve()
    metadata = load_workspace_metadata(workspace)
    run_name = str(metadata.get("run_name") or "")
    level = int(metadata.get("level") or "0")
    problem_id = int(metadata.get("problem_id") or "0")
    validate_workspace_assignment(
        workspace,
        run_name=run_name,
        level=level,
        problem_id=problem_id,
    )
    provenance = _load_archive_provenance(run_name=run_name, level=level, problem_id=problem_id)
    return ServerContext(
        workspace=workspace,
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        dataset_src=str(metadata.get("dataset_src") or "local"),
        kernelbench_root=(
            str(provenance.get("kernelbench_root"))
            if provenance.get("kernelbench_root")
            else None
        ),
        num_gpu_slots=int(metadata.get("num_gpus") or 1),
        precision=str(metadata.get("precision") or "bf16"),
        client_tool=normalize_tool_name(_env("KBH_CLIENT_TOOL", required=False, default="codex")),
        events_path=Path(_env("KBH_MCP_EVENTS_PATH") or "").expanduser().resolve(),
    )


def _tool_descriptor(spec: ToolSpec) -> dict[str, Any]:
    payload = {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": spec.input_schema,
    }
    if spec.annotations is not None:
        payload["annotations"] = spec.annotations
    return payload


def _resolve_workspace_path(ctx: ServerContext, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = ctx.workspace / candidate
    return candidate.resolve()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return "." if str(rel) == "" else str(rel)
    except ValueError:
        return str(path)


def _allowed_directory(path: Path, workspace: Path) -> bool:
    allowed_dirs = {
        workspace.resolve(),
        (workspace / "samples").resolve(),
        (workspace / "profiles").resolve(),
    }
    return path.resolve() in allowed_dirs


def _assert_allowed_read(ctx: ServerContext, path: Path) -> None:
    allowed_read_paths, allowed_read_roots = workspace_read_surface(ctx.workspace)
    resolved = path.resolve()
    if resolved in allowed_read_paths:
        return
    if any(
        resolved == root or root in resolved.parents
        for root in allowed_read_roots
    ):
        return
    raise RuntimeError(f"read path is outside the allowed workspace surface: {path}")


def _assert_allowed_edit(ctx: ServerContext, path: Path) -> None:
    allowed_edit_paths = workspace_edit_surface(ctx.workspace)
    resolved = path.resolve()
    if resolved not in allowed_edit_paths:
        raise RuntimeError(f"edit path is outside the allowed workspace edit surface: {path}")


def _append_trace_event(ctx: ServerContext, *, kind: str, tool_name: str, command: str | None = None, path: str | None = None, text: str | None = None, metadata: dict[str, Any] | None = None) -> None:
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


def _invoke_command(handler: Any, namespace: argparse.Namespace) -> dict[str, Any]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        handler(namespace)
    output = buffer.getvalue().strip()
    if not output:
        return {}
    return json.loads(output)


def _text_result(text: str, *, structured: dict[str, Any] | None = None, is_error: bool = False) -> dict[str, Any]:
    payload = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if structured is not None:
        payload["structuredContent"] = structured
    return payload


def handle_workspace_overview(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    metadata = load_workspace_metadata(ctx.workspace)
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
        "key_files": [
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
            "goal_status.json",
            "problem.json",
            "workspace_contract.json",
            "problem_reference.py",
            CANDIDATE_FILENAME,
            "samples/",
            "profiles/",
        ],
        "mcp_tools": [spec.name for spec in TOOL_SPECS if spec.name != "workspace_overview"],
    }
    text = (
        f"Problem {ctx.level}/{ctx.problem_id} ({overview['assignment']['problem_name'] or 'unknown'}). "
        f"Read AGENTS.md, SPEC.md, HARDWARE.md, and GOAL_STATUS.md first. "
        "Use only the kernelbench MCP tools for local reads, candidate edits, measured runs, profiling, status refreshes, and completion."
    )
    return _text_result(text, structured=overview)


def handle_list_workspace_dir(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or ".")
    path = _resolve_workspace_path(ctx, raw_path)
    if not _allowed_directory(path, ctx.workspace):
        raise RuntimeError("directory listing is limited to '.', 'samples', and 'profiles'")
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"directory does not exist: {raw_path}")
    entries = []
    for child in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        entries.append(
            {
                "name": child.name,
                "path": _safe_relative(child, ctx.workspace),
                "is_dir": child.is_dir(),
            }
        )
    relative_path = _safe_relative(path, ctx.workspace)
    _append_trace_event(
        ctx,
        kind="file_read",
        tool_name="list_workspace_dir",
        path=relative_path,
        metadata={"listed": len(entries)},
    )
    return _text_result(
        json.dumps(entries, indent=2, sort_keys=True),
        structured={"path": relative_path, "entries": entries},
    )


def handle_read_workspace_file(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise RuntimeError("path is required")
    path = _resolve_workspace_path(ctx, raw_path)
    _assert_allowed_read(ctx, path)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"file does not exist: {raw_path}")
    text = path.read_text(encoding="utf-8")
    relative_path = _safe_relative(path, ctx.workspace)
    _append_trace_event(
        ctx,
        kind="file_read",
        tool_name="read_workspace_file",
        path=relative_path,
        metadata={"bytes": len(text.encode('utf-8'))},
    )
    return _text_result(text, structured={"path": relative_path, "text": text})


def handle_write_candidate(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, str):
        raise RuntimeError("content must be a string")
    candidate_path = workspace_candidate_path(ctx.workspace)
    _assert_allowed_edit(ctx, candidate_path)
    write_text(candidate_path, content)
    _append_trace_event(
        ctx,
        kind="file_change",
        tool_name="write_candidate",
        path=_safe_relative(candidate_path, ctx.workspace),
        metadata={"bytes": len(content.encode('utf-8'))},
    )
    return _text_result(
        f"Wrote {CANDIDATE_FILENAME} ({len(content.encode('utf-8'))} bytes).",
        structured={"path": CANDIDATE_FILENAME, "bytes": len(content.encode('utf-8'))},
    )


def handle_run_candidate(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = _invoke_command(
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
    _append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="run_candidate",
        command="./bin/run_candidate.sh",
        metadata={"status": payload.get("status"), "sample_id": payload.get("sample_id")},
    )
    return _text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)


def handle_profile_ncu(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = _invoke_command(
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
    _append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="profile_ncu",
        command="./bin/profile_ncu.sh",
        metadata={"status": payload.get("status"), "profile_id": payload.get("profile_id")},
    )
    return _text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)


def handle_goal_status(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = _invoke_command(
        command_goal_status,
        argparse.Namespace(
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
            workspace=str(ctx.workspace),
        ),
    )
    goal_status_text = (ctx.workspace / "GOAL_STATUS.md").read_text(encoding="utf-8")
    _append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="goal_status",
        command="./bin/goal_status.sh",
        metadata={"status_mode": payload.get("status_mode")},
    )
    return _text_result(goal_status_text, structured={"snapshot": payload, "markdown": goal_status_text})


def handle_best_result(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    payload = _invoke_command(
        command_best_result,
        argparse.Namespace(
            run_name=ctx.run_name,
            level=ctx.level,
            problem_id=ctx.problem_id,
        ),
    )
    _append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="best_result",
        command="./bin/best_result.sh",
        metadata={"sample_id": payload.get("sample_id")},
    )
    return _text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)


def handle_complete_problem(ctx: ServerContext, arguments: dict[str, Any]) -> dict[str, Any]:
    summary = arguments.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("summary is required")
    payload = _invoke_command(
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
    _append_trace_event(
        ctx,
        kind="command_execution",
        tool_name="complete_problem",
        command="./bin/complete_problem.sh",
        metadata={"summary": summary},
        text=summary,
    )
    return _text_result(json.dumps(payload, indent=2, sort_keys=True), structured=payload)


TOOL_HANDLERS = {
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


def _workspace_resource_uri(relative_path: str) -> str:
    return f"{RESOURCE_URI_PREFIX}{relative_path}"



def _workspace_resource_descriptors(ctx: ServerContext) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for relative_path in RESOURCE_PATHS:
        path = ctx.workspace / relative_path
        if not path.exists() or not path.is_file():
            continue
        resources.append(
            {
                "uri": _workspace_resource_uri(relative_path),
                "name": relative_path,
                "description": f"Workspace file: {relative_path}",
                "mimeType": "text/markdown" if relative_path.endswith('.md') else "text/plain",
            }
        )
    return resources



def _read_workspace_resource(ctx: ServerContext, uri: str) -> dict[str, Any]:
    if not uri.startswith(RESOURCE_URI_PREFIX):
        raise RuntimeError(f"unknown resource uri: {uri}")
    relative_path = uri[len(RESOURCE_URI_PREFIX):]
    path = _resolve_workspace_path(ctx, relative_path)
    _assert_allowed_read(ctx, path)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"resource does not exist: {relative_path}")
    text = path.read_text(encoding="utf-8")
    _append_trace_event(
        ctx,
        kind="file_read",
        tool_name="read_workspace_resource",
        path=_safe_relative(path, ctx.workspace),
        metadata={"bytes": len(text.encode('utf-8'))},
    )
    mime_type = "text/markdown" if path.suffix == ".md" else "text/plain"
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": mime_type,
                "text": text,
            }
        ]
    }



def _list_workspace_resource_templates() -> dict[str, Any]:
    return {"resourceTemplates": []}


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        decoded = line.decode("utf-8").strip()
        if not decoded:
            continue
        name, _, value = decoded.partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _write_response(request_id: Any, result: dict[str, Any]) -> None:
    _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def _write_error(request_id: Any, code: int, message: str) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def main() -> None:
    ctx = load_context()
    while True:
        request = _read_message()
        if request is None:
            return
        method = request.get("method")
        request_id = request.get("id")

        if method == "notifications/initialized":
            continue
        if method == "initialize":
            params = request.get("params") or {}
            protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
            _write_response(
                request_id,
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            continue
        if method == "ping":
            _write_response(request_id, {})
            continue
        if method == "resources/list":
            _write_response(request_id, {"resources": _workspace_resource_descriptors(ctx)})
            continue
        if method == "resources/templates/list":
            _write_response(request_id, _list_workspace_resource_templates())
            continue
        if method == "resources/read":
            params = request.get("params") or {}
            try:
                result = _read_workspace_resource(ctx, str(params.get("uri") or ""))
            except Exception as exc:  # pragma: no cover - best effort server error surface
                _write_error(request_id, -32602, f"resource read failed: {exc}")
            else:
                _write_response(request_id, result)
            continue
        if method == "tools/list":
            _write_response(request_id, {"tools": [_tool_descriptor(spec) for spec in TOOL_SPECS]})
            continue
        if method == "tools/call":
            params = request.get("params") or {}
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            handler = TOOL_HANDLERS.get(str(tool_name))
            if handler is None:
                _write_error(request_id, -32602, f"unknown tool: {tool_name}")
                continue
            try:
                result = handler(ctx, arguments)
            except SystemExit as exc:
                _write_response(
                    request_id,
                    _text_result(str(exc), is_error=True),
                )
            except Exception as exc:  # pragma: no cover - best effort server error surface
                _write_response(
                    request_id,
                    _text_result(f"{type(exc).__name__}: {exc}", is_error=True),
                )
            else:
                _write_response(request_id, result)
            continue
        if request_id is None:
            continue
        _write_error(request_id, -32601, f"unsupported method: {method}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - startup failures should be visible in stderr logs
        print(
            f"kernelbench MCP server failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
