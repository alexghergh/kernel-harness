"""Build the KernelBench MCP server on top of the official Python MCP SDK.

This module keeps the MCP layer thin: FastMCP owns protocol lifecycle, stdio framing, and
capability negotiation, while the harness-specific code here only wires existing workspace and
command helpers into SDK-managed resources and tools.
"""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP

from ..policy_model import MCP_TOOL_SPECS, McpToolSpec
from . import SERVER_NAME
from .context import ServerContext, load_context
from .filesystem import RESOURCE_LIST_DIRS, assert_allowed_read, resolve_workspace_path
from .handlers import TOOL_HANDLERS
from .resources import RESOURCE_PATHS, workspace_resource_uri


mcp = FastMCP(SERVER_NAME)


@lru_cache(maxsize=1)
def server_context() -> ServerContext:
    return load_context()


def tool_spec(name: str) -> McpToolSpec:
    for spec in MCP_TOOL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def tool_result(payload: dict[str, Any]) -> types.CallToolResult:
    content: list[
        types.TextContent
        | types.ImageContent
        | types.AudioContent
        | types.ResourceLink
        | types.EmbeddedResource
    ] = []
    for item in payload.get("content", []):
        if item.get("type") == "text":
            content.append(types.TextContent(type="text", text=str(item.get("text") or "")))
    if not content:
        content = [types.TextContent(type="text", text="")]
    structured = payload.get("structuredContent")
    if structured is not None and not isinstance(structured, dict):
        structured = {"result": structured}
    return types.CallToolResult(
        content=content,
        structuredContent=structured,
        isError=bool(payload.get("isError")),
    )


def invoke_tool(name: str, arguments: dict[str, Any] | None = None) -> types.CallToolResult:
    ctx = server_context()
    handler = TOOL_HANDLERS[name]
    try:
        payload = handler(ctx, arguments or {})
    except SystemExit as exc:
        return tool_result(
            {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }
        )
    except Exception as exc:
        return tool_result(
            {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        )
    return tool_result(payload)


def read_workspace_resource(path: str) -> str:
    ctx = server_context()
    resolved = resolve_workspace_path(ctx, path)
    assert_allowed_read(ctx, resolved)
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"resource does not exist: {path}")
    payload = TOOL_HANDLERS["read_workspace_file"](ctx, {"path": path})
    content = payload.get("content") or []
    if not content:
        return ""
    return str(content[0].get("text") or "")


def listed_workspace_resources() -> list[str]:
    """Return the canonical solver-visible workspace files without touching live workspace state.

    Codex probes MCP servers with initialize/resources discovery before the agent can do any useful
    work. Keep that startup path lazy: resource templates and tool schemas should not require the
    per-problem workspace env or filesystem until an actual resource/tool call arrives.
    """
    return list(RESOURCE_PATHS)


@mcp.tool(
    name="workspace_overview",
    description=tool_spec("workspace_overview").purpose,
)
def workspace_overview() -> types.CallToolResult:
    return invoke_tool("workspace_overview")


@mcp.tool(
    name="list_workspace_dir",
    description=tool_spec("list_workspace_dir").purpose,
)
def list_workspace_dir(path: str = ".") -> types.CallToolResult:
    return invoke_tool("list_workspace_dir", {"path": path})


@mcp.tool(
    name="read_workspace_file",
    description=tool_spec("read_workspace_file").purpose,
)
def read_workspace_file(path: str) -> types.CallToolResult:
    return invoke_tool("read_workspace_file", {"path": path})


@mcp.tool(
    name="write_candidate",
    description=tool_spec("write_candidate").purpose,
)
def write_candidate(content: str) -> types.CallToolResult:
    return invoke_tool("write_candidate", {"content": content})


@mcp.tool(
    name="run_candidate",
    description=tool_spec("run_candidate").purpose,
)
def run_candidate() -> types.CallToolResult:
    return invoke_tool("run_candidate")


@mcp.tool(
    name="profile_ncu",
    description=tool_spec("profile_ncu").purpose,
)
def profile_ncu() -> types.CallToolResult:
    return invoke_tool("profile_ncu")


@mcp.tool(
    name="goal_status",
    description=tool_spec("goal_status").purpose,
)
def goal_status() -> types.CallToolResult:
    return invoke_tool("goal_status")


@mcp.tool(
    name="best_result",
    description=tool_spec("best_result").purpose,
)
def best_result() -> types.CallToolResult:
    return invoke_tool("best_result")


@mcp.tool(
    name="complete_problem",
    description=tool_spec("complete_problem").purpose,
)
def complete_problem(summary: str) -> types.CallToolResult:
    return invoke_tool("complete_problem", {"summary": summary})


@mcp.resource(
    "kb://workspace/{path}",
    name="workspace_file",
    description="Read one allowed workspace file by relative path.",
    mime_type="text/plain",
)
def workspace_file(path: str) -> str:
    return read_workspace_resource(path)


@mcp.resource(
    "kb://workspace/files",
    name="workspace_files",
    description="Canonical solver-visible workspace files.",
    mime_type="application/json",
)
def workspace_files() -> str:
    return json.dumps(listed_workspace_resources(), indent=2)


# Expose the safe directory-listing roots in one small note resource so clients can discover the
# intended workspace surface without shelling out or guessing path policy.
@mcp.resource(
    "kb://workspace/directories",
    name="workspace_directories",
    description="Safe workspace directories that list_workspace_dir can inspect.",
    mime_type="application/json",
)
def workspace_directories() -> str:
    return json.dumps(list(RESOURCE_LIST_DIRS), indent=2)


def run() -> None:
    mcp.run(transport="stdio")
