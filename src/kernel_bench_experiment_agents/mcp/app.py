"""Build the KernelBench MCP server on top of the official Python MCP SDK.

This module keeps the MCP layer thin: FastMCP owns protocol lifecycle, stdio framing, and
capability negotiation, while the harness-specific code here only wires existing workspace and
command helpers into SDK-managed resources and tools.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP

from kernel_bench_experiment_agents.agent_contract.policy import MCP_TOOL_SPECS, McpToolSpec
from . import SERVER_NAME
from .context import ServerContext, load_context
from .filesystem import assert_allowed_read, resolve_workspace_path, safe_relative
from .handlers import TOOL_HANDLERS, append_trace_event
from .resources import RESOURCE_PATHS, workspace_resource_name, workspace_resource_uri


mcp = FastMCP(SERVER_NAME)


@lru_cache(maxsize=1)
def server_context() -> ServerContext:
    return load_context()


def tool_spec(name: str) -> McpToolSpec:
    for spec in MCP_TOOL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def mcp_tool(name: str):
    """Register a solver-facing tool with MCP safety annotations when supported."""
    spec = tool_spec(name)
    kwargs = {
        "name": name,
        "description": spec.purpose,
        "annotations": spec.annotations,
    }
    try:
        return mcp.tool(**kwargs)
    except TypeError as exc:
        if "annotations" not in str(exc):
            raise
        kwargs.pop("annotations")
        return mcp.tool(**kwargs)


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
    text = resolved.read_text(encoding="utf-8")
    append_trace_event(
        ctx,
        kind="file_read",
        tool_name="read_workspace_resource",
        path=safe_relative(resolved, ctx.workspace),
        metadata={"bytes": len(text.encode("utf-8")), "source": "resource"},
    )
    return text


@mcp_tool("workspace_overview")
def workspace_overview() -> types.CallToolResult:
    return invoke_tool("workspace_overview")


@mcp_tool("list_workspace_dir")
def list_workspace_dir(path: str = "samples") -> types.CallToolResult:
    return invoke_tool("list_workspace_dir", {"path": path})


@mcp_tool("read_workspace_file")
def read_workspace_file(path: str) -> types.CallToolResult:
    return invoke_tool("read_workspace_file", {"path": path})


@mcp_tool("write_candidate")
def write_candidate(content: str) -> types.CallToolResult:
    return invoke_tool("write_candidate", {"content": content})


@mcp_tool("run_candidate")
def run_candidate() -> types.CallToolResult:
    return invoke_tool("run_candidate")


@mcp_tool("profile_ncu")
def profile_ncu() -> types.CallToolResult:
    return invoke_tool("profile_ncu")


@mcp_tool("goal_status")
def goal_status() -> types.CallToolResult:
    return invoke_tool("goal_status")


@mcp_tool("best_result")
def best_result() -> types.CallToolResult:
    return invoke_tool("best_result")


@mcp_tool("complete_problem")
def complete_problem(summary: str) -> types.CallToolResult:
    return invoke_tool("complete_problem", {"summary": summary})


def register_fixed_workspace_resource(relative_path: str):
    """Register one static read-only workspace resource without exposing a path template.

    Resources are for the tiny canonical read surface only. History browsing under `samples/` and
    `profiles/` stays on explicit MCP tools so the agent can read useful artifacts without learning
    that there is a generic local-file resource API.

    FastMCP validates that function parameters match URI template parameters. These fixed resources
    have literal URIs, so the registered reader must take no arguments at all.
    """
    uri = workspace_resource_uri(relative_path)
    name = workspace_resource_name(relative_path)

    def _make_reader(path: str):
        def _read() -> str:
            return read_workspace_resource(path)

        return _read

    reader = _make_reader(relative_path)
    reader.__name__ = f"resource_{name}"
    reader.__qualname__ = reader.__name__
    return mcp.resource(
        uri,
        name=name,
        description=f"Read the workspace file `{relative_path}`.",
        mime_type="text/plain",
    )(reader)


_REGISTERED_FIXED_RESOURCES = tuple(
    register_fixed_workspace_resource(relative_path) for relative_path in RESOURCE_PATHS
)


def run() -> None:
    mcp.run(transport="stdio")
