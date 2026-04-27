"""Expose the launcher-owned command broker as a tiny MCP server.

The solver gets direct workspace file access from native file tools, while
privileged harness actions stay behind the launcher-owned Unix-socket broker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP

from kernel_bench_experiment_agents.command_client import send_request


SERVER_NAME = "kernelbench_commands"

mcp = FastMCP(SERVER_NAME)


def _required_socket_path() -> Path:
    value = os.environ.get("KBH_COMMAND_SOCKET", "").strip()
    if not value:
        raise RuntimeError("KBH_COMMAND_SOCKET is required for command MCP access")
    return Path(value).expanduser().resolve()


def _tool_result(payload: dict[str, Any]) -> types.CallToolResult:
    text = str(payload.get("stdout") or "")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=payload.get("payload")
        if isinstance(payload.get("payload"), dict)
        else None,
        isError=bool(payload.get("is_error")),
    )


def _invoke(command: str, **arguments: Any) -> types.CallToolResult:
    request: dict[str, Any] = {"command": command}
    request.update(arguments)
    response = send_request(socket_path=_required_socket_path(), payload=request)
    if response.get("ok") is True:
        return _tool_result(response)
    error = str(response.get("error") or "command broker request failed")
    stdout = str(response.get("stdout") or "")
    payload = response.get("payload")
    detail = error if not stdout else f"{error}\n\n{stdout}"
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=detail)],
        structuredContent=payload if isinstance(payload, dict) else None,
        isError=True,
    )


def _annotations(*, read_only: bool = False) -> types.ToolAnnotations:
    return types.ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=False,
        openWorldHint=False,
        idempotentHint=read_only,
    )


@mcp.resource(
    "kb://commands",
    name="workspace_commands",
    description="The privileged command tools available through the launcher-owned broker.",
    mime_type="application/json",
)
def workspace_commands() -> str:
    return json.dumps(
        {
            "server": SERVER_NAME,
            "tools": [
                "run_candidate",
                "profile_ncu",
                "goal_status",
                "best_result",
                "complete_problem",
            ],
        },
        indent=2,
        sort_keys=True,
    )


@mcp.tool(
    name="run_candidate",
    description="Evaluate correctness and runtime for the current candidate through the launcher-owned broker.",
    annotations=_annotations(),
)
def run_candidate() -> types.CallToolResult:
    return _invoke("run_candidate")


@mcp.tool(
    name="profile_ncu",
    description="Profile the current candidate with Nsight Compute through the launcher-owned broker.",
    annotations=_annotations(),
)
def profile_ncu() -> types.CallToolResult:
    return _invoke("profile_ncu")


@mcp.tool(
    name="goal_status",
    description="Refresh and return the live goal-status snapshot through the launcher-owned broker.",
    annotations=_annotations(read_only=True),
)
def goal_status() -> types.CallToolResult:
    return _invoke("goal_status")


@mcp.tool(
    name="best_result",
    description="Return the best measured correct result so far through the launcher-owned broker.",
    annotations=_annotations(read_only=True),
)
def best_result() -> types.CallToolResult:
    return _invoke("best_result")


@mcp.tool(
    name="complete_problem",
    description="Record a terminal completion summary through the launcher-owned broker.",
    annotations=_annotations(),
)
def complete_problem(summary: str) -> types.CallToolResult:
    return _invoke("complete_problem", summary=summary)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
