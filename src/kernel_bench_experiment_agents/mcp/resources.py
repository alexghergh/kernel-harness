"""Expose the small read-only workspace resource surface for MCP clients."""

from __future__ import annotations

from typing import Any

from ..candidate_contract import CANDIDATE_FILENAME
from . import SERVER_NAME
from .context import ServerContext
from .filesystem import assert_allowed_read, resolve_workspace_path, safe_relative
from .trace import append_mcp_event


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



def workspace_resource_uri(relative_path: str) -> str:
    return f"{RESOURCE_URI_PREFIX}{relative_path}"



def workspace_resource_descriptors(ctx: ServerContext) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for relative_path in RESOURCE_PATHS:
        path = ctx.workspace / relative_path
        if not path.exists() or not path.is_file():
            continue
        resources.append(
            {
                "uri": workspace_resource_uri(relative_path),
                "name": relative_path,
                "description": f"Workspace file: {relative_path}",
                "mimeType": "text/markdown" if relative_path.endswith(".md") else "text/plain",
            }
        )
    return resources



def read_workspace_resource(ctx: ServerContext, uri: str) -> dict[str, Any]:
    if not uri.startswith(RESOURCE_URI_PREFIX):
        raise RuntimeError(f"unknown resource uri: {uri}")
    relative_path = uri[len(RESOURCE_URI_PREFIX):]
    path = resolve_workspace_path(ctx, relative_path)
    assert_allowed_read(ctx, path)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"resource does not exist: {relative_path}")
    text = path.read_text(encoding="utf-8")
    append_mcp_event(
        ctx.events_path,
        {
            "tool": ctx.client_tool,
            "kind": "file_read",
            "tool_name": f"mcp__{SERVER_NAME}__read_workspace_resource",
            "path": safe_relative(path, ctx.workspace),
            "metadata": {"bytes": len(text.encode("utf-8"))},
        },
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



def list_workspace_resource_templates() -> dict[str, Any]:
    return {"resourceTemplates": []}
