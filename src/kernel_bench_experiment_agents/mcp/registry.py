"""Describe the solver-visible MCP tools and their request schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..policy_model import MCP_TOOL_SPECS
from .context import ServerContext

ToolHandler = Callable[[ServerContext, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    annotations: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations is not None:
            payload["annotations"] = self.annotations
        return payload


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "workspace_overview": {"type": "object", "properties": {}, "additionalProperties": False},
    "list_workspace_dir": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
        },
        "additionalProperties": False,
    },
    "read_workspace_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "write_candidate": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
        },
        "required": ["content"],
        "additionalProperties": False,
    },
    "run_candidate": {"type": "object", "properties": {}, "additionalProperties": False},
    "profile_ncu": {"type": "object", "properties": {}, "additionalProperties": False},
    "goal_status": {"type": "object", "properties": {}, "additionalProperties": False},
    "best_result": {"type": "object", "properties": {}, "additionalProperties": False},
    "complete_problem": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
}

TOOL_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "workspace_overview": {"readOnlyHint": True},
    "list_workspace_dir": {"readOnlyHint": True},
    "read_workspace_file": {"readOnlyHint": True},
    "write_candidate": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": False,
    },
    "run_candidate": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "profile_ncu": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "goal_status": {"readOnlyHint": True},
    "best_result": {"readOnlyHint": True},
    "complete_problem": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": False,
        "idempotentHint": False,
    },
}



def build_tool_descriptors(handlers: dict[str, ToolHandler]) -> tuple[ToolDescriptor, ...]:
    descriptors: list[ToolDescriptor] = []
    for spec in MCP_TOOL_SPECS:
        descriptors.append(
            ToolDescriptor(
                name=spec.name,
                description=spec.purpose,
                input_schema=TOOL_SCHEMAS[spec.name],
                annotations=TOOL_ANNOTATIONS.get(spec.name),
                handler=handlers[spec.name],
            )
        )
    return tuple(descriptors)
