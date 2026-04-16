"""Small shared resource constants for the KernelBench MCP server."""

from __future__ import annotations

from ..candidate_contract import CANDIDATE_FILENAME

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
