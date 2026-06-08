"""Static resource descriptors for the KernelBench MCP server.

Resources stay intentionally narrow and read-only: a fixed set of canonical docs/code files.
History browsing for `samples/` and `profiles/` stays on explicit MCP tools so the agent can see
past attempts without getting a vague "read any path" resource template.

The actual list of registered resources is computed per-workspace at server startup from the
active problem source; see ``app._register_workspace_resources``.
"""

from __future__ import annotations

RESOURCE_URI_PREFIX = "kb://workspace/"


def workspace_resource_uri(relative_path: str) -> str:
    return f"{RESOURCE_URI_PREFIX}{relative_path}"


def workspace_resource_name(relative_path: str) -> str:
    slug = relative_path.replace("/", "_").replace(".", "_")
    return f"workspace_{slug}"
