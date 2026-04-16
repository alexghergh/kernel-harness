"""Tiny standalone MCP server for manual Codex/Claude smoke tests.

This server is intentionally separate from the harness workspace contract. It exposes a minimal
rooted file surface so we can verify MCP startup and tool calling outside the full harness.
"""

from __future__ import annotations

from pathlib import Path
import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("kernelbench-dev")


def root_dir() -> Path:
    root = Path(os.environ.get("MCP_DEV_ROOT") or ".").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_path(path: str) -> Path:
    candidate = (root_dir() / path).resolve()
    try:
        candidate.relative_to(root_dir())
    except ValueError as exc:
        raise RuntimeError("path escapes MCP_DEV_ROOT") from exc
    return candidate


@mcp.tool(name="list_dir", description="List files under MCP_DEV_ROOT.")
def list_dir(path: str = ".") -> str:
    target = resolve_path(path)
    if not target.exists() or not target.is_dir():
        raise RuntimeError(f"directory does not exist: {path}")
    lines: list[str] = []
    for child in sorted(target.iterdir(), key=lambda entry: entry.name):
        suffix = "/" if child.is_dir() else ""
        rel = child.relative_to(root_dir())
        lines.append(f"{rel}{suffix}")
    return "\n".join(lines)


@mcp.tool(name="read_text_file", description="Read one UTF-8 text file under MCP_DEV_ROOT.")
def read_text_file(path: str) -> str:
    target = resolve_path(path)
    if not target.exists() or not target.is_file():
        raise RuntimeError(f"file does not exist: {path}")
    return target.read_text(encoding="utf-8")


@mcp.tool(name="write_text_file", description="Write one UTF-8 text file under MCP_DEV_ROOT.")
def write_text_file(path: str, content: str) -> str:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.relative_to(root_dir()))


@mcp.resource(
    "kbdev://root",
    name="root",
    description="The rooted directory used by the dev MCP smoke-test server.",
    mime_type="text/plain",
)
def root_resource() -> str:
    return str(root_dir())


def run() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
