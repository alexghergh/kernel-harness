"""Apply the harness workspace read/write policy inside the MCP server."""

from __future__ import annotations

from pathlib import Path

from kernel_bench_experiment_agents.agent_contract.policy import WORKSPACE_BROWSE_DIRS, workspace_edit_surface, workspace_read_surface
from .context import ServerContext


RESOURCE_LIST_DIRS: tuple[str, ...] = tuple(entry.rstrip("/") for entry in WORKSPACE_BROWSE_DIRS)



def resolve_workspace_path(ctx: ServerContext, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = ctx.workspace / candidate
    return candidate.resolve()



def safe_relative(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return "." if str(rel) == "" else str(rel)
    except ValueError:
        return str(path)



def allowed_directory(path: Path, workspace: Path) -> bool:
    allowed_dirs = {
        (workspace / entry).resolve() if entry != "." else workspace.resolve()
        for entry in RESOURCE_LIST_DIRS
    }
    return path.resolve() in allowed_dirs



def assert_allowed_read(ctx: ServerContext, path: Path) -> None:
    allowed_read_paths, allowed_read_roots = workspace_read_surface(ctx.workspace)
    resolved = path.resolve()
    if resolved in allowed_read_paths:
        return
    if any(resolved == root or root in resolved.parents for root in allowed_read_roots):
        return
    raise RuntimeError(f"read path is outside the allowed workspace surface: {path}")



def assert_allowed_edit(ctx: ServerContext, path: Path) -> None:
    allowed_edit_paths = workspace_edit_surface(ctx.workspace)
    resolved = path.resolve()
    if resolved not in allowed_edit_paths:
        raise RuntimeError(f"edit path is outside the allowed workspace edit surface: {path}")
