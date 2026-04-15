"""Load the per-problem MCP context from the launcher environment and workspace metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import normalize_tool_name
from ..project import archive_contract_dir
from ..workspace_paths import load_workspace_metadata, validate_workspace_assignment


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
