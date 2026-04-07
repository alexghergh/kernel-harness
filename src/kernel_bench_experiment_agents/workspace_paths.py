from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME
from .project import workspace_dir, write_json, write_text


def workspace_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workspace_metadata(workspace: Path) -> dict[str, Any]:
    return read_json_file(workspace / "problem.json")


def load_workspace_baseline(workspace: Path) -> dict[str, Any]:
    return read_json_file(workspace / "baseline.json")


def problem_workspace_paths(
    run_name: str,
    level: int,
    problem_id: int,
    workspace_root: str | None,
) -> dict[str, Path]:
    workspace = workspace_dir(run_name, level, problem_id, explicit_root=workspace_root)
    return {
        "workspace": workspace,
        "samples": workspace / "samples",
        "profiles": workspace / "profiles",
        "bin": workspace / "bin",
    }


def workspace_candidate_path(workspace: Path) -> Path:
    return workspace / CANDIDATE_FILENAME


def workspace_samples_dir(workspace: Path) -> Path:
    return workspace / "samples"


def workspace_profiles_dir(workspace: Path) -> Path:
    return workspace / "profiles"


def workspace_relpath(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def write_workspace_sample_copy(
    workspace: Path,
    sample_id: int,
    candidate_src: str,
) -> None:
    write_text(
        workspace_samples_dir(workspace) / f"sample_{sample_id}.py",
        candidate_src,
    )


def write_workspace_best_sample(
    workspace: Path,
    payload: dict[str, Any] | None,
) -> None:
    best_sample_path = workspace_samples_dir(workspace) / "best_sample.py"
    best_result_path = workspace_samples_dir(workspace) / "best_result.json"
    if payload is None:
        if best_sample_path.exists():
            best_sample_path.unlink()
        if best_result_path.exists():
            best_result_path.unlink()
        return

    official_kernel = payload.get("official_kernel_path")
    if isinstance(official_kernel, str):
        official_kernel_path = Path(official_kernel)
        if official_kernel_path.exists():
            write_text(
                best_sample_path,
                official_kernel_path.read_text(encoding="utf-8"),
            )
        elif best_sample_path.exists():
            best_sample_path.unlink()
    elif best_sample_path.exists():
        best_sample_path.unlink()
    write_json(best_result_path, payload)


def latest_workspace_profile_paths(workspace: Path) -> dict[str, Path]:
    profiles_dir = workspace_profiles_dir(workspace)
    return {
        "details": profiles_dir / "latest.details.txt",
        "details_stderr": profiles_dir / "latest.details.stderr.txt",
        "raw_csv": profiles_dir / "latest.raw.csv",
        "raw_csv_stderr": profiles_dir / "latest.raw.stderr.txt",
        "summary": profiles_dir / "latest.summary.txt",
        "stdout": profiles_dir / "latest.stdout.txt",
        "stderr": profiles_dir / "latest.stderr.txt",
        "json": profiles_dir / "latest.json",
    }
