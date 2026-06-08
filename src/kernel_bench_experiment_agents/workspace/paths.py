"""Resolve live workspace paths and read the generated workspace metadata files.

Run, profile, status, and completion commands all use these helpers to verify they are operating on the assigned workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.kernelbench.attempt_summary import solver_attempt_summary
from kernel_bench_experiment_agents.problem_source import (
    DEFAULT_PROBLEM_SOURCE,
    ProblemSource,
    get_problem_source,
)
from kernel_bench_experiment_agents.runtime.project import archive_problem_dir, workspace_dir, write_json, write_text


def workspace_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workspace_metadata(workspace: Path) -> dict[str, Any]:
    return read_json_file(workspace / "problem.json")


def workspace_problem_source(workspace: Path) -> ProblemSource:
    """Look up the registered ProblemSource recorded in this workspace's problem.json."""
    metadata = load_workspace_metadata(workspace)
    name = metadata.get("problem_source") if isinstance(metadata, dict) else None
    return get_problem_source(str(name) if name else DEFAULT_PROBLEM_SOURCE)


def load_workspace_baseline(workspace: Path) -> dict[str, Any]:
    """Return the unified single-baseline payload recorded for this workspace."""
    problem = read_json_file(workspace / "problem.json")
    baseline_runtime_ms = problem.get("baseline_runtime_ms") if isinstance(problem, dict) else None
    baseline_label = problem.get("baseline_label") if isinstance(problem, dict) else None
    baseline_extras = problem.get("baseline_extras") if isinstance(problem, dict) else None
    return {
        "runtime_ms": baseline_runtime_ms,
        "label": baseline_label,
        "extras": baseline_extras if isinstance(baseline_extras, dict) else {},
    }


def validate_workspace_assignment(
    workspace: Path,
    *,
    run_name: str,
    level: int,
    problem_id: int,
) -> dict[str, Any]:
    metadata = load_workspace_metadata(workspace)
    expected = {
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
    }
    actual = {
        "run_name": metadata.get("run_name"),
        "level": metadata.get("level"),
        "problem_id": metadata.get("problem_id"),
    }
    if actual != expected:
        raise RuntimeError(
            "Workspace assignment does not match the requested run/problem: "
            f"expected {expected}, got {actual}."
        )
    return metadata


def problem_workspace_paths(
    run_name: str,
    level: int,
    problem_id: int,
) -> dict[str, Path]:
    workspace = workspace_dir(run_name, level, problem_id)
    return {
        "workspace": workspace,
        "samples": workspace / "samples",
        "profiles": workspace / "profiles",
        "bin": workspace / "bin",
    }


def workspace_candidate_path(workspace: Path) -> Path:
    return workspace / workspace_problem_source(workspace).candidate_filename


def workspace_reference_path(workspace: Path) -> Path:
    return workspace / workspace_problem_source(workspace).reference_filename


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
    extension = workspace_problem_source(workspace).extension
    write_text(
        workspace_samples_dir(workspace) / f"sample_{sample_id}{extension}",
        candidate_src,
    )


def write_workspace_sample_summary(
    workspace: Path,
    sample_id: int,
    payload: dict[str, Any],
) -> None:
    """Mirror the solver-facing attempt summary next to samples/sample_<id>.cu.

    The agent cannot read the full archive payload (it lives outside the workspace
    surface); this drops the summary into samples/ so each past attempt's outcome
    stays readable via `read_workspace_file` after the immediate run_candidate
    response has scrolled out of context.
    """
    summary = solver_attempt_summary(payload)
    write_json(
        workspace_samples_dir(workspace) / f"sample_{sample_id}.summary.json",
        summary,
    )


def write_workspace_sample_diagnostics(
    workspace: Path,
    sample_id: int,
    payload: dict[str, Any],
) -> None:
    """Persist the verbose nvcc + binary stderr/stdout into samples/ as plain text.

    The runner captures these streams in `result.metadata` (`build_stderr`,
    `build_stdout`, `stderr_tail`, `stdout_tail`). The MCP response carries only
    paths to these files so the agent can pull the full diagnostic via
    `read_workspace_file` instead of receiving multi-KB strings inline.
    """
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    samples_dir = workspace_samples_dir(workspace)

    file_writes = {
        f"sample_{sample_id}.build_stderr.txt": metadata.get("build_stderr"),
        f"sample_{sample_id}.build_stdout.txt": (
            metadata.get("build_stdout") or metadata.get("build_stdout_tail")
        ),
        f"sample_{sample_id}.run_stderr.txt": metadata.get("stderr_tail"),
        f"sample_{sample_id}.run_stdout.txt": metadata.get("stdout_tail"),
    }
    for filename, text in file_writes.items():
        if isinstance(text, str) and text.strip():
            write_text(samples_dir / filename, text)


def write_workspace_best_sample(
    workspace: Path,
    payload: dict[str, Any] | None,
) -> None:
    extension = workspace_problem_source(workspace).extension
    best_sample_path = workspace_samples_dir(workspace) / f"best_sample{extension}"
    best_result_path = workspace_samples_dir(workspace) / "best_result.json"
    if payload is None:
        if best_sample_path.exists():
            best_sample_path.unlink()
        if best_result_path.exists():
            best_result_path.unlink()
        return

    archive_kernel = payload.get("archive_kernel_path") or payload.get("official_kernel_path")
    if isinstance(archive_kernel, str):
        metadata = load_workspace_metadata(workspace)
        kernel_path = Path(archive_kernel)
        if not kernel_path.is_absolute():
            kernel_path = archive_problem_dir(
                metadata["run_name"],
                int(metadata["level"]),
                int(metadata["problem_id"]),
            ) / kernel_path
        if kernel_path.exists():
            write_text(
                best_sample_path,
                kernel_path.read_text(encoding="utf-8"),
            )
        elif best_sample_path.exists():
            best_sample_path.unlink()
    elif best_sample_path.exists():
        best_sample_path.unlink()
    write_json(best_result_path, solver_attempt_summary(payload))


def latest_workspace_profile_paths(workspace: Path) -> dict[str, Path]:
    profiles_dir = workspace_profiles_dir(workspace)
    return {
        "details": profiles_dir / "latest.details.txt",
        "summary": profiles_dir / "latest.summary.txt",
        "stdout": profiles_dir / "latest.stdout.txt",
        "stderr": profiles_dir / "latest.stderr.txt",
        "json": profiles_dir / "latest.json",
    }
