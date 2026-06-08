"""Build compact solver-facing summaries for archived candidate attempts."""

from __future__ import annotations

import re
from typing import Any

from kernel_bench_experiment_agents.kernelbench.metrics import (
    candidate_runtime,
    payload_counts_toward_progress,
    payload_execution_failed,
    result_is_correct_with_runtime,
    result_runtime_error,
)


def compact_error_summary(message: str | None) -> str | None:
    if not message:
        return None
    for line in str(message).splitlines():
        stripped = line.strip()
        if not stripped or "error:" not in stripped.lower():
            continue
        cuda_match = re.search(r"([^/\s]+\.cu\(\d+\):\s*error:.*)$", stripped)
        if cuda_match:
            return cuda_match.group(1)
        return stripped[:500]
    return str(message).splitlines()[0].strip()[:500]


def _has_text(metadata: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _diagnostic_artifact_paths(
    metadata: dict[str, Any], sample_id: int | None
) -> dict[str, str | None]:
    """Workspace-relative paths for the per-attempt diagnostic files written into samples/.

    Returns one entry per stream that the runner actually populated. Paths are workspace
    relative so the agent can pass them directly to `read_workspace_file`.
    """
    if sample_id is None:
        return {}
    paths: dict[str, str | None] = {}
    if _has_text(metadata, "build_stderr"):
        paths["workspace_build_stderr"] = f"samples/sample_{sample_id}.build_stderr.txt"
    if _has_text(metadata, "build_stdout", "build_stdout_tail"):
        paths["workspace_build_stdout"] = f"samples/sample_{sample_id}.build_stdout.txt"
    if _has_text(metadata, "stderr_tail"):
        paths["workspace_run_stderr"] = f"samples/sample_{sample_id}.run_stderr.txt"
    if _has_text(metadata, "stdout_tail"):
        paths["workspace_run_stdout"] = f"samples/sample_{sample_id}.run_stdout.txt"
    return paths


def solver_attempt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    runtime_error = result_runtime_error(result)
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    error_message = (
        compact_error_summary(metadata.get("build_stderr"))
        or compact_error_summary(metadata.get("stderr_tail"))
        or compact_error_summary(runtime_error)
        or compact_error_summary(str(error.get("message") or ""))
    )
    runtime_ms = candidate_runtime(result)
    execution_failed = payload_execution_failed(payload)
    status = payload.get("status")
    if execution_failed and status == "succeeded":
        status = "execution_failed"
    sample_id = payload.get("sample_id")
    artifacts: dict[str, Any] = {
        "sample": f"attempts/sample_{sample_id}.json" if sample_id is not None else None,
        "workspace_summary": (
            f"samples/sample_{sample_id}.summary.json" if sample_id is not None else None
        ),
        "kernel": payload.get("archive_kernel_path"),
        "stdout": payload.get("stdout_path"),
        "stderr": payload.get("stderr_path"),
    }
    artifacts.update(_diagnostic_artifact_paths(metadata, sample_id))
    return {
        "status": status,
        "sample_id": sample_id,
        "counts_toward_progress": payload_counts_toward_progress(payload),
        "execution_failed": execution_failed,
        "correctness": result_is_correct_with_runtime(result),
        "runtime_ms": runtime_ms,
        "ref_runtime_ms": result.get("ref_runtime") if runtime_ms is not None else None,
        "correctness_trials": metadata.get("correctness_trials"),
        "warnings": payload.get("warnings") if isinstance(payload.get("warnings"), list) else [],
        "error": error_message,
        "build_command": metadata.get("build_command"),
        "artifacts": artifacts,
    }
