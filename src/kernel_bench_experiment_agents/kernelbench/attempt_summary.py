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


def solver_attempt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    runtime_error = result_runtime_error(result)
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    error_message = (
        compact_error_summary(runtime_error)
        or compact_error_summary(str(error.get("message") or ""))
    )
    runtime_ms = candidate_runtime(result)
    execution_failed = payload_execution_failed(payload)
    status = payload.get("status")
    if execution_failed and status == "succeeded":
        status = "execution_failed"
    sample_id = payload.get("sample_id")
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
        "artifacts": {
            "sample": f"attempts/sample_{sample_id}.json" if sample_id is not None else None,
            "kernel": payload.get("archive_kernel_path"),
            "stdout": payload.get("stdout_path"),
            "stderr": payload.get("stderr_path"),
        },
    }
