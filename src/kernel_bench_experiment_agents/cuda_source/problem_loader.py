"""Load one cuda_source problem definition from disk and measure its baseline.

A cuda_source problem lives in a directory containing:

* ``problem.json``  — metadata (name, shapes, tolerance, build flags, arch, baseline label)
* ``reference.cu``  — the locked candidate scaffold seeded with the starter kernel; the
  workspace uses this as both the read-only reference and the initial ``candidate.cu``.

At prep time the harness compiles ``reference.cu`` once and runs it. The cuBLAS GFLOPS
print from the locked driver becomes the per-problem baseline so the agent's goal is
unambiguous: beat that number.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.cuda_source.build import build_candidate_binary
from kernel_bench_experiment_agents.cuda_source.runner import parse_result_line
from kernel_bench_experiment_agents.problem_source import BaselinePayload, ReferencePayload


def _resolve_problem_dir(problem_dir: str | None) -> Path:
    candidate = problem_dir or os.environ.get("PROBLEM_DIR")
    if not candidate:
        raise RuntimeError(
            "cuda_source.prepare_problem requires --problem-dir or PROBLEM_DIR to be set."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"problem directory does not exist: {path}")
    return path


def _load_problem_metadata(problem_dir: Path) -> dict[str, Any]:
    payload = json.loads((problem_dir / "problem.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"problem.json at {problem_dir / 'problem.json'} is not a JSON object")
    return payload


def _measure_baseline(
    *,
    reference_path: Path,
    problem_metadata: dict[str, Any],
    build_timeout_seconds: float,
    run_timeout_seconds: float,
) -> BaselinePayload:
    """Compile + run the reference once at prep time to capture the locked baseline."""
    with tempfile.TemporaryDirectory(prefix="cuda_source_baseline_") as scratch:
        scratch_path = Path(scratch)
        build_result = build_candidate_binary(
            candidate_path=reference_path,
            build_dir=scratch_path,
            problem_metadata=problem_metadata,
            timeout_seconds=build_timeout_seconds,
        )
        if build_result.returncode != 0:
            raise RuntimeError(
                "Failed to compile the cuda_source reference at prep time. nvcc stderr:\n"
                + (build_result.stderr or "")
            )
        completed = subprocess.run(
            [str(build_result.binary_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(scratch_path),
            timeout=run_timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "cuda_source reference binary failed during baseline measurement."
                f" returncode={completed.returncode}\nstderr:\n{(completed.stderr or '')[-2000:]}"
            )
        parsed = parse_result_line(completed.stdout or "")
        if parsed is None:
            raise RuntimeError(
                "cuda_source reference binary did not emit a RESULT line during baseline measurement."
            )
    return BaselinePayload(
        runtime_ms=parsed.cublas_ms,
        label=str(problem_metadata.get("baseline_label") or "cublas"),
        extras={
            "cublas_gflops": parsed.cublas_gflops,
            "cublas_ms": parsed.cublas_ms,
            "starter_candidate_gflops": parsed.candidate_gflops,
            "starter_candidate_ms": parsed.candidate_ms,
            "num_flops": parsed.num_flops,
            "mean_abs_error": parsed.mean_abs_error,
            "tolerance": float(problem_metadata.get("tolerance") or 1e-2),
            "measurement_source": "prep_time_nvcc_run",
        },
    )


def prepare_problem(
    *,
    level: int,
    problem_id: int,
    problem_dir: str | None = None,
    skip_baseline_measurement: bool = False,
    build_timeout_seconds: float = 240.0,
    run_timeout_seconds: float = 300.0,
    **_ignored: Any,
) -> ReferencePayload:
    resolved_dir = _resolve_problem_dir(problem_dir)
    problem_metadata = _load_problem_metadata(resolved_dir)
    reference_path = resolved_dir / "reference.cu"
    if not reference_path.exists():
        raise RuntimeError(f"reference.cu not found at {reference_path}")

    reference_source = reference_path.read_text(encoding="utf-8")

    if skip_baseline_measurement or problem_metadata.get("skip_baseline_measurement"):
        runtime_hint = problem_metadata.get("baseline_runtime_ms_hint")
        if runtime_hint is None:
            raise RuntimeError(
                "baseline measurement is skipped but problem.json has no baseline_runtime_ms_hint."
            )
        baseline = BaselinePayload(
            runtime_ms=float(runtime_hint),
            label=str(problem_metadata.get("baseline_label") or "cublas"),
            extras={
                "measurement_source": "problem_json_hint",
                "tolerance": float(problem_metadata.get("tolerance") or 1e-2),
            },
        )
    else:
        baseline = _measure_baseline(
            reference_path=reference_path,
            problem_metadata=problem_metadata,
            build_timeout_seconds=build_timeout_seconds,
            run_timeout_seconds=run_timeout_seconds,
        )

    enriched_metadata = dict(problem_metadata)
    enriched_metadata.setdefault("level", level)
    enriched_metadata.setdefault("problem_id", problem_id)
    enriched_metadata["problem_dir"] = str(resolved_dir)
    return ReferencePayload(
        reference_source=reference_source,
        problem_name=str(problem_metadata.get("name") or resolved_dir.name),
        problem_metadata=enriched_metadata,
        baseline=baseline,
    )
