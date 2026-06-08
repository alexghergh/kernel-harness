"""CUDA-source problem-source implementation.

Where the KernelBench source runs Python ``Model`` classes against ``ModelNew`` candidates
under ``eval_kernel_against_ref``, this source compiles a single ``candidate.cu`` file
through ``nvcc`` and runs it. The candidate keeps a fixed locked driver section (which
times a frozen reference such as cuBLAS) and exposes only the kernel implementation and
its launch configuration through editable markers. Measured outcomes come from parsing
the binary's stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kernel_bench_experiment_agents.cuda_source.candidate_scaffold import (
    CANDIDATE_FILENAME,
    REFERENCE_FILENAME,
    candidate_template,
    normalize_candidate_template,
)
from kernel_bench_experiment_agents.cuda_source.problem_loader import prepare_problem
from kernel_bench_experiment_agents.cuda_source.validator import validate_candidate_source
from kernel_bench_experiment_agents.problem_source import (
    EvalContext,
    PROBLEM_SOURCE_CUDA,
    ProblemSource,
    register_problem_source,
)
from kernel_bench_experiment_agents.runtime.project import write_text


_RUNNER_MODULE = "kernel_bench_experiment_agents.cuda_source.runner"
_PROFILER_MODULE = "kernel_bench_experiment_agents.cuda_source.profiler_runner"


def _require(value: str | None, *, label: str) -> str:
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"cuda_source command requires {label}")
    return str(value)


def _build_eval_command(context: EvalContext) -> list[str]:
    return [
        sys.executable, "-m", _RUNNER_MODULE,
        "--candidate", context.candidate_path,
        "--output-path", context.output_path,
        "--run-name", context.run_name,
        "--level", str(context.level),
        "--problem-id", str(context.problem_id),
        "--sample-id", str(context.sample_id),
        "--gpu-id", str(context.gpu_id),
        "--problem-dir", _require(context.problem_dir, label="--problem-dir"),
    ]


def _build_profile_command(context: EvalContext) -> list[str]:
    return [
        sys.executable, "-m", _PROFILER_MODULE,
        "--candidate", context.candidate_path,
        "--run-name", context.run_name,
        "--level", str(context.level),
        "--problem-id", str(context.problem_id),
        "--sample-label", context.sample_label or f"sample_{context.sample_id}",
        "--gpu-id", str(context.gpu_id),
        "--problem-dir", _require(context.problem_dir, label="--problem-dir"),
    ]


def _read_validated_candidate_source(candidate_path: Path) -> str:
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)
    return candidate_src


def _write_run_candidate_snapshot(*, snapshot_path: Path, candidate_src: str, **_ignored) -> Path:
    write_text(snapshot_path, candidate_src)
    return snapshot_path


def _write_profile_candidate_snapshot(
    *, profiles_dir: Path, profile_name: str, candidate_src: str, **_ignored
) -> Path:
    snapshot_path = profiles_dir / f"{profile_name}.candidate.cu"
    write_text(snapshot_path, candidate_src)
    return snapshot_path


PROBLEM_SOURCE = register_problem_source(
    ProblemSource(
        name=PROBLEM_SOURCE_CUDA,
        extension=".cu",
        candidate_filename=CANDIDATE_FILENAME,
        reference_filename=REFERENCE_FILENAME,
        validate_candidate_source=validate_candidate_source,
        candidate_template=candidate_template,
        normalize_candidate_template=normalize_candidate_template,
        prepare_problem=prepare_problem,
        build_eval_command=_build_eval_command,
        build_profile_command=_build_profile_command,
        read_validated_candidate_source=_read_validated_candidate_source,
        write_run_candidate_snapshot=_write_run_candidate_snapshot,
        write_profile_candidate_snapshot=_write_profile_candidate_snapshot,
    )
)
