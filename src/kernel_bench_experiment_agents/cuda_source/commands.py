"""Build the subprocess argv that the cuda_source eval and profile commands exec."""

from __future__ import annotations

import sys

from kernel_bench_experiment_agents.problem_source import EvalContext


_RUNNER_MODULE = "kernel_bench_experiment_agents.cuda_source.runner"
_PROFILER_MODULE = "kernel_bench_experiment_agents.cuda_source.profiler_runner"


def _require(value: str | None, *, label: str) -> str:
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"cuda_source command requires {label}")
    return str(value)


def build_eval_command(context: EvalContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        "--candidate",
        context.candidate_path,
        "--output-path",
        context.output_path,
        "--run-name",
        context.run_name,
        "--level",
        str(context.level),
        "--problem-id",
        str(context.problem_id),
        "--sample-id",
        str(context.sample_id),
        "--gpu-id",
        str(context.gpu_id),
        "--problem-dir",
        _require(context.problem_dir, label="--problem-dir"),
    ]


def build_profile_command(context: EvalContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        _PROFILER_MODULE,
        "--candidate",
        context.candidate_path,
        "--run-name",
        context.run_name,
        "--level",
        str(context.level),
        "--problem-id",
        str(context.problem_id),
        "--sample-label",
        context.sample_label or f"sample_{context.sample_id}",
        "--gpu-id",
        str(context.gpu_id),
        "--problem-dir",
        _require(context.problem_dir, label="--problem-dir"),
    ]
