"""Profiling subprocess entrypoint for cuda_source.

The profile-ncu command launches this module under ``ncu``. The module compiles the
candidate (reusing the runner's nvcc invocation) and then execs the resulting binary so
NCU has a single CUDA-using process to attach to. The binary still runs the full
cuBLAS reference + custom kernel loops on the assigned shape; NCU records both.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kernel_bench_experiment_agents.cuda_source.build import build_candidate_binary
from kernel_bench_experiment_agents.cuda_source.runner import load_problem_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cuda_source candidate binary for ncu profiling.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--problem-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--sample-label", default="profile_scratch")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--build-timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    problem_dir = Path(args.problem_dir).expanduser().resolve()
    problem_metadata = load_problem_metadata(problem_dir)

    build_dir = candidate_path.parent / f"profile_build_{args.sample_label}"
    build_result = build_candidate_binary(
        candidate_path=candidate_path,
        build_dir=build_dir,
        problem_metadata=problem_metadata,
        timeout_seconds=args.build_timeout_seconds,
    )
    if build_result.returncode != 0:
        sys.stderr.write(build_result.stderr or "")
        raise SystemExit(
            f"nvcc returned {build_result.returncode} while compiling candidate for profiling"
        )

    # The parent command sets CUDA_VISIBLE_DEVICES via the GPU lease before
    # spawning this subprocess. Do not overwrite it here.
    #
    # Pass --profile-only so the locked driver skips the cuBLAS reference loop:
    # NCU should record only the candidate kernel, not the cuBLAS GEMM that runs
    # first in the normal run-candidate flow.
    env = os.environ.copy()
    binary = str(build_result.binary_path)
    os.execve(binary, [binary, "--profile-only"], env)


if __name__ == "__main__":
    main()
