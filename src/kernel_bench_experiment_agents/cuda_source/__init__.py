"""CUDA-source problem-source implementation.

Where the KernelBench source runs Python ``Model`` classes against ``ModelNew`` candidates
under ``eval_kernel_against_ref``, this source compiles a single ``candidate.cu`` file
through ``nvcc`` and runs it. The candidate keeps a fixed locked driver section (which
times a frozen reference such as cuBLAS) and exposes only the kernel implementation and
its launch configuration through editable markers. Measured outcomes come from parsing
the binary's stdout.
"""

from __future__ import annotations

from kernel_bench_experiment_agents.cuda_source.candidate_scaffold import (
    CANDIDATE_FILENAME,
    REFERENCE_FILENAME,
    candidate_template,
    normalize_candidate_template,
)
from kernel_bench_experiment_agents.cuda_source.commands import (
    build_eval_command,
    build_profile_command,
)
from kernel_bench_experiment_agents.cuda_source.problem_loader import prepare_problem
from kernel_bench_experiment_agents.cuda_source.snapshot import (
    read_validated_candidate_source,
    write_profile_candidate_snapshot,
    write_run_candidate_snapshot,
)
from kernel_bench_experiment_agents.cuda_source.validator import validate_candidate_source
from kernel_bench_experiment_agents.problem_source import (
    PROBLEM_SOURCE_CUDA,
    ProblemSource,
    register_problem_source,
)


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
        build_eval_command=build_eval_command,
        build_profile_command=build_profile_command,
        read_validated_candidate_source=read_validated_candidate_source,
        write_run_candidate_snapshot=write_run_candidate_snapshot,
        write_profile_candidate_snapshot=write_profile_candidate_snapshot,
    )
)
