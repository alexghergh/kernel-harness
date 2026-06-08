"""Adapter that registers the KernelBench package as a harness-level ProblemSource.

Importing this module side-effects ``problem_source._REGISTRY`` so the rest of the harness
can look up the KernelBench source by name. The actual KernelBench-specific behavior lives
in the existing submodules; this file just wires them onto the typed surface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.kernelbench.candidate.contract import (
    CANDIDATE_FILENAME,
    candidate_template,
    normalize_candidate_template,
)
from kernel_bench_experiment_agents.kernelbench.candidate.snapshot import (
    read_validated_candidate_source as kb_read_validated_candidate_source,
    write_profile_candidate_snapshot as kb_write_profile_candidate_snapshot,
    write_run_candidate_snapshot as kb_write_run_candidate_snapshot,
)
from kernel_bench_experiment_agents.kernelbench.candidate.validation import (
    validate_candidate_source as kb_validate_candidate_source,
)
from kernel_bench_experiment_agents.kernelbench.metrics import (
    baseline_file_paths,
    baseline_payload_for_problem,
)
from kernel_bench_experiment_agents.kernelbench.problems import load_problem
from kernel_bench_experiment_agents.problem_source import (
    BaselinePayload,
    EvalContext,
    PROBLEM_SOURCE_KERNELBENCH,
    ProblemSource,
    ReferencePayload,
    register_problem_source,
)
from kernel_bench_experiment_agents.runtime.project import official_kernel_path


REFERENCE_FILENAME = "problem_reference.py"
EXTENSION = ".py"

EVAL_SUBPROCESS_MODULE = "kernel_bench_experiment_agents.kernelbench.runners.evaluation"
PROFILE_SUBPROCESS_MODULE = "kernel_bench_experiment_agents.kernelbench.profiling.runner"


def _kb_build_eval_command(context: EvalContext) -> list[str]:
    command = [
        sys.executable,
        "-m",
        EVAL_SUBPROCESS_MODULE,
        "--candidate",
        context.candidate_path,
        "--output-path",
        context.output_path,
        "--level",
        str(context.level),
        "--problem-id",
        str(context.problem_id),
        "--dataset-src",
        context.dataset_src or "local",
        "--gpu-id",
        str(context.gpu_id),
        "--run-name",
        context.run_name,
        "--sample-id",
        str(context.sample_id),
        "--backend",
        context.backend or "cuda",
        "--precision",
        context.precision or "bf16",
        "--num-correct-trials",
        str(context.num_correct_trials or 5),
        "--num-perf-trials",
        str(context.num_perf_trials or 100),
    ]
    if context.kernelbench_root:
        command.extend(["--kernelbench-root", context.kernelbench_root])
    if context.timing_method is not None:
        command.extend(["--timing-method", context.timing_method])
    return command


def _kb_build_profile_command(context: EvalContext) -> list[str]:
    command = [
        sys.executable,
        "-m",
        PROFILE_SUBPROCESS_MODULE,
        "--candidate",
        context.candidate_path,
        "--level",
        str(context.level),
        "--problem-id",
        str(context.problem_id),
        "--dataset-src",
        context.dataset_src or "local",
        "--gpu-id",
        str(context.gpu_id),
        "--run-name",
        context.run_name,
        "--sample-label",
        context.sample_label or f"sample_{context.sample_id}",
        "--precision",
        context.precision or "bf16",
    ]
    if context.kernelbench_root:
        command.extend(["--kernelbench-root", context.kernelbench_root])
    return command


def _kb_write_run_snapshot(
    *,
    snapshot_path: Path,
    candidate_src: str,
    **_ignored: Any,
) -> Path:
    # The KernelBench writer keeps the original behavior of writing through
    # ``write_text``; passing the harness-computed ``snapshot_path`` directly
    # lets the command own filename selection across all problem sources.
    from kernel_bench_experiment_agents.runtime.project import write_text as _write_text

    _write_text(snapshot_path, candidate_src)
    return snapshot_path


def _kb_write_profile_snapshot(
    *,
    profiles_dir: Path,
    profile_name: str,
    candidate_src: str,
    **_ignored: Any,
) -> Path:
    return kb_write_profile_candidate_snapshot(
        profiles_dir=profiles_dir,
        profile_name=profile_name,
        candidate_src=candidate_src,
    )


def _kb_prepare_problem(
    *,
    level: int,
    problem_id: int,
    dataset_src: str = "local",
    kernelbench_root: str | None = None,
    timings_dir: str | None = None,
    hardware_name: str | None = None,
    **_ignored: Any,
) -> ReferencePayload:
    if not hardware_name:
        raise RuntimeError("kernelbench.prepare_problem requires --hardware-name")
    problem = load_problem(
        level=level,
        problem_id=problem_id,
        dataset_src=dataset_src,
        explicit_kernelbench_root=kernelbench_root,
    )
    eager_baseline_file, compile_baseline_file = baseline_file_paths(
        kernelbench_root=kernelbench_root,
        timings_dir=timings_dir,
        hardware_name=hardware_name,
    )
    payload = baseline_payload_for_problem(
        level=level,
        problem_id=problem_id,
        problem_name=problem.name,
        eager_baseline_file=eager_baseline_file,
        compile_baseline_file=compile_baseline_file,
    )
    eager_runtime_ms = float(payload["eager"]["runtime_ms"])
    compile_runtime_ms = float(payload["compile"]["runtime_ms"])
    # The unified baseline is the harder of the two PyTorch references — whichever
    # is faster, since "beating both" was the original goal and we collapse to one.
    primary_runtime_ms = min(eager_runtime_ms, compile_runtime_ms)
    primary_label = "torch.compile" if compile_runtime_ms <= eager_runtime_ms else "torch.eager"
    baseline = BaselinePayload(
        runtime_ms=primary_runtime_ms,
        label=primary_label,
        extras={
            "eager_runtime_ms": eager_runtime_ms,
            "compile_runtime_ms": compile_runtime_ms,
            "eager_baseline_file": str(eager_baseline_file),
            "compile_baseline_file": str(compile_baseline_file),
            "measurement_source": "kernelbench_results_timing",
        },
    )
    metadata = {
        "name": problem.name,
        "dataset_src": dataset_src,
        "source_path": getattr(problem, "path", None),
    }
    return ReferencePayload(
        reference_source=problem.code,
        problem_name=problem.name or f"level_{level}_problem_{problem_id}",
        problem_metadata=metadata,
        baseline=baseline,
    )


PROBLEM_SOURCE = register_problem_source(
    ProblemSource(
        name=PROBLEM_SOURCE_KERNELBENCH,
        extension=EXTENSION,
        candidate_filename=CANDIDATE_FILENAME,
        reference_filename=REFERENCE_FILENAME,
        validate_candidate_source=kb_validate_candidate_source,
        candidate_template=candidate_template,
        normalize_candidate_template=normalize_candidate_template,
        prepare_problem=_kb_prepare_problem,
        build_eval_command=_kb_build_eval_command,
        build_profile_command=_kb_build_profile_command,
        read_validated_candidate_source=kb_read_validated_candidate_source,
        write_run_candidate_snapshot=_kb_write_run_snapshot,
        write_profile_candidate_snapshot=_kb_write_profile_snapshot,
    )
)


def archived_kernel_path(run_name: str, level: int, problem_id: int, sample_id: int) -> Path:
    return official_kernel_path(run_name, level, problem_id, sample_id)
