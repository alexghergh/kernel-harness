"""Dispatch problem-source-specific behavior used by the harness commands.

The harness orchestration (run, profile, prepare, complete) is mostly generic. The
problem-source-specific bits are the candidate filename and template, the source-level
validator, the evaluator subprocess entrypoint, and how the reference source and baseline
runtime are obtained at workspace prep time. This module exposes one ``ProblemSource``
dataclass and a tiny ``get_problem_source(name)`` helper so the rest of the harness can
stay agnostic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol

PROBLEM_SOURCE_KERNELBENCH = "kernelbench"
PROBLEM_SOURCE_CUDA = "cuda_source"

DEFAULT_PROBLEM_SOURCE = PROBLEM_SOURCE_KERNELBENCH


class CandidateValidationError(ValueError):
    """Shared error type raised by every ProblemSource's candidate validator."""


class _CandidateValidator(Protocol):
    def __call__(self, source: str) -> None: ...


@dataclass(frozen=True)
class BaselinePayload:
    """Result of measuring or loading the per-problem baseline at prep time."""

    runtime_ms: float
    label: str
    extras: dict[str, Any]


@dataclass(frozen=True)
class ReferencePayload:
    """Bundle of facts collected for one problem at prep time."""

    reference_source: str
    problem_name: str
    problem_metadata: dict[str, Any]
    baseline: BaselinePayload


@dataclass(frozen=True)
class EvalContext:
    """Inputs both run-candidate and profile-ncu commands hand to the active source.

    Field semantics:

    * shared: ``candidate_path``, ``output_path``, ``run_name``, ``level``,
      ``problem_id``, ``sample_id`` (or ``profile_id`` for profiling), ``gpu_id``,
      ``workspace``, ``problem_dir``, ``precision``.
    * KernelBench-only: ``kernelbench_root``, ``dataset_src``, ``backend``,
      ``num_correct_trials``, ``num_perf_trials``, ``timing_method``.

    Each problem source consumes the subset it cares about.
    """

    candidate_path: str
    output_path: str
    run_name: str
    level: int
    problem_id: int
    sample_id: int
    gpu_id: int
    workspace: str | None = None
    problem_dir: str | None = None
    precision: str | None = None
    kernelbench_root: str | None = None
    dataset_src: str | None = None
    backend: str | None = None
    num_correct_trials: int | None = None
    num_perf_trials: int | None = None
    timing_method: str | None = None
    sample_label: str | None = None


@dataclass(frozen=True)
class ProblemSource:
    """Minimal description of one problem source the harness can run."""

    name: str
    extension: str  # ".py" or ".cu"; drives archived/snapshot/sample filenames
    candidate_filename: str
    reference_filename: str

    # Candidate source validator. Raises CandidateValidationError on rejection.
    validate_candidate_source: _CandidateValidator
    # Returns the locked candidate template as a string. The initial workspace
    # candidate is seeded with this template.
    candidate_template: Callable[[], str]
    # Normalizes a candidate by collapsing the contents of every editable block to
    # ``<editable>``. The validator uses this to compare non-editable spans against
    # the locked template byte-for-byte.
    normalize_candidate_template: Callable[[str], str]

    # Returns the per-problem reference + metadata + baseline at prep time.
    prepare_problem: Callable[..., ReferencePayload]

    # Build the subprocess argv the run-candidate command should exec for one
    # measured evaluation.
    build_eval_command: Callable[["EvalContext"], list[str]]
    # Build the subprocess argv to wrap inside ncu for one profiling run.
    build_profile_command: Callable[["EvalContext"], list[str]]

    # Reads + validates the candidate at ``path`` and returns the source text.
    read_validated_candidate_source: Callable[..., str]
    # Writes the validated source as the immutable archived attempt snapshot.
    write_run_candidate_snapshot: Callable[..., Any]
    # Writes the validated source as a profile snapshot under the profiles dir.
    write_profile_candidate_snapshot: Callable[..., Any]


_REGISTRY: dict[str, ProblemSource] = {}


def register_problem_source(source: ProblemSource) -> ProblemSource:
    _REGISTRY[source.name] = source
    return source


def get_problem_source(name: str | None) -> ProblemSource:
    key = (name or DEFAULT_PROBLEM_SOURCE).strip()
    if key not in _REGISTRY:
        # Lazy import the two known implementations on first lookup. Importing the
        # subpackage executes its module-level register call below.
        if key == PROBLEM_SOURCE_KERNELBENCH:
            importlib.import_module("kernel_bench_experiment_agents.kernelbench.source")
        elif key == PROBLEM_SOURCE_CUDA:
            importlib.import_module("kernel_bench_experiment_agents.cuda_source")
        else:
            raise RuntimeError(f"unknown problem source: {key!r}")
    if key not in _REGISTRY:
        raise RuntimeError(f"problem source {key!r} failed to register itself")
    return _REGISTRY[key]


