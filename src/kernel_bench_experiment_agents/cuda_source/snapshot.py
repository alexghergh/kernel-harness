"""Freeze validated CUDA candidate sources into immutable archived snapshots.

This mirrors ``kernelbench.candidate.snapshot``: the run and profile paths operate on the
exact bytes that were validated, rather than re-reading the mutable workspace file later.
"""

from __future__ import annotations

from pathlib import Path

from kernel_bench_experiment_agents.cuda_source.validator import validate_candidate_source
from kernel_bench_experiment_agents.runtime.project import write_text


def read_validated_candidate_source(candidate_path: Path) -> str:
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)
    return candidate_src


def write_run_candidate_snapshot(
    *,
    snapshot_path: Path,
    candidate_src: str,
    **_ignored,
) -> Path:
    write_text(snapshot_path, candidate_src)
    return snapshot_path


def write_profile_candidate_snapshot(
    *,
    profiles_dir: Path,
    profile_name: str,
    candidate_src: str,
    **_ignored,
) -> Path:
    snapshot_path = profiles_dir / f"{profile_name}.candidate.cu"
    write_text(snapshot_path, candidate_src)
    return snapshot_path
