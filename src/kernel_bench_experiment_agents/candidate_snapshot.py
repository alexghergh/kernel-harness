"""Helpers for freezing validated candidate sources into immutable archived snapshots.

Run and profile paths both need to operate on the exact bytes that were validated and
archived, instead of re-reading a mutable workspace file path later in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from .candidate_validation import validate_candidate_source
from .project import official_kernel_path, write_text


# Candidate snapshots are small, so keeping the shared freeze/write logic in one module
# makes the run/profile paths consistent without duplicating validation code.
def read_validated_candidate_source(candidate_path: Path) -> str:
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)
    return candidate_src


def write_run_candidate_snapshot(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    sample_id: int,
    candidate_src: str,
) -> Path:
    snapshot_path = official_kernel_path(run_name, level, problem_id, sample_id)
    write_text(snapshot_path, candidate_src)
    return snapshot_path


def write_profile_candidate_snapshot(
    *,
    profiles_dir: Path,
    profile_name: str,
    candidate_src: str,
) -> Path:
    snapshot_path = profiles_dir / f"{profile_name}.candidate.py"
    write_text(snapshot_path, candidate_src)
    return snapshot_path
