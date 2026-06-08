"""Drive ``nvcc`` to compile one candidate.cu into an executable binary.

Build flags and the target architecture come from the problem-definition ``problem.json``
so different cuda_source problems can pin different toolchain parameters.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildResult:
    binary_path: Path
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def _resolve_nvcc(extra_search_paths: list[str] | None = None) -> str:
    candidate = shutil.which("nvcc")
    if candidate:
        return candidate
    for entry in extra_search_paths or []:
        path = Path(entry).expanduser() / "nvcc"
        if path.exists():
            return str(path)
    raise RuntimeError(
        "nvcc is not on PATH. Load the CUDA toolchain (e.g. ``module load cuda``) before running this command."
    )


def build_candidate_binary(
    *,
    candidate_path: Path,
    build_dir: Path,
    problem_metadata: dict[str, Any],
    timeout_seconds: float = 240.0,
) -> BuildResult:
    """Compile ``candidate.cu`` with ``nvcc`` using the flags from ``problem.json``.

    Returns a ``BuildResult`` with the compiler return code, stdout, stderr, and the
    final binary path (whether or not the compile succeeded).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / "candidate_binary"
    arch = problem_metadata.get("arch") or problem_metadata.get("cuda_arch") or "sm_90"
    build_flags = problem_metadata.get("build_flags")
    if isinstance(build_flags, list):
        extra_flags = [str(entry) for entry in build_flags]
    else:
        extra_flags = ["-O3", "-std=c++17"]
    libs = problem_metadata.get("libs")
    if isinstance(libs, list):
        link_flags = [f"-l{entry}" if not str(entry).startswith("-l") else str(entry) for entry in libs]
    else:
        link_flags = ["-lcublas"]

    nvcc = _resolve_nvcc()
    command = [
        nvcc,
        *extra_flags,
        f"-arch={arch}",
        str(candidate_path),
        "-o",
        str(binary_path),
        *link_flags,
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return BuildResult(
        binary_path=binary_path,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=command,
    )
