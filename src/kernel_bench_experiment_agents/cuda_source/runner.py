"""Compile + run one cuda_source candidate.cu and persist the measured payload.

This module is the subprocess entrypoint spawned by the run-candidate command. The
higher-level command owns GPU leasing, artifact locking, and goal-status updates; this
runner only does:

1. compile the candidate with nvcc using the problem's build flags;
2. execute the resulting binary inside the assigned GPU slot;
3. parse the locked driver's structured RESULT line for cublas / candidate timings and
   the mean-absolute-error verification;
4. write a KB-compatible evaluation_result.json so downstream metrics/summary code does
   not need to know which problem source produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.cuda_source.build import BuildResult, build_candidate_binary
from kernel_bench_experiment_agents.cuda_source.candidate_scaffold import RESULT_LINE_PREFIX


_RESULT_LINE_PATTERN = re.compile(
    re.escape(RESULT_LINE_PREFIX) + r"(?P<body>.+)$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class ParsedResult:
    cublas_gflops: float
    candidate_gflops: float
    cublas_ms: float
    candidate_ms: float
    mean_abs_error: float
    num_flops: int
    extras: dict[str, Any]


def parse_result_line(stdout: str) -> ParsedResult | None:
    match = _RESULT_LINE_PATTERN.search(stdout)
    if match is None:
        return None
    body = match.group("body").strip()
    pairs: dict[str, str] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key.strip()] = value.strip()
    try:
        return ParsedResult(
            cublas_gflops=float(pairs["cublas_gflops"]),
            candidate_gflops=float(pairs["candidate_gflops"]),
            cublas_ms=float(pairs["cublas_ms"]),
            candidate_ms=float(pairs["candidate_ms"]),
            mean_abs_error=float(pairs["mean_abs_error"]),
            num_flops=int(pairs["num_flops"]),
            extras={k: v for k, v in pairs.items() if k not in {
                "cublas_gflops",
                "candidate_gflops",
                "cublas_ms",
                "candidate_ms",
                "mean_abs_error",
                "num_flops",
            }},
        )
    except (KeyError, ValueError):
        return None


def load_problem_metadata(problem_dir: Path) -> dict[str, Any]:
    metadata_path = problem_dir / "problem.json"
    if not metadata_path.exists():
        raise RuntimeError(f"problem.json not found at {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"problem.json at {metadata_path} did not contain a JSON object")
    return payload


_DIAGNOSTIC_TEXT_CAP: int = 200_000


def _tail(text: str | None, cap: int = _DIAGNOSTIC_TEXT_CAP) -> str:
    if not text:
        return ""
    return text[-cap:]


def _payload_for_compile_failure(build_result: BuildResult) -> dict[str, Any]:
    return {
        "compiled": False,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "runtime_error": f"nvcc returned {build_result.returncode}",
            "build_stdout": _tail(build_result.stdout),
            "build_stderr": _tail(build_result.stderr),
            "build_command": build_result.command,
        },
        "raw_repr": f"nvcc returned {build_result.returncode}",
    }


def _payload_for_execution_failure(
    *,
    completed: subprocess.CompletedProcess,
    build_result: BuildResult,
) -> dict[str, Any]:
    return {
        "compiled": True,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "runtime_error": (
                f"candidate binary exited with returncode={completed.returncode}; "
                f"see workspace samples/ for full stderr"
            ),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "build_command": build_result.command,
        },
        "raw_repr": f"binary returncode={completed.returncode}",
    }


def _payload_for_parse_failure(
    *,
    completed: subprocess.CompletedProcess,
    build_result: BuildResult,
) -> dict[str, Any]:
    return {
        "compiled": True,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "runtime_error": (
                "the candidate binary ran to completion but did not emit a RESULT line."
                " The locked driver may have been altered, or the kernel may have produced"
                " no output."
            ),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "build_command": build_result.command,
        },
        "raw_repr": "no RESULT line",
    }


def evaluate_candidate(
    *,
    candidate_path: Path,
    problem_dir: Path,
    build_dir: Path,
    gpu_id: int,
    run_timeout_seconds: float,
    build_timeout_seconds: float,
) -> dict[str, Any]:
    problem_metadata = load_problem_metadata(problem_dir)
    tolerance = float(problem_metadata.get("tolerance") or 1e-2)

    build_result = build_candidate_binary(
        candidate_path=candidate_path,
        build_dir=build_dir,
        problem_metadata=problem_metadata,
        timeout_seconds=build_timeout_seconds,
    )
    if build_result.returncode != 0:
        return _payload_for_compile_failure(build_result)

    # The parent command sets CUDA_VISIBLE_DEVICES via the GPU lease before
    # spawning this subprocess. Do not overwrite it here — the binary will pick
    # up whatever the lease isolated for us as ``cuda:0``.
    env = os.environ.copy()
    completed = subprocess.run(
        [str(build_result.binary_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(build_dir),
        env=env,
        timeout=run_timeout_seconds,
    )
    if completed.returncode != 0:
        return _payload_for_execution_failure(
            completed=completed,
            build_result=build_result,
        )

    parsed = parse_result_line(completed.stdout or "")
    if parsed is None:
        return _payload_for_parse_failure(
            completed=completed,
            build_result=build_result,
        )

    correctness = parsed.mean_abs_error < tolerance
    runtime_stats = {
        "mean": parsed.candidate_ms,
        "mean_runtime_ms": parsed.candidate_ms,
        "gflops": parsed.candidate_gflops,
    }
    ref_runtime_stats = {
        "mean": parsed.cublas_ms,
        "mean_runtime_ms": parsed.cublas_ms,
        "gflops": parsed.cublas_gflops,
    }
    metadata = {
        "tolerance": tolerance,
        "mean_abs_error": parsed.mean_abs_error,
        "num_flops": parsed.num_flops,
        "cublas_gflops": parsed.cublas_gflops,
        "candidate_gflops": parsed.candidate_gflops,
        "extras": parsed.extras,
        "build_command": build_result.command,
        "build_stdout_tail": build_result.stdout[-500:] if build_result.stdout else "",
        "stdout_tail": (completed.stdout or "")[-1000:],
    }
    if not correctness:
        metadata["runtime_error"] = (
            f"mean absolute error {parsed.mean_abs_error:.6g} exceeds tolerance {tolerance:.6g}"
        )
    return {
        "compiled": True,
        "correctness": correctness,
        "runtime": parsed.candidate_ms,
        "runtime_stats": runtime_stats,
        "ref_runtime": parsed.cublas_ms,
        "ref_runtime_stats": ref_runtime_stats,
        "metadata": metadata,
        "raw_repr": f"candidate_ms={parsed.candidate_ms} cublas_ms={parsed.cublas_ms} error={parsed.mean_abs_error}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and time one CUDA candidate.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--problem-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--build-timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    problem_dir = Path(args.problem_dir).expanduser().resolve()
    output_path = Path(args.output_path)
    build_dir = output_path.parent
    payload = evaluate_candidate(
        candidate_path=candidate_path,
        problem_dir=problem_dir,
        build_dir=build_dir,
        gpu_id=args.gpu_id,
        run_timeout_seconds=args.run_timeout_seconds,
        build_timeout_seconds=args.build_timeout_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
