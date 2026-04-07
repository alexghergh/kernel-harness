from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .archive_layout import history_path, sample_manifest_path
from .candidate_contract import CANDIDATE_FILENAME
from .candidate_validation import CandidateValidationError, validate_candidate_source
from .common import emit_json
from .goal_status import write_goal_status_files
from .gpu_pool import isolated_gpu_environment, lease_gpu_slot, lease_problem_artifacts
from .project import (
    append_jsonl,
    build_problem_dir,
    next_sample_id,
    now_iso,
    official_kernel_path,
    official_prompt_path,
    write_json,
    write_text,
)
from .subprocess_tools import excerpt, load_json_object, run_subprocess_capture, serialize_exception
from .workspace_paths import (
    workspace_candidate_path,
    workspace_path,
    write_workspace_sample_copy,
)


def command_run_candidate(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).resolve()
    workspace = workspace_path(args.workspace) if args.workspace else None
    lease_name = f"artifacts:{args.run_name}:level_{args.level}:problem_{args.problem_id}"
    sample_id: int | None = None
    prompt_path: Path | None = None
    payload: dict[str, Any] | None = None
    sample_json_path: Path | None = None
    history_path_value = history_path(args.run_name, args.level, args.problem_id)
    failure: Exception | None = None
    persist_failure: Exception | None = None

    try:
        with lease_problem_artifacts(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            lease_name=lease_name,
        ) as artifact_lease:
            sample_id = next_sample_id(args.run_name, args.level, args.problem_id)
            kernel_path = official_kernel_path(
                args.run_name,
                args.level,
                args.problem_id,
                sample_id,
            )
            sample_json_path = sample_manifest_path(
                args.run_name,
                args.level,
                args.problem_id,
                sample_id,
            )
            stdout_path = sample_json_path.with_suffix(".stdout.txt")
            stderr_path = sample_json_path.with_suffix(".stderr.txt")
            if args.prompt_path:
                prompt_path = official_prompt_path(
                    args.run_name,
                    args.level,
                    args.problem_id,
                    sample_id,
                )

            payload = {
                "status": "started",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "run_name": args.run_name,
                "level": args.level,
                "problem_id": args.problem_id,
                "sample_id": sample_id,
                "candidate_path": str(candidate_path),
                "official_kernel_path": str(kernel_path),
                "official_prompt_path": str(prompt_path) if prompt_path else None,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "backend": args.backend,
                "precision": args.precision,
                "artifact_reservation_wait_seconds": artifact_lease.wait_seconds,
                "artifact_commit_wait_seconds": None,
                "gpu_id": None,
                "gpu_device_selector": None,
                "gpu_visible_devices": None,
                "gpu_logical_id": None,
                "gpu_selector_source": None,
                "gpu_wait_seconds": None,
                "result": {},
                "error": None,
            }

            if workspace is not None:
                expected_candidate_path = workspace_candidate_path(workspace)
                if candidate_path != expected_candidate_path:
                    raise CandidateValidationError(
                        f"Only {CANDIDATE_FILENAME} may be evaluated from the problem workspace."
                    )

            candidate_src = candidate_path.read_text(encoding="utf-8")
            validate_candidate_source(candidate_src)
            write_text(kernel_path, candidate_src)
            if workspace is not None:
                write_workspace_sample_copy(workspace, sample_id, candidate_src)
            if prompt_path is not None:
                write_text(prompt_path, Path(args.prompt_path).read_text(encoding="utf-8"))
            write_json(sample_json_path, payload)

        with lease_gpu_slot(
            num_slots=args.num_gpu_slots,
            requested_slot=args.gpu_id,
            lease_name=f"run:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
        ) as lease:
            runner_output_path = build_problem_dir(
                args.run_name,
                args.level,
                args.problem_id,
                f"sample_{sample_id}",
            ) / "evaluation_result.json"
            command = [
                sys.executable,
                "-m",
                "kernel_bench_experiment_agents.evaluation_runner",
                "--candidate",
                str(candidate_path),
                "--output-path",
                str(runner_output_path),
                "--level",
                str(args.level),
                "--problem-id",
                str(args.problem_id),
                "--dataset-src",
                args.dataset_src,
                "--gpu-id",
                str(lease.logical_gpu_id),
                "--run-name",
                args.run_name,
                "--sample-id",
                str(sample_id),
                "--backend",
                args.backend,
                "--precision",
                args.precision,
                "--num-correct-trials",
                str(args.num_correct_trials),
                "--num-perf-trials",
                str(args.num_perf_trials),
            ]
            if args.kernelbench_root:
                command.extend(["--kernelbench-root", args.kernelbench_root])
            if args.timing_method is not None:
                command.extend(["--timing-method", args.timing_method])

            completed = run_subprocess_capture(
                command,
                env=isolated_gpu_environment(device_selector=lease.device_selector),
            )
            write_text(stdout_path, completed.stdout)
            write_text(stderr_path, completed.stderr)
            payload["gpu_id"] = lease.slot_id
            payload["gpu_device_selector"] = lease.device_selector
            payload["gpu_visible_devices"] = lease.isolated_visible_devices
            payload["gpu_logical_id"] = lease.logical_gpu_id
            payload["gpu_selector_source"] = lease.selector_source
            payload["gpu_wait_seconds"] = lease.wait_seconds

        if completed.returncode != 0:
            raise RuntimeError(
                "Candidate evaluation subprocess failed "
                f"(return code {completed.returncode}); see {stderr_path}.\n"
                f"stderr excerpt:\n{excerpt(completed.stderr or completed.stdout)}"
            )
        if not runner_output_path.exists():
            raise RuntimeError(
                f"Candidate evaluation subprocess produced no result payload at {runner_output_path}."
            )
        result = load_json_object(runner_output_path)
        payload["status"] = "succeeded"
        payload["updated_at"] = now_iso()
        payload["result"] = result
    except Exception as exc:
        failure = exc
        if payload is None or sample_id is None or sample_json_path is None:
            raise
        payload["status"] = "failed"
        payload["updated_at"] = now_iso()
        payload["error"] = serialize_exception(exc)
    finally:
        if payload is not None and sample_json_path is not None:
            try:
                with lease_problem_artifacts(
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    lease_name=lease_name,
                ) as artifact_lease:
                    payload["artifact_commit_wait_seconds"] = artifact_lease.wait_seconds
                    payload["updated_at"] = now_iso()
                    write_json(sample_json_path, payload)
                    append_jsonl(history_path_value, payload)
                    if workspace is not None:
                        write_goal_status_files(
                            run_name=args.run_name,
                            level=args.level,
                            problem_id=args.problem_id,
                            workspace=workspace,
                        )
            except Exception as exc:
                persist_failure = exc

    emit_json(payload)
    if failure is not None:
        if persist_failure is not None:
            print(
                f"warning: artifact persistence also failed for sample {sample_id}: {persist_failure}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"Candidate evaluation failed for sample {sample_id}: {failure}"
        ) from failure
    if persist_failure is not None:
        raise SystemExit(
            f"Artifact persistence failed for sample {sample_id}: {persist_failure}"
        ) from persist_failure
