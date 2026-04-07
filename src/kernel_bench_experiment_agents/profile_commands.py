from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .archive_layout import archive_problem_profiles_dir, next_archive_profile_index, profile_index_path
from .candidate_contract import CANDIDATE_FILENAME
from .candidate_validation import validate_candidate_source
from .common import emit_json
from .goal_status import write_goal_status_files
from .gpu_pool import isolated_gpu_environment, lease_gpu_slot, lease_problem_artifacts
from .ncu_summary import summarize_ncu_raw_csv
from .project import append_jsonl, artifact_problem_dir, now_iso, relative_path_within, write_json, write_text
from .subprocess_tools import run_subprocess_capture
from .workspace_paths import (
    latest_workspace_profile_paths,
    validate_workspace_assignment,
    workspace_candidate_path,
    workspace_path,
    workspace_profiles_dir,
    workspace_relpath,
)


def _workspace_profile_local_paths(workspace: Path, sample_label: str) -> dict[str, Path]:
    profile_base = workspace_profiles_dir(workspace) / sample_label
    return {
        "details_path": profile_base.with_suffix(".details.txt"),
        "details_stderr_path": profile_base.with_suffix(".details.stderr.txt"),
        "raw_csv_path": profile_base.with_suffix(".raw.csv"),
        "raw_csv_stderr_path": profile_base.with_suffix(".raw.stderr.txt"),
        "summary_path": profile_base.with_suffix(".summary.txt"),
        "stdout_path": profile_base.with_suffix(".stdout.txt"),
        "stderr_path": profile_base.with_suffix(".stderr.txt"),
        "json_path": profile_base.with_suffix(".json"),
    }


def _write_workspace_profile_mirrors(
    *,
    workspace: Path,
    sample_label: str,
    payload: dict[str, Any],
    completed_stdout: str,
    completed_stderr: str,
    details_stdout: str,
    details_stderr: str,
    raw_csv_stdout: str,
    raw_csv_stderr: str,
    summary_text: str,
) -> dict[str, Path]:
    local_paths = _workspace_profile_local_paths(workspace, sample_label)
    latest_paths = latest_workspace_profile_paths(workspace)

    write_text(local_paths["details_path"], details_stdout)
    write_text(local_paths["details_stderr_path"], details_stderr)
    write_text(local_paths["raw_csv_path"], raw_csv_stdout)
    write_text(local_paths["raw_csv_stderr_path"], raw_csv_stderr)
    write_text(local_paths["summary_path"], summary_text)
    write_text(local_paths["stdout_path"], completed_stdout)
    write_text(local_paths["stderr_path"], completed_stderr)
    write_json(local_paths["json_path"], payload)

    write_text(latest_paths["details"], details_stdout)
    write_text(latest_paths["details_stderr"], details_stderr)
    write_text(latest_paths["raw_csv"], raw_csv_stdout)
    write_text(latest_paths["raw_csv_stderr"], raw_csv_stderr)
    write_text(latest_paths["summary"], summary_text)
    write_text(latest_paths["stdout"], completed_stdout)
    write_text(latest_paths["stderr"], completed_stderr)
    write_json(latest_paths["json"], payload)

    return {**local_paths, **{f"latest_{key}": value for key, value in latest_paths.items()}}


def _workspace_candidate_reference(candidate_path: Path, workspace: Path | None) -> str:
    if workspace is not None:
        return workspace_relpath(candidate_path, workspace)
    return candidate_path.name


def command_profile_ncu(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).resolve()
    workspace: Path | None = None
    if args.workspace:
        workspace = workspace_path(args.workspace)
        validate_workspace_assignment(
            workspace,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        )
        expected_candidate_path = workspace_candidate_path(workspace)
        if candidate_path != expected_candidate_path:
            raise SystemExit(
                f"Only {CANDIDATE_FILENAME} may be profiled from the problem workspace."
            )
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)

    lease_name = f"profile:{args.run_name}:level_{args.level}:problem_{args.problem_id}"
    problem_archive_root = artifact_problem_dir(args.run_name, args.level, args.problem_id)
    profiles_dir = archive_problem_profiles_dir(args.run_name, args.level, args.problem_id)

    with lease_problem_artifacts(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        lease_name=f"{lease_name}:reserve",
    ) as artifact_lease:
        reservation_wait_seconds = artifact_lease.wait_seconds
        if args.sample_id is not None:
            sample_label = f"sample_{args.sample_id}"
        else:
            sample_label = (
                f"profile_{next_archive_profile_index(args.run_name, args.level, args.problem_id)}"
            )
        report_prefix = profiles_dir / sample_label
        report_path = Path(str(report_prefix) + ".ncu-rep")
        stdout_path = report_prefix.with_suffix(".stdout.txt")
        stderr_path = report_prefix.with_suffix(".stderr.txt")
        details_path = report_prefix.with_suffix(".details.txt")
        details_stderr_path = report_prefix.with_suffix(".details.stderr.txt")
        raw_csv_path = report_prefix.with_suffix(".raw.csv")
        raw_csv_stderr_path = report_prefix.with_suffix(".raw.stderr.txt")
        summary_path = report_prefix.with_suffix(".summary.txt")
        profile_json_path = report_prefix.with_suffix(".json")
        candidate_ref = _workspace_candidate_reference(candidate_path, workspace)
        archive_report_prefix = relative_path_within(report_prefix, problem_archive_root)
        write_json(
            profile_json_path,
            {
                "status": "started",
                "timestamp": now_iso(),
                "run_name": args.run_name,
                "level": args.level,
                "problem_id": args.problem_id,
                "sample_label": sample_label,
                "candidate_path": candidate_ref,
                "artifact_reservation_wait_seconds": reservation_wait_seconds,
            },
        )

    with lease_gpu_slot(
        num_slots=args.num_gpu_slots,
        requested_slot=args.gpu_id,
        lease_name=lease_name,
    ) as lease:
        isolated_env = isolated_gpu_environment(device_selector=lease.device_selector)
        command = [
            "ncu",
            "--set",
            args.ncu_set,
            "--force-overwrite",
            "--target-processes",
            "all",
            "--export",
            str(report_prefix),
            sys.executable,
            "-m",
            "kernel_bench_experiment_agents.ncu_runner",
            "--candidate",
            str(candidate_path),
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
            "--sample-label",
            sample_label,
        ]
        if args.kernelbench_root:
            command.extend(["--kernelbench-root", args.kernelbench_root])
        completed = run_subprocess_capture(command, env=isolated_env)
        gpu_id = lease.slot_id
        gpu_device_selector = lease.device_selector
        gpu_visible_devices = lease.isolated_visible_devices
        gpu_logical_id = lease.logical_gpu_id
        gpu_selector_source = lease.selector_source
        gpu_wait_seconds = lease.wait_seconds

    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)

    details_command = [
        "ncu",
        "--import",
        str(report_path),
        "--page",
        "details",
    ]
    details_completed = run_subprocess_capture(details_command)
    write_text(details_path, details_completed.stdout)
    write_text(details_stderr_path, details_completed.stderr)

    raw_csv_command = [
        "ncu",
        "--import",
        str(report_path),
        "--page",
        "raw",
        "--csv",
    ]
    raw_csv_completed = run_subprocess_capture(raw_csv_command)
    write_text(raw_csv_path, raw_csv_completed.stdout)
    write_text(raw_csv_stderr_path, raw_csv_completed.stderr)
    summary_text = summarize_ncu_raw_csv(raw_csv_completed.stdout)
    write_text(summary_path, summary_text)

    keep_report = os.environ.get("KBE_KEEP_NCU_REP", "").strip().lower() in {"1", "true", "yes", "on"}
    if not keep_report and report_path.exists():
        report_path.unlink()

    profile_ok = (
        completed.returncode == 0
        and details_completed.returncode == 0
        and raw_csv_completed.returncode == 0
        and bool(details_completed.stdout.strip())
        and bool(raw_csv_completed.stdout.strip())
    )
    payload = {
        "status": "succeeded" if profile_ok else "failed",
        "timestamp": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "sample_label": sample_label,
        "candidate_path": candidate_ref,
        "report_path": f"{archive_report_prefix}.ncu-rep" if keep_report and report_path.exists() else None,
        "details_path": relative_path_within(details_path, problem_archive_root),
        "details_stderr_path": relative_path_within(details_stderr_path, problem_archive_root),
        "raw_csv_path": relative_path_within(raw_csv_path, problem_archive_root),
        "raw_csv_stderr_path": relative_path_within(raw_csv_stderr_path, problem_archive_root),
        "summary_path": relative_path_within(summary_path, problem_archive_root),
        "stdout_path": relative_path_within(stdout_path, problem_archive_root),
        "stderr_path": relative_path_within(stderr_path, problem_archive_root),
        "returncode": completed.returncode,
        "command": [
            "ncu",
            "--set",
            args.ncu_set,
            "--force-overwrite",
            "--target-processes",
            "all",
            "--export",
            archive_report_prefix,
            "python",
            "-m",
            "kernel_bench_experiment_agents.ncu_runner",
            "--candidate",
            candidate_ref,
            "--level",
            str(args.level),
            "--problem-id",
            str(args.problem_id),
            "--dataset-src",
            args.dataset_src,
            "--gpu-id",
            str(gpu_logical_id),
            "--run-name",
            args.run_name,
            "--sample-label",
            sample_label,
        ],
        "details_command": ["ncu", "--import", f"{archive_report_prefix}.ncu-rep", "--page", "details"],
        "details_returncode": details_completed.returncode,
        "raw_csv_command": [
            "ncu",
            "--import",
            f"{archive_report_prefix}.ncu-rep",
            "--page",
            "raw",
            "--csv",
        ],
        "raw_csv_returncode": raw_csv_completed.returncode,
        "gpu_id": gpu_id,
        "gpu_device_selector": gpu_device_selector,
        "gpu_visible_devices": gpu_visible_devices,
        "gpu_logical_id": gpu_logical_id,
        "gpu_selector_source": gpu_selector_source,
        "gpu_wait_seconds": gpu_wait_seconds,
        "artifact_reservation_wait_seconds": reservation_wait_seconds,
        "artifact_commit_wait_seconds": None,
    }

    emit_payload = dict(payload)
    try:
        with lease_problem_artifacts(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            lease_name=f"{lease_name}:commit",
        ) as artifact_lease:
            payload["artifact_commit_wait_seconds"] = artifact_lease.wait_seconds
            write_json(profile_json_path, payload)
            append_jsonl(profile_index_path(args.run_name, args.level, args.problem_id), payload)

            if workspace is not None:
                mirror_paths = _write_workspace_profile_mirrors(
                    workspace=workspace,
                    sample_label=sample_label,
                    payload=payload,
                    completed_stdout=completed.stdout,
                    completed_stderr=completed.stderr,
                    details_stdout=details_completed.stdout,
                    details_stderr=details_completed.stderr,
                    raw_csv_stdout=raw_csv_completed.stdout,
                    raw_csv_stderr=raw_csv_completed.stderr,
                    summary_text=summary_text,
                )
                write_goal_status_files(
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    workspace=workspace,
                )
                emit_payload = {
                    "timestamp": payload["timestamp"],
                    "run_name": args.run_name,
                    "level": args.level,
                    "problem_id": args.problem_id,
                    "sample_label": sample_label,
                    "candidate_path": workspace_relpath(candidate_path, workspace),
                    "details_path": workspace_relpath(mirror_paths["latest_details"], workspace),
                    "raw_csv_path": workspace_relpath(mirror_paths["latest_raw_csv"], workspace),
                    "summary_path": workspace_relpath(mirror_paths["latest_summary"], workspace),
                    "stdout_path": workspace_relpath(mirror_paths["latest_stdout"], workspace),
                    "stderr_path": workspace_relpath(mirror_paths["latest_stderr"], workspace),
                    "profile_details_path": workspace_relpath(mirror_paths["details_path"], workspace),
                    "profile_raw_csv_path": workspace_relpath(mirror_paths["raw_csv_path"], workspace),
                    "profile_summary_path": workspace_relpath(mirror_paths["summary_path"], workspace),
                    "profile_stdout_path": workspace_relpath(mirror_paths["stdout_path"], workspace),
                    "profile_stderr_path": workspace_relpath(mirror_paths["stderr_path"], workspace),
                    "returncode": completed.returncode,
                    "details_returncode": details_completed.returncode,
                    "raw_csv_returncode": raw_csv_completed.returncode,
                    "gpu_id": gpu_id,
                    "gpu_wait_seconds": gpu_wait_seconds,
                }
    except Exception as exc:
        raise SystemExit(f"Failed to persist profiling metadata for {sample_label}: {exc}") from exc

    if completed.returncode != 0:
        raise SystemExit(
            "ncu profiling failed "
            f"(return code {completed.returncode}); see {stderr_path}"
        )
    if details_completed.returncode != 0:
        raise SystemExit(
            "ncu text summary export failed "
            f"(return code {details_completed.returncode}); see {details_stderr_path}"
        )
    if raw_csv_completed.returncode != 0:
        raise SystemExit(
            "ncu raw csv export failed "
            f"(return code {raw_csv_completed.returncode}); see {raw_csv_stderr_path}"
        )
    if not details_completed.stdout.strip():
        raise SystemExit(
            f"ncu details export produced no readable output; see {details_path}"
        )
    if not raw_csv_completed.stdout.strip():
        raise SystemExit(
            f"ncu raw csv export produced no readable output; see {raw_csv_path}"
        )
    emit_json(emit_payload)
