from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME
from .candidate_validation import CandidateValidationError, validate_candidate_source
from .common import emit_json
from .gpu_pool import lease_gpu_slot, lease_problem_artifacts
from .kernelbench import evaluate_candidate
from .project import (
    append_jsonl,
    next_sample_id,
    now_iso,
    official_kernel_path,
    official_prompt_path,
    write_json,
    write_text,
)
from .workspace_state import (
    archive_problem_profiles_dir,
    history_path,
    latest_workspace_profile_paths,
    next_archive_profile_index,
    profile_index_path,
    sample_manifest_path,
    serialize_exception,
    workspace_candidate_path,
    workspace_path,
    workspace_profiles_dir,
    workspace_relpath,
    write_goal_status_files,
    write_workspace_sample_copy,
)
from .common import as_float


def summarize_ncu_raw_csv(raw_csv_text: str) -> str:
    rows = list(csv.DictReader(io.StringIO(raw_csv_text)))
    if not rows:
        return (
            "NCU summary could not be generated because the raw CSV had no data rows.\n"
            "Read profiles/latest.details.txt for the full text report.\n"
        )

    def score(row: dict[str, str]) -> int:
        return sum(
            1 for value in row.values() if isinstance(value, str) and any(ch.isdigit() for ch in value)
        )

    row = max(rows, key=score)

    def first_value(*keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    lines = [
        "# NCU Summary",
        "",
        "Prefer this file first. Read `profiles/latest.details.txt` only when you need the full report.",
        "",
    ]

    metric_groups = (
        (
            "Key performance metrics",
            (
                ("duration", "gpu__time_duration.sum"),
                ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                (
                    "compute+memory throughput",
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                ),
                ("registers per thread", "launch__registers_per_thread"),
                ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
            ),
        ),
        (
            "Memory and shared-memory indicators",
            (
                ("L1/TEX throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
                ("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_active"),
                ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                ("shared-memory conflict n-way", "derived__memory_l1_conflicts_shared_nway"),
                (
                    "shared-memory excessive wavefronts",
                    "derived__memory_l1_wavefronts_shared_excessive",
                ),
            ),
        ),
        (
            "Occupancy limiters",
            (
                ("block limit by registers", "launch__occupancy_limit_registers"),
                ("block limit by shared memory", "launch__occupancy_limit_shared_mem"),
                ("block limit by warps", "launch__occupancy_limit_warps"),
            ),
        ),
    )

    for title, metrics in metric_groups:
        lines.append(f"## {title}")
        wrote_any = False
        for label, key in metrics:
            value = first_value(key)
            if value is None:
                continue
            lines.append(f"- {label}: {value}")
            wrote_any = True
        if not wrote_any:
            lines.append("- no values found in the exported raw CSV")
        lines.append("")

    stall_entries: list[tuple[str, float, str]] = []
    for key, value in row.items():
        if "smsp__average_warps_issue_stalled_" not in key:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        numeric = as_float(value)
        if numeric is None or numeric <= 0:
            continue
        stall_name = key.split("stalled_", 1)[1].split("_per_", 1)[0]
        stall_entries.append((stall_name, numeric, value.strip()))

    lines.append("## Top warp stalls")
    if stall_entries:
        for stall_name, _, raw_value in sorted(stall_entries, key=lambda item: -item[1])[:8]:
            lines.append(f"- {stall_name}: {raw_value}")
    else:
        lines.append("- no positive warp-stall metrics were found in the exported raw CSV")
    lines.append("")
    lines.append("## Next step")
    lines.append(
        "- Re-read `HARDWARE.md`, then use this summary plus `profiles/latest.details.txt` to pick the next branch."
    )
    lines.append("")
    return "\n".join(lines)


def run_subprocess_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
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
    status_refresh_failure: Exception | None = None

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
            sample_json_path = sample_manifest_path(args.run_name, args.level, args.problem_id, sample_id)
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
                "backend": args.backend,
                "precision": args.precision,
                "artifact_reservation_wait_seconds": artifact_lease.wait_seconds,
                "artifact_commit_wait_seconds": None,
                "gpu_id": None,
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
            payload["gpu_id"] = lease.slot_id
            payload["gpu_wait_seconds"] = lease.wait_seconds
            result = evaluate_candidate(
                candidate_src=candidate_src,
                level=args.level,
                problem_id=args.problem_id,
                dataset_src=args.dataset_src,
                run_name=args.run_name,
                sample_id=sample_id,
                gpu_id=lease.slot_id,
                timing_method=args.timing_method,
                backend=args.backend,
                precision=args.precision,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                explicit_kernelbench_root=args.kernelbench_root,
            )

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
            except Exception as exc:
                persist_failure = exc
        if workspace is not None:
            try:
                write_goal_status_files(
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    workspace=workspace,
                )
            except Exception as exc:
                status_refresh_failure = exc

    emit_json(payload)
    if failure is not None:
        if persist_failure is not None:
            print(
                f"warning: artifact persistence also failed for sample {sample_id}: {persist_failure}",
                file=sys.stderr,
            )
        if status_refresh_failure is not None:
            print(
                f"warning: goal-status refresh also failed for sample {sample_id}: {status_refresh_failure}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"Candidate evaluation failed for sample {sample_id}: {failure}"
        ) from failure
    if persist_failure is not None:
        raise SystemExit(
            f"Artifact persistence failed for sample {sample_id}: {persist_failure}"
        ) from persist_failure
    if status_refresh_failure is not None:
        print(
            f"warning: failed to refresh goal status after sample {sample_id}: {status_refresh_failure}",
            file=sys.stderr,
        )


def command_profile_ncu(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).resolve()
    workspace: Path | None = None
    if args.workspace:
        workspace = workspace_path(args.workspace)
        expected_candidate_path = workspace_candidate_path(workspace)
        if candidate_path != expected_candidate_path:
            raise SystemExit(
                f"Only {CANDIDATE_FILENAME} may be profiled from the problem workspace."
            )
    candidate_src = candidate_path.read_text(encoding="utf-8")
    validate_candidate_source(candidate_src)

    lease_name = f"profile:{args.run_name}:level_{args.level}:problem_{args.problem_id}"
    profiles_dir = archive_problem_profiles_dir(args.run_name, args.level, args.problem_id)
    reservation_wait_seconds = 0.0
    status_refresh_failure: Exception | None = None

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
            sample_label = f"profile_{next_archive_profile_index(args.run_name, args.level, args.problem_id)}"
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
        write_json(
            profile_json_path,
            {
                "status": "started",
                "timestamp": now_iso(),
                "run_name": args.run_name,
                "level": args.level,
                "problem_id": args.problem_id,
                "sample_label": sample_label,
                "candidate_path": str(candidate_path),
                "artifact_reservation_wait_seconds": reservation_wait_seconds,
            },
        )

    with lease_gpu_slot(
        num_slots=args.num_gpu_slots,
        requested_slot=args.gpu_id,
        lease_name=lease_name,
    ) as lease:
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
            str(lease.slot_id),
            "--run-name",
            args.run_name,
            "--sample-label",
            sample_label,
        ]
        if args.kernelbench_root:
            command.extend(["--kernelbench-root", args.kernelbench_root])
        completed = run_subprocess_capture(command)
        gpu_id = lease.slot_id
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
        "candidate_path": str(candidate_path),
        "report_path": str(report_path) if keep_report and report_path.exists() else None,
        "details_path": str(details_path),
        "details_stderr_path": str(details_stderr_path),
        "raw_csv_path": str(raw_csv_path),
        "raw_csv_stderr_path": str(raw_csv_stderr_path),
        "summary_path": str(summary_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": completed.returncode,
        "command": command,
        "details_command": details_command,
        "details_returncode": details_completed.returncode,
        "raw_csv_command": raw_csv_command,
        "raw_csv_returncode": raw_csv_completed.returncode,
        "gpu_id": gpu_id,
        "gpu_wait_seconds": gpu_wait_seconds,
        "artifact_reservation_wait_seconds": reservation_wait_seconds,
        "artifact_commit_wait_seconds": None,
    }

    emit_payload = dict(payload)
    if workspace is not None:
        profiles_workspace_dir = workspace_profiles_dir(workspace)
        profile_base = profiles_workspace_dir / sample_label
        local_paths = {
            "details_path": profile_base.with_suffix(".details.txt"),
            "details_stderr_path": profile_base.with_suffix(".details.stderr.txt"),
            "raw_csv_path": profile_base.with_suffix(".raw.csv"),
            "raw_csv_stderr_path": profile_base.with_suffix(".raw.stderr.txt"),
            "summary_path": profile_base.with_suffix(".summary.txt"),
            "stdout_path": profile_base.with_suffix(".stdout.txt"),
            "stderr_path": profile_base.with_suffix(".stderr.txt"),
            "json_path": profile_base.with_suffix(".json"),
        }
        latest_paths = latest_workspace_profile_paths(workspace)

        write_text(local_paths["details_path"], details_completed.stdout)
        write_text(local_paths["details_stderr_path"], details_completed.stderr)
        write_text(local_paths["raw_csv_path"], raw_csv_completed.stdout)
        write_text(local_paths["raw_csv_stderr_path"], raw_csv_completed.stderr)
        write_text(local_paths["summary_path"], summary_text)
        write_text(local_paths["stdout_path"], completed.stdout)
        write_text(local_paths["stderr_path"], completed.stderr)
        write_json(local_paths["json_path"], payload)
        write_text(latest_paths["details"], details_completed.stdout)
        write_text(latest_paths["details_stderr"], details_completed.stderr)
        write_text(latest_paths["raw_csv"], raw_csv_completed.stdout)
        write_text(latest_paths["raw_csv_stderr"], raw_csv_completed.stderr)
        write_text(latest_paths["summary"], summary_text)
        write_text(latest_paths["stdout"], completed.stdout)
        write_text(latest_paths["stderr"], completed.stderr)
        write_json(latest_paths["json"], payload)
        emit_payload = {
            "timestamp": payload["timestamp"],
            "run_name": args.run_name,
            "level": args.level,
            "problem_id": args.problem_id,
            "sample_label": sample_label,
            "candidate_path": workspace_relpath(candidate_path, workspace),
            "details_path": workspace_relpath(latest_paths["details"], workspace),
            "raw_csv_path": workspace_relpath(latest_paths["raw_csv"], workspace),
            "summary_path": workspace_relpath(latest_paths["summary"], workspace),
            "stdout_path": workspace_relpath(latest_paths["stdout"], workspace),
            "stderr_path": workspace_relpath(latest_paths["stderr"], workspace),
            "profile_details_path": workspace_relpath(local_paths["details_path"], workspace),
            "profile_raw_csv_path": workspace_relpath(local_paths["raw_csv_path"], workspace),
            "profile_summary_path": workspace_relpath(local_paths["summary_path"], workspace),
            "profile_stdout_path": workspace_relpath(local_paths["stdout_path"], workspace),
            "profile_stderr_path": workspace_relpath(local_paths["stderr_path"], workspace),
            "returncode": completed.returncode,
            "details_returncode": details_completed.returncode,
            "raw_csv_returncode": raw_csv_completed.returncode,
            "gpu_id": gpu_id,
            "gpu_wait_seconds": gpu_wait_seconds,
        }

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
    except Exception as exc:
        raise SystemExit(f"Failed to persist profiling metadata for {sample_label}: {exc}") from exc

    if workspace is not None:
        try:
            write_goal_status_files(
                run_name=args.run_name,
                level=args.level,
                problem_id=args.problem_id,
                workspace=workspace,
            )
        except Exception as exc:
            status_refresh_failure = exc

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
    if status_refresh_failure is not None:
        print(
            f"warning: failed to refresh goal status after profiling {sample_label}: {status_refresh_failure}",
            file=sys.stderr,
        )
    emit_json(emit_payload)
