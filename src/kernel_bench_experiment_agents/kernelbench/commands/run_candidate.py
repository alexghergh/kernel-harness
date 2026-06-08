"""Implement the measured candidate-evaluation command used by the workspace run wrapper.

This module validates the candidate, records archived attempt metadata, leases a GPU slot, and updates goal status after each run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.workspace.archive import sample_manifest_path
from kernel_bench_experiment_agents.kernelbench.metrics import result_runtime_error
from kernel_bench_experiment_agents.problem_source import (
    CandidateValidationError,
    DEFAULT_PROBLEM_SOURCE,
    EvalContext,
    get_problem_source,
)
from kernel_bench_experiment_agents.runtime.common import as_float, emit_json
from kernel_bench_experiment_agents.agent_contract.goal_status import write_goal_status_files
from kernel_bench_experiment_agents.runtime.live_gpu_wait import (
    clear_live_gpu_wait_marker,
    create_live_gpu_wait_marker,
    mark_live_gpu_wait_operation_started,
    settle_live_gpu_wait_marker,
)
from kernel_bench_experiment_agents.runtime.gpu_pool import (
    isolated_gpu_environment,
    lease_gpu_slot,
    lease_problem_artifacts,
    quarantine_gpu_slot,
)
from kernel_bench_experiment_agents.runtime.project import (
    archive_problem_dir,
    build_problem_dir,
    next_sample_id,
    now_iso,
    official_kernel_path,
    relative_path_within,
    write_json,
)
from kernel_bench_experiment_agents.runtime.subprocess_tools import (
    SubprocessStart,
    SubprocessTimeoutError,
    excerpt,
    load_json_object,
    run_subprocess_streaming,
    serialize_exception,
    subprocess_cleanup_incomplete,
    subprocess_result_metadata,
    subprocess_start_metadata,
    timeout_seconds_from_env,
)
from kernel_bench_experiment_agents.workspace.paths import (
    load_workspace_metadata,
    validate_workspace_assignment,
    workspace_candidate_path,
    workspace_path,
    workspace_relpath,
    write_workspace_sample_copy,
)


def _workspace_candidate_reference(candidate_path: Path, workspace: Path | None) -> str:
    if workspace is not None:
        return workspace_relpath(candidate_path, workspace)
    return candidate_path.name


def _result_warnings(
    result: dict[str, Any],
    workspace: Path | None,
    *,
    stdout_text: str = "",
) -> list[str]:
    warnings: list[str] = []
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if metadata.get("excessive_speedup"):
        warnings.append(
            "KernelBench flagged this run as suspicious because the measured speedup is excessively large. This run does not count toward progress. Discard it as possible reward hacking and continue iterating until you have a non-suspicious result."
        )
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[WARNING]"):
            warnings.append(stripped)
    if workspace is None:
        return warnings
    metadata = load_workspace_metadata(workspace)
    baseline_runtime_ms = metadata.get("baseline_runtime_ms") if isinstance(metadata, dict) else None
    baseline_runtime_ms = baseline_runtime_ms if isinstance(baseline_runtime_ms, dict) else {}
    eager_baseline = as_float(baseline_runtime_ms.get("eager"))
    ref_runtime = as_float(result.get("ref_runtime"))
    if eager_baseline is None or ref_runtime is None or eager_baseline <= 0:
        return warnings
    relative_delta = abs(ref_runtime - eager_baseline) / eager_baseline
    if relative_delta > 0.15:
        warnings.append(
            f"KernelBench reported ref_runtime={ref_runtime} ms but the archived eager baseline is {eager_baseline} ms; relative delta {relative_delta:.1%}. Review this problem manually before trusting the baseline comparison."
        )
    return warnings



def command_run_candidate(args: argparse.Namespace) -> None:
    """Evaluate one frozen candidate snapshot and persist the measured attempt payload."""
    candidate_path = Path(args.candidate).resolve()
    workspace = workspace_path(args.workspace) if args.workspace else None
    workspace_metadata: dict[str, Any] = {}
    if workspace is not None:
        from kernel_bench_experiment_agents.workspace.paths import load_workspace_metadata as _load_workspace_metadata

        workspace_metadata = _load_workspace_metadata(workspace)
    source_name = (
        getattr(args, "problem_source", None)
        or workspace_metadata.get("problem_source")
        or DEFAULT_PROBLEM_SOURCE
    )
    problem_source = get_problem_source(str(source_name))
    problem_dir_arg = getattr(args, "problem_dir", None) or workspace_metadata.get("problem_dir")
    problem_archive_root = archive_problem_dir(args.run_name, args.level, args.problem_id)
    lease_name = f"artifacts:{args.run_name}:level_{args.level}:problem_{args.problem_id}"
    sample_id: int | None = None
    payload: dict[str, Any] | None = None
    sample_json_path: Path | None = None
    live_gpu_wait_marker = None
    failure: Exception | None = None
    persist_failure: Exception | None = None
    subprocess_timeout_seconds = timeout_seconds_from_env(
        "KBHARNESS_RUN_CANDIDATE_TIMEOUT_SECONDS",
        540.0,
    )

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
                extension=problem_source.extension,
            )
            sample_json_path = sample_manifest_path(
                args.run_name,
                args.level,
                args.problem_id,
                sample_id,
            )
            stdout_path = sample_json_path.with_suffix(".stdout.txt")
            stderr_path = sample_json_path.with_suffix(".stderr.txt")
            if workspace is not None:
                validate_workspace_assignment(
                    workspace,
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                )
                expected_candidate_path = workspace_candidate_path(workspace)
                if candidate_path != expected_candidate_path:
                    raise CandidateValidationError(
                        f"Only {problem_source.candidate_filename} may be evaluated from the problem workspace."
                    )

            candidate_ref = _workspace_candidate_reference(candidate_path, workspace)
            payload = {
                "status": "started",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "run_name": args.run_name,
                "level": args.level,
                "problem_id": args.problem_id,
                "sample_id": sample_id,
                "candidate_path": candidate_ref,
                "archive_kernel_path": relative_path_within(kernel_path, problem_archive_root),
                "stdout_path": relative_path_within(stdout_path, problem_archive_root),
                "stderr_path": relative_path_within(stderr_path, problem_archive_root),
                "backend": args.backend,
                "precision": args.precision,
                "problem_source": problem_source.name,
                "artifact_reservation_wait_seconds": artifact_lease.wait_seconds,
                "artifact_commit_wait_seconds": None,
                "gpu_id": None,
                "gpu_device_selector": None,
                "gpu_visible_devices": None,
                "gpu_logical_id": None,
                "gpu_selector_source": None,
                "gpu_wait_seconds": None,
                "gpu_quarantine_path": None,
                "gpu_quarantine_reason": None,
                "result": {},
                "warnings": [],
                "error": None,
                "subprocess": None,
            }

            candidate_src = problem_source.read_validated_candidate_source(candidate_path)
            kernel_path = problem_source.write_run_candidate_snapshot(
                snapshot_path=kernel_path,
                candidate_src=candidate_src,
                run_name=args.run_name,
                level=args.level,
                problem_id=args.problem_id,
                sample_id=sample_id,
            )
            if workspace is not None:
                write_workspace_sample_copy(workspace, sample_id, candidate_src)
            write_json(sample_json_path, payload)

        # The launcher polls goal status while this wrapper may still be queued for a
        # GPU lease, so record the live wait immediately instead of only after the
        # command eventually persists gpu_wait_seconds at the end of the run.
        live_gpu_wait_marker = create_live_gpu_wait_marker(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            operation="run_candidate",
            requested_gpu=args.gpu_id,
            num_gpu_slots=args.num_gpu_slots,
        )
        with lease_gpu_slot(
            num_slots=args.num_gpu_slots,
            requested_slot=args.gpu_id,
            lease_name=f"run:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
        ) as lease:
            settle_live_gpu_wait_marker(live_gpu_wait_marker, wait_seconds=lease.wait_seconds)
            mark_live_gpu_wait_operation_started(live_gpu_wait_marker)

            runner_output_path = build_problem_dir(
                args.run_name,
                args.level,
                args.problem_id,
                f"sample_{sample_id}",
            ) / "evaluation_result.json"
            command = problem_source.build_eval_command(
                EvalContext(
                    candidate_path=str(kernel_path),
                    output_path=str(runner_output_path),
                    run_name=args.run_name,
                    level=args.level,
                    problem_id=args.problem_id,
                    sample_id=sample_id,
                    gpu_id=lease.logical_gpu_id,
                    workspace=str(workspace) if workspace else None,
                    problem_dir=problem_dir_arg,
                    precision=args.precision,
                    kernelbench_root=args.kernelbench_root,
                    dataset_src=args.dataset_src,
                    backend=args.backend,
                    num_correct_trials=args.num_correct_trials,
                    num_perf_trials=args.num_perf_trials,
                    timing_method=args.timing_method,
                )
            )

            payload.update(
                {
                    "gpu_id": lease.slot_id,
                    "gpu_device_selector": lease.device_selector,
                    "gpu_visible_devices": lease.isolated_visible_devices,
                    "gpu_logical_id": lease.logical_gpu_id,
                    "gpu_selector_source": lease.selector_source,
                    "gpu_wait_seconds": lease.wait_seconds,
                }
            )

            def record_subprocess_start(start: SubprocessStart) -> None:
                payload["subprocess"] = subprocess_start_metadata(start)
                payload["updated_at"] = now_iso()
                write_json(sample_json_path, payload)

            try:
                completed = run_subprocess_streaming(
                    command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    env=isolated_gpu_environment(device_selector=lease.device_selector),
                    timeout_seconds=subprocess_timeout_seconds,
                    on_start=record_subprocess_start,
                )
            except SubprocessTimeoutError as exc:
                if subprocess_cleanup_incomplete(exc.result):
                    payload["gpu_quarantine_reason"] = "subprocess_cleanup_incomplete"
                    payload["gpu_quarantine_path"] = quarantine_gpu_slot(
                        lease,
                        reason="subprocess_cleanup_incomplete",
                        metadata=subprocess_result_metadata(exc.result),
                    )
                    payload["updated_at"] = now_iso()
                    write_json(sample_json_path, payload)
                raise
            payload["subprocess"] = subprocess_result_metadata(completed)

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
        runtime_error = result_runtime_error(result)
        payload["status"] = "execution_failed" if runtime_error else "succeeded"
        payload["updated_at"] = now_iso()
        payload["result"] = result
        payload["warnings"] = _result_warnings(result, workspace, stdout_text=completed.stdout)
        if runtime_error:
            payload["error"] = {
                "type": "KernelBenchExecutionError",
                "message": excerpt(runtime_error),
            }
    except Exception as exc:
        failure = exc
        if payload is None or sample_id is None or sample_json_path is None:
            raise
        payload["status"] = "failed"
        payload["updated_at"] = now_iso()
        if isinstance(exc, SubprocessTimeoutError):
            payload["subprocess"] = subprocess_result_metadata(exc.result)
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
                    clear_live_gpu_wait_marker(live_gpu_wait_marker)
                    live_gpu_wait_marker = None
                    if workspace is not None:
                        write_goal_status_files(
                            run_name=args.run_name,
                            level=args.level,
                            problem_id=args.problem_id,
                            workspace=workspace,
                        )
            except Exception as exc:
                persist_failure = exc
        else:
            clear_live_gpu_wait_marker(live_gpu_wait_marker)

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
