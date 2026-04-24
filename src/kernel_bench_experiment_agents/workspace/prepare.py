"""Prepare a fresh per-problem workspace plus its matching archived contract bundle.

This is the top-level setup step that problem launchers call before handing control to the solver agent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from kernel_bench_experiment_agents.agent_contract.agent_specs import write_archive_helper_agent_specs
from kernel_bench_experiment_agents.workspace.archive import archive_problem_contract_dir, write_archive_problem_manifest
from kernel_bench_experiment_agents.runtime.common import emit_json, normalize_tool_name
from kernel_bench_experiment_agents.agent_contract.goal_status import write_goal_status_files
from kernel_bench_experiment_agents.agent_contract.hardware import resolve_hardware_spec
from kernel_bench_experiment_agents.kernelbench.problems import load_problem
from kernel_bench_experiment_agents.runtime.project import archive_problem_dir, build_problem_root, kernelbench_root, write_json
from kernel_bench_experiment_agents.kernelbench.metrics import baseline_file_paths, baseline_payload_for_problem
from kernel_bench_experiment_agents.workspace.materialization import (
    build_archive_provenance,
    build_hardware_payload,
    build_problem_metadata,
    write_contract_bundle,
)
from kernel_bench_experiment_agents.workspace.paths import problem_workspace_paths, workspace_candidate_path
from kernel_bench_experiment_agents.workspace.wrappers import write_default_workspace_wrappers


def command_prepare_problem_workspace(args: argparse.Namespace) -> None:
    """Create the workspace, archived contract, workspace wrappers, and initial goal status for one problem."""
    resolved_kernelbench_root = str(kernelbench_root(args.kernelbench_root))
    try:
        hardware = resolve_hardware_spec(args.hardware_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=resolved_kernelbench_root,
    )
    paths = problem_workspace_paths(
        args.run_name,
        args.level,
        args.problem_id,
    )
    problem_archive_dir = archive_problem_dir(args.run_name, args.level, args.problem_id)
    shutil.rmtree(paths["workspace"], ignore_errors=True)
    shutil.rmtree(problem_archive_dir, ignore_errors=True)
    shutil.rmtree(build_problem_root(args.run_name, args.level, args.problem_id), ignore_errors=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    eager_baseline_file, compile_baseline_file = baseline_file_paths(
        kernelbench_root=resolved_kernelbench_root,
        timings_dir=args.timings_dir,
        hardware_name=args.hardware_name,
    )

    baseline = baseline_payload_for_problem(
        level=args.level,
        problem_id=args.problem_id,
        problem_name=problem.name,
        eager_baseline_file=eager_baseline_file,
        compile_baseline_file=compile_baseline_file,
    )
    metadata = build_problem_metadata(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        tool=normalize_tool_name(args.tool),
        problem=problem,
        hardware=hardware,
        hardware_name=args.hardware_name,
        num_gpus=args.num_gpus,
        model=args.model,
        time_budget_minutes=args.time_budget_minutes,
        precision=args.precision,
    )
    hardware_payload = build_hardware_payload(hardware)
    provenance_payload = build_archive_provenance(
        kernelbench_root_path=resolved_kernelbench_root,
        timings_dir=str((Path(args.timings_dir).expanduser().resolve() if args.timings_dir else eager_baseline_file.parent)),
        problem=problem,
        eager_baseline_file=eager_baseline_file,
        compile_baseline_file=compile_baseline_file,
    )

    write_contract_bundle(
        target_dir=paths["workspace"],
        metadata=metadata,
        baseline=baseline,
        hardware_payload=hardware_payload,
        problem_code=problem.code,
    )

    contract_dir = archive_problem_contract_dir(args.run_name, args.level, args.problem_id)
    archive_manifest_path = write_archive_problem_manifest(args.run_name, args.level, args.problem_id)
    helper_agent_paths = write_archive_helper_agent_specs(
        archive_contract_dir=contract_dir,
    )
    write_contract_bundle(
        target_dir=contract_dir,
        metadata=metadata,
        baseline=baseline,
        hardware_payload=hardware_payload,
        problem_code=problem.code,
    )
    write_json(contract_dir / "provenance.json", provenance_payload)

    write_default_workspace_wrappers(
        bin_dir=paths["bin"],
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        num_gpus=args.num_gpus,
        precision=args.precision,
    )

    status_snapshot = write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=paths["workspace"],
    )

    emit_json(
        {
            "workspace": str(paths["workspace"]),
            "contract_dir": str(contract_dir),
            "archive_problem_dir": str(problem_archive_dir),
            "candidate": str(workspace_candidate_path(paths["workspace"])),
            "goal_status": str(paths["workspace"] / "goal_status.json"),
            "status_snapshot": status_snapshot,
            "helper_agent_specs": [str(path) for path in helper_agent_paths],
            "archive_manifest": str(archive_manifest_path),
        }
    )
