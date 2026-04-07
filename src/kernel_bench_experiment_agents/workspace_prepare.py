from __future__ import annotations

import argparse
import shutil

from .agent_specs import write_workspace_helper_agent_specs
from .archive_layout import archive_problem_contract_dir, write_archive_problem_manifest
from .common import emit_json, normalize_tool_name
from .goal_status import write_goal_status_files
from .hardware_catalog import resolve_hardware_spec
from .kernelbench import load_problem
from .project import artifact_problem_dir, kernelbench_root
from .run_metrics import baseline_payload_for_problem
from .workspace_materialization import (
    build_hardware_payload,
    build_problem_metadata,
    write_contract_bundle,
)
from .workspace_paths import problem_workspace_paths, workspace_candidate_path
from .workspace_wrappers import write_default_workspace_wrappers


def command_prepare_problem_workspace(args: argparse.Namespace) -> None:
    resolved_kernelbench_root = str(kernelbench_root(args.kernelbench_root))
    try:
        hardware = resolve_hardware_spec(args.gpu_name)
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
        args.workspace_root,
    )
    archive_problem_dir = artifact_problem_dir(args.run_name, args.level, args.problem_id)
    shutil.rmtree(paths["workspace"], ignore_errors=True)
    shutil.rmtree(archive_problem_dir, ignore_errors=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    baseline = baseline_payload_for_problem(
        level=args.level,
        problem_id=args.problem_id,
        problem_name=problem.name,
        eager_baseline_file=args.eager_baseline_file,
        compile_baseline_file=args.compile_baseline_file,
    )
    metadata = build_problem_metadata(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        tool=normalize_tool_name(args.tool),
        problem=problem,
        hardware=hardware,
        num_gpus=args.num_gpus,
        model=args.model,
        time_budget_minutes=args.time_budget_minutes,
        kernelbench_root_path=resolved_kernelbench_root,
        kernelbench_python=args.kernelbench_python,
        workspace=paths["workspace"],
        baseline=baseline,
    )
    hardware_payload = build_hardware_payload(hardware)

    contract = write_contract_bundle(
        target_dir=paths["workspace"],
        metadata=metadata,
        baseline=baseline,
        hardware_payload=hardware_payload,
        problem_code=problem.code,
        include_candidate=True,
    )

    contract_dir = archive_problem_contract_dir(args.run_name, args.level, args.problem_id)
    archive_manifest_path = write_archive_problem_manifest(args.run_name, args.level, args.problem_id)
    helper_agent_paths = write_workspace_helper_agent_specs(
        workspace=paths["workspace"],
        archive_contract_dir=contract_dir,
    )
    write_contract_bundle(
        target_dir=contract_dir,
        metadata=metadata,
        baseline=baseline,
        hardware_payload=hardware_payload,
        problem_code=problem.code,
        include_candidate=False,
    )

    write_default_workspace_wrappers(
        bin_dir=paths["bin"],
        kernelbench_root_path=resolved_kernelbench_root,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        num_gpus=args.num_gpus,
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
            "archive_problem_dir": str(archive_problem_dir),
            "candidate": str(workspace_candidate_path(paths["workspace"])),
            "goal_status": str(paths["workspace"] / "goal_status.json"),
            "status_snapshot": status_snapshot,
            "helper_agent_specs": [str(path) for path in helper_agent_paths],
            "archive_manifest": str(archive_manifest_path),
        }
    )
