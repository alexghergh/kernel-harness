from __future__ import annotations

import argparse
import shlex
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any

from .agent_specs import write_workspace_helper_agent_specs
from .candidate_contract import CANDIDATE_FILENAME, candidate_template
from .common import emit_json, normalize_tool_name
from .hardware_catalog import render_hardware_markdown, resolve_hardware_spec
from .kernelbench import load_problem
from .project import (
    artifact_problem_dir,
    kernelbench_root,
    make_executable,
    now_iso,
    write_json,
    write_text,
)
from .workspace_contract import (
    build_workspace_contract,
    render_initial_prompt,
    render_workspace_agents_md,
    render_workspace_spec_md,
)
from .workspace_state import (
    archive_problem_contract_dir,
    baseline_payload_for_problem,
    problem_workspace_paths,
    workspace_candidate_path,
    write_goal_status_files,
)


def write_workspace_script(path: Path, content: str) -> None:
    write_text(path, content)
    make_executable(path)


def workspace_spec_markdown(
    *,
    problem: Any,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    hardware_markdown_name: str,
) -> str:
    return render_workspace_spec_md(
        problem_name=getattr(problem, "name", None),
        metadata=metadata,
        baseline=baseline,
        hardware_markdown_name=hardware_markdown_name,
    )


def workspace_wrapper_common(*, kernelbench_python: str, kernelbench_root_path: str | None) -> str:
    kb_root_line = (
        f'KERNELBENCH_ROOT={shlex.quote(str(Path(kernelbench_root_path).resolve()))}\n'
        if kernelbench_root_path
        else 'KERNELBENCH_ROOT=""\n'
    )
    return dedent(
        f"""
        #!/usr/bin/env bash
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        WORKSPACE="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
        KERNELBENCH_PYTHON={shlex.quote(kernelbench_python)}
        KBE_CLI="${{KBE_CLI:-kbe}}"
        {kb_root_line.rstrip()}

        if ! command -v "${{KBE_CLI}}" >/dev/null 2>&1; then
          echo "kbe CLI is not on PATH. Install this repo into the KernelBench environment first (pip install -e .)." >&2
          exit 1
        fi
        """
    ).lstrip()


def shell_multiline_command(lines: list[str]) -> str:
    return " \\\n".join(lines) + "\n"


def generate_run_wrapper(
    *,
    kernelbench_python: str,
    kernelbench_root_path: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=kernelbench_root_path,
    )
    command_lines = [
        '"${KBE_CLI}" run-candidate',
        f'  --candidate "${{WORKSPACE}}/{CANDIDATE_FILENAME}"',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        f'  --dataset-src {shlex.quote(dataset_src)}',
        '  --workspace "${WORKSPACE}"',
    ]
    if kernelbench_root_path:
        command_lines.append('  --kernelbench-root "${KERNELBENCH_ROOT}"')
    command_lines.append(f'  --num-gpu-slots {num_gpus}')
    return common + shell_multiline_command(command_lines) + (
        'echo ">>> Re-read GOAL_STATUS.md and SPEC.md before your next decision."\n'
    )


def generate_profile_wrapper(
    *,
    kernelbench_python: str,
    kernelbench_root_path: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=kernelbench_root_path,
    )
    command_lines = [
        '"${KBE_CLI}" profile-ncu',
        f'  --candidate "${{WORKSPACE}}/{CANDIDATE_FILENAME}"',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        f'  --dataset-src {shlex.quote(dataset_src)}',
        '  --workspace "${WORKSPACE}"',
    ]
    if kernelbench_root_path:
        command_lines.append('  --kernelbench-root "${KERNELBENCH_ROOT}"')
    command_lines.append(f'  --num-gpu-slots {num_gpus}')
    return common + shell_multiline_command(command_lines) + (
        'echo ">>> Read profiles/latest.summary.txt first, then profiles/latest.details.txt if needed."\n'
    )


def generate_info_wrapper(
    *,
    kernelbench_python: str,
    kernelbench_root_path: str | None,
    level: int,
    problem_id: int,
    dataset_src: str,
) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=kernelbench_root_path,
    )
    command_lines = [
        '"${KBE_CLI}" problem-info',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        f'  --dataset-src {shlex.quote(dataset_src)}',
    ]
    if kernelbench_root_path:
        command_lines.append('  --kernelbench-root "${KERNELBENCH_ROOT}"')
    return common + shell_multiline_command(command_lines)


def generate_hardware_info_wrapper(*, kernelbench_python: str) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=None,
    )
    return common + 'cat "${WORKSPACE}/hardware.json"\n'


def generate_goal_status_wrapper(*, kernelbench_python: str, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=None,
    )
    command_lines = [
        '"${KBE_CLI}" goal-status',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        '  --workspace "${WORKSPACE}"',
    ]
    return common + shell_multiline_command(command_lines)


def generate_best_wrapper(*, kernelbench_python: str, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=None,
    )
    command_lines = [
        '"${KBE_CLI}" best-result',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
    ]
    return common + shell_multiline_command(command_lines)


def generate_complete_wrapper(*, kernelbench_python: str, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(
        kernelbench_python=kernelbench_python,
        kernelbench_root_path=None,
    )
    command_lines = [
        '"${KBE_CLI}" complete-problem',
        '  "$@"',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        '  --workspace "${WORKSPACE}"',
    ]
    return common + shell_multiline_command(command_lines)


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
    metadata = {
        "created_at": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "tool": normalize_tool_name(args.tool),
        "dataset_src": args.dataset_src,
        "problem_name": problem.name,
        "problem_path": problem.path,
        "gpu_name": hardware.display_name,
        "gpu_architecture": hardware.architecture,
        "gpu_compute_capability": hardware.compute_capability,
        "num_gpus": args.num_gpus,
        "model": args.model,
        "time_budget_minutes": args.time_budget_minutes,
        "kernelbench_root": resolved_kernelbench_root,
        "kernelbench_python": args.kernelbench_python,
        "workspace": str(paths["workspace"]),
        "candidate_path": str(workspace_candidate_path(paths["workspace"])),
        "eager_baseline_file": baseline["eager"]["source_file"],
        "compile_baseline_file": baseline["compile"]["source_file"],
    }
    hardware_payload = {
        "display_name": hardware.display_name,
        "architecture": hardware.architecture,
        "compute_capability": hardware.compute_capability,
        "registers_per_sm": hardware.registers_per_sm,
        "max_registers_per_thread": hardware.max_registers_per_thread,
        "max_warps_per_sm": hardware.max_warps_per_sm,
        "max_blocks_per_sm": hardware.max_blocks_per_sm,
        "shared_memory_per_sm_kb": hardware.shared_memory_per_sm_kb,
        "max_shared_memory_per_block_kb": hardware.max_shared_memory_per_block_kb,
        "shared_memory_carveout_kb": list(hardware.shared_memory_carveout_kb),
        "guidance": list(hardware.guidance),
        "doc_urls": list(hardware.doc_urls),
    }
    contract = build_workspace_contract(metadata=metadata)

    write_json(paths["workspace"] / "problem.json", metadata)
    write_json(paths["workspace"] / "baseline.json", baseline)
    write_json(paths["workspace"] / "hardware.json", hardware_payload)
    write_json(paths["workspace"] / "workspace_contract.json", contract)
    write_text(paths["workspace"] / "problem_reference.py", problem.code)
    write_text(workspace_candidate_path(paths["workspace"]), candidate_template())
    write_text(paths["workspace"] / "HARDWARE.md", render_hardware_markdown(hardware))
    write_text(
        paths["workspace"] / "SPEC.md",
        workspace_spec_markdown(
            problem=problem,
            metadata=metadata,
            baseline=baseline,
            hardware_markdown_name="HARDWARE.md",
        ),
    )
    write_text(paths["workspace"] / "AGENTS.md", render_workspace_agents_md(contract=contract))
    write_text(paths["workspace"] / "INITIAL_PROMPT.md", render_initial_prompt(contract=contract, baseline=baseline))

    contract_dir = archive_problem_contract_dir(args.run_name, args.level, args.problem_id)
    helper_agent_paths = write_workspace_helper_agent_specs(
        workspace=paths["workspace"],
        archive_contract_dir=contract_dir,
    )
    write_json(contract_dir / "problem.json", metadata)
    write_json(contract_dir / "baseline.json", baseline)
    write_json(contract_dir / "hardware.json", hardware_payload)
    write_json(contract_dir / "workspace_contract.json", contract)
    write_text(contract_dir / "problem_reference.py", problem.code)
    write_text(contract_dir / "HARDWARE.md", render_hardware_markdown(hardware))
    write_text(
        contract_dir / "SPEC.md",
        workspace_spec_markdown(
            problem=problem,
            metadata=metadata,
            baseline=baseline,
            hardware_markdown_name="HARDWARE.md",
        ),
    )
    write_text(contract_dir / "AGENTS.md", render_workspace_agents_md(contract=contract))
    write_text(contract_dir / "INITIAL_PROMPT.md", render_initial_prompt(contract=contract, baseline=baseline))

    write_workspace_script(
        paths["bin"] / "run_candidate.sh",
        generate_run_wrapper(
            kernelbench_python=args.kernelbench_python,
            kernelbench_root_path=resolved_kernelbench_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
            num_gpus=args.num_gpus,
        ),
    )
    write_workspace_script(
        paths["bin"] / "profile_ncu.sh",
        generate_profile_wrapper(
            kernelbench_python=args.kernelbench_python,
            kernelbench_root_path=resolved_kernelbench_root,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
            num_gpus=args.num_gpus,
        ),
    )
    write_workspace_script(
        paths["bin"] / "problem_info.sh",
        generate_info_wrapper(
            kernelbench_python=args.kernelbench_python,
            kernelbench_root_path=resolved_kernelbench_root,
            level=args.level,
            problem_id=args.problem_id,
            dataset_src=args.dataset_src,
        ),
    )
    write_workspace_script(
        paths["bin"] / "hardware_info.sh",
        generate_hardware_info_wrapper(kernelbench_python=args.kernelbench_python),
    )
    write_workspace_script(
        paths["bin"] / "goal_status.sh",
        generate_goal_status_wrapper(
            kernelbench_python=args.kernelbench_python,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
    )
    write_workspace_script(
        paths["bin"] / "best_result.sh",
        generate_best_wrapper(
            kernelbench_python=args.kernelbench_python,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
    )
    write_workspace_script(
        paths["bin"] / "complete_problem.sh",
        generate_complete_wrapper(
            kernelbench_python=args.kernelbench_python,
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
        ),
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
        }
    )


def command_problem_info(args: argparse.Namespace) -> None:
    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=args.kernelbench_root,
    )
    emit_json(
        {
            "level": problem.level,
            "problem_id": problem.problem_id,
            "dataset_src": problem.dataset_src,
            "name": problem.name,
            "path": problem.path,
            "code": problem.code,
        }
    )
