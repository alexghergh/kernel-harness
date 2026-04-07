from __future__ import annotations

import shlex
from pathlib import Path
from textwrap import dedent

from .candidate_contract import CANDIDATE_FILENAME
from .project import make_executable, write_text


def write_workspace_script(path: Path, content: str) -> None:
    write_text(path, content)
    make_executable(path)


def workspace_wrapper_common(*, kernelbench_root_path: str | None) -> str:
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
    kernelbench_root_path: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=kernelbench_root_path)
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
        'echo ">>> Read GOAL_STATUS.md now. If it still says UNRESOLVED, choose the next action yourself and keep iterating."\n'
    )


def generate_profile_wrapper(
    *,
    kernelbench_root_path: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=kernelbench_root_path)
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
        'echo ">>> Read profiles/latest.summary.txt first, then GOAL_STATUS.md, then pick the next optimization step yourself."\n'
    )


def generate_info_wrapper(
    *,
    kernelbench_root_path: str | None,
    level: int,
    problem_id: int,
    dataset_src: str,
) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=kernelbench_root_path)
    command_lines = [
        '"${KBE_CLI}" problem-info',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        f'  --dataset-src {shlex.quote(dataset_src)}',
    ]
    if kernelbench_root_path:
        command_lines.append('  --kernelbench-root "${KERNELBENCH_ROOT}"')
    return common + shell_multiline_command(command_lines)


def generate_hardware_info_wrapper() -> str:
    common = workspace_wrapper_common(kernelbench_root_path=None)
    return common + 'cat "${WORKSPACE}/hardware.json"\n'


def generate_goal_status_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=None)
    command_lines = [
        '"${KBE_CLI}" goal-status',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        '  --workspace "${WORKSPACE}"',
    ]
    return common + shell_multiline_command(command_lines)


def generate_best_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=None)
    command_lines = [
        '"${KBE_CLI}" best-result',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
    ]
    return common + shell_multiline_command(command_lines)


def generate_complete_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common(kernelbench_root_path=None)
    command_lines = [
        '"${KBE_CLI}" complete-problem',
        '  "$@"',
        f'  --run-name {shlex.quote(run_name)}',
        f'  --level {level}',
        f'  --problem-id {problem_id}',
        '  --workspace "${WORKSPACE}"',
    ]
    return common + shell_multiline_command(command_lines)


def write_default_workspace_wrappers(
    *,
    bin_dir: Path,
    kernelbench_root_path: str | None,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
) -> list[Path]:
    wrappers = {
        "run_candidate.sh": generate_run_wrapper(
            kernelbench_root_path=kernelbench_root_path,
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            dataset_src=dataset_src,
            num_gpus=num_gpus,
        ),
        "profile_ncu.sh": generate_profile_wrapper(
            kernelbench_root_path=kernelbench_root_path,
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            dataset_src=dataset_src,
            num_gpus=num_gpus,
        ),
        "problem_info.sh": generate_info_wrapper(
            kernelbench_root_path=kernelbench_root_path,
            level=level,
            problem_id=problem_id,
            dataset_src=dataset_src,
        ),
        "hardware_info.sh": generate_hardware_info_wrapper(),
        "goal_status.sh": generate_goal_status_wrapper(
            run_name=run_name,
            level=level,
            problem_id=problem_id,
        ),
        "best_result.sh": generate_best_wrapper(
            run_name=run_name,
            level=level,
            problem_id=problem_id,
        ),
        "complete_problem.sh": generate_complete_wrapper(
            run_name=run_name,
            level=level,
            problem_id=problem_id,
        ),
    }
    written: list[Path] = []
    for name, content in wrappers.items():
        path = bin_dir / name
        write_workspace_script(path, content)
        written.append(path)
    return written
