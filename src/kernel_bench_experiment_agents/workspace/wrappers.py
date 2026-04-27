"""Render the small shell wrappers that expose the harness CLI inside a workspace."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from kernel_bench_experiment_agents.runtime.project import make_executable, write_text


def write_workspace_script(path: Path, content: str) -> None:
    write_text(path, content)
    make_executable(path)


def workspace_wrapper_common() -> str:
    return dedent(
        """
        #!/usr/bin/env bash
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
        SOCKET_PATH="${KBH_COMMAND_SOCKET:-}"
        PYTHON_BIN="${PYTHON:-python}"

        if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
          echo "python is not on PATH. Launch through ./kb run or scripts/run_agent_problem.sh so the configured interpreter is exported." >&2
          exit 1
        fi
        if [[ -z "${SOCKET_PATH}" ]]; then
          echo "KBH_COMMAND_SOCKET is not set. Launch through ./kb run or scripts/run_agent_problem.sh so the privileged command broker is exported." >&2
          exit 1
        fi
        """
    ).lstrip()


def shell_multiline_command(lines: list[str]) -> str:
    return " \\\n".join(lines) + "\n"


def generate_run_wrapper(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
    precision: str,
) -> str:
    common = workspace_wrapper_common()
    command_lines = [
        '"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_client',
        '  --socket "${SOCKET_PATH}"',
        "  run-candidate",
    ]
    return common + shell_multiline_command(command_lines) + (
        'echo ">>> Read GOAL_STATUS.md now. If it still says UNRESOLVED, choose the next action yourself and keep iterating."\n'
    )


def generate_profile_wrapper(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
    precision: str,
) -> str:
    common = workspace_wrapper_common()
    command_lines = [
        '"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_client',
        '  --socket "${SOCKET_PATH}"',
        "  profile-ncu",
    ]
    return common + shell_multiline_command(command_lines) + (
        'echo ">>> Read profiles/latest.summary.txt first, then GOAL_STATUS.md, then pick the next optimization step yourself."\n'
    )


def generate_hardware_info_wrapper() -> str:
    common = workspace_wrapper_common()
    return common + 'cat "${WORKSPACE}/hardware.json"\n'


def generate_goal_status_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common()
    command_lines = [
        '"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_client',
        '  --socket "${SOCKET_PATH}"',
        "  goal-status",
    ]
    return common + shell_multiline_command(command_lines)


def generate_best_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common()
    command_lines = [
        '"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_client',
        '  --socket "${SOCKET_PATH}"',
        "  best-result",
    ]
    return common + shell_multiline_command(command_lines)


def generate_complete_wrapper(*, run_name: str, level: int, problem_id: int) -> str:
    common = workspace_wrapper_common()
    validation = dedent(
        """
        ORIGINAL_ARGS=("$@")
        HAVE_SUMMARY=false
        while [[ $# -gt 0 ]]; do
          case "$1" in
            --summary)
              shift
              if [[ $# -eq 0 ]]; then
                echo "complete_problem.sh: --summary requires a value" >&2
                exit 2
              fi
              HAVE_SUMMARY=true
              ;;
            --summary=*)
              HAVE_SUMMARY=true
              ;;
            *)
              echo "complete_problem.sh only accepts --summary" >&2
              exit 2
              ;;
          esac
          shift
        done

        if [[ "${HAVE_SUMMARY}" != true ]]; then
          echo "complete_problem.sh requires --summary" >&2
          exit 2
        fi
        """
    ).lstrip()
    command_lines = [
        '"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_client',
        '  --socket "${SOCKET_PATH}"',
        "  complete-problem",
        '  "${ORIGINAL_ARGS[@]}"',
    ]
    return common + validation + shell_multiline_command(command_lines)


def write_default_workspace_wrappers(
    *,
    bin_dir: Path,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    num_gpus: int,
    precision: str,
) -> list[Path]:
    wrappers = {
        "run_candidate.sh": generate_run_wrapper(
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            dataset_src=dataset_src,
            num_gpus=num_gpus,
            precision=precision,
        ),
        "profile_ncu.sh": generate_profile_wrapper(
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            dataset_src=dataset_src,
            num_gpus=num_gpus,
            precision=precision,
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
