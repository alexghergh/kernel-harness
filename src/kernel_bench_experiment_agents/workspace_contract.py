from __future__ import annotations

from textwrap import dedent
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME

ALLOWED_WEB_DOMAINS = ["docs.nvidia.com"]
SOLVER_TERMINAL_STATES = ["done", "stalled", "harness_failure"]
LAUNCHER_TERMINAL_STATES = ["budget_exhausted", "failed_to_generate"]
WORKSPACE_COMMANDS = [
    {"name": "problem_info", "path": "./bin/problem_info.sh", "gpu": False, "purpose": "print the KernelBench reference problem"},
    {"name": "hardware_info", "path": "./bin/hardware_info.sh", "gpu": False, "purpose": "print frozen hardware facts for this workspace"},
    {"name": "run_candidate", "path": "./bin/run_candidate.sh", "gpu": True, "purpose": "evaluate correctness and runtime for the current candidate"},
    {"name": "profile_ncu", "path": "./bin/profile_ncu.sh", "gpu": True, "purpose": "profile the current candidate with Nsight Compute"},
    {"name": "goal_status", "path": "./bin/goal_status.sh", "gpu": False, "purpose": "refresh and print live goal status"},
    {"name": "best_result", "path": "./bin/best_result.sh", "gpu": False, "purpose": "print the best measured correct result so far"},
    {"name": "finish_problem", "path": "./bin/complete_problem.sh", "gpu": False, "purpose": "record a terminal solver state"},
]


def build_workspace_contract(*, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "assignment": {
            "run_name": metadata["run_name"],
            "level": metadata["level"],
            "problem_id": metadata["problem_id"],
            "dataset_src": metadata["dataset_src"],
            "problem_name": metadata.get("problem_name"),
            "gpu_name": metadata.get("gpu_name"),
            "num_gpus": metadata.get("num_gpus"),
            "time_budget_minutes": metadata.get("time_budget_minutes"),
            "model": metadata.get("model"),
        },
        "reads": [
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
            "goal_status.json",
            "hardware.json",
            "workspace_contract.json",
            "problem.json",
            "problem_reference.py",
            CANDIDATE_FILENAME,
            "baseline.json",
            "samples/",
            "profiles/",
        ],
        "edits": [CANDIDATE_FILENAME],
        "wrapper_commands": WORKSPACE_COMMANDS,
        "web": {
            "allowed_domains": ALLOWED_WEB_DOMAINS,
            "shell_network_forbidden": True,
        },
        "solver_terminal_states": SOLVER_TERMINAL_STATES,
        "launcher_terminal_states": LAUNCHER_TERMINAL_STATES,
    }


def render_workspace_agents_md(*, contract: dict[str, Any]) -> str:
    assignment = contract["assignment"]
    command_lines = "\n".join(
        f"- `{command['path']}` — {command['purpose']}"
        for command in contract["wrapper_commands"]
    )
    read_lines = "\n".join(f"- `{path}`" for path in contract["reads"])
    edit_lines = "\n".join(f"- `{path}`" for path in contract["edits"])
    terminal_lines = "\n".join(
        f"- `{state}`" for state in contract["solver_terminal_states"]
    )
    return dedent(
        f"""
        # Solver Instructions

        You are the solver for exactly one KernelBench problem.

        Assignment:

        - run name: `{assignment['run_name']}`
        - level: `{assignment['level']}`
        - problem id: `{assignment['problem_id']}`
        - dataset source: `{assignment['dataset_src']}`
        - problem name: `{assignment.get('problem_name') or 'unknown'}`
        - reported GPU name: `{assignment.get('gpu_name') or 'not provided'}`
        - available GPU slots for wrapper execution: `{assignment.get('num_gpus')}`
        - total solver budget: `{assignment.get('time_budget_minutes')}` minutes

        Read order:

        1. `AGENTS.md`
        2. `SPEC.md`
        3. `HARDWARE.md`
        4. `GOAL_STATUS.md`

        Scope:

        - stay inside this workspace
        - do not read, edit, or execute anything outside this workspace
        - do not inspect repository-maintainer docs or harness internals
        - do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas
        - use `problem_reference.py` only as the problem reference
        - edit only `{CANDIDATE_FILENAME}` for the candidate solution, and only inside its marked editable blocks
        - the judged path is `fp32`; internal mixed precision is allowed only if the final candidate still passes the `fp32` correctness checks

        Allowed wrapper commands:

        {command_lines}

        Allowed reads:

        {read_lines}

        Allowed edits:

        {edit_lines}

        Web policy:

        - if hosted web search is enabled, use it only for `docs.nvidia.com`
        - do not use shell networking at all
        - do not use online code, papers, forums, or blogs for solution ideas

        Completion:

        - the solver may terminate only through `./bin/complete_problem.sh`
        - valid solver-written terminal states are:
        {terminal_lines}
        - `./bin/complete_problem.sh --state done --summary "..."` means “I am done; the harness will infer the measured baseline outcome from actual artifacts”
        - `budget_exhausted` and `failed_to_generate` are launcher-only states
        - do not end with a plain assistant message

        Rules:

        - LOOP UNTIL DONE. DO NOT STOP EARLY.
        - every measured attempt must go through `./bin/run_candidate.sh`
        - profiling is a normal tool, not a last resort
        - trust wrapper output; do not monitor wrapper progress with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection
        - do not use ad hoc shell, Python, or benchmarking commands outside the local wrappers
        - re-read `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` before any major strategy change and before any termination decision
        - keep working until both baselines are beaten or a truthful terminal state is reached
        """
    ).strip() + "\n"


def render_workspace_spec_md(
    *,
    problem_name: str | None,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    hardware_markdown_name: str,
) -> str:
    return dedent(
        f"""
        # Orders

        You are optimizing one KernelBench problem. The harness decides the measured outcome from actual runs; your job is to keep iterating until you have either solved the problem or truthfully exhausted the allowed stopping conditions.

        ## Target

        - problem: `{problem_name or 'unknown'}` (level `{metadata['level']}`, problem `{metadata['problem_id']}`)
        - eager PyTorch baseline: `{baseline['eager']['runtime_ms']}` ms
        - `torch.compile` baseline: `{baseline['compile']['runtime_ms']}` ms
        - the strongest outcome is to beat both baselines with one correct candidate
        - optimize `problem_reference.py` by editing only `{CANDIDATE_FILENAME}`
        - the evaluated implementation must be raw custom CUDA/C++ extension code with minimal glue; vendor-library wrappers, Triton, and ATen compute helpers are forbidden
        - correctness and runtime are evaluated on the harness `fp32` path

        ## Termination

        The only valid exit path is `./bin/complete_problem.sh`.

        Solver-written terminal states:

        - `done` — you believe the current search is complete; the harness will infer whether you beat eager, compile, both, or neither from measured artifacts
        - `stalled` — substantial exploration has failed and the remaining time no longer justifies continued search
        - `harness_failure` — the environment or harness is broken in a way that blocks truthful progress

        Launcher-only terminal states:

        - `budget_exhausted`
        - `failed_to_generate`

        ## Loop

        1. Edit `{CANDIDATE_FILENAME}`.
        2. Run `./bin/run_candidate.sh`.
        3. Read `GOAL_STATUS.md`.
        4. If needed, run `./bin/profile_ncu.sh` and read `profiles/latest.summary.txt` first.
        5. Repeat until `./bin/complete_problem.sh` is justified.

        ## Budget and status

        - total budget: `{metadata['time_budget_minutes']}` minutes
        - remaining budget: read `GOAL_STATUS.md` or `goal_status.json`
        - recorded GPU lock wait time is excluded from the budget
        - failed attempts are normal; they are not a stop signal
        - there is no human confirmation step during the run

        ## References

        - problem code: `problem_reference.py`
        - solution file: `{CANDIDATE_FILENAME}`
        - hardware facts: `{hardware_markdown_name}` and `hardware.json`
        - live status: `GOAL_STATUS.md` and `goal_status.json`
        - exact workspace contract: `workspace_contract.json`
        """
    ).strip() + "\n"


def render_initial_prompt(*, contract: dict[str, Any], baseline: dict[str, Any]) -> str:
    assignment = contract["assignment"]
    return dedent(
        f"""
        Optimize exactly one KernelBench problem.

        - run name: {assignment['run_name']}
        - level: {assignment['level']}
        - problem id: {assignment['problem_id']}
        - dataset source: {assignment['dataset_src']}
        - problem name: {assignment.get('problem_name') or 'unknown'}
        - eager baseline: {baseline['eager']['runtime_ms']} ms
        - compile baseline: {baseline['compile']['runtime_ms']} ms
        - total solver budget: {assignment.get('time_budget_minutes')} minutes

        Start by reading `AGENTS.md`, then `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md`.
        Stay inside this workspace. Only edit `{CANDIDATE_FILENAME}`. Use only the local `./bin/*.sh` wrapper commands.
        Do not stop early. When you are truly finished, terminate only through `./bin/complete_problem.sh --state done --summary "..."` or another truthful terminal state.
        The harness will infer the measured outcome from the recorded runs.
        """
    ).strip() + "\n"
