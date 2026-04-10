"""Build the solver-visible workspace contract and markdown docs for one problem.

Workspace materialization uses these renderers to turn typed metadata and shared policy into the files the agent actually reads.
"""

from __future__ import annotations

from typing import Any

from .candidate_contract import CANDIDATE_FILENAME
from .policy_model import (
    ALLOWED_WEB_DOMAINS,
    HELPER_SPECS,
    LAUNCHER_TERMINAL_STATES,
    SOLVER_TERMINAL_STATES,
    WORKSPACE_COMMAND_SPECS,
    WORKSPACE_EDIT_PATHS,
    WORKSPACE_READ_PATHS,
    WORKSPACE_STANDING_ORDERS,
    WORKSPACE_STUCK_PROTOCOL,
)


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
            "precision": metadata.get("precision", "bf16"),
        },
        "reads": list(WORKSPACE_READ_PATHS),
        "edits": list(WORKSPACE_EDIT_PATHS),
        "wrapper_commands": [
            {
                "name": spec.name,
                "path": spec.path,
                "gpu": spec.uses_gpu,
                "purpose": spec.purpose,
            }
            for spec in WORKSPACE_COMMAND_SPECS
        ],
        "helper_agents": [spec.name for spec in HELPER_SPECS],
        "behavior": {
            "independent_execution": True,
            "no_user_confirmation": True,
            "no_plain_message_exit": True,
            "standing_orders": list(WORKSPACE_STANDING_ORDERS),
            "stuck_protocol": list(WORKSPACE_STUCK_PROTOCOL),
        },
        "web": {
            "allowed_domains": list(ALLOWED_WEB_DOMAINS),
            "shell_network_forbidden": True,
        },
        "solver_terminal_states": list(SOLVER_TERMINAL_STATES),
        "launcher_terminal_states": list(LAUNCHER_TERMINAL_STATES),
    }


# These markdown renderers are the solver-facing contract for a prepared workspace,
# so they must stay aligned with the wrapper interface and runtime policy.
def render_workspace_agents_md(*, contract: dict[str, Any]) -> str:
    assignment = contract["assignment"]
    behavior = contract.get("behavior") or {}
    helper_names = ", ".join(f"`{name}`" for name in contract.get("helper_agents", []))
    precision = assignment.get("precision") or "bf16"
    lines = [
        "# Solver Instructions",
        "",
        "You are the autonomous solver for exactly one optimization problem.",
        "",
        "Assignment:",
        "",
        f"- run name: `{assignment['run_name']}`",
        f"- level: `{assignment['level']}`",
        f"- problem id: `{assignment['problem_id']}`",
        f"- dataset source: `{assignment['dataset_src']}`",
        f"- problem name: `{assignment.get('problem_name') or 'unknown'}`",
        f"- reported GPU name: `{assignment.get('gpu_name') or 'not provided'}`",
        f"- available GPU slots for wrapper execution: `{assignment.get('num_gpus')}`",
        f"- total solver budget: `{assignment.get('time_budget_minutes')}` minutes",
        f"- judged precision path: `{precision}`",
        "",
        "Read order:",
        "",
        "1. `AGENTS.md`",
        "2. `SPEC.md`",
        "3. `HARDWARE.md`",
        "4. `GOAL_STATUS.md`",
        "",
        "Scope:",
        "",
        "- stay inside this workspace-visible surface; do not treat hidden harness storage as part of your usable environment",
        "- do not read, edit, or execute anything outside this workspace",
        "- do not inspect repository-maintainer docs or harness internals",
        "- treat `samples/` and `profiles/` as local mirrors of archived outputs, not as separate sources of truth",
        "- do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas",
        "- use `problem_reference.py` as the problem reference",
        f"- edit only `{CANDIDATE_FILENAME}` for the candidate solution, and only inside its marked editable blocks",
        f"- the judged path is `{precision}`; internal mixed precision is allowed only if the final candidate still passes the `{precision}` correctness checks",
        "",
        "Allowed wrapper commands:",
        "",
        *[
            f"- `{command['path']}` — {command['purpose']}"
            for command in contract["wrapper_commands"]
        ],
        "",
        "Wrapper argument policy:",
        "",
        "- treat `./bin/hardware_info.sh`, `./bin/run_candidate.sh`, `./bin/profile_ncu.sh`, `./bin/goal_status.sh`, and `./bin/best_result.sh` as fixed commands with no solver-supplied flags",
        "- `./bin/complete_problem.sh` is the only wrapper that accepts solver-supplied flags, and only for `--summary`",
        "- never overlap wrapper calls; start a new `./bin/*.sh` command only after the previous wrapper has returned",
        "- `./bin/run_candidate.sh` and `./bin/profile_ncu.sh` may take a while; trust them and wait for them to return instead of treating them as hung",
        "",
        "Allowed reads:",
        "",
        *[f"- `{path}`" for path in contract["reads"]],
        "",
        "Allowed edits:",
        "",
        *[f"- `{path}`" for path in contract["edits"]],
        "",
        "Web policy:",
        "",
        "- if hosted web search is enabled, use it only for `docs.nvidia.com`",
        "- do not use shell networking at all",
        "- do not use online code, papers, forums, or blogs for solution ideas",
        "",
        "Optional helper agents:",
        "",
        f"- if your runtime exposes generated helper agents, use them proactively for isolated evaluation or profiling work: {helper_names}",
        "- helper agents are execution aids, not a reason to stop or ask the user for approval",
        "",
        "Completion:",
        "",
        "- the solver may terminate only through `./bin/complete_problem.sh --summary \"...\"`",
        "- solver-written completion is always recorded as `done`; the harness infers the measured outcome from actual artifacts",
        "- `budget_exhausted` and `failed_to_generate` are launcher-only states",
        "- post-hoc harness invalidation is handled by archive policy, not by solver-supplied `--state` flags",
        "- do not end with a plain assistant message",
        "",
        "Standing orders:",
        "",
        *[f"- {value}" for value in list(behavior.get("standing_orders") or [])],
        "",
        "When stuck:",
        "",
        *[f"- {value}" for value in list(behavior.get("stuck_protocol") or [])],
        "",
        "Rules:",
        "",
        "- LOOP UNTIL DONE. DO NOT STOP EARLY.",
        "- every measured attempt must go through `./bin/run_candidate.sh`",
        "- profiling is a normal tool, not a last resort",
        "- trust wrapper output; do not monitor wrapper progress with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection",
        "- `HARDWARE.md` and `hardware.json` are the only supported hardware surface; do not probe the machine yourself",
        "- do not use ad hoc shell, Python, or benchmarking commands outside the local wrappers",
        "- re-read `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` before any major strategy change and before any termination decision",
        "- keep working until both baselines are beaten or a truthful terminal state is reached",
    ]
    return "\n".join(lines) + "\n"


def render_workspace_spec_md(
    *,
    problem_name: str | None,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    hardware_markdown_name: str,
) -> str:
    precision = metadata.get("precision", "bf16")
    lines = [
        "# Orders",
        "",
        "You are optimizing one problem. The harness decides the measured outcome from actual runs; your job is to keep iterating until you have either solved the problem or truthfully exhausted the allowed stopping conditions.",
        "",
        "## Target",
        "",
        f"- problem: `{problem_name or 'unknown'}` (level `{metadata['level']}`, problem `{metadata['problem_id']}`)",
        f"- eager PyTorch baseline: `{baseline['eager']['runtime_ms']}` ms",
        f"- `torch.compile` baseline: `{baseline['compile']['runtime_ms']}` ms",
        "- the strongest outcome is to beat both baselines with one correct candidate",
        f"- optimize `problem_reference.py` by editing only `{CANDIDATE_FILENAME}`",
        "- the evaluated implementation must be raw custom CUDA/C++ extension code with minimal glue; vendor-library wrappers, Triton, and ATen compute helpers are forbidden",
        f"- correctness and runtime are evaluated on the harness `{precision}` path",
        "",
        "## Autonomy",
        "",
        "- there is no human confirmation step during the run",
        "- do not ask whether to proceed, whether a plan is acceptable, or whether the user wants you to continue",
        "- if one attempt fails, make the next plan yourself and continue",
        "- if stuck, re-read the local docs, use profiling, consult allowed NVIDIA docs, and try another implementation branch",
        "",
        "## Termination",
        "",
        "The only valid exit path is `./bin/complete_problem.sh --summary \"...\"`.",
        "",
        "Solver-written completion:",
        "",
        "- use `./bin/complete_problem.sh --summary \"...\"` when you believe the current search is complete",
        "- the harness records that as `done` and infers whether you beat eager, compile, both, or neither from measured artifacts",
        "",
        "Launcher-only terminal states:",
        "",
        *[f"- `{state}`" for state in LAUNCHER_TERMINAL_STATES],
        "",
        "## Loop",
        "",
        f"1. Edit `{CANDIDATE_FILENAME}`.",
        "2. Run `./bin/run_candidate.sh`.",
        "3. Read `GOAL_STATUS.md`.",
        "4. If needed, run `./bin/profile_ncu.sh` and read `profiles/latest.summary.txt` first.",
        "5. Repeat until `./bin/complete_problem.sh` is justified.",
        "",
        "All wrappers other than `./bin/complete_problem.sh` are fixed commands. Do not pass alternate paths, run ids, or extra control flags to them.",
        "Never overlap wrapper calls. Start a new `./bin/*.sh` command only after the previous wrapper has fully returned.",
        "`./bin/run_candidate.sh` and `./bin/profile_ncu.sh` may take a while; trust them and wait for the wrapper result instead of treating them as hung.",
        "",
        "## Budget and status",
        "",
        f"- total budget: `{metadata['time_budget_minutes']}` minutes",
        "- remaining budget: read `GOAL_STATUS.md` or `goal_status.json`",
        "- the budget clock is wall time since workspace creation minus recorded GPU wait time and any live GPU lease wait currently in progress",
        "- recorded GPU lock wait time and an active GPU lease wait in progress are excluded from the budget",
        "- failed attempts are normal; they are not a stop signal",
        "- there is no human confirmation step during the run",
        "",
        "## References",
        "",
        "- problem code: `problem_reference.py`",
        f"- solution file: `{CANDIDATE_FILENAME}`",
        f"- hardware facts: `{hardware_markdown_name}` and `hardware.json`",
        "- live status: `GOAL_STATUS.md` and `goal_status.json`",
        "- local mirrors of measured attempts/profiles: `samples/` and `profiles/`",
        "- exact workspace contract: `workspace_contract.json`",
    ]
    return "\n".join(lines) + "\n"


def render_initial_prompt(*, contract: dict[str, Any], baseline: dict[str, Any]) -> str:
    assignment = contract["assignment"]
    precision = assignment.get("precision") or "bf16"
    lines = [
        "Optimize exactly one problem.",
        "",
        f"- run name: {assignment['run_name']}",
        f"- level: {assignment['level']}",
        f"- problem id: {assignment['problem_id']}",
        f"- dataset source: {assignment['dataset_src']}",
        f"- problem name: {assignment.get('problem_name') or 'unknown'}",
        f"- eager baseline: {baseline['eager']['runtime_ms']} ms",
        f"- compile baseline: {baseline['compile']['runtime_ms']} ms",
        f"- total solver budget: {assignment.get('time_budget_minutes')} minutes",
        f"- judged precision path: {precision}",
        "",
        "Start by reading `AGENTS.md`, then `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md`.",
        f"Stay inside this workspace. Only edit `{CANDIDATE_FILENAME}`. Use only the local `./bin/*.sh` wrapper commands. Treat every wrapper except `./bin/complete_problem.sh` as a fixed command with no extra flags.",
        "Work independently. There is no user approval step in this run. Do not ask for permission, confirmation, or whether to continue.",
        "Never overlap wrapper calls. Start a new `./bin/*.sh` command only after the previous wrapper has returned.",
        "If a strategy fails, re-read the docs, profile when useful, consult allowed NVIDIA docs when needed, and start the next strategy yourself.",
        "`./bin/run_candidate.sh` and `./bin/profile_ncu.sh` may take a while; trust the wrapper output and wait for them to finish.",
        "Do not stop early. When you are truly finished, terminate only through `./bin/complete_problem.sh --summary \"...\"`.",
        "The harness will infer the measured outcome from the recorded runs.",
    ]
    return "\n".join(lines) + "\n"
