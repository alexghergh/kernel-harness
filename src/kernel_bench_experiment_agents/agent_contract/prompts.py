"""Render the solver-facing markdown and opening prompt for one workspace."""

from __future__ import annotations

from typing import Any

from kernel_bench_experiment_agents.agent_contract.policy import LAUNCHER_TERMINAL_STATES
from kernel_bench_experiment_agents.kernelbench.candidate.contract import CANDIDATE_FILENAME
from kernel_bench_experiment_agents.runtime.common import as_float


RAW_KERNEL_RULE_LINES = (
    "You may rewrite the candidate file freely; there are no protected markers or scaffold lines to preserve.",
    "Define `ModelNew(nn.Module)` and build your own custom CUDA/C++ extension via `load_inline` or `load`.",
    "Write your own `__global__` kernel, `<<<...>>>` launch path, and pybind-visible entrypoint.",
    "Do not call PyTorch ops that perform the same math: `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`, or Python `@` are forbidden.",
    "Do not use ATen wrappers, Triton, or vendor-library shortcuts such as cuBLAS, cuDNN, CUTLASS, or similar helpers.",
)


def _direct_command_tool_name(*, tool: str | None, name: str) -> str | None:
    if (tool or "").strip().lower() in {"codex", "claude"}:
        return f"mcp__kernelbench_commands__{name}"
    return None


def render_workspace_agents_md(*, contract: dict[str, Any]) -> str:
    assignment = contract["assignment"]
    behavior = contract.get("behavior") or {}
    helper_names = ", ".join(f"`{name}`" for name in contract.get("helper_agents", []))
    precision = assignment.get("precision") or "bf16"
    command_tools = list(contract.get("command_tools") or [])
    direct_tool_names = ", ".join(
        f"`{tool['tool_name']}`" for tool in command_tools if tool.get("tool_name")
    )
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
        f"- available GPU slots for measured tool execution: `{assignment.get('num_gpus')}`",
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
        "- work directly in this prepared workspace and nowhere else",
        "- read only the allowed workspace files below; do not inspect repository-maintainer docs, hidden harness storage, or tool-private config state",
        "- do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas",
        "- use `problem_reference.py` as the problem reference",
        f"- edit only `{CANDIDATE_FILENAME}` with normal file-edit tools; the rest of the workspace is mounted read-only during solver execution",
        f"- the judged path is `{precision}`; internal mixed precision is allowed only if the final candidate still passes the `{precision}` correctness checks",
        *[f"- {value}" for value in RAW_KERNEL_RULE_LINES],
        "",
        "Backend wrapper commands (compatibility/debug surface, not the primary solver tools):",
        "",
        *[
            f"- `{command['path']}` - {command['purpose']}"
            for command in contract["commands"]
        ],
    ]
    if command_tools:
        lines.extend(
            [
                "",
                "Equivalent privileged command tools exposed in this runtime:",
                "",
                *[
                    f"- `{tool['tool_name']}` - {tool['purpose']}"
                    for tool in command_tools
                ],
            ]
        )
    lines.extend(
        [
            "",
            "Command policy:",
            "",
            "- the direct `complete_problem` tool is the only valid harness exit path when direct command tools are exposed",
            "- never overlap direct command tools; start a new one only after the previous one has returned",
            "- `run_candidate` and `profile_ncu` may take a while; trust them and wait for them to return instead of treating them as hung",
            (
                f"- this runtime exposes direct command tools {direct_tool_names}; use those exact tool names instead of attempting shell commands"
                if command_tools
                else "- if your runtime exposes direct command tools, use those instead of shelling out"
            ),
            (
                "- do not attempt `Bash` in this runtime; it is not exposed here"
                if command_tools
                else "- if `Bash` is unavailable, do not keep retrying it"
            ),
            "- backend wrapper scripts exist for compatibility and human debugging; they are not permission to use general shell commands",
            "- use normal file tools for reads and edits, but do not inspect hidden launcher runtime state or tool-private config",
            "",
            "Allowed workspace reads:",
            "",
            *[f"- `{path}`" for path in contract["reads"]],
            "",
            "Allowed workspace edits:",
            "",
            *[f"- `{path}`" for path in contract["edits"]],
            "",
            "Web policy:",
            "",
            "- prefer the direct `research_nvidia_docs` command tool for official NVIDIA docs research when it is exposed",
            "- use hosted web search or fetch only for `docs.nvidia.com`",
            "- when CUDA, PTX, WMMA/MMA, tensor-core, memory-hierarchy, occupancy, Nsight Compute, or compiler behavior is uncertain, call `research_nvidia_docs` before guessing",
            "- after repeated CUDA API, compile, profiling-metric, or hardware-tuning failures, call `research_nvidia_docs` before the next code edit",
            "- useful research roots include `https://docs.nvidia.com/llms.txt`, `https://docs.nvidia.com/cuda/llms.txt`, and the official URLs in `HARDWARE.md`",
            "- do not use shell networking at all",
            "- do not use online code, papers, forums, or blogs for solution ideas",
            "",
            "Optional helper agents:",
            "",
            f"- if your runtime exposes generated helper agents, use them proactively for isolated evaluation or profiling work: {helper_names}",
            "- helper agents should use the same direct command tools for measured actions",
            "- helper agents are execution aids, not a reason to stop or ask the user for approval",
            "",
            "Completion:",
            "",
            "- the solver may terminate only through the direct `complete_problem` tool",
            "- solver-written completion is always recorded as `done`; the harness infers the measured outcome from actual artifacts",
            "- `budget_exhausted` and `failed_to_generate` are launcher-only states",
            "- post-hoc harness invalidation is handled by archive policy, not by solver-supplied terminal states",
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
            "- every measured attempt must go through the direct `run_candidate` tool",
            "- profiling is a required optimization step, not a last resort",
            "- when a candidate compiles and runs but is slower than either baseline, use `profile_ncu` before more than one further optimization edit",
            "- do not declare a slow correct candidate fundamentally limited unless you have profiled it and read `profiles/latest.summary.txt`",
            "- trust command-tool output; do not monitor progress with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection",
            "- `HARDWARE.md` and `hardware.json` are the only supported hardware surface; do not probe the machine yourself",
            "- do not use ad hoc shell, Python, or benchmarking commands",
            "- re-read `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` before any major strategy change and before any termination decision",
            "- keep working until both baselines are beaten or a truthful terminal state is reached",
        ]
    )
    return "\n".join(lines) + "\n"


def render_workspace_spec_md(
    *,
    problem_name: str | None,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    hardware_markdown_name: str,
) -> str:
    precision = metadata.get("precision", "bf16")
    tool = str(metadata.get("tool") or "")
    direct_run_tool = _direct_command_tool_name(tool=tool, name="run_candidate")
    direct_profile_tool = _direct_command_tool_name(tool=tool, name="profile_ncu")
    direct_research_tool = _direct_command_tool_name(tool=tool, name="research_nvidia_docs")
    direct_complete_tool = _direct_command_tool_name(tool=tool, name="complete_problem")
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
        f"- optimize `problem_reference.py` by editing only `{CANDIDATE_FILENAME}` directly",
        "- the evaluated implementation must be raw custom CUDA/C++ extension code with minimal glue",
        *[f"- {value}" for value in RAW_KERNEL_RULE_LINES],
        f"- correctness and runtime are evaluated on the harness `{precision}` path",
        "",
        "## Tool loop",
        "",
        "1. Read `AGENTS.md`, `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` directly in the workspace.",
        f"2. Edit `{CANDIDATE_FILENAME}` directly.",
        (
            f"3. Run `{direct_run_tool}` directly."
            if direct_run_tool
            else "3. Run the direct `run_candidate` tool if your runtime exposes it."
        ),
        "4. Re-read `GOAL_STATUS.md`.",
        (
            f"5. As soon as a candidate compiles and runs but is slower than either baseline, run `{direct_profile_tool}` directly before making more than one further optimization edit; read `profiles/latest.summary.txt` first."
            if direct_profile_tool
            else "5. As soon as a candidate compiles and runs but is slower than either baseline, run the direct `profile_ncu` tool before making more than one further optimization edit; read `profiles/latest.summary.txt` first."
        ),
        (
            f"6. When CUDA/NVIDIA-specific behavior is uncertain or repeated compile/profile failures occur, call `{direct_research_tool}` before the next code edit."
            if direct_research_tool
            else "6. When CUDA/NVIDIA-specific behavior is uncertain or repeated compile/profile failures occur, call the direct `research_nvidia_docs` tool before the next code edit."
        ),
        (
            f"7. Repeat until `{direct_complete_tool}` is justified."
            if direct_complete_tool
            else "7. Repeat until the direct `complete_problem` tool is justified."
        ),
        "",
        "Never overlap direct command tools. Start a new one only after the previous one has fully returned.",
        "`run_candidate` and `profile_ncu` may take a while; trust them and wait for the result instead of treating them as hung.",
        "",
        "## Termination",
        "",
        (
            f"The only valid exit path is `{direct_complete_tool}`."
            if direct_complete_tool
            else "The only valid exit path is the direct `complete_problem` tool."
        ),
        "",
        "Solver-written completion:",
        "",
        (
            f"- use `{direct_complete_tool}` when you believe the current search is complete"
            if direct_complete_tool
            else "- use the direct `complete_problem` tool when you believe the current search is complete"
        ),
        "- the harness records that as `done` and infers whether you beat eager, compile, both, or neither from measured artifacts",
        "",
        "Launcher-only terminal states:",
        "",
        *[f"- `{state}`" for state in LAUNCHER_TERMINAL_STATES],
        "",
        "## Budget and status",
        "",
        f"- total budget: `{metadata['time_budget_minutes']}` minutes",
        "- remaining budget: re-read `GOAL_STATUS.md` or `goal_status.json` in the workspace",
        "- failed attempts are normal; they are not a stop signal",
        "",
        "## References",
        "",
        "- problem code: `problem_reference.py`",
        f"- solution file: `{CANDIDATE_FILENAME}`",
        f"- hardware facts: `{hardware_markdown_name}` and `hardware.json`",
        "- live status: `GOAL_STATUS.md` and `goal_status.json`",
        "- local mirrors of measured attempts/profiles: `samples/latest.json`, `samples/latest.stdout.txt`, `samples/latest.stderr.txt`, and `profiles/`",
        "- backend compatibility wrappers: `bin/`",
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
        "Start by reading `AGENTS.md`, `SPEC.md`, `HARDWARE.md`, and `GOAL_STATUS.md` directly in the workspace.",
        f"Stay inside this prepared workspace. Edit only `{CANDIDATE_FILENAME}` directly.",
        "Use only the direct command tools `run_candidate`, `profile_ncu`, `research_nvidia_docs`, `goal_status`, `best_result`, and `complete_problem` for harness actions.",
        "Do not use shell commands or Python snippets for harness actions.",
        "After each measured run, read `samples/latest.json` first. If that attempt failed, inspect `samples/latest.stdout.txt` and `samples/latest.stderr.txt` before revising the candidate.",
        "You may rewrite the candidate file freely. Define `ModelNew(nn.Module)` and build your own custom CUDA/C++ extension via `load_inline` or `load`.",
        "Write your own `__global__` kernel, `<<<...>>>` launch path, and pybind-visible entrypoint.",
        "Do not replace the computation with `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`, Python `@`, Triton, ATen helpers, cuBLAS, cuDNN, CUTLASS, or similar shortcuts.",
        "Work independently. There is no user approval step in this run. Do not ask for permission, confirmation, or whether to continue.",
        "Never overlap direct command tools. Start a new one only after the previous one has returned.",
        "If a strategy fails, re-read the docs, profile any compiling slow candidate, call `research_nvidia_docs` when CUDA/NVIDIA details are uncertain, and start the next strategy yourself.",
        "Do not decide a slow correct candidate is fundamentally limited until you have run `profile_ncu` and read `profiles/latest.summary.txt`.",
        "`run_candidate` and `profile_ncu` may take a while; trust the command output and wait for them to finish.",
        "Do not stop early. When you are truly finished, terminate only through the direct `complete_problem` tool.",
        "The harness will infer the measured outcome from the recorded runs.",
    ]
    return "\n".join(lines) + "\n"


def render_goal_status_markdown(snapshot: dict[str, Any]) -> str:
    best_runtime = snapshot.get("best_correct_runtime_ms")
    eager_baseline = snapshot.get("eager_baseline_ms")
    compile_baseline = snapshot.get("compile_baseline_ms")
    problem_name = snapshot.get("problem_name") or "unknown"
    wall_clock_elapsed_minutes = as_float(snapshot.get("wall_clock_elapsed_minutes"))
    elapsed_minutes = as_float(snapshot.get("elapsed_minutes"))
    recorded_gpu_wait_minutes = as_float(snapshot.get("recorded_gpu_wait_minutes"))
    live_gpu_wait_minutes = as_float(snapshot.get("live_gpu_wait_minutes"))
    gpu_wait_minutes_total = as_float(snapshot.get("gpu_wait_minutes_total"))
    remaining_minutes = as_float(snapshot.get("remaining_minutes"))
    time_budget_minutes = as_float(snapshot.get("time_budget_minutes"))
    substantial_budget_remains = (
        remaining_minutes is not None
        and time_budget_minutes is not None
        and remaining_minutes > max(60.0, time_budget_minutes * 0.25)
    )
    latest_attempt_blocked_reason = str(snapshot.get("latest_attempt_blocked_reason") or "").strip()
    unresolved = not snapshot["beats_both"]
    if unresolved:
        heading = "# Goal Status: UNRESOLVED - keep working"
        if latest_attempt_blocked_reason:
            heading = "# Goal Status: UNRESOLVED - latest attempt was discarded; keep working"
        standing_orders = [
            "- You MUST NOT stop, summarize, or hand back control. Keep working.",
            "- Do NOT ask the user for confirmation, approval, or whether to continue. Choose the next action yourself.",
            "- Re-read `SPEC.md` and `HARDWARE.md` before every major strategy change.",
            "- Timing and profiling are normal tools, not expensive last resorts. Profile any compiling slow candidate before making more than one further optimization edit.",
            "- Do not decide a slow correct candidate is fundamentally limited until you have run the direct `profile_ncu` tool and read `profiles/latest.summary.txt`.",
            "- Never overlap direct command tools. Start a new harness action only after the previous one has fully returned.",
            "- If one is slow, wait for it. Do NOT monitor it with `ps`, `pgrep`, `top`, `htop`, `nvidia-smi`, `strace`, `/proc`, or build-tree inspection.",
            "- If the latest run was discarded as suspicious, cheating, or invalid, it does not count. Fix the exact reported issue and keep working.",
            "- If stuck: run the direct `profile_ncu` tool once the candidate compiles and runs, read `profiles/latest.summary.txt`, read `HARDWARE.md`, call `research_nvidia_docs` for NVIDIA-specific uncertainty, make a new plan, and try a new branch without asking for approval.",
            "- The budget clock is wall time since workspace creation minus recorded GPU wait time and any live GPU lease wait currently in progress. End through the direct `complete_problem` tool before remaining time reaches zero.",
            "- A plain assistant message is NEVER a valid way to end this run.",
            "- `run_candidate` and `profile_ncu` may take a while; wait for the result instead of treating them as hung.",
        ]
    else:
        heading = "# Goal Status: RESOLVED - both baselines beaten; complete with success"
        standing_orders = [
            "- Re-check `SPEC.md` once, then end through the direct `complete_problem` tool with summary `both baselines beaten`.",
        ]

    if remaining_minutes is None:
        remaining_line = "unknown"
    elif substantial_budget_remains:
        remaining_line = (
            f"{remaining_minutes} (most of your budget remains - stopping now wastes it)"
        )
    else:
        remaining_line = str(remaining_minutes)

    if best_runtime is None:
        best_runtime_line = "none yet"
    else:
        best_runtime_line = (
            f"{best_runtime} ms (must be below {eager_baseline} ms and {compile_baseline} ms)"
        )

    attempt_breakdown = (
        f"{snapshot['num_correct_attempts']} correct, "
        f"{snapshot['num_incorrect_attempts']} incorrect, "
        f"{snapshot['num_execution_failed_attempts']} execution-failed"
    )
    if snapshot.get("num_other_attempts"):
        attempt_breakdown += f", {snapshot['num_other_attempts']} other"

    lines = [
        heading,
        "",
        "Standing orders (active until both baselines are beaten):",
        "",
        *standing_orders,
        "",
        "## Current State",
        "",
        f"- problem: level {snapshot['level']} problem {snapshot['problem_id']} ({problem_name})",
        f"- best correct runtime: {best_runtime_line}",
        f"- beats eager ({eager_baseline} ms): {snapshot['beats_eager']}",
        f"- beats compile ({compile_baseline} ms): {snapshot['beats_compile']}",
        f"- beats both: {snapshot['beats_both']}",
        f"- latest attempt sample: {snapshot.get('latest_attempt_sample_id')}",
        f"- latest attempt counts toward progress: {snapshot.get('latest_attempt_counts_toward_progress', True)}",
        f"- latest attempt discard reason: {snapshot.get('latest_attempt_blocked_reason') or 'none'}",
        f"- attempts counted toward progress: {snapshot['num_attempts']} ({attempt_breakdown})",
        f"- timing calls: {snapshot['num_timing_runs']}",
        f"- profiler calls: {snapshot['num_profile_runs']}",
        f"- best correct sample: {snapshot.get('best_correct_sample_id')}",
        f"- best result warnings: {snapshot.get('best_result_warnings') or []}",
        f"- wall-clock minutes since workspace creation: {wall_clock_elapsed_minutes}",
        f"- completed gpu wait minutes excluded from budget: {recorded_gpu_wait_minutes}",
        f"- currently active gpu queue-wait minutes excluded from budget: {live_gpu_wait_minutes}",
        f"- total gpu wait minutes excluded from budget: {gpu_wait_minutes_total}",
        f"- elapsed minutes counted against budget: {elapsed_minutes}",
        f"- remaining minutes: {remaining_line}",
        "- static docs: `AGENTS.md`, `SPEC.md`, `HARDWARE.md`",
        "- live docs: `GOAL_STATUS.md`, `goal_status.json`",
        "- local sample mirrors: `samples/latest.json`, `samples/latest.stdout.txt`, `samples/latest.stderr.txt`, `samples/best_sample.py`, `samples/best_result.json`",
        "- latest profiler mirrors: `profiles/latest.summary.txt` and `profiles/latest.details.txt`",
        "",
        "Source of truth: measured run history plus the live solver trace. Refresh via the direct `goal_status` or `run_candidate` command tools.",
    ]
    return "\n".join(lines) + "\n"
