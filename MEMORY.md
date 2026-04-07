# Rolling Memory

## Current focus

Tighten the runtime boundary so the solver-visible workspace, archive layout, and measured execution path are explicit and stable, with GPU-bound subprocess execution instead of loose in-process device access.

## Locked decisions

- `archive/` is the only durable copy-out root.
- `state/` is disposable runtime state.
- root `SPEC.md` is removed.
- root `AGENTS.md` and workspace `AGENTS.md` intentionally serve different audiences.
- solver terminal states are `done`, `stalled`, and `harness_failure`.
- launcher-only terminal states are `budget_exhausted` and `failed_to_generate`.
- measured baseline outcomes are inferred by the harness, not declared by the solver.
- helper-agent specs are generated per workspace from `src/kernel_bench_experiment_agents/agent_specs.py` and archived under `contract/helper_agents/`.
- structured JSON is the canonical machine-readable state; rendered Markdown is the solver-facing view.
- launcher and workspace wrappers should rely on the installed `kbe` entrypoint, not on repo-root `PYTHONPATH` injection.
- every wrapper except `./bin/complete_problem.sh` should behave as a fixed command with no solver-supplied control flags.
- hardware facts should come from frozen workspace files (`HARDWARE.md`, `hardware.json`, `./bin/hardware_info.sh`), not ad hoc host probing.

## Recently completed

- rewrote the root docs around explicit audiences and the `archive/` vs `state/` split
- narrowed completion handling to solver intent plus harness-inferred measured outcome
- moved helper-agent spec generation into per-problem workspaces and archived rendered copies with the run contract
- removed repo-root generated `.codex/agents/*` and `.claude/agents/*`
- added shared trace modules (`trace_ir.py`, `trace_analysis.py`) and switched trace materialization to a mostly-lossless IR written to `trace_ir.json`
- split `cli.py` into a thin parser/dispatcher plus dedicated modules:
  - `workspace_builder.py`
  - `workspace_state.py`
  - `execution_commands.py`
  - `status_commands.py`
  - `summary_commands.py`
  - `trace_commands.py`
  - `completion_policy.py`
- switched `scripts/run_agent_problem.sh` from `python -m kernel_bench_experiment_agents.cli` plus `PYTHONPATH` to the installed `kbe` CLI
- updated `scripts/setup_kernelbench_env.sh` to install this harness into the same KernelBench environment
- moved measured evaluation onto `evaluation_runner.py`, an isolated subprocess bound to one leased GPU slot through `CUDA_VISIBLE_DEVICES`
- tightened GPU slot resolution so the harness can lease against either the inherited visible GPU list or `KBE_VISIBLE_GPU_DEVICES`
- moved Nsight Compute execution onto the same isolated GPU-binding model and made profile metadata/workspace mirrors refresh under the per-problem state lock
- archived per-sample evaluation stdout/stderr alongside the sample manifest
- removed solver pass-through flags from every wrapper except `./bin/complete_problem.sh`

## In progress

- keep the new execution/runtime behavior aligned with the docs and archive contract
- continue smoke-testing the isolated GPU-binding path and profile metadata flow
- keep trimming remaining catch-all modules (`workspace_state.py`, `execution_commands.py`) once the behavior settles

## Next steps

- continue decomposing the larger state/execution modules once the runtime path feels stable
- consider whether more archive metadata should move from Markdown renders into structured JSON without adding redundant drift
- later, tighten any remaining boundary assumptions that still depend on the external sandbox rather than the harness itself
