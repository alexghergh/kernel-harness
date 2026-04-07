# Rolling Memory

## Current focus

Tighten the harness so the runtime contract, archive layout, and solver-visible surface are explicit and stable while the internals keep moving out of `cli.py`.

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

## In progress

- keep the new modules aligned while smoke-testing the refactor
- improve runtime hardening later, especially GPU/device isolation and profiling execution details
- continue tightening documentation where the installed-CLI flow or archive contract changed

## Next steps

- harden GPU execution isolation and profiling flow in a later pass
- consider whether `workspace_state.py` should be split further once the runtime behavior settles
- keep workspace contract, trace materialization, and summarization aligned as the code is split further
