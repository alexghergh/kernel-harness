# Rolling Memory

## Current focus

Tighten the harness so the runtime contract, archive layout, and solver-visible surface are explicit and stable while the internals are split out of `cli.py`.

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

## Recently completed

- rewrote the root docs around explicit audiences and the `archive/` vs `state/` split
- narrowed completion handling to solver intent plus harness-inferred measured outcome
- moved helper-agent spec generation into per-problem workspaces and archived rendered copies with the run contract
- removed repo-root generated `.codex/agents/*` and `.claude/agents/*`
- added shared trace modules (`trace_ir.py`, `trace_analysis.py`) and switched trace materialization to a mostly-lossless IR written to `trace_ir.json`
- updated the launcher so Claude workspace settings no longer wipe workspace-generated helper agents

## In progress

- continue splitting `cli.py`; trace handling now delegates to dedicated modules but more command logic still lives there
- tighten profiling and evaluation execution paths further where needed
- keep archive outputs stable while refactoring internals

## Next steps

- move more command/workspace logic out of `cli.py`
- harden GPU execution isolation and profiling flow in a later pass
- keep workspace contract, trace materialization, and summarization aligned as the code is split further
