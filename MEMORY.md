# Rolling Memory

## Current focus

Tighten the harness contract so docs, launcher behavior, archive layout, and completion semantics all describe the same system.

## Locked decisions

- `archive/` is the only durable copy-out root.
- `state/` is disposable runtime state.
- root `SPEC.md` is removed.
- root `AGENTS.md` and workspace `AGENTS.md` intentionally serve different audiences.
- solver terminal states are `done`, `stalled`, and `harness_failure`.
- launcher-only terminal states are `budget_exhausted` and `failed_to_generate`.
- measured baseline outcomes are inferred by the harness, not declared by the solver.
- Codex and Claude helper-agent specs are generated from `src/kernel_bench_experiment_agents/agent_specs.py`.
- structured JSON is the canonical machine-readable state; rendered Markdown is the solver-facing view.

## Recently completed

- moved durable-vs-temp repo docs to `archive/` vs `state/`
- updated launcher scripts to use the new layout
- narrowed completion handling to solver state plus harness-inferred measured outcome
- refreshed root docs and added ADRs
- added canonical helper-agent spec generation
- added a generated workspace contract module

## In progress

- continue splitting `cli.py` into smaller modules
- tighten profiling and evaluation execution paths further where needed
- keep archive layout stable while refactoring internals

## Next steps

- move more command logic out of `cli.py`
- preserve archive compatibility for runs created after this refactor
- keep workspace contract, trace materialization, and summarization aligned as the code is split further
