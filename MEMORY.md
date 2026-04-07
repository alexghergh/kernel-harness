# Rolling Memory

## Current focus

Keep tightening the harness contract so the solver-facing workspace is explicit, autonomous, and self-contained, while the codebase keeps moving away from large catch-all modules.

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
- solver docs should explicitly push independent execution: no approval-seeking, no plan-and-wait handoff, keep iterating from measured evidence.

## Recently completed

- rewrote the root docs around explicit audiences and the `archive/` vs `state/` split
- narrowed completion handling to solver intent plus harness-inferred measured outcome
- moved helper-agent spec generation into per-problem workspaces and archived rendered copies with the run contract
- removed repo-root generated `.codex/agents/*` and `.claude/agents/*`
- added shared trace modules (`trace_ir.py`, `trace_analysis.py`) and switched trace materialization to a mostly-lossless IR written to `trace_ir.json`
- split `cli.py` into a thin parser/dispatcher plus dedicated modules
- split the runtime/state code into:
  - `archive_layout.py`
  - `workspace_paths.py`
  - `run_metrics.py`
  - `goal_status.py`
  - `candidate_commands.py`
  - `profile_commands.py`
  - `subprocess_tools.py`
  - `ncu_summary.py`
- moved measured evaluation onto `evaluation_runner.py`, an isolated subprocess bound to one leased GPU slot through `CUDA_VISIBLE_DEVICES`
- tightened GPU slot resolution so the harness can lease against either the inherited visible GPU list or `KBE_VISIBLE_GPU_DEVICES`
- moved Nsight Compute execution onto the same isolated GPU-binding model and made profile metadata/workspace mirrors refresh under the per-problem state lock
- archived per-sample evaluation stdout/stderr alongside the sample manifest
- removed solver pass-through flags from every wrapper except `./bin/complete_problem.sh`
- split workspace preparation again into dedicated modules:
  - `workspace_wrappers.py`
  - `workspace_materialization.py`
  - `workspace_prepare.py`
  - `workspace_info.py`
- split run summarization again into dedicated modules:
  - `summary_math.py`
  - `summary_scan.py`
  - `summary_report.py`
- tightened the generated solver contract, goal-status text, wrapper reminders, and helper-agent specs so the model is explicitly told to work independently, keep iterating, and not ask for approval

## In progress

- keep the new solver-autonomy wording aligned across `workspace_contract.json`, rendered workspace docs, goal-status text, and helper-agent specs
- verify that the remaining docs describe the current code layout instead of earlier pre-split versions

## Next steps

- later, tighten any remaining boundary assumptions that still depend on the external sandbox rather than the harness itself
- consider whether more generated solver text should be driven by even more structured contract fields without overcomplicating the contract JSON
- decide whether any further run-level archive manifest is needed beyond per-problem `archive_manifest.json`
