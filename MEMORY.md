# Rolling Memory

## Current focus

Keep tightening the harness contract so the solver-facing workspace is explicit, autonomous, and self-contained, while the codebase stays split into focused modules instead of catch-all files.

## Locked decisions

- `archive/` is the only durable copy-out root.
- `state/` is disposable runtime state.
- root `SPEC.md` is removed.
- root `AGENTS.md` and workspace `AGENTS.md` intentionally serve different audiences.
- solver terminal states are `done` and `harness_failure`.
- launcher-only terminal states are `budget_exhausted` and `failed_to_generate`.
- measured baseline outcomes are inferred by the harness, not declared by the solver.
- helper-agent specs are generated per workspace from `src/kernel_bench_experiment_agents/agent_specs.py` and archived under `contract/helper_agents/`.
- structured JSON is the canonical machine-readable state; rendered Markdown is the solver-facing view.
- the archived contract should preserve both the initial candidate scaffold and the final captured candidate when completion is written.
- launcher and workspace wrappers should rely on the installed `kbe` entrypoint, not on repo-root `PYTHONPATH` injection.
- every wrapper except `./bin/complete_problem.sh` should behave as a fixed command with no solver-supplied control flags.
- `./bin/complete_problem.sh` should accept only `--state done|harness_failure` and `--summary`.
- hardware facts should come from frozen workspace files (`HARDWARE.md`, `hardware.json`, `./bin/hardware_info.sh`), not ad hoc host probing.
- solver docs should explicitly push independent execution: no approval-seeking, no plan-and-wait handoff, keep iterating from measured evidence.
- the live solver workspace should not contain external filesystem paths; outward provenance belongs only in archived provenance metadata.

## Recently completed

- rewrote the root docs around explicit audiences and the `archive/` vs `state/` split
- narrowed completion handling to solver intent plus harness-inferred measured outcome
- moved helper-agent spec generation into per-problem workspaces and archived rendered copies with the run contract
- removed repo-root generated `.codex/agents/*` and `.claude/agents/*`
- added shared trace modules (`trace_ir.py`, `trace_analysis.py`) and switched trace materialization to a mostly-lossless IR written to `trace_ir.json`
- split `cli.py` into a thin parser/dispatcher plus dedicated modules
- split runtime/state code into focused archive, workspace-path, metrics, goal-status, candidate-execution, profiling, subprocess, and summary modules
- moved measured evaluation onto `evaluation_runner.py`, an isolated subprocess bound to one leased GPU selector through `CUDA_VISIBLE_DEVICES`
- tightened GPU slot resolution so the harness can lease against either the inherited visible GPU list or `KBE_VISIBLE_GPU_DEVICES`
- moved Nsight Compute execution onto the same isolated GPU-binding model and made profile metadata/workspace mirrors refresh under the per-problem state lock
- archived per-sample evaluation stdout/stderr alongside the sample manifest
- removed solver pass-through flags from every wrapper except `./bin/complete_problem.sh`
- tightened the generated solver contract, goal-status text, wrapper reminders, and helper-agent specs so the model is explicitly told to work independently, keep iterating, and not ask for approval
- removed `problem_info.sh` from the solver workspace surface and kept the workspace self-contained around local reference code and local mirrors only
- sanitized workspace-facing JSON so it no longer carries external filesystem paths; outward provenance now lives only in archived `contract/provenance.json`
- made archive sample/profile metadata use portable relative references instead of absolute paths
- made the completion wrapper reject launcher-only terminal states and unknown flags

## In progress

- verify that the remaining docs describe the current code layout instead of earlier pre-split versions
- validate real cluster behavior for Codex isolation, Claude isolation, GPU selector locking, and Nsight Compute output parsing

## Next steps

- later, tighten any remaining boundary assumptions that still depend on the external sandbox rather than the harness itself
- decide whether to keep trace audit as a hard invalidation step or demote it to diagnostics once the external sandbox is fully trusted
- verify on-cluster that `archive/<run_name>/` really contains everything needed and that `state/` can be deleted safely between inactive runs
