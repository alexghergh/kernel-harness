# KernelBench experiment harness

This repository runs one solver agent at a time on one KernelBench problem, records the full run, and keeps the durable artifacts in one place.

## What it does

For each `(run_name, level, problem_id)` the harness:

1. creates a fresh solver workspace
2. renders solver-facing docs and wrapper scripts into that workspace
3. launches Codex or Claude inside that workspace
4. lets the solver iterate through local wrapper commands only
5. records attempts, traces, completion state, and profiler outputs
6. stores the durable record under `archive/<run_name>/...`

The harness, not the solver, decides whether a final correct candidate beat eager PyTorch, `torch.compile`, both, or neither.

## Install

Install this repo into the same Python environment used for KernelBench:

```bash
pip install -e .
```

That provides the `kbe` CLI used by the generated workspace wrappers.

## Main entrypoints

Top-level launcher:

```bash
scripts/run_agent_problem.sh
```

Cleanup for one run:

```bash
scripts/clear_run.sh <run_name>
```

Summarize one archived run:

```bash
kbe summarize-run --run-name <run_name>
```

Helper-agent specs are generated automatically during `prepare-problem-workspace`. To regenerate them for one existing workspace explicitly:

```bash
kbe sync-helper-agent-specs \
  --workspace /path/to/workspace \
  --archive-contract-dir /path/to/archive/<run>/level_<level>/problem_<problem_id>/contract
```

## Durable vs temporary state

### Copy this out

If you want the full durable record for a run, copy:

```text
archive/<run_name>/
```

That directory is the canonical archive. It contains everything worth keeping for later analysis. Each problem archive now also includes `archive_manifest.json`, which spells out which subdirectories and file patterns are canonical versus merely workspace mirrors.

### Safe to discard

The harness keeps live runtime state under:

```text
state/
```

That includes:

- live workspaces
- isolated agent-home state
- build directories
- lock files

Do not treat `state/` as archival storage.

## Archive layout

Per problem, the durable record is:

```text
archive/<run_name>/level_<level>/problem_<problem_id>/
  archive_manifest.json
  contract/
  agent/
  attempts/
  profiles/
```

### `archive_manifest.json`

A machine-readable map of the canonical archive contents for that problem, including what should be copied out and which workspace files are only mirrors.

### `contract/`

The exact solver-facing contract for that workspace:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `INITIAL_PROMPT.md`
- `problem.json`
- `baseline.json`
- `hardware.json`
- `workspace_contract.json`
- `problem_reference.py`
- `candidate_final.py` once completion is written

### `agent/`

Agent-side run record:

- `events.jsonl` — raw CLI event stream
- `trace_ir.json` — normalized mostly-lossless trace IR materialization
- `final_message.txt`
- `completion.json`
- `goal_status.json`

### `attempts/`

Measured candidate attempts:

- `history.jsonl`
- `sample_<id>.json`
- `sample_<id>.stdout.txt`
- `sample_<id>.stderr.txt`
- `kernels/level_<level>_problem_<problem_id>_sample_<id>_kernel.py`
- `prompts/...` when a prompt snapshot is provided

### `profiles/`

Nsight Compute outputs:

- `index.jsonl`
- `profile_<id>.json`
- `profile_<id>.summary.txt`
- `profile_<id>.details.txt`
- `profile_<id>.raw.csv`
- `profile_<id>.stdout.txt`
- `profile_<id>.stderr.txt`

The text and CSV exports are first-class. The raw `.ncu-rep` file is optional debug retention.

## Workspace contract

Each solver workspace contains a small, explicit surface:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `GOAL_STATUS.md`
- `goal_status.json`
- `hardware.json`
- `workspace_contract.json`
- `problem_reference.py`
- `candidate_model_new.py`
- `samples/`
- `profiles/`
- `bin/*.sh`

The solver is expected to stay inside that workspace and use only the local wrapper scripts. `samples/` and `profiles/` inside the workspace are convenience mirrors of the durable archive, not a second source of truth. Every wrapper except `./bin/complete_problem.sh` is a fixed command with no solver-supplied control flags.

## Canonical JSON vs rendered Markdown

Some workspace information is kept in both JSON and Markdown on purpose.

- JSON is the machine-readable source used by the harness
- Markdown is the human-readable surface shown to the solver

Current pairs are:

- `workspace_contract.json` -> rendered `AGENTS.md` and `INITIAL_PROMPT.md`
- `hardware.json` -> rendered `HARDWARE.md`
- `goal_status.json` -> rendered `GOAL_STATUS.md`

These are not intended to be independently edited. The harness writes both forms from the same underlying payload so the solver gets readable text while the harness keeps a structured source of truth.

## Solver wrapper commands

The generated workspace exposes these commands:

- `./bin/problem_info.sh`
- `./bin/hardware_info.sh`
- `./bin/run_candidate.sh`
- `./bin/profile_ncu.sh`
- `./bin/goal_status.sh`
- `./bin/best_result.sh`
- `./bin/complete_problem.sh`

`run_candidate.sh` is the only supported path for measured evaluation.
`profile_ncu.sh` is the only supported path for profiling.
`complete_problem.sh` is the only wrapper that accepts solver-supplied flags, and only for `--state` plus `--summary`.

## Completion model

The solver may terminate only through `./bin/complete_problem.sh`.

Solver-written terminal states:

- `done`
- `stalled`
- `harness_failure`

Launcher-only terminal states:

- `budget_exhausted`
- `failed_to_generate`

Measured outcomes are computed by the harness from recorded attempts:

- `beats_both`
- `beats_eager_only`
- `beats_compile_only`
- `beats_none`
- `no_correct_candidate`

The live budget clock shown in `GOAL_STATUS.md` and `goal_status.json` is wall time since workspace creation minus recorded GPU wait time.

A successful run means the measured outcome is `beats_both`.

## Codex and Claude support

Both tools are first-class. The harness keeps tool-specific parsing and runtime setup separate from the generic run model.

Helper-agent specs are generated from the canonical definitions in:

```text
src/kernel_bench_experiment_agents/agent_specs.py
```

Generated outputs live inside each prepared workspace and are also archived under `contract/helper_agents/` for the exact rendered run contract:

- `state/workspaces/.../.codex/agents/*.toml`
- `state/workspaces/.../.claude/agents/*.md`
- `archive/.../contract/helper_agents/codex/*.toml`
- `archive/.../contract/helper_agents/claude/*.md`

## GPU/runtime isolation notes

- GPU slot leases are logical harness slots, not guaranteed physical device ids
- measured evaluation and Nsight Compute profiling now run in isolated subprocesses with `CUDA_VISIBLE_DEVICES` bound to the leased slot's selector
- inside that isolated subprocess the runner always uses logical device `cuda:0`
- if the cluster already restricts visibility through `CUDA_VISIBLE_DEVICES`, or you set `KBE_VISIBLE_GPU_DEVICES`, the harness leases against that visible selector list

## Current implementation notes

- `cli.py` is now a thin parser/dispatcher; workspace generation, execution, status, trace, and summary logic live in dedicated modules under `src/kernel_bench_experiment_agents/`
- the runtime layout, completion ownership, helper-agent generation, and GPU isolation model are aligned with the current architecture documents
- the larger state/execution catch-all modules have now been split further into archive, workspace-path, run-metric, goal-status, candidate-execution, and profiling modules
- the next major cleanup target is continued runtime hardening around external sandbox enforcement and any remaining archive-surface polish
