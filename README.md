# KernelBench experiment harness

This repository runs one autonomous solver agent at a time on one optimization problem.

The harness owns measured outcomes. The solver works inside a fresh, self-contained workspace, uses only local wrapper commands, and terminates through one narrow completion wrapper.

## What it does

For each `(run_name, level, problem_id)` the harness:

1. creates a fresh solver workspace
2. renders solver-facing docs and wrapper scripts into that workspace
3. launches Codex or Claude inside that workspace
4. lets the solver iterate through local wrapper commands only
5. records attempts, traces, completion state, and profiler outputs
6. stores the durable record under `archive/<run_name>/...`

The harness, not the solver, decides whether the final correct candidate beat eager PyTorch, `torch.compile`, both, or neither.

## Install

The harness should live in the same Python environment as the official KernelBench checkout.

### Recommended: use the helper script

```bash
export KERNELBENCH_ROOT=/path/to/KernelBench
./scripts/setup_kernelbench_env.sh uv
```

That does two things:

1. creates or updates the KernelBench `uv` environment
2. installs this harness into that same environment so the `kbe` CLI is available

### Manual `uv` flow

```bash
cd /path/to/KernelBench
uv sync --extra gpu
.venv/bin/python -m pip install -e /path/to/this/harness
```

### Manual `pip` flow

```bash
cd /path/to/KernelBench
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gpu]"
python -m pip install -e /path/to/this/harness
```

After install, `kbe` should resolve inside the KernelBench environment.

## Main entrypoints

Top-level launcher:

```bash
scripts/run_agent_problem.sh
```

Cleanup for one archived run and its disposable state:

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

That directory is the canonical archive. Each problem archive also includes `archive_manifest.json`, which explains which files are canonical and which workspace files are only mirrors.

### Safe to discard

Live mutable state is kept under:

```text
state/
```

That includes:

- live workspaces
- isolated agent-home state
- build directories
- lock files

`state/` is disposable. It is safe to delete **only when no run is active**. Do not treat it as archival storage.

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
- `candidate_model_new.py` — initial candidate scaffold shown to the solver
- `candidate_final.py` once completion is written
- `provenance.json` — archive-only provenance for the original KernelBench checkout and baseline input files

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

The solver is expected to stay inside that workspace and use only the local wrapper scripts. `samples/` and `profiles/` inside the workspace are convenience mirrors of the durable archive, not a second source of truth. Every wrapper except `./bin/complete_problem.sh` is a fixed command with no solver-supplied control flags. `./bin/complete_problem.sh` accepts only `--state` and `--summary`, and only the solver states `done` or `harness_failure`.

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

`done` means only that the solver believes the search is complete. It does **not** imply success. The live budget clock shown in `GOAL_STATUS.md` and `goal_status.json` is wall time since workspace creation minus recorded GPU wait time.

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

- GPU slot leases are logical harness slots backed by visible-device selectors
- measured evaluation and Nsight Compute profiling run in isolated subprocesses with `CUDA_VISIBLE_DEVICES` bound to the leased selector
- inside that isolated subprocess the runner always uses logical device `cuda:0`
