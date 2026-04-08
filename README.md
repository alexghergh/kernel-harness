# KernelBench harness

This repository runs autonomous coding agents on KernelBench optimization problems, one problem at a time, and records the durable results under `archive/`.

At a high level:

- the harness prepares a fresh problem workspace
- the agent works only inside that workspace
- local wrapper commands handle timing, profiling, status refresh, and completion
- the harness records attempts, traces, completion state, and summaries under `archive/`

For the detailed system contract, archive layout, workspace layout, and runtime boundary notes, read `ARCHITECTURE.md`.

## Before you use this repo

Set up the official KernelBench environment first, following the KernelBench repository instructions:

- <https://github.com/ScalingIntelligence/KernelBench>

This harness assumes:

- KernelBench is already installed and working in the **currently active** Python environment
- the active environment is the one you want this harness to use
- the KernelBench timing files already exist for your hardware
- if your timing results live outside the default KernelBench timing tree, set `KERNELBENCH_TIMINGS_DIR` when launching runs

Once that environment is active, install this harness into it from this repo:

```bash
uv pip install -e .
```

## Authenticate the agent tools

### Codex

Use a repo-local Codex home so the launcher can reuse that login:

```bash
CODEX_HOME="$(pwd)/.codex" codex login --device-auth
CODEX_HOME="$(pwd)/.codex" codex login status
```

### Claude Code

Export your Anthropic API key in the active shell:

```bash
export ANTHROPIC_API_KEY=...
```

## Most common runs

### Run one problem

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v3 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=gpt-5-codex \
TIME_BUDGET_MINUTES=180 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_problem.sh
```

### Run one problem with Claude

```bash
TOOL=claude \
RUN_NAME=kernelbench-claude-h100-v3 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=opus \
TIME_BUDGET_MINUTES=180 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_problem.sh
```

### Run a contiguous range

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v3 \
LEVEL=1 \
START_PROBLEM_ID=1 \
END_PROBLEM_ID=10 \
MODEL=gpt-5-codex \
TIME_BUDGET_MINUTES=180 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_range.sh
```

### Run an explicit problem list

```bash
TOOL=claude \
RUN_NAME=kernelbench-claude-h100-v3 \
LEVEL=1 \
PROBLEM_IDS=1,4,9 \
MODEL=opus \
TIME_BUDGET_MINUTES=180 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_range.sh
```

### Submit the Slurm wrapper

```bash
sbatch \
  --export=TOOL=codex,RUN_NAME=kernelbench-codex-h100-v3,LEVEL=1,START_PROBLEM_ID=1,END_PROBLEM_ID=10,MODEL=gpt-5-codex,TIME_BUDGET_MINUTES=180,KERNELBENCH_ROOT=/path/to/KernelBench,HARDWARE_NAME=H100 \
  ./scripts/run_agent_problem.slurm.sh
```


### Summarize one archived run

```bash
kbharness summarize-run --run-name kernelbench-codex-h100-v3
```

This scans only `archive/<run_name>/` and writes `archive/<run_name>/run_summary.json`.

## Operator knobs you will actually use

These are the main variables worth changing:

- `TOOL=codex|claude`
- `MODEL=...`
- `RUN_NAME=...`
- `LEVEL=...`
- `PROBLEM_ID=...`
- `START_PROBLEM_ID=...` / `END_PROBLEM_ID=...`
- `PROBLEM_IDS=1,4,9`
- `TIME_BUDGET_MINUTES=...`
- `KERNELBENCH_ROOT=/path/to/KernelBench`
- `HARDWARE_NAME=H100`
- `KERNELBENCH_TIMINGS_DIR=/path/to/results/timing/<hardware>` when you need a non-default timings location
- inherited `CUDA_VISIBLE_DEVICES` when you want to pin visible GPUs from the scheduler or shell

## Launcher exit semantics

`./scripts/run_agent_problem.sh` exits `0` for any valid archived run, even if the solver did **not** beat the baselines. It exits non-zero only for harness or launcher failures such as `harness_failure` or `failed_to_generate`. Use `completion.json`, `goal_status.json`, or `kbharness summarize-run` to judge optimization success.

## Where to look after a run

The only durable copy-out root is:

```text
archive/<run_name>/
```

Live workspaces, locks, and build products live under `state/` and are disposable once no run is active.

## CLI surface

Installing this repo exposes the harness CLI:

```bash
kbharness --help
```

The launcher scripts are the normal entrypoints. The CLI exists mainly so those scripts and generated workspace wrappers can call the harness internals in a stable way.

## Need more detail?

Read `ARCHITECTURE.md` for:

- archive contents and file meanings
- workspace contents and solver boundaries
- completion semantics
- how profiling, attempts, traces, and summaries are recorded
- what parts of the runtime policy are hard enforcement vs documented intent
