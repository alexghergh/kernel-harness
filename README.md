# KernelBench agent harness

A reproducible agent harness for running Codex or Claude on **one KernelBench problem at a time** through a narrow Landrun boundary and launcher-owned command broker.

It does four things:

- prepares a fresh per-problem workspace and solver contract
- exposes a small self-contained workspace for direct reads/edits while keeping privileged actions behind command tools
- archives every attempt, profile, trace, and completion record under `archive/`
- scores the final result against eager and `torch.compile` baselines

The disposable live state is under `state/`. The durable record is under `archive/`.

If you only read one more file after this README, read `ARCHITECTURE.md`.


## What the model is actually told

The solver policy is not hidden in one giant prompt. It is split across a few generated files that the model re-reads during the run:

- `AGENTS.md` — the durable top-level solver contract
- `INITIAL_PROMPT.md` — the opening run-specific instructions
- `GOAL_STATUS.md` — the live progress/status file the solver should re-read after measured actions
- direct command tools — the brokered harness action and NVIDIA-docs research surface
- helper-agent specs for `runner` and `profiler` — narrow delegated roles when the client runtime supports sub-agents

The intended behavior is:

- the main agent acts as the **planner-manager**
- `runner` handles measured evaluation by default
- `profiler` handles Nsight Compute work by default
- direct `run_candidate` / `profile_ncu` from the main agent are valid paths when helper spawning is unavailable
- direct `research_nvidia_docs` is the canonical audited path for NVIDIA-specific docs lookup

## Actual solver surface

For **both** Codex and Claude, the model starts inside the prepared problem workspace under Landrun. It may read the generated workspace docs/history directly, while Landrun mounts only `candidate_model_new.py` writable.

Privileged actions are still launcher-owned. Direct command tools route to a Unix-socket broker outside the Landrun policy; workspace `bin/*.sh` wrappers remain for compatibility and human debugging. Hosted web access stays separate and tool-native.

The exact live model trace is saved as `archive/.../agent/events.jsonl`. Broker-mediated activity is saved as `archive/.../agent/activity_ir_events.jsonl`. `trace_ir.json` is the normalized merged view used for counts and summaries.

- fixed direct reads: `AGENTS.md`, `INITIAL_PROMPT.md`, `SPEC.md`, `HARDWARE.md`, `GOAL_STATUS.md`, `problem_reference.py`, `candidate_model_new.py`, `workspace_contract.json`, `problem.json`, `hardware.json`
- bounded history reads: `samples/` and `profiles/`
- only workspace edit: `candidate_model_new.py`
- OS-enforced local write surface: `candidate_model_new.py`; other workspace paths are read-only to the agent process
- privileged actions: `run_candidate`, `profile_ncu`, `research_nvidia_docs`, `goal_status`, `best_result`, `complete_problem`
- `goal_status` returns the live JSON status snapshot and refreshes `GOAL_STATUS.md`
- `best_result` returns the current best measured correct attempt, including `sample_id` and archive-relative artifact paths
- native web stays separate from local actions and is limited to `docs.nvidia.com`; comparable enforced-docs runs should use brokered `research_nvidia_docs`

The client-specific enforcement differs slightly:

- **Codex** runs in the workspace with native file/edit tools and the shell tool disabled. A per-problem `CODEX_HOME` is generated under disposable `state/tool_state/...`, and a tiny command MCP server exposes only brokered privileged actions.
- **Claude** runs in the workspace with normal file tools and a tiny command MCP server for brokered privileged actions. A per-problem `CLAUDE_CONFIG_DIR` is generated under disposable `state/tool_state/...`; Bash is not exposed, and file-tool hooks deny reads outside the assigned workspace and writes outside `candidate_model_new.py`.


## Setup

The clean repo-root entrypoint is:

```bash
./kb setup
```

Optional setup knobs:

- `./kb setup --python 3.10`
- `./kb setup --venv-dir /path/to/venv`

This harness assumes:

- Go 1.18 or newer is available during `./kb setup` so the vendored Landrun submodule can be built
- the official KernelBench checkout exists either at `./third_party/KernelBench/` or wherever `KERNELBENCH_ROOT` points
- the KernelBench timing files already exist for your hardware
- `KERNELBENCH_TIMINGS_DIR` is optional; set it only when your timing results live outside the default KernelBench timing tree

Notes:

- `./kb setup` always syncs and initializes the vendored `third_party/KernelBench` and `third_party/landrun` submodules first, so older clones pick up `.gitmodules` URL changes automatically.
- `./kb setup` builds the vendored Landrun checkout into `third_party/bin/landrun`, verifies `landrun --version`, and runs a small sandbox smoke before installing the Python environment.
- `./kb setup` always uses `uv` and provisions a Python 3.10 environment under `./.venv` by default.
- When `./.venv` already exists, `./kb setup` removes and recreates it non-interactively before reinstalling packages.
- For safety, `--venv-dir` only auto-replaces the repo-managed `./.venv` or an existing directory that already looks like a virtualenv; it refuses to delete arbitrary existing directories.
- `./kb setup` defaults `uv` to `UV_LINK_MODE=copy`, which avoids noisy hardlink fallback warnings on NFS or cross-filesystem setups.
- `./kb run` and `./kb range` require `--hardware-name` unless `HARDWARE_NAME` is already set in the environment.
- `./kb submit` requires `--partition` and `--hardware-name`, plus one of `--problem-id`, `--problem-ids`, or `--start-problem-id/--end-problem-id`.
- When the active submit command is `ybatch`, `./kb submit` also requires `--ybatch-resource` or `KB_YBATCH_RESOURCE`.
- When `uv` is missing, `./kb setup` prints the official install command plus the installation docs URL and exits.
- KernelBench upstream currently publishes `requires-python = "==3.10.*"` in its `pyproject.toml`, so the supported setup path today is still Python 3.10.x.
- Pass `--gpu-extras` when you want `KernelBench[gpu]`. For compatibility, `INSTALL_KERNELBENCH_GPU_EXTRAS=1` is still honored too.
- `./kb setup` records the selected interpreter in `./.kb-python`, and the launchers reuse that exact Python on later runs instead of guessing from a stale `./.venv`.
- Launchers use the repo-built `third_party/bin/landrun` by default. Set `LANDRUN=/path/to/landrun` only when intentionally testing a different Landrun binary.
- When `RUN_NAME` is unset, the launchers generate a unique default like `kernelbench-codex-20260423T081530Z-12345` so reruns do not reuse the same archive tree by accident.

For compatibility, `./scripts/bootstrap_uv.sh` still works and now forwards to `./kb setup`.

## Authenticate the agent tools

Run these commands from the harness repo root.

The harness generates per-problem tool homes under `state/tool_state/` on launch. Authenticate once into repo-root tool dirs, and the harness will mirror only those repo-root auth files into the per-problem tool home for each run. It intentionally does not read `~/.codex` or `~/.claude`.

### Codex

Preferred path: sign in once into repo-root `./.codex/`, using file-backed credentials so the harness can copy `auth.json` into `state/tool_state/<run>/level_<n>/problem_<m>/codex/` on launch.

```bash
mkdir -p .codex
CODEX_HOME="./.codex" codex -c cli_auth_credentials_store=file login --device-auth
CODEX_HOME="./.codex" codex login status
```

Alternative: export an API key instead.

```bash
export OPENAI_API_KEY=...
```

### Claude Code

Preferred path: sign in once into repo-root `./.claude/`. The harness copies only `./.claude/.credentials.json` into `state/tool_state/<run>/level_<n>/problem_<m>/claude/` on launch. If a fresh `claude login` works in your normal shell but the harness does not, refresh the repo-root `./.claude/.credentials.json` file; the harness intentionally ignores `~/.claude`.

```bash
mkdir -p .claude
CLAUDE_CONFIG_DIR="./.claude" claude login
```

Alternatives: export API credentials or an OAuth token.

```bash
export ANTHROPIC_API_KEY=...
# or
export ANTHROPIC_AUTH_TOKEN=...
# or
export CLAUDE_CODE_OAUTH_TOKEN=...
```

## Command broker smoke test

The active launcher path uses the command broker rather than a file-access MCP server. A lightweight smoke is to prepare one workspace, start `python -m kernel_bench_experiment_agents.command_broker` with a state-local Unix socket, and call the generated workspace wrappers with `PYTHON` and `KBH_COMMAND_SOCKET` exported.

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v4 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=gpt-5.4 \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./kb run --problem-id 1 --hardware-name H100
```

The full smoke path is a real `./kb run`, because that exercises workspace preparation, Landrun, the command broker, trace materialization, and completion handling together.

For a broker-only smoke without launching a model, run `scripts/test_command_broker.sh` with `KERNELBENCH_ROOT` and `HARDWARE_NAME` set.

The helper agents `runner` and `profiler` are generated into each per-problem tool home when the client runtime supports them.

## Most common runs

Run these commands from the harness repo root.

### Run one problem

```bash
./kb run \
  --tool codex \
  --run-name kernelbench-codex-h100-v4 \
  --level 1 \
  --problem-id 1 \
  --model gpt-5.4 \
  --time-budget-minutes 180 \
  --precision bf16 \
  --hardware-name H100
```

### Run one problem with Claude

```bash
./kb run \
  --tool claude \
  --run-name kernelbench-claude-h100-v4 \
  --level 1 \
  --problem-id 1 \
  --model claude-opus-4-7 \
  --time-budget-minutes 180 \
  --precision bf16 \
  --hardware-name H100
```

If you are **not** using the vendored submodule, add:

```bash
--kernelbench-root /path/to/KernelBench
```

### Run a contiguous range

```bash
./kb range \
  --tool codex \
  --run-name kernelbench-codex-h100-v4 \
  --level 1 \
  --start-problem-id 1 \
  --end-problem-id 10 \
  --model gpt-5.4 \
  --time-budget-minutes 180 \
  --precision bf16 \
  --hardware-name H100
```

### Run an explicit problem list

```bash
./kb range \
  --tool claude \
  --run-name kernelbench-claude-h100-v4 \
  --level 1 \
  --problem-ids 1,4,9 \
  --model claude-opus-4-7 \
  --time-budget-minutes 180 \
  --precision bf16 \
  --hardware-name H100
```

### Submit one problem to Slurm

```bash
./kb submit \
  --partition h100 \
  --hardware-name H100 \
  --ybatch-resource h100_1 \
  --tool codex \
  --problem-id 1
```

`./kb submit` uses `ybatch` automatically when that site-local command exists; otherwise it uses `sbatch`.
Use `--dry-run` first when you want to inspect the exact submit command and any generated `ybatch` wrapper without queueing a job.

### Submit a range to Slurm

```bash
./kb submit \
  --partition a100 \
  --hardware-name A100 \
  --ybatch-resource a100_1 \
  --tool codex \
  --start-problem-id 1 \
  --end-problem-id 10 \
  --time 13:00:00
```

Keep the scheduler choice explicit. `./kb submit` does not try to autodetect free hardware or choose GPU fallbacks for you.
On clusters with a site-local `ybatch`, the resource name is still site-specific, so set `--ybatch-resource` or `KB_YBATCH_RESOURCE` yourself.

### Summarize one archived run

```bash
./kb summarize-run --run-name kernelbench-codex-h100-v4
```

This scans only `archive/<run_name>/` and writes `archive/<run_name>/run_summary.json`. Summary beat-rates and best-runtime fields exclude suspicious or otherwise non-counting attempts.

## User knobs you will actually use

These are the main variables worth changing:

- `DATA_ROOT=/path/for/archive-and-state` if you want artifacts somewhere other than `./`
- `TOOL=codex|claude`
- `MODEL=...`
- `RUN_NAME=...`
- `LEVEL=...`
- `PROBLEM_ID=...`
- `START_PROBLEM_ID=...` / `END_PROBLEM_ID=...`
- `PROBLEM_IDS=1,4,9`
- `TIME_BUDGET_MINUTES=...`
- `PRECISION=bf16`
- `KERNELBENCH_ROOT=/path/to/KernelBench` when you are not using `./third_party/KernelBench`
- `HARDWARE_NAME=H100`
- `KERNELBENCH_TIMINGS_DIR=/path/to/results/timing/<hardware>` when you need a non-default timings location
- inherited `CUDA_VISIBLE_DEVICES` when you want to pin visible GPUs from the scheduler or shell

## Where to look after a run

The only durable copy-out root is:

```text
archive/<run_name>/
```

Live workspaces, locks, per-problem tool config/runtime dirs, broker sockets, and build products live under `state/` and are disposable once no run is active.

## Lower-level entrypoints

`./kb` is the clean user-facing wrapper. These still exist underneath it:

```bash
./scripts/run_agent_problem.sh
./scripts/run_agent_range.sh
./scripts/run_agent_problem.slurm.sh
./scripts/kbharness --help
```

The shell launchers and workspace wrappers call the repo-local `scripts/kbharness` wrapper, which runs `python -m kernel_bench_experiment_agents.runtime.cli` against the repo source tree. An installed `kbharness` console script is still fine, but it is no longer required just to use this repo.

## Need more detail?

Read `ARCHITECTURE.md` for:

- archive contents and file meanings
- workspace contents and solver boundaries
- per-problem Codex / Claude config layout under `state/tool_state/`
- the direct-workspace plus broker local tool surface
- how profiling, attempts, traces, and summaries are recorded
