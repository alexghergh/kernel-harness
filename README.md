# KernelBench Experiment Agents

Tool-aware scaffold for running autonomous KernelBench optimization experiments with Codex or Claude on a remote GPU node.

This project does five things:

- constrains one agent session to one KernelBench problem at a time
- gives the agent a small local tool surface for timing and profiling kernels
- preserves official KernelBench run naming so downstream analysis stays compatible
- serializes GPU-consuming operations so parallel agent sessions do not overcommit the available GPUs
- generates a workspace-local `HARDWARE.md` from `GPU_NAME` so the solver has explicit architecture limits and official NVIDIA doc links

Start here:

- `SPEC.md` for experiment goals and outputs
- `ARCHITECTURE.md` for the runtime design
- `AGENTS.md` for the repo-local operating rules
- `MEMORY.md` for the evolving project log

The intended top-level entrypoint is a shell launcher under `scripts/` that prepares a single-problem workspace and then runs the selected agent CLI inside that workspace. The generic launch surface is:

- `./scripts/run_agent_problem.sh`
- `./scripts/run_agent_range.sh`
- `./scripts/run_agent_problem.slurm.sh`

Set `TOOL=codex` or `TOOL=claude` to pick the agent.

The checked-in Slurm wrapper is just one convenient cluster profile, not the generic baseline. Its current file defaults are:

- `TOOL=claude`
- `MAX_PARALLEL_SOLVERS=10`
- `TIME_BUDGET_MINUTES=180`

Override those in the file or via `sbatch --export=...` when you want a different run profile.

## setup KernelBench first

This repository itself does not need installation.

What must be set up first is the official KernelBench checkout referenced by `KERNELBENCH_ROOT`.

## full environment setup

Run the full setup on the target GPU node, not on a login node.

1. install the agent CLI you want to use and put it on `PATH`

Codex:

```bash
npm install --prefix "$HOME/.local" @openai/codex
export PATH="$HOME/.local/node_modules/.bin:$PATH"
hash -r
codex --version
```

Claude Code:

```bash
npm install --prefix "$HOME/.local" @anthropic-ai/claude-code
export PATH="$HOME/.local/node_modules/.bin:$PATH"
hash -r
claude --version
```

2. configure authentication for the chosen agent

Codex uses a repo-local saved login:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
export CODEX_HOME="$(pwd)/.codex"
codex login --device-auth
codex login status
```

The launcher reuses that saved repo-local login for every problem. Each Codex run then gets its own derived runtime `CODEX_HOME` under `.runtime/agent_home/...`, so parallel planners do not share mutable Codex state. No `OPENAI_API_KEY` export is needed once this login succeeds.

Claude Code currently uses `ANTHROPIC_API_KEY` directly:

```bash
export ANTHROPIC_API_KEY=...
```

The launcher checks the relevant auth path depending on `TOOL`.

3. create and activate a Python 3.10 environment

```bash
pyenv install 3.10.16
pyenv virtualenv 3.10.16 kernelbench-3.10
pyenv activate kernelbench-3.10
python --version
```

If that environment already exists, just activate it:

```bash
pyenv activate kernelbench-3.10
python --version
```

4. install `uv` in that environment if it is not already available

```bash
python -m pip install --upgrade pip
python -m pip install uv
uv --version
```

5. clone KernelBench and install its dependencies with the official `uv` flow

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git
cd KernelBench
uv sync --extra gpu
```

6. generate the hardware baselines from the KernelBench checkout

First set the `hardware_name` in `scripts/generate_baseline_time.py`, then run:

```bash
cd /path/to/KernelBench
uv run python scripts/generate_baseline_time.py
```

That produces:

- `results/timing/<HARDWARE_NAME>/baseline_time_torch.json`
- `results/timing/<HARDWARE_NAME>/baseline_time_torch_compile_inductor_default.json`

Important precision note:

- this harness evaluates candidate kernels on the `fp32` path by default
- the current `KernelBench/scripts/generate_baseline_time.py` in this checkout hardcodes `bf16` when generating baseline JSONs
- if you want apples-to-apples baseline comparison, either regenerate those baseline JSONs at `fp32` or point `EAGER_BASELINE_FILE` / `COMPILE_BASELINE_FILE` at an older `fp32` baseline checkpoint
- the harness does not infer baseline precision from those JSONs; it simply reads them as external targets

7. export the runtime paths for this harness

```bash
export KERNELBENCH_ROOT=/home/alexghergh/KernelBench
export KERNELBENCH_PYTHON="${KERNELBENCH_ROOT}/.venv/bin/python"
export EAGER_BASELINE_FILE="${KERNELBENCH_ROOT}/results/timing/H100_tsubame/baseline_time_torch.json"
export COMPILE_BASELINE_FILE="${KERNELBENCH_ROOT}/results/timing/H100_tsubame/baseline_time_torch_compile_inductor_default.json"
export NUM_GPUS=1
export GPU_NAME=H100
```

`GPU_NAME` is required and must match a supported family alias. The current static catalog supports `H100`, `A100`, `L40S`, and `B200` families, with common aliases such as `H100 NVL`, `A100 80GB`, `RTX 6000 Ada`, `GB200`, `H200`, and `GH200`. Unknown names fail fast during workspace preparation.

8. run a smoke-test problem from this repository

Generic launcher:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v2 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=gpt-5-codex \
TIME_BUDGET_MINUTES=720 \
./scripts/run_agent_problem.sh
```

Claude example:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
export ANTHROPIC_API_KEY=...
TOOL=claude \
RUN_NAME=kernelbench-claude-h100-v2 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=opus \
TIME_BUDGET_MINUTES=720 \
./scripts/run_agent_problem.sh
```

KernelBench currently ships a `pyproject.toml` and supports an editable install path, but the official repo workflow is `uv`-based. The recommended path is:

```bash
export KERNELBENCH_ROOT=/path/to/KernelBench
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
./scripts/setup_kernelbench_env.sh uv
```

That runs:

- `uv sync --extra gpu` inside `${KERNELBENCH_ROOT}`
- then uses `${KERNELBENCH_ROOT}/.venv/bin/python` as the runtime interpreter

Fallback path if you do not want `uv`:

```bash
export KERNELBENCH_ROOT=/path/to/KernelBench
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
./scripts/setup_kernelbench_env.sh pip
```

That creates a virtualenv and runs:

- `pip install -e ".[gpu]"`

inside the official KernelBench repo.

KernelBench currently declares `requires-python = "==3.10.*"`, so use Python 3.10 for real experiment runs even though this harness package itself is looser about Python versions.

## how to start a run

The default entrypoint is `scripts/run_agent_problem.sh`, which:

- prepares a single-problem workspace under `.runtime/workspaces/...` inside this repo by default
- writes a workspace-local `SPEC.md`, `AGENTS.md`, `HARDWARE.md`, generated status files, a fixed-scaffold `candidate_model_new.py`, and local wrapper scripts into that workspace
- checks the selected agent CLI and auth path (`.codex/` login for Codex, `ANTHROPIC_API_KEY` for Claude)
- creates isolated per-problem runtime state under `.runtime/agent_home/...` when the selected tool needs it
- keeps the helper import path on the wrapper side instead of exposing the whole harness to the solver by default
- enforces one active solver per `(run_name, level, problem_id)` with a per-problem session lock
- enforces the corrected per-problem budget in the launcher and records `budget_exhausted` if the agent does not finish first
- captures the raw agent event stream for the problem
- launches exactly one agent session on exactly that problem

The launcher reads its run settings from environment variables such as `TOOL`, `RUN_NAME`, `LEVEL`, `PROBLEM_ID`, `MODEL`, `NUM_GPUS`, `GPU_NAME`, `WORKSPACE_ROOT`, `KERNELBENCH_PYTHON`, `EAGER_BASELINE_FILE`, and `COMPILE_BASELINE_FILE`.

Example:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
export KERNELBENCH_ROOT=/path/to/KernelBench
export KERNELBENCH_PYTHON="${KERNELBENCH_ROOT}/.venv/bin/python"
export EAGER_BASELINE_FILE="${KERNELBENCH_ROOT}/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch.json"
export COMPILE_BASELINE_FILE="${KERNELBENCH_ROOT}/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch_compile_inductor_default.json"
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v2 \
LEVEL=1 \
PROBLEM_ID=23 \
MODEL=gpt-5-codex \
NUM_GPUS=1 \
GPU_NAME=H100 \
TIME_BUDGET_MINUTES=720 \
./scripts/run_agent_problem.sh
```

Default time budget is 12 hours per problem for the generic `run_agent_problem.sh` / `run_agent_range.sh` launchers unless you override `TIME_BUDGET_MINUTES`.
The checked-in Slurm wrapper currently overrides that with its own `TIME_BUDGET_MINUTES=180` default.
Recorded GPU lock wait is excluded from the remaining budget shown to the solver and from the launcher-side stop check.
Candidate evaluation currently uses the `fp32` KernelBench path unless you explicitly override the precision in the helper CLI.

Keep `EAGER_BASELINE_FILE` and `COMPILE_BASELINE_FILE` exported for later single-problem and range invocations.

## how to run subsets on different days

Runs accumulate naturally as long as you reuse the same `RUN_NAME`.

Single problems:

```bash
TOOL=codex RUN_NAME=kernelbench-codex-h100-v2 LEVEL=1 PROBLEM_ID=23 ./scripts/run_agent_problem.sh
TOOL=codex RUN_NAME=kernelbench-codex-h100-v2 LEVEL=1 PROBLEM_ID=24 ./scripts/run_agent_problem.sh
```

Ranges:

```bash
TOOL=codex RUN_NAME=kernelbench-codex-h100-v2 LEVEL=1 START_PROBLEM_ID=0 END_PROBLEM_ID=20 ./scripts/run_agent_range.sh
TOOL=codex RUN_NAME=kernelbench-codex-h100-v2 LEVEL=2 START_PROBLEM_ID=50 END_PROBLEM_ID=100 ./scripts/run_agent_range.sh
```

Explicit lists:

```bash
TOOL=codex RUN_NAME=kernelbench-codex-h100-v2 LEVEL=3 PROBLEM_IDS=1,7,9,42 ./scripts/run_agent_range.sh
```

`run_agent_range.sh` launches one agent session per problem and supports limited planning parallelism via `MAX_PARALLEL_SOLVERS`. GPU work still respects the shared lease count controlled by `NUM_GPUS`. Launching the same `(run_name, level, problem_id)` twice is unsupported and now fails fast through the per-problem solver lock.

## what the solver agent sees

By default, the solver agent is launched from a problem-specific workspace inside this repository under `.runtime/workspaces/...`.
You can still override `WORKSPACE_ROOT` if you want those workspaces elsewhere.

Its intended working set is:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `GOAL_STATUS.md`
- `problem.json`
- `baseline.json`
- `problem_reference.py`
- `candidate_model_new.py`
- `samples/`
- `profiles/`
- `./bin/problem_info.sh`
- `./bin/run_candidate.sh`
- `./bin/profile_ncu.sh`
- `./bin/goal_status.sh`
- `./bin/best_result.sh`
- `./bin/complete_problem.sh`

The maintainer docs in this repository, including `SPEC.md`, `ARCHITECTURE.md`, and `MEMORY.md`, are not part of the solver agent’s normal workflow.
The evaluated path is currently `fp32`: KernelBench generates inputs, casts tensors to `fp32`, runs the reference `Model` and candidate `ModelNew`, and checks correctness there. Internal mixed-precision math is still allowed if the candidate passes that judged `fp32` path.

`SPEC.md` is the fixed target for the solver:

- a short imperative document, not a project brief
- baselines and success condition at the top
- explicit stopping rules as a checklist
- an explicit edit -> run -> check -> profile -> retry loop
- budget guidance that makes it clear many failed attempts are normal and not a reason to stop
- pointers back to `HARDWARE.md` and `AGENTS.md` instead of duplicating all constraints inline

`HARDWARE.md` is the fixed hardware reference for the solver:

- resolved from the configured `GPU_NAME` through a static alias catalog
- architecture family and compute capability
- register, warp, block, and shared-memory limits that matter for tiling decisions
- official `docs.nvidia.com` links for the relevant architecture tuning guide and the CUDA programming guide
- explicit guidance that larger micro-searches and periodic profiling are expected

`AGENTS.md` is the operating contract:

- read `SPEC.md` first and keep referring back to it
- stay inside the problem workspace
- use only the local wrappers
- edit only `candidate_model_new.py` for the evaluated solution, and only inside its marked editable blocks
- do not inspect harness internals, repository code, or wrapper sources to infer hidden evaluator behavior
- do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for solution ideas
- do not run ad hoc Python or shell GPU experiments outside the wrapper commands
- do not stop early
- terminate only through `./bin/complete_problem.sh`
- valid solver-written completion decisions are `beats_both_baselines`, `beats_eager_only`, `beats_compile_only`, `budget_exhausted`, `stalled`, and `harness_failure`
- `failed_to_generate` is reserved for the launcher when Codex exits without writing completion
- long autonomous runs are expected; the solver should use the remaining-time budget in `GOAL_STATUS.md` and keep working until it succeeds or truthfully exhausts the budget
- if substantial budget remains, the solver should not declare `stalled` until it has at least profiled a strong candidate and tried a new branch informed by that evidence
- there is no wrapper-use limit; even simple tests should be done by editing `candidate_model_new.py` and calling the local `.sh` wrappers
- timing and profiling are framed as routine tools, not expensive last resorts
- quick file inspection should use plain read-only commands inside the workspace, not `python -c`, heredoc Python, or `git diff`
- `HARDWARE.md` should be re-read before major strategy changes that affect tile sizes, shared-memory use, register pressure, or tensor-core math choices

`GOAL_STATUS.md` is the re-anchoring status file:

- it starts with an explicit verdict such as `UNRESOLVED — keep working`
- it puts standing orders above the stats so the solver sees the directive first
- it records timing-call and profiler-call counts from the live trace
- it makes the no-profiler gate visible when `stalled` would be invalid
- it points the solver to workspace-local `samples/` and `profiles/` instead of external artifact paths

Taken together, `SPEC.md`, `AGENTS.md`, `GOAL_STATUS.md`, `HARDWARE.md`, `samples/`, `profiles/`, and the per-attempt manifests are the durable runtime state of the experiment. The harness is intentionally designed so future resumable runners can recover progress from those files instead of depending on one uninterrupted model context.

## how timing results are obtained

Inside a problem workspace, timing should happen through:

```bash
./bin/run_candidate.sh
```

That wrapper:

- reserves the next `sample_id` under a per-problem artifact lock
- copies the candidate into the official KernelBench-style path
- leases one GPU slot from the shared GPU pool
- calls KernelBench evaluation code on the leased GPU only
- writes a per-attempt manifest before evaluation starts
- persists the terminal manifest and `history.jsonl` entry even when evaluation fails
- refreshes the workspace `goal_status.json` and `GOAL_STATUS.md`
- rejects candidates that violate the solution contract before GPU evaluation
- prints a post-run reminder to re-read `GOAL_STATUS.md` and `SPEC.md` before the next decision

The evaluated file must define `ModelNew` and uses a fixed scaffold that compiles inline extension code through `torch.utils.cpp_extension.load_inline(...)`.

The harness rejects:

- redefining `Model`, `get_inputs`, or `get_init_inputs`
- `torch.compile`
- Triton
- environment-variable mutation
- torch backend flag mutation
- `register_buffer`
- pure PyTorch matmul shortcuts
- `out=` output-buffer reuse tricks
- vendor-library wrappers or shortcuts such as cuBLAS, cuBLASLt, CUTLASS, ATen compute helpers, and similar library-dispatch paths
- inspection of generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels as reverse-engineering aids
- any edit outside the marked editable blocks in `candidate_model_new.py`

The direct underlying helper command is:

```bash
${KERNELBENCH_PYTHON} -m kernel_bench_experiment_agents.cli run-candidate ...
```

That direct command is maintainer-facing. Solver agents should stick to `./bin/run_candidate.sh`.

Official evaluated kernel artifacts are written to:

- `runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py`

Per-attempt local manifests are written to:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/sample_<sample_id>.json`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/history.jsonl`

Failed compile or runtime attempts are kept in those same manifests and history entries.

The current measured objective status is available through:

```bash
./bin/goal_status.sh
```

That command refreshes and prints the measured best runtime, the two baselines, and whether the current best correct sample beats eager, compile, or both.
It also records:

- `num_timing_runs`
- `num_profile_runs`

If a timing or profiling command cannot obtain a GPU lease within the configured wait window, it fails explicitly instead of waiting forever. The default lease timeout is 1800 seconds and can be changed with:

```bash
export KBE_GPU_LEASE_MAX_WAIT_SECONDS=1800
```

## how profiling results are obtained

Inside a problem workspace, profiling should happen through:

```bash
./bin/profile_ncu.sh
```

That wrapper also leases a GPU slot before running `ncu`, so profiling and timing share the same global GPU-capacity limit.
The profiling call is only considered successful if it also produces readable text artifacts for the solver.

Profiler outputs are written under:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/ncu/`

The solver should read the generated text artifacts only:

- `*.details.txt`
- `*.raw.csv`

If those text exports are missing or empty, the profiling command is treated as failed.

## how the solver stops

The solver is not supposed to casually hand control back.

The intended contract is:

- keep working autonomously inside the problem workspace
- aim to beat both baselines
- if that happens, stop through:

```bash
./bin/complete_problem.sh --decision beats_both_baselines --summary "..."
```

- if the budget is exhausted, the search is stalled, or the harness is broken, stop only through an explicit non-success decision such as:

```bash
./bin/complete_problem.sh --decision beats_eager_only --summary "..."
./bin/complete_problem.sh --decision beats_compile_only --summary "..."
./bin/complete_problem.sh --decision budget_exhausted --summary "..."
./bin/complete_problem.sh --decision stalled --summary "..."
./bin/complete_problem.sh --decision harness_failure --summary "..."
```

`stalled` is intentionally stricter than “I tried several things.” If substantial budget remains, the harness expects the solver to have profiled at least one strong candidate first. A solver-side `stalled` decision without any recorded `./bin/profile_ncu.sh` call is rejected and later materialized as `harness_failure`.

The launcher treats missing completion as `failed_to_generate`. That decision value exists in the CLI for completeness, but it is normally launcher-written rather than solver-written.

Canonical completion artifact:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/completion.json`

If that file is missing when the agent CLI exits, the launcher synthesizes a `failed_to_generate` completion record and the run is counted as failed.

## manual verification of a saved kernel

If the problem workspace still exists, use the workspace-local best mirror directly:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
export PROJECT_ROOT="$(pwd)"
export BEST_KERNEL="${PROJECT_ROOT}/.runtime/workspaces/kernelbench-codex-h100-v2/level_1/problem_1/samples/best_sample.py"
```

That file is the exact winning `ModelNew` source mirrored into the workspace for local inspection and reruns.

### rerun the saved kernel through the harness directly

This reproduces the harness evaluation path without launching Codex again:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
export PROJECT_ROOT="$(pwd)"
export BEST_KERNEL="${PROJECT_ROOT}/.runtime/workspaces/kernelbench-codex-h100-v2/level_1/problem_1/samples/best_sample.py"

PYTHONPATH="${PROJECT_ROOT}/src" \
"${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli run-candidate \
  --candidate "${BEST_KERNEL}" \
  --run-name manual_repro \
  --level 1 \
  --problem-id 1 \
  --dataset-src local \
  --kernelbench-root "${KERNELBENCH_ROOT}" \
  --gpu-id 0 \
  --num-gpu-slots "${NUM_GPUS:-1}"
```

This writes a fresh manifest under:

- `artifacts/manual_repro/level_1/problem_1/sample_<id>.json`

and a fresh saved kernel under:

- `runs/manual_repro/level_1_problem_1_sample_<id>_kernel.py`

### cross-check the same kernel with official KernelBench

The official KernelBench single-sample path is `scripts/run_and_check.py`. Run it from the KernelBench checkout:

```bash
cd "${KERNELBENCH_ROOT}"
"${KERNELBENCH_PYTHON}" scripts/run_and_check.py \
  ref_origin=kernelbench \
  level=1 \
  problem_id=1 \
  kernel_src_path="${BEST_KERNEL}" \
  gpu_arch='["Hopper"]'
```

That prints:

- the custom-kernel runtime
- the eager PyTorch runtime
- the `torch.compile` runtime
- the two speedups

### compare against the recorded hardware baselines

The harness and KernelBench both compare against the baseline JSON files for your hardware. To inspect the stored baseline entries for one problem:

```bash
python - <<'PY'
import json, os

problem_name = "1_Square_matrix_multiplication_.py"
for label, env_name in [
    ("eager", "EAGER_BASELINE_FILE"),
    ("compile", "COMPILE_BASELINE_FILE"),
]:
    path = os.environ[env_name]
    payload = json.load(open(path))
    level_payload = payload["level1"]
    entry = level_payload[problem_name]
    print(label, problem_name, entry["mean"])
PY
```

For a faithful manual cross-check, the three numbers to compare are:

- the runtime from the direct harness `run-candidate` invocation
- the runtime from KernelBench `scripts/run_and_check.py`
- the baseline means from `EAGER_BASELINE_FILE` and `COMPILE_BASELINE_FILE`

If those align closely, the saved sample is behaving consistently outside the autonomous harness loop.

## agent traces

Each problem now stores normalized trace artifacts under:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/events.jsonl`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/conversation.json`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/final_message.txt`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/completion.json`

`events.jsonl` is the raw CLI event stream for that problem. The exact raw event shape depends on `TOOL` (`codex exec --json` for Codex, `claude --print --output-format stream-json` for Claude).

`conversation.json` is a normalized, easier-to-read projection of that raw event stream. It is meant for inspection, not as the canonical source of truth.

`conversation.json` also carries an `audit` section. The audit invalidates a run if the agent:

- manually changes directories or otherwise leaves the problem workspace contract for shell work
- edits files outside `candidate_model_new.py`
- runs commands outside a narrow allowlist of local wrapper calls and simple workspace-local read-only inspection commands
- inspects generated PTX, cubins, Triton output, Inductor output, or similar compiled artifacts

`completion.json` also carries token accounting derived from the raw trace:

- `token_usage.input_tokens`
- `token_usage.cached_input_tokens`
- `token_usage.cache_creation_input_tokens`
- `token_usage.uncached_input_tokens`
- `token_usage.output_tokens`
- `token_usage.turns_completed`

For Claude, `completion.json.token_usage` now means whole-artifact billed usage, not just the main solve session:

- when Claude emits cumulative `result.modelUsage`, the harness uses the maximum cumulative row
- otherwise it falls back to the maximum cumulative `result.usage` row
- `turns_completed` still records the highest observed `num_turns`, because later cumulative bookkeeping rows often reset to `1`
- `uncached_input_tokens` includes both direct input tokens and cache-creation input tokens

In observed Claude traces, this whole-artifact total appears to include `Task` subagent work when that work is reflected in cumulative `result.modelUsage`.

`completion.json` also carries `trace_counts`, including:

- `run_candidate_calls`
- `profile_ncu_calls`
- `goal_status_calls`
- `best_result_calls`
- `complete_problem_calls`
- `spawn_agent_calls`
- `wait_calls`
- `web_search_calls`
- aggregate wrapper, command, and file-change counts

`completion.json` also carries `web_searches`, including the recorded search queries and any surfaced domains from the trace payload.

`completion.json` also carries diagnostic raw-outcome fields such as:

- `raw_best_correct_runtime_ms`
- `raw_beats_eager`
- `raw_beats_compile`
- `raw_beats_both`
- `outside_harness_success`

These fields are advisory only. They do not override audit invalidation, and a run still counts as failed inside the harness when `decision = "harness_failure"` or `success = false`.

That is the per-problem source for later API-cost calculations.

If the trace audit fails, `completion.json` is rewritten to `decision = "harness_failure"` and `success = false`, even if the solver claimed it beat the baselines.

## how to reset a run

To remove the local state for a run before rerunning it:

```bash
cd /home/alex/projects/uni-research/titech/hpc-agent/kernel_bench_experiment_agents
RUN_NAME=kernelbench-claude-h100-v2 ./scripts/clear_run.sh
```

Positional form:

```bash
./scripts/clear_run.sh kernelbench-claude-h100-v2
```

The top-level run directories have different roles:

- `runs/` is durable and worth keeping. It holds the official saved kernel sources and prompts in a KernelBench-like naming scheme.
- `artifacts/` is durable and worth keeping. It holds manifests, `history.jsonl`, completion state, normalized traces, profiler outputs, and goal-status snapshots.
- `build/` is disposable scratch for compilation/build outputs.
- `.runtime/` is disposable live runtime state: workspaces, per-problem agent homes, and lock files.

If you only care about preserving final kernels and analysis/provenance, keep `runs/` and `artifacts/`. `build/` and `.runtime/` can be removed once no run is active.

This removes:

- `runs/<RUN_NAME>/`
- `artifacts/<RUN_NAME>/`
- `build/<RUN_NAME>/`
- `.runtime/agent_home/<RUN_NAME>/`
- `.runtime/workspaces/<RUN_NAME>/`
- stale run-scoped artifact and solver lock files

## GPU serialization and artifact locking

Thinking can happen in parallel across multiple agent sessions.

GPU work cannot.

The harness uses three separate lock classes:

- `.runtime/solver_locks/` for one active solver per `(run_name, level, problem_id)`
- `.runtime/artifact_locks/` for per-problem `sample_id` reservation and manifest/history commits
- `.runtime/gpu_locks/` for timing and profiling access to the configured GPU slots

## sandbox reality on the cluster

Codex's `workspace-write` sandbox is the preferred mode, but on the current shared cluster it fails because the required Linux namespace setup is blocked. The checked-in `.codex/config.toml` still declares that preferred mode, while the current launcher overrides Codex to `danger-full-access` by default on the cluster.

Claude uses its own permission model instead of the Codex sandbox. The current launcher defaults Claude to `bypassPermissions` on the cluster, so Claude also does not provide hard filesystem isolation there.

That means the hard path boundary is not enforced by the agent CLI itself today. The practical controls are:

- the generated workspace contract
- candidate validation
- post-run trace auditing that invalidates out-of-scope behavior
- hosted web search restricted to `docs.nvidia.com` only

If you need real path isolation, it has to come from the environment around the agent, not from prompt text alone.

Timing and profiling commands acquire a shared GPU lease from:

- `.runtime/gpu_locks/`

The number of simultaneous GPU consumers is controlled by `NUM_GPUS`. If `NUM_GPUS=1`, all timing and profiling commands serialize. If `NUM_GPUS=4`, at most four such commands may run at once across all sessions using this harness.

This works across separate agent processes because the lease uses kernel-managed advisory file locks (`flock`) on shared lock files. That is a valid cross-process serialization mechanism on the same machine as long as all GPU-consuming paths cooperate and use the wrappers.

The GPU lock is not used for `sample_id` allocation. That is handled separately through the per-problem artifact lock so concurrent evaluations on the same problem cannot overwrite each other.

## how to print run statistics

The helper CLI includes a summary command:

```bash
export PYTHONPATH="$(pwd)/src"
${KERNELBENCH_PYTHON} -m kernel_bench_experiment_agents.cli summarize-run \
  --run-name kernelbench-codex-h100-v2 \
  --kernelbench-root "${KERNELBENCH_ROOT}" \
  --level 1 \
  --eager-baseline-file "${KERNELBENCH_ROOT}/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch.json" \
  --compile-baseline-file "${KERNELBENCH_ROOT}/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch_compile_inductor_default.json"
```

That JSON summary includes:

- total problems and total samples
- compiled sample count and compiled sample rate
- correct sample count and correct sample rate
- effective correct sample count and rate after audit invalidation
- problem-level compile hit rate
- problem-level correct hit rate
- audit-invalid problem count
- terminal-decision counts such as `beats_both_baselines`, `budget_exhausted`, `stalled`, and `failed_to_generate`
- pass@k estimates for the requested `k` values
- count and rate of problems whose best correct kernel beats the eager baseline
- count and rate of problems whose best correct kernel beats the chosen compile baseline
- aggregated token totals from per-problem `completion.json` files
- aggregated trace-count totals from per-problem `completion.json` files, including timing calls, profiler calls, subagent spawns, and web-search usage

Per-problem rows in the summary also keep raw best-runtime and raw baseline-beat fields for later manual inspection, but audited harness success remains the source of truth for aggregate win metrics.

Baseline comparison is computed at summary time from the official KernelBench baseline JSON files you pass in. The per-attempt manifests do not currently embed those baseline values.
If a problem run is invalidated by the Codex trace audit, its raw samples remain in the artifact tree, but the summary excludes that problem from effective-correct and baseline-win metrics.

To request different pass@k values:

```bash
${KERNELBENCH_PYTHON} -m kernel_bench_experiment_agents.cli summarize-run \
  --run-name kernelbench-codex-h100-v2 \
  --kernelbench-root "${KERNELBENCH_ROOT}" \
  --pass-k 1,5,20
```

To summarize multiple disjoint batches accumulated under one run name, just reuse the same `RUN_NAME`. The summary command reads the whole accumulated artifact tree.

## what to send back after a cluster run

If you want me to inspect one problem run or make prompt/harness changes, the most useful artifacts are:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/events.jsonl`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/conversation.json`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/completion.json`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/history.jsonl`
- `artifacts/<run_name>/level_<level>/problem_<problem_id>/agent/goal_status.json`
- any relevant `ncu` outputs for the best candidate

For a quick first-pass review, the minimum useful set is:

- `agent/events.jsonl`
- `agent/completion.json`
- `history.jsonl`

If the run did useful profiling, also send:

- `artifacts/<run_name>/level_<level>/problem_<problem_id>/ncu/`

## official KernelBench timing scripts

KernelBench itself provides the main baseline and analysis scripts:

- `scripts/generate_baseline_time.py`
- `scripts/eval_from_generations.py`
- `scripts/benchmark_eval_analysis.py`

Those are the scripts to look at when you want the official baseline timing generation and downstream result analysis.
