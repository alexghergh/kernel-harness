# Architecture

## design summary

This project is agent-first.

The selected agent CLI is the planner. Local scripts and Python helpers are the tools. The Python package does not orchestrate the agent loop.

The flow is:

1. a shell launcher prepares a problem-specific workspace
2. the launcher writes a workspace-local `SPEC.md`, `AGENTS.md`, `HARDWARE.md`, generated status files, and local wrapper scripts into that workspace
3. the launcher starts the selected agent CLI from that workspace
4. the agent reads the local instruction chain, plans work, and uses local commands to iterate
5. the agent optionally spawns narrow subagents for timing or profiling summaries
6. timing and profiling commands acquire shared GPU leases before touching the GPU
7. evaluated kernels are copied into official KernelBench-style run paths
8. per-problem artifact locks reserve `sample_id`s and serialize manifest commits
9. the launcher records the raw agent event stream and requires an explicit completion artifact

## directory layout

The experiment root contains:

- docs: `README.md`, `SPEC.md`, `ARCHITECTURE.md`, `AGENTS.md`, `MEMORY.md`
- Codex config: `.codex/config.toml`
- custom Codex agents: `.codex/agents/*.toml`
- Claude config: `.claude/settings.json`
- custom Claude agents: `.claude/agents/*.md`
- shell entrypoints: `scripts/*.sh`
- helper package: `src/kernel_bench_experiment_agents/`

Runtime-created directories will include:

- `runs/<run_name>/...` for official-style evaluated kernel artifacts
- `.runtime/agent_home/<run_name>/level_<level>/problem_<problem_id>/...` for isolated per-problem runtime state when the selected tool needs it
- `.runtime/workspaces/<run_name>/level_<level>/problem_<problem_id>/...` by default for a single agent session
- `artifacts/<run_name>/...` for manifests, summaries, and profiler outputs
- `build/<run_name>/...` for compilation/build cache outputs
- `.runtime/artifact_locks/` for per-problem sample allocation and manifest commits
- `.runtime/gpu_locks/` for shared GPU lease files
- `.runtime/solver_locks/` for one active solver per `(run_name, level, problem_id)`

Runs are append-only by `run_name`, so separate invocations on different problem subsets can accumulate into the same output tree. The supported execution model is still one active solver per `(run_name, level, problem_id)`.

## instruction layering

Stable maintainer rules belong in the experiment root `AGENTS.md`.

The solver agent should not rely on the repository-root docs. Instead, the launcher generates a workspace-local `SPEC.md`, `AGENTS.md`, and `HARDWARE.md` and launches the selected agent from that workspace. This keeps the agent tightly scoped to a single problem and prevents broad file exploration.

The initial prompt should contain only the run-specific assignment and budget details.

The generated workspace contract is intentionally solution-only and hardware-aware:

- `problem_reference.py` is read-only reference code
- `candidate_model_new.py` is the only evaluated solution file
- `candidate_model_new.py` uses a fixed scaffold; the solver may edit only its marked blocks
- the judged KernelBench path currently uses `fp32` inputs and `fp32` correctness/runtime checks
- internal mixed-precision math is allowed if the candidate still passes that judged `fp32` path
- `HARDWARE.md` is generated from the configured `GPU_NAME` through a static alias catalog and is part of the required solver working set
- `SPEC.md` is generated as a short imperative document with explicit target, stopping rules, and loop steps
- `GOAL_STATUS.md` is generated as a directive-first status file that re-anchors the solver after each measurement
- `samples/` holds workspace-local mirrors of all measured attempts plus `best_sample.py`
- `profiles/` holds workspace-local profiler exports plus `latest.details.txt` and `latest.raw.csv`
- the solver should not read harness source or wrapper internals to reverse-engineer the evaluator
- the solver should not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels for reverse-engineering
- the solver should not edit any file outside `candidate_model_new.py`
- the solver should treat `stalled` as a late terminal state; if substantial budget remains, it is expected to profile a strong candidate and try a new branch before stopping

This workspace layout is also the harness's durable external memory. The intent is that instructions, measured progress, prior samples, and profiler summaries live on disk in a structured form, so later resumable runners can recover from files rather than trusting one uninterrupted agent context.

The root `MEMORY.md` is maintainer-only and is not part of the problem-agent write path.

## agent configuration

The shared harness is tool-neutral, but the checked-in CLI configuration is tool-specific:

- `.codex/config.toml` and `.codex/agents/*.toml` for Codex
- `.claude/settings.json` and `.claude/agents/*.md` for Claude

Both tool layers define:

- hosted web-search policy
- subagent behavior
- preferred sandbox or permission defaults where the CLI supports them

The launcher currently overrides Codex's preferred `workspace-write` setting and defaults to `danger-full-access` on the shared cluster, because the restricted sandbox fails there. Claude uses its own permission mode and is currently launched with `bypassPermissions` on that cluster.

The main session remains the planner. The helper agents are narrow and summary-oriented:

- `runner`: executes timing commands and returns concise summaries
- `profiler`: runs `ncu` and summarizes counters and bottlenecks

The planner is expected to use the full per-problem budget when needed, and should lean on subagents during long searches to keep its own context compact. Critical solver guidance now lives directly in the generated workspace docs rather than in any optional discovery layer.
For Codex, the launcher treats repo-local `.codex/` as the canonical login/config source and derives a fresh runtime `CODEX_HOME` per problem so parallel planners do not share mutable Codex state. For Claude, the launcher copies repo-local `.claude/` settings into the prepared workspace before launch.

## kernelbench integration boundary

The experiment depends on a separate official KernelBench checkout referenced by `KERNELBENCH_ROOT`.

KernelBench setup is environment-first:

- recommended: `uv sync --extra gpu`
- fallback: `pip install -e ".[gpu]"`
- runtime uses an explicit interpreter path, `KERNELBENCH_PYTHON`

Integration rules:

- preserve the official `runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py` naming
- prefer official KernelBench Python entry points when they are easy to call directly
- fall back to invoking official scripts where the Python API is unclear or likely to drift
- keep helper manifests in this repo rather than modifying KernelBench internals
- match KernelBench's solution contract: the candidate defines `ModelNew` with custom CUDA/C++ extension code, while the reference `Model` stays in the original problem file
- narrow the optimization target further than the generic KernelBench prompt: raw custom CUDA is allowed, but vendor-library wrappers and ATen compute helpers are forbidden for this experiment

## local tool surface

The helper package exposes a small command set:

- `prepare-problem-workspace`
- `problem-info`
- `run-candidate`
- `profile-ncu`
- `goal-status`
- `best-result`
- `complete-problem`
- `materialize-agent-trace`
- `summarize-run`

This keeps the tool surface narrow and auditable.

The solver agent should usually call the generated workspace-local wrappers instead of invoking the Python module directly.

The timing wrapper also prints a post-run reminder to re-read `SPEC.md` and `GOAL_STATUS.md`, because the common failure mode is drift immediately after a disappointing timing result.

`run-candidate` and `profile-ncu` now validate the candidate before doing GPU work. The current validator rejects:

- redefining `Model`, `get_inputs`, or `get_init_inputs`
- `torch.compile`
- Triton
- environment-variable mutation
- torch backend flag mutation
- pure PyTorch matmul shortcuts
- `register_buffer`
- `out=` output-buffer reuse tricks
- vendor-library wrappers such as cuBLAS, cuBLASLt, CUTLASS, and ATen compute helpers
- edits outside the fixed editable blocks in `candidate_model_new.py`

## subagent design

Subagents are advisory helpers, not independent planners.

`runner`:

- executes timing/evaluation commands for the current problem
- summarizes correctness, runtime, and failure signals
- does not broaden scope or inspect unrelated problems

`profiler`:

- runs `ncu` for the current candidate
- summarizes dominant kernels, occupancy signals, memory bottlenecks, and obvious optimization clues
- does not change source code

The main agent decides when to call them and how to act on the results.
Long optimization runs should prefer subagents for wrapper execution and profiler summarization instead of carrying all raw output in the planner context.

## GPU lease model

GPU access is one constrained resource, but it is not the only shared resource.

Multiple agent sessions may think in parallel, but timing and profiling must respect the configured GPU capacity.

The helper commands therefore acquire a lease from `.runtime/gpu_locks/` before:

- evaluating a candidate kernel
- running `ncu`

The lease count equals the configured number of available GPUs. This serializes or partially serializes GPU work without forcing planning itself to be single-threaded.

GPU lease acquisition is bounded. By default, timing and profiling commands fail after 1800 seconds of waiting for a slot instead of blocking forever. That timeout can be overridden with `KBE_GPU_LEASE_MAX_WAIT_SECONDS`.

`run-candidate` also acquires a separate per-problem artifact lock before it:

- reserves the next `sample_id`
- writes the official kernel artifact
- writes or updates the per-attempt manifest
- appends the terminal row to `history.jsonl`

The artifact lock is not held while the expensive GPU evaluation itself runs.

The top-level launcher holds a third lock for the lifetime of the whole supervised problem session so the same problem cannot receive multiple solvers at once.

## runtime artifacts

Official-style artifacts:

- evaluated kernel files in `runs/`

Project-local artifacts:

- `problem.json` with problem metadata and assignment
- `baseline.json` with the two resolved runtime targets for this exact problem
- workspace-local `SPEC.md` as the stable optimization target
- workspace-local `HARDWARE.md` as the stable hardware reference derived from `GPU_NAME`
- `goal_status.json` and `GOAL_STATUS.md` as measured status snapshots derived from artifacts, including current timing-call and profiler-call counts
- corrected remaining budget computed from wall-clock elapsed time minus recorded GPU lock wait time
- `history.jsonl` for terminal evaluated attempts, including failures
- `sample_<id>.json` per-attempt manifests that are created before evaluation and finalized afterward
- `candidate_model_new.py` as the only evaluated solution file
- workspace-local `samples/sample_<id>.py` mirrors for every measured attempt
- workspace-local `samples/best_sample.py` and `samples/best_result.json` for the best correct attempt so far
- workspace-local `profiles/profile_<n>.*` mirrors for each profiler run plus `profiles/latest.*` convenience files
- workspace-local `AGENTS.md`
- `bin/` wrapper scripts for timing, profiling, and inspection
- raw and normalized trace artifacts under `artifacts/<run_name>/.../agent/`
- `completion.json` as the required terminal state record for each problem run
- profiler reports and related manifests under `artifacts/<run_name>/...`
- profiler text exports under `artifacts/<run_name>/.../ncu/`, mirrored into the workspace as the solver-readable profiling surface

`completion.json` also stores `token_usage` totals extracted from the raw agent event stream, so run cost can be computed later without re-parsing raw traces.
For Claude, this token payload is defined as whole-artifact billed usage: the harness prefers cumulative `result.modelUsage` when present, otherwise cumulative `result.usage`, and records the maximum cumulative row while keeping `turns_completed` as the highest observed `num_turns`.
`completion.json` also stores `cost_usd` when the raw agent trace reports an explicit cumulative price. Today that means Claude: the harness prefers cumulative `result.total_cost_usd`, falls back to cumulative `result.modelUsage.*.costUSD`, and records the maximum cumulative row. Codex traces do not currently expose an equivalent explicit dollar-cost field.
`completion.json` also stores `trace_counts` derived from the trace, including wrapper usage, profiler usage, subagent spawns, and hosted web-search calls.
`completion.json` also stores `web_searches`, including recorded queries and any surfaced domains from the trace payload.
`completion.json` also stores raw diagnostic outcome fields such as `raw_best_correct_runtime_ms`, `raw_beats_*`, and `outside_harness_success`. These are reference-only fields for later manual inspection; they do not override audit invalidation or harness success semantics.
`conversation.json` stores a normalized aggregate trace plus an `audit` result. If the audit detects out-of-scope shell work, forbidden compiled-artifact inspection, or forbidden file changes, the run is invalidated and `completion.json` is rewritten to `decision = "harness_failure"`.
The same completion-materialization path also enforces a simple stop-policy check: a solver-written `stalled` decision is invalidated if substantial budget remains and no profiler run was recorded.
Separately, the launcher enforces the corrected budget directly: if remaining time reaches zero before the agent writes a completion artifact, the launcher terminates the process and records `budget_exhausted`.

The profiling wrapper now has a stricter contract than before: it must produce readable text exports from the Nsight Compute report, not just a binary report artifact. The solver is expected to read those text exports only.

## baselines

Two baselines are tracked separately:

- eager PyTorch
- `torch.compile`

The launcher resolves those per-problem values during workspace preparation and writes them into `baseline.json` and `SPEC.md`. Aggregate summary still reads the external baseline JSON files for batch analysis.
Aggregate summary also rolls up per-problem `trace_counts`, so total timing calls, profiler calls, subagent usage, and web-search usage are visible at the run level.

## heavy dependencies

Avoid heavy orchestration frameworks.

Planned dependencies:

- standard library
- the local KernelBench environment

Possible small additions if needed:

- `PyYAML` if YAML output materially improves compatibility

Avoid for v2:

- LangChain
- Autogen
- CrewAI
- custom long-running Python agent frameworks

## later extensions

- batched multi-problem launcher with GPU-aware concurrency
- broader result analysis helpers
- broader static GPU catalog coverage when new target families matter
- paper and LaTeX integration

## sandbox limits

Codex's preferred `workspace-write` sandbox can restrict writable roots, but only if the host allows the underlying Linux sandbox setup. On the current shared cluster, that sandbox fails, so the launcher falls back to `danger-full-access`.

Claude does not use the same filesystem sandbox. On the current shared cluster it runs under its own permission mode, currently `bypassPermissions`.

That means this project currently relies on:

- a narrow generated workspace contract
- candidate validation
- trace auditing after the run
- hosted web search restricted to `docs.nvidia.com`

for policy enforcement. If hard filesystem isolation is required, it must come from the surrounding cluster/container environment rather than from the agent CLI configuration alone.
