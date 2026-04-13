# Architecture

This document describes the current system contract for the KernelBench harness.

## Scope

The harness runs one autonomous coding agent on one KernelBench optimization problem at a time, records the run, and stores the durable result under `archive/`.

The harness is responsible for:

- preparing a fresh workspace
- rendering the solver-facing contract into that workspace
- launching Codex or Claude in that workspace
- exposing a narrow local wrapper surface for timing, profiling, status refresh, and completion
- recording attempts, traces, status, completion, and profiler outputs
- aggregating archived results through `summarize-run`

The harness is **not** responsible for installing KernelBench itself. KernelBench setup belongs to the official KernelBench repository and the active environment you choose to run this harness in.

## Documentation audiences

There are two documentation surfaces.

Root docs are for maintainers and users:

- `README.md`
- `ARCHITECTURE.md`
- `AGENTS.md`
- `MEMORY.md`

Generated workspace docs are for the solver agent:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `GOAL_STATUS.md`

The same filename does not imply the same audience.

## High-level flow

For one `(run_name, level, problem_id)` tuple, the harness does this:

1. resolves the problem and baseline information from the KernelBench checkout
2. prepares a fresh self-contained workspace
3. renders solver-facing docs and wrapper scripts into that workspace
4. prepares the shared tool-private config under `state/config/` once at the start of a run
5. launches Codex or Claude with the current working directory set to that workspace
6. lets the solver iterate through wrapper commands
7. records attempts, traces, completion state, and optional profiles
8. writes the durable record under `archive/<run_name>/level_<level>/problem_<problem_id>/`

The solver does not decide measured outcomes. The harness decides, from recorded attempts, whether the best correct solution beat eager PyTorch, `torch.compile`, both, or neither.

## Durable vs disposable state

### Durable: `archive/`

`archive/` is the only directory you should copy out for post-run analysis.

Per problem, the durable record is:

```text
archive/<run_name>/level_<level>/problem_<problem_id>/
  archive_manifest.json
  contract/
  agent/
  attempts/
  profiles/
```

### Disposable: `state/`

`state/` contains live mutable runtime data only:

- `state/workspaces/`
- `state/build/`
- `state/locks/`
- `state/config/codex/`
- `state/config/claude/`

It is safe to delete `state/` only when no run is active.

`DATA_ROOT` controls where `archive/` and `state/` are written. It does **not** decide where repository source files live.

## Shared tool-private config

The workspace is solver-visible. Tool-private config and auth live outside it.

- Codex uses `CODEX_HOME=state/config/codex/`
- Claude uses `CLAUDE_CONFIG_DIR=state/config/claude/`

Those shared tool dirs are where the harness writes:

- generated Codex `config.toml`
- generated Claude `settings.json`
- generated helper-agent definitions for both tools
- tool-managed local state such as auth/session/history files

The harness does not rely on repo-root `.codex/` or `.claude/` config files.

This split is deliberate:

- the workspace should contain only problem files the solver is meant to read or edit
- tool auth/config should not sit inside the solver-visible workspace
- traces are **not** recovered from shared tool history files; each problem captures its own streamed `agent/events.jsonl` directly from the launcher

## Archive contents

### `archive_manifest.json`

Machine-readable description of the problem archive. It explains what is canonical, what is mirrored into the live workspace, and what should be copied out.

### `contract/`

The exact problem contract shown to the solver, plus frozen structured metadata.

Important files:

- `AGENTS.md` — solver instructions and boundaries
- `SPEC.md` — problem goal and success criteria
- `HARDWARE.md` — human-readable hardware facts and guidance
- `INITIAL_PROMPT.md` — the exact initial launch prompt
- `problem_reference.py` — local copy of the reference PyTorch problem code
- `candidate_model_new.py` — initial solver scaffold shown at workspace creation
- `candidate_final.py` — final captured candidate when completion exists
- `problem.json` — machine-readable problem metadata, embedded eager/compile baseline runtimes, budget, and run identifiers
- `hardware.json` — machine-readable hardware facts
- `workspace_contract.json` — machine-readable solver contract
- `provenance.json` — archive-only outward provenance for the source KernelBench checkout and timing files
- `helper_agents/` — archived copies of the generated Codex and Claude helper-agent definitions

The live workspace should not expose outward filesystem paths. That information belongs only in archive-only provenance.

### `agent/`

The agent-side run record.

Important files:

- `events.jsonl` — raw streamed output from the agent CLI for this problem only
- `trace_ir.json` — normalized, mostly-lossless trace representation used by the harness
- `completion.json` — terminal state plus measured outcome and trace-derived metadata
- `goal_status.json` — archived final goal-status snapshot for where the solver ended
- `final_message.txt` — the last assistant-visible model message, when one is available

`final_message.txt` is intentionally a convenience artifact. It is the last assistant-visible text extracted from the raw CLI event stream (or Codex `--output-last-message` when available), so a user can quickly inspect how the run ended without reading the full event log.

### `attempts/`

Measured evaluation results for submitted candidates.

Important files:

- `sample_<id>.json` — one measured attempt record with correctness/runtime/build outcome and archive-relative file references
- `sample_<id>.stdout.txt` — evaluation subprocess stdout for that attempt
- `sample_<id>.stderr.txt` — evaluation subprocess stderr for that attempt
- `kernels/level_<level>_problem_<problem_id>_sample_<id>_kernel.py` — archived candidate source measured for that attempt

### `profiles/`

Nsight Compute profiling artifacts.

Important files:

- `profile_<id>.json` — one profile manifest with attempt linkage and archive-relative file references
- `profile_<id>.summary.txt` — short solver-facing summary of the selected metrics
- `profile_<id>.details.txt` — fuller textual profiler output
- `profile_<id>.stdout.txt` — profiler command stdout
- `profile_<id>.stderr.txt` — profiler command stderr

The harness keeps text-first profiler outputs. It does not archive `.ncu-rep` files.

## Workspace surface

Each solver workspace is fresh, self-contained, and intentionally small.

Solver-visible files:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `GOAL_STATUS.md`
- `goal_status.json`
- `hardware.json`
- `workspace_contract.json`
- `problem.json`
- `problem_reference.py`
- `candidate_model_new.py`
- `samples/`
- `profiles/`
- `bin/*.sh`

Workspace-local `samples/` and `profiles/` are mirrors for solver convenience. They are not the durable source of truth.

Notably absent from the workspace:

- Codex `auth.json`
- Codex `config.toml`
- Claude `settings.json`
- Claude credential files
- tool-private helper-agent config

Those live under `state/config/` instead.

## Canonical JSON vs rendered Markdown

Where both JSON and Markdown exist, the JSON form is the canonical structured source and the Markdown form is a rendered solver-facing view.

Examples:

- `workspace_contract.json` renders workspace `AGENTS.md` and `INITIAL_PROMPT.md`
- `hardware.json` renders `HARDWARE.md`
- `goal_status.json` renders `GOAL_STATUS.md`

This is intentional, not independent duplicated documentation.

## Wrapper command surface

The solver is supposed to work through generated wrapper commands, not ad hoc shell workflows.

The workspace wrapper surface is:

- `./bin/hardware_info.sh`
- `./bin/run_candidate.sh`
- `./bin/profile_ncu.sh`
- `./bin/goal_status.sh`
- `./bin/best_result.sh`
- `./bin/complete_problem.sh`

Semantics:

- `run_candidate.sh` is the only supported measured-evaluation path
- `profile_ncu.sh` is the only supported profiling path
- `goal_status.sh` refreshes live status explicitly
- `complete_problem.sh` is the only valid solver termination path

All wrappers except `complete_problem.sh` are fixed commands with no solver-supplied control flags.

`complete_problem.sh` accepts only:

- `--summary "..."`

## Completion model

Solver-written completion is intentionally narrow:

- the only solver-visible exit path is `./bin/complete_problem.sh --summary "..."`
- the solver does not choose between `done`, `budget_exhausted`, or other launcher-owned outcomes
- the harness records solver completion as `done` and infers the measured outcome from archived attempts

Launcher-owned terminal states are:

- `budget_exhausted`
- `failed_to_generate`

The harness computes the measured outcome from archived attempts:

- `beats_both`
- `beats_eager_only`
- `beats_compile_only`
- `beats_none`
- `no_correct_candidate`

This keeps measured state inside the harness instead of duplicating it in the solver.

## Goal status and the budget watcher

`goal_status.json` and `GOAL_STATUS.md` are live status views.

They are refreshed:

- when wrapper commands update state
- by the launcher budget watcher while the solver is running

The watcher runs periodically so the remaining-time view does not depend only on wrapper usage. Tool calls still trigger immediate corrections after measured work such as timing or profiling.

The budget clock is wall time since workspace creation minus:

- recorded historical GPU wait time from completed run/profile operations
- any currently active live GPU lease wait in progress

Live GPU-queue wait markers live under `state/locks/live_gpu_wait/`.

## Runtime policy and sandbox boundary

The runtime policy should be semantically the same for Codex and Claude, rendered into each tool's native settings surface.

Intended policy:

- workspace-only writes
- workspace-only intended reads
- documentation web access only to `docs.nvidia.com`
- no ad hoc shell networking
- no host probing
- helper agents allowed
- autonomous work, no approval-seeking

Important boundary note:

The external sandbox is the real enforcement layer. The harness itself adds two softer layers:

- fixed wrapper commands and generated local docs
- trace audit as a bring-up/regression check

Trace audit is useful, but it is not the long-term security boundary.

## Codex vs Claude integration notes

The policy target is the same, but the config surfaces are different.

- Codex uses `CODEX_HOME` for config, auth, helper agents, history, and other local state. The harness writes the generated Codex policy directly into `state/config/codex/config.toml`.
- Claude uses `CLAUDE_CONFIG_DIR` for config, credentials, helper agents, and other local state. The harness writes the generated Claude policy directly into `state/config/claude/settings.json`.

In both cases, the current working directory remains the per-problem workspace.

## KernelBench integration surface

The harness relies on a narrow KernelBench integration surface:

- load the reference problem code
- evaluate a candidate for correctness/runtime
- run the profiler path against the problem inputs/model
- read the standard eager and `torch.compile` timing JSONs for the selected hardware

The harness ignores any alternate reference runtime fields returned by KernelBench evaluation for decision-making. The source of truth for baseline targets is the archived baseline information already resolved into the workspace and archive.

## Lock model

The harness uses three lock classes:

- solver lock — one active top-level solver per problem
- problem-state lock — serialized mutation of per-problem durable state
- GPU lock — shared GPU-selector leasing across problems

GPU leasing is based on the visible `CUDA_VISIBLE_DEVICES` view. The harness binds evaluation and profiling subprocesses to one leased selector and then uses logical `cuda:0` inside that isolated view.

## Aggregation

`summarize-run` is a user-facing archive scanner.

Its job is to aggregate archived results for one `run_name` after the fact. It does not participate in solver execution. It exists so the harness has one built-in, archive-only aggregation path instead of pushing every result inspection into ad hoc notebooks.
