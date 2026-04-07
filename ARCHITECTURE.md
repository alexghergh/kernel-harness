# Architecture

This file describes the current harness contract, not historical behavior.

## Audiences

There are two distinct documentation surfaces:

- root docs in this repository for maintainers and operators
- generated docs inside each solver workspace for the solver agent

Root `AGENTS.md` is for maintainers. Workspace `AGENTS.md` is for the solver. The same filename does not imply the same audience.

## System model

A run is keyed by:

```text
(run_name, level, problem_id)
```

For one such key, the harness performs this loop:

1. prepare a fresh workspace
2. render the solver contract into that workspace
3. launch Codex or Claude inside that workspace
4. let the solver iterate through wrapper commands only
5. record attempts, traces, and optional profiler outputs
6. archive the durable record under `archive/<run_name>/...`

## Canonical durable state

The canonical durable root is:

```text
archive/
```

Everything worth copying out after a run lives there.

Per problem, the archive is split into:

- `contract/` — what the solver saw
- `agent/` — raw and normalized agent outputs
- `attempts/` — measured candidate attempts
- `profiles/` — profiler outputs

`archive/` is the source of truth for post-run analysis.

## Disposable runtime state

Live mutable state is kept under:

```text
state/
```

This includes:

- `state/workspaces/`
- `state/agent_home/`
- `state/build/`
- `state/locks/`

The harness may delete and recreate these paths freely. They are not archival.

## Workspace boundary

The solver workspace is intended to be self-contained.

The solver-visible surface is:

- generated docs
- `candidate_model_new.py`
- `problem_reference.py`
- `samples/`
- `profiles/`
- `bin/*.sh`

The solver contract explicitly forbids reading or editing outside that workspace.

## Canonical structured state

Where the same information appears in JSON and Markdown, the JSON form is the structured source of truth and the Markdown form is a rendered solver-facing view.

Examples:

- `workspace_contract.json` drives workspace `AGENTS.md` and the initial prompt
- `hardware.json` and `HARDWARE.md` are rendered from the same hardware payload
- `goal_status.json` is the measured status snapshot and `GOAL_STATUS.md` is its readable view

The intended rule is simple: do not hand-edit both forms. Change the generator or the source payload instead.

The launcher also avoids granting extra directories to the solver runtime. This is important because root maintainer docs and repo internals must not leak into solver sessions.

## Tool/runtime isolation

### Codex

The launcher gives each problem its own runtime `CODEX_HOME` under `state/agent_home/...` and patches the copied Codex config so project discovery stops at the workspace instead of walking up to the repo root.

### Claude

The launcher copies project-level Claude settings into the workspace and gives each problem its own isolated `CLAUDE_CONFIG_DIR` under `state/agent_home/...`.

## Wrapper command surface

The workspace contract currently exposes exactly these wrapper commands:

- `./bin/problem_info.sh`
- `./bin/hardware_info.sh`
- `./bin/run_candidate.sh`
- `./bin/profile_ncu.sh`
- `./bin/goal_status.sh`
- `./bin/best_result.sh`
- `./bin/complete_problem.sh`

The solver should not use ad hoc shell commands to benchmark, profile, inspect hardware, or terminate the run.

## Hardware surface

Hardware facts are frozen into:

- `HARDWARE.md`
- `hardware.json`
- `./bin/hardware_info.sh`

The intended model is that the solver reads these files rather than probing the machine through `nvidia-smi`, `/proc`, `/etc`, or one-off scripts.

## Completion ownership

The solver does not declare measured performance outcomes.

Solver-written terminal states are narrow:

- `done`
- `stalled`
- `harness_failure`

Launcher-only terminal states are:

- `budget_exhausted`
- `failed_to_generate`

The harness computes the measured outcome from recorded attempts and goal status:

- `beats_both`
- `beats_eager_only`
- `beats_compile_only`
- `beats_none`
- `no_correct_candidate`

This keeps measured state inside the harness instead of duplicating it in the solver.

## Goal status

`GOAL_STATUS.md` and `goal_status.json` are live, harness-generated status views.

They are derived from:

- attempt history
- baseline files
- profiler activity
- solver trace counts
- wall-clock budget minus recorded GPU-wait time

The solver should re-read goal status after evaluation, after profiling, and before terminating.

## Attempts and measured evaluation

Every measured attempt goes through `./bin/run_candidate.sh`.

That command:

- validates the candidate source
- reserves a per-problem sample id
- snapshots the candidate into `archive/.../attempts/kernels/`
- evaluates correctness and runtime
- appends to `attempts/history.jsonl`
- refreshes goal status

## Profiling

`./bin/profile_ncu.sh` is the supported profiling path.

The profiler flow:

- reserves a per-problem profile id
- runs Nsight Compute under the harness
- exports summary/details/raw CSV text outputs
- writes archive metadata under `profiles/`
- mirrors the latest text outputs into the live workspace
- refreshes goal status afterward

The text and CSV exports are the first-class solver-facing profiling surface.

## Trace handling and audit

The launcher records raw CLI output to `agent/events.jsonl`.

A later materialization step produces `agent/trace.json` and updates `completion.json` with:

- token usage totals when available
- cost totals when available
- trace counts
- web-search summary
- audit result

Trace audit validates the solver session against the workspace contract, including wrapper-command usage and out-of-scope file edits.

If audit invalidates a run, the final terminal state is rewritten to `harness_failure` while preserving the originally reported state separately.

## Lock model

The harness uses three lock classes:

- `state/locks/solver/` — one active top-level solver per problem
- `state/locks/problem_state/` — serialized mutation of per-problem durable state
- `state/locks/gpu/` — shared GPU slot leasing across problems

## Current refactor status

The current direction is:

- keep archive and workspace contracts stable
- move generic logic out of `cli.py`
- keep Codex/Claude-specific parsing and runtime setup in thin adapters
- keep one canonical source for duplicated helper-agent specifications
