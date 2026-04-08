# Maintainer Instructions

This `AGENTS.md` is for people and coding agents working **on this repository**.
It is **not** the solver contract used inside per-problem workspaces. Those workspaces render their own `AGENTS.md`, `SPEC.md`, and status files for the solver.

## Read order

1. Read this file.
2. Read `MEMORY.md`.
3. Read the code that implements the area you are changing.
4. Read `ARCHITECTURE.md` when you need the current system contract or archive/workspace layout.

## Memory policy

`MEMORY.md` is the short rolling handoff for the next maintainer or agent session.

- Update `MEMORY.md` every turn.
- Keep it short.
- Preserve only current direction, locked decisions, open questions, and next steps.
- Compress older detail instead of letting it grow indefinitely.
- Do not turn it into a changelog.

## Commit policy

- Make a git commit at logical boundaries of completed work such as a feature slice, cleanup pass, or issue fix.
- Do **not** ask the user whether to commit.
- Do **not** commit mid-change when the current state is obviously incomplete or broken.
- When you do commit, use a proper subject and body that explain what changed and why.
- Always tell the user after you commit.

## Repo rules

- Treat root docs and workspace docs as different audiences, even when filenames match.
- Keep the solver contract explicit, narrow, and self-contained.
- Do not leak external filesystem paths into the live solver workspace.
- Keep generic harness logic separate from tool-specific adapters.
- Prefer one canonical source over duplicated vendor-specific files.
- Do not hand-edit generated workspace helper-agent specs. Edit the canonical generator and let workspace preparation render them.
- Keep `archive/` as the single durable copy-out root.
- Keep `state/` disposable.
- Do not re-introduce root `SPEC.md`.
- Prefer a small doc surface. Stable design decisions belong in `ARCHITECTURE.md`; temporary direction and reminders belong in `MEMORY.md`.

## Validation before finishing

At minimum, run:

```bash
python -m py_compile src/kernel_bench_experiment_agents/*.py
bash -n scripts/run_agent_problem.sh
bash -n scripts/clear_run.sh
bash -n scripts/run_agent_range.sh
bash -n scripts/run_agent_problem.slurm.sh
```

If you change helper-agent or runtime policy rendering, regenerate a workspace and inspect the generated tool-specific files inside that workspace.

## Current architecture direction

- solver-facing workspaces are self-contained under `state/workspaces/...`
- durable run outputs live under `archive/<run_name>/...`
- launcher and harness own measured outcomes
- solver terminal states are narrow: `done`, `harness_failure`
- Codex and Claude remain equal first-class adapters
- trace audit is a bring-up guard, not the primary sandbox boundary
