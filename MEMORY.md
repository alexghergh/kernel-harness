# Memory

Short rolling maintainer handoff for the KernelBench harness.

## Current stable baseline

- MCP-backed problem access is now the intended default for both Codex and Claude.
- The real solver surface is intentionally narrow:
  - fixed read-only resources: `AGENTS.md`, `INITIAL_PROMPT.md`, `SPEC.md`, `HARDWARE.md`, `GOAL_STATUS.md`, `problem_reference.py`, `candidate_model_new.py`
  - bounded history reads through `list_workspace_dir(samples|profiles)` and `read_workspace_file(...)`
  - actions through `write_candidate`, `run_candidate`, `profile_ncu`, `goal_status`, `best_result`, `complete_problem`
- Shared tool config lives under `state/config/` and should stay disposable.
- Repo-root auth is the only credential source the harness should mirror:
  - `./.codex/auth.json`
  - `./.claude/.credentials.json`
- Codex should use one shared `state/config/codex/config.toml` with MCP `env_vars`; do not reintroduce per-problem `CODEX_HOME` unless a concrete client bug forces it.

## Fixed in the current pass

- goal-status attempt counts now distinguish `correct`, `incorrect`, and `execution-failed`, so the displayed breakdown sums to total attempts
- Codex config moved back toward the shared static MCP shape rather than per-problem generated homes
- auth mirroring now follows repo-root files only and removes stale mirrored auth when the repo-root source is absent
- tool descriptions were tightened so the MCP surface is clearer about arguments and outputs

## Still on the radar

- verify the shared static Codex `env_vars` path on a real cluster run after the current edits land
- confirm Claude MCP registration/auth behavior stays stable with repo-root `./.claude/.credentials.json` only
- keep reviewing real traces for any client-side surface drift (especially unexpected non-MCP tools)
- destructive workspace preparation ordering and process-group kill semantics are still known cleanup targets

## Review reminders

- after live runs, inspect `archive/.../agent/events.jsonl`, `mcp_ir_events.jsonl`, and `trace_ir.json` together
- manually inspect final kernels for forbidden escapes (vendor libraries, ATen compute helpers, Triton, etc.)
- if Codex or Claude behavior drifts, prefer narrowing the advertised surface and docs before adding more policy layers
