# ADR 0002: Use `archive/` as the canonical durable run root

## Status
Accepted.

## Decision
`archive/` is the only directory that must be copied out to preserve a run.
`state/` is disposable runtime state.

## Consequence
Post-run analysis should read from `archive/<run_name>/...`.
Workspace mirrors and build directories are convenience/runtime state, not the archival source of truth.
