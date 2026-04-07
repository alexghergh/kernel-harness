# ADR 0001: Separate root docs from workspace docs

## Status
Accepted.

## Decision
Root docs are for maintainers and operators.
Generated workspace docs are for the solver agent.

Matching filenames across those two surfaces are allowed when the audiences are different.

## Consequence
Root `AGENTS.md` must never be treated as the solver contract.
Workspace `AGENTS.md` and `SPEC.md` are generated artifacts owned by the harness.
