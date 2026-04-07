# ADR 0003: The harness owns measured outcomes

## Status
Accepted.

## Decision
The solver may report only narrow terminal states such as `done`, `stalled`, and `harness_failure`.
The harness computes measured outcomes like `beats_both` from actual recorded attempts.

## Consequence
Measured performance state is not duplicated in the solver contract.
This reduces drift and keeps final outcome logic in one place.
