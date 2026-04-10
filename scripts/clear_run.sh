#!/usr/bin/env bash
# Delete one archived run plus its disposable state.
#
# Usage:
#   PROJECT_ROOT=/path/to/repo ./scripts/clear_run.sh <run_name>
# or:
#   PROJECT_ROOT=/path/to/repo RUN_NAME=<run_name> ./scripts/clear_run.sh
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "PROJECT_ROOT must point at the harness repository root." >&2
  exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
STATE_ROOT="${PROJECT_ROOT}/state"
ARCHIVE_ROOT="${PROJECT_ROOT}/archive"

RUN_NAME="${RUN_NAME:-${1:-}}"

if [[ -z "${RUN_NAME}" ]]; then
  echo "Set RUN_NAME or pass it as the first argument." >&2
  exit 1
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_NAME may contain only ASCII letters, digits, dot, underscore, and hyphen." >&2
  exit 1
fi

rm -rf \
  "${ARCHIVE_ROOT}/${RUN_NAME}" \
  "${STATE_ROOT}/build/${RUN_NAME}" \
  "${STATE_ROOT}/problem_runtime/${RUN_NAME}" \
  "${STATE_ROOT}/agent_home/${RUN_NAME}" \
  "${STATE_ROOT}/claude_home/${RUN_NAME}" \
  "${STATE_ROOT}/workspaces/${RUN_NAME}"

rm -f \
  "${STATE_ROOT}/locks/solver/${RUN_NAME}"_level_*_problem_*.lock \
  "${STATE_ROOT}/locks/problem_state/${RUN_NAME}"_level_*_problem_*.lock

echo "Cleared run state for ${RUN_NAME}"
