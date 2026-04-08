#!/usr/bin/env bash
# Delete one archived run plus its disposable state.
#
# Usage:
#   ./scripts/clear_run.sh <run_name>
# or:
#   RUN_NAME=<run_name> ./scripts/clear_run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_ROOT="${PROJECT_ROOT}/state"
ARCHIVE_ROOT="${PROJECT_ROOT}/archive"

RUN_NAME="${RUN_NAME:-${1:-}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${STATE_ROOT}/workspaces}"

if [[ -z "${RUN_NAME}" ]]; then
  echo "Set RUN_NAME or pass it as the first argument." >&2
  exit 1
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_NAME may contain only ASCII letters, digits, dot, underscore, and hyphen." >&2
  exit 1
fi

SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9._-' '_')"

rm -rf \
  "${ARCHIVE_ROOT}/${RUN_NAME}" \
  "${STATE_ROOT}/build/${RUN_NAME}" \
  "${STATE_ROOT}/agent_home/${SAFE_RUN_NAME}" \
  "${WORKSPACE_ROOT}/${RUN_NAME}"

rm -f \
  "${STATE_ROOT}/locks/solver/${SAFE_RUN_NAME}"_level_*_problem_*.lock \
  "${STATE_ROOT}/locks/problem_state/${SAFE_RUN_NAME}"_level_*_problem_*.lock

echo "Cleared run state for ${RUN_NAME}"
