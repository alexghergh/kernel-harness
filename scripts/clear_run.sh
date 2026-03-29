#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="${RUN_NAME:-${1:-}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PROJECT_ROOT}/.runtime/workspaces}"

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
  "${PROJECT_ROOT}/runs/${RUN_NAME}" \
  "${PROJECT_ROOT}/artifacts/${RUN_NAME}" \
  "${PROJECT_ROOT}/build/${RUN_NAME}" \
  "${WORKSPACE_ROOT}/${RUN_NAME}"

rm -f \
  "${PROJECT_ROOT}/.runtime/solver_locks/${SAFE_RUN_NAME}"_level_*_problem_*.lock \
  "${PROJECT_ROOT}/.runtime/artifact_locks/${SAFE_RUN_NAME}"_level_*_problem_*.lock

echo "Cleared run state for ${RUN_NAME}"
