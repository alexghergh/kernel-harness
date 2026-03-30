#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_NAME="${RUN_NAME:-kernelbench-codex-h100-v1}"
LEVEL="${LEVEL:-1}"
MAX_PARALLEL_SOLVERS="${MAX_PARALLEL_SOLVERS:-1}"
RUN_STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
RUN_STARTED_EPOCH="$(date +%s)"

if [[ -n "${PROBLEM_IDS:-}" ]]; then
  IFS=',' read -r -a PROBLEM_ID_LIST <<< "${PROBLEM_IDS}"
elif [[ -n "${START_PROBLEM_ID:-}" && -n "${END_PROBLEM_ID:-}" ]]; then
  PROBLEM_ID_LIST=()
  for (( pid=START_PROBLEM_ID; pid<=END_PROBLEM_ID; pid++ )); do
    PROBLEM_ID_LIST+=("${pid}")
  done
else
  echo "Set either PROBLEM_IDS=1,2,3 or START_PROBLEM_ID and END_PROBLEM_ID." >&2
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

report_elapsed_time() {
  local exit_code=$?
  local finished_at elapsed hours minutes seconds

  finished_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  elapsed=$(( $(date +%s) - RUN_STARTED_EPOCH ))
  hours=$(( elapsed / 3600 ))
  minutes=$(( (elapsed % 3600) / 60 ))
  seconds=$(( elapsed % 60 ))

  echo "Range run ${RUN_NAME} finished at ${finished_at} with exit code ${exit_code} after ${hours}h ${minutes}m ${seconds}s" >&2
}

trap report_elapsed_time EXIT

echo "Range run ${RUN_NAME} started at ${RUN_STARTED_AT} (level=${LEVEL}, problems=${#PROBLEM_ID_LIST[@]}, max_parallel_solvers=${MAX_PARALLEL_SOLVERS})" >&2

active_jobs=0
failures=0
for raw_pid in "${PROBLEM_ID_LIST[@]}"; do
  pid="$(trim "${raw_pid}")"
  [[ -n "${pid}" ]] || continue

  echo "Launching level ${LEVEL} problem ${pid} under run ${RUN_NAME}" >&2
  RUN_NAME="${RUN_NAME}" LEVEL="${LEVEL}" PROBLEM_ID="${pid}" \
    "${SCRIPT_DIR}/run_codex_problem.sh" &
  active_jobs=$((active_jobs + 1))

  if (( active_jobs >= MAX_PARALLEL_SOLVERS )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active_jobs=$((active_jobs - 1))
  fi
done

while (( active_jobs > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active_jobs=$((active_jobs - 1))
done

if (( failures > 0 )); then
  echo "${failures} problem runs finished with non-success terminal status or launcher failure." >&2
  exit 1
fi
