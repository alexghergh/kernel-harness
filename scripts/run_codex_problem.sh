#!/usr/bin/env bash
set -euo pipefail

# example single-problem launcher
# edit the variables below for your environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

prepare_runtime_codex_home() {
  local base_home="$1"
  local runtime_home="$2"
  local entry

  rm -rf "${runtime_home}"
  mkdir -p "${runtime_home}"

  for entry in .personality_migration auth.json config.toml version.json yusa_auth.json; do
    if [[ -f "${base_home}/${entry}" ]]; then
      cp -a "${base_home}/${entry}" "${runtime_home}/${entry}"
    fi
  done

  for entry in agents rules; do
    if [[ -d "${base_home}/${entry}" ]]; then
      cp -a "${base_home}/${entry}" "${runtime_home}/${entry}"
    fi
  done
}

terminate_codex_pipeline() {
  local parent_pid="$1"

  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -P "${parent_pid}" 2>/dev/null || true
  fi
  kill -TERM "${parent_pid}" 2>/dev/null || true
  sleep 5
  if kill -0 "${parent_pid}" 2>/dev/null; then
    if command -v pkill >/dev/null 2>&1; then
      pkill -KILL -P "${parent_pid}" 2>/dev/null || true
    fi
    kill -KILL "${parent_pid}" 2>/dev/null || true
  fi
}

RUN_NAME="${RUN_NAME:-kernelbench-codex-h100-v2}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-gpt-5-codex}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-720}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_NAME="${GPU_NAME:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PROJECT_ROOT}/.runtime/workspaces}"
CODEX_SANDBOX_MODE="${CODEX_SANDBOX_MODE:-danger-full-access}"
CODEX_SANDBOX_NETWORK_ACCESS="${CODEX_SANDBOX_NETWORK_ACCESS:-true}"
KERNELBENCH_PYTHON="${KERNELBENCH_PYTHON:-${KERNELBENCH_ROOT:-}/.venv/bin/python}"
EAGER_BASELINE_FILE="${EAGER_BASELINE_FILE:-}"
COMPILE_BASELINE_FILE="${COMPILE_BASELINE_FILE:-}"
BUDGET_POLL_SECONDS="${BUDGET_POLL_SECONDS:-30}"

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi

if [[ ! -x "${KERNELBENCH_PYTHON}" ]]; then
  echo "KernelBench Python interpreter not found at ${KERNELBENCH_PYTHON}" >&2
  echo "Run ./scripts/setup_kernelbench_env.sh uv in the KernelBench repo or set KERNELBENCH_PYTHON explicitly." >&2
  exit 1
fi

if [[ -z "${EAGER_BASELINE_FILE}" || ! -f "${EAGER_BASELINE_FILE}" ]]; then
  echo "EAGER_BASELINE_FILE must point to the official eager baseline JSON for this hardware." >&2
  exit 1
fi

if [[ -z "${COMPILE_BASELINE_FILE}" || ! -f "${COMPILE_BASELINE_FILE}" ]]; then
  echo "COMPILE_BASELINE_FILE must point to the official torch.compile baseline JSON for this hardware." >&2
  exit 1
fi

CODEX_BASE_HOME="${PROJECT_ROOT}/.codex"
CODEX_RUNTIME_ROOT="${PROJECT_ROOT}/.runtime/codex_home"
PROJECT_PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

RUN_OUTPUT_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
ARTIFACT_PROBLEM_DIR="${PROJECT_ROOT}/artifacts/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
CODEX_ARTIFACT_DIR="${ARTIFACT_PROBLEM_DIR}/codex"
BUILD_PROBLEM_DIR="${PROJECT_ROOT}/build/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"

mkdir -p \
  "${RUN_OUTPUT_DIR}" \
  "${ARTIFACT_PROBLEM_DIR}" \
  "${CODEX_ARTIFACT_DIR}" \
  "${BUILD_PROBLEM_DIR}" \
  "${PROJECT_ROOT}/.runtime" \
  "${CODEX_RUNTIME_ROOT}" \
  "${PROJECT_ROOT}/.runtime/gpu_locks" \
  "${PROJECT_ROOT}/.runtime/artifact_locks" \
  "${PROJECT_ROOT}/.runtime/solver_locks"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to enforce one active solver per problem." >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/.runtime/solver_locks"
SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
SOLVER_LOCK_PATH="${PROJECT_ROOT}/.runtime/solver_locks/${SAFE_RUN_NAME}_level_${LEVEL}_problem_${PROBLEM_ID}.lock"
CODEX_RUNTIME_HOME="${CODEX_RUNTIME_ROOT}/${SAFE_RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
exec 9>"${SOLVER_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another solver is already active for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID}." >&2
  exit 1
fi

PREP_OUTPUT="$(
  PYTHONPATH="${PROJECT_PYTHONPATH}" "${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli prepare-problem-workspace \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --dataset-src "${DATASET_SRC}" \
    --kernelbench-root "${KERNELBENCH_ROOT}" \
    --kernelbench-python "${KERNELBENCH_PYTHON}" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --gpu-name "${GPU_NAME}" \
    --num-gpus "${NUM_GPUS}" \
    --model "${MODEL}" \
    --time-budget-minutes "${TIME_BUDGET_MINUTES}" \
    --eager-baseline-file "${EAGER_BASELINE_FILE}" \
    --compile-baseline-file "${COMPILE_BASELINE_FILE}"
)"

WORKSPACE="$(
  PREP_OUTPUT="${PREP_OUTPUT}" "${KERNELBENCH_PYTHON}" - <<'PY'
import json
import os
payload = json.loads(os.environ["PREP_OUTPUT"])
print(payload["workspace"])
PY
)"

INITIAL_PROMPT_PATH="${WORKSPACE}/INITIAL_PROMPT.md"
EVENTS_PATH="${CODEX_ARTIFACT_DIR}/events.jsonl"
FINAL_MESSAGE_PATH="${CODEX_ARTIFACT_DIR}/final_message.txt"
CONVERSATION_PATH="${CODEX_ARTIFACT_DIR}/conversation.json"
COMPLETION_PATH="${CODEX_ARTIFACT_DIR}/completion.json"
WORKSPACE_COMPLETION_PATH="${WORKSPACE}/completion.json"
BUDGET_EXHAUSTED_MARKER_PATH="${CODEX_ARTIFACT_DIR}/budget_exhausted_goal_status.json"

if ! CODEX_HOME="${CODEX_BASE_HOME}" codex login status >/dev/null 2>&1; then
  echo "Codex is not logged in for CODEX_HOME=${CODEX_BASE_HOME}." >&2
  echo "Run: CODEX_HOME=\"${CODEX_BASE_HOME}\" codex login --device-auth" >&2
  exit 1
fi

prepare_runtime_codex_home "${CODEX_BASE_HOME}" "${CODEX_RUNTIME_HOME}"
export CODEX_HOME="${CODEX_RUNTIME_HOME}"

echo "Launching Codex in ${WORKSPACE} with runtime CODEX_HOME=${CODEX_HOME}" >&2
rm -f "${FINAL_MESSAGE_PATH}" "${CONVERSATION_PATH}" "${COMPLETION_PATH}" "${WORKSPACE_COMPLETION_PATH}" "${BUDGET_EXHAUSTED_MARKER_PATH}"

CODEX_ARGS=(
  -a never
  exec
  --sandbox "${CODEX_SANDBOX_MODE}"
  --cd "${WORKSPACE}"
  --skip-git-repo-check
  --add-dir "${RUN_OUTPUT_DIR}"
  --add-dir "${ARTIFACT_PROBLEM_DIR}"
  --add-dir "${BUILD_PROBLEM_DIR}"
  --add-dir "${PROJECT_ROOT}/.runtime/gpu_locks"
  --add-dir "${PROJECT_ROOT}/.runtime/artifact_locks"
  --add-dir "${PROJECT_ROOT}/.runtime/solver_locks"
  --model "${MODEL}"
  --json
)

if [[ "${CODEX_SANDBOX_MODE}" == "workspace-write" ]]; then
  CODEX_ARGS+=(-c "sandbox_workspace_write.network_access=${CODEX_SANDBOX_NETWORK_ACCESS}")
fi

refresh_goal_status() {
  PYTHONPATH="${PROJECT_PYTHONPATH}" "${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli goal-status \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --workspace "${WORKSPACE}" >/dev/null 2>&1
}

mark_budget_exhausted_if_needed() {
  local status_path="${WORKSPACE}/goal_status.json"
  local exhausted=""

  refresh_goal_status || return 1
  exhausted="$(
    STATUS_PATH="${status_path}" "${KERNELBENCH_PYTHON}" -c '
import json
import os

path = os.environ["STATUS_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("false" if remaining is None else ("true" if float(remaining) <= 0 else "false"))
'
  )"
  if [[ "${exhausted}" == "true" ]]; then
    cp -f "${status_path}" "${BUDGET_EXHAUSTED_MARKER_PATH}"
    return 0
  fi
  return 1
}

watch_budget_limit() {
  local remaining=""
  while kill -0 "${CODEX_PIPE_PID}" 2>/dev/null; do
    if [[ -f "${COMPLETION_PATH}" ]]; then
      return 0
    fi
    if mark_budget_exhausted_if_needed; then
      remaining="$(
        STATUS_PATH="${BUDGET_EXHAUSTED_MARKER_PATH}" "${KERNELBENCH_PYTHON}" -c '
import json
import os

path = os.environ["STATUS_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("" if remaining is None else remaining)
'
      )"
      echo "Budget exhausted for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID} (remaining=${remaining}); stopping Codex." >&2
      terminate_codex_pipeline "${CODEX_PIPE_PID}"
      return 0
    fi
    sleep "${BUDGET_POLL_SECONDS}"
  done
}

set +e
(
  codex "${CODEX_ARGS[@]}" \
    --output-last-message "${FINAL_MESSAGE_PATH}" \
    "$(cat "${INITIAL_PROMPT_PATH}")" | tee "${EVENTS_PATH}"
) &
CODEX_PIPE_PID=$!
watch_budget_limit &
BUDGET_WATCH_PID=$!
wait "${CODEX_PIPE_PID}"
CODEX_EXIT=$?
kill "${BUDGET_WATCH_PID}" 2>/dev/null || true
wait "${BUDGET_WATCH_PID}" 2>/dev/null || true
set -e

if [[ ! -f "${COMPLETION_PATH}" ]]; then
  mark_budget_exhausted_if_needed >/dev/null 2>&1 || true
  if [[ -f "${BUDGET_EXHAUSTED_MARKER_PATH}" ]]; then
    PYTHONPATH="${PROJECT_PYTHONPATH}" "${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --decision budget_exhausted \
      --summary "launcher stopped Codex after the corrected remaining budget reached zero without a solver-written completion" \
      --allow-overwrite >/dev/null
  else
    PYTHONPATH="${PROJECT_PYTHONPATH}" "${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --decision failed_to_generate \
      --summary "codex exited with code ${CODEX_EXIT} without writing completion.json" \
      --allow-overwrite >/dev/null
  fi
fi

if ! PYTHONPATH="${PROJECT_PYTHONPATH}" "${KERNELBENCH_PYTHON}" -m kernel_bench_experiment_agents.cli materialize-codex-trace \
  --events-path "${EVENTS_PATH}" \
  --completion-path "${COMPLETION_PATH}" \
  --output-path "${CONVERSATION_PATH}" \
  --workspace "${WORKSPACE}" >/dev/null; then
  echo "warning: failed to materialize normalized Codex trace at ${CONVERSATION_PATH}" >&2
fi

readarray -t COMPLETION_STATE < <(
  COMPLETION_PATH="${COMPLETION_PATH}" "${KERNELBENCH_PYTHON}" - <<'PY'
import json
import os
path = os.environ["COMPLETION_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
print(payload.get("decision", ""))
print("true" if payload.get("success") else "false")
PY
)
DECISION="${COMPLETION_STATE[0]:-}"
SUCCESS="${COMPLETION_STATE[1]:-false}"

echo "Completion decision: ${DECISION}" >&2
if [[ "${SUCCESS}" != "true" || "${CODEX_EXIT}" -ne 0 ]]; then
  exit 1
fi
