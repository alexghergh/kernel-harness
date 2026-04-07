#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_ROOT="${PROJECT_ROOT}/state"
ARCHIVE_ROOT="${PROJECT_ROOT}/archive"

prepare_runtime_codex_home() {
  local base_home="$1"
  local runtime_home="$2"
  local entry
  local config_path

  rm -rf "${runtime_home}"
  mkdir -p "${runtime_home}"

  for entry in .personality_migration auth.json config.toml version.json yusa_auth.json; do
    if [[ -f "${base_home}/${entry}" ]]; then
      cp -a "${base_home}/${entry}" "${runtime_home}/${entry}"
    fi
  done

  for entry in rules; do
    if [[ -d "${base_home}/${entry}" ]]; then
      cp -a "${base_home}/${entry}" "${runtime_home}/${entry}"
    fi
  done

  config_path="${runtime_home}/config.toml"
  if [[ -f "${config_path}" ]]; then
    python - "${config_path}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if re.search(r"(?m)^project_root_markers\s*=", text):
    text = re.sub(r"(?m)^project_root_markers\s*=.*$", "project_root_markers = []", text)
else:
    if not text.endswith("\n"):
        text += "\n"
    text += "project_root_markers = []\n"
path.write_text(text, encoding="utf-8")
PY
  fi
}

prepare_runtime_claude_project_config() {
  local base_dir="$1"
  local workspace="$2"
  local target_dir="${workspace}/.claude"
  local entry

  mkdir -p "${target_dir}"
  rm -f "${target_dir}/settings.json" "${target_dir}/settings.local.json"

  for entry in settings.json settings.local.json; do
    if [[ -f "${base_dir}/${entry}" ]]; then
      cp -a "${base_dir}/${entry}" "${target_dir}/${entry}"
    fi
  done

}

terminate_agent_pipeline() {
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

TOOL="${TOOL:-codex}"
case "${TOOL}" in
  codex|claude) ;;
  *)
    echo "Unsupported TOOL=${TOOL}. Expected codex or claude." >&2
    exit 1
    ;;
esac

DEFAULT_RUN_NAME="kernelbench-${TOOL}-h100-v2"
DEFAULT_MODEL="gpt-5-codex"
if [[ "${TOOL}" == "claude" ]]; then
  DEFAULT_MODEL="opus"
fi

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-720}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_NAME="${GPU_NAME:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${STATE_ROOT}/workspaces}"
CODEX_SANDBOX_MODE="${CODEX_SANDBOX_MODE:-workspace-write}"
CODEX_SANDBOX_NETWORK_ACCESS="${CODEX_SANDBOX_NETWORK_ACCESS:-false}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-}"
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

export PATH="$(dirname "${KERNELBENCH_PYTHON}"):${PATH}"
KBE_CLI="${KBE_CLI:-kbe}"

if ! command -v "${KBE_CLI}" >/dev/null 2>&1; then
  echo "kbe CLI is not on PATH. Install this repo into the KernelBench environment first (pip install -e .)." >&2
  exit 1
fi

CODEX_BASE_HOME="${PROJECT_ROOT}/.codex"
CLAUDE_PROJECT_DIR="${PROJECT_ROOT}/.claude"
AGENT_RUNTIME_ROOT="${STATE_ROOT}/agent_home"

ARCHIVE_PROBLEM_DIR="${ARCHIVE_ROOT}/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
AGENT_ARTIFACT_DIR="${ARCHIVE_PROBLEM_DIR}/agent"
BUILD_PROBLEM_DIR="${STATE_ROOT}/build/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
SOLVER_LOCK_DIR="${STATE_ROOT}/locks/solver"
PROBLEM_STATE_LOCK_DIR="${STATE_ROOT}/locks/problem_state"
GPU_LOCK_DIR="${STATE_ROOT}/locks/gpu"

mkdir -p \
  "${ARCHIVE_PROBLEM_DIR}" \
  "${AGENT_ARTIFACT_DIR}" \
  "${BUILD_PROBLEM_DIR}" \
  "${STATE_ROOT}" \
  "${AGENT_RUNTIME_ROOT}" \
  "${SOLVER_LOCK_DIR}" \
  "${PROBLEM_STATE_LOCK_DIR}" \
  "${GPU_LOCK_DIR}"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to enforce one active solver per problem." >&2
  exit 1
fi

SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
SOLVER_LOCK_PATH="${SOLVER_LOCK_DIR}/${SAFE_RUN_NAME}_level_${LEVEL}_problem_${PROBLEM_ID}.lock"
AGENT_RUNTIME_HOME="${AGENT_RUNTIME_ROOT}/${SAFE_RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
exec 9>"${SOLVER_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another solver is already active for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID}." >&2
  exit 1
fi

if [[ "${TOOL}" == "codex" ]]; then
  if ! command -v codex >/dev/null 2>&1; then
    echo "codex CLI is not on PATH." >&2
    exit 1
  fi
  if ! CODEX_HOME="${CODEX_BASE_HOME}" codex login status >/dev/null 2>&1; then
    echo "Codex is not logged in for CODEX_HOME=${CODEX_BASE_HOME}." >&2
    echo "Run: CODEX_HOME=\"${CODEX_BASE_HOME}\" codex login --device-auth" >&2
    exit 1
  fi
else
  if ! command -v claude >/dev/null 2>&1; then
    echo "claude CLI is not on PATH." >&2
    exit 1
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY must be exported before launching Claude Code." >&2
    exit 1
  fi
fi

PREP_OUTPUT="$({
  "${KBE_CLI}" prepare-problem-workspace \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --dataset-src "${DATASET_SRC}" \
    --kernelbench-root "${KERNELBENCH_ROOT}" \
    --kernelbench-python "${KERNELBENCH_PYTHON}" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --gpu-name "${GPU_NAME}" \
    --num-gpus "${NUM_GPUS}" \
    --tool "${TOOL}" \
    --model "${MODEL}" \
    --time-budget-minutes "${TIME_BUDGET_MINUTES}" \
    --eager-baseline-file "${EAGER_BASELINE_FILE}" \
    --compile-baseline-file "${COMPILE_BASELINE_FILE}"
})"

WORKSPACE="$({
  PREP_OUTPUT="${PREP_OUTPUT}" "${KERNELBENCH_PYTHON}" - <<'PY'
import json
import os
payload = json.loads(os.environ["PREP_OUTPUT"])
print(payload["workspace"])
PY
})"

INITIAL_PROMPT_PATH="${WORKSPACE}/INITIAL_PROMPT.md"
EVENTS_PATH="${AGENT_ARTIFACT_DIR}/events.jsonl"
FINAL_MESSAGE_PATH="${AGENT_ARTIFACT_DIR}/final_message.txt"
TRACE_PATH="${AGENT_ARTIFACT_DIR}/trace_ir.json"
COMPLETION_PATH="${AGENT_ARTIFACT_DIR}/completion.json"
WORKSPACE_COMPLETION_PATH="${WORKSPACE}/completion.json"
BUDGET_EXHAUSTED_MARKER_PATH="${AGENT_ARTIFACT_DIR}/budget_exhausted_goal_status.json"

if [[ "${TOOL}" == "codex" ]]; then
  prepare_runtime_codex_home "${CODEX_BASE_HOME}" "${AGENT_RUNTIME_HOME}"
  export CODEX_HOME="${AGENT_RUNTIME_HOME}"
  echo "Launching Codex in ${WORKSPACE} with isolated CODEX_HOME=${CODEX_HOME}" >&2
else
  prepare_runtime_claude_project_config "${CLAUDE_PROJECT_DIR}" "${WORKSPACE}"
  rm -rf "${AGENT_RUNTIME_HOME}"
  export CLAUDE_CONFIG_DIR="${AGENT_RUNTIME_HOME}/claude_home"
  mkdir -p "${CLAUDE_CONFIG_DIR}"
  echo "Launching Claude Code in ${WORKSPACE} with isolated CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR}" >&2
fi

rm -f "${FINAL_MESSAGE_PATH}" "${TRACE_PATH}" "${COMPLETION_PATH}" "${WORKSPACE_COMPLETION_PATH}" "${BUDGET_EXHAUSTED_MARKER_PATH}"

refresh_goal_status() {
  "${KBE_CLI}" goal-status \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --workspace "${WORKSPACE}" >/dev/null 2>&1
}

mark_budget_exhausted_if_needed() {
  local status_path="${WORKSPACE}/goal_status.json"
  local exhausted=""

  refresh_goal_status || return 1
  exhausted="$({
    STATUS_PATH="${status_path}" "${KERNELBENCH_PYTHON}" -c '
import json
import os

path = os.environ["STATUS_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("false" if remaining is None else ("true" if float(remaining) <= 0 else "false"))
'
  })"
  if [[ "${exhausted}" == "true" ]]; then
    cp -f "${status_path}" "${BUDGET_EXHAUSTED_MARKER_PATH}"
    return 0
  fi
  return 1
}

watch_budget_limit() {
  local remaining=""
  while kill -0 "${AGENT_PIPE_PID}" 2>/dev/null; do
    if [[ -f "${COMPLETION_PATH}" ]]; then
      return 0
    fi
    if mark_budget_exhausted_if_needed; then
      remaining="$({
        STATUS_PATH="${BUDGET_EXHAUSTED_MARKER_PATH}" "${KERNELBENCH_PYTHON}" -c '
import json
import os

path = os.environ["STATUS_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("" if remaining is None else remaining)
'
      })"
      echo "Budget exhausted for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID} (remaining=${remaining}); stopping ${TOOL}." >&2
      terminate_agent_pipeline "${AGENT_PIPE_PID}"
      return 0
    fi
    sleep "${BUDGET_POLL_SECONDS}"
  done
}

set +e
if [[ "${TOOL}" == "codex" ]]; then
  CODEX_ARGS=(
    -a never
    exec
    --sandbox "${CODEX_SANDBOX_MODE}"
    --cd "${WORKSPACE}"
    --skip-git-repo-check
    --model "${MODEL}"
    --json
  )

  if [[ "${CODEX_SANDBOX_MODE}" == "workspace-write" ]]; then
    CODEX_ARGS+=( -c "sandbox_workspace_write.network_access=${CODEX_SANDBOX_NETWORK_ACCESS}" )
  fi

  (
    codex "${CODEX_ARGS[@]}" \
      --output-last-message "${FINAL_MESSAGE_PATH}" \
      "$(cat "${INITIAL_PROMPT_PATH}")" | tee "${EVENTS_PATH}"
  ) &
else
  CLAUDE_ARGS=(
    -p
    --verbose
    --output-format stream-json
    --no-session-persistence
    --setting-sources project,local
    --model "${MODEL}"
  )
  if [[ -n "${CLAUDE_PERMISSION_MODE}" ]]; then
    CLAUDE_ARGS+=( --permission-mode "${CLAUDE_PERMISSION_MODE}" )
  fi

  (
    cd "${WORKSPACE}" && claude "${CLAUDE_ARGS[@]}" \
      "$(cat "${INITIAL_PROMPT_PATH}")" | tee "${EVENTS_PATH}"
  ) &
fi
AGENT_PIPE_PID=$!
watch_budget_limit &
BUDGET_WATCH_PID=$!
wait "${AGENT_PIPE_PID}"
AGENT_EXIT=$?
kill "${BUDGET_WATCH_PID}" 2>/dev/null || true
wait "${BUDGET_WATCH_PID}" 2>/dev/null || true
set -e

if [[ ! -f "${COMPLETION_PATH}" ]]; then
  mark_budget_exhausted_if_needed >/dev/null 2>&1 || true
  if [[ -f "${BUDGET_EXHAUSTED_MARKER_PATH}" ]]; then
    "${KBE_CLI}" complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state budget_exhausted \
      --summary "launcher stopped ${TOOL} after the corrected remaining budget reached zero without a solver-written completion" \
      --allow-overwrite >/dev/null
  else
    "${KBE_CLI}" complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state failed_to_generate \
      --summary "${TOOL} exited with code ${AGENT_EXIT} without writing completion.json" \
      --allow-overwrite >/dev/null
  fi
fi

if ! "${KBE_CLI}" materialize-agent-trace \
  --tool "${TOOL}" \
  --events-path "${EVENTS_PATH}" \
  --completion-path "${COMPLETION_PATH}" \
  --final-message-path "${FINAL_MESSAGE_PATH}" \
  --output-path "${TRACE_PATH}" \
  --workspace "${WORKSPACE}" >/dev/null; then
  echo "warning: failed to materialize normalized ${TOOL} trace at ${TRACE_PATH}" >&2
fi

readarray -t COMPLETION_STATE < <(
  COMPLETION_PATH="${COMPLETION_PATH}" "${KERNELBENCH_PYTHON}" - <<'PY'
import json
import os
path = os.environ["COMPLETION_PATH"]
payload = json.loads(open(path, "r", encoding="utf-8").read())
print(payload.get("terminal_state", ""))
print("true" if payload.get("success") else "false")
PY
)
TERMINAL_STATE="${COMPLETION_STATE[0]:-}"
SUCCESS="${COMPLETION_STATE[1]:-false}"

echo "Completion state: ${TERMINAL_STATE}" >&2
if [[ "${SUCCESS}" != "true" || "${AGENT_EXIT}" -ne 0 ]]; then
  exit 1
fi
