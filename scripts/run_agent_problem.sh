#!/usr/bin/env bash
# Run exactly one solver session for one KernelBench problem.
#
# Required environment:
#   TOOL=codex|claude
#   KERNELBENCH_ROOT=/path/to/KernelBench
#   HARDWARE_NAME=<timings-subdir name, e.g. H100 or H100_tsubame>
#
# Common overrides:
#   RUN_NAME=kernelbench-codex-h100-v3
#   LEVEL=1
#   PROBLEM_ID=1
#   MODEL=gpt-5-codex|opus
#   TIME_BUDGET_MINUTES=180
#   KERNELBENCH_TIMINGS_DIR=/path/to/KernelBench/results/timing/<hardware>
#
# Example:
#   TOOL=codex RUN_NAME=kernelbench-codex-h100-v3 LEVEL=1 PROBLEM_ID=1 \
#   MODEL=gpt-5-codex TIME_BUDGET_MINUTES=180 \
#   KERNELBENCH_ROOT=/path/to/KernelBench HARDWARE_NAME=H100 \
#   ./scripts/run_agent_problem.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_ROOT="${PROJECT_ROOT}/state"
ARCHIVE_ROOT="${PROJECT_ROOT}/archive"
KBHARNESS_CLI="kbharness"

# Prepare a per-problem Codex home so sessions do not share state across runs.
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

  if [[ -d "${base_home}/rules" ]]; then
    cp -a "${base_home}/rules" "${runtime_home}/rules"
  fi
}

# Copy the project-scoped Claude settings into the fresh workspace before launch.
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

# Regenerate the repo-root runtime configs from the shared policy source.
refresh_runtime_configs() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python - "${PROJECT_ROOT}" <<'PY'
from pathlib import Path
import sys
from kernel_bench_experiment_agents.runtime_policy import write_repo_runtime_configs

write_repo_runtime_configs(Path(sys.argv[1]))
PY
}

# Stop the launched agent process tree when the budget watcher fires.
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

# Fail early if the active environment does not expose a required executable.
require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Required command is not on PATH: ${name}" >&2
    exit 1
  fi
}

# Resolve operator-facing launcher settings.
TOOL="${TOOL:-codex}"
case "${TOOL}" in
  codex|claude) ;;
  *)
    echo "Unsupported TOOL=${TOOL}. Expected codex or claude." >&2
    exit 1
    ;;
esac

DEFAULT_RUN_NAME="kernelbench-${TOOL}-h100-v3"
DEFAULT_MODEL="gpt-5-codex"
if [[ "${TOOL}" == "claude" ]]; then
  DEFAULT_MODEL="opus"
fi

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
NUM_GPUS="${NUM_GPUS:-1}"
HARDWARE_NAME="${HARDWARE_NAME:-}"
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${STATE_ROOT}/workspaces}"
CODEX_SANDBOX_MODE="${CODEX_SANDBOX_MODE:-workspace-write}"
CODEX_SANDBOX_NETWORK_ACCESS="${CODEX_SANDBOX_NETWORK_ACCESS:-false}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-}"
BUDGET_POLL_SECONDS="${BUDGET_POLL_SECONDS:-30}"

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi
if [[ -z "${HARDWARE_NAME}" ]]; then
  echo "HARDWARE_NAME must name the KernelBench timings subdirectory to use." >&2
  exit 1
fi

# Validate the active environment and regenerate tool runtime configs.
require_command python
require_command flock
require_command "${KBHARNESS_CLI}"
refresh_runtime_configs

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

SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
SOLVER_LOCK_PATH="${SOLVER_LOCK_DIR}/${SAFE_RUN_NAME}_level_${LEVEL}_problem_${PROBLEM_ID}.lock"
AGENT_RUNTIME_HOME="${AGENT_RUNTIME_ROOT}/${SAFE_RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
exec 9>"${SOLVER_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another solver is already active for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID}." >&2
  exit 1
fi

# Validate authentication before mutating workspace state.
if [[ "${TOOL}" == "codex" ]]; then
  require_command codex
  if ! CODEX_HOME="${CODEX_BASE_HOME}" codex login status >/dev/null 2>&1; then
    echo "Codex is not logged in for CODEX_HOME=${CODEX_BASE_HOME}." >&2
    echo "Run: CODEX_HOME=\"${CODEX_BASE_HOME}\" codex login --device-auth" >&2
    exit 1
  fi
else
  require_command claude
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY must be exported before launching Claude Code." >&2
    exit 1
  fi
fi

# Prepare the fresh workspace and archive contract before the agent starts.
PREP_OUTPUT="$({
  "${KBHARNESS_CLI}" prepare-problem-workspace \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --dataset-src "${DATASET_SRC}" \
    --kernelbench-root "${KERNELBENCH_ROOT}" \
    --timings-dir "${KERNELBENCH_TIMINGS_DIR}" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --hardware-name "${HARDWARE_NAME}" \
    --num-gpus "${NUM_GPUS}" \
    --tool "${TOOL}" \
    --model "${MODEL}" \
    --time-budget-minutes "${TIME_BUDGET_MINUTES}"
})"

WORKSPACE="$({
  PREP_OUTPUT="${PREP_OUTPUT}" python - <<'PY'
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

# Materialize isolated per-problem runtime state for the chosen tool.
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

# Recompute live goal status in place without printing it.
refresh_goal_status() {
  "${KBHARNESS_CLI}" goal-status \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --workspace "${WORKSPACE}" >/dev/null 2>&1
}

# Refresh status and return success only when the corrected remaining budget is zero.
mark_budget_exhausted_if_needed() {
  local status_path="${WORKSPACE}/goal_status.json"
  local exhausted=""

  refresh_goal_status || return 1
  exhausted="$({
    STATUS_PATH="${status_path}" python - <<'PY'
import json
import os
payload = json.loads(open(os.environ["STATUS_PATH"], "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("false" if remaining is None else ("true" if float(remaining) <= 0 else "false"))
PY
  })"
  if [[ "${exhausted}" == "true" ]]; then
    cp -f "${status_path}" "${BUDGET_EXHAUSTED_MARKER_PATH}"
    return 0
  fi
  return 1
}

# Periodically refresh the live budget view and stop the agent when time is exhausted.
watch_budget_limit() {
  local remaining=""
  while kill -0 "${AGENT_PIPE_PID}" 2>/dev/null; do
    if [[ -f "${COMPLETION_PATH}" ]]; then
      return 0
    fi
    if mark_budget_exhausted_if_needed; then
      remaining="$({
        STATUS_PATH="${BUDGET_EXHAUSTED_MARKER_PATH}" python - <<'PY'
import json
import os
payload = json.loads(open(os.environ["STATUS_PATH"], "r", encoding="utf-8").read())
remaining = payload.get("remaining_minutes")
print("" if remaining is None else remaining)
PY
      })"
      echo "Budget exhausted for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID} (remaining=${remaining}); stopping ${TOOL}." >&2
      terminate_agent_pipeline "${AGENT_PIPE_PID}"
      return 0
    fi
    sleep "${BUDGET_POLL_SECONDS}"
  done
}

# Launch the agent CLI, capture its raw event stream, and run the budget watcher in parallel.
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

# If the solver never wrote completion.json, the launcher writes the fallback terminal state.
if [[ ! -f "${COMPLETION_PATH}" ]]; then
  mark_budget_exhausted_if_needed >/dev/null 2>&1 || true
  if [[ -f "${BUDGET_EXHAUSTED_MARKER_PATH}" ]]; then
    "${KBHARNESS_CLI}" complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state budget_exhausted \
      --summary "launcher stopped ${TOOL} after the corrected remaining budget reached zero without a solver-written completion" \
      --allow-overwrite >/dev/null
  else
    "${KBHARNESS_CLI}" complete-problem \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state failed_to_generate \
      --summary "${TOOL} exited with code ${AGENT_EXIT} without writing completion.json" \
      --allow-overwrite >/dev/null
  fi
fi

# Normalize the raw streamed trace into the archive-friendly IR after the session ends.
if ! "${KBHARNESS_CLI}" materialize-agent-trace \
  --tool "${TOOL}" \
  --events-path "${EVENTS_PATH}" \
  --completion-path "${COMPLETION_PATH}" \
  --final-message-path "${FINAL_MESSAGE_PATH}" \
  --output-path "${TRACE_PATH}" \
  --workspace "${WORKSPACE}" >/dev/null; then
  echo "warning: failed to materialize normalized ${TOOL} trace at ${TRACE_PATH}" >&2
fi

readarray -t COMPLETION_STATE < <(
  COMPLETION_PATH="${COMPLETION_PATH}" python - <<'PY'
import json
import os
payload = json.loads(open(os.environ["COMPLETION_PATH"], "r", encoding="utf-8").read())
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
