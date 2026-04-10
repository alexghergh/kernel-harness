#!/usr/bin/env bash
# Run exactly one solver session for one KernelBench problem.
#
# Required environment:
#   PROJECT_ROOT=/abs/path/to/this/repo
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
#   PRECISION=bf16
#   KERNELBENCH_TIMINGS_DIR=/path/to/KernelBench/results/timing/<hardware>
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "PROJECT_ROOT must point at the harness repository root." >&2
  exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
STATE_ROOT="${PROJECT_ROOT}/state"
ARCHIVE_ROOT="${PROJECT_ROOT}/archive"
KBHARNESS_CLI="kbharness"
CODEX_BASE_HOME="${PROJECT_ROOT}/.codex"
CLAUDE_PROJECT_DIR="${PROJECT_ROOT}/.claude"
CLAUDE_BASE_CONFIG_DIR="${CLAUDE_BASE_CONFIG_DIR:-${HOME}/.claude}"
CLAUDE_HOME_STATE_ROOT="${STATE_ROOT}/claude_home"

# Prepare a per-problem Codex home so sessions do not share mutable state.
prepare_runtime_codex_home() {
  local base_home="$1"
  local runtime_home="$2"
  local entry

  rm -rf "${runtime_home}"
  mkdir -p "${runtime_home}"

  for entry in auth.json config.toml; do
    if [[ -f "${base_home}/${entry}" ]]; then
      cp -a "${base_home}/${entry}" "${runtime_home}/${entry}"
    fi
  done
}

# Copy only the project-scoped Claude settings into the fresh workspace.
prepare_runtime_claude_project_config() {
  local base_dir="$1"
  local workspace="$2"
  local target_dir="${workspace}/.claude"

  mkdir -p "${target_dir}"
  rm -f "${target_dir}/settings.json"
  if [[ -f "${base_dir}/settings.json" ]]; then
    cp -a "${base_dir}/settings.json" "${target_dir}/settings.json"
  fi
}

# Seed the isolated Claude config dir with subscription credentials when present.
prepare_runtime_claude_home() {
  local base_dir="$1"
  local runtime_dir="$2"

  rm -rf "${runtime_dir}"
  mkdir -p "${runtime_dir}"

  if [[ -f "${base_dir}/.credentials.json" ]]; then
    cp -a "${base_dir}/.credentials.json" "${runtime_dir}/.credentials.json"
  fi
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

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Required command is not on PATH: ${name}" >&2
    exit 1
  fi
}

visible_gpu_slot_count() {
  local raw="${CUDA_VISIBLE_DEVICES:-}"
  local trimmed entry count=0

  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${SLURM_GPUS_ON_NODE}"
    return
  fi

  if [[ -z "${raw}" ]]; then
    printf '1\n'
    return
  fi

  IFS=',' read -r -a entries <<< "${raw}"
  for entry in "${entries[@]}"; do
    trimmed="${entry#${entry%%[![:space:]]*}}"
    trimmed="${trimmed%${trimmed##*[![:space:]]}}"
    [[ -n "${trimmed}" ]] || continue
    count=$((count + 1))
  done

  if (( count == 0 )); then
    printf '1\n'
  else
    printf '%s\n' "${count}"
  fi
}

# Resolve user-facing launcher settings.
TOOL="${TOOL:-codex}"
case "${TOOL}" in
  codex|claude) ;;
  *)
    echo "Unsupported TOOL=${TOOL}. Expected codex or claude." >&2
    exit 1
    ;;
esac

DEFAULT_MODEL="gpt-5-codex"
if [[ "${TOOL}" == "claude" ]]; then
  DEFAULT_MODEL="opus"
fi

RUN_NAME="${RUN_NAME:-kernelbench-${TOOL}-h100-v3}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
HARDWARE_NAME="${HARDWARE_NAME:-}"
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR:-}"
PRECISION="${PRECISION:-bf16}"
NUM_GPU_SLOTS="$(visible_gpu_slot_count)"
CODEX_SANDBOX_MODE="workspace-write"
CODEX_SANDBOX_NETWORK_ACCESS="false"
BUDGET_POLL_SECONDS=30

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_NAME may contain only ASCII letters, digits, dot, underscore, and hyphen." >&2
  exit 1
fi
if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi
if [[ -z "${HARDWARE_NAME}" ]]; then
  echo "HARDWARE_NAME must name the KernelBench timings subdirectory to use." >&2
  exit 1
fi

require_command python
require_command flock
require_command "${KBHARNESS_CLI}"
refresh_runtime_configs

AGENT_RUNTIME_ROOT="${STATE_ROOT}/agent_home"
ARCHIVE_PROBLEM_DIR="${ARCHIVE_ROOT}/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
AGENT_ARTIFACT_DIR="${ARCHIVE_PROBLEM_DIR}/agent"
SOLVER_LOCK_DIR="${STATE_ROOT}/locks/solver"
PROBLEM_STATE_LOCK_DIR="${STATE_ROOT}/locks/problem_state"
GPU_LOCK_DIR="${STATE_ROOT}/locks/gpu"

mkdir -p \
  "${ARCHIVE_PROBLEM_DIR}" \
  "${AGENT_ARTIFACT_DIR}" \
  "${STATE_ROOT}" \
  "${AGENT_RUNTIME_ROOT}" \
  "${CLAUDE_HOME_STATE_ROOT}" \
  "${SOLVER_LOCK_DIR}" \
  "${PROBLEM_STATE_LOCK_DIR}" \
  "${GPU_LOCK_DIR}"

SOLVER_LOCK_PATH="${SOLVER_LOCK_DIR}/${RUN_NAME}_level_${LEVEL}_problem_${PROBLEM_ID}.lock"
AGENT_RUNTIME_HOME="${AGENT_RUNTIME_ROOT}/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
CLAUDE_RUNTIME_HOME="${CLAUDE_HOME_STATE_ROOT}/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
exec 9>"${SOLVER_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another solver is already active for run=${RUN_NAME} level=${LEVEL} problem=${PROBLEM_ID}." >&2
  exit 1
fi

# Validate authentication before mutating workspace state.
if [[ "${TOOL}" == "codex" ]]; then
  require_command codex
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    :
  elif ! CODEX_HOME="${CODEX_BASE_HOME}" codex login status >/dev/null 2>&1; then
    echo "Codex needs either a repo-local ChatGPT login or OPENAI_API_KEY before launch." >&2
    echo "Preferred: CODEX_HOME=\"${CODEX_BASE_HOME}\" codex login --device-auth" >&2
    echo "Alternative: export OPENAI_API_KEY=..." >&2
    exit 1
  fi
else
  require_command claude
  if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${ANTHROPIC_AUTH_TOKEN:-}" || -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
    :
  elif [[ -f "${CLAUDE_BASE_CONFIG_DIR}/.credentials.json" || -f "${HOME}/.claude.json" ]]; then
    :
  else
    echo "Claude Code needs a subscription login or exported API credentials before launch." >&2
    echo "Preferred: run claude login first so ${CLAUDE_BASE_CONFIG_DIR}/.credentials.json exists." >&2
    echo "Alternatives: export ANTHROPIC_API_KEY=..., ANTHROPIC_AUTH_TOKEN=..., or CLAUDE_CODE_OAUTH_TOKEN=..." >&2
    exit 1
  fi
fi

PREP_OUTPUT="$({
  "${KBHARNESS_CLI}" prepare-problem-workspace \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --dataset-src "${DATASET_SRC}" \
    --kernelbench-root "${KERNELBENCH_ROOT}" \
    --timings-dir "${KERNELBENCH_TIMINGS_DIR}" \
    --hardware-name "${HARDWARE_NAME}" \
    --num-gpus "${NUM_GPU_SLOTS}" \
    --tool "${TOOL}" \
    --model "${MODEL}" \
    --time-budget-minutes "${TIME_BUDGET_MINUTES}" \
    --precision "${PRECISION}"
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

if [[ "${TOOL}" == "codex" ]]; then
  prepare_runtime_codex_home "${CODEX_BASE_HOME}" "${AGENT_RUNTIME_HOME}"
  export CODEX_HOME="${AGENT_RUNTIME_HOME}"
  echo "Launching Codex in ${WORKSPACE} with isolated CODEX_HOME=${CODEX_HOME}" >&2
else
  prepare_runtime_claude_project_config "${CLAUDE_PROJECT_DIR}" "${WORKSPACE}"
  prepare_runtime_claude_home "${CLAUDE_BASE_CONFIG_DIR}" "${CLAUDE_RUNTIME_HOME}"
  export CLAUDE_CONFIG_DIR="${CLAUDE_RUNTIME_HOME}"
  echo "Launching Claude Code in ${WORKSPACE} with isolated CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR}" >&2
fi

rm -f "${FINAL_MESSAGE_PATH}" "${TRACE_PATH}" "${COMPLETION_PATH}" "${WORKSPACE_COMPLETION_PATH}" "${BUDGET_EXHAUSTED_MARKER_PATH}"

refresh_goal_status() {
  "${KBHARNESS_CLI}" goal-status \
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
    --setting-sources project
    --model "${MODEL}"
  )
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
    "${KBHARNESS_CLI}" record-launcher-completion \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state budget_exhausted \
      --summary "launcher stopped ${TOOL} after the corrected remaining budget reached zero without a solver-written completion" \
      --allow-overwrite >/dev/null
  else
    "${KBHARNESS_CLI}" record-launcher-completion \
      --run-name "${RUN_NAME}" \
      --level "${LEVEL}" \
      --problem-id "${PROBLEM_ID}" \
      --workspace "${WORKSPACE}" \
      --state failed_to_generate \
      --summary "${TOOL} exited with code ${AGENT_EXIT} without writing completion.json" \
      --allow-overwrite >/dev/null
  fi
fi

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
print(payload.get("measured_outcome", ""))
PY
)
TERMINAL_STATE="${COMPLETION_STATE[0]:-}"
SUCCESS="${COMPLETION_STATE[1]:-false}"
MEASURED_OUTCOME="${COMPLETION_STATE[2]:-}"

echo "Completion state: ${TERMINAL_STATE} (measured_outcome=${MEASURED_OUTCOME}, success=${SUCCESS})" >&2
case "${TERMINAL_STATE}" in
  harness_failure|failed_to_generate)
    exit 1
    ;;
  done|budget_exhausted)
    exit 0
    ;;
  *)
    echo "Unexpected terminal state: ${TERMINAL_STATE}" >&2
    exit 1
    ;;
esac
