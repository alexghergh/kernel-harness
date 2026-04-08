#!/usr/bin/env bash
#SBATCH --job-name=kernelbench-harness
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=slurm-out/%x-%j.out
#SBATCH --error=slurm-err/%x-%j.err
# Thin Slurm wrapper around run_agent_range.sh.
#
# Submit from a shell where the intended Python environment is already active,
# or make sure `python`, `kbharness`, and the chosen agent CLI are on PATH in
# the batch environment.
#
# Example:
#   sbatch --export=TOOL=codex,RUN_NAME=kernelbench-codex-h100-v3,LEVEL=1,START_PROBLEM_ID=1,END_PROBLEM_ID=10,MODEL=gpt-5-codex,TIME_BUDGET_MINUTES=180,KERNELBENCH_ROOT=/path/to/KernelBench,HARDWARE_NAME=H100 ./scripts/run_agent_problem.slurm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

module load cuda || true

TOOL="${TOOL:-claude}"
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
PROBLEM_IDS="${PROBLEM_IDS:-}"
START_PROBLEM_ID="${START_PROBLEM_ID:-1}"
END_PROBLEM_ID="${END_PROBLEM_ID:-100}"
MAX_PARALLEL_SOLVERS="${MAX_PARALLEL_SOLVERS:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
HARDWARE_NAME="${HARDWARE_NAME:-H100}"
KERNELBENCH_ROOT="${KERNELBENCH_ROOT:?KERNELBENCH_ROOT must be set}"
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR:-}"

cd "${PROJECT_ROOT}"

if [[ "${TOOL}" == "codex" ]]; then
  if ! command -v codex >/dev/null 2>&1; then
    echo "codex CLI is not on PATH on this node." >&2
    exit 1
  fi
  if ! CODEX_HOME="${PROJECT_ROOT}/.codex" codex login status >/dev/null 2>&1; then
    echo "Codex is not logged in for CODEX_HOME=${PROJECT_ROOT}/.codex." >&2
    echo "Run once before sbatch: CODEX_HOME=\"${PROJECT_ROOT}/.codex\" codex login --device-auth" >&2
    exit 1
  fi
else
  if ! command -v claude >/dev/null 2>&1; then
    echo "claude CLI is not on PATH on this node." >&2
    exit 1
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY must be exported before submitting a Claude Code job." >&2
    exit 1
  fi
fi

TOOL="${TOOL}" \
RUN_NAME="${RUN_NAME}" \
LEVEL="${LEVEL}" \
PROBLEM_IDS="${PROBLEM_IDS}" \
START_PROBLEM_ID="${START_PROBLEM_ID}" \
END_PROBLEM_ID="${END_PROBLEM_ID}" \
MAX_PARALLEL_SOLVERS="${MAX_PARALLEL_SOLVERS}" \
DATASET_SRC="${DATASET_SRC}" \
MODEL="${MODEL}" \
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES}" \
NUM_GPUS="${NUM_GPUS}" \
HARDWARE_NAME="${HARDWARE_NAME}" \
KERNELBENCH_ROOT="${KERNELBENCH_ROOT}" \
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR}" \
"${SCRIPT_DIR}/run_agent_range.sh"
