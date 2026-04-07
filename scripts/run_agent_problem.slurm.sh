#!/usr/bin/env bash
#SBATCH --job-name=kernelbench-agent
#YBATCH -r h100_1
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=slurm-out/%x-%j.out
#SBATCH --error=slurm-err/%x-%j.err

module load cuda

# edit these defaults before submitting with plain `sbatch`
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
KERNELBENCH_ROOT="${KERNELBENCH_ROOT:-/home/alexghergh/KernelBench}"
PYENV_ENV="${PYENV_ENV:-kernelbench-3.10}"
HARDWARE_NAME="${HARDWARE_NAME:-H100_tsubame}"

TOOL="${TOOL:-claude}"
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
PROBLEM_IDS="${PROBLEM_IDS:-}"
START_PROBLEM_ID="${START_PROBLEM_ID:-1}"
END_PROBLEM_ID="${END_PROBLEM_ID:-100}"
MAX_PARALLEL_SOLVERS="${MAX_PARALLEL_SOLVERS:-10}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
GPU_NAME="${GPU_NAME:-H100}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

export PATH="$HOME/.local/node_modules/.bin:$PATH"
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="${PYENV_ROOT}/bin:${PATH}"
export OPENAI_API_KEY
export ANTHROPIC_API_KEY

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is required on the compute node for this wrapper." >&2
  exit 1
fi

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate "${PYENV_ENV}"

export KERNELBENCH_ROOT
export KERNELBENCH_PYTHON="${KERNELBENCH_PYTHON:-${KERNELBENCH_ROOT}/.venv/bin/python}"
export EAGER_BASELINE_FILE="${EAGER_BASELINE_FILE:-${KERNELBENCH_ROOT}/results/timing/${HARDWARE_NAME}/baseline_time_torch.json}"
export COMPILE_BASELINE_FILE="${COMPILE_BASELINE_FILE:-${KERNELBENCH_ROOT}/results/timing/${HARDWARE_NAME}/baseline_time_torch_compile_inductor_default.json}"
export NUM_GPUS
export GPU_NAME

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
GPU_NAME="${GPU_NAME}" \
"${SCRIPT_DIR}/run_agent_range.sh"
