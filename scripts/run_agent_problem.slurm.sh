#!/usr/bin/env bash
# Thin Slurm entrypoint that forwards the launcher environment to run_agent_range.sh.
#
# Submit this script from the harness repo root.
#
# Required environment:
#   KERNELBENCH_ROOT=/path/to/KernelBench
set -euo pipefail

if [[ ! -f "./pyproject.toml" || ! -d "./src/kernel_bench_experiment_agents" ]]; then
  echo "Submit scripts/run_agent_problem.slurm.sh from the harness repo root." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-.}"
mkdir -p "${DATA_ROOT}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
export DATA_ROOT

module load cuda || true

TOOL="${TOOL:-claude}"
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
PROBLEM_IDS="${PROBLEM_IDS:-}"
START_PROBLEM_ID="${START_PROBLEM_ID:-1}"
END_PROBLEM_ID="${END_PROBLEM_ID:-100}"
MAX_PARALLEL_SOLVERS="${MAX_PARALLEL_SOLVERS:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
HARDWARE_NAME="${HARDWARE_NAME:-H100}"
KERNELBENCH_ROOT="${KERNELBENCH_ROOT:?KERNELBENCH_ROOT must be set}"
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR:-}"
PRECISION="${PRECISION:-bf16}"

DATA_ROOT="${DATA_ROOT}" \
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
HARDWARE_NAME="${HARDWARE_NAME}" \
KERNELBENCH_ROOT="${KERNELBENCH_ROOT}" \
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR}" \
PRECISION="${PRECISION}" \
./scripts/run_agent_range.sh
