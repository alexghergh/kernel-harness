#!/usr/bin/env bash
#SBATCH --job-name=kernelbench-harness
#YBATCH -r h100_1
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=slurm-out/%x-%j.out
#SBATCH --error=slurm-err/%x-%j.err
# Thin Slurm wrapper around run_agent_range.sh.
#
# Submit from the harness repo root in a shell where the intended Python
# environment is already active, or make sure `python`, `kbharness`, and the
# chosen agent CLI are on PATH in the batch environment.
#
# Example:
#   ybatch --export=TOOL=codex,RUN_NAME=kernelbench-codex-h100-v4,LEVEL=1,START_PROBLEM_ID=1,END_PROBLEM_ID=10,MODEL=gpt-5.4,TIME_BUDGET_MINUTES=180,PRECISION=bf16,KERNELBENCH_ROOT=/path/to/KernelBench,HARDWARE_NAME=H100 ./scripts/run_agent_problem.slurm.sh
#
# Replace `ybatch` with `sbatch` on clusters that use plain Slurm submission.
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

./scripts/run_agent_range.sh
