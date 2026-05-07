#!/usr/bin/env bash
#SBATCH --job-name=kernelbench-harness
#YBATCH -r h100_1
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-out/%x-%j.out
#SBATCH --error=slurm-err/%x-%j.err
# Thin Slurm wrapper around run_agent_range.sh.
#
# Submit from the harness repo root in a shell where the intended Python
# environment is already active, or make sure `python`, `kbharness`, and the
# chosen agent CLI are on PATH in the batch environment.
#
# Example (export the run config in the calling shell so Slurm propagates it
# under `--export=ALL`; some `ybatch` wrappers do not pass `--export=` through):
#   TOOL=codex \
#   RUN_NAME=kernelbench-codex-h100-v5 \
#   LEVEL=1 \
#   START_PROBLEM_ID=1 \
#   END_PROBLEM_ID=10 \
#   MODEL=gpt-5.5 \
#   TIME_BUDGET_MINUTES=180 \
#   PRECISION=bf16 \
#   KERNELBENCH_ROOT=/path/to/KernelBench \
#   HARDWARE_NAME=H100 \
#   ybatch ./scripts/run_agent_problem.slurm.sh
#
# Replace `ybatch` with `sbatch` on clusters that use plain Slurm submission.
set -euo pipefail

if [[ ! -f "./pyproject.toml" || ! -d "./src/kernel_bench_experiment_agents" ]]; then
  echo "Submit scripts/run_agent_problem.slurm.sh from the harness repo root." >&2
  exit 1
fi

# Slurm batch shells do not run interactive shell init, so pyenv shims and the
# active virtualenv from the submitting shell are not visible. Re-activate the
# pyenv virtualenv that has KernelBench and this harness installed. Override
# PYENV_VIRTUALENV from the submitting shell to use a different env.
PYENV_VIRTUALENV="${PYENV_VIRTUALENV:-kernelbench-3.10}"
if [[ -d "${HOME}/.pyenv" ]]; then
  export PYENV_ROOT="${HOME}/.pyenv"
  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)" 2>/dev/null || true
  pyenv activate "${PYENV_VIRTUALENV}"
fi

module load cuda || true

./scripts/run_agent_range.sh
