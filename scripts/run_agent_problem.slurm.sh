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

# Mint the launcher UUID before the banner so the run config printed at the
# top of the slurm output identifies which lock-file slug this job will use.
# Prefer the kernel UUID source / `uuidgen` here because pyenv has not been
# activated yet, so `python` may not be on PATH. run_agent_range.sh inherits
# this value and falls back to python when invoked outside Slurm.
if [[ -z "${KBHARNESS_LAUNCHER_UUID:-}" ]]; then
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    KBHARNESS_LAUNCHER_UUID="$(cat /proc/sys/kernel/random/uuid)"
  elif command -v uuidgen >/dev/null 2>&1; then
    KBHARNESS_LAUNCHER_UUID="$(uuidgen)"
  fi
fi
export KBHARNESS_LAUNCHER_UUID

# Banner: print the run config at the top of the slurm output so `head -20`
# is enough to identify what this job is. Auth credentials are intentionally
# never echoed.
echo "=================================================================="
echo "Slurm job ${SLURM_JOB_ID:-<no-slurm>} on host $(hostname)"
echo "Submitted from: ${SLURM_SUBMIT_DIR:-?}"
echo "Started at:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "------------------------------------------------------------------"
for var in TOOL RUN_NAME LEVEL PROBLEM_IDS START_PROBLEM_ID END_PROBLEM_ID \
           MODEL TIME_BUDGET_MINUTES PRECISION KERNELBENCH_ROOT HARDWARE_NAME \
           MAX_PARALLEL_SOLVERS DATA_ROOT PYENV_VIRTUALENV \
           KBHARNESS_LAUNCHER_UUID; do
  raw="${!var:-}"
  if [[ -z "$raw" ]]; then
    printf '  %-22s (unset)\n' "$var"
  else
    printf '  %-22s %s\n' "$var" "$raw"
  fi
done
echo "=================================================================="

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
