#!/usr/bin/env bash
#SBATCH --job-name=kernelbench-claude
#YBATCH -r h100_1
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=slurm-out/%x-%j.out
#SBATCH --error=slurm-err/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOOL=claude "${SCRIPT_DIR}/run_agent_problem.slurm.sh"
