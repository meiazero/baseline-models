#!/bin/bash
# ---------------------------------------------------------------------------
# eval_utilise.sbatch - Evaluate trained U-TILISE on the Brazil test subset.
# ---------------------------------------------------------------------------
#SBATCH -J eval-utilise
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs

echo "=== Evaluating U-TILISE ==="
uv run python -m eval.evaluate --model utilise --config configs/utilise_eval.yaml
