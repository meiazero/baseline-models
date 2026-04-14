#!/bin/bash
# ---------------------------------------------------------------------------
# eval_uncrtaints.sbatch - Evaluate trained UnCRtainTS on the Brazil test subset.
# ---------------------------------------------------------------------------
#SBATCH -J eval-uncrtaints
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o evals-logs/slurm_%x_%j.out
#SBATCH -e evals-logs/slurm_%x_%j.err

set -euo pipefail

module load miniconda3/py312_25.1.1
module load cuda/12.6.2

export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p evals-logs

echo "=== Evaluating UnCRtainTS ==="
uv run python -m eval.evaluate --model uncrtaints --config configs/uncrtaints_eval.yaml
