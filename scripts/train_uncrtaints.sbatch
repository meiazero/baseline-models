#!/bin/bash
# ---------------------------------------------------------------------------
# train_uncrtaints.sbatch - Train the UnCRtainTS model on a GPU node.
#
# Override repo dir via REPO_DIR:
#   REPO_DIR=/path/to/repo sbatch scripts/train_uncrtaints.sbatch
# ---------------------------------------------------------------------------
#SBATCH -J uncrtaints-brazil
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
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

mkdir -p logs results/uncrtaints

echo "=== Environment ==="
uv run python --version
nvidia-smi

set +e
uv run python -c "import torch; assert torch.cuda.is_available()"
TORCH_STATUS=$?
set -e

if [[ $TORCH_STATUS -ne 0 ]]; then
    echo "ERROR: CUDA not available to torch. Check module load / uv deps."
    exit 1
fi

echo "=== Training UnCRtainTS ==="
uv run python -m train.train_uncrtaints --config configs/uncrtaints.yaml
