#!/bin/bash
#SBATCH --job-name=utilise_brazil
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/utilise_%j.out
#SBATCH --error=logs/utilise_%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results/utilise

module purge
module load pytorch/25.03 || true

srun uv run python -m train.train_utilise --config configs/utilise.yaml
