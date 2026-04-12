#!/usr/bin/env bash
# Submit all three baseline training jobs to CENAPAD Apollo gpuq.
# Run from baseline-models/scripts/.
set -euo pipefail

cd "$(dirname "$0")"

for job in train_uncrtaints.sbatch train_ctgan.sbatch train_utilise.sbatch; do
    echo "Submitting $job"
    sbatch "$job"
done
