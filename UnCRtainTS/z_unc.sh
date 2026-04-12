#!/bin/bash
#SBATCH --mail-user=ck696@cornell.edu    # Email for status updates
#SBATCH --mail-type=END                  # Request status by email
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 80:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition
#SBATCH --constraint="gpu-high|gpu-mid"  # GPU constraint
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=60G
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH -N 1                             # Number of nodes
#SBATCH--output=watch_folder/data-%x-%j.log   # Output file name
#SBATCH --requeue                        # Requeue job if it fails

# Setup python path and env
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate H3
# conda activate db
cd /share/hariharan/ck696/allclear/baselines/UnCRtainTS/

python ACinterface_main_0515.py --dataset "SEN12MSCRTS" --experiment_name "repro"
# python ACinterface_main_0515.py --dataset "ALLCLEAR"    --experiment_name "allclear_v1"