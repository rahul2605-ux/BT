#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:00:00
#SBATCH --job-name=final_cnn
#SBATCH --output=runs/train_cnn_%j.out
#SBATCH --error=runs/train_cnn_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
# a CNN retrain peaks at 2.3 GB (sacct MaxRSS, jobs 2274865/2274946; README §C.4)
#SBATCH --mem=6G

# Chain step 1 (README §3.4): retrain Li et al.'s CNN on env $1 + every CFAR threshold.
#   sbatch submit_train_cnn.sh base      (usually via submit_env.sh)
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u train_cnn.py --env "$1"
