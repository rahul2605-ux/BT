#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=08:00:00
#SBATCH --job-name=cgan_base
#SBATCH --output=runs/base_%A_%a_%N.out
#SBATCH --error=runs/base_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Baselines sweep (README §2.10). K = SLURM_ARRAY_TASK_ID (or --k). K=1 first (it
# builds the test drops and the universal curve); then --array=2-4.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u baselines.py "$@"
