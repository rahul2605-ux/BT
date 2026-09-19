#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=08:00:00
#SBATCH --job-name=cgan_shaped
#SBATCH --output=runs/shaped_%A_%a_%N.out
#SBATCH --error=runs/shaped_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# D2a: shaped-noise jammer, CMA-ES against one detector (README §3.4).
# Task = SLURM_ARRAY_TASK_ID (train_shaped.TASKS): sbatch --array=0-9 submit_train_shaped.sh
# Smoke first: sbatch submit_train_shaped.sh --smoke. --mem from sacct MaxRSS of the baselines (2.1 GB).
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u train_shaped.py "$@"
