#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=04:00:00
#SBATCH --job-name=cgan_train_gan
#SBATCH --output=runs/train_gan_%A_%a_%N.out
#SBATCH --error=runs/train_gan_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D2: white-box detector-aware GAN training (README §3.4).
# Task = SLURM_ARRAY_TASK_ID (train_gan.TASKS): sbatch --array=0-6 submit_train_gan.sh
# Smoke first: sbatch submit_train_gan.sh --smoke
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u train_gan.py "$@"
