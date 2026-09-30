#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:00:00
#SBATCH --job-name=final_gan
#SBATCH --output=runs/train_gan_%A_%a.out
#SBATCH --error=runs/train_gan_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G

# Chain step 3 (README §3.4): the 8 generators of env $1, one per array task
# (train_gan.ARRAY: control r0-r2, cnn_b10 r0-r2, energy_b10 r0, kurtosis_b10 r0).
#   sbatch --array=0-7 submit_train_gan.sh base
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u train_gan.py --env "$1" --index "${SLURM_ARRAY_TASK_ID}"
