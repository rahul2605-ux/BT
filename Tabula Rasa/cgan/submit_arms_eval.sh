#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:30:00
#SBATCH --job-name=arms_eval
#SBATCH --output=runs/arms_eval_%A_%a_%N.out
#SBATCH --error=runs/arms_eval_%A_%a.err
#SBATCH --gres=gpu:1
# ONE card type: CUDA's random streams differ between GPU models, and the evals are
# paired across defenders only if they draw the same frames (arms_eval.py docstring).
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D6 round 1 (README §3.3o): one (SNR, defender) per array task, arms_eval.TASKS:
#   sbatch --array=0-5 submit_arms_eval.sh
# --mem as the E2/S1 evals (<= 2.2 GB measured, README §3.3k).
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u arms_eval.py --task "${SLURM_ARRAY_TASK_ID}" "$@"
