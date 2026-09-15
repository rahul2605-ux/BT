#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=03:00:00
#SBATCH --job-name=cgan_eval
#SBATCH --output=runs/eval_%j.out
#SBATCH --error=runs/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# C6: evaluate the trained CGAN vs Zhou Fig. 6.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u evaluate.py "$@"
