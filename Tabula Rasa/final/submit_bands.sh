#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:40:00
#SBATCH --job-name=final_bands
#SBATCH --output=runs/bands_%j.out
#SBATCH --error=runs/bands_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G

# Chain step 2 (README §3.4): the JSR training band of each target on env $1.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u bands.py --env "$1"
