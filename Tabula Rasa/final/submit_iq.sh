#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=final_iq
#SBATCH --output=runs/iq_%j.out
#SBATCH --error=runs/iq_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G

# IQ pictures of the jammers at matched damage (iq_plots.py) for env $1.
#   sbatch submit_iq.sh base
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u iq_plots.py --env "$1" "${@:2}"
