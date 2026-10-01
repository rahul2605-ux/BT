#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=final_pulse
#SBATCH --output=runs/pulse_%j.out
#SBATCH --error=runs/pulse_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G

# where the jammers put their energy in time (pulse_plots.py) for env $1.
#   sbatch submit_pulse.sh base
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u pulse_plots.py --env "$1" "${@:2}"
