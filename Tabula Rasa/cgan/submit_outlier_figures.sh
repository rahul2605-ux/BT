#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:20:00
#SBATCH --job-name=cgan_outfig
#SBATCH --output=runs/outfig_%j.out
#SBATCH --error=runs/outfig_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# E2 noise-ablation figures + the summary table (README §3.3g). Reads the ablation
# JSONs only: no Sionna, no GPU, so this schedules on cpu.normal while the GPU nodes
# are full -- and it runs on the login node too.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u outlier_figures.py "$@"
