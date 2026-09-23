#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:20:00
#SBATCH --job-name=gan_figs
#SBATCH --output=runs/ganfig_%j.out
#SBATCH --error=runs/ganfig_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D1/D2 figures + the confirmed stealthy-BER table. Reads the eval JSONs only:
# no Sionna, no GPU, so this schedules on cpu.normal while the GPU nodes are full.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u gan_figures.py "$@"
