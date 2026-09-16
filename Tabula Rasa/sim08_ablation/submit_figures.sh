#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:15:00
#SBATCH --job-name=abl08_figs
#SBATCH --output=runs/figures_%j.out
#SBATCH --error=runs/figures_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU-only post-processing of the sweep JSONs: 5 figures + summary.json.
# Submit with --dependency=afterok:<sweep array id>.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u figures.py "$@"
