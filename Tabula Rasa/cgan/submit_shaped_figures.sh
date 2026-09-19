#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:20:00
#SBATCH --job-name=shaped_figs
#SBATCH --output=runs/shapefig_%j.out
#SBATCH --error=runs/shapefig_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# CPU-ish post-processing (needs Sionna for link.py import + test drops). Figures + summary.json.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u shaped_figures.py "$@"
