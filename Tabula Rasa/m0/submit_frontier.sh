#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=m0_frontier
#SBATCH --output=runs/frontier_%A_%a.out
#SBATCH --error=runs/frontier_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-7
# One sigma per array task. Possible only since the ITET migration -- the old
# INFK cluster capped us at ONE running GPU job (MaxJobsPU=1).

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
SIGMAS=(0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5)
SG=${SIGMAS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> sigma=$SG on $(hostname -s)"
python -u frontier_m0.py --sigma "$SG"
