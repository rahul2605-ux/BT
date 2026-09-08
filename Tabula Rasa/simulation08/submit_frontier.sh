#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:15:00
#SBATCH --job-name=sim08_suite
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

# Frontier-only re-run against the already-trained channel-valid detector
# (retrain_detector_channel.py already saved run001_best.pt). Produces the
# full-suite (CNN + energy) effectiveness-detectability frontier.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs


python -u frontier_channel.py --detector-model "../artifacts/sim08/detector/run001_best.pt"
