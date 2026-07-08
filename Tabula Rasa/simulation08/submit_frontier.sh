#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:15:00
#SBATCH --job-name=sim08_suite
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

# Frontier-only re-run against the already-trained channel-valid detector
# (retrain_detector_channel.py already saved run001_best.pt). Produces the
# full-suite (CNN + energy) effectiveness-detectability frontier.
source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation08"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

python -u frontier_channel.py --detector-model "../artifacts/sim08/detector/run001_best.pt"
