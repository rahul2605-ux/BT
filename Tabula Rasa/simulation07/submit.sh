#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=08:00:00
#SBATCH --job-name=sim07_jam
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation07"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

DETECTOR_MODEL="../artifacts/sim06/detector/run002_best.pt"

# To resume a previous run, uncomment and set the checkpoint path:
# RESUME="--resume ../artifacts/sim07/jammer/run001_ckpt.pt"

python -u train_jammer.py --detector-model "$DETECTOR_MODEL" ${RESUME:-}
