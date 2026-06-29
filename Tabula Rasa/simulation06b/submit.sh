#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=04:00:00
#SBATCH --job-name=sim06b_1sc
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation06b"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

DETECTOR_MODEL="../artifacts/sim06/detector/run002_best.pt"

python -u train_jammer_1sc.py --detector-model "$DETECTOR_MODEL"
