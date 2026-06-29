#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=08:00:00
#SBATCH --job-name=sim06_jam
#SBATCH --output=runs/jammer/slurm_%j.out
#SBATCH --error=runs/jammer/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation06"
mkdir -p runs/jammer

pip install --quiet torchvision 2>&1 | tail -1

DETECTOR_MODEL="../artifacts/sim06/detector/run002_best.pt"

python -u train_jammer.py --detector-model "$DETECTOR_MODEL"
