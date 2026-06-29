#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=02:00:00
#SBATCH --job-name=sim05_cnn
#SBATCH --output=runs/detector/slurm_%j.out
#SBATCH --error=runs/detector/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation05"
mkdir -p runs/detector

# torchvision needed for EfficientNet-B0
pip install --quiet torchvision 2>&1 | tail -1

python -u train_detector.py
