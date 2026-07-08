#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:15:00
#SBATCH --job-name=specfig
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/frontier"
mkdir -p runs
pip install --quiet torchvision 2>&1 | tail -1
python -u spectrogram_figure.py
