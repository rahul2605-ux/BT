#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:10:00
#SBATCH --job-name=sim06_probe
#SBATCH --output=runs/jammer/probe_%j.out
#SBATCH --error=runs/jammer/probe_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation06"

pip install --quiet torchvision 2>&1 | tail -1

python -u probe_1sc.py
