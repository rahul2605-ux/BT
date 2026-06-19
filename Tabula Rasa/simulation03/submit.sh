#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ppo_jammer
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation03"
mkdir -p runs

python -u train_ppo.py
