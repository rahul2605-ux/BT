#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=03:00:00
#SBATCH --job-name=gen_jammer
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation03b"
mkdir -p runs

CUDA_VISIBLE_DEVICES="" python -u train.py
