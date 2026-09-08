#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=03:00:00
#SBATCH --job-name=gen_jammer
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

CUDA_VISIBLE_DEVICES="" python -u train.py
