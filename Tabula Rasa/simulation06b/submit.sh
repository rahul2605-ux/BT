#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=04:00:00
#SBATCH --job-name=sim06b_1sc
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs


DETECTOR_MODEL="../artifacts/sim06/detector/run002_best.pt"

python -u train_jammer_1sc.py --detector-model "$DETECTOR_MODEL"
