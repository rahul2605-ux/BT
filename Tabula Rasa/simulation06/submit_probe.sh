#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:10:00
#SBATCH --job-name=sim06_probe
#SBATCH --output=runs/jammer/probe_%j.out
#SBATCH --error=runs/jammer/probe_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"


python -u probe_1sc.py
