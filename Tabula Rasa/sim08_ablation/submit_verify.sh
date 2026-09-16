#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=abl08_verify
#SBATCH --output=runs/verify_%j.out
#SBATCH --error=runs/verify_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Sionna cannot be imported on the login node, so the test suite is a job.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u verify.py "$@"
