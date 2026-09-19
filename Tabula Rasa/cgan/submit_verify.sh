#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=cgan_verify
#SBATCH --output=runs/verify_%j.out
#SBATCH --error=runs/verify_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it. 24 GB cards only:
# §10's geometry check holds ~9 GB at once and OOMs an 11 GB 2080 Ti (job 2267046).
#SBATCH --constraint=titan_rtx|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Sionna cannot be imported on the login node, so the cgan/ test suite is a job.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u verify.py -v "$@"
