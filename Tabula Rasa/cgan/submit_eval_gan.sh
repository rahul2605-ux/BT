#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=02:00:00
#SBATCH --job-name=cgan_eval_gan
#SBATCH --output=runs/eval_gan_%j_%N.out
#SBATCH --error=runs/eval_gan_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D1: put a plain/trained GAN generator on the D0 BER-P(det) plane (README §3.4).
# Smoke first: sbatch submit_eval_gan.sh --smoke
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u eval_gan.py "$@"
