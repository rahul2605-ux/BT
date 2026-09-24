#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:20:00
#SBATCH --job-name=snr_examples
#SBATCH --output=runs/snr_examples_%j_%N.out
#SBATCH --error=runs/snr_examples_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# E2 example frames for the report (README §3.3g): what the CNN sees at 0 / 15 / 30 dB
# and the noiseless anchor, clean vs D1 vs D2 at matched BER. -> snr_ablation/run001/examples.npz
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u snr_examples.py "$@"
