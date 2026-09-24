#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:00:00
#SBATCH --job-name=cgan_snr_abl
#SBATCH --output=runs/snr_abl_%A_%a_%N.out
#SBATCH --error=runs/snr_abl_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# E2 noise ablation on the live models (README §3.3g): one SNR level per array task.
#   sbatch --array=6 submit_snr_ablation.sh      # 30 dB alone: MUST reproduce §3.3f
#   sbatch --array=0-9 submit_snr_ablation.sh    # the full grid
# --cpus-per-task=2, not 4: 4 stalled the D2 jobs in PENDING (BadConstraints) when the
# only node with a free GPU of the allowed type had 2 free CPUs (README §3.3f).
# --mem is a first guess; re-size it from `sacct --format=MaxRSS` after this runs (§C.4).
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u snr_ablation.py --task "${SLURM_ARRAY_TASK_ID}" "$@"
