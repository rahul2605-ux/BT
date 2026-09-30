#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:30:00
#SBATCH --job-name=final_verify
#SBATCH --output=runs/verify_%j.out
#SBATCH --error=runs/verify_%j.err
#SBATCH --gres=gpu:1
# titan_rtx: 24 GB (cgan's verify OOMed an 11 GB card, README §C.4) and the card type the
# chains are pinned to, so the verify and the runs draw the same CUDA random streams.
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

# THE TEST SUITE for final/ (verify.py), then the regression gate vs S4 (regress.py).
# Exit 0 only if both pass. No environment runs before this exits 0 (README §3.4).
#   sbatch submit_verify.sh              # both
#   sbatch submit_verify.sh --gate-only  # regress.py alone
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [ "$1" != "--gate-only" ]; then
    python -u verify.py -v || { echo "VERIFY FAILED"; exit 1; }
fi
python -u regress.py || { echo "REGRESSION GATE FAILED"; exit 1; }
