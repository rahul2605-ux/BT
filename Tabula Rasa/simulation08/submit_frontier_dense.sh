#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=01:30:00
#SBATCH --job-name=sim08_dense
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

# DENSE full-suite frontier — step-1 (matched-detectability) follow-up.
# Reuses the trained channel-valid detector (run001_best.pt). Finer power +
# n_active grid, concentrated in the low-power/stealthy regime, at larger batch
# (B=512) to cut the per-frame detection-rate noise that makes the coarse
# blind-vs-channel-aware frontier hard to trust. Writes to a SEPARATE out dir
# so the canonical m2 results/figures (../artifacts/sim08/frontier) stay intact.
source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation08"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

python -u frontier_channel.py \
    --detector-model "../artifacts/sim08/detector/run001_best.pt" \
    --out "../artifacts/sim08/frontier_dense" \
    --batch 512 \
    --powers "0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0" \
    --n-active "1,2,3,4,6,8,12,16,24,32,52"

# Matched-detectability figure/summary on the dense data (CPU-only post-proc).
python -u matched_detectability.py \
    --results "../artifacts/sim08/frontier_dense/results.json" \
    --out "../artifacts/sim08/frontier_dense"
