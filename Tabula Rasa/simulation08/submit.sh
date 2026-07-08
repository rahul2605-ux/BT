#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:45:00
#SBATCH --job-name=sim08_m2
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation08"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

echo "=== sim08 m2: retrain CNN detector on the FADED channel ==="
python -u retrain_detector_channel.py
DET=$(realpath "$(ls -t ../artifacts/sim08/detector/run*_best.pt | head -1)")
echo "channel-valid detector checkpoint: $DET"

echo "=== sim08 m2: full-suite (CNN + energy) frontier on the realistic channel ==="
python -u frontier_channel.py --detector-model "$DET"
