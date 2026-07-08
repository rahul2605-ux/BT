#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:30:00
#SBATCH --job-name=recheck
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/frontier"
mkdir -p runs
pip install --quiet torchvision 2>&1 | tail -1

echo "=== retrain detector on CORRECTED complex-STFT spectrogram ==="
cd ../simulation06
python -u train_detector.py
DET=$(realpath "$(ls -t ../artifacts/sim06/detector/run*_best.pt | head -1)")
echo "new detector checkpoint: $DET"

echo "=== recheck Phase 0: complex-STFT CNN + energy detector suite ==="
cd ../frontier
python -u recheck_suite.py --detector-model "$DET"
