#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=00:45:00
#SBATCH --job-name=phase05
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/frontier"
mkdir -p runs

pip install --quiet torchvision 2>&1 | tail -1

echo "=== Phase 0.5: retrain detector with in-band jammers ==="
python -u retrain_detector_inband.py

DET=$(ls -t ../artifacts/frontier/detector/run*_best.pt | head -1)
echo "=== re-sweeping frontier against retrained detector: $DET ==="
python -u frontier_sweep.py --detector-model "$DET" --out ../artifacts/frontier_inband
