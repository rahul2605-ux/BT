#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:30:00
#SBATCH --job-name=recheck
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

echo "=== retrain detector on CORRECTED complex-STFT spectrogram ==="
cd ../simulation06
python -u train_detector.py
DET=$(realpath "$(ls -t ../artifacts/sim06/detector/run*_best.pt | head -1)")
echo "new detector checkpoint: $DET"

echo "=== recheck Phase 0: complex-STFT CNN + energy detector suite ==="
cd ../frontier
python -u recheck_suite.py --detector-model "$DET"
