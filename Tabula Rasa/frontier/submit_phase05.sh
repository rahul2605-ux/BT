#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:45:00
#SBATCH --job-name=phase05
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs


echo "=== Phase 0.5: retrain detector with in-band jammers ==="
python -u retrain_detector_inband.py

DET=$(ls -t ../artifacts/frontier/detector/run*_best.pt | head -1)
echo "=== re-sweeping frontier against retrained detector: $DET ==="
python -u frontier_sweep.py --detector-model "$DET" --out ../artifacts/frontier_inband
