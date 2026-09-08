#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:00:00
#SBATCH --job-name=sim05_eval
#SBATCH --output=runs/jammer/slurm_%j.out
#SBATCH --error=runs/jammer/slurm_%j.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090

source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/jammer

# Use sim04 run007 (latest clean run) and sim05 detector run001
JAMMER_MODEL="../artifacts/sim04/run007_model.pt"
DETECTOR_MODEL="../artifacts/sim05/detector/run001_best.pt"

# Fall back to run006 if run007 doesn't exist
if [ ! -f "$JAMMER_MODEL" ]; then
    JAMMER_MODEL="../artifacts/sim04/run006_model.pt"
fi

python -u eval_jammer_vs_detector.py \
    --jammer-model "$JAMMER_MODEL" \
    --detector-model "$DETECTOR_MODEL" \
    --n-samples 200
