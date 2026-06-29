#!/bin/bash
#SBATCH --account=projects
#SBATCH --partition=jobs
#SBATCH --time=01:00:00
#SBATCH --job-name=sim05_eval
#SBATCH --output=runs/jammer/slurm_%j.out
#SBATCH --error=runs/jammer/slurm_%j.err
#SBATCH --gpus=1

source /work/scratch/rrahman/bt_env/bin/activate
cd "/home/rrahman/StudentClusterBT/Tabula Rasa/simulation05"
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
