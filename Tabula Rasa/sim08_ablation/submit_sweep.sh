#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=00:10:00      # a task takes ~3 min; a short limit also backfills better
#SBATCH --job-name=abl08_sweep
#SBATCH --output=runs/sweep_%A_%a.out
#SBATCH --error=runs/sweep_%A_%a.err
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=2500M       # measured peak 1.7G; small enough for the RAM left on a busy node

# One noise level per task (0..16 = Eb/N0 0:2.5:40 dB, 17 = noiseless): sweep
# N_J x power x n_active, then confirm the frontier picks on fresh frames.
# Run only after submit_verify.sh has exited 0.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u ablation.py --task "$SLURM_ARRAY_TASK_ID" "$@"
