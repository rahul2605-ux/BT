#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=02:00:00
#SBATCH --job-name=cgan_eval_arr
#SBATCH --output=runs/eval_gan_%A_%a_%N.out
#SBATCH --error=runs/eval_gan_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D2 eval: place trained generator task{ID}_G.pt on the BER-P(det) plane.
# sbatch --array=0-12 submit_eval_gan_array.sh
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u eval_gan.py --task "${SLURM_ARRAY_TASK_ID}" "$@"
