#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:30:00
#SBATCH --job-name=cgan_team_pol
#SBATCH --output=runs/team_policy_%A_%a_%N.out
#SBATCH --error=runs/team_policy_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# D4b learned coordination policy (team_policy.py): one (SNR, beta) per array task.
#   sbatch submit_team_policy.sh --smoke                     # task 0, tiny train + eval
#   sbatch --array=0-3 submit_team_policy.sh                 # TRAIN (SNR 15,30)x(beta 0,10)
#   sbatch --array=0-3 submit_team_policy.sh --mode eval     # EVAL, after training done
# --mem 8G: training keeps one EfficientNet-B0 graph (grad=True) like train_gan.py.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u team_policy.py --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
