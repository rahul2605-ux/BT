#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:30:00
#SBATCH --job-name=cgan_team_fad
#SBATCH --output=runs/team_fading_%A_%a_%N.out
#SBATCH --error=runs/team_fading_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# E3 team-under-fading pre-check (team_fading.py): one (SNR, channel) per array task.
#   sbatch submit_team_fading.sh --smoke          # no array: task 5 (30 dB, Rayleigh), tiny grid
#   sbatch --array=0-5 submit_team_fading.sh      # (15, 30 dB) x (lossless, rician10, rayleigh)
# --mem 4G: the E2 tasks this reuses peaked at 2.0-2.2 GB MaxRSS (sacct, jobs 2270449/2270556).
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u team_fading.py --task "${SLURM_ARRAY_TASK_ID:-5}" "$@"
