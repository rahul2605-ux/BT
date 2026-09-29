#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=01:00:00
#SBATCH --job-name=arms_cnn
#SBATCH --output=runs/arms_cnn_%A_%a_%N.out
#SBATCH --error=runs/arms_cnn_%A_%a.err
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): tikgpu02/03 (titan_xp) cannot run it.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=2
# a retrain peaks at 2.3 GB (sacct MaxRSS, jobs 2274865/2274946; README §C.4)
#SBATCH --mem=6G

# D6 round 1 (README §3.3o): the defender's retrains, one per array task:
#   sbatch --array=0-4 submit_arms_cnn.sh
# The 30 dB round-0 defender is the deployed CNN and is not retrained. The widened
# range is arms_eval.WIDE_DB; the round-0 attackers are arms_eval.GENS' "seen" rows.
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
OUT=../artifacts/cgan/baselines/arms
R0="../artifacts/cgan/gan/run003/task13_G.pt ../artifacts/cgan/gan/run003/task14_G.pt"
case "${SLURM_ARRAY_TASK_ID}" in
    0) ARGS="--snr-db 15 --out-dir $OUT/snr15_r0" ;;
    1) ARGS="--jsr-range -35 10 --out-dir $OUT/snr30_A" ;;
    2) ARGS="--jsr-range -35 10 --extra-gens $R0 --out-dir $OUT/snr30_B" ;;
    3) ARGS="--snr-db 15 --jsr-range -35 10 --out-dir $OUT/snr15_A" ;;
    4) ARGS="--snr-db 15 --jsr-range -35 10 --extra-gens $R0 --out-dir $OUT/snr15_B" ;;
    *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
python -u train_spectrogram_cnn.py $ARGS "$@"
