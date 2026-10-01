#!/bin/bash
# The Li-protocol readout of one environment (README §3.4 "Li-protocol readout in noise and fading"),
# submitted from the login node (it only calls sbatch):
#   surrogate CNN (seed 12 -> cnn_s12/) -> its band (bands_cnn_s12.json) -> cnnGAN1 grey-box
#   (train_gan array 11-13) -> evaluate --out eval_li.json (every attacker, the CNN's argmax recorded)
#   -> IQ pictures + pulse pictures (in parallel), each --dependency=afterok.
#   ./submit_li_env.sh noise          # the environment's final/ chain (submit_env.sh) must be done
# Run `sbatch submit_verify.sh` once before the first environment. Afterwards, on the login node:
#   python li_figures.py --env E && python li_figures.py --env E --clean
#   python iq_plots.py --env E --plot-only --clean
# cnnGAN2 (train_gan indices 8-10 and 14-16) is dropped (user 2026-09-30) and is not trained here.
set -e
export SLURM_CONF=${SLURM_CONF:-/home/sladmitet/slurm/slurm.conf}
E="$1"
[ -z "$E" ] && { echo "usage: submit_li_env.sh <env>"; exit 1; }
cd "$(dirname "$0")"
mkdir -p runs
j1=$(sbatch --parsable --job-name="li_cnn_$E" --output="runs/train_cnn_s12_${E}_%j.out" \
     --error="runs/train_cnn_s12_${E}_%j.err" submit_train_cnn.sh "$E" --seed 12 --name cnn_s12)
j2=$(sbatch --parsable --dependency=afterok:$j1 --job-name="li_bands_$E" \
     --output="runs/bands_s12_${E}_%j.out" --error="runs/bands_s12_${E}_%j.err" submit_bands.sh "$E" --cnn cnn_s12)
j3=$(sbatch --parsable --dependency=afterok:$j2 --array=11-13 --job-name="li_gan_$E" \
     --output="runs/train_gan_grey_${E}_%A_%a.out" --error="runs/train_gan_grey_${E}_%A_%a.err" \
     submit_train_gan.sh "$E")
j4=$(sbatch --parsable --dependency=afterok:$j3 --job-name="li_eval_$E" \
     --output="runs/evaluate_li_${E}_%j.out" --error="runs/evaluate_li_${E}_%j.err" \
     submit_evaluate.sh "$E" --out eval_li.json)
j5=$(sbatch --parsable --dependency=afterok:$j4 --job-name="li_iq_$E" \
     --output="runs/iq_${E}_%j.out" --error="runs/iq_${E}_%j.err" submit_iq.sh "$E")
j6=$(sbatch --parsable --dependency=afterok:$j4 --job-name="li_pulse_$E" \
     --output="runs/pulse_${E}_%j.out" --error="runs/pulse_${E}_%j.err" submit_pulse.sh "$E")
echo "$E: surrogate CNN $j1 -> bands $j2 -> cnnGAN1 grey $j3 (array 11-13) -> evaluate $j4 -> iq $j5, pulse $j6"
