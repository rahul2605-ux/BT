#!/bin/bash
# The per-environment chain of the final experiment (README §3.4), submitted from the login
# node (it only calls sbatch):
#   train_cnn -> bands -> 8 generators (array) -> evaluate, each --dependency=afterok.
#   ./submit_env.sh base              # or: bash submit_env.sh base
#   ./submit_env.sh base --smoke      # tiny end-to-end run into artifacts/final/_smoke/
# Prints the four job IDs. If a generator task fails, the evaluate job never starts
# (DependencyNeverSatisfied): resubmit that task, then submit_evaluate.sh by hand.
set -e
export SLURM_CONF=${SLURM_CONF:-/home/sladmitet/slurm/slurm.conf}
E="$1"
[ -z "$E" ] && { echo "usage: submit_env.sh <env> [--smoke]"; exit 1; }
cd "$(dirname "$0")"
mkdir -p runs
TAG="$E"
if [ "$2" == "--smoke" ]; then export FINAL_SMOKE=1; TAG="smoke_$E"; fi
j1=$(sbatch --parsable --job-name="f_cnn_$TAG" --output="runs/train_cnn_${TAG}_%j.out" \
     --error="runs/train_cnn_${TAG}_%j.err" submit_train_cnn.sh "$E")
j2=$(sbatch --parsable --dependency=afterok:$j1 --job-name="f_bands_$TAG" \
     --output="runs/bands_${TAG}_%j.out" --error="runs/bands_${TAG}_%j.err" submit_bands.sh "$E")
j3=$(sbatch --parsable --dependency=afterok:$j2 --array=0-7 --job-name="f_gan_$TAG" \
     --output="runs/train_gan_${TAG}_%A_%a.out" --error="runs/train_gan_${TAG}_%A_%a.err" submit_train_gan.sh "$E")
j4=$(sbatch --parsable --dependency=afterok:$j3 --job-name="f_eval_$TAG" \
     --output="runs/evaluate_${TAG}_%j.out" --error="runs/evaluate_${TAG}_%j.err" submit_evaluate.sh "$E")
echo "$TAG: train_cnn $j1 -> bands $j2 -> train_gan $j3 (array 0-7) -> evaluate $j4"
