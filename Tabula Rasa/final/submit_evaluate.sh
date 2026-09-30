#!/bin/bash
#SBATCH --account=disco-med
#SBATCH --time=02:00:00
#SBATCH --job-name=final_eval
#SBATCH --output=runs/evaluate_%j.out
#SBATCH --error=runs/evaluate_%j.err
#SBATCH --gres=gpu:1
# ONE card type for every evaluation (CUDA's random streams differ between models, §C.4)
#SBATCH --constraint=titan_rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Chain step 4 (README §3.4): every attacker vs every detector on env $1 -> eval.json.
#   sbatch submit_evaluate.sh base [--only tag1,tag2]
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
echo "node: $(hostname -f) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
E="$1"; shift
python -u evaluate.py --env "$E" "$@"
