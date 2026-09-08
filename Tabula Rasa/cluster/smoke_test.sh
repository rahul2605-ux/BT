#!/bin/bash
# ---------------------------------------------------------------------------
# ITET/TIK cluster smoke test — confirms the batch environment behaves like the
# old INFK student cluster before any real work is submitted.
#
# Checks: node identity, GPU allocation + visibility, CUDA from python,
# filesystem reachability (home / node-local scratch / repo), and the
# SLURM_CONF fix surviving into the job.
#
#   sbatch cluster/smoke_test.sh      (from the repo root)
# ---------------------------------------------------------------------------
#SBATCH --account=disco-med
#SBATCH --partition=gpu.normal,disco.med
#SBATCH --job-name=smoke
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/runs/smoke_%j.out
#SBATCH --error=cluster/runs/smoke_%j.err

set -u

echo "=============================================================="
echo " ITET smoke test — job $SLURM_JOB_ID"
echo "=============================================================="
echo "node            : $(hostname -f)"
echo "partition       : ${SLURM_JOB_PARTITION:-?}"
echo "account         : ${SLURM_JOB_ACCOUNT:-?}"
echo "cpus            : ${SLURM_CPUS_PER_TASK:-?}"
echo "mem             : ${SLURM_MEM_PER_NODE:-?} MB"
echo "SLURM_JOB_GPUS  : ${SLURM_JOB_GPUS:-<none>}"
echo "CUDA_VISIBLE_DEV: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "submit dir      : ${SLURM_SUBMIT_DIR:-?}"
echo "started         : $(date -Is)"
echo

echo "--- OS on the compute node (submit host is Debian 12) ---"
grep PRETTY_NAME /etc/os-release
echo

echo "--- GPU ---"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
    echo "FAIL: nvidia-smi not on PATH"
fi
echo

echo "--- filesystems ---"
for p in "$HOME" /scratch "$SLURM_SUBMIT_DIR"; do
    if [ -d "$p" ]; then
        printf 'ok   %-42s %s avail\n' "$p" "$(df -h "$p" 2>/dev/null | tail -1 | awk '{print $4}')"
    else
        printf 'FAIL %-42s (not reachable from node)\n' "$p"
    fi
done
echo

echo "--- repo reachable from the node ---"
if [ -f "$SLURM_SUBMIT_DIR/README.md" ]; then
    echo "ok   README.md found ($(wc -l < "$SLURM_SUBMIT_DIR/README.md") lines)"
else
    echo "FAIL README.md not found under \$SLURM_SUBMIT_DIR"
fi
echo

echo "--- python / torch ---"
python3 -V
python3 - <<'PY'
try:
    import torch
    print(f"ok   torch {torch.__version__}")
    print(f"     cuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     device count   : {torch.cuda.device_count()}")
        print(f"     device 0       : {torch.cuda.get_device_name(0)}")
        print(f"     capability     : {torch.cuda.get_device_capability(0)}")
        x = torch.randn(2048, 2048, device="cuda")
        print(f"     matmul check   : {(x @ x).sum().item():.3e}  <- GPU compute works")
except ImportError:
    print("note torch not installed in the system python "
          "(expected: the project venv still has to be rebuilt on ITET)")
PY
echo

echo "--- slurm client works from inside the job ---"
echo "SLURM_CONF=${SLURM_CONF:-<unset>}"
squeue -j "$SLURM_JOB_ID" -o "%.10i %.14P %.10j %.8T %.10M %R" 2>&1 | head -3
echo
echo "finished        : $(date -Is)"
echo "=============================================================="
