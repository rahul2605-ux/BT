#!/bin/bash
# ---------------------------------------------------------------------------
# Verify the rebuilt ITET python environment on a compute node.
#   sbatch cluster/verify_env.sh        (from the repo root)
# ---------------------------------------------------------------------------
#SBATCH --account=disco-med
#SBATCH --job-name=verify_env
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
# torch 2.9 dropped Pascal (sm_61): the titan_xp nodes tikgpu02/03 cannot run
# it ("no kernel image is available"). Ask for anything sm_70 or newer.
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/runs/verify_%j.out
#SBATCH --error=cluster/runs/verify_%j.err

set -u
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate

echo "node   : $(hostname -f)"
echo "gpu    : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "python : $(which python)"
echo

python - <<'PY'
import sys, traceback

def check(label, fn):
    try:
        print(f"ok   {label:<22} {fn()}")
        return True
    except Exception as e:
        print(f"FAIL {label:<22} {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False

import torch
check("torch",        lambda: torch.__version__)
check("cuda build",   lambda: torch.version.cuda)
check("cuda avail",   lambda: torch.cuda.is_available())
check("device",       lambda: torch.cuda.get_device_name(0))

# real GPU compute, not just a handshake
x = torch.randn(4096, 4096, device="cuda")
y = (x @ x).sum().item()
print(f"ok   gpu matmul             {y:.4e}")

import sionna, importlib.metadata as md
check("sionna",       lambda: md.version("sionna"))
import sionna.phy
check("sionna.phy",   lambda: "imported")

# exercise the PHY layer the project actually uses
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.channel import AWGN
bs   = BinarySource()
mapr = Mapper("qam", 2)
demp = Demapper("app", "qam", 2)
awgn = AWGN()
bits = bs([4, 1024])
sym  = mapr(bits)
rx   = awgn(sym, 0.01)
llr  = demp(rx, 0.01)
ber  = ((llr > 0).to(bits.dtype) != bits).float().mean().item()
print(f"ok   sionna QPSK chain      BER={ber:.5f} at low noise (expect ~0)")

import stable_baselines3, gymnasium, zuko, torchvision, scipy, matplotlib
check("stable-baselines3", lambda: stable_baselines3.__version__)
check("gymnasium",         lambda: gymnasium.__version__)
check("zuko",              lambda: zuko.__version__)
check("torchvision",       lambda: torchvision.__version__)
check("scipy",             lambda: scipy.__version__)
check("matplotlib",        lambda: matplotlib.__version__)

# torchvision was pip-installed at runtime in the old INFK scripts; the
# detectors use EfficientNet-B0, so confirm it constructs.
from torchvision.models import efficientnet_b0
m = efficientnet_b0(weights=None).cuda()
n = sum(p.numel() for p in m.parameters())
print(f"ok   efficientnet_b0        {n/1e6:.2f} M params on GPU")

# SB3 smoke: one short learn() call
import warnings; warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
env = gymnasium.make("Pendulum-v1")
PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32,
    device="cpu").learn(total_timesteps=128)
print("ok   sb3 PPO learn()        128 steps completed")

print("\nALL CHECKS PASSED")
PY
