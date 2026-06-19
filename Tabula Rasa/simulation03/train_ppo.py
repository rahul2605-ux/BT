#GPU Session: srun --account=projects --partition=interactive --gres=gpu:1 --pty bash

import os
print("imports starting...", flush=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — no display needed
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from jammer_env import JammerEnv, N_SYMBOLS, KURT_THRESH, BETA, GAMMA
import torch
print(f"imports done — CUDA available: {torch.cuda.is_available()}", flush=True)

TOTAL_STEPS = 50_000

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim03")
os.makedirs(RUNS_DIR, exist_ok=True)
existing = [int(f[3:6]) for f in os.listdir(RUNS_DIR) if f.startswith("run") and f.endswith(".png")]
run_id   = f"{max(existing, default=0) + 1:03d}"
print(f"Run {run_id}")

SNAPSHOT_STEPS = 500

class LoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.bers       = []
        self.detections = []
        self.kurts      = []
        self.powers     = []
        self.actions    = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        self.bers.append(np.mean([i["ber"] for i in infos]))
        self.detections.append(np.mean([float(i["detected"]) for i in infos]))
        self.kurts.append(np.mean([i["kurtosis"] for i in infos]))
        self.powers.append(np.mean([i["power"] for i in infos]))
        self.actions.append(self.locals["actions"][0].copy())
        return True

print("creating env...", flush=True)
env      = JammerEnv()
print("env created, building model...", flush=True)
callback = LoggingCallback()
model    = PPO("MlpPolicy", env, verbose=1, device="auto", learning_rate=1e-4, clip_range=0.1, n_steps=512)
print("model ready, starting training...", flush=True)
model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

model_path = os.path.join(RUNS_DIR, f"run{run_id}_model")
model.save(model_path)
print(f"Saved model → {model_path}.zip")

W = max(1, TOTAL_STEPS // 100)

def rolling(data):
    return np.convolve(data, np.ones(W) / W, mode='valid')

meta = (f"run={run_id}  |  steps={TOTAL_STEPS:,}  |  N_SYMBOLS={N_SYMBOLS}  |  "
        f"BETA={BETA}  |  GAMMA={GAMMA}  |  KURT_THRESH={KURT_THRESH}  |  rolling_W={W}")

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axes[0].plot(rolling(callback.bers))
axes[0].set_ylabel("BER (rolling avg)")
axes[0].set_title(f"PPO Jammer — Kurtosis Detector\n{meta}", fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(rolling(callback.detections), color='red')
axes[1].set_ylabel("Detection rate (rolling avg)")
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(alpha=0.3)

axes[2].plot(rolling(callback.kurts), color='green')
axes[2].axhline(KURT_THRESH, color='orange', ls='--', label=f'KURT_THRESH ({KURT_THRESH})')
axes[2].axhline(0,           color='gray',   ls=':',  label='Gaussian reference (0)')
axes[2].axhline(-2,          color='blue',   ls=':',  label='QPSK reference (-2)')
axes[2].set_ylabel("Received kurtosis (rolling avg)")
axes[2].legend()
axes[2].grid(alpha=0.3)

axes[3].plot(rolling(callback.powers), color='purple')
axes[3].set_ylabel("Jammer power (rolling avg)")
axes[3].set_xlabel("Step")
axes[3].grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(RUNS_DIR, f"run{run_id}.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
pass  # no display on cluster
print(f"Saved → {out_path}")

# ── IQ scatter snapshots ──────────────────────────────────────────────────────
actions = np.array(callback.actions)   # (total_steps, 2*N_SYMBOLS)
n       = len(actions)
thirds  = [
    ("early", actions[:SNAPSHOT_STEPS]),
    ("mid",   actions[n//2 : n//2 + SNAPSHOT_STEPS]),
    ("late",  actions[-SNAPSHOT_STEPS:]),
]

qpsk   = np.array([ 1,  1, -1, -1]) / np.sqrt(2)
qpsk_q = np.array([ 1, -1,  1, -1]) / np.sqrt(2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"IQ scatter — jammer actions  [{meta}]", fontsize=8)

for ax, (label, chunk) in zip(axes, thirds):
    iq = chunk.reshape(-1, N_SYMBOLS, 2)
    I  = iq[:, :, 0].flatten()
    Q  = iq[:, :, 1].flatten()
    ax.scatter(I, Q, s=1, alpha=0.3, color='steelblue')
    ax.scatter(qpsk, qpsk_q, s=60, color='red', marker='x', label='QPSK ref', zorder=5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(label)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

plt.tight_layout()
iq_path = os.path.join(RUNS_DIR, f"run{run_id}_iq.png")
plt.savefig(iq_path, dpi=150, bbox_inches='tight')
pass  # no display on cluster
print(f"Saved → {iq_path}")