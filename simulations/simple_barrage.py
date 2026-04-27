"""
simulations/simple_barrage.py
------------------------------
Stage 1 baseline: a single TimedBarrageJamming agent vs a static legitimate node.

Runs one episode and produces two plots:
  1. Mean SINR over time — shows the drop when the barrage starts.
  2. Per-subcarrier SINR heatmap — shows which subcarriers are hit.

Usage (from repo root):
    conda run -n sionna-thesis python simulations/simple_barrage.py
"""

import numpy as np
import matplotlib.pyplot as plt

from core.config import AgentRole, AgentSpec, EnvironmentConfig, OFDMConfig, ScatterMode
from core.strategy import TimedBarrageJamming
from envs.wireless_env import WirelessEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SILENT_STEPS = 30     # jammer listens before barraing
MAX_STEPS    = 150
N_SUB        = 76
SEED         = 42

jammer_spec = AgentSpec(
    role=AgentRole.JAMMER,
    count=1,
    ofdm=OFDMConfig(n_subcarriers=N_SUB),
    scatter_mode=ScatterMode.RANDOM,
    max_power_dbm=30.0,
    policy=TimedBarrageJamming(
        n_subcarriers=N_SUB,
        silent_steps=SILENT_STEPS,
        max_power_dbm=30.0,
    ),
)

config = EnvironmentConfig(
    legitimate=AgentSpec(
        role=AgentRole.LEGITIMATE,
        count=1,
        ofdm=OFDMConfig(n_subcarriers=N_SUB),
        scatter_mode=ScatterMode.RANDOM,
        max_power_dbm=20.0,
    ),
    jammers=jammer_spec,
    max_steps=MAX_STEPS,
    seed=SEED,
)

# ---------------------------------------------------------------------------
# Run episode
# ---------------------------------------------------------------------------

env = WirelessEnv(config)
obs, _ = env.reset()

sinr_per_step = []   # (MAX_STEPS, N_SUB)
mean_sinr     = []   # (MAX_STEPS,)

dummy_action = np.zeros(N_SUB, dtype=np.float32)  # ignored — policy is set

for _ in range(MAX_STEPS):
    obs, reward, terminated, truncated, info = env.step(dummy_action)
    sinr_per_step.append(obs[:N_SUB].copy())   # first N_SUB elements = SINR per subcarrier
    mean_sinr.append(info["mean_sinr_db"])
    if terminated or truncated:
        break

sinr_matrix = np.stack(sinr_per_step)   # (T, N_SUB)
steps = np.arange(1, len(mean_sinr) + 1)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
fig.suptitle("Stage 1 Baseline — Timed Barrage Jamming", fontsize=13)

# -- Mean SINR over time
ax1.plot(steps, mean_sinr, color="steelblue", linewidth=1.5, label="Mean SINR")
ax1.axvline(SILENT_STEPS, color="crimson", linestyle="--", linewidth=1.2, label=f"Barrage starts (step {SILENT_STEPS})")
ax1.set_xlabel("Step")
ax1.set_ylabel("Mean SINR (dB)")
ax1.set_title("Mean SINR over Episode")
ax1.legend()
ax1.grid(True, alpha=0.3)

# -- Per-subcarrier SINR heatmap
im = ax2.imshow(
    sinr_matrix.T,
    aspect="auto",
    origin="lower",
    extent=[1, len(steps), 0, N_SUB],
    cmap="RdYlGn",
)
ax2.axvline(SILENT_STEPS, color="white", linestyle="--", linewidth=1.2, label=f"Barrage starts")
ax2.set_xlabel("Step")
ax2.set_ylabel("Subcarrier index")
ax2.set_title("SINR per Subcarrier (dB)")
ax2.legend(loc="upper right")
fig.colorbar(im, ax=ax2, label="SINR (dB)")

plt.savefig("simulations/barrage_baseline.png", dpi=150)
print("Saved: simulations/barrage_baseline.png")
plt.show()
