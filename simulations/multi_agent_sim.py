"""
simulations/multi_agent_sim.py
--------------------------------
Multi-jammer / multi-defender simulation with fixed strategies.

Experiments:
  1. Sweep number of jammers (1–5), fixed N_leg=1 — how much does adding
     more jammers help? Diminishing returns? Interference budget?
  2. Sweep number of defenders (1–3), fixed N_jam=3 — does each defender
     see similar SINR degradation, or does position matter a lot?
  3. Single episode trace: 3 jammers vs 2 defenders, per-subcarrier heatmap.

All jammers use TimedBarrageJamming (Stage 1a baseline).

Usage (local, MockChannel):
    PYTHONPATH=. conda run -n sionna-thesis python simulations/multi_agent_sim.py

Usage (Colab, SionnaChannel):
    !git clone <repo>
    import sys; sys.path.insert(0, '/content/BT')
    !python simulations/multi_agent_sim.py --channel sionna
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.config import AgentRole, AgentSpec, EnvironmentConfig, OFDMConfig, ScatterMode
from core.strategy import TimedBarrageJamming
from envs.wireless_env import WirelessEnv

N_SUB     = 76
MAX_STEPS = 150
SEED      = 42

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_jammers: int, n_defenders: int, channel_mode: str, seed: int) -> WirelessEnv:
    return WirelessEnv(EnvironmentConfig(
        legitimate=AgentSpec(
            role=AgentRole.LEGITIMATE, count=n_defenders,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM,
            max_power_dbm=20.0,
        ),
        jammers=AgentSpec(
            role=AgentRole.JAMMER, count=n_jammers,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM,
            max_power_dbm=30.0,
            policy=TimedBarrageJamming(
                n_subcarriers=N_SUB,
                silent_steps=20,   # listen first, then jam
                max_power_dbm=30.0,
            ),
        ),
        channel_mode=channel_mode,
        max_steps=MAX_STEPS,
        seed=seed,
    ))


def _run_episode(env: WirelessEnv) -> dict:
    """
    Run one episode with the env's built-in policy.
    Returns a dict with per-step SINR traces for all defenders.
    """
    obs, _ = env.reset()
    env.config.jammers.policy.reset()

    dummy_action = np.zeros(N_SUB, dtype=np.float32)

    sinr_trace    = []   # list of (N_rx, N_sub) per step
    mean_sinr_log = []   # mean over all RX and subcarriers

    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(dummy_action)
        done = terminated or truncated
        if env._last_sinr is not None:
            sinr_trace.append(env._last_sinr.copy())     # (N_rx, N_sub)
            mean_sinr_log.append(info["mean_sinr_db"])

    return {
        "sinr_trace": np.stack(sinr_trace),  # (T, N_rx, N_sub)
        "mean_sinr":  np.array(mean_sinr_log),
    }


# ---------------------------------------------------------------------------
# Experiment 1 — sweep N_jammers
# ---------------------------------------------------------------------------

def exp1_sweep_jammers(channel_mode: str, out_dir: str):
    jam_counts  = [1, 2, 3, 4, 5]
    results     = {}

    print("Experiment 1: sweeping number of jammers (N_leg=1) ...")
    for n_jam in jam_counts:
        env = _make_env(n_jammers=n_jam, n_defenders=1,
                        channel_mode=channel_mode, seed=SEED)
        ep  = _run_episode(env)
        env.close()
        results[n_jam] = ep
        mean_after_jam = ep["mean_sinr"][20:].mean()
        print(f"  N_jam={n_jam}  mean SINR after barrage starts: {mean_after_jam:.1f} dB")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("Experiment 1: Effect of Number of Jammers (N_leg=1, Barrage)", fontsize=11)

    # Left: mean SINR over time for each N_jam
    ax = axes[0]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(jam_counts)))
    for color, n_jam in zip(colors, jam_counts):
        sinr = results[n_jam]["mean_sinr"]
        ax.plot(sinr, color=color, linewidth=1.5, label=f"N_jam={n_jam}")
    ax.axvline(20, color="black", linestyle="--", linewidth=0.8, label="Barrage starts")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean SINR at Defender (dB)")
    ax.set_title("SINR over episode")
    ax.legend(fontsize=8)

    # Right: boxplot of post-barrage SINR per N_jam
    ax2 = axes[1]
    data   = [results[n]["mean_sinr"][20:] for n in jam_counts]
    labels = [str(n) for n in jam_counts]
    bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel("Number of Jammers")
    ax2.set_ylabel("Post-barrage SINR (dB)")
    ax2.set_title("SINR distribution vs jammer count")

    path = os.path.join(out_dir, "exp1_sweep_jammers.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")
    return results


# ---------------------------------------------------------------------------
# Experiment 2 — sweep N_defenders
# ---------------------------------------------------------------------------

def exp2_sweep_defenders(channel_mode: str, out_dir: str):
    def_counts = [1, 2, 3]
    n_jam      = 3

    print(f"\nExperiment 2: sweeping number of defenders (N_jam={n_jam}) ...")

    fig, axes = plt.subplots(1, len(def_counts), figsize=(13, 4), constrained_layout=True)
    fig.suptitle(f"Experiment 2: Per-Defender SINR (N_jam={n_jam}, Barrage)", fontsize=11)

    for ax, n_def in zip(axes, def_counts):
        env = _make_env(n_jammers=n_jam, n_defenders=n_def,
                        channel_mode=channel_mode, seed=SEED)
        ep  = _run_episode(env)
        env.close()

        # ep["sinr_trace"]: (T, N_rx, N_sub) — average over subcarriers per RX
        sinr_per_rx = ep["sinr_trace"].mean(axis=2)  # (T, N_rx)

        for i in range(n_def):
            ax.plot(sinr_per_rx[:, i], linewidth=1.3, label=f"Defender {i+1}")
        ax.axvline(20, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"N_def={n_def}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean SINR (dB)")
        ax.legend(fontsize=7)

        mean_post = sinr_per_rx[20:].mean()
        print(f"  N_def={n_def}  mean SINR after barrage: {mean_post:.1f} dB")

    path = os.path.join(out_dir, "exp2_sweep_defenders.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Experiment 3 — per-subcarrier SINR heatmap, N_jam=3, N_leg=2
# ---------------------------------------------------------------------------

def exp3_heatmap(channel_mode: str, out_dir: str):
    n_jam, n_def = 3, 2
    print(f"\nExperiment 3: per-subcarrier heatmap (N_jam={n_jam}, N_def={n_def}) ...")

    env = _make_env(n_jammers=n_jam, n_defenders=n_def,
                    channel_mode=channel_mode, seed=SEED)
    ep  = _run_episode(env)
    env.close()

    # sinr_trace: (T, N_rx, N_sub)
    sinr = ep["sinr_trace"]
    T    = sinr.shape[0]

    fig, axes = plt.subplots(n_def, 1, figsize=(11, 3 * n_def), constrained_layout=True)
    fig.suptitle(
        f"Experiment 3: SINR heatmap per defender (N_jam={n_jam}, N_def={n_def})", fontsize=11
    )
    if n_def == 1:
        axes = [axes]

    vmin = sinr.min() - 2
    vmax = sinr.max() + 2

    for i, ax in enumerate(axes):
        im = ax.imshow(
            sinr[:, i, :].T,        # (N_sub, T)
            aspect="auto", origin="lower",
            extent=[0, T, 0, N_SUB],
            cmap="RdYlGn", vmin=vmin, vmax=vmax,
        )
        ax.axvline(20, color="white", linestyle="--", linewidth=1.0, label="Barrage starts")
        ax.set_ylabel("Subcarrier")
        ax.set_title(f"Defender {i+1}")
        fig.colorbar(im, ax=ax, label="SINR (dB)")

    axes[-1].set_xlabel("Step")

    path = os.path.join(out_dir, "exp3_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="mock", choices=["mock", "sionna"])
    parser.add_argument("--out",     default="plots/multi_agent")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    exp1_sweep_jammers(args.channel, args.out)
    exp2_sweep_defenders(args.channel, args.out)
    exp3_heatmap(args.channel, args.out)

    print(f"\nAll plots saved to {args.out}/")
