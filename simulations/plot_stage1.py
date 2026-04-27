"""
simulations/plot_stage1.py
--------------------------
Generates and saves plots from a Stage 1 training run.

Can be called automatically at the end of train_stage1.py,
or run standalone after training:

    PYTHONPATH=. python simulations/plot_stage1.py \
        --model models/stage1/best/best_model \
        --log   logs/stage1/metrics.npz \
        --out   plots/stage1
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.config import AgentRole, AgentSpec, EnvironmentConfig, OFDMConfig, ScatterMode
from core.strategy import RandomStrategy, TimedBarrageJamming
from envs.wireless_env import WirelessEnv
from wrappers.sinr_normalize import SINRNormalizeObservation
from wrappers.rescale_action import PowerRescaleAction

N_SUB = 76
SEED  = 99  # different from training seed

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_eval_env(channel_mode: str = "mock", seed: int = SEED) -> WirelessEnv:
    config = EnvironmentConfig(
        legitimate=AgentSpec(
            role=AgentRole.LEGITIMATE, count=1,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM, max_power_dbm=20.0,
        ),
        jammers=AgentSpec(
            role=AgentRole.JAMMER, count=1,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM, max_power_dbm=30.0,
        ),
        channel_mode=channel_mode, max_steps=200, seed=seed,
    )
    env = WirelessEnv(config)
    env = SINRNormalizeObservation(env, n_subcarriers=N_SUB)
    env = PowerRescaleAction(env)
    return env


# ---------------------------------------------------------------------------
# Strategy runners  (returns per-step SINR arrays and per-step actions)
# ---------------------------------------------------------------------------

def _run_policy(model, env, n_episodes: int = 20):
    """Run a trained SB3 model. Returns (sinr_per_step, actions_per_step)."""
    all_sinr, all_actions = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get("mean_sinr_db") is not None:
                all_sinr.append(info["mean_sinr_db"])
            # Recover original dBm action for plotting
            wrapped_act = env.reverse_action(action)   # back to [0, 30]
            all_actions.append(wrapped_act)
    return np.array(all_sinr), np.array(all_actions)


def _run_fixed_strategy(strategy, channel_mode: str = "mock", n_episodes: int = 20):
    """Run a fixed BaseStrategy. Returns sinr array."""
    config = EnvironmentConfig(
        legitimate=AgentSpec(role=AgentRole.LEGITIMATE, count=1,
                             ofdm=OFDMConfig(n_subcarriers=N_SUB),
                             scatter_mode=ScatterMode.RANDOM, max_power_dbm=20.0),
        jammers=AgentSpec(role=AgentRole.JAMMER, count=1,
                          ofdm=OFDMConfig(n_subcarriers=N_SUB),
                          scatter_mode=ScatterMode.RANDOM, max_power_dbm=30.0,
                          policy=strategy),
        channel_mode=channel_mode, max_steps=200, seed=SEED,
    )
    env = WirelessEnv(config)
    all_sinr = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        strategy.reset()
        done = False
        while not done:
            action = strategy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get("mean_sinr_db") is not None:
                all_sinr.append(info["mean_sinr_db"])
    env.close()
    return np.array(all_sinr)


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Plot 1 — Training curves (reward + SINR)
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_path: str, out_dir: str) -> str:
    data = np.load(metrics_path)
    steps   = data["steps"]
    rewards = data["ep_rewards"]
    sinr    = data["ep_sinr"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    fig.suptitle("Stage 1 PPO — Training Curves", fontsize=12)

    ax1.plot(steps, rewards, alpha=0.25, color="steelblue", linewidth=0.8)
    if len(rewards) > 50:
        ax1.plot(steps[49:], _smooth(rewards), color="steelblue", linewidth=1.8, label="smoothed")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(fontsize=8)

    ax2.plot(steps, sinr, alpha=0.25, color="tomato", linewidth=0.8)
    if len(sinr) > 50:
        ax2.plot(steps[49:], _smooth(sinr), color="tomato", linewidth=1.8, label="smoothed")
    ax2.set_ylabel("Mean SINR at Defender (dB)")
    ax2.set_xlabel("Environment Steps")
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()  # lower SINR = better jamming, so show it going "up"

    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 2 — Strategy comparison (box plots per subcarrier)
# ---------------------------------------------------------------------------

def plot_policy_comparison(
    model_path: str,
    out_dir: str,
    channel_mode: str = "mock",
    n_episodes: int = 30,
) -> str:
    from stable_baselines3 import PPO

    env = _make_eval_env(channel_mode=channel_mode)

    model    = PPO.load(model_path)
    ppo_sinr, ppo_actions = _run_policy(model, env, n_episodes)
    env.close()

    barrage_sinr  = _run_fixed_strategy(
        TimedBarrageJamming(N_SUB, silent_steps=0, max_power_dbm=30.0),
        channel_mode=channel_mode, n_episodes=n_episodes,
    )
    random_sinr   = _run_fixed_strategy(
        RandomStrategy(
            WirelessEnv(EnvironmentConfig(channel_mode=channel_mode, seed=SEED)).action_space
        ),
        channel_mode=channel_mode, n_episodes=n_episodes,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    fig.suptitle("Stage 1 — Strategy Comparison", fontsize=12)

    # Left: SINR distribution per strategy
    ax = axes[0]
    data   = [ppo_sinr, barrage_sinr, random_sinr]
    labels = ["PPO (learned)", "Barrage (full power)", "Random"]
    colors = ["steelblue", "tomato", "grey"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean SINR at Defender (dB)")
    ax.set_title("SINR distribution\n(lower = more effective jamming)")

    # Right: mean per-subcarrier power of learned policy
    ax2 = axes[1]
    if len(ppo_actions) > 0:
        mean_power = ppo_actions.mean(axis=0)  # (N_sub,)
        ax2.bar(np.arange(N_SUB), mean_power, color="steelblue", alpha=0.7, width=1.0)
        ax2.set_xlabel("Subcarrier index")
        ax2.set_ylabel("Mean TX power (dBm)")
        ax2.set_title("PPO learned power allocation\n(per subcarrier)")
        ax2.set_ylim(0, 35)

    path = os.path.join(out_dir, "policy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 3 — SINR heatmap over an episode (PPO vs barrage)
# ---------------------------------------------------------------------------

def plot_episode_heatmap(
    model_path: str,
    out_dir: str,
    channel_mode: str = "mock",
) -> str:
    from stable_baselines3 import PPO

    def _collect_sinr_trace(policy_fn, env):
        obs, _ = env.reset()
        trace = []
        done = False
        while not done:
            action = policy_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trace.append(obs[:N_SUB].copy())  # normalised SINR slice
        return np.stack(trace).T  # (N_sub, T)

    model = PPO.load(model_path)
    env_ppo = _make_eval_env(channel_mode=channel_mode)
    env_bar = _make_eval_env(channel_mode=channel_mode)

    ppo_trace = _collect_sinr_trace(
        lambda obs: model.predict(obs, deterministic=True)[0], env_ppo
    )
    bar_trace = _collect_sinr_trace(
        lambda obs: env_bar.action_space.high * np.ones(N_SUB),  # full barrage
        env_bar,
    )
    env_ppo.close()
    env_bar.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    fig.suptitle("SINR per Subcarrier over Episode (normalised [0,1])", fontsize=11)

    for ax, data, title in [
        (ax1, ppo_trace, "PPO (learned)"),
        (ax2, bar_trace, "Barrage (full power)"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[0, data.shape[1], 0, N_SUB],
                       vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_ylabel("Subcarrier")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Norm. SINR")

    ax2.set_xlabel("Step")

    path = os.path.join(out_dir, "episode_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all(
    model_path: str,
    metrics_path: str | None,
    out_dir: str,
    channel_mode: str = "mock",
    n_episodes: int = 30,
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Generating plots → {out_dir} ===")

    if metrics_path and os.path.exists(metrics_path):
        plot_training_curves(metrics_path, out_dir)
    else:
        print("  [skip] training_curves — no metrics file found")

    plot_policy_comparison(model_path, out_dir, channel_mode, n_episodes)
    plot_episode_heatmap(model_path, out_dir, channel_mode)
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="models/stage1/best/best_model")
    parser.add_argument("--log",     default="logs/stage1/metrics.npz")
    parser.add_argument("--out",     default="plots/stage1")
    parser.add_argument("--channel", default="mock", choices=["mock", "sionna"])
    parser.add_argument("--n-eps",   type=int, default=30)
    args = parser.parse_args()

    generate_all(args.model, args.log, args.out, args.channel, args.n_eps)
