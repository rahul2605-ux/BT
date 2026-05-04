"""
simulations/train_stage1.py
----------------------------
Stage 1 baseline: single jammer trained with SB3 PPO against a static defender.

Uses:
  - WirelessEnv (MockChannel, 1 jammer, 1 defender)
  - SINRNormalizeObservation + PowerRescaleAction wrappers
  - SB3 PPO with MLP policy (no custom architecture)

The goal is NOT a novel RL implementation — it's a reproducible baseline
whose weaknesses we document before building Stage 3.

Usage (local):
    PYTHONPATH=. conda run -n sionna-thesis python simulations/train_stage1.py

Usage (Colab):
    !git clone <repo_url>
    import sys; sys.path.insert(0, '/content/BT')
    %run /content/BT/simulations/train_stage1.py
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO


def _tb_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from core.config import AgentRole, AgentSpec, EnvironmentConfig, OFDMConfig, ScatterMode
from envs.wireless_env import WirelessEnv
from wrappers.sinr_normalize import SINRNormalizeObservation
from wrappers.rescale_action import PowerRescaleAction


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SUB = 76
SEED  = 42

TRAIN_STEPS   = 500_000   # increase for Colab/Euler runs
EVAL_FREQ     = 10_000    # evaluate every N env steps
N_EVAL_EPS    = 10        # episodes per evaluation
SAVE_DIR      = "models/stage1"
LOG_DIR       = "logs/stage1"


def make_env(seed: int = SEED, channel_mode: str = "mock") -> WirelessEnv:
    config = EnvironmentConfig(
        legitimate=AgentSpec(
            role=AgentRole.LEGITIMATE,
            count=1,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM,
            max_power_dbm=20.0,
        ),
        jammers=AgentSpec(
            role=AgentRole.JAMMER,
            count=1,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM,
            max_power_dbm=30.0,
            policy=None,  # RL controls this side
        ),
        channel_mode=channel_mode,
        max_steps=200,
        seed=seed,
    )
    env = WirelessEnv(config)
    env = SINRNormalizeObservation(env, n_subcarriers=N_SUB)
    env = PowerRescaleAction(env)
    return env


# ---------------------------------------------------------------------------
# Custom callback: log mean SINR (jammer's primary metric)
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """
    Tracks episode reward and mean SINR throughout training.
    Saves to a .npz file at the end so plot_stage1.py can read it.
    Also logs to TensorBoard if available.
    """

    def __init__(self, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self._ep_rewards: list[float] = []
        self._ep_sinr:    list[float] = []
        self._ep_steps:   list[int]   = []
        self._sinr_buf:   list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            sinr = info.get("mean_sinr_db")
            if sinr is not None:
                self._sinr_buf.append(sinr)
            # Monitor wraps episode end in "episode" key
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(float(ep["r"]))
                self._ep_sinr.append(
                    float(np.mean(self._sinr_buf)) if self._sinr_buf else float("nan")
                )
                self._ep_steps.append(self.num_timesteps)
                self._sinr_buf.clear()
                if _tb_available():
                    self.logger.record("custom/mean_sinr_db", self._ep_sinr[-1])
        return True

    def _on_training_end(self) -> None:
        np.savez(
            self.save_path,
            steps=np.array(self._ep_steps),
            ep_rewards=np.array(self._ep_rewards),
            ep_sinr=np.array(self._ep_sinr),
        )
        print(f"  Metrics saved → {self.save_path}.npz")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ch = getattr(args, "channel_mode", "mock")

    # Training env (vectorized for SB3)
    train_env = make_vec_env(
        lambda: Monitor(make_env(seed=SEED, channel_mode=ch)),
        n_envs=args.n_envs,
        seed=SEED,
    )

    # Separate env for evaluation (not used in training updates)
    eval_env = Monitor(make_env(seed=SEED + 1, channel_mode=ch))

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # small entropy bonus — stops policy collapsing to all-on or all-off
        verbose=1,
        tensorboard_log=LOG_DIR if _tb_available() else None,
        seed=SEED,
    )

    metrics_path = os.path.join(LOG_DIR, "metrics")
    callbacks = [
        MetricsCallback(save_path=metrics_path),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(SAVE_DIR, "best"),
            log_path=os.path.join(LOG_DIR, "eval"),
            eval_freq=max(EVAL_FREQ // args.n_envs, 1),
            n_eval_episodes=N_EVAL_EPS,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=os.path.join(SAVE_DIR, "checkpoints"),
            name_prefix="ppo_jammer",
        ),
    ]

    print(f"\n=== Stage 1 PPO training: {args.total_steps:,} steps ===")
    print(f"    n_envs={args.n_envs}  |  obs={train_env.observation_space.shape}  |  act={train_env.action_space.shape}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=False,
    )

    final_path = os.path.join(SAVE_DIR, "ppo_jammer_final")
    model.save(final_path)
    print(f"\nSaved final model → {final_path}.zip")

    train_env.close()
    eval_env.close()

    # Auto-generate plots
    from simulations.plot_stage1 import generate_all
    best_path = os.path.join(SAVE_DIR, "best", "best_model")
    eval_model = best_path if os.path.exists(best_path + ".zip") else final_path
    generate_all(
        model_path=eval_model,
        metrics_path=metrics_path + ".npz",
        out_dir="plots/stage1",
        channel_mode=args.channel_mode if hasattr(args, "channel_mode") else "mock",
    )

    return model


# ---------------------------------------------------------------------------
# Quick evaluation of a saved model
# ---------------------------------------------------------------------------

def evaluate(model_path: str, n_episodes: int = 20):
    model = PPO.load(model_path)
    env = make_env(seed=99)

    sinr_per_episode, rewards_per_episode = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_sinr = 0.0, []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info.get("mean_sinr_db") is not None:
                ep_sinr.append(info["mean_sinr_db"])
            done = terminated or truncated
        sinr_per_episode.append(np.mean(ep_sinr))
        rewards_per_episode.append(ep_reward)

    env.close()
    print(f"\n=== Evaluation over {n_episodes} episodes ===")
    print(f"  Mean episode reward : {np.mean(rewards_per_episode):.2f} ± {np.std(rewards_per_episode):.2f}")
    print(f"  Mean SINR (defender): {np.mean(sinr_per_episode):.2f} ± {np.std(sinr_per_episode):.2f} dB")
    print(f"  (Lower SINR = more effective jamming)")
    return np.mean(sinr_per_episode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--channel-mode", default="mock", choices=["mock", "sionna"],
                        help="Channel model (mock=local dev, sionna=Colab/Euler)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to saved model — skip training and just evaluate")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.eval_only)
    else:
        model = train(args)
        evaluate(os.path.join(SAVE_DIR, "best", "best_model"))
