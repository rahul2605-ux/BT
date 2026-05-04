"""
simulations/train_ctde.py
--------------------------
Stage 1 CTDE baseline: N jammers trained with parameter-sharing PPO.

Each jammer acts on its own local obs (decentralized execution);
the critic sees the full global state (centralized training).
This is the MLP baseline whose permutation sensitivity is exposed in Stage 2.

Usage
-----
    # Local (MockChannel, N_jam=3)
    PYTHONPATH=. conda run -n sionna-thesis python simulations/train_ctde.py

    # Colab (SionnaChannel, N_jam=3)
    python simulations/train_ctde.py --channel-mode sionna --n-jammers 3 --total-steps 1000000

    # Evaluate saved model
    python simulations/train_ctde.py --eval-only models/ctde/best/best_model --n-jammers 3
"""

import argparse
import os
import subprocess

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from core.config import AgentRole, AgentSpec, EnvironmentConfig, OFDMConfig, ScatterMode
from envs.wireless_env import WirelessEnv
from policies.ctde_mlp import CTDEMlpPolicy
from simulations.train_stage1 import MetricsCallback, _tb_available, evaluate


class GpuStatsCallback(BaseCallback):
    def __init__(self, every_n_steps: int = 5000):
        super().__init__()
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                gpu_pct, mem_used, mem_total = r.stdout.strip().split(", ")
                print(f"  [GPU] step {self.num_timesteps:>8,} | "
                      f"util {gpu_pct}% | VRAM {mem_used}/{mem_total} MiB")
        return True


N_SUB  = 76
SEED   = 42

TRAIN_STEPS = 1_000_000
EVAL_FREQ   = 20_000
N_EVAL_EPS  = 10
SAVE_DIR    = "models/ctde"
LOG_DIR     = "logs/ctde"


def make_env(n_jammers: int, channel_mode: str, seed: int) -> WirelessEnv:
    config = EnvironmentConfig(
        legitimate=AgentSpec(
            role=AgentRole.LEGITIMATE, count=1,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM, max_power_dbm=20.0,
        ),
        jammers=AgentSpec(
            role=AgentRole.JAMMER, count=n_jammers,
            ofdm=OFDMConfig(n_subcarriers=N_SUB),
            scatter_mode=ScatterMode.RANDOM, max_power_dbm=30.0,
            policy=None,
        ),
        channel_mode=channel_mode, max_steps=200, seed=seed,
    )
    return WirelessEnv(config)


def train(args) -> PPO:
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n_jam = args.n_jammers
    ch    = args.channel_mode

    train_env = make_vec_env(
        lambda: Monitor(make_env(n_jam, ch, SEED)),
        n_envs=args.n_envs, seed=SEED,
    )
    eval_env = Monitor(make_env(n_jam, ch, SEED + 1))

    model = PPO(
        policy=CTDEMlpPolicy,
        env=train_env,
        device="cpu",
        policy_kwargs=dict(
            n_jammers=n_jam,
            n_subcarriers=N_SUB,
            hidden_size=256,
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR if _tb_available() else None,
        seed=SEED,
    )

    metrics_path = os.path.join(LOG_DIR, "metrics")
    callbacks = [
        MetricsCallback(save_path=metrics_path),
        GpuStatsCallback(every_n_steps=5000),
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
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=os.path.join(SAVE_DIR, "checkpoints"),
            name_prefix=f"ctde_n{n_jam}",
        ),
    ]

    print(f"\n=== CTDE PPO training: N_jam={n_jam} | {args.total_steps:,} steps ===")
    print(f"    obs={train_env.observation_space.shape} | act={train_env.action_space.shape}\n")

    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=False)

    final_path = os.path.join(SAVE_DIR, f"ctde_n{n_jam}_final")
    model.save(final_path)
    print(f"\nSaved → {final_path}.zip")

    train_env.close()
    eval_env.close()

    # Auto-plot
    from simulations.plot_stage1 import generate_all
    best = os.path.join(SAVE_DIR, "best", "best_model")
    use  = best if os.path.exists(best + ".zip") else final_path
    generate_all(
        model_path=use,
        metrics_path=metrics_path + ".npz",
        out_dir=f"plots/ctde_n{n_jam}",
        channel_mode=ch,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jammers",    type=int,   default=3)
    parser.add_argument("--total-steps",  type=int,   default=TRAIN_STEPS)
    parser.add_argument("--n-envs",       type=int,   default=4)
    parser.add_argument("--channel-mode", default="mock", choices=["mock", "sionna"])
    parser.add_argument("--eval-only",    type=str,   default=None)
    args = parser.parse_args()

    if args.eval_only:
        env = make_env(args.n_jammers, args.channel_mode, SEED + 99)
        model = PPO.load(
            args.eval_only,
            custom_objects={"policy_class": CTDEMlpPolicy},
        )
        evaluate(args.eval_only, n_episodes=20)
    else:
        train(args)
