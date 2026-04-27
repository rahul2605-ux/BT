"""
wrappers/fixed_opponent.py
--------------------------
Reduces a two-team dict env to a single-agent env by freezing one side
at a fixed BaseStrategy.

WHY THIS EXISTS
---------------
Once wireless_env exposes a multi-agent dict interface:
    obs_dict  = {"jammer": ..., "legitimate": ...}
    step(action_dict) -> obs_dict, reward_dict, terminated, truncated, info

standard SB3 training loops can't consume it directly. This wrapper freezes
the opponent at any BaseStrategy (e.g. TimedBarrageJamming) and exposes only
the learner's obs/action — so SB3 sees a normal single-agent gym.Env.

CURRICULUM TRAINING
-------------------
Phase 1: train jammer vs a fixed passive/barrage legitimate node
Phase 3: swap strategies, unfreeze defender — co-train both sides

NOTE: This wrapper targets the FUTURE multi-agent env interface. It will
raise NotImplementedError if called against the current flat WirelessEnv,
which does not yet return dict obs/rewards.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from core.strategy import BaseStrategy


class FixedOpponentWrapper(gym.Wrapper):
    """
    Parameters
    ----------
    env : gym.Env
        A multi-agent env whose step() returns (obs_dict, reward_dict, ...).
    opponent : BaseStrategy
        Fixed strategy for the frozen side.
    learner_key : str
        Which team the RL algorithm controls: "jammer" or "legitimate".
    """

    def __init__(
        self,
        env: gym.Env,
        opponent: BaseStrategy,
        learner_key: str = "jammer",
    ):
        if learner_key not in ("jammer", "legitimate"):
            raise ValueError("learner_key must be 'jammer' or 'legitimate'.")
        super().__init__(env)
        self.opponent = opponent
        self.learner_key = learner_key
        self._opp_key = "legitimate" if learner_key == "jammer" else "jammer"
        self._opp_obs: np.ndarray | None = None

        # Expose only the learner's spaces to SB3
        self.observation_space = env.observation_space[learner_key]
        self.action_space      = env.action_space[learner_key]

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self.opponent.reset()
        self._opp_obs = obs_dict[self._opp_key]
        return obs_dict[self.learner_key], info

    def step(self, action: np.ndarray):
        opp_action = self.opponent.act(self._opp_obs)
        action_dict = {
            self.learner_key: action,
            self._opp_key:    opp_action,
        }
        obs_dict, reward_dict, terminated, truncated, info = self.env.step(action_dict)
        self._opp_obs = obs_dict[self._opp_key]
        return (
            obs_dict[self.learner_key],
            reward_dict[self.learner_key],
            terminated,
            truncated,
            info,
        )
