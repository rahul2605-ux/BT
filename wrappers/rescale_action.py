"""
wrappers/rescale_action.py
--------------------------
Maps the policy's [-1, 1] action output to the env's [0, max_power_dbm] space.

Most RL policy networks use a tanh output head, which saturates at ±1.
The env action space is per-subcarrier power in dBm (e.g. [0, 30]).
Without this wrapper the policy wastes half its output range — everything
negative gets clipped to 0 — and SB3 warns about out-of-bounds actions.

The mapping is a simple linear stretch:
    env_action = low + (policy_action + 1) / 2 * (high - low)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PowerRescaleAction(gym.ActionWrapper):
    """
    Wraps an env whose action space is Box([low, high]) and presents
    Box([-1, 1]) to the policy.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low  = env.action_space.low.copy()
        self._high = env.action_space.high.copy()
        self._half_range = 0.5 * (self._high - self._low)

        self.action_space = spaces.Box(
            low  = -np.ones_like(self._low),
            high =  np.ones_like(self._high),
            dtype=np.float32,
        )

    def action(self, act: np.ndarray) -> np.ndarray:
        """[-1, 1]  →  [low, high]  (policy → env)"""
        return (self._low + (act + 1.0) * self._half_range).astype(np.float32)

    def reverse_action(self, act: np.ndarray) -> np.ndarray:
        """[low, high]  →  [-1, 1]  (env → policy, useful for logging)"""
        return ((act - self._low) / self._half_range - 1.0).astype(np.float32)
