"""
wrappers/sinr_normalize.py
--------------------------
Normalises the SINR portion of the observation from raw dB to [0, 1]
using a fixed physics-based clip range.

The observation layout is:  [SINR x n_sub | position x 3 | step_frac x 1]
Only the first n_sub elements are SINR — the rest are already in [0, 1]
and are passed through untouched.

Why fixed range instead of running stats (gymnasium.NormalizeObservation):
  The SINR distribution shifts as the jammer policy improves during training,
  so a running mean/std chases a moving target and can destabilise early
  training. A fixed clip range grounded in physics is more robust.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SINRNormalizeObservation(gym.ObservationWrapper):
    """
    Parameters
    ----------
    env : gym.Env
    n_subcarriers : int
        Number of leading observation elements that are SINR values.
    sinr_min_db : float
        SINR below this is clipped to 0 (fully jammed floor).
    sinr_max_db : float
        SINR above this is clipped to 1 (clean channel ceiling).
    """

    def __init__(
        self,
        env: gym.Env,
        n_subcarriers: int,
        sinr_min_db: float = -20.0,
        sinr_max_db: float = 40.0,
    ):
        super().__init__(env)
        self.n_sub = n_subcarriers
        self.sinr_min = sinr_min_db
        self.sinr_max = sinr_max_db
        self._range = sinr_max_db - sinr_min_db

        # Update the observation space bounds for the SINR slice only
        low  = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        low[:n_subcarriers]  = 0.0
        high[:n_subcarriers] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[:self.n_sub] = np.clip(
            (obs[:self.n_sub] - self.sinr_min) / self._range,
            0.0, 1.0,
        )
        return obs.astype(np.float32)
