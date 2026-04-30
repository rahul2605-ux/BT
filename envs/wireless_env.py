"""
envs/wireless_env.py
--------------------
Multi-agent wireless environment.

Flat gymnasium-compatible interface so standard SB3 wrappers work out of the box.
Internally supports N jammers, each with its own action vector.

Observation layout  (shape = N_jam * (N_sub + 3) + 1):
  [SINR(N_sub)] × N_jam    — shared defender SINR, repeated once per jammer
                              (same signal, different slot so each jammer's
                               position slot lines up in the reshape below)
  [norm_pos(3)] × N_jam    — each jammer's own normalised position
  [step_frac(1)]           — current_step / max_steps

  Reshape to (N_jam, N_sub + 3) to get per-jammer tokens for Set Transformer.
  For N_jam=1 this collapses to the old (80,) obs — fully backward compatible.

Action layout  (shape = N_jam * N_sub):
  Per-jammer, per-subcarrier power in dBm, range [0, max_power_dbm].
  Reshape to (N_jam, N_sub) inside step().
  For N_jam=1 this is the old (76,) action — fully backward compatible.

Channel matrices H_tx, H_jam are computed once per step and stored on the
instance so they can be re-used for complex-signal SINR (Phase 1.4+) without
re-calling the channel model.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.config import AgentRole, EnvironmentConfig, ScatterMode
from core.scatter import make_scatter, SymmetricScatter, CustomScatter
from channel.base import BaseChannel, MockChannel


# ---------------------------------------------------------------------------
# Channel factory
# ---------------------------------------------------------------------------

def _build_channel(config: EnvironmentConfig) -> BaseChannel:
    if config.channel_mode == "mock":
        return MockChannel(seed=config.seed)
    elif config.channel_mode == "sionna":
        from channel.sionna_channel import SionnaChannel
        return SionnaChannel(
            carrier_frequency_hz=config.legitimate.ofdm.carrier_frequency_hz,
            n_subcarriers=config.legitimate.ofdm.n_subcarriers,
            subcarrier_spacing_hz=config.legitimate.ofdm.subcarrier_spacing_hz,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown channel_mode: {config.channel_mode!r}")


# ---------------------------------------------------------------------------
# WirelessEnv
# ---------------------------------------------------------------------------

class WirelessEnv(gym.Env):
    """
    Flat gym.Env controlling the jammer team against passive defenders.

    Reward: −mean(SINR_dB)   jammer wants SINR low at the defender.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config      = config
        self._rng        = np.random.default_rng(config.seed)
        self._channel    = _build_channel(config)

        self._n_sub      = config.legitimate.ofdm.n_subcarriers
        self._n_leg      = config.legitimate.count
        self._n_jam      = config.jammers.count
        self._max_power  = config.jammers.max_power_dbm
        self._noise_dbm  = -90.0   # thermal noise floor — confirm value with ADM

        # Positions (populated on reset)
        self.leg_positions: np.ndarray = np.zeros((self._n_leg, 3), dtype=np.float32)
        self.jam_positions: np.ndarray = np.zeros((self._n_jam, 3), dtype=np.float32)

        self._leg_scatter = self._build_scatter(config.legitimate)
        self._jam_scatter = self._build_scatter(config.jammers)

        # Channel matrices — updated every step, stored for reuse in Phase 1.4+
        self._H_tx:  Optional[np.ndarray] = None  # (N_rx, N_tx,  N_sub) complex64
        self._H_jam: Optional[np.ndarray] = None  # (N_rx, N_jam, N_sub) complex64

        # Gym spaces
        # obs: [SINR×N_jam | pos×N_jam | step_frac]
        obs_dim = self._n_jam * (self._n_sub + 3) + 1
        # act: [power_jam0 | power_jam1 | ... ]
        act_dim = self._n_jam * self._n_sub

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=self._max_power, shape=(act_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._last_sinr: Optional[np.ndarray] = None   # (N_rx, N_sub)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scatter(self, spec):
        mode = spec.scatter_mode.value
        if mode == "symmetric":
            side = "left" if spec.role == AgentRole.LEGITIMATE else "right"
            return SymmetricScatter(side=side)
        if mode == "custom":
            return CustomScatter(spec.positions)
        return make_scatter("random")

    def _compute_observation(self) -> np.ndarray:
        """
        Build the flat observation vector.

        Layout: [SINR(N_sub)] * N_jam | [norm_pos(3)] * N_jam | [step_frac]

        The N_jam copies of SINR carry the same values — they're duplicated so
        that each jammer's token (row in a (N_jam, N_sub+3) reshape) contains
        the shared feedback signal + its own spatial coordinate.
        """
        sinr = (
            self._last_sinr[0].astype(np.float32)   # first defender RX
            if self._last_sinr is not None
            else np.zeros(self._n_sub, dtype=np.float32)
        )
        bounds    = np.array(self.config.space_bounds, dtype=np.float32)
        step_frac = np.array([self._step_count / self.config.max_steps], dtype=np.float32)

        sinr_block = np.tile(sinr, self._n_jam)                          # (N_jam * N_sub,)
        pos_block  = (self.jam_positions / bounds).flatten()             # (N_jam * 3,)
        return np.concatenate([sinr_block, pos_block, step_frac])        # (N_jam*(N_sub+3)+1,)

    def _jammer_obs_for_policy(self, jammer_idx: int) -> np.ndarray:
        """Local obs for a fixed-strategy policy (evaluation mode)."""
        sinr = (
            self._last_sinr[0]
            if self._last_sinr is not None
            else np.zeros(self._n_sub, dtype=np.float32)
        )
        bounds    = np.array(self.config.space_bounds, dtype=np.float32)
        norm_pos  = self.jam_positions[jammer_idx] / bounds
        step_frac = np.array([self._step_count / self.config.max_steps], dtype=np.float32)
        return np.concatenate([sinr.astype(np.float32), norm_pos, step_frac])

    def _build_info(self) -> dict:
        return {
            "step":         self._step_count,
            "leg_positions": self.leg_positions.copy(),
            "jam_positions": self.jam_positions.copy(),
            "mean_sinr_db": (
                float(np.mean(self._last_sinr))
                if self._last_sinr is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        for spec in (self.config.legitimate, self.config.jammers):
            if spec.policy is not None:
                spec.policy.reset()

        self._step_count = 0
        self._last_sinr  = None
        self._H_tx       = None
        self._H_jam      = None

        bounds = self.config.space_bounds
        self.leg_positions = self._leg_scatter.place(self._n_leg, bounds, self._rng)
        self.jam_positions = self._jam_scatter.place(self._n_jam, bounds, self._rng)

        return self._compute_observation(), self._build_info()

    def step(self, action: np.ndarray):
        self._step_count += 1
        tx_power_dbm = np.full(self._n_leg, self.config.legitimate.max_power_dbm)

        # --- Resolve per-jammer action ---
        if self.config.jammers.policy is not None:
            # Evaluation mode: fixed strategy, call once per jammer
            # (most built-in strategies produce (N_sub,) per call)
            rows = [
                np.clip(
                    self.config.jammers.policy.act(self._jammer_obs_for_policy(i)),
                    0.0, self._max_power,
                )
                for i in range(self._n_jam)
            ]
            jam_power_dbm = np.stack(rows).astype(np.float32)  # (N_jam, N_sub)
        else:
            # Training mode: RL framework supplies flat (N_jam * N_sub,) action
            jam_power_dbm = (
                np.clip(action, 0.0, self._max_power)
                .astype(np.float32)
                .reshape(self._n_jam, self._n_sub)
            )

        # --- Channel: compute H once, store for reuse (Phase 1.4+) ---
        # TODO: leg_positions used for both TX and RX — needs separate arrays
        #       once the TX/RX system model is confirmed with ADM.
        self._H_tx, self._H_jam = self._channel.get_coefficients(
            tx_positions=self.leg_positions,
            rx_positions=self.leg_positions,
            jam_positions=self.jam_positions,
            n_subcarriers=self._n_sub,
        )

        self._last_sinr = BaseChannel.sinr_from_power(
            self._H_tx, self._H_jam,
            tx_power_dbm, jam_power_dbm,
            self._noise_dbm,
        )  # (N_rx, N_sub)

        reward   = float(-np.mean(self._last_sinr))
        truncated = self._step_count >= self.config.max_steps

        return self._compute_observation(), reward, False, truncated, self._build_info()

    def render(self):
        if self._last_sinr is None:
            print("No data yet — call reset() first.")
            return
        print(
            f"Step {self._step_count:3d} | "
            f"Mean SINR: {np.mean(self._last_sinr):+6.2f} dB | "
            f"N_jam={self._n_jam} | "
            f"JAM[0] pos: {self.jam_positions[0].round(1)}"
        )
