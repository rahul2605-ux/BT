"""
envs/wireless_env.py
--------------------
Multi-agent wireless environment skeleton.

Design decisions:
- Exposes a flat gymnasium-compatible interface for now so it can be used
  with standard SB3 wrappers. PettingZoo AEC API is a future extension.
- The environment is mode-aware: training (jammers.policy=None) vs evaluation.
- Channel backend is fully swappable via EnvironmentConfig.channel_mode.

TODOs:
- Wire in SionnaChannel when channel_mode == "sionna" (Euler only).
- Add mobility: agents currently remain at reset positions each episode.
- Separate TX and RX positions for legitimate nodes (currently same array).
- Extend to true multi-agent dict interface (one obs/action per jammer).
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
        # Import only when needed — Sionna requires GPU (Euler cluster)
        # from channel.sionna_channel import SionnaChannel
        # return SionnaChannel(config)
        raise NotImplementedError(
            "SionnaChannel is not yet implemented. "
            "Use channel_mode='mock' for local and Colab development."
        )
    else:
        raise ValueError(f"Unknown channel_mode: {config.channel_mode!r}")


# ---------------------------------------------------------------------------
# WirelessEnv
# ---------------------------------------------------------------------------

class WirelessEnv(gym.Env):
    """
    Single-agent wrapper around the multi-agent wireless scenario.

    The RL agent controls the jammer team (attacker side).
    Legitimate nodes are passive (fixed TX power, no learned policy yet).

    Observation (from the first legitimate RX node — proxy for jammer feedback):
        SINR per subcarrier : (n_subcarriers,)   in dB
        Own position        : (3,)               normalised to [0, 1]
        Step fraction       : (1,)               current_step / max_steps

    Action (jammer team, training mode):
        Transmit power per subcarrier : (n_subcarriers,)  in dBm, [0, max_power_dbm]
        All jammers share this action for now (true MARL dict interface is future work).

    Reward: −mean(SINR) — jammer wants to minimise defender link quality.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._channel = _build_channel(config)

        self._n_sub = config.legitimate.ofdm.n_subcarriers
        self._n_leg = config.legitimate.count
        self._n_jam = config.jammers.count
        self._max_power = config.jammers.max_power_dbm
        self._noise_dbm = -90.0  # thermal noise floor — placeholder, confirm with ADM

        # Positions (populated on reset)
        self.leg_positions: np.ndarray = np.zeros((self._n_leg, 3), dtype=np.float32)
        self.jam_positions: np.ndarray = np.zeros((self._n_jam, 3), dtype=np.float32)

        self._leg_scatter = self._build_scatter(config.legitimate)
        self._jam_scatter = self._build_scatter(config.jammers)

        obs_dim = self._n_sub + 3 + 1  # SINR + normalised position + step fraction
        act_dim = self._n_sub           # per-subcarrier power

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=self._max_power, shape=(act_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._last_sinr: Optional[np.ndarray] = None

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
        sinr = (
            self._last_sinr[0]
            if self._last_sinr is not None
            else np.zeros(self._n_sub, dtype=np.float32)
        )
        norm_pos = (
            self.leg_positions[0]
            / np.array(self.config.space_bounds, dtype=np.float32)
        )
        step_frac = np.array(
            [self._step_count / self.config.max_steps], dtype=np.float32
        )
        return np.concatenate([sinr, norm_pos, step_frac])

    def _compute_jammer_observation(self) -> np.ndarray:
        """Placeholder: jammer sees its own position + last SINR reading."""
        norm_pos = (
            self.jam_positions[0]
            / np.array(self.config.space_bounds, dtype=np.float32)
        )
        sinr = (
            self._last_sinr[0]
            if self._last_sinr is not None
            else np.zeros(self._n_sub, dtype=np.float32)
        )
        step_frac = np.array(
            [self._step_count / self.config.max_steps], dtype=np.float32
        )
        return np.concatenate([sinr, norm_pos, step_frac])

    def _build_info(self) -> dict:
        return {
            "step": self._step_count,
            "leg_positions": self.leg_positions.copy(),
            "jam_positions": self.jam_positions.copy(),
            "mean_sinr_db": (
                float(np.mean(self._last_sinr))
                if self._last_sinr is not None
                else None
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
        self._last_sinr = None

        bounds = self.config.space_bounds
        self.leg_positions = self._leg_scatter.place(self._n_leg, bounds, self._rng)
        self.jam_positions = self._jam_scatter.place(self._n_jam, bounds, self._rng)

        return self._compute_observation(), self._build_info()

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Resolve jammer action
        if self.config.jammers.policy is not None:
            jam_action = self.config.jammers.policy.act(self._compute_jammer_observation())
        else:
            jam_action = np.clip(action, 0.0, self._max_power).astype(np.float32)

        # All jammers broadcast the same action for now
        jam_power_dbm = np.tile(jam_action, (self._n_jam, 1))  # (N_jam, N_sub)

        # Legitimate TX: fixed uniform power (defender side not trained yet)
        tx_power_dbm = np.full(self._n_leg, self.config.legitimate.max_power_dbm)

        # TODO: leg_positions used for both TX and RX — needs separate TX/RX once
        #       the system model is finalised with ADM.
        self._last_sinr = self._channel.compute_sinr(
            tx_positions=self.leg_positions,
            rx_positions=self.leg_positions,
            jam_positions=self.jam_positions,
            tx_power_dbm=tx_power_dbm,
            jam_power_dbm=jam_power_dbm,
            noise_power_dbm=self._noise_dbm,
            n_subcarriers=self._n_sub,
        )

        reward = float(-np.mean(self._last_sinr))  # jammer: minimise defender SINR
        truncated = self._step_count >= self.config.max_steps

        return self._compute_observation(), reward, False, truncated, self._build_info()

    def render(self):
        if self._last_sinr is None:
            print("No data yet — call reset() first.")
            return
        print(
            f"Step {self._step_count:3d} | "
            f"Mean SINR: {np.mean(self._last_sinr):+6.2f} dB | "
            f"LEG pos: {self.leg_positions[0].round(1)} | "
            f"JAM pos: {self.jam_positions[0].round(1)}"
        )
