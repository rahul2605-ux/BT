"""
channel/base.py
---------------
BaseChannel ABC + MockChannel.

Interface
---------
get_coefficients(tx_pos, rx_pos, jam_pos, n_sub)
    → H_tx  : (N_rx, N_tx,  N_sub) complex64   TX-to-RX channels
    → H_jam : (N_rx, N_jam, N_sub) complex64   jammer-to-RX channels

    H[rx, src, sub] = total complex channel coefficient (path loss + fading).

compute_sinr(...)           — power-only actions, incoherent combining (Stage 1, backward compat)
compute_sinr_complex(...)   — complex signals, coherent combining (Stage 3+)

Why complex coefficients?
    With only power-level actions the jammer interference adds incoherently:
        received power  ∝  Σ_jam |H_jam|² · P_jam      (∝ N jammers)
    With complex signals and coordinated phases it can add coherently:
        received power  ∝  |Σ_jam H_jam · j_jam|²      (∝ N² jammers)
    compute_sinr_complex() implements the coherent formula once actions carry phase.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseChannel(ABC):

    @abstractmethod
    def get_coefficients(
        self,
        tx_positions:  np.ndarray,   # (N_tx,  3)
        rx_positions:  np.ndarray,   # (N_rx,  3)
        jam_positions: np.ndarray,   # (N_jam, 3)
        n_subcarriers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return complex channel matrices for one timestep.

        Returns
        -------
        H_tx  : (N_rx, N_tx,  N_sub) complex64
        H_jam : (N_rx, N_jam, N_sub) complex64
        """
        ...

    # ------------------------------------------------------------------
    # Concrete SINR methods — subclasses get these for free
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Static SINR helpers — operate on pre-computed H matrices.
    # Use these when the same H must be shared between observation
    # building and SINR computation (i.e. every env.step() call).
    # ------------------------------------------------------------------

    @staticmethod
    def sinr_from_power(
        H_tx:            np.ndarray,   # (N_rx, N_tx,  N_sub) complex64
        H_jam:           np.ndarray,   # (N_rx, N_jam, N_sub) complex64
        tx_power_dbm:    np.ndarray,   # (N_tx,)
        jam_power_dbm:   np.ndarray,   # (N_jam, N_sub) real
        noise_power_dbm: float,
    ) -> np.ndarray:                   # (N_rx, N_sub) dB
        """
        Incoherent SINR from pre-computed H matrices.
        Interference = Σ_jam |H_jam|² · P_jam  (powers add independently).
        """
        tx_power_lin = 10 ** (tx_power_dbm / 10.0)
        signal_lin = np.sum(
            np.abs(H_tx) ** 2 * tx_power_lin[np.newaxis, :, np.newaxis],
            axis=1,
        )  # (N_rx, N_sub)

        if H_jam.shape[1] > 0:
            jam_power_lin = 10 ** (jam_power_dbm / 10.0)
            jam_lin = np.sum(
                np.abs(H_jam) ** 2 * jam_power_lin[np.newaxis, :, :],
                axis=1,
            )  # (N_rx, N_sub)
        else:
            jam_lin = np.zeros(H_tx.shape[::2], dtype=np.float32)  # (N_rx, N_sub)

        noise_lin = 10 ** (noise_power_dbm / 10.0)
        sinr_lin  = signal_lin / (jam_lin + noise_lin)
        return (10.0 * np.log10(np.clip(sinr_lin, 1e-10, None))).astype(np.float32)

    @staticmethod
    def sinr_from_signals(
        H_tx:            np.ndarray,   # (N_rx, N_tx,  N_sub) complex64
        H_jam:           np.ndarray,   # (N_rx, N_jam, N_sub) complex64
        tx_power_dbm:    np.ndarray,   # (N_tx,)
        jam_signals:     np.ndarray,   # (N_jam, N_sub) complex — jammer action
        noise_power_dbm: float,
    ) -> np.ndarray:                   # (N_rx, N_sub) dB
        """
        Coherent SINR from pre-computed H matrices and complex jam signals.

        Interference = |Σ_jam H_jam[rx,jam,sub] · j[jam,sub]|²
                              ↑ coherent field — phase coordination matters

        This is the correct call path for Phase 1.3+: the env calls
        get_coefficients() once per step, builds observations from H, then
        passes H and the policy action here to compute SINR.  Using the same
        H for both obs and SINR ensures the jammer coordinates against the
        channel it actually observes.
        """
        tx_power_lin = 10 ** (tx_power_dbm / 10.0)
        signal_lin = np.sum(
            np.abs(H_tx) ** 2 * tx_power_lin[np.newaxis, :, np.newaxis],
            axis=1,
        )  # (N_rx, N_sub)

        if H_jam.shape[1] > 0:
            jam_field = np.einsum(
                'rjs,js->rs',
                H_jam,
                jam_signals.astype(np.complex64),
            )  # (N_rx, N_sub) complex
            jam_lin = np.abs(jam_field) ** 2
        else:
            jam_lin = np.zeros(H_tx.shape[::2], dtype=np.float32)

        noise_lin = 10 ** (noise_power_dbm / 10.0)
        sinr_lin  = signal_lin / (jam_lin + noise_lin)
        return (10.0 * np.log10(np.clip(sinr_lin, 1e-10, None))).astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience wrappers — call get_coefficients() then the static helpers.
    # Suitable for Stage 1 (power actions) and for scripts that don't need
    # to share H between obs-building and SINR computation.
    # ------------------------------------------------------------------

    def compute_sinr(
        self,
        tx_positions:    np.ndarray,
        rx_positions:    np.ndarray,
        jam_positions:   np.ndarray,
        tx_power_dbm:    np.ndarray,
        jam_power_dbm:   np.ndarray,
        noise_power_dbm: float,
        n_subcarriers:   int,
    ) -> np.ndarray:
        """Backward-compatible power-only SINR (Stage 1 / existing code)."""
        H_tx, H_jam = self.get_coefficients(
            tx_positions, rx_positions, jam_positions, n_subcarriers
        )
        return self.sinr_from_power(
            H_tx, H_jam, tx_power_dbm, jam_power_dbm, noise_power_dbm
        )

    def compute_sinr_complex(
        self,
        tx_positions:    np.ndarray,
        rx_positions:    np.ndarray,
        jam_positions:   np.ndarray,
        tx_power_dbm:    np.ndarray,
        jam_signals:     np.ndarray,
        noise_power_dbm: float,
        n_subcarriers:   int,
    ) -> np.ndarray:
        """Coherent SINR wrapper — calls get_coefficients() internally.
        Only use when you do NOT need to share H with observation building.
        Otherwise call get_coefficients() + sinr_from_signals() directly."""
        H_tx, H_jam = self.get_coefficients(
            tx_positions, rx_positions, jam_positions, n_subcarriers
        )
        return self.sinr_from_signals(
            H_tx, H_jam, tx_power_dbm, jam_signals, noise_power_dbm
        )


class MockChannel(BaseChannel):
    """
    Fast channel for local dev and unit tests.

    Path loss:  PL(d) = PL0 + 10·α·log10(d/d0)  [dB]
    Phase:      uniform random per (link, subcarrier) — models rich scattering

    Behaviour of compute_sinr() is identical to the old implementation because
    |exp(j·φ)|² = 1, so random phases cancel when squaring for power.
    compute_sinr_complex() uses the phases — but since they are re-drawn each call,
    a jammer cannot coordinate against MockChannel. Use SionnaChannel for
    phase-coordinated jamming experiments (it returns consistent CDL-D phases).
    """

    def __init__(
        self,
        alpha:  float = 3.5,
        pl0_db: float = 40.0,
        d0_m:   float = 1.0,
        seed:   int | None = None,
    ):
        self.alpha  = alpha
        self.pl0_db = pl0_db
        self.d0_m   = d0_m
        self._rng   = np.random.default_rng(seed)

    def _path_loss_db(
        self, pos_a: np.ndarray, pos_b: np.ndarray
    ) -> np.ndarray:
        """Pairwise path loss (M, N) in dB.  pos_a (M,3), pos_b (N,3)."""
        diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1).clip(self.d0_m, None)
        return self.pl0_db + 10.0 * self.alpha * np.log10(dist / self.d0_m)

    def get_coefficients(
        self,
        tx_positions:  np.ndarray,
        rx_positions:  np.ndarray,
        jam_positions: np.ndarray,
        n_subcarriers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        N_tx  = tx_positions.shape[0]
        N_rx  = rx_positions.shape[0]
        N_jam = jam_positions.shape[0] if len(jam_positions) > 0 else 0

        # TX → RX
        pl_tx  = self._path_loss_db(tx_positions, rx_positions)   # (N_tx, N_rx)
        amp_tx = 10 ** (-pl_tx.T / 20.0)                          # (N_rx, N_tx) voltage amp
        # Random phase per (rx, tx, subcarrier) — models rich scattering
        phi_tx = self._rng.uniform(0.0, 2 * np.pi, (N_rx, N_tx, n_subcarriers))
        H_tx   = (amp_tx[:, :, np.newaxis] * np.exp(1j * phi_tx)).astype(np.complex64)

        # Jammer → RX
        if N_jam > 0:
            pl_jam  = self._path_loss_db(jam_positions, rx_positions)   # (N_jam, N_rx)
            amp_jam = 10 ** (-pl_jam.T / 20.0)                          # (N_rx, N_jam)
            phi_jam = self._rng.uniform(0.0, 2 * np.pi, (N_rx, N_jam, n_subcarriers))
            H_jam   = (amp_jam[:, :, np.newaxis] * np.exp(1j * phi_jam)).astype(np.complex64)
        else:
            H_jam = np.zeros((N_rx, 0, n_subcarriers), dtype=np.complex64)

        return H_tx, H_jam
