from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseChannel(ABC):

    @abstractmethod
    def compute_sinr(
        self,
        tx_positions:   np.ndarray,   # (N_tx, 3)
        rx_positions:   np.ndarray,   # (N_rx, 3)
        jam_positions:  np.ndarray,   # (N_jam, 3)
        tx_power_dbm:   np.ndarray,   # (N_tx,)
        jam_power_dbm:  np.ndarray,   # (N_jam, N_subcarriers)
        noise_power_dbm: float,
        n_subcarriers:  int,
    ) -> np.ndarray:
        ...

class MockChannel(BaseChannel):
    """
    Extremely simplified SINR model for local development and unit tests.

    Path loss model:  PL(d) = PL0 + 10 * alpha * log10(d/d0)  [dB]

    This is NOT physically accurate — it is just fast and dependency-free.
    Replace with SionnaChannel once on Euler.

    Parameters
    ----------
    alpha : float
        Path loss exponent. Typical values: free space=2, urban=3.5.
    pl0_db : float
        Reference path loss at d0.
    d0_m : float
        Reference distance in metres.
    noise_floor_dbm : float
        Thermal noise power in dBm (default: -90 dBm).
    """

    def __init__(
        self,
        alpha: float = 3.5,
        pl0_db: float = 40.0,
        d0_m: float = 1.0,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.pl0_db = pl0_db
        self.d0_m = d0_m
        self._rng = np.random.default_rng(seed)

    def _path_loss_db(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise path loss matrix.

        Parameters
        ----------
        pos_a : (M, 3)
        pos_b : (N, 3)

        Returns
        -------
        pl : (M, N) path loss in dB
        """
        # Euclidean distance, clipped to avoid log(0)
        diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]  # (M, N, 3)
        dist = np.linalg.norm(diff, axis=-1)                        # (M, N)
        dist = np.clip(dist, self.d0_m, None)
        pl = self.pl0_db + 10.0 * self.alpha * np.log10(dist / self.d0_m)
        return pl   # (M, N)

    def compute_sinr(
        self,
        tx_positions,
        rx_positions,
        jam_positions,
        tx_power_dbm,
        jam_power_dbm,
        noise_power_dbm,
        n_subcarriers,
    ):
        N_rx  = rx_positions.shape[0]
        N_tx  = tx_positions.shape[0]
        N_jam = jam_positions.shape[0] if len(jam_positions) > 0 else 0

        # --- Signal power at each RX from each TX (broadcast across subcarriers)
        # pl_tx : (N_tx, N_rx)
        pl_tx = self._path_loss_db(tx_positions, rx_positions)
        # received_power : (N_rx, N_tx, 1) → broadcast over subcarriers
        rx_signal_dbm = (tx_power_dbm[np.newaxis, :] - pl_tx.T)  # (N_rx, N_tx)
        # Sum legitimate signal power (linear, then back to dB)
        rx_signal_lin = np.sum(
            10 ** (rx_signal_dbm / 10.0), axis=1
        )  # (N_rx,)

        # --- Interference power at each RX from each jammer per subcarrier
        if N_jam > 0:
            # pl_jam : (N_jam, N_rx)
            pl_jam = self._path_loss_db(jam_positions, rx_positions)
            # jam_power_dbm : (N_jam, N_subcarriers)
            # received_jam  : (N_rx, N_jam, N_subcarriers)
            rx_jam_dbm = (
                jam_power_dbm[np.newaxis, :, :]          # (1, N_jam, N_sub)
                - pl_jam.T[:, :, np.newaxis]             # (N_rx, N_jam, 1)
            )
            rx_jam_lin = np.sum(
                10 ** (rx_jam_dbm / 10.0), axis=1
            )  # (N_rx, N_subcarriers)
        else:
            rx_jam_lin = np.zeros((N_rx, n_subcarriers))

        noise_lin = 10 ** (noise_power_dbm / 10.0)

        # --- SINR: signal / (interference + noise) — same for all subcarriers
        sinr_lin = (
            rx_signal_lin[:, np.newaxis]              # (N_rx, 1)
            / (rx_jam_lin + noise_lin)                # (N_rx, N_subcarriers)
        )
        sinr_db = 10.0 * np.log10(np.clip(sinr_lin, 1e-10, None))
        return sinr_db.astype(np.float32)  # (N_rx, N_subcarriers)