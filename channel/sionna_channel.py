"""
channel/sionna_channel.py
--------------------------
CDL-D channel model backed by NVIDIA Sionna 2.x (PyTorch backend).

Requires: sionna>=2.0, torch, GPU recommended.
Only imported when EnvironmentConfig.channel_mode == "sionna".

Two-component model:
  1. Large-scale path loss  — log-distance (same as MockChannel, placeholder
                               until 3GPP UMa/UMi path-loss is wired in)
  2. Small-scale fading     — CDL-D (LOS-dominant, frequency-selective)
                               One independent realization per TX-RX pair per step.

The combination gives frequency-selective SINR per OFDM subcarrier, which is
the key difference vs. MockChannel (where all subcarriers see the same channel).

Colab usage
-----------
    from core.config import EnvironmentConfig
    config = EnvironmentConfig(channel_mode="sionna", ...)
    from envs.wireless_env import WirelessEnv
    env = WirelessEnv(config)   # SionnaChannel is instantiated here
"""

from __future__ import annotations
import numpy as np
from channel.base import BaseChannel


class SionnaChannel(BaseChannel):
    """
    Parameters
    ----------
    carrier_frequency_hz : float
        Centre carrier frequency (Hz). Default 3.5 GHz.
    n_subcarriers : int
        Number of OFDM subcarriers.
    subcarrier_spacing_hz : float
        OFDM subcarrier spacing (Hz). Default 15 kHz.
    delay_spread_s : float
        CDL-D RMS delay spread (s). Default 30 ns (typical outdoor LOS/UAV).
    alpha : float
        Path-loss exponent for the log-distance model.
    pl0_db : float
        Reference path loss at d0=1 m.
    device : str | None
        PyTorch device string, e.g. "cuda" or "cpu".
        None = auto-select (cuda if available, else cpu).
    seed : int | None
        Seed for PyTorch RNG (reproducible fading realisations).
    """

    def __init__(
        self,
        carrier_frequency_hz: float = 3.5e9,
        n_subcarriers: int = 76,
        subcarrier_spacing_hz: float = 15e3,
        delay_spread_s: float = 30e-9,
        alpha: float = 3.5,
        pl0_db: float = 40.0,
        seed: int | None = None,
    ):
        import torch
        from sionna.phy import config as sionna_config
        from sionna.phy.channel.tr38901 import CDL, PanelArray

        self.carrier_freq        = carrier_frequency_hz
        self.n_sub               = n_subcarriers
        self.subcarrier_spacing  = subcarrier_spacing_hz
        self.delay_spread        = delay_spread_s
        self.alpha               = alpha
        self.pl0_db              = pl0_db

        if seed is not None:
            torch.manual_seed(seed)

        # Single isotropic panel at UT and BS — no MIMO / beamforming in Stage 1
        _array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency_hz,
        )
        # Pass device=None — let Sionna pick from its own available_devices registry.
        # On CPU-only machines that's "cpu"; on Colab/Euler with CUDA it auto-selects GPU.
        self._cdl = CDL(
            model="D",
            delay_spread=delay_spread_s,
            carrier_frequency=carrier_frequency_hz,
            ut_array=_array,
            bs_array=_array,
            direction="uplink",
            min_speed=0.0,
            max_speed=0.0,
            device=None,   # Sionna auto-selects; don't pass torch-style "cuda"
        )

        # Resolve the actual device Sionna chose, then use it for our tensors
        self.device = sionna_config.device   # e.g. "cpu" or "gpu"
        torch_device = "cuda" if self.device != "cpu" else "cpu"

        # Subcarrier frequency offsets from carrier centre  (N_sub,)
        self._freq_offsets = torch.tensor(
            (np.arange(n_subcarriers) - n_subcarriers // 2) * subcarrier_spacing_hz,
            dtype=torch.float32,
            device=torch_device,
        )

        print(f"[SionnaChannel] CDL-D ready on {self.device} | "
              f"fc={carrier_frequency_hz/1e9:.1f} GHz | "
              f"Δf={subcarrier_spacing_hz/1e3:.0f} kHz | "
              f"τ_rms={delay_spread_s*1e9:.0f} ns")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_loss_db(
        self, pos_a: np.ndarray, pos_b: np.ndarray
    ) -> np.ndarray:
        """Pairwise log-distance path loss matrix (M, N) in dB."""
        diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]  # (M, N, 3)
        dist = np.linalg.norm(diff, axis=-1).clip(1.0, None)       # (M, N)
        return self.pl0_db + 10.0 * self.alpha * np.log10(dist)    # (M, N)

    def _cdl_fading_db(self, n_pairs: int) -> np.ndarray:
        """
        Generate CDL-D frequency responses for n_pairs independent links.

        Each pair gets an independent channel realization (i.i.d. fading).
        This is the per-step fading update — positions enter via path loss,
        not via the CDL model itself.

        Returns
        -------
        H_db : (n_pairs, N_sub) — fading gain in dB per subcarrier
        """
        import torch

        # h   : (n_pairs, 1, 1, 1, 1, n_paths, 1)  complex64
        # tau : (n_pairs, 1, 1, n_paths)             float32
        h, tau = self._cdl(
            batch_size=n_pairs,
            num_time_steps=1,
            sampling_frequency=self.subcarrier_spacing,
        )

        h   = h.squeeze()    # (n_pairs, n_paths)  or (n_paths,) if n_pairs=1
        tau = tau.squeeze()  # same

        # Guard against single-pair edge case
        if h.dim() == 1:
            h   = h.unsqueeze(0)    # (1, n_paths)
            tau = tau.unsqueeze(0)

        # H[k] = Σ_l  h_l · exp(−j·2π·f_k·τ_l)
        # shapes after broadcasting:
        #   h   : (n_pairs, n_paths, 1)
        #   tau : (n_pairs, n_paths, 1)
        #   f   : (1,       1,       N_sub)
        h_c   = h.unsqueeze(-1).to(torch.complex64)           # (..., 1)
        tau_c = tau.unsqueeze(-1).to(torch.complex64)         # (..., 1)
        f     = self._freq_offsets.to(torch.complex64)        # (N_sub,)
        f     = f.unsqueeze(0).unsqueeze(0)                   # (1, 1, N_sub)

        phase  = torch.exp(-2j * torch.pi * tau_c * f)        # (n_pairs, n_paths, N_sub)
        H_freq = (h_c * phase).sum(dim=1)                     # (n_pairs, N_sub)

        H_mag2_db = 10.0 * torch.log10(
            H_freq.abs().pow(2).clamp(min=1e-12)
        )
        return H_mag2_db.cpu().float().numpy()  # (n_pairs, N_sub)

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def compute_sinr(
        self,
        tx_positions:    np.ndarray,   # (N_tx, 3)
        rx_positions:    np.ndarray,   # (N_rx, 3)
        jam_positions:   np.ndarray,   # (N_jam, 3)
        tx_power_dbm:    np.ndarray,   # (N_tx,)
        jam_power_dbm:   np.ndarray,   # (N_jam, N_sub)
        noise_power_dbm: float,
        n_subcarriers:   int,
    ) -> np.ndarray:                   # (N_rx, N_sub) in dB
        N_rx  = rx_positions.shape[0]
        N_tx  = tx_positions.shape[0]
        N_jam = jam_positions.shape[0] if len(jam_positions) > 0 else 0

        # --- Signal: legitimate TX → each RX ---
        pl_tx  = self._path_loss_db(tx_positions, rx_positions)          # (N_tx, N_rx)
        H_tx   = self._cdl_fading_db(N_tx * N_rx)                       # (N_tx*N_rx, N_sub)
        H_tx   = H_tx.reshape(N_tx, N_rx, n_subcarriers)

        sig_dbm = (
            tx_power_dbm[:, np.newaxis, np.newaxis]   # (N_tx, 1,    1)
            - pl_tx[:, :, np.newaxis]                 # (N_tx, N_rx, 1)
            + H_tx                                    # (N_tx, N_rx, N_sub)
        )
        signal_lin = np.sum(10 ** (sig_dbm / 10.0), axis=0)             # (N_rx, N_sub)

        # --- Interference: each jammer → each RX ---
        if N_jam > 0:
            pl_jam = self._path_loss_db(jam_positions, rx_positions)     # (N_jam, N_rx)
            H_jam  = self._cdl_fading_db(N_jam * N_rx)                  # (N_jam*N_rx, N_sub)
            H_jam  = H_jam.reshape(N_jam, N_rx, n_subcarriers)

            jam_dbm = (
                jam_power_dbm[:, np.newaxis, :]       # (N_jam, 1,    N_sub)
                - pl_jam[:, :, np.newaxis]            # (N_jam, N_rx, 1)
                + H_jam                               # (N_jam, N_rx, N_sub)
            )
            jam_lin = np.sum(10 ** (jam_dbm / 10.0), axis=0)            # (N_rx, N_sub)
        else:
            jam_lin = np.zeros((N_rx, n_subcarriers), dtype=np.float32)

        # --- SINR ---
        noise_lin = 10 ** (noise_power_dbm / 10.0)
        sinr_lin  = signal_lin / (jam_lin + noise_lin)
        sinr_db   = 10.0 * np.log10(np.clip(sinr_lin, 1e-10, None))
        return sinr_db.astype(np.float32)                                # (N_rx, N_sub)
