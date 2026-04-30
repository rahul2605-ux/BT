"""
channel/sionna_channel.py
--------------------------
CDL-D channel model backed by NVIDIA Sionna 2.x (PyTorch backend).

Requires: sionna>=2.0, torch, GPU recommended.
Only imported when EnvironmentConfig.channel_mode == "sionna".

Two-component model
-------------------
1. Large-scale path loss  — log-distance (placeholder for 3GPP UMa/UMi)
2. Small-scale fading     — CDL-D (LOS-dominant, 13 paths, 30 ns delay spread)
                            Full complex coefficient returned — phase is preserved.

The complex H[rx, src, sub] enables coherent combining in compute_sinr_complex():
jammers that coordinate their phase through the CDL channel can achieve N²
interference gain at the defender.

Sionna 2.0 device note
----------------------
Sionna 2.0 uses its own device registry (sionna.phy.config.available_devices).
Never pass device="cuda" — always pass device=None and read back
sionna.phy.config.device for the chosen device string.
"""

from __future__ import annotations
import numpy as np
from channel.base import BaseChannel


class SionnaChannel(BaseChannel):
    """
    Parameters
    ----------
    carrier_frequency_hz : float
    n_subcarriers : int
    subcarrier_spacing_hz : float
    delay_spread_s : float
        CDL-D RMS delay spread. Default 30 ns (outdoor LOS/UAV).
    alpha : float
        Log-distance path-loss exponent.
    pl0_db : float
        Reference path loss at d0=1 m.
    seed : int | None
    """

    def __init__(
        self,
        carrier_frequency_hz: float = 3.5e9,
        n_subcarriers:        int   = 76,
        subcarrier_spacing_hz: float = 15e3,
        delay_spread_s:       float = 30e-9,
        alpha:                float = 3.5,
        pl0_db:               float = 40.0,
        seed:                 int | None = None,
    ):
        import torch
        from sionna.phy import config as sionna_config
        from sionna.phy.channel.tr38901 import CDL, PanelArray

        self.carrier_freq       = carrier_frequency_hz
        self.n_sub              = n_subcarriers
        self.subcarrier_spacing = subcarrier_spacing_hz
        self.delay_spread       = delay_spread_s
        self.alpha              = alpha
        self.pl0_db             = pl0_db

        if seed is not None:
            torch.manual_seed(seed)

        _array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency_hz,
        )
        self._cdl = CDL(
            model="D",
            delay_spread=delay_spread_s,
            carrier_frequency=carrier_frequency_hz,
            ut_array=_array,
            bs_array=_array,
            direction="uplink",
            min_speed=0.0,
            max_speed=0.0,
            device=None,  # Sionna auto-selects; never pass torch-style "cuda"
        )

        self.device = sionna_config.device
        torch_device = "cuda" if self.device != "cpu" else "cpu"

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
        """Pairwise log-distance path loss (M, N) in dB."""
        diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1).clip(1.0, None)
        return self.pl0_db + 10.0 * self.alpha * np.log10(dist)

    def _cdl_coefficients(self, n_pairs: int) -> np.ndarray:
        """
        Generate CDL-D frequency-domain channel coefficients for n_pairs links.

        Returns
        -------
        H : (n_pairs, N_sub) complex64

        Each row is one independent CDL-D realization evaluated at the N_sub
        OFDM subcarrier frequencies.  Phase is preserved — this is what enables
        coherent combining in compute_sinr_complex().

        Previously this method discarded the phase by returning |H|² in dB.
        That prevented any downstream code from exploiting phase coordination.
        """
        import torch

        h, tau = self._cdl(
            batch_size=n_pairs,
            num_time_steps=1,
            sampling_frequency=self.subcarrier_spacing,
        )
        # h   : (n_pairs, 1, 1, 1, 1, n_paths, 1)  complex64
        # tau : (n_pairs, 1, 1, n_paths)             float32
        h   = h.squeeze()    # (n_pairs, n_paths) or (n_paths,) if n_pairs=1
        tau = tau.squeeze()
        if h.dim() == 1:
            h   = h.unsqueeze(0)
            tau = tau.unsqueeze(0)

        # H[k] = Σ_l  h_l · exp(−j·2π·f_k·τ_l)
        h_c   = h.unsqueeze(-1).to(torch.complex64)            # (n_pairs, n_paths, 1)
        tau_c = tau.unsqueeze(-1).to(torch.complex64)          # (n_pairs, n_paths, 1)
        f     = self._freq_offsets.to(torch.complex64)
        f     = f.unsqueeze(0).unsqueeze(0)                    # (1, 1, N_sub)

        phase  = torch.exp(-2j * torch.pi * tau_c * f)         # (n_pairs, n_paths, N_sub)
        H_freq = (h_c * phase).sum(dim=1)                      # (n_pairs, N_sub) complex

        return H_freq.cpu().numpy().astype(np.complex64)       # phase preserved

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def get_coefficients(
        self,
        tx_positions:  np.ndarray,
        rx_positions:  np.ndarray,
        jam_positions: np.ndarray,
        n_subcarriers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns H_tx (N_rx, N_tx, N_sub) and H_jam (N_rx, N_jam, N_sub), complex64.

        H_total = path_loss_amplitude × H_CDL_fading
              = 10^(−PL/20) × H_fading

        The CDL coefficient is kept complex so downstream code can compute
        coherent interference fields.
        """
        N_tx  = tx_positions.shape[0]
        N_rx  = rx_positions.shape[0]
        N_jam = jam_positions.shape[0] if len(jam_positions) > 0 else 0

        # TX → RX
        pl_tx    = self._path_loss_db(tx_positions, rx_positions)   # (N_tx, N_rx)
        amp_tx   = 10 ** (-pl_tx / 20.0)                            # (N_tx, N_rx) voltage amp
        H_fad_tx = self._cdl_coefficients(N_tx * N_rx)              # (N_tx*N_rx, N_sub)
        H_fad_tx = H_fad_tx.reshape(N_tx, N_rx, n_subcarriers)      # (N_tx, N_rx, N_sub)
        # Combine and transpose to (N_rx, N_tx, N_sub)
        H_tx = (amp_tx[:, :, np.newaxis] * H_fad_tx).transpose(1, 0, 2).astype(np.complex64)

        # Jammer → RX
        if N_jam > 0:
            pl_jam    = self._path_loss_db(jam_positions, rx_positions)  # (N_jam, N_rx)
            amp_jam   = 10 ** (-pl_jam / 20.0)                           # (N_jam, N_rx)
            H_fad_jam = self._cdl_coefficients(N_jam * N_rx)             # (N_jam*N_rx, N_sub)
            H_fad_jam = H_fad_jam.reshape(N_jam, N_rx, n_subcarriers)
            H_jam = (amp_jam[:, :, np.newaxis] * H_fad_jam).transpose(1, 0, 2).astype(np.complex64)
        else:
            H_jam = np.zeros((N_rx, 0, n_subcarriers), dtype=np.complex64)

        return H_tx, H_jam

    # compute_sinr() and compute_sinr_complex() are inherited from BaseChannel.
    # No override needed.
