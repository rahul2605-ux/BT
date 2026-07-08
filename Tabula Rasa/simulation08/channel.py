"""
sim08 channel — realistic multi-link frequency-selective OFDM channel.

rx_grid = h_tx * tx  +  Σ_k h_jam_k * jam_k  +  AWGN   (per subcarrier, freq domain)

- Independent frequency-selective block fading per link via Sionna tr38901.TDL
  (coherent within a frame; different gain per subcarrier → which SC to jam matters).
- Per-link average path gain (geometry): the TX and each jammer can have different
  average SNR to the RX, so power allocation across jammers is a real decision.
- AWGN at a target Eb/N0 (added on the effective/in-band subcarriers, consistent
  with the demapper's `no`).

Physical model: the jammer transmits jam_k, which propagates through its OWN channel
h_jam_k before reaching the RX — a BLIND jammer does not know h_jam_k. The legit RX
equalizes the TX link (perfect CSI here). After ZF equalization, the effective
interference on subcarrier n is (h_jam_k[n] / h_tx[n]) * jam_k[n]: subcarriers where
the jammer is strong relative to the TX dominate — which is exactly why channel-aware
subcarrier selection beats blind selection, and why a learned jammer can help.
"""

import torch
import sionna.phy as sn
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import TDL


class MultiLinkChannel:
    def __init__(self, resource_grid, n_jammers, device,
                 model="C", delay_spread=100e-9, carrier_freq=5.2e9,
                 tx_gain_db=0.0, jammer_gains_db=None):
        self.rg = resource_grid
        self.n_jammers = n_jammers
        self.device = device

        def make_gen():
            tdl = TDL(model=model, delay_spread=delay_spread,
                      carrier_frequency=carrier_freq, num_rx_ant=1, num_tx_ant=1)
            return GenerateOFDMChannel(tdl, resource_grid)

        self.tx_gen = make_gen()
        self.jam_gens = [make_gen() for _ in range(n_jammers)]
        self.tx_gain = 10 ** (tx_gain_db / 20.0)
        if jammer_gains_db is None:
            jammer_gains_db = [0.0] * n_jammers
        self.jam_gains = [10 ** (g / 20.0) for g in jammer_gains_db]

        fft = resource_grid.fft_size
        eff = resource_grid.effective_subcarrier_ind
        eff = eff.cpu().numpy() if hasattr(eff, "cpu") else eff
        mask = torch.zeros(fft, device=device)
        mask[list(eff)] = 1.0
        self.eff_mask = mask  # [FFT], 1 on in-band subcarriers

    def sample(self, B):
        """Fresh per-link frequency responses.
        Returns h_tx [B, N_OFDM, FFT], h_jam list of [B, N_OFDM, FFT] (complex)."""
        h_tx = self.tx_gain * self.tx_gen(B)[:, 0, 0, 0, 0]
        h_jam = [g * gen(B)[:, 0, 0, 0, 0]
                 for g, gen in zip(self.jam_gains, self.jam_gens)]
        return h_tx, h_jam

    def apply(self, tx_grid, jam_grids, h_tx, h_jam, no):
        """Return faded, jammed, noisy rx_grid [B, 1, 1, N_OFDM, FFT].

        tx_grid:   [B, 1, 1, N_OFDM, FFT] complex (clean resource grid)
        jam_grids: list of [B, N_OFDM, FFT] complex (per jammer, pre-channel)
        h_tx:      [B, N_OFDM, FFT]; h_jam: list of [B, N_OFDM, FFT]
        no:        per-subcarrier noise variance (from ebnodb2no)
        """
        rx = h_tx * tx_grid[:, 0, 0]
        for hk, jk in zip(h_jam, jam_grids):
            rx = rx + hk * jk
        std = (no / 2.0) ** 0.5
        noise = std * (torch.randn_like(rx.real) + 1j * torch.randn_like(rx.real))
        rx = rx + noise * self.eff_mask                 # AWGN on in-band subcarriers
        out = tx_grid.clone()
        out[:, 0, 0] = rx
        return out

    @staticmethod
    def equalize_zf(rx_grid, h_tx, eps=1e-6):
        """Perfect-CSI zero-forcing equalization of the TX link (for BER)."""
        h = h_tx.clone()
        small = h.abs() < eps
        h[small] = eps
        out = rx_grid.clone()
        out[:, 0, 0] = rx_grid[:, 0, 0] / h
        return out
