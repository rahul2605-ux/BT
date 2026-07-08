"""
Illustrative figure for the Phase 0 finding: in-band vs out-of-band.

Top row  — per-subcarrier power (the intuitive "which frequencies carry energy"
           view), with the deliberately-empty guard/DC bins shaded red.
Bottom row — the STFT spectrogram the CNN detector actually classifies.

Cases (equal power where jammed): clean, sparse in-band, broadband in-band,
broadband out-of-band. The last two have the SAME power and nearly the same BER;
only the out-of-band one touches the guard bins, and only it is detected.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import ofdm
from detector import load_detector, time_signal_to_spectrogram
from frontier_sweep import build_jam, detect_chunked

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ofdm.init_device(device)
det = load_detector(os.path.join(os.path.dirname(__file__),
                    "../artifacts/sim06/detector/run002_best.pt"), device=device)
for p in det.parameters():
    p.requires_grad = False

eff = ofdm.resource_grid.effective_subcarrier_ind
eff_full = torch.as_tensor(eff.cpu().numpy() if hasattr(eff, "cpu") else eff,
                           dtype=torch.long, device=device)
GUARD = [i for i in range(ofdm.FFT_SIZE) if i not in eff_full.cpu().tolist()]

torch.manual_seed(1)
B = 64
tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)

cases = [
    ("Clean\n(no jammer)",              "clean",         0,  0.0),
    ("Sparse in-band\n(4 subcarriers)", "sparse_blind",  4,  8.0),
    ("Broadband in-band\n(52 subcarriers)", "broadband",  52, 8.0),
    ("Broadband OUT-of-band\n(all 64 bins)", "broadband_oob", 64, 8.0),
]

results = []
for name, strat, n, pw in cases:
    if strat == "clean":
        jam = torch.zeros(B, ofdm.N_OFDM_SYMBOLS, ofdm.FFT_SIZE,
                          dtype=torch.complex64, device=device)
    else:
        jam = build_jam(strat, n, pw, tx_grid, eff_full, device)
    rx = tx_grid.clone()
    rx[:, 0, 0] = rx[:, 0, 0] + jam
    rx_time = ofdm.modulator(rx)
    pj = detect_chunked(det, rx_time[:, 0, 0]).mean().item()
    _, rx_eff = ofdm.demodulate_frame(rx_time)
    ber = ofdm.compute_ber(rx_eff, tx_bits).mean().item()
    scpow = rx[:, 0, 0].abs().pow(2).mean(dim=(0, 1)).cpu().numpy()
    spec = time_signal_to_spectrogram(rx_time[0, 0, 0]).cpu().numpy()
    results.append((name, pj, ber, scpow, spec))
    print(f"{name.splitlines()[0]:24s} P(det)={pj:.3f} BER={ber:.3f}", flush=True)

ncol = len(cases)
fig, axes = plt.subplots(2, ncol, figsize=(4.3 * ncol, 7.4))

for j, (name, pj, ber, scpow, spec) in enumerate(results):
    caught = pj > 0.5
    col = "#A8443B" if caught else "#2F7A55"
    verdict = "DETECTED" if caught else "evades"

    ax = axes[0, j]
    ax.bar(range(ofdm.FFT_SIZE), scpow, width=1.0, color="#0E7C88")
    for g in GUARD:
        ax.axvspan(g - 0.5, g + 0.5, color="#C2410C", alpha=0.16, lw=0)
    ax.set_title(f"{name}\nP(detect)={pj:.3f}  ·  BER={ber:.3f}\n{verdict}",
                 color=col, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, ofdm.FFT_SIZE - 0.5)
    ax.set_xlabel("subcarrier index (0–63)")
    if j == 0:
        ax.set_ylabel("power per subcarrier")

    ax = axes[1, j]
    ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("time frame")
    if j == 0:
        ax.set_ylabel("STFT frequency bin\n(what the CNN sees)")

fig.suptitle(
    "In-band jamming hides in the signal; out-of-band jamming lights up the guard bands\n"
    "Top: power per subcarrier (red shading = deliberately-empty guard/DC bins).  "
    "Bottom: the spectrogram the CNN classifies.",
    fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.dirname(__file__),
                   "../artifacts/frontier/inband_vs_outofband.png")
fig.savefig(out, dpi=140)
print("saved", out, flush=True)
