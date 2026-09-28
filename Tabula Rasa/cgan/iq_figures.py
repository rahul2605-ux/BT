"""
Received IQ at the symbol instants -- why the sim04 genie team hid and the GAN does not
(README §3.3g / A.3, user question 2026-09-27).

    sbatch submit_iq_figures.sh

Panels (SNR 30 dB, the deployed defender): clean; the genie flip (omniscient, eta = 1 --
sim04's mechanism: it knows the current symbol and moves it onto another QPSK point);
matched QPSK at sim04's power (asynchronous, symbol-blind); the CNN-targeted GAN
(run003, beta 10) at its matched-BER JSR, at 0 dB and at +10 dB. Each panel carries the
BER and P(det) at alpha 0.05 of one-sided power, kurtosis and the CNN, measured on
N_MEAS fresh frames at the same point. Matched-filter samples z (the victim's decision
statistic): unit-energy pulse, so the QPSK points sit at (±1 ± j)/√2.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import attacks
import baselines
import link as lk
import models
import scene

N_MEAS, N_SCATTER = 512, 24
OUT = os.path.join(scene.ART, "..", "iq")
GEN = os.path.join(scene.ART, "..", "gan", "run003", "task14_G.pt")    # spec_cnn_b10, random init


def main():
    device = lk.setup(seed=6000)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = baselines.Defender(L)
    G, scale, _ = models.load_generator(GEN, device)
    gan = dict(name="gan", G=G, scale=scale, tag="spec_cnn_b10")
    panels = [
        ("clean", None, None),
        ("genie flip (sim04 mechanism)\nknows the symbol, JSR 0 dB", dict(name="omniscient", eta=1.0), 0.0),
        ("matched QPSK, symbol-blind\nJSR +1 dB (sim04's power)", dict(name="pulsed", p=1.0), [1.0]),
        ("GAN vs CNN, β=10\nJSR −23 dB (matched BER 3e-4)", gan, [-23.0]),
        ("GAN vs CNN, β=10\nJSR 0 dB", gan, [0.0]),
        ("GAN vs CNN, β=10\nJSR +10 dB", gan, [10.0]),
    ]
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6))
    for ax, (title, spec, jsr) in zip(axes.flat, panels):
        m = baselines.measure(L, dfd, spec, jsr, N_MEAS)
        out = attacks.frames(L, N_SCATTER, spec, jsr, dfd.snr_db)
        z = out["z"].reshape(-1).cpu().numpy()
        outside = float(np.mean((np.abs(z.real) > 2.6) | (np.abs(z.imag) > 2.6)))
        ax.scatter(z.real, z.imag, s=2, alpha=0.25, color="#882255", lw=0, rasterized=True)
        q = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        ax.scatter(q.real, q.imag, marker="x", s=60, color="#1f1f1e", lw=1.8, zorder=3)
        ax.axhline(0, color="#e4e3de", lw=0.8, zorder=0); ax.axvline(0, color="#e4e3de", lw=0.8, zorder=0)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal")
        pd = m["pdet"]
        ax.set_title(title, fontsize=9.5, loc="left", color="#1f1f1e")
        ax.text(0.02, 0.02, f"BER {m['ber']:.1e}" + (f"   ({100 * outside:.0f}% of points off-axis)" if outside > 0.005 else "")
                + f"\nP(det): power {pd['power_one_sided']['0.05']:.2f}  "
                f"kurt {pd['kurtosis']['0.05']:.2f}  CNN {pd['spec_cnn']['0.05']:.2f}",
                transform=ax.transAxes, fontsize=8.5, color="#1f1f1e", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e4e3de", alpha=0.9))
        print(f"{title.splitlines()[0]:32s} BER {m['ber']:.2e}  pdet "
              f"{ {d: round(v['0.05'], 3) for d, v in pd.items()} }", flush=True)
    fig.suptitle("Received matched-filter samples at 30 dB SNR (×: QPSK points; α = 0.05, FAR 0.05)",
                 fontsize=11.5, x=0.012, ha="left", color="#1f1f1e")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "fig_iq_sim04_vs_gan.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
