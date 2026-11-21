"""
Attack study, step 0 (README §3.1): QPSK SER vs N0 with noise only, no jammer.

This is the reference every attack is later measured against (excess SER =
attacked minus this). Monte Carlo on M0's link (y = s + w, Es = 1, N0 = 2*sigma^2);
the closed form BER = Q(1/sqrt(N0)) is only printed as a check.

    python noise_baseline.py      # CPU, seconds; the login node is fine (M0)
    -> ../artifacts/m0/noise_baseline.{png,json}
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import link

N0_DB = list(range(-10, 31))        # Es = 1, so Es/N0 runs from +10 to -30 dB
N_SYM = 1_000_000                   # per point: 2e6 bits, ~1500 errors at the lowest N0
SEED = 0
OUT = os.path.join("..", "artifacts", "m0")

BLUE, ORANGE = "#2a78d6", "#eb6834"  # dataviz reference palette, slots 1-2
INK, MUTED = "#1f1f1e", "#6b6a64"


def measure():
    gen = torch.Generator().manual_seed(SEED)
    rows = []
    for n0_db in N0_DB:
        n0 = 10.0 ** (n0_db / 10.0)
        sigma = math.sqrt(n0 / 2.0)
        bits = link.random_bits(1, N_SYM, generator=gen)
        sym = link.bits_to_symbols(bits)
        _, yhat = link.receive(sym, torch.zeros_like(sym), sigma, generator=gen)
        ber, ser = link.ber_ser(bits, link.symbols_to_bits(yhat))
        p = link.theoretical_ber(sigma)
        rows.append(dict(n0_db=n0_db, n0=n0, sigma=sigma, ber=ber, ser=ser,
                         ber_theory=p, ser_theory=1.0 - (1.0 - p) ** 2))
        print(f"N0 {n0_db:+3d} dB  BER {ber:.3e} (theory {p:.3e})  "
              f"SER {ser:.3e} (theory {rows[-1]['ser_theory']:.3e})")
    return rows


def plot(rows, path, log=True):
    x = [r["n0_db"] for r in rows]
    y = [r["ser"] if r["ser"] > 0 else float("nan") for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(x, y, "o-", ms=6, lw=2, color=BLUE,
            markeredgecolor="white", markeredgewidth=1.2)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel(r"$N_0$ (dB; $E_s = 1$, so $E_s/N_0 = -N_0$)", color=INK)
    ax.set_ylabel("SER", color=INK)
    ax.set_title(f"QPSK over AWGN, no jammer ({N_SYM:,} symbols per point)",
                 color=INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = measure()
    with open(os.path.join(OUT, "noise_baseline.json"), "w") as f:
        json.dump(dict(n_sym=N_SYM, seed=SEED, es=1.0, rows=rows), f, indent=1)
    plot(rows, os.path.join(OUT, "noise_baseline.png"))
    plot(rows, os.path.join(OUT, "noise_baseline_linear.png"), log=False)
    print("->", os.path.join(OUT, "noise_baseline.{png,json}"))


if __name__ == "__main__":
    main()
