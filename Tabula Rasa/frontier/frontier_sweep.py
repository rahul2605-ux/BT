"""
Phase 0 — Effectiveness–detectability frontier (no RL).

Characterizes the frozen sim06 CNN detector against controlled interference:
for a grid of (number of active subcarriers, per-subcarrier power, structure),
inject a temporally-stable interfering waveform into a clean OFDM frame, then
measure BOTH:
  - P(detect)  — frozen EfficientNet-B0 score on the frame spectrogram
  - BER        — hard-decision bit error rate on the data subcarriers

This produces the achievable BER-vs-P(detect) frontier WITHOUT any learning.
It bounds what any learned jammer could reach at a given stealth level, and
answers the make-or-break question: is there room for RL to beat a random
sparse strategy, or is the frontier fully explained by "few active SCs, any
quiet ones, enough power"?

Strategies swept:
  sparse_blind   — n_active random effective SCs, HELD across all 14 symbols,
                   fixed magnitude + random phase (blind, temporally stable =
                   stealthiest). This is the main frontier sweep.
  sparse_hopping — same, but RESAMPLED every OFDM symbol (fresh SCs + phases).
                   Reproduces the sim07 frequency-hopping regime. The held-vs-
                   hopping contrast isolates temporal coherence as a detection
                   axis independent of spectral occupancy.
  sparse_matched — n_active random effective SCs, jam = -2*tx per symbol
                   (omniscient cancellation → rx = -tx, an upper bound).
  broadband      — all effective SCs, per-symbol complex Gaussian (classical
                   barrage reference; temporally white).
  single_tone    — one fixed SC, held (classical single-tone reference).

No training. Reuses simulation06/{ofdm,detector}.py unchanged.
"""

import os
import sys
import json
import math
import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import ofdm
from detector import load_detector, detect


# ── Detector helper (chunked to bound memory) ───────────────────────────────

@torch.no_grad()
def detect_chunked(detector, rx_time_flat, chunk=64):
    """P(jammed) for [B, T] complex, run through the detector in chunks."""
    out = []
    for i in range(0, rx_time_flat.shape[0], chunk):
        out.append(detect(detector, rx_time_flat[i:i + chunk]))
    return torch.cat(out, dim=0)


# ── Interference construction ───────────────────────────────────────────────

def random_active_mask(B, n_active, eff_full, fft_size, device):
    """Boolean [B, fft_size] mask with n_active random effective SCs per row."""
    n_eff = eff_full.shape[0]
    n_active = min(n_active, n_eff)
    sel = torch.rand(B, n_eff, device=device).argsort(dim=1)[:, :n_active]  # into eff
    full_idx = eff_full[sel]                                                # [B, n_active]
    mask = torch.zeros(B, fft_size, device=device)
    mask.scatter_(1, full_idx, 1.0)
    return mask


def build_jam(strategy, n_active, power, tx_grid, eff_full, device):
    """Return jam_full [B, N_OFDM, FFT] complex to add to the clean grid.

    `power` is the per-active-subcarrier power (mean |jam|^2 on active SCs).
    """
    B = tx_grid.shape[0]
    fft = ofdm.FFT_SIZE
    n_ofdm = ofdm.N_OFDM_SYMBOLS

    if strategy == "sparse_blind":
        mask = random_active_mask(B, n_active, eff_full, fft, device)      # [B, FFT]
        phase = torch.rand(B, fft, device=device) * (2 * math.pi)
        vals = mask * (power ** 0.5) * torch.exp(1j * phase.to(torch.complex64))
        return vals[:, None, :].expand(B, n_ofdm, fft).contiguous()

    if strategy == "sparse_hopping":
        jam = torch.zeros(B, n_ofdm, fft, dtype=torch.complex64, device=device)
        for t in range(n_ofdm):
            mask = random_active_mask(B, n_active, eff_full, fft, device)
            phase = torch.rand(B, fft, device=device) * (2 * math.pi)
            jam[:, t, :] = mask * (power ** 0.5) * torch.exp(1j * phase.to(torch.complex64))
        return jam

    if strategy == "sparse_matched":
        mask = random_active_mask(B, n_active, eff_full, fft, device)      # [B, FFT]
        return (-2.0) * tx_grid[:, 0, 0] * mask[:, None, :]                # rx = -tx on active

    if strategy == "broadband":  # in-band only (respects spectral mask)
        mask = torch.zeros(fft, device=device)
        mask[eff_full] = 1.0
        noise = (torch.randn(B, n_ofdm, fft, device=device)
                 + 1j * torch.randn(B, n_ofdm, fft, device=device)) * ((power / 2) ** 0.5)
        return noise * mask

    if strategy == "broadband_oob":  # all 64 bins incl guard/DC — replicates sim06 probe
        return (torch.randn(B, n_ofdm, fft, device=device)
                + 1j * torch.randn(B, n_ofdm, fft, device=device)) * ((power / 2) ** 0.5)

    if strategy == "single_tone":
        mask = torch.zeros(B, fft, device=device)
        mask[:, int(eff_full[len(eff_full) // 2])] = 1.0
        vals = mask * (power ** 0.5)
        return vals.to(torch.complex64)[:, None, :].expand(B, n_ofdm, fft).contiguous()

    raise ValueError(f"unknown strategy {strategy}")


@torch.no_grad()
def evaluate(strategy, n_active, power, detector, eff_full, B, device):
    """Inject one config, return dict of P(detect)/BER stats."""
    tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    jam_full = build_jam(strategy, n_active, power, tx_grid, eff_full, device)

    rx_grid = tx_grid.clone()
    rx_grid[:, 0, 0] = rx_grid[:, 0, 0] + jam_full

    rx_time = ofdm.modulator(rx_grid)
    p_jam = detect_chunked(detector, rx_time[:, 0, 0])

    _, rx_eff = ofdm.demodulate_frame(rx_time)
    ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)

    per_sc_power = power if strategy != "sparse_matched" else \
        (jam_full.abs().pow(2)[jam_full.abs() > 0].mean().item() if (jam_full.abs() > 0).any() else 0.0)
    total_power = jam_full.abs().pow(2).mean().item()

    return {
        "strategy": strategy,
        "n_active": int(n_active),
        "per_sc_power": float(per_sc_power),
        "total_power": float(total_power),
        "p_jam_mean": float(p_jam.mean()),
        "p_jam_std": float(p_jam.std()),
        "ber_mean": float(ber.mean()),
        "ber_std": float(ber.std()),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def pareto_envelope(points):
    """Upper-left staircase: max BER achievable at each P(detect) ceiling."""
    pts = sorted(points, key=lambda d: d["p_jam_mean"])
    xs, ys, best = [], [], -1.0
    for d in pts:
        if d["ber_mean"] > best:
            best = d["ber_mean"]
            xs.append(d["p_jam_mean"])
            ys.append(best)
    return xs, ys


def save_plots(rows, n_eff, out_dir, taus=(0.05, 0.10, 0.50)):
    blind = [r for r in rows if r["strategy"] == "sparse_blind"]
    hopping = [r for r in rows if r["strategy"] == "sparse_hopping"]
    matched = [r for r in rows if r["strategy"] == "sparse_matched"]
    broad = [r for r in rows if r["strategy"] == "broadband"]
    oob = [r for r in rows if r["strategy"] == "broadband_oob"]
    tone = [r for r in rows if r["strategy"] == "single_tone"]

    # ── Fig 1: the frontier ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    na = np.array([r["n_active"] for r in blind])
    norm = matplotlib.colors.LogNorm(vmin=max(na.min(), 1), vmax=na.max())
    sc = ax.scatter([r["p_jam_mean"] for r in blind],
                    [r["ber_mean"] for r in blind],
                    c=na, cmap="viridis", norm=norm, s=45,
                    edgecolor="k", linewidth=0.3, label="sparse blind", zorder=3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("# active subcarriers (blind)")

    if hopping:
        ax.scatter([r["p_jam_mean"] for r in hopping], [r["ber_mean"] for r in hopping],
                   marker="+", c="red", s=45, linewidth=1.0,
                   label="sparse hopping (per-symbol)", zorder=2)
    if matched:
        ax.scatter([r["p_jam_mean"] for r in matched], [r["ber_mean"] for r in matched],
                   marker="X", c="crimson", s=60, label="omniscient (jam=-2·tx)", zorder=4)
    if broad:
        ax.scatter([r["p_jam_mean"] for r in broad], [r["ber_mean"] for r in broad],
                   marker="s", c="gray", s=40, label="broadband in-band", zorder=2)
    if oob:
        ax.scatter([r["p_jam_mean"] for r in oob], [r["ber_mean"] for r in oob],
                   marker="v", c="black", s=40, label="broadband out-of-band (sim06 probe)", zorder=2)
    if tone:
        ax.scatter([r["p_jam_mean"] for r in tone], [r["ber_mean"] for r in tone],
                   marker="^", c="orange", s=45, label="single tone", zorder=2)

    xs, ys = pareto_envelope(rows)
    ax.step(xs, ys, where="post", color="k", lw=1.5, ls="--", label="Pareto envelope", zorder=1)

    for tau in taus:
        ax.axvline(tau, color="steelblue", lw=0.8, alpha=0.5)
        ax.text(tau, ax.get_ylim()[1] * 0.98, f"τ={tau}", rotation=90,
                va="top", ha="right", fontsize=7, color="steelblue")

    ax.set_xlabel("P(detect)  — frozen CNN")
    ax.set_ylabel("BER (data subcarriers)")
    ax.set_title("Effectiveness–detectability frontier\n(temporally-stable interference, lossless OFDM)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "frontier.png"), dpi=150)
    plt.close(fig)

    # ── Fig 2: detection cliff, P(detect) vs n_active per power ─────────
    fig, ax = plt.subplots(figsize=(8, 5))
    powers = sorted({r["per_sc_power"] for r in blind})
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(powers)))
    for c, p in zip(cmap, powers):
        pts = sorted([r for r in blind if abs(r["per_sc_power"] - p) < 1e-9],
                     key=lambda d: d["n_active"])
        ax.plot([r["n_active"] for r in pts], [r["p_jam_mean"] for r in pts],
                "-o", color=c, ms=4, label=f"power={p:g}")
    ax.set_xlabel("# active subcarriers"); ax.set_ylabel("P(detect)")
    ax.set_title("Detection vs sparsity (the cliff)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, title="per-SC")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cliff.png"), dpi=150)
    plt.close(fig)

    # ── Fig 3: BER vs n_active with analytic ceilings ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = max((r["per_sc_power"] for r in blind), default=1.0)
    b_hi = sorted([r for r in blind if abs(r["per_sc_power"] - hi) < 1e-9],
                  key=lambda d: d["n_active"])
    ax.plot([r["n_active"] for r in b_hi], [r["ber_mean"] for r in b_hi],
            "-o", color="teal", label=f"blind measured (power={hi:g})")
    if matched:
        m = sorted(matched, key=lambda d: d["n_active"])
        ax.plot([r["n_active"] for r in m], [r["ber_mean"] for r in m],
                "-X", color="crimson", label="omniscient measured")
    nn = np.array(sorted({r["n_active"] for r in blind}))
    ax.plot(nn, 0.5 * nn / n_eff, ":", color="teal", label="blind ceiling 0.5·n/N")
    ax.plot(nn, nn / n_eff, ":", color="crimson", label="omniscient ceiling n/N")
    ax.set_xlabel("# active subcarriers"); ax.set_ylabel("BER")
    ax.set_title("Effectiveness vs sparsity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ber_vs_nactive.png"), dpi=150)
    plt.close(fig)

    # ── Fig 4: temporal coherence — held vs hopping at high power ───────
    if hopping:
        fig, ax = plt.subplots(figsize=(8, 5))
        hi = max((r["per_sc_power"] for r in blind), default=1.0)
        for data, lbl, col in [(blind, "held (temporally stable)", "seagreen"),
                               (hopping, "hopping (per-symbol)", "red")]:
            pts = sorted([r for r in data if abs(r["per_sc_power"] - hi) < 1e-9],
                         key=lambda d: d["n_active"])
            ax.plot([r["n_active"] for r in pts], [r["p_jam_mean"] for r in pts],
                    "-o", color=col, ms=5, label=lbl)
        ax.set_xlabel("# active subcarriers"); ax.set_ylabel("P(detect)")
        ax.set_title(f"Temporal coherence is the detection axis (per-SC power={hi:g})\n"
                     "same spectral occupancy, opposite detectability")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "coherence.png"), dpi=150)
        plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-model", default="../artifacts/sim06/detector/run002_best.pt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="../artifacts/frontier")
    ap.add_argument("--smoke", action="store_true", help="tiny sweep for a quick sanity run")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)
    out_dir = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(out_dir, exist_ok=True)

    ofdm.init_device(device)
    detector = load_detector(
        os.path.join(os.path.dirname(__file__), args.detector_model), device=device)
    for p in detector.parameters():
        p.requires_grad = False

    eff_arr = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(
        eff_arr.cpu().numpy() if hasattr(eff_arr, "cpu") else np.asarray(eff_arr),
        dtype=torch.long, device=device)
    n_eff = eff_full.shape[0]
    print(f"effective subcarriers: {n_eff}", flush=True)

    if args.smoke:
        n_active_list = [1, 4, 52]
        power_list = [0.3, 4.0]
        B = min(args.batch, 16)
    else:
        n_active_list = [1, 2, 3, 4, 5, 6, 8, 12, 16, 26, 52]
        power_list = [0.03, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0]
        B = args.batch

    rows = []
    t0 = time.time()

    # main frontier: sparse_blind (held) and sparse_hopping over the full grid
    for strat in ("sparse_blind", "sparse_hopping"):
        for n_active in n_active_list:
            for power in power_list:
                r = evaluate(strat, n_active, power, detector, eff_full, B, device)
                rows.append(r)
                print(f"{strat:<14} n={n_active:>3} pwr={power:>5g} | "
                      f"P(det)={r['p_jam_mean']:.4f} BER={r['ber_mean']:.4f}", flush=True)

    # omniscient ceiling (power irrelevant): one point per n_active
    for n_active in n_active_list:
        r = evaluate("sparse_matched", n_active, 0.0, detector, eff_full, B, device)
        rows.append(r)
        print(f"matched  n={n_active:>3}            | "
              f"P(det)={r['p_jam_mean']:.4f} BER={r['ber_mean']:.4f}", flush=True)

    # classical references + out-of-band control (replicates sim06 broadband probe)
    for power in power_list:
        rb = evaluate("broadband", n_eff, power, detector, eff_full, B, device)
        rt = evaluate("single_tone", 1, power, detector, eff_full, B, device)
        ro = evaluate("broadband_oob", n_eff, power, detector, eff_full, B, device)
        rows.extend([rb, rt, ro])
        print(f"broadband      n={n_eff:>3} pwr={power:>5g} | "
              f"P(det)={rb['p_jam_mean']:.4f} BER={rb['ber_mean']:.4f}   ||  "
              f"OOB P(det)={ro['p_jam_mean']:.4f} BER={ro['ber_mean']:.4f}", flush=True)

    dt = time.time() - t0
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"n_eff": n_eff, "batch": B, "rows": rows}, f, indent=2)
    save_plots(rows, n_eff, out_dir)

    # ── Headline: max stealthy BER at each detection ceiling ────────────
    blind = [r for r in rows if r["strategy"] == "sparse_blind"]
    print("\n=== HEADLINE: best stealthy operating point (sparse blind) ===", flush=True)
    for tau in (0.05, 0.10, 0.50):
        cand = [r for r in blind if r["p_jam_mean"] <= tau]
        if cand:
            best = max(cand, key=lambda d: d["ber_mean"])
            print(f"  P(det) <= {tau:.2f}: max BER = {best['ber_mean']:.4f} "
                  f"(n_active={best['n_active']}, per-SC power={best['per_sc_power']:g})",
                  flush=True)
        else:
            print(f"  P(det) <= {tau:.2f}: no configuration stays this stealthy", flush=True)

    print(f"\nDone in {dt:.1f}s. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
