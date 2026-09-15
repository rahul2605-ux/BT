"""
C6 -- evaluate a trained CGAN against Zhou et al. 2025, Fig. 6 (README §2.10, §3.4).

    sbatch submit_eval.sh --run run002 --sensitivity-sync locked
    sbatch submit_eval.sh --run run001 --tag async --sensitivity-sync locked

Zhou's protocol: SNR 30 dB, JSR -10:2:10 dB, jammers noise / optimal / gan, 1e4
symbols x 100 trials. EXTENDED by an error stopping rule (>= 100 bit errors or
max-bits) so the low-BER points are real; zero-error points are plotted at their
95% upper bound. The link is the calibrated one (link.LINK).

SYNCHRONISATION. `--sync` (default LINK["optimal_sync"] = async) applies to BOTH
structured jammers, optimal and gan -- one model, decided 2026-09-15 (README
§2.10). `--sensitivity-sync` adds optimal_<s> and gan_<s> sweeps under a second
model, so the effect of the assumption is reported with both jammers held
consistent. Noise does not depend on it.

Reports JSR at BER 1e-3 for each jammer, and the two gaps -- Noise - GAN and
GAN - Optimal -- next to the paper's own (4.44 and 1.31 dB). REPORT-ONLY: there
is no pass/fail bar (§2.10, decided at the C2 checkpoint).

Two diagnostics, written next to the BER figure:
  <out>_implied_gain.png  the Gaussian processing gain g each BER point implies,
                          BER = Q(sqrt(1/(1/(sps*SNR) + JSR/g))), paper vs ours.
                          A flat g is Gaussian-like interference; g -> infinity at
                          low JSR is a bounded jammer that cannot reach the decision
                          boundary (README §4.2 Q7).
  <out>_iq.png            G's output through the victim's matched filter at the
                          symbol instants (timing offset 0 and sps/2), next to the
                          clean signal: did G learn clean, symbol-aligned QPSK?
"""

import argparse
import json
import math
import os
from statistics import NormalDist

import numpy as np
import torch

import link as lk
import jammers
import models

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "cgan")

C_OPT, C_NOISE, C_GAN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3de"
COLOR = {"optimal": C_OPT, "noise": C_NOISE, "gan": C_GAN}
LABEL = {"optimal": "Optimal", "noise": "Noise", "gan": "GAN"}
MAIN = ["optimal", "noise", "gan"]


def load_generator(run, device):
    ckpt = torch.load(os.path.join(ART, f"{run}_G.pt"), map_location=device, weights_only=False)
    G = models.Generator(n_classes=ckpt["n_classes"], seg_len=ckpt["seg_len"]).to(device)
    G.load_state_dict(ckpt["state_dict"]); G.eval()
    return G, ckpt


def sweep(L, jsr_db, jammer_fn, snr_db, min_errors, max_bits):
    rows = []
    for j in jsr_db:
        r = lk.measure_ber(L, jammer_fn(float(j)), snr_db=snr_db,
                           min_errors=min_errors, max_bits=max_bits)
        r["zero"] = r["errors"] == 0
        r["plot"] = r["upper"] if r["zero"] else r["ber"]
        rows.append(r)
        tag = f">={r['upper']:.1e} (0 err)" if r["zero"] else f"{r['ber']:.2e}"
        print(f"  JSR {j:+5.1f} dB  BER {tag:>16}  ({r['errors']} err / {r['bits']:.1e} bits)",
              flush=True)
    return rows


# ---------------------------------------------------------------- implied Gaussian gain
def implied_gain_db(jsr_db, ber, sps, snr_db, max_ber=0.2):
    """
    Per point, the g that makes BER = Q(sqrt(1/(1/(sps*SNR) + JSR/g))) -- the
    link's own Gaussian-jammer law (link.py, g = sps for white noise). None where
    the BER is missing, >= max_ber (the Q-inversion is meaningless on the plateau)
    or below what the noise floor alone would give.
    """
    floor = 1.0 / (sps * 10.0 ** (snr_db / 10.0))
    out = []
    for j, b in zip(jsr_db, ber):
        if b is None or not (0 < b < max_ber):
            out.append(None); continue
        z = -NormalDist().inv_cdf(b)
        denom = 1.0 / z ** 2 - floor
        out.append(round(10 * math.log10(10 ** (j / 10) / denom), 3) if denom > 0 else None)
    return out


# ---------------------------------------------------------------- G's constellation
def constellation(L, G, ckpt, n_frames=64, n_sym=512):
    """
    G's output (sync locked, scaled to the signal's power) through the victim's
    matched filter, sampled at the symbol instants and half a symbol later; and the
    clean signal for reference. EVM is rms distance to the nearest unit-energy QPSK
    point: ~0 for clean aligned QPSK, large for a smeared or misaligned one.
    """
    j = jammers.gan(L, n_frames, n_sym, 0.0, G=G, scale=ckpt["scale"], z_dim=models.Z_DIM,
                    seg_len=ckpt["seg_len"], sync="locked")
    zf = L.filt(j, padding="full")
    span = n_sym * L.sps
    half = L.sps // 2
    pts = {
        "clean": L.matched_filter(L.modulate(n_frames, n_sym)[2], n_sym),
        "gan_offset0": zf[..., L.delay:L.delay + span:L.sps],
        "gan_offset_half": zf[..., L.delay + half:L.delay + half + span:L.sps],
    }
    stats = {}
    for k, z in pts.items():
        hard = torch.complex(torch.sign(z.real), torch.sign(z.imag)) / math.sqrt(2.0)
        stats[k] = dict(evm_rms=round(float((z - hard).abs().pow(2).mean().sqrt()), 4),
                        mean_power=round(float(z.abs().pow(2).mean()), 4))
    return {k: v.flatten().cpu().numpy() for k, v in pts.items()}, stats


# ---------------------------------------------------------------- figures
def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def _clean_axes(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(jsr, ours, paper, pflags, sync, sens, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for key in MAIN:
        c, lab = COLOR[key], LABEL[key]
        b = np.array([r["plot"] for r in ours[key]])
        zero = np.array([r["zero"] for r in ours[key]])
        ax.plot(jsr, b, color=c, lw=2, label=f"{lab} (ours{'' if key == 'noise' else ', ' + sync})")
        if zero.any():
            ax.plot(jsr[zero], b[zero], ls="none", marker="v", ms=8, color=c)
        if sens and key != "noise":
            s = ours[f"{key}_{sens}"]
            ax.plot(jsr, [r["plot"] for r in s], color=c, lw=1.2, ls=(0, (4, 2)),
                    label=f"{lab} (ours, {sens})")
        pb = np.array([np.nan if v is None else v for v in paper[key]], float)
        vis = np.array(["occluded" not in fl for fl in pflags[key]])
        ax.plot(jsr[vis], pb[vis], ls="none", marker="o", ms=8, mfc="white", mec=c, mew=1.5,
                label=f"{lab} (Zhou Fig. 6)")
    ax.axhline(1e-3, color=INK2, lw=0.8, ls="--")
    ax.text(9.5, 1.3e-3, "BER = 1e-3", ha="right", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-11, 1)
    ax.set_xlabel("Jamming-to-signal ratio (dB)"); ax.set_ylabel("Bit error rate")
    ax.set_title("CGAN reproduction vs Zhou 2025 Fig. 6 (QPSK, SNR 30 dB)", loc="left")
    _clean_axes(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def figure_implied_gain(jsr, gains_ours, zero, gains_paper, sync, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key in MAIN:
        c, lab = COLOR[key], LABEL[key]
        g = np.array([np.nan if v is None else v for v in gains_ours[key]], float)
        z = np.array(zero[key])
        ax.plot(jsr[~z], g[~z], color=c, lw=2, marker="s", ms=6,
                label=f"{lab} (ours{'' if key == 'noise' else ', ' + sync})")
        if (z & np.isfinite(g)).any():
            ax.plot(jsr[z], g[z], ls="none", marker="^", ms=8, mfc="white", mec=c, mew=1.5)
        gp = np.array([np.nan if v is None else v for v in gains_paper[key]], float)
        ax.plot(jsr, gp, color=c, lw=1.2, ls=(0, (4, 2)), marker="o", ms=7, mfc="white", mec=c,
                mew=1.5, label=f"{lab} (Zhou Fig. 6)")
    ax.set_xlabel("Jamming-to-signal ratio (dB)")
    ax.set_ylabel("Implied Gaussian processing gain g (dB)")
    ax.set_title("Implied g from BER = Q(√(1/(1/(sps·SNR) + JSR/g)))  —  flat = Gaussian-like;"
                 "  △ = lower bound (0 errors)", loc="left", fontsize=9)
    _clean_axes(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def figure_constellation(pts, stats, out, n_show=4000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    panels = [("clean", "Victim signal", INK2), ("gan_offset0", "G output, symbol instants", C_GAN),
              ("gan_offset_half", "G output, +sps/2 offset", C_GAN)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    rng = np.random.default_rng(0)
    for ax, (k, title, c) in zip(axes, panels):
        z = pts[k]
        z = z[rng.choice(len(z), size=min(n_show, len(z)), replace=False)]
        ax.scatter(z.real, z.imag, s=3, color=c, alpha=0.35, linewidths=0)
        ax.scatter(np.array([1, 1, -1, -1]) / math.sqrt(2), np.array([1, -1, 1, -1]) / math.sqrt(2),
                   s=40, marker="+", color=INK, linewidths=1.2)
        ax.set_title(f"{title}\nEVM {stats[k]['evm_rms']:.3f}, power {stats[k]['mean_power']:.3f}",
                     loc="left", fontsize=9)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        _clean_axes(ax)
    fig.suptitle("After the victim's matched filter, scaled to the signal's power (JSR 0 dB)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run001")
    ap.add_argument("--tag", default="", help="output suffix: <run>_<tag>_*")
    ap.add_argument("--sync", choices=jammers.SYNC_VARIANTS, default=lk.LINK["optimal_sync"])
    ap.add_argument("--sensitivity-sync", choices=jammers.SYNC_VARIANTS, default=None)
    ap.add_argument("--min-errors", type=int, default=100)
    ap.add_argument("--max-bits", type=float, default=1e9)
    args = ap.parse_args()
    sens = args.sensitivity_sync if args.sensitivity_sync != args.sync else None
    out = args.run + (f"_{args.tag}" if args.tag else "")

    device = lk.setup(seed=7)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    G, ckpt = load_generator(args.run, device)
    jsr, paper, pflags = lk.load_paper_fig6()

    def opt_fn(s):
        return lambda j: jammers.make("optimal", j, sync=s)

    def gan_fn(s):
        return lambda j: jammers.make("gan", j, G=G, scale=ckpt["scale"], z_dim=models.Z_DIM,
                                      seg_len=ckpt["seg_len"], sync=s)

    jammer_fns = {"noise": lambda j: jammers.make("noise", j, band=lk.LINK["noise_band"]),
                  "optimal": opt_fn(args.sync), "gan": gan_fn(args.sync)}
    if sens:
        jammer_fns[f"optimal_{sens}"] = opt_fn(sens)
        jammer_fns[f"gan_{sens}"] = gan_fn(sens)
    ours = {}
    for name, fn in jammer_fns.items():
        print(f"\n{name}:", flush=True)
        ours[name] = sweep(L, jsr, fn, lk.SNR_DB, args.min_errors, args.max_bits)

    # Crossing uses the plotted values, so a zero-error point enters at its 95%
    # upper bound. Per link.jsr_at_ber that makes the result a LOWER BOUND on the
    # true crossing JSR when the left bracket had zero errors; `cross_is_bound`
    # flags those. (Excluding zero-error points instead returns None whenever the
    # crossing sits just past one, which hid the Optimal crossing in run001.)
    cross, cross_is_bound = {}, {}
    for k in jammer_fns:
        cross[k] = lk.jsr_at_ber(jsr, [r["plot"] for r in ours[k]])
        i = None if cross[k] is None else int(np.searchsorted(jsr, cross[k])) - 1
        cross_is_bound[k] = bool(i is not None and 0 <= i < len(ours[k]) and ours[k][i]["zero"])
    paper_cross = {k: lk.jsr_at_ber(jsr, paper[k]) for k in MAIN}

    def gap(a, b, d):
        return None if (d.get(a) is None or d.get(b) is None) else round(d[a] - d[b], 2)

    gains_ours = {k: implied_gain_db(jsr, [r["plot"] for r in ours[k]], L.sps, lk.SNR_DB) for k in MAIN}
    gains_paper = {k: implied_gain_db(jsr, paper[k], L.sps, lk.SNR_DB) for k in MAIN}
    pts, cstats = constellation(L, G, ckpt)

    report = dict(
        run=args.run, tag=args.tag, snr_db=lk.SNR_DB, jsr_db=jsr.tolist(), link=lk.LINK,
        sync=args.sync, sensitivity_sync=sens, recipe=ckpt.get("recipe"),
        ber={k: [r["plot"] for r in ours[k]] for k in jammer_fns},
        zero_error={k: [r["zero"] for r in ours[k]] for k in jammer_fns},
        errors={k: [r["errors"] for r in ours[k]] for k in jammer_fns},
        bits={k: [r["bits"] for r in ours[k]] for k in jammer_fns},
        jsr_at_1e3={k: (None if v is None else round(v, 2)) for k, v in cross.items()},
        jsr_at_1e3_is_lower_bound=cross_is_bound,
        paper_jsr_at_1e3={k: round(v, 2) for k, v in paper_cross.items()},
        gaps_ours=dict(noise_minus_gan=gap("noise", "gan", cross),
                       gan_minus_optimal=gap("gan", "optimal", cross)),
        gaps_sensitivity=None if not sens else dict(
            noise_minus_gan=gap("noise", f"gan_{sens}", cross),
            gan_minus_optimal=gap(f"gan_{sens}", f"optimal_{sens}", cross)),
        gaps_paper=dict(noise_minus_gan=gap("noise", "gan", paper_cross),
                        gan_minus_optimal=gap("gan", "optimal", paper_cross)),
        implied_gain_db=dict(ours=gains_ours, paper=gains_paper),
        gan_constellation=cstats,
    )
    with open(os.path.join(ART, f"{out}_ber_vs_jsr.json"), "w") as f:
        json.dump(report, f, indent=2)
    figure(jsr, ours, paper, pflags, args.sync, sens, os.path.join(ART, f"{out}_ber_vs_jsr.png"))
    figure_implied_gain(jsr, gains_ours, {k: [r["zero"] for r in ours[k]] for k in MAIN},
                        gains_paper, args.sync, os.path.join(ART, f"{out}_implied_gain.png"))
    figure_constellation(pts, cstats, os.path.join(ART, f"{out}_iq.png"))

    print("\n--- REPORT (no pass/fail; §2.10) ---")
    print(f"sync {args.sync}" + (f" | sensitivity {sens}" if sens else ""))
    print(f"JSR at BER 1e-3:  ours {report['jsr_at_1e3']}")
    print(f"                 paper {report['paper_jsr_at_1e3']}")
    print(f"Noise - GAN:      ours {report['gaps_ours']['noise_minus_gan']} dB  "
          f"(paper {report['gaps_paper']['noise_minus_gan']} dB)")
    print(f"GAN - Optimal:    ours {report['gaps_ours']['gan_minus_optimal']} dB  "
          f"(paper {report['gaps_paper']['gan_minus_optimal']} dB)")
    if sens:
        print(f"[{sens}] Noise - GAN {report['gaps_sensitivity']['noise_minus_gan']} dB, "
              f"GAN - Optimal {report['gaps_sensitivity']['gan_minus_optimal']} dB")
    print(f"G constellation: {cstats}")
    print(f"wrote {out}_ber_vs_jsr.json/.png, {out}_implied_gain.png, {out}_iq.png to {ART}")


if __name__ == "__main__":
    main()
