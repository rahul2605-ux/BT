"""
S1 -- the shadowing ablation's read-out (README §3.3k). Login node, no Sionna.

    python shadow_figures.py                  # tables from snr_ablation/shadow_run003/

Per shadowing level: each detector's clean FAR, P(det) at matched excess BER 3e-4
(the E2 convention, snr_figures._cross_log), frames broken per extra alarm (PER over
P(det) - FAR, best over JSR, excess >= 2 sigma -- the E2b power-free unit), and the
best confirmed stealthy BER inside the budget. At sigma 0 there is no power_csi row:
it equals power_one_sided there by construction (detectors.power_csi).
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_figures import BER_FLOOR, GRID, INK, INK2, _cross_log, style

AD = "../artifacts/cgan/snr_ablation"
DETS = ["power_one_sided", "power_csi", "kurtosis", "spec_cnn"]
SHORT = {"power_one_sided": "pow", "power_csi": "csi", "power_two_sided": "pow2",
         "kurtosis": "kurt", "spec_cnn": "cnn", "lrt_noise": "lrt"}
A, BER_REF = "0.05", 3e-4
JAM = [("noise", "noise"), ("pulsed_p1", "matched QPSK"), ("pulsed_p0.1", "Amuru p0.1"),
       ("eff", "GAN beta0"), ("power_one_sided_b10", "GAN pow1s b10"),
       ("spec_cnn_b1", "GAN cnn b1"), ("spec_cnn_b10", "GAN cnn b10"), ("omniscient_e1", "genie")]


def pdet(p, det):
    if det == "power_csi" and det not in p["pdet"]:
        det = "power_one_sided"          # sigma 0: identical by construction
    return p["pdet"].get(det, {}).get(A, np.nan)


def load(run):
    fs = [f for f in glob.glob(os.path.join(AD, run, "shadow_*.json")) if not f.endswith("_smoke.json")]
    return sorted((json.load(open(f)) for f in fs), key=lambda d: d["meta"]["shadow_db"])


def matched(d, pts, det):
    """(P(det), JSR) at the first JSR reaching BER_REF excess BER."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = np.array([p["ber"] for p in pts]) - d["clean"]["ber"]
    x = _cross_log(jsr, np.where(exc >= BER_FLOOR, exc, 0.0), BER_REF)
    if x is None:
        return np.nan, None
    return float(np.interp(x, jsr, [pdet(p, det) for p in pts])), x


def frames_per_alarm(d, pts, det):
    """Best over JSR of excess PER / (P(det) - FAR), extra alarms >= 2 binomial sigma."""
    far, per0 = pdet(d["clean"], det), d["clean"].get("per", 0.0)
    best = np.nan
    for p in pts:
        extra = pdet(p, det) - far
        if "per" in p and extra >= 2 * math.sqrt(far * (1 - far) / p["frames"]):
            v = (p["per"] - per0) / extra
            best = v if np.isnan(best) else max(best, v)
    return best


def summary(levels):
    for d in levels:
        s, c = d["meta"]["shadow_db"], d["clean"]
        print(f"\n=== shadowing {s:g} dB | clean BER {c['ber']:.1e} PER {c.get('per', 0):.4f} | FAR "
              + " ".join(f"{SHORT[k]} {pdet(c, k):.3f}" for k in DETS))
        print(f"  {'jammer':<15}{'JSR@3e-4':>9}  P(det): " + " ".join(f"{SHORT[k]:>6}" for k in DETS)
              + "   frames/extra alarm: " + " ".join(f"{SHORT[k]:>6}" for k in DETS))
        for key, lab in JAM:
            pts = d["attacks"].get(key) or d["generators"].get(key)
            if not pts:
                continue
            row = [matched(d, pts, det) for det in DETS]
            x = row[0][1]
            print(f"  {lab:<15}{('%+.1f' % x) if x is not None else '--':>9}          "
                  + " ".join(f"{p:6.3f}" for p, _ in row) + "                       "
                  + " ".join(f"{frames_per_alarm(d, pts, det):6.2f}" for det in DETS))
        best = {}
        for grp in ("confirmed_attacks", "confirmed_generators"):
            for name, conf in d.get(grp, {}).items():
                if name.startswith("omniscient"):
                    continue                          # the genie is the reference, not a result
                for det, by_a in conf.items():
                    pick = by_a.get(A)
                    if pick and pick.get("passed") and pick["confirmed_ber"] > best.get(det, (0.0, ""))[0]:
                        best[det] = (pick["confirmed_ber"], name)
        print("  confirmed stealthy BER @ alpha 0.05 (non-genie): "
              + "  ".join(f"{SHORT.get(k, k)} {v[0]:.1e} ({v[1]})" for k, v in best.items()))


# ---------------------------------------------------------------- linear frontier
# The whole damage range on a LINEAR axis (user, 2026-09-28): the stealthy region is
# one end of the curve, not the evaluation. Reference jammers dashed, generators
# solid, one marker each -- two pairs sit at CVD dE 6.4-7.6 (genie/GAN-CNN protan,
# GAN-beta0/GAN-power deutan), legal only with that secondary encoding.
FRONT = [("omniscient_e1", "genie (knows the symbols)", "#333333", "X", (0, (5, 2))),
         ("noise", "white noise", "#0072B2", "o", (0, (5, 2))),
         ("pulsed_p1", "matched QPSK", "#D55E00", "P", (0, (5, 2))),
         ("pulsed_p0.1", "Amuru pulsed (p = 0.1)", "#E69F00", "D", (0, (5, 2))),
         ("eff", "GAN, no detection term (β = 0)", "#009E73", "^", "-"),
         ("power_one_sided_b10", "GAN vs power (β = 10)", "#CC79A7", "v", "-"),
         ("spec_cnn_b10", "GAN vs CNN (β = 10)", "#882255", "s", "-")]
FRONT_DETS = [("power_one_sided", "power threshold (one-sided)"), ("kurtosis", "kurtosis"),
              ("spec_cnn", "spectrogram CNN")]
BUDGET = 0.5


def at_budget(pts, det, key, budget=BUDGET):
    """Max damage `key` over sweep points with P(det) <= budget (None if none qualifies)."""
    ok = [p[key] for p in pts if pdet(p, det) <= budget and key in p]
    return max(ok) if ok else None


def fig_linear(d, out):
    fig, axes = plt.subplots(1, len(FRONT_DETS), figsize=(13.5, 4.6), sharey=True)
    for ax, (det, title) in zip(axes, FRONT_DETS):
        style(ax)
        far = pdet(d["clean"], det)
        ax.axvline(far, color=INK2, lw=1, ls=":", zorder=1)
        ax.axvline(BUDGET, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0.5, color=GRID, lw=1.2, zorder=1)
        for key, lab, col, mk, ls in FRONT:
            pts = d["attacks"].get(key) or d["generators"].get(key)
            x = [pdet(p, det) for p in pts]
            y = [p["ber"] for p in pts]
            ax.plot(x, y, color=col, lw=2, ls=ls, marker=mk, ms=6, markevery=5, label=lab,
                    zorder=3, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("P(det) per frame, α = 0.05", color=INK2)
        ax.text(far + 0.015, 0.97, "FAR", color=INK2, fontsize=8, va="top")
        ax.text(BUDGET + 0.015, 0.97, "P(det) = 0.5", color=INK2, fontsize=8, va="top")
        ax.text(0.99, 0.51, "random guessing", color=INK2, fontsize=8, ha="right", va="bottom")
    axes[0].set_ylabel("BER (linear)", color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.12), labelcolor=INK)
    s = d["meta"]["shadow_db"]
    fig.suptitle(f"Damage vs detection over the whole range — 30 dB SNR, "
                 f"{'ideal link' if not s else f'shadowing {s:g} dB'}, one point per JSR (−50…+15 dB)",
                 color=INK, fontsize=11, y=1.02)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


def budget_table(d):
    print(f"\nmax damage with P(det) <= {BUDGET} (over the JSR grid), BER / PER:")
    print(f"  {'jammer':<34}" + "".join(f"{t[:22]:>24}" for _, t in FRONT_DETS))
    for key, lab, *_ in FRONT:
        pts = d["attacks"].get(key) or d["generators"].get(key)
        cells = []
        for det, _ in FRONT_DETS:
            b, p = at_budget(pts, det, "ber"), at_budget(pts, det, "per")
            cells.append("--" if b is None else f"{b:.4f} / {p:.3f}")
        print(f"  {lab:<34}" + "".join(f"{c:>24}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shadow_run003")
    ap.add_argument("--linear", type=float, default=None, metavar="SIGMA",
                    help="linear BER-vs-P(det) figure + P(det)<=0.5 table at this shadowing level")
    args = ap.parse_args()
    levels = load(args.run)
    if args.linear is None:
        summary(levels)
        return
    d = next(x for x in levels if x["meta"]["shadow_db"] == args.linear)
    fig_linear(d, os.path.join(AD, args.run, f"fig_frontier_linear_{args.linear:g}.png"))
    budget_table(d)


if __name__ == "__main__":
    main()
