"""
D6 round 1 -- the read-out (README §3.3o). Login node, no Sionna.

    python arms_figures.py [--figs]           # tables (+ fig_gain/fig_cost.png) from baselines/arms/eval/round1/

Per SNR, one column per defender (r0 / A / B, arms_eval.py):
  * the COST: the CNN on Li et al.'s own classes -- argmax accuracy and FAR (Li's
    metric, as Phase 0.5 reported it), detection rate at alpha, AUC, and per type
    the JSR where P(det) first reaches 0.5;
  * the GAIN: per jammer, CNN P(det) at matched excess BER 3e-4, frames broken per
    extra alarm (the E2b unit) and the most BER / PER with P(det) <= 0.5 --
    shadow_figures' helpers unchanged, so the numbers are S1's definitions. Energy
    (one-sided power) is the floor line: the same frames, a detector nobody retrains.
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_figures import at_budget, frames_per_alarm, matched
from snr_figures import GRID, INK, INK2, style

AD = "../artifacts/cgan/baselines/arms/eval"
ARMS = ["r0", "A", "B"]
ROLES = ["classical", "control", "seen", "twin", "held-out"]
# one hue per DEFENDER, fixed order (the reference palette's first three slots, which
# pass all-pairs CVD / normal-vision in light mode); a marker each, because aqua sits
# below 3:1 contrast on white
ARM_STYLE = {"r0": ("#2a78d6", "o", "round 0: Li's range, never saw the attacker"),
             "A": ("#eb6834", "s", "A: range widened to −35 dB, no attacker frames"),
             "B": ("#1baf7a", "D", "B: A + the round-0 attackers' frames")}
LABEL = {"noise": "white noise", "pulsed_p1": "matched QPSK", "pulsed_p0.1": "Amuru pulsed p 0.1",
         "eff": "GAN β = 0 (control)", "r0_b1": "GAN vs CNN β 1 (seen)", "r0_b10": "GAN vs CNN β 10 (seen)",
         "twin_cold_b10": "β 10, same init (twin)", "twin_grey_b1": "grey-box β 1 (twin)",
         "twin_grey_b10": "grey-box β 10 (twin)", "held_r1_b10": "β 10 seed 1 (held out)",
         "held_r2_b10": "β 10 seed 2 (held out)", "held_r3_b10": "β 10 seed 3 (held out)"}


def load(run):
    """{snr: {arm: eval JSON}}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(AD, run, "snr*_*.json"))):
        if f.endswith("_smoke.json"):
            continue
        d = json.load(open(f))
        if "runtime_s" not in d:                  # still being written by arms_eval.py
            print(f"skipping {os.path.relpath(f)}: incomplete")
            continue
        out.setdefault(d["meta"]["snr_db"], {})[d["meta"]["arm"]] = d
    return out


def pts_of(d, name):
    return d["attacks"].get(name) or d["generators"].get(name)


def half_jsr(curve, grid):
    """First JSR [dB] where P(det) reaches 0.5 (None if never)."""
    return next((g for g, p in zip(grid, curve) if p >= 0.5), None)


def cost_table(by_arm):
    arms = [a for a in ARMS if a in by_arm]
    print("  Li et al.'s classes (the cost side)" + "".join(f"{a:>12}" for a in arms))
    for rng in ("li_range", "wide_range"):
        m = {a: by_arm[a]["li"]["mixture"][rng] for a in arms}
        lab = f"[{m[arms[0]]['jsr_range_db'][0]:+.0f}, {m[arms[0]]['jsr_range_db'][1]:+.0f}] dB"
        for key, name in (("accuracy", "accuracy (argmax)"), ("far_argmax", "FAR (argmax)"),
                          ("dr_alpha", "DR at alpha 0.05"), ("far_alpha", "FAR at alpha 0.05"),
                          ("auc", "AUC")):
            print(f"    {lab:<14}{name:<22}" + "".join(f"{m[a][key]:12.4f}" for a in arms))
    li = by_arm[arms[0]]["li"]
    for k in li["pdet_vs_jsr"]:
        print(f"    P(det) >= 0.5 from JSR [dB], {k:<17}" + "".join(
            f"{str(half_jsr(by_arm[a]['li']['pdet_vs_jsr'][k], li['jsr_db'])):>12}" for a in arms))


def gain_table(by_arm):
    arms = [a for a in ARMS if a in by_arm]
    ref = by_arm[arms[0]]
    roles = ref["meta"]["roles"]
    print(f"\n  {'jammer':<16}{'role':<10}" + "".join(f"{'CNN P(det)@3e-4 ' + a:>20}" for a in arms)
          + "".join(f"{'frames/alarm ' + a:>16}" for a in arms) + f"{'energy':>9}"
          + "".join(f"{'BER/PER @P<=.5 ' + a:>22}" for a in arms))
    groups = {}
    for role in ROLES:
        for name, r in roles.items():
            if r != role:
                continue
            cells = []
            for a in arms:
                d = by_arm[a]
                p, x = matched(d, pts_of(d, name), "spec_cnn")
                cells.append(f"{p:.3f} ({'%+.0f' % x if x is not None else '--'})")
            fpa = [frames_per_alarm(by_arm[a], pts_of(by_arm[a], name), "spec_cnn") for a in arms]
            energy = frames_per_alarm(ref, pts_of(ref, name), "power_one_sided")
            bud = []
            for a in arms:
                pts = pts_of(by_arm[a], name)
                b, q = at_budget(pts, "spec_cnn", "ber"), at_budget(pts, "spec_cnn", "per")
                bud.append("--" if b is None else f"{b:.4f} / {q:.3f}")
            print(f"  {name:<16}{role:<10}" + "".join(f"{c:>20}" for c in cells)
                  + "".join(f"{v:16.2f}" for v in fpa) + f"{energy:9.2f}" + "".join(f"{c:>22}" for c in bud))
            groups.setdefault(role, []).append(fpa)
    for role in ("seen", "twin", "held-out"):
        if role in groups:
            v = np.array(groups[role])
            print(f"  {'mean ± sd':<16}{role:<10}" + " " * 20 * len(arms)
                  + "".join(f"{m:9.2f} ± {s:4.2f}" for m, s in zip(v.mean(0), v.std(0))))


# ---------------------------------------------------------------- figures
def fig_gain(data, out):
    """Per jammer and defender: CNN P(det) at matched BER, and frames broken per extra alarm."""
    snrs = sorted(data, reverse=True)
    ref = data[snrs[0]][next(a for a in ARMS if a in data[snrs[0]])]
    order = [n for role in reversed(ROLES) for n, r in ref["meta"]["roles"].items() if r == role]
    y = np.arange(len(order))
    fig, axes = plt.subplots(2, len(snrs), figsize=(6.2 * len(snrs), 10.5), sharey=True, squeeze=False)
    for col, snr in enumerate(snrs):
        by_arm = data[snr]
        for row, (title, xlab) in enumerate((("P(det) of the CNN at matched excess BER 3e-4",
                                               "P(det) per frame, α = 0.05 (lower = stealthier)"),
                                              ("frames broken per extra CNN alarm",
                                               "frames per extra alarm (log; higher = the jammer gains)"))):
            ax = axes[row][col]
            style(ax)
            for j, arm in enumerate(a for a in ARMS if a in by_arm):
                c, mk, lab = ARM_STYLE[arm]
                d = by_arm[arm]
                v = [matched(d, pts_of(d, n), "spec_cnn")[0] if row == 0
                     else frames_per_alarm(d, pts_of(d, n), "spec_cnn") for n in order]
                ax.plot(v, y + (j - 1) * 0.22, ls="none", marker=mk, ms=7, color=c, label=lab,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            if row == 1:
                ax.set_xscale("log")
                d0 = by_arm[next(a for a in ARMS if a in by_arm)]
                e = [frames_per_alarm(d0, pts_of(d0, n), "power_one_sided") for n in order]
                ax.axvline(np.nanmedian(e), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.text(np.nanmedian(e) * 1.06, len(order) - 0.6, "energy detector\n(every jammer ≈ 1)",
                        color=INK2, fontsize=8, va="top")
            else:
                ax.set_xlim(-0.02, 1.02)
                far = by_arm[next(a for a in ARMS if a in by_arm)]["clean"]["pdet"]["spec_cnn"]["0.05"]
                ax.axvline(far, color=INK2, lw=1, ls=":", zorder=1)
                ax.text(far + 0.01, len(order) - 0.6, "FAR", color=INK2, fontsize=8, va="top")
            for k in range(1, len(order)):
                if ref["meta"]["roles"][order[k]] != ref["meta"]["roles"][order[k - 1]]:
                    ax.axhline(k - 0.5, color=GRID, lw=1.2, zorder=0)
            ax.set_title(f"{snr:g} dB SNR — {title}", color=INK, fontsize=10.5)
            ax.set_xlabel(xlab, color=INK2, fontsize=9)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([LABEL.get(n, n) for n in order], color=INK, fontsize=9)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("D6 round 1 — the CNN retrains, the attackers stay frozen", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


def fig_cost(data, out):
    """The CNN on Li et al.'s own classes: P(det) at alpha vs JSR, per type and defender."""
    snrs = sorted(data, reverse=True)
    types = list(next(iter(data[snrs[0]].values()))["li"]["pdet_vs_jsr"])
    fig, axes = plt.subplots(len(snrs), len(types), figsize=(3.4 * len(types), 3.3 * len(snrs)),
                             sharex=True, sharey=True, squeeze=False)
    for row, snr in enumerate(snrs):
        for col, k in enumerate(types):
            ax = axes[row][col]
            style(ax)
            for arm in (a for a in ARMS if a in data[snr]):
                c, mk, lab = ARM_STYLE[arm]
                li = data[snr][arm]["li"]
                ax.plot(li["jsr_db"], li["pdet_vs_jsr"][k], color=c, lw=2, marker=mk, ms=5, markevery=3,
                        markeredgecolor="white", markeredgewidth=0.6, label=lab)
            ax.axvspan(-35, -20, color=GRID, alpha=0.5, lw=0, zorder=0)
            ax.axhline(0.05, color=INK2, lw=1, ls=":")
            ax.set_title(f"{k} — {snr:g} dB", color=INK, fontsize=10)
            if row == len(snrs) - 1:
                ax.set_xlabel("JSR [dB]", color=INK2, fontsize=9)
        axes[row][0].set_ylabel("P(det), α = 0.05", color=INK2, fontsize=9)
    axes[0][0].text(-27.5, 0.97, "added by\nthe widening", color=INK2, fontsize=7.5, ha="center", va="top")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("The cost side — the retrained CNNs on Li et al.'s four jammer types", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="round1")
    ap.add_argument("--figs", action="store_true", help="also write fig_gain.png and fig_cost.png")
    args = ap.parse_args()
    data = load(args.run)
    for snr in sorted(data, reverse=True):
        by_arm = data[snr]
        c = by_arm[next(a for a in ARMS if a in by_arm)]["clean"]
        print(f"\n=== {snr:g} dB | defenders {', '.join(a for a in ARMS if a in by_arm)} | clean BER "
              f"{c['ber']:.1e} | CNN FAR " + " ".join(f"{a} {by_arm[a]['clean']['pdet']['spec_cnn']['0.05']:.3f}"
                                                  for a in ARMS if a in by_arm))
        cost_table(by_arm)
        gain_table(by_arm)
    if args.figs:
        fig_gain(data, os.path.join(AD, args.run, "fig_gain.png"))
        fig_cost(data, os.path.join(AD, args.run, "fig_cost.png"))


if __name__ == "__main__":
    main()
