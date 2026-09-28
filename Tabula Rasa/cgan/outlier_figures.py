"""
E2b figure -- which jammer wins depends on the damage metric (README §3.3g, E2b).

    CGAN_RUN=run003 sbatch submit_outlier_figures.sh

Rows: average BER, frame error rate (uncoded frames with >= 1 bit error). Columns:
SNR 15 and 30 dB. x: per-frame P(det) of one detector at alpha (default the CNN at
0.05). Each jammer is drawn as its frontier -- the best damage it reaches at P(det)
<= x when it may choose its (constant) power, i.e. the running maximum over the JSR
sweep -- and the on/off strategy (matched QPSK, loud on a fraction of frames, silent
otherwise) as the straight line it traces. Reads outlier_alarm.py's grid JSONs.
"""

import json
import os
import sys

import numpy as np

from snr_figures import style, C, MARK, INK, INK2, GRID, plt

RUN = os.environ.get("CGAN_RUN", "run003") + "_per"
AD = os.path.join("..", "artifacts", "cgan", "outlier_alarm", RUN)
ROWS = [("pulsed_p0.1", "amuru", "Amuru pulsed (p=0.1), classical"),
        ("eff", "eff", "trained on damage only, β=0"),
        ("spec_cnn_b10", "g_cnn", "trained against the CNN, β=10")]


def frontier(pts, det, alpha, key, far):
    """(x, y): running max of `key` over P(det) <= x, P(det) clipped at the FAR."""
    xy = sorted((max(p["pdet"][det][alpha], far), p[key]) for p in pts)
    xs, ys, best = [], [], 0.0
    for x, y in xy:
        best = max(best, y)
        if best > 0:
            xs.append(x); ys.append(best)
    return np.array(xs), np.array(ys)


def main(det="spec_cnn", alpha="0.05"):
    levels = [json.load(open(os.path.join(AD, f"snr_{s}.json"))) for s in (15, 30)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    for col, d in enumerate(levels):
        far = d["far"][det][alpha]
        loud = d["jammers"]["pulsed_p1"][-1]
        for row, (key, lab) in enumerate([("ber", "average BER"), ("per", "frame error rate")]):
            ax = axes[row, col]; style(ax)
            for tag, ck, name in ROWS:
                x, y = frontier(d["jammers"][tag], det, alpha, key, far)
                ax.plot(x, y, color=C[ck], lw=2.2, marker=MARK[ck], ms=5, markevery=4,
                        label=name, zorder=3)
            f = np.geomspace(1e-6, 1.0, 300)            # fraction of frames the jammer is loud on
            xs = far + f * (1.0 - far)
            ax.plot(xs, f * loud[key], color=INK2, lw=1.6, ls=(0, (5, 3)), zorder=2,
                    label=f"on/off: matched QPSK at {loud['jsr']:+.0f} dB on a fraction of frames")
            ax.axvline(far, color=INK2, lw=1.0, ls=(0, (2, 2)))
            ax.set_yscale("log"); ax.set_ylim(1e-5, 1.5)
            ax.set_xscale("log"); ax.set_xlim(0.03, 1.05)
            ax.set_ylabel(lab, color=INK2, fontsize=9)
            if row == 0:
                ax.set_title(f"SNR {d['meta']['snr_db']:g} dB", color=INK, fontsize=10.5, loc="left")
            else:
                ax.set_xlabel(f"per-frame P(det) of the {'spectrogram CNN' if det == 'spec_cnn' else det} "
                              f"at α = {alpha} (dotted: false-alarm rate)", color=INK2, fontsize=9)
    fig.suptitle("Best damage at a given detectability, the jammer choosing its power — "
                 "average BER rewards on/off, frame errors reward spreading",
                 color=INK, fontsize=11.5, x=0.012, ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    out = os.path.join(AD, f"fig_frontier_ber_vs_per_{det}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"wrote {out}")


SLOPE_JAMMERS = [("noise", "noise", "white noise (barrage)"),
                 ("pulsed_p1", "optimal", "matched QPSK"),
                 ("pulsed_p0.1", "amuru", "Amuru pulsed (p=0.1)"),
                 ("eff", "eff", "trained on damage only, β=0"),
                 ("spec_cnn_b10", "g_cnn", "trained against the CNN, β=10")]


def slopes(alpha="0.05"):
    """
    Damage per extra alarm: max over the JSR sweep of damage / (P(det) - FAR), counting
    only points whose excess over the FAR is at least 2 binomial sigma (512 frames), so a
    noise-level denominator cannot inflate it. It is the slope of the jammer's frontier
    from the no-attack point -- what time-sharing that operating point with silence
    achieves -- and it needs no power axis.
    """
    import math
    out = {}
    for s in (15, 30):
        d = json.load(open(os.path.join(AD, f"snr_{s}.json")))
        n = d["meta"]["frames"]
        for det in ("spec_cnn", "power_one_sided"):
            far = d["far"][det][alpha]
            tol = 2 * math.sqrt(float(alpha) * (1 - float(alpha)) / n)
            for tag, _, _ in SLOPE_JAMMERS:
                for key in ("ber", "per"):
                    c = [(p[key] / (p["pdet"][det][alpha] - far), p["jsr"]) for p in d["jammers"][tag]
                         if p["pdet"][det][alpha] - far >= tol and p[key] > 0]
                    v, x = max(c) if c else (0.0, None)
                    out[f"{s}|{det}|{tag}|{key}"] = dict(value=v, jsr=x)
    return out


def fig_slopes(alpha="0.05"):
    S = slopes(alpha)
    with open(os.path.join(AD, "damage_per_extra_alarm.json"), "w") as f:
        json.dump(S, f, indent=1)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharey=True)
    rows = [("ber", "bit errors per extra alarm"), ("per", "broken frames per extra alarm")]
    cols = [("spec_cnn", "against the spectrogram CNN"), ("power_one_sided", "against one-sided power")]
    ys = np.arange(len(SLOPE_JAMMERS))[::-1]
    for r, (key, xl) in enumerate(rows):
        for c, (det, title) in enumerate(cols):
            ax = axes[r, c]; style(ax)
            for y, (tag, ck, name) in zip(ys, SLOPE_JAMMERS):
                v15 = S[f"15|{det}|{tag}|{key}"]["value"]; v30 = S[f"30|{det}|{tag}|{key}"]["value"]
                ax.plot([v15, v30], [y, y], color=GRID, lw=1.2, zorder=1)
                ax.scatter([v15], [y], s=58, facecolors="white", edgecolors=C[ck], lw=1.8, zorder=3,
                           marker=MARK[ck], label="15 dB SNR" if y == ys[0] else None)
                ax.scatter([v30], [y], s=58, color=C[ck], zorder=3, marker=MARK[ck],
                           label="30 dB SNR" if y == ys[0] else None)
            ax.set_xscale("log"); ax.set_xlim(0.01, 60)
            ax.axvline(1.0, color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=0)
            ax.set_yticks(ys); ax.set_yticklabels([n for _, _, n in SLOPE_JAMMERS], fontsize=9, color=INK)
            if r == 0:
                ax.set_title(title, color=INK, fontsize=10.5, loc="left")
            ax.set_xlabel(xl + f" (α = {alpha})", color=INK2, fontsize=9)
    fig.suptitle("Damage per extra alarm, each jammer at its best power (open: 15 dB SNR, filled: 30 dB)",
                 color=INK, fontsize=11.5, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(AD, "fig_damage_per_extra_alarm.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for det in ("spec_cnn", "power_one_sided"):
        main(det)
    fig_slopes()
