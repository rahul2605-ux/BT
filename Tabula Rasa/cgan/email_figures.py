"""
Jargon-free figures for the supervisor emails (README Sec. 3.1). Login node: reads the
E2 / E2b JSONs through snr_figures' and outlier_figures' loaders, so every number is the
one behind README Sec. 3.3g. CGAN_RUN picks the data AND the email:

    cd cgan && python email_figures.py                   # run001: the 2026-09-24 email (D1 vs D2)
    cd cgan && CGAN_RUN=run003 python email_figures.py   # run003: the 2026-09-28 email

run001 -> ../artifacts/cgan/snr_ablation/run001/email/
  email_fig1_tradeoff.png   BER vs P(det) of the spectrogram CNN at one SNR (the
                            whole trade-off curve, FAR dotted -- never only <= alpha)
  email_fig2_vs_snr.png     P(det) at matched BER and the power it takes, over SNR
run003 -> ../artifacts/cgan/snr_ablation/run003/email/
  email_fig1_cnn_vs_control.png    the paper method against its beta = 0 control: CNN
                                   P(det) at matched BER over SNR (no power panel, §2.7)
  email_fig2_frames_per_alarm.png  broken frames per extra alarm, CNN vs one-sided power
                                   (the bottom row of outlier_figures.fig_slopes)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import snr_figures as sf
from snr_figures import ALPHA, BER_REF, C, INK, INK2, GRID, MARK, style

OUT = os.path.join(sf.AD, "email")
DET = "spec_cnn"
ROWS = [("noise", "noise", "white noise"),
        ("amuru", "pulsed_p0.1", "best classical jammer (pulsed)"),
        ("plain", "plain_run001", "GAN jammer (Zhou et al., reproduced)"),
        ("g_cnn", "spec_cnn_b10", "same GAN, trained against the detector")]
GREY = "#9a9994"


def level(levels, snr):
    return next(d for d in levels if d["snr"] == snr)


def fig_tradeoff(levels, snr=15.0):
    d = level(levels, snr)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    style(ax)
    for key, tag, lab in ROWS:
        jsr, exc, pdet = sf.series(d, tag)
        ok = exc > 0                     # below the measurement floor there is no damage to plot
        col = GREY if key == "noise" else C[key]
        ax.plot(pdet[DET][ok], exc[ok], color=col, marker=MARK[key], ms=4.5,
                lw=2.4 if key in ("plain", "g_cnn") else 1.5,
                ls="-" if key in ("plain", "g_cnn") else (0, (4, 2)), label=lab, zorder=3)
    ax.axvline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax.text(ALPHA + 0.012, 0.3, "detector's\nfalse-alarm rate", color=INK2, fontsize=8, va="top")
    p1, _ = sf.at_matched_ber(d, "plain_run001", DET)
    p2, _ = sf.at_matched_ber(d, "spec_cnn_b10", DET)
    ax.annotate("", xy=(p2, BER_REF), xytext=(p1, BER_REF),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text((p1 + p2) / 2, BER_REF * 1.6, f"same damage: flagged in {p1:.0%} → {p2:.0%} of frames",
            ha="center", fontsize=8.8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("P(detect) — fraction of jammed frames the detector flags", color=INK2, fontsize=9)
    ax.set_ylabel("bit error rate caused (above clean)", color=INK2, fontsize=9)
    ax.set_title(f"Spectrogram-CNN detector, SNR {snr:g} dB — curves traced by jammer power\n"
                 "up-left is better for the attacker", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    return fig


def fig_vs_snr(levels):
    cur = {k: sf._curve(levels, t, DET, BER_REF) for k, t, _ in ROWS if k != "noise"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    style(ax1); style(ax2)
    x, p1, j1 = cur["plain"]
    _, p2, j2 = cur["g_cnn"]
    ok = ~np.isnan(p1) & ~np.isnan(p2)
    ax1.fill_between(x[ok], p2[ok], p1[ok], color=C["g_cnn"], alpha=0.10, lw=0)
    for key, tag, lab in ROWS[1:]:
        xs, p, j = cur[key]
        kw = dict(color=C[key], marker=MARK[key], ms=5.5, label=lab, zorder=3,
                  lw=1.5 if key == "amuru" else 2.4, ls=(0, (4, 2)) if key == "amuru" else "-")
        ax1.plot(xs, p, **kw)
        ax2.plot(xs, j, **kw)
    ax1.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax1.text(40.4, ALPHA, "false-\nalarm", color=INK2, fontsize=7.5, va="center")
    i15, i30 = int(np.argmin(np.abs(x - 15))), int(np.argmin(np.abs(x - 30)))
    ax1.annotate(f"{p1[i15]:.0%} → {p2[i15]:.0%}\nat 15 dB", xy=(15, (p1[i15] + p2[i15]) / 2),
                 xytext=(1.0, 0.72), fontsize=8.8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax1.annotate(f"{p1[i30]:.0%} → {p2[i30]:.0%} at 30 dB,\nthe SNR in Zhou et al.'s protocol",
                 xy=(30, p2[i30]), xytext=(26.5, 0.21), fontsize=8.8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax1.set_ylabel(f"P(detect) at equal damage (BER {BER_REF:g})", color=INK2, fontsize=9)
    ax1.set_title("how often the CNN flags it (lower = stealthier)", color=INK, fontsize=10, loc="left")
    sav = (j1 - j2)[(x >= 10) & ~np.isnan(j1 - j2)]
    ax2.annotate(f"{sav.min():.1f}–{sav.max():.1f} dB less power\nat every SNR ≥ 10 dB",
                 xy=(30, -13), xytext=(28, -31), ha="center", va="center", fontsize=8.8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID),
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax2.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax2.set_ylabel("jammer power needed for that damage (JSR) [dB]", color=INK2, fontsize=9)
    ax2.set_title("the jammer power it takes", color=INK, fontsize=10, loc="left")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.8, labelcolor=INK2, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------- the 2026-09-28 email (run003)
ROWS_CTRL = [("eff", "eff", "learned jammer, trained for damage only (control)"),
             ("g_cnn", "spec_cnn_b10", "same, also trained against the CNN"),
             ("amuru", "pulsed_p0.1", "best classical jammer (pulsed QPSK)")]


def fig_cnn_vs_control(levels):
    cur = {k: sf._curve(levels, t, DET, BER_REF) for k, t, _ in ROWS_CTRL}
    fig, ax = plt.subplots(figsize=(7.4, 4.5))       # no power panel: power is secondary (README §2.7)
    style(ax)
    x, p0, _ = cur["eff"]
    _, p1, _ = cur["g_cnn"]
    ok = ~np.isnan(p0) & ~np.isnan(p1)
    ax.fill_between(x[ok], p1[ok], p0[ok], color=C["g_cnn"], alpha=0.10, lw=0)
    for key, tag, lab in ROWS_CTRL:
        xs, p, _ = cur[key]
        ax.plot(xs, p, color=C[key], marker=MARK[key], ms=5.5, label=lab, zorder=3,
                lw=1.5 if key == "amuru" else 2.4, ls=(0, (4, 2)) if key == "amuru" else "-")
    ax.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax.text(40.4, ALPHA, "false-\nalarm", color=INK2, fontsize=7.5, va="center")
    i30 = int(np.argmin(np.abs(x - 30)))
    ax.annotate(f"{p0[i30]:.0%} → {p1[i30]:.0%} of frames at 30 dB,\nthe SNR the CNN was trained at",
                xy=(30, (p0[i30] + p1[i30]) / 2), xytext=(1.0, 0.72), fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax.set_ylabel(f"P(detect) at equal damage (BER {BER_REF:g})", color=INK2, fontsize=9)
    ax.set_title("How often the CNN detector flags the jammer (lower = stealthier)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0, 0.98))
    fig.tight_layout()
    return fig


def fig_frames_per_alarm():
    import outlier_figures as of
    S = of.slopes()
    names = {"noise": "white noise", "pulsed_p1": "matched QPSK, constant",
             "pulsed_p0.1": "pulsed QPSK (classical)", "eff": "learned, damage only (control)",
             "spec_cnn_b10": "learned, also trained against the CNN"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ys = np.arange(len(of.SLOPE_JAMMERS))[::-1]
    for ax, (det, title) in zip(axes, [("spec_cnn", "against the learned CNN detector"),
                                       ("power_one_sided", "against a plain energy detector")]):
        style(ax)
        for y, (tag, ck, _) in zip(ys, of.SLOPE_JAMMERS):
            v15, v30 = (S[f"{s}|{det}|{tag}|per"]["value"] for s in (15, 30))
            ax.plot([v15, v30], [y, y], color=GRID, lw=1.2, zorder=1)
            ax.scatter([v15], [y], s=58, facecolors="white", edgecolors=C[ck], lw=1.8, zorder=3,
                       marker=MARK[ck])
            ax.scatter([v30], [y], s=58, color=C[ck], zorder=3, marker=MARK[ck])
            if tag == "spec_cnn_b10" and det == "spec_cnn":
                for v in (v15, v30):
                    ax.text(v, y + 0.32, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center",
                            fontsize=8.5, color=INK)
        ax.set_xscale("log"); ax.set_xlim(0.3, 60)
        ax.axvline(1.0, color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=0)
        ax.set_yticks(ys); ax.set_yticklabels([names[t] for t, _, _ in of.SLOPE_JAMMERS],
                                              fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("frames broken per alarm above the false-alarm rate (log)", color=INK2, fontsize=9)
    for face, lab in (("white", "SNR 15 dB"), (INK2, "SNR 30 dB")):   # neutral: no jammer's colour
        axes[1].scatter([], [], s=58, facecolors=face, edgecolors=INK2, lw=1.8, marker="o", label=lab)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    fig.suptitle("Damage per extra alarm, each jammer at its best power (false-alarm rate 5 %)",
                 color=INK, fontsize=10.5, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    levels = sf.load()
    os.makedirs(OUT, exist_ok=True)
    figs = ([("email_fig1_tradeoff.png", fig_tradeoff(levels)),
             ("email_fig2_vs_snr.png", fig_vs_snr(levels))] if sf.RUN == "run001" else
            [("email_fig1_cnn_vs_control.png", fig_cnn_vs_control(levels)),
             ("email_fig2_frames_per_alarm.png", fig_frames_per_alarm())])
    for name, fig in figs:
        path = os.path.join(OUT, name)
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        print("wrote", os.path.relpath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
