"""
Baselines -- figures and summary.json from the sweep JSONs (README §2.10, §3.4).

    sbatch submit_baselines_figures.sh                # CPU only

Reads artifacts/cgan/baselines/run001/sweep_K*.json (+ examples.pt, the CNN
training report, the test drops) and writes, into the same directory:

    fig1_tradeoff.png       dual-axis BER/SER (left) + P(det) per detector (right)
                            vs received JSR, per attack, K = 1 -- the supervisor's
                            format, with the clean floor and the omniscient ceiling.
                            (Dual axis is the ONE mandated exception to one-axis; the
                            frontier below is its matched-detectability companion.)
    fig2_frontier.png       BER vs P(det) per detector, all attacks, K = 1, with the
                            FAR budget marked -- matched detectability.
    fig3_iq.png             matched-filter constellation under each attack (examples.pt).
    fig4_detector_view.png  what each detector sees: spectrogram images + kurtosis.
    fig5_ber_vs_jsr.png     BER/SER vs received JSR per attack (K = 1) + closed-form noise.
    fig6_pdet_vs_jsr.png    P(det) per detector x attack, FAR line, LRT ceiling on noise.
    fig7_njammers.png       K = 1..4: BER vs total budget (median + IQR) with the
                            best-single-of-K reference; BER at matched detectability vs K.
    fig8_scene.png          the test drops in 3-D; received-JSR spread at a fixed budget.
    fig9_cnn_repro.png      CNN training/val curves + ROC vs Li et al.; accuracy at +10 dB.
    summary.json            the headline numbers behind every figure.

Colours follow the validated categorical palette (dataviz skill): fixed hue order,
never cycled; text in ink tokens, identity from the coloured mark + legend.
"""

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import scene

RUN = "run001"
OUT = os.path.join(scene.ART, RUN)

# validated categorical palette (light), fixed order; ink tokens for all text
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
ATTACK_ORDER = ["noise", "pulsed_p1", "pulsed_p0.5", "pulsed_p0.25", "pulsed_p0.1",
                "omniscient_e0.1", "omniscient_e1"]
ATTACK_LABEL = {"noise": "Noise (barrage)", "pulsed_p1": "Matched QPSK (p=1)",
                "pulsed_p0.5": "Pulsed p=0.5", "pulsed_p0.25": "Pulsed p=0.25",
                "pulsed_p0.1": "Pulsed p=0.1", "omniscient_e0.1": "Omniscient push (η=0.1)",
                "omniscient_e1": "Omniscient flip (η=1)"}
ACOLOR = {a: PAL[i] for i, a in enumerate(ATTACK_ORDER)}
DET_ORDER = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn", "lrt_noise"]
DET_LABEL = {"power_one_sided": "Power (1-sided)", "power_two_sided": "Power (2-sided)",
             "kurtosis": "Kurtosis", "spec_cnn": "Spectrogram CNN", "lrt_noise": "Optimal (noise)"}
DCOLOR = {d: PAL[i] for i, d in enumerate(DET_ORDER)}
ALPHA = 0.05
SA = str(ALPHA)


def _style():
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "legend.frameon": False})


def _clean(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8, solid_capstyle="round")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(k):
    p = os.path.join(OUT, f"sweep_K{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pdet_curve(points, det, a=SA):
    return np.array([p["pdet"].get(det, {}).get(a, np.nan) for p in points])


def ber_curve(points):
    return np.array([p["ber"] for p in points])


# ---------------------------------------------------------------- fig 1: dual axis
def fig_tradeoff(k1):
    jsr = np.array(k1["jsr_db"])
    attacks_plot = ["noise", "pulsed_p1", "pulsed_p0.1", "omniscient_e0.1"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    floor = k1["clean"]["ber"]
    for ax, name in zip(axes.flat, attacks_plot):
        pts = k1["attacks"][name]
        b, s = ber_curve(pts), np.array([p["ser"] for p in pts])
        ax.plot(jsr, np.clip(b, 1e-6, 1), color=INK, lw=2, label="BER")
        ax.plot(jsr, np.clip(s, 1e-6, 1), color=INK2, lw=1.5, ls=(0, (4, 2)), label="SER")
        ax.axhline(max(floor, 1e-6), color=INK2, lw=0.8, ls=":", label="clean BER floor")
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2); ax.set_ylabel("BER / SER")
        ax.set_title(ATTACK_LABEL[name], loc="left")
        _clean(ax)
        axr = ax.twinx()
        for det in DET_ORDER:
            pc = pdet_curve(pts, det)
            if np.isnan(pc).all():
                continue
            axr.plot(jsr, pc, color=DCOLOR[det], lw=1.6, marker="o", ms=3, label=DET_LABEL[det])
        axr.axhline(ALPHA, color=INK2, lw=0.8, ls="--")
        axr.set_ylim(-0.02, 1.02); axr.set_ylabel("P(detect)")
        for sp in ("top",):
            axr.spines[sp].set_visible(False)
    axes.flat[-1].set_xlabel("Received JSR (dB)"); axes.flat[-2].set_xlabel("Received JSR (dB)")
    lh = [plt.Line2D([], [], color=INK, lw=2), plt.Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2))),
          plt.Line2D([], [], color=INK2, lw=0.8, ls=":")]
    fig.legend(lh, ["BER", "SER", "clean BER floor"], loc="upper left", bbox_to_anchor=(0.07, 1.07),
               ncol=3, fontsize=8)
    dh = [plt.Line2D([], [], color=DCOLOR[d], marker="o", lw=1.6, ms=4) for d in DET_ORDER]
    fig.legend(dh, [DET_LABEL[d] for d in DET_ORDER], loc="upper right", bbox_to_anchor=(0.97, 1.07),
               ncol=3, fontsize=8, title="P(detect), α=0.05 dashed")
    fig.suptitle("Effectiveness vs detectability, single jammer (K=1)", x=0.5, y=1.14,
                 fontsize=13, ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_tradeoff.png"), dpi=140,
                                    bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 2: frontier
def fig_frontier(k1):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.axvspan(-0.02, ALPHA, color=PAL[2], alpha=0.06)
    for name in ATTACK_ORDER:
        pts = k1["attacks"][name]
        # strongest DEPLOYED detector = max P(det) over {power 1/2-sided, kurtosis, CNN}
        deployed = np.vstack([pdet_curve(pts, d) for d in
                              ("power_one_sided", "power_two_sided", "kurtosis", "spec_cnn")])
        pmax = np.nanmax(deployed, axis=0)
        b = np.clip(ber_curve(pts), 1e-6, 1)
        genie = name.startswith("omniscient")
        ax.scatter(pmax, b, s=26 if genie else 16, color=ACOLOR[name],
                   marker="D" if genie else "o", alpha=0.9 if genie else 0.6,
                   edgecolors=SURFACE, linewidths=0.5, label=ATTACK_LABEL[name], zorder=3 if genie else 2)
    ax.axvline(ALPHA, color=INK2, lw=1.0, ls="--")
    ax.text(ALPHA + 0.01, 3e-1, "← stealth region\n   (P(det) ≤ FAR budget α=0.05)", fontsize=8,
            color=INK2, va="center")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1.4); ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("P(detect) by the strongest deployed detector (power 1/2-sided · kurtosis · CNN)")
    ax.set_ylabel("BER"); _clean(ax)
    ax.set_title("Only the genie reaches the stealth region — every realisable attack is at the floor",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_frontier.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 3: IQ clouds
def fig_iq(ex):
    keys = [f"clean@-10", "noise@-10", "pulsed_p1@-10", "pulsed_p0.1@-10", "omniscient_e0.1@+0",
            "omniscient_e1@+0"]
    keys = [k for k in keys if k in ex]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.2))
    for ax, key in zip(np.atleast_1d(axes), keys):
        z = ex[key]["z"].numpy()
        ax.scatter(z.real, z.imag, s=3, color=ACOLOR.get(key.split("@")[0], INK2), alpha=0.3,
                   linewidths=0)
        pts = np.array([1, 1, -1, -1]) / math.sqrt(2), np.array([1, -1, 1, -1]) / math.sqrt(2)
        ax.scatter(*pts, s=45, marker="+", color=INK, linewidths=1.3)
        ax.set_title(f"{key}\nBER {ex[key]['ber']:.2g}", loc="left", fontsize=9)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); _clean(ax)
    fig.suptitle("Matched-filter constellation under attack (JSR as labelled)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_iq.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 4: detector view
def fig_detector_view(ex):
    keys = [k for k in ["clean@-10", "noise@-10", "pulsed_p1@-10", "pulse_comb@-10",
                        "omniscient_e1@+0"] if k in ex]
    keys = keys or [k for k in ex if k.endswith("@-10")][:5]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6))
    for j, key in enumerate(keys):
        db = ex[key]["spec_db"][0].numpy()
        axes[0, j].imshow(db.T, aspect="auto", origin="lower", cmap="viridis")
        axes[0, j].set_title(key, loc="left", fontsize=9)
        axes[0, j].set_xlabel("time"); axes[0, j].set_ylabel("freq bin")
        k = ex[key]["kurtosis"].numpy()
        axes[1, j].hist(k, bins=24, color=ACOLOR.get(key.split("@")[0], INK2), alpha=0.85)
        axes[1, j].axvline(2.0, color=INK2, lw=0.8, ls="--")
        axes[1, j].set_title(f"kurtosis μ={k.mean():.2f}", loc="left", fontsize=9)
        axes[1, j].set_xlabel("|r|⁴/|r|²²"); _clean(axes[1, j])
    fig.suptitle("What the detectors see: spectrogram (top), kurtosis distribution (bottom)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_detector_view.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 5/6: vs JSR
def fig_ber_vs_jsr(k1, link_noise_ref):
    jsr = np.array(k1["jsr_db"])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name in ATTACK_ORDER:
        b = np.clip(ber_curve(k1["attacks"][name]), 1e-6, 1)
        ax.plot(jsr, b, color=ACOLOR[name], lw=2, label=ATTACK_LABEL[name])
    if link_noise_ref is not None:
        ax.plot(jsr, np.clip(link_noise_ref, 1e-6, 1), color=INK, lw=1.0, ls=(0, (1, 1)),
                label="Noise closed form")
    ax.axhline(max(k1["clean"]["ber"], 1e-6), color=INK2, lw=0.8, ls=":", label="clean BER floor")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2)
    ax.set_xlabel("Received JSR (dB)"); ax.set_ylabel("BER"); _clean(ax)
    ax.set_title("BER vs received JSR, single jammer", loc="left")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_ber_vs_jsr.png"), dpi=140); plt.close(fig)


def fig_pdet_vs_jsr(k1):
    jsr = np.array(k1["jsr_db"])
    show = ["noise", "pulsed_p1", "pulsed_p0.1"]
    fig, axes = plt.subplots(1, len(show), figsize=(5.0 * len(show), 5), sharey=True)
    for ax, name in zip(axes, show):
        pts = k1["attacks"][name]
        for det in DET_ORDER:
            pc = pdet_curve(pts, det)
            if np.isnan(pc).all():
                continue
            ax.plot(jsr, pc, color=DCOLOR[det], lw=1.8, marker="o", ms=3, label=DET_LABEL[det])
        ax.axhline(ALPHA, color=INK2, lw=0.8, ls="--")
        ax.set_title(ATTACK_LABEL[name], loc="left"); ax.set_xlabel("Received JSR (dB)")
        ax.set_ylim(-0.02, 1.02); _clean(ax)
    axes[0].set_ylabel("P(detect)")
    axes[-1].legend(fontsize=8, loc="center right")
    fig.suptitle("Detectability vs received JSR (α=0.05 dashed; optimal test on noise only)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_pdet_vs_jsr.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 7: n jammers
def matched_ber_vs_k(sweeps, det="spec_cnn", which="pulsed_p1"):
    """
    Median confirmed stealthy BER per K at alpha=0.05, for ONE realisable attack
    (the matched-QPSK jammer) so K=1..4 are the same jammer split more ways. K=1
    is its single confirmed pick; K>1 is the median over drops.
    """
    out = {}
    for k, sw in sweeps.items():
        if sw is None or which not in sw["confirmed"]:
            continue
        c = sw["confirmed"][which]
        if k == 1:
            pick = c.get(det, {}).get(SA)
            vals = [pick["confirmed_ber"]] if pick else []
        else:
            vals = [d[det][SA]["confirmed_ber"] for d in c if d.get(det, {}).get(SA)]
        if vals:
            out[k] = (float(np.median(vals)), float(np.percentile(vals, 25)),
                      float(np.percentile(vals, 75)))
    return out


def fig_njammers(sweeps):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.5))
    # left: BER vs total budget, pulsed p=1, median + IQR over drops
    for k in (2, 3, 4):
        sw = sweeps.get(k)
        if sw is None:
            continue
        per_drop = sw["attacks"].get("pulsed_p1")
        if not per_drop:
            continue
        budget = np.array(sw["budget_dbm"])
        B = np.array([[p["ber"] for p in drop] for drop in per_drop])
        med, lo, hi = np.median(B, 0), np.percentile(B, 25, 0), np.percentile(B, 75, 0)
        axl.plot(budget, np.clip(med, 1e-6, 1), color=PAL[k], lw=2, label=f"K={k}")
        axl.fill_between(budget, np.clip(lo, 1e-6, 1), np.clip(hi, 1e-6, 1), color=PAL[k], alpha=0.12)
    axl.set_yscale("log"); axl.set_ylim(1e-6, 1.2)
    axl.set_xlabel("Total jammer budget (dBm)"); axl.set_ylabel("BER (median, IQR over drops)")
    axl.set_title("Matched QPSK, equal split, K jammers", loc="left"); _clean(axl)
    axl.legend(fontsize=9)
    # right: matched-detectability BER vs K
    mv = matched_ber_vs_k(sweeps)
    if mv:
        ks = sorted(mv)
        med = [mv[k][0] for k in ks]
        yerr = [[mv[k][0] - mv[k][1] for k in ks], [mv[k][2] - mv[k][0] for k in ks]]
        axr.errorbar(ks, med, yerr=np.abs(yerr), color=PAL[0], lw=2, marker="o", ms=7, capsize=4)
    axr.set_xlabel("Number of jammers K (equal total power)")
    axr.set_ylabel("Stealthy BER at α=0.05 (matched QPSK, CNN)"); axr.set_xticks([1, 2, 3, 4])
    axr.set_title("Uncoordinated multi-jammer at matched detectability", loc="left"); _clean(axr)
    fig.suptitle("Does splitting power across jammers help? (equal split, no coordination)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_njammers.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 8: scene
def fig_scene(drops):
    pos = drops["positions"]
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for d in range(min(len(pos), 12)):
        p = np.array(pos[d])
        ax.scatter(*p[0], color=PAL[0], marker="^", s=40)
        ax.scatter(*p[1], color=PAL[1], marker="s", s=40)
        ax.scatter(p[2:, 0], p[2:, 1], p[2:, 2], color=INK2, marker="x", s=25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Test drops (12 shown): ▲ TX  ■ RX  ✕ jammers", loc="left", fontsize=9)
    axr = fig.add_subplot(1, 2, 2)
    g = drops["g"][:, 1:]
    p_t_rx = drops["p_t_rx_w"]
    jsr = 10 * np.log10(scene.dbm_to_w(20.0) / 1 * g[:, 0] / (p_t_rx * 1e3))
    axr.hist(jsr, bins=20, color=PAL[2], alpha=0.85)
    axr.set_xlabel("Received JSR of one jammer at budget +20 dBm (dB)")
    axr.set_ylabel("drops"); axr.set_title("Geometry spread across drops", loc="left"); _clean(axr)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_scene.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- fig 9: CNN reproduction
def fig_cnn_repro():
    p = os.path.join(scene.ART, "spec_cnn_training.json")
    if not os.path.exists(p):
        return None
    rep = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    h = rep["history"]
    ep = [r["epoch"] for r in h]
    axes[0].plot(ep, [r["train_acc"] for r in h], color=PAL[0], lw=2, label="train")
    axes[0].plot(ep, [r["val_acc"] for r in h], color=PAL[1], lw=2, label="val")
    axes[0].axhline(rep["paper"]["two_class_va"] / 100, color=INK2, lw=0.8, ls="--",
                    label=f"Li et al. VA {rep['paper']['two_class_va']}%")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.4, 1.02)
    axes[0].set_title("Training", loc="left"); _clean(axes[0]); axes[0].legend(fontsize=8)
    v = rep["validation"]
    axes[1].plot(v["roc_fpr"], v["roc_tpr"], color=PAL[0], lw=2, label=f"AUC {v['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    axes[1].set_xlabel("false-alarm rate"); axes[1].set_ylabel("detection rate")
    axes[1].set_title("ROC (two-class)", loc="left"); _clean(axes[1]); axes[1].legend(fontsize=8)
    hi = rep["high_jsr_10db"]["per_type"]
    types = list(hi.keys())
    acc = [hi[t]["argmax_correct"] for t in types]
    axes[2].bar(range(len(types)), acc, color=[PAL[i] for i in range(len(types))], width=0.6)
    axes[2].axhline(rep["paper"]["five_class_dr"] / 100, color=INK2, lw=0.8, ls="--")
    axes[2].set_xticks(range(len(types))); axes[2].set_xticklabels(types, rotation=30, ha="right",
                                                                   fontsize=8)
    axes[2].set_ylim(0, 1.05); axes[2].set_ylabel("accuracy at JSR +10 dB")
    axes[2].set_title(f"Reproduction (val acc {v['accuracy']:.3f} vs {rep['paper']['two_class_va']}%)",
                      loc="left"); _clean(axes[2])
    fig.suptitle("Li et al. spectrogram CNN, retrained on this link", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_cnn_repro.png"), dpi=140); plt.close(fig)
    return rep


# ---------------------------------------------------------------- summary
def summarise(sweeps, cnn_rep):
    k1 = sweeps[1]
    jsr = np.array(k1["jsr_db"])
    s = dict(run=RUN, clean=k1["clean"], jsr_at_ber_1e3={}, stealthy_ber={}, matched_ber_vs_k={})
    for name, pts in k1["attacks"].items():
        b = ber_curve(pts)
        cross = next((float(jsr[i]) for i in range(len(b)) if b[i] >= 1e-3), None)
        s["jsr_at_ber_1e3"][name] = cross
        # strongest confirmed stealthy BER per detector at alpha 0.05
        s["stealthy_ber"][name] = {d: (k1["confirmed"][name].get(d, {}).get(SA)) for d in DET_ORDER}
    s["matched_ber_vs_k"] = {str(k): v for k, v in matched_ber_vs_k(sweeps).items()}
    if cnn_rep:
        s["cnn"] = dict(val_accuracy=cnn_rep["validation"]["accuracy"],
                        paper_va=cnn_rep["paper"]["two_class_va"], auc=cnn_rep["validation"]["auc"],
                        high_jsr_accuracy=cnn_rep["high_jsr_10db"]["accuracy"])
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    return s


def main():
    _style()
    import link as lk
    sweeps = {k: load(k) for k in (1, 2, 3, 4)}
    if sweeps[1] is None:
        raise SystemExit("no sweep_K1.json yet -- run baselines.py --k 1 first")
    ex = None
    import torch
    ex_path = os.path.join(OUT, "examples.pt")
    if os.path.exists(ex_path):
        ex = torch.load(ex_path, weights_only=False)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    noise_ref = L.ber_noise_jammer(np.array(sweeps[1]["jsr_db"]), band="full")

    fig_tradeoff(sweeps[1])
    fig_frontier(sweeps[1])
    if ex:
        fig_iq(ex); fig_detector_view(ex)
    fig_ber_vs_jsr(sweeps[1], noise_ref)
    fig_pdet_vs_jsr(sweeps[1])
    fig_njammers(sweeps)
    fig_scene(scene.load_test_drops())
    cnn_rep = fig_cnn_repro()
    s = summarise(sweeps, cnn_rep)
    print("wrote figures + summary.json to", OUT)
    print(json.dumps(s["jsr_at_ber_1e3"], indent=2))


if __name__ == "__main__":
    main()
