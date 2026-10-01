"""
Final experiment -- the CNN scored the way Li et al. score it (README §3.3q follow-up,
2026-09-30). New file; login node (JSON in, PNG/Markdown/JSON out, no Sionna, no torch).

    python li_figures.py --env base        # reads artifacts/final/<env>/eval_li.json

Li et al. (IEEE Access 2022) score their spectrogram CNN per SAMPLE -- one image, which on
our link is one 128-symbol frame -- with the network's own argmax decision, on a test set of
762 clean : 204 per jammer type, by detection rate DR = correct / all (their eq. 2a),
precision, recall, F-score and FAR = FP / (FP + TN). This script applies that to our CNN
(the environment's, retrained on Li's four types) for Li's four types, the CNN-targeted
cGAN (cnn_b10) and its continuation at Li's damage level (cnn_li), each white-box (trained on
the deployed CNN, dashed) and grey-box (trained on the surrogate cnn_s12, solid; user
2026-09-30: knowing the deployed weights is not realistic), with the genie flip as the
upper-bound reference. Attackers not yet in eval_li.json are skipped. evaluate.py --out eval_li.json records the argmax
decision next to the alpha thresholds.

Two windows for the jammed samples (user, 2026-09-30):
  A  Li's literal window: every jammer over the same JSRs, uniform in dB over the CNN's
     training range [-20, +10] dB. Matched configuration, so one comparison row only.
  B  Li's "effective jamming" (a complete loss of signal) read as matched damage: each
     jammer at the JSR where it breaks 10 / 50 / 90 % of frames (excess PER, interpolated as
     in figures.py). PER 0.9 is Li's damage. The main table.
  C  §2.7's standing read-out: the most excess PER a jammer reaches at any grid JSR while
     the CNN flags at most half of its samples. Needed where detection is not monotonic in
     JSR (cnn_li is invisible inside its training band and caught on both sides of it).
Li's metrics per jammer use Li's class balance with the measured rates: TP = 204 x recall,
FP = 762 x FAR. "Recall" is the detection rate of that jammer's frames (our P_det at the
argmax decision).

Outputs, artifacts/final/<env>/li/:
    fig_li_vs_jsr.png         excess BER (log), excess PER and the CNN's detection rate on one
                              JSR axis; Li's window shaded
    fig_li_damage_vs_dr.png   the same points with JSR hidden: excess PER and BER against the
                              detection rate -- where equal damage is compared
    table.md, li.json
"""

import argparse
import json
import os

import numpy as np

import env as E
import figures as F

DECISIONS = {"argmax": "argmax (Li's decision)", "0.05": "α 0.05 (the paper's operating point)"}
WINDOW_A = (-20.0, 10.0)                 # the CNN's training range, train_cnn.JSR_RANGE_DB
LEVELS = (0.1, 0.5, 0.9)                 # excess PER; 0.9 = Li's complete loss of signal
N_CLEAN, N_JAM = 762, 204                # Li's class balance (their Table 3 / train_cnn.py)
LI_REPORTED = "Li et al., EfficientNet-B0: DR 100 % two-class, 99.79 % five-class, weighted FAR 0.03 % (five-class)"
BUDGET = 0.5                             # §2.7: damage while flagged on at most half of the samples
MULTI = ("cnn_b10", "cnn_li", "cnn_b10_grey", "cnn_li_grey")
LI_FOUR = ["noise", "tone", "pulse_comb", "pulsed_p0.25"]
ROSTER = {   # key -> (label, colour, line kwargs), in drawing order (last on top); hues validated all-pairs
    "omniscient_e1": ("genie flip s → −s (upper bound)", F.INK, dict(lw=2.4)),
    "tone":          ("Li: single tone (constant vector)", F.MUTED, dict(marker="s", markevery=4, ms=4)),
    "pulse_comb":    ("Li: successive pulse", F.MUTED, dict(marker="^", markevery=4, ms=4, ls="--")),
    "noise":         ("Li: barrage", "#4a3aa7", {}),
    "pulsed_p0.25":  ("Li: protocol-aware (Amuru p 0.25)", "#1baf7a", {}),
    "cnn_li":        ("cGAN at Li's damage, white-box", "#eb6834", dict(ls="--", lw=1.1)),
    "cnn_b10":       ("cGAN vs CNN, white-box", "#2a78d6", dict(ls="--", lw=1.1)),
    "cnn_li_grey":   ("cGAN at Li's damage, grey-box", "#eb6834", {}),
    "cnn_b10_grey":  ("cGAN vs CNN, grey-box", "#2a78d6", {}),
}
BOX = "dashed cGAN = white-box (trained on the deployed CNN); solid = grey-box (trained on a surrogate CNN)"
AMURU_LOW = {   # Li_<env>/ only (user, 2026-10-01): the duty cycles that beat cnnGAN1 at every budget (§3.3r).
    # Not in ROSTER: magenta and dark yellow fail the normal-vision floor against cnn_li's orange.
    "pulsed_p0.05":  ("Amuru pulsed p 0.05", "#c98500", dict(marker="o", markevery=4, ms=4)),
    "pulsed_p0.02":  ("Amuru pulsed p 0.02", "#e87ba4", dict(marker="D", markevery=4, ms=4)),
}
CLEAN = ["omniscient_e1", "tone", "pulse_comb", "noise", "pulsed_p0.25", "pulsed_p0.05", "pulsed_p0.02",
         "cnn_b10_grey"]


def clean_mode():
    """The Li_<env>/ roster (user, 2026-09-30): Li's four, the genie flip and cnnGAN1 grey-box, plus
    Amuru p 0.05 / 0.02 (user, 2026-10-01)."""
    global ROSTER, BOX
    ROSTER = {k: {**ROSTER, **AMURU_LOW}[k] for k in CLEAN}
    ROSTER["cnn_b10_grey"] = ("cnnGAN1: cGAN vs CNN, grey-box",) + ROSTER["cnn_b10_grey"][1:]
    BOX = "cnnGAN1 is grey-box: trained on a surrogate CNN, never on the deployed one"


def load(env):
    with open(E.art(env, "eval_li.json")) as f:
        return json.load(f)


def seeds(d, key):
    if key in MULTI:
        return [d["attackers"][f"{key}_r{r}"]["points"] for r in range(3) if f"{key}_r{r}" in d["attackers"]]
    return [d["attackers"][key]["points"]] if key in d["attackers"] else []


def rate(p, k):
    return p["pdet"]["spec_cnn"][k]


def li_metrics(recall, far, n_jam=N_JAM, n_clean=N_CLEAN):
    """Li's eq. 2a-e on a clean : jammed test set of Li's balance, from the measured rates."""
    tp, fp = n_jam * recall, n_clean * far
    tn = n_clean - fp
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    return dict(dr=(tp + tn) / (n_jam + n_clean), precision=prec, recall=recall, f_score=f, far=far)


def window_a(d, pts, k):
    jsr = np.array(d["meta"]["jsr_db"], float)
    m = (jsr >= WINDOW_A[0]) & (jsr <= WINDOW_A[1])
    return (float(np.mean([rate(p, k) for p, w in zip(pts, m) if w])),
            float(np.mean(F.excess(pts, "per", d["clean"])[m])))


def matched(d, pts, k, level):
    """(recall, JSR) at the first JSR where excess PER reaches level (figures.matched, any decision)."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = np.clip(F.excess(pts, "per", d["clean"]), 0.0, None)
    x = F.cross_log(jsr, exc, level)
    if x is None:
        return np.nan, np.nan
    return float(np.interp(x, jsr, [rate(p, k) for p in pts])), x


def damage_under(d, pts, k, budget=None):
    """(max excess PER, its JSR) over the grid points where recall <= budget (§2.7's standing read-out)."""
    budget = BUDGET if budget is None else budget
    jsr = d["meta"]["jsr_db"]
    ok = [(e, j) for e, j, p in zip(F.excess(pts, "per", d["clean"]), jsr, pts) if rate(p, k) <= budget]
    return max(ok) if ok else (0.0, None)


def summary(d):
    out = dict(meta=dict(env=d["meta"]["env"], jsr_window_a=WINDOW_A, levels=LEVELS, n_clean=N_CLEAN, n_jam=N_JAM,
                         li_reported=LI_REPORTED),
               far={k: rate(d["clean"], k) for k in DECISIONS}, attackers={})
    for key in ROSTER:
        S = seeds(d, key)
        if not S:
            continue
        e = dict(label=ROSTER[key][0], n_seeds=len(S))
        for k in DECISIONS:
            far = rate(d["clean"], k)
            a = [window_a(d, pts, k) for pts in S]
            b = {str(l): [matched(d, pts, k, l) for pts in S] for l in LEVELS}
            u = [damage_under(d, pts, k) for pts in S]
            e[k] = dict(
                under_budget=dict(budget=BUDGET, max_excess_per=[x[0] for x in u], jsr=[x[1] for x in u]),
                window_a=dict(recall=[x[0] for x in a], mean_excess_per=[x[1] for x in a],
                              li=[li_metrics(x[0], far) for x in a]),
                matched={l: dict(recall=[x[0] for x in v], jsr=[x[1] for x in v],
                                 li=[li_metrics(x[0], far) if not np.isnan(x[0]) else None for x in v])
                         for l, v in b.items()})
        out["attackers"][key] = e
    for k in DECISIONS:          # Li's own two-class setting: clean vs their four types pooled
        rec = float(np.mean([out["attackers"][t][k]["window_a"]["recall"][0] for t in LI_FOUR]))
        out.setdefault("pooled_li_four", {})[k] = li_metrics(rec, rate(d["clean"], k), n_jam=4 * N_JAM)
    return out


def fmt(v, pct=False, nd=3):
    """mean ± std over seeds; pct in %, one decimal; nd decimals otherwise (1 for a JSR)."""
    v = [x for x in v if x is not None and not np.isnan(x)]
    if not v:
        return "—"
    s, nd = (100.0, 1) if pct else (1.0, nd)
    return f"{s * np.mean(v):.{nd}f}" + (f" ± {s * np.std(v, ddof=1):.{nd}f}" if len(v) > 1 else "")


def table_md(s):
    L = [f"# env {s['meta']['env']}: the CNN scored Li et al.'s way (per sample = one 128-symbol frame)", "",
         f"Clean FAR: argmax {s['far']['argmax']:.4f}, α 0.05 {s['far']['0.05']:.4f}. Reference: {LI_REPORTED}.",
         f"Li metrics on a {N_CLEAN} clean : {N_JAM} jammed test set (Li's balance) from the measured rates. "
         "Mean ± std over seeds where a jammer has three.", ""]
    for k, name in DECISIONS.items():
        L += [f"## Decision: {name}", "",
              f"### A. Li's window: every jammer at the same JSRs, uniform over [{WINDOW_A[0]:+g}, {WINDOW_A[1]:+g}] dB "
              "(matched configuration)", "",
              "| jammer | recall | mean excess PER | DR (eq. 2a) % | precision | F-score |", "|---|---|---|---|---|---|"]
        for key, e in s["attackers"].items():
            a = e[k]["window_a"]
            L.append(f"| {e['label']} | {fmt(a['recall'])} | {fmt(a['mean_excess_per'])} | "
                     f"{fmt([m['dr'] for m in a['li']], pct=True)} | {fmt([m['precision'] for m in a['li']])} | "
                     f"{fmt([m['f_score'] for m in a['li']])} |")
        p = s["pooled_li_four"][k]
        L += [f"| **Li's four pooled** (Li's two-class setting, 762 : 816) | {p['recall']:.3f} | — | "
              f"{100 * p['dr']:.1f} | {p['precision']:.3f} | {p['f_score']:.3f} |", "",
              "### B. Matched damage: each jammer where it breaks 10 / 50 / 90 % of frames (PER 0.9 = Li's damage)", "",
              "| jammer | JSR @ PER 0.1 | recall | JSR @ PER 0.5 | recall | JSR @ PER 0.9 | recall | "
              "at PER 0.9: DR % / precision / F |", "|---|---|---|---|---|---|---|---|"]
        for key, e in s["attackers"].items():
            row = f"| {e['label']} |"
            for l in LEVELS:
                m = e[k]["matched"][str(l)]
                row += f" {fmt(m['jsr'], nd=1)} | {fmt(m['recall'])} |"
            li = [m for m in e[k]["matched"]["0.9"]["li"] if m]
            row += (f" {fmt([m['dr'] for m in li], pct=True)} / {fmt([m['precision'] for m in li])} / "
                    f"{fmt([m['f_score'] for m in li])} |" if li else " — |")
            L.append(row)
        L += ["", f"### C. Most damage while flagged on at most {BUDGET:g} of samples (§2.7 standing read-out; grid points)", "",
              "| jammer | max excess PER | at JSR [dB] |", "|---|---|---|"]
        for key, e in s["attackers"].items():
            u = e[k]["under_budget"]
            L.append(f"| {e['label']} | {fmt(u['max_excess_per'])} | {fmt([x for x in u['jsr'] if x is not None], nd=1)} |")
        L.append("")
    return "\n".join(L)


def fig_vs_jsr(env, d, far, out):
    jsr = np.array(d["meta"]["jsr_db"], float)
    fig, ax = F.new_fig(3, 1, 7.6, 9.4, sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
    ax = ax[:, 0]
    for a in ax:
        a.axvspan(*WINDOW_A, color=F.GRID, alpha=0.45, lw=0)
    for key, (lab, col, kw) in ROSTER.items():
        S = seeds(d, key)
        if not S:
            continue
        kw = dict(kw)
        lw = kw.pop("lw", 1.6)
        X = np.tile(jsr, (len(S), 1))
        F.draw_role(ax[0], X, np.array([F.excess(p, "ber", d["clean"]) for p in S]), col, lw, kw, lab, logy=True)
        F.draw_role(ax[1], X, np.array([F.excess(p, "per", d["clean"]) for p in S]), col, lw, kw, None)
        F.draw_role(ax[2], X, np.array([[rate(q, "argmax") for q in p] for p in S]), col, lw, kw, None)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-6, 1.2)
    ax[1].set_ylim(-0.02, 1.02)
    for l in LEVELS:
        ax[1].axhline(l, color=F.AXIS, lw=0.9)
    ax[1].text(-34.5, 0.915, "PER 0.9 = Li's damage (complete loss of signal)", fontsize=7.5, color=F.INK2, va="bottom")
    ax[2].set_ylim(-0.02, 1.02)
    ax[2].axhline(far, color=F.MUTED, ls=":", lw=1.2)
    ax[2].text(-34.5, 0.8, f"argmax decision\nclean FAR {far:.4f} dotted\nshaded: Li's window\n"
               f"[{WINDOW_A[0]:+g}, {WINDOW_A[1]:+g}] dB (the CNN's\ntraining range)", fontsize=7.5, color=F.INK2, va="top")
    ax[0].set_ylabel("excess BER (log)")
    ax[1].set_ylabel("excess PER")
    ax[2].set_ylabel("CNN detection rate (recall)")
    ax[2].set_xlabel("received JSR [dB]")
    ax[2].set_xlim(-35, 15)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False,
               labelcolor=F.INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"The CNN scored Li et al.'s way — env {env} ({E.LABEL[env]}), SNR 15 dB = Es/N0 24 dB\n"
                 "One sample = one 128-symbol frame. Read down: at one JSR, how much damage and how often flagged.\n"
                 f"Thick: seed mean; thin: seeds.\n{BOX}", fontsize=9.5, color=F.INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.subplots_adjust(top=0.885)
    fig.savefig(os.path.join(out, "fig_li_vs_jsr.png"), dpi=150, facecolor=F.SURFACE)
    F.plt.close(fig)


def fig_damage_vs_dr(env, d, far, out):
    fig, ax = F.new_fig(2, 1, 7.6, 8.2)
    ax = ax[:, 0]
    for key, (lab, col, kw) in ROSTER.items():
        S = seeds(d, key)
        if not S:
            continue
        kw = dict(kw)
        lw = kw.pop("lw", 1.6)
        X = np.array([[rate(q, "argmax") for q in p] for p in S])
        F.draw_role(ax[0], X, np.array([F.excess(p, "per", d["clean"]) for p in S]), col, lw, kw, lab)
        F.draw_role(ax[1], X, np.array([F.excess(p, "ber", d["clean"]) for p in S]), col, lw, kw, None, logy=True)
    for l in LEVELS:
        ax[0].axhline(l, color=F.AXIS, lw=0.9)
    ax[0].set_ylim(-0.02, 1.02)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-6, 1.2)
    for a in ax:
        a.axvline(far, color=F.MUTED, ls=":", lw=1.2)
        a.set_xlim(-0.02, 1.02)
        a.set_xlabel("CNN detection rate (recall), argmax decision")
    ax[0].set_ylabel("excess PER")
    ax[1].set_ylabel("excess BER (log)")
    ax[0].text(0.03, 0.915, "PER 0.9 = Li's damage", fontsize=7.5, color=F.INK2, va="bottom")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False,
               labelcolor=F.INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Damage against detection, JSR hidden — env {env} ({E.LABEL[env]})\n"
                 f"Up and to the left is better for the jammer; clean FAR {far:.4f} dotted\n{BOX}",
                 fontsize=9.5, color=F.INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.subplots_adjust(top=0.9)
    fig.savefig(os.path.join(out, "fig_li_damage_vs_dr.png"), dpi=150, facecolor=F.SURFACE)
    F.plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--clean", action="store_true",
                    help="only the CLEAN roster, graphs only, into artifacts/final/Li_<env>/")
    args = ap.parse_args()
    d = load(args.env)
    if args.clean:
        clean_mode()
        out = os.path.join(E.ART_ROOT, f"Li_{args.env}")
        os.makedirs(out, exist_ok=True)
        far = rate(d["clean"], "argmax")
        fig_vs_jsr(args.env, d, far, out)
        fig_damage_vs_dr(args.env, d, far, out)
        print(f"wrote {os.path.relpath(out)}/: fig_li_vs_jsr.png, fig_li_damage_vs_dr.png")
        return
    out = E.art(args.env, "li")
    os.makedirs(out, exist_ok=True)
    s = summary(d)
    with open(os.path.join(out, "li.json"), "w") as f:
        json.dump(s, f, indent=1)
    md = table_md(s)
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write(md)
    fig_vs_jsr(args.env, d, s["far"]["argmax"], out)
    fig_damage_vs_dr(args.env, d, s["far"]["argmax"], out)
    print(md)
    print(f"\nwrote {os.path.relpath(out)}/: fig_li_vs_jsr.png, fig_li_damage_vs_dr.png, table.md, li.json")


if __name__ == "__main__":
    main()
