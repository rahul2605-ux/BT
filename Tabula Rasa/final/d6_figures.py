"""
Quick D6 (README §3.3s): the CNN before and after one retrain that adds cnnGAN1's frames.
New file; login node (JSON in, PNG/Markdown/JSON out, no Sionna, no torch).

    python d6_figures.py --env base     # reads <env>/eval_li.json (deployed CNN) and eval_d6.json (cnn_d6)

Both files score the SAME frames (evaluate.py re-seeds each attacker by its position, same card
type), so every difference is the defender's. The retrained CNN (train_cnn.py --name cnn_d6
--extra-gens cnn_b10_grey_r0 cnn_b10_grey_r1) saw grey-box cnnGAN1 seeds r0 and r1 as a fifth
jammed type; seed r2 is held out (its burst sits on another symbol, §3.3r).

Per jammer, the CNN's detection rate (Li's recall) at its argmax decision, read two ways
(li_figures.py's): A, Li's window -- every jammer over the same JSRs, uniform in [-20, +10] dB;
B, matched damage -- at the JSR where the jammer breaks half of the frames. The clean row is
1 - FAR. Li's pooled DR (eq. 2a, clean vs Li's four) is the "overall accuracy". Both at the
argmax (Li's decision, whose FAR the retrain moves) and at the calibrated alpha 0.05 (same FAR).

Outputs, artifacts/final/<env>/d6/: fig_d6_accuracy.png, table.md, d6.json
"""

import argparse
import json
import os

import numpy as np

import env as E
import figures as F
import li_figures as LF

BEFORE, AFTER = ("deployed CNN", F.MUTED), ("retrained with cnnGAN1 frames", "#2a78d6")
LEVEL = 0.5                      # panel B: matched excess PER
ROWS = [   # label -> attacker tags (a seed mean when several); None = the clean frames
    ("clean frames (1 − FAR)", None),
    ("Li: barrage", ["noise"]),
    ("Li: single tone", ["tone"]),
    ("Li: successive pulse", ["pulse_comb"]),
    ("Li: protocol-aware (Amuru p 0.25)", ["pulsed_p0.25"]),
    ("Amuru pulsed p 0.05", ["pulsed_p0.05"]),
    ("Amuru pulsed p 0.02", ["pulsed_p0.02"]),
    ("genie flip (upper bound)", ["omniscient_e1"]),
    ("cnnGAN1 grey-box, seen (r0, r1)", ["cnn_b10_grey_r0", "cnn_b10_grey_r1"]),
    ("cnnGAN1 grey-box, held out (r2)", ["cnn_b10_grey_r2"]),
]
DECISIONS = ["argmax", "0.05"]


def load(env, name):
    with open(E.art(env, name)) as f:
        return json.load(f)


def read(d, tags, k):
    """(Li's window recall, matched-damage recall) for one row, seed mean; clean -> 1 - FAR twice."""
    if tags is None:
        v = 1.0 - LF.rate(d["clean"], k)
        return v, v
    a = [LF.window_a(d, d["attackers"][t]["points"], k)[0] for t in tags]
    b = [LF.matched(d, d["attackers"][t]["points"], k, LEVEL)[0] for t in tags]
    return float(np.mean(a)), float(np.nanmean(b))


def pooled(d, k):
    """Li's two-class setting: clean vs Li's four pooled, eq. 2a on 762 : 816."""
    rec = float(np.mean([LF.window_a(d, d["attackers"][t]["points"], k)[0] for t in LF.LI_FOUR]))
    return LF.li_metrics(rec, LF.rate(d["clean"], k), n_jam=4 * LF.N_JAM)


def summary(d0, d1):
    out = dict(level=LEVEL, window_a=LF.WINDOW_A, decisions={})
    for k in DECISIONS:
        rows = {lab: dict(zip(("before", "after"), (read(d, tags, k) for d in (d0, d1)))) for lab, tags in ROWS}
        every = {t: [LF.window_a(d, d["attackers"][t]["points"], k)[0] for d in (d0, d1)]
                 + [LF.matched(d, d["attackers"][t]["points"], k, LEVEL)[0] for d in (d0, d1)]
                 for t in d0["attackers"] if t in d1["attackers"]}
        out["decisions"][k] = dict(far=[LF.rate(d["clean"], k) for d in (d0, d1)],
                                   pooled=[pooled(d, k) for d in (d0, d1)], rows=rows, every=every)
    return out


def table_md(env, s, train):
    L = [f"# env {env}: the CNN before / after one retrain with cnnGAN1's frames (README §3.3s)", "",
         "Same frames for both defenders (evaluate.py, paired by position). Retrained on Li's four + cnnGAN1 "
         "grey-box r0, r1 (204 frames, JSR U[−22, +10] dB); r2 held out.", "",
         "| training (validation split) | deployed `cnn/` | `cnn_d6/` |", "|---|---|---|"]
    for key, lab in (("accuracy", "accuracy"), ("far_argmax", "FAR (argmax)"), ("auc", "AUC")):
        L.append(f"| {lab} | {train[0]['validation'][key]:.3f} | {train[1]['validation'][key]:.3f} |")
    for kind in train[1]["validation"]["per_type_correct"]:
        b = train[0]["validation"]["per_type_correct"].get(kind)
        L.append(f"| correct on {kind} | {'—' if b is None else f'{b:.3f}'} | "
                 f"{train[1]['validation']['per_type_correct'][kind]:.3f} |")
    for k, e in s["decisions"].items():
        p0, p1 = e["pooled"]
        L += ["", f"## Decision: {LF.DECISIONS[k]}", "",
              f"Clean FAR {e['far'][0]:.4f} → {e['far'][1]:.4f}. **Li's pooled DR (clean vs Li's four, eq. 2a): "
              f"{100 * p0['dr']:.1f} % → {100 * p1['dr']:.1f} %** (recall {p0['recall']:.3f} → {p1['recall']:.3f}).", "",
              f"| jammer | Li's window [−20, +10] dB: before | after | Δ | at PER {LEVEL:g}: before | after | Δ |",
              "|---|---|---|---|---|---|---|"]
        for lab, r in e["rows"].items():
            (a0, b0), (a1, b1) = r["before"], r["after"]
            L.append(f"| {lab} | {a0:.3f} | {a1:.3f} | {a1 - a0:+.3f} | {b0:.3f} | {b1:.3f} | {b1 - b0:+.3f} |")
        L += ["", "Every attacker evaluated (one row per tag; recall, window A and matched PER 0.5):", "",
              "| tag | window: before | after | PER 0.5: before | after |", "|---|---|---|---|---|"]
        for t, v in e["every"].items():
            L.append(f"| {t} | " + " | ".join("—" if np.isnan(x) else f"{x:.3f}" for x in v) + " |")
    return "\n".join(L) + "\n"


def fig(env, s, out):
    """Rows: Li's argmax decision, then the calibrated alpha 0.05 (same FAR for both defenders).
    Columns: Li's window, matched damage."""
    labs = [lab for lab, _ in ROWS]
    y = np.arange(len(labs))[::-1].astype(float)
    h = 0.36
    fig, ax = F.new_fig(2, 2, 12.5, 10.4, sharey=True)
    cols = [f"Li's window: each jammer over [{LF.WINDOW_A[0]:+g}, {LF.WINDOW_A[1]:+g}] dB JSR",
            f"matched damage: each jammer where it breaks {LEVEL:.0%} of frames"]
    for r, k in enumerate(DECISIONS):
        e = s["decisions"][k]
        p0, p1 = e["pooled"]
        dec = ("Li's argmax decision (the retrain moves its FAR)" if k == "argmax"
               else f"calibrated α {k}: both CNNs at the same FAR")
        for i, a in enumerate(ax[r]):
            v0 = np.array([e["rows"][l]["before"][i] for l in labs])
            v1 = np.array([e["rows"][l]["after"][i] for l in labs])
            a.barh(y + h / 2, v0, h, color=BEFORE[1], label=BEFORE[0], edgecolor=F.SURFACE, lw=1)
            a.barh(y - h / 2, v1, h, color=AFTER[1], label=AFTER[0], edgecolor=F.SURFACE, lw=1)
            for yy, b0, b1 in zip(y, v0, v1):
                moved = abs(b1 - b0) >= 0.02
                a.text(1.03, yy, f"{b0:.2f} → {b1:.2f}" + (f" ({b1 - b0:+.2f})" if moved else ""), va="center",
                       fontsize=7.5, color=F.INK if moved else F.INK2, weight="bold" if moved else "normal")
            a.axhline(y[0] - 0.5, color=F.AXIS, lw=0.9)
            a.set_xlim(0, 1.0)
            a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            a.grid(False, axis="y")
            note = (f"\noverall: Li's pooled DR (clean vs Li's four, eq. 2a) {100 * p0['dr']:.1f} % → "
                    f"{100 * p1['dr']:.1f} %") if i == 0 else "\n"
            a.set_title(f"{'AB'[i] if r == 0 else 'CD'[i]}. {dec}\n{cols[i]}{note}", fontsize=8.5, loc="left")
            a.set_xlabel("CNN detection rate per frame")
        ax[r][0].set_yticks(y)
        ax[r][0].set_yticklabels(labs, fontsize=8.5)
    fig.legend(*ax[0][0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               labelcolor=F.INK2, bbox_to_anchor=(0.5, 0.0))
    where = env if E.LABEL[env] == env else f"{env} ({E.LABEL[env]})"
    fig.suptitle(f"Retraining the CNN on cnnGAN1 — env {where}, SNR 15 dB = Es/N0 24 dB. Same frames for both "
                 "defenders.\nRetrained = Li's four types + cnnGAN1 grey-box seeds r0, r1 as a fifth jammed type; "
                 "seed r2 held out.", fontsize=9.5, color=F.INK, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(left=0.2, right=0.87, top=0.89, bottom=0.08, wspace=0.5, hspace=0.38)
    fig.savefig(os.path.join(out, "fig_d6_accuracy.png"), dpi=150, facecolor=F.SURFACE)
    F.plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    args = ap.parse_args()
    d0, d1 = load(args.env, "eval_li.json"), load(args.env, "eval_d6.json")
    train = [load(args.env, os.path.join(n, "spec_cnn_training.json")) for n in ("cnn", "cnn_d6")]
    out = E.art(args.env, "d6")
    os.makedirs(out, exist_ok=True)
    s = summary(d0, d1)
    with open(os.path.join(out, "d6.json"), "w") as f:
        json.dump(s, f, indent=1)
    md = table_md(args.env, s, train)
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write(md)
    fig(args.env, s, out)
    print(md)
    print(f"wrote {os.path.relpath(out)}/: fig_d6_accuracy.png, table.md, d6.json")


if __name__ == "__main__":
    main()
