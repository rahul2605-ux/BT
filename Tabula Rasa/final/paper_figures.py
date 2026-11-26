"""
The SECON 2027 paper's figures and tables, from the final experiment's eval files. New file; login node
(JSON in, PDF/TeX out, no Sionna, no torch). Reuses figures.py's readers, so every number is the one
figures.py / README §3.3q report.

    python paper_figures.py     # -> ../paper_drafts/secon27/figs/{fig_base.pdf, fig_envs.pdf, tab_envs.tex}

Grey-box (README §2.8, §3.3q "CNN seeds 13 and 14"): the generators were trained against CNN seed 11 (the
surrogate), so every CNN number here is against the two DEPLOYED CNNs, seeds 13 and 14 (eval_cnn_s13/14.json,
same frames as eval.json; damage is identical across the three files). A generator seed's P_det is its mean
over the two deployed CNNs; ± is the std over generator seeds. Parametric attackers: mean over the two CNNs.
The white-box value (against seed 11) is tabled as a reference. Energy and kurtosis numbers are eval.json's,
except the frame-timed pulsed jammer (pulsed_sync_k1), which only the deployed-CNN files hold: its energy and
kurtosis numbers are the mean over those two files (same frames, thresholds from their own clean draws).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import env as E
import figures as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_drafts", "secon27", "figs")
DEPLOYED = ("eval_cnn_s13.json", "eval_cnn_s14.json")
DETS = ["energy", "kurtosis", "spec_cnn"]
SYNC = "pulsed_sync_k1"     # pulsed with exactly one pulse per frame: the generator's frame timing (README 3.3q)
DET_TITLE = {"energy": "Energy (after matched filter)", "kurtosis": "Kurtosis", "spec_cnn": "CNN classifier"}
ROLES = {   # role -> (label, colour, kwargs); F.DRAWN's roster and colours, paper labels
    "omniscient_e1":    ("genie flip (bound)", F.INK, dict(lw=1.8)),
    "random_push_e0.1": ("random push", F.INK2, dict(ls="--", lw=1.0)),
    "noise":            ("barrage", "#4a3aa7", {}),
    "amuru":            ("pulsed, best $p$", "#1baf7a", {}),
    "control":          (r"generator, $\beta=0$", "#eb6834", {}),
    "conditioned":      (r"generator vs this detector, $\beta=10$", "#2a78d6", {}),
}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def load(env, fn="eval.json"):
    with open(E.art(env, fn)) as f:
        return json.load(f)


def cnn_pdet(Ds, key):
    """[seeds, points] P_det of spec_cnn for one attacker key, mean over the deployed CNNs."""
    seeds = [F.seeds(d, key) for d in Ds]
    return np.mean([[[F.pd(q, "spec_cnn") for q in pts] for pts in s] for s in seeds], axis=0)


def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1.5)


# ---------------------------------------------------------------- Fig. base: damage and P_det vs JSR
def fig_base():
    d = load("base")
    Ds = [load("base", fn) for fn in DEPLOYED]
    jsr = np.array(d["meta"]["jsr_db"], float)
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.5), sharex=True, sharey=True)
    for j, det in enumerate(DETS):
        src = Ds[0] if det == "spec_cnn" else d          # best p on the deployed CNN (0.02 on both)
        for role, (lab, col, kw) in ROLES.items():
            key = F.CONDITIONED[det] if role == "conditioned" else (F.best_p(src, det) if role == "amuru" else role)
            S = F.seeds(d, key) if key in F.SEEDED or not key.startswith("pulsed") else [d["attackers"][key]["points"]]
            per = np.array([F.excess(p, "per", d["clean"]) for p in S])
            pdt = cnn_pdet(Ds, key) if det == "spec_cnn" else np.array([[F.pd(q, det) for q in p] for p in S])
            kw = dict(kw)
            lw = kw.pop("lw", 1.2)
            for a, y in ((ax[0, j], per), (ax[1, j], pdt)):
                if len(y) > 1:
                    for yi in y:
                        a.plot(jsr, yi, color=col, lw=0.5, alpha=0.35, **kw)
                a.plot(jsr, y.mean(0), color=col, lw=lw, label=lab if j == 0 and a is ax[0, 0] else None, **kw)
        ax[0, j].set_title(DET_TITLE[det] + (f"  (pulsed $p$={F.best_p(src, det)[8:]})"), fontsize=7.5)
        ax[0, j].axhline(F.PER_REF, color="#999999", lw=0.6)
        ax[1, j].axhline(E.ALPHA, color="#999999", ls=":", lw=0.8)
        ax[1, j].set_xlabel("received JSR [dB]")
        for a in ax[:, j]:
            style(a)
            a.set_xlim(-40, 15)
            a.set_ylim(-0.02, 1.02)
    ax[0, 0].set_ylabel("excess frame error rate")
    ax[1, 0].set_ylabel(r"$P_{\mathrm{det}}$")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               fontsize=7, handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.4, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_base.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- matched-damage numbers, six environments
def env_numbers(env, measure="per", ref=F.PER_REF):
    """{det: {key: [P_det per seed]}} at matched damage; spec_cnn grey-box (deployed mean), + 'white' ref."""
    d = load(env)
    Ds = [load(env, fn) for fn in DEPLOYED]
    keys = ["omniscient_e1", "random_push_e0.1", "zhou_cgan", "envelope", "control", "cnn_b10", "energy_b10",
            "kurtosis_b10"]
    out = {}
    for det in ["energy", "energy_csi", "kurtosis"]:
        if det in d["meta"]["dets"]:
            out[det] = {k: F.matched_entry(d, k, det, measure, ref) for k in keys}
    out["spec_cnn"] = {k: list(np.mean([F.matched_entry(D, k, "spec_cnn", measure, ref) for D in Ds], axis=0))
                       for k in keys}
    out["spec_cnn_white"] = {k: F.matched_entry(d, k, "spec_cnn", measure, ref) for k in keys}
    for det in ["energy", "energy_csi", "kurtosis", "spec_cnn"]:      # deployed files only
        if det in out:
            out[det][SYNC] = [float(np.mean([F.matched_entry(D, SYNC, det, measure, ref)[0] for D in Ds]))]
    out["best_p"] = {det: F.best_p(Ds[0] if det == "spec_cnn" else d, det) for det in DETS}
    return out


def fig_envs(N):
    envs = list(E.PAPER_ENVS)
    x = np.arange(len(envs))
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
    marks = [("omniscient_e1", "genie flip (bound)", F.INK, "*"),
             ("random_push_e0.1", "random push", F.INK2, "x"),
             ("envelope", "best parametric", "#1baf7a", "D"),
             (SYNC, "pulsed, frame-timed (1 per frame)", "#1baf7a", "d"),
             ("control", r"generator, $\beta=0$", "#eb6834", "o"),
             ("conditioned", r"generator vs this detector, $\beta=10$", "#2a78d6", "s")]
    off = np.linspace(-0.3, 0.3, len(marks))
    for j, det in enumerate(DETS):
        a = ax[j]
        for i, (key, lab, col, m) in enumerate(marks):
            k = F.CONDITIONED[det] if key == "conditioned" else key
            v = [N[e][det][k] for e in envs]
            mean = np.array([np.nanmean(s) for s in v])
            std = np.array([np.nanstd(s, ddof=1) if len(s) > 1 else 0 for s in v])
            a.errorbar(x + off[i], np.minimum(mean, 0.47), yerr=std, fmt=m, color=col, ms=3.6 if m != "*" else 5.5,
                       lw=0.8, capsize=1.5, label=lab if j == 0 else None,
                       mfc="white" if m == "d" else (col if m != "x" else None))
            for xi, mi in zip(x + off[i], mean):
                if mi > 0.47:
                    a.annotate(f"{mi:.2f}", (xi, 0.47), xytext=(0, 2), textcoords="offset points", fontsize=5.5,
                               ha="center", va="bottom", color=col, rotation=90)
        a.axhline(E.ALPHA, color="#999999", ls=":", lw=0.8)
        a.set_xticks(x)
        a.set_xticklabels(["base", "noise", "noise\n1 dB", "fading", "fading\nK28", "both"], fontsize=6.5)
        a.set_title(DET_TITLE[det], fontsize=7.5)
        style(a)
        a.set_ylim(0, 0.5)
    ax[0].set_ylabel(r"$P_{\mathrm{det}}$ at excess FER 0.1")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_envs.pdf"))
    plt.close(fig)


def cell(v, pm=True):
    v = [x for x in v if not np.isnan(x)]
    if not v:
        return "--"
    s = f"{np.mean(v):.2f}"
    return s + (f"\\,$\\pm$\\,{np.std(v, ddof=1):.2f}" if pm and len(v) > 1 else "")


def tab_envs(N):
    """LaTeX rows: P_det at excess FER 0.1, per detector block, six environments."""
    envs = list(E.PAPER_ENVS)
    rows = [("spec_cnn", "CNN classifier (deployed, grey-box)",
             [("envelope", "best parametric"), (SYNC, "pulsed, frame-timed ($k=1$)"), ("control", r"generator, $\beta=0$"),
              ("cnn_b10", r"generator vs CNN, $\beta=10$"), ("zhou_cgan", "Zhou et al.\\ CGAN"),
              ("random_push_e0.1", "random push"), ("omniscient_e1", "genie flip")]),
            ("spec_cnn_white", "CNN the generators trained against (white-box reference)",
             [("cnn_b10", r"generator vs CNN, $\beta=10$")]),
            ("energy", "Energy (one-sided, after the matched filter)",
             [("envelope", "best parametric"), (SYNC, "pulsed, frame-timed ($k=1$)"), ("control", r"generator, $\beta=0$"),
              ("energy_b10", r"generator vs energy, $\beta=10$"), ("cnn_b10", r"generator vs CNN, $\beta=10$")]),
            ("energy_csi", "Energy, channel gain known (fading environments)",
             [("envelope", "best parametric"), ("control", r"generator, $\beta=0$"),
              ("cnn_b10", r"generator vs CNN, $\beta=10$")]),
            ("kurtosis", "Kurtosis",
             [("envelope", "best parametric"), (SYNC, "pulsed, frame-timed ($k=1$)"), ("control", r"generator, $\beta=0$"),
              ("kurtosis_b10", r"generator vs kurtosis, $\beta=10$"), ("cnn_b10", r"generator vs CNN, $\beta=10$")])]
    L = []
    for det, head, items in rows:
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
        for k, lab in items:
            cells = [cell(N[e][det][k]) if det in N[e] else "--" for e in envs]
            L.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
        L.append("\\addlinespace[2pt]")
    with open(os.path.join(OUT, "tab_envs.tex"), "w") as f:
        f.write("% generated by final/paper_figures.py -- do not edit by hand\n"
                "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
                " & \\emph{base} & \\emph{noise} & \\emph{noise\\,1\\,dB} & \\emph{fading} & \\emph{fading\\,K28} & \\emph{both} \\\\\n"
                "\\midrule\n" + "\n".join(L[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    N = {e: env_numbers(e) for e in E.PAPER_ENVS}
    with open(os.path.join(OUT, "numbers.json"), "w") as f:
        json.dump(N, f, indent=1, default=float)
    fig_base()
    fig_envs(N)
    tab_envs(N)
    print("->", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
