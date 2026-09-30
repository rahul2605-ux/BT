"""
Final experiment -- the frozen figure set (README §2.7, §3.4 decision 6) and the
cross-environment comparison. New file; login node (JSON in, PNG/Markdown/JSON out, no
Sionna, no torch).

    python figures.py --env base       # the outputs for one environment
    python figures.py --all            # every environment that has an eval.json
    python figures.py --compare        # across the six environments

The figures draw six attackers per detector column (DRAWN; user 2026-09-30): the genie
flip, the random push, barrage, Amuru at its best p for that detector, the vanilla cGAN
(the damage-only control) and the cGAN conditioned on that detector. Every other attacker
stays in the tables. Columns: one-sided energy, kurtosis, the CNN (the three detectors a
generator is conditioned on); two-sided energy and energy with CSI are in the tables.

Per environment (artifacts/final/<env>/):
    fig1_vs_jsr.png          excess BER (log), excess PER and P_det stacked on one JSR axis
                             per detector, so damage and detection read off the same JSR
    fig2_damage_vs_pdet.png  the same points with JSR hidden: BER (log) and PER (linear)
                             against P_det -- where equal damage is compared (comparing at
                             equal JSR is the matched-config trap)
    table.md, summary.json   P_det at excess PER 0.1 and excess BER 3e-4, mean +- std
                             over seeds, every attacker x detector
Across environments (artifacts/final/compare/):
    fig_compare_pdet.png     P_det at matched damage per drawn attacker, per detector, per env
    fig_compare_gains.png    the CNN-conditioned cGAN's gains against the CNN, over Amuru's
                             best p and over the vanilla cGAN
    fig_dc_share.png         the DC-share diagnostic (1 = the supervisor's constant vector)
    compare.md, compare.json

Conventions. Damage is EXCESS over the same environment's clean frames. "Matched damage"
reads P_det at the first JSR where the excess damage reaches the level, interpolated in
log damage between grid points (cgan/snr_figures._cross_log; a zero left bracket reports
the first point above). The Amuru family is drawn as ONE p per detector: the p with the
lowest P_det at excess PER 0.1 (best_p), i.e. the table's "Amuru, best p" row; the envelope
uses the minimum matched P_det over p.
The on/off baseline is analytic from matched QPSK at +15 dB (E2b, §3.3g): a fraction f of
frames jammed, P_det = f P_det(+15) + (1 - f) FAR, excess damage = f x excess damage(+15),
drawn against its mean JSR 15 + 10 log10 f. The classical envelope is the minimum matched
P_det over barrage, the Amuru family (p = 1 included), on/off, tone and pulse comb.
Seed spreads are std over seeds (ddof 1); a gain's +- is the standard error of the
difference of two seed means. In the figures a multi-seed attacker is its seed mean at each
JSR (thick) over its seeds (thin).
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import env as E

A = "0.05"
PER_REF, BER_REF = 0.1, 3e-4
BER_FLOOR = 3e-6          # below this an excess BER is not a measurement (1 error in 3e5 bits)
DET_LABEL = {"energy": "energy (one-sided)", "energy_2s": "energy (two-sided)", "kurtosis": "kurtosis",
             "spec_cnn": "CNN (Li et al., retrained)", "energy_csi": "energy, gain known (CSI)"}
SEEDED = {"control": 3, "cnn_b10": 3}
ONOFF_F = np.logspace(-3, 0, 61)

# ---------------------------------------------------------------- palette (dataviz reference,
# validated in order: light surface, 8 slots, adjacent CVD dE >= 9.1, normal >= 19.6)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {   # key -> (label, colour, linestyle kwargs)
    "cnn_b10":         ("cGAN vs CNN (β 10)", "#2a78d6", {}),
    "control":         ("cGAN vanilla (β 0, damage only)", "#eb6834", {}),
    "energy_b10":      ("cGAN vs energy (β 10)", "#1baf7a", {}),
    "kurtosis_b10":    ("cGAN vs kurtosis (β 10)", "#eda100", {}),
    "zhou_cgan":       ("Zhou's CGAN (not retrained)", "#e87ba4", {}),
    "pulsed_p0.1":     ("Amuru pulsed p 0.1", "#008300", {}),
    "amuru":           ("Amuru pulsed, envelope over p", "#008300", {}),
    "pulsed_p1":       ("matched QPSK (Amuru p 1)", "#4a3aa7", {}),
    "noise":           ("barrage (white noise)", "#e34948", {}),
    "tone":            ("tone = constant vector", MUTED, dict(marker="s", markevery=6, ms=4)),
    "pulse_comb":      ("pulse comb", MUTED, dict(marker="^", markevery=6, ms=4)),
    "onoff":           ("on/off, matched QPSK at +15 dB", MUTED, dict(marker="D", markevery=6, ms=3.5)),
    "omniscient_e0.1": ("genie push (η 0.1, knows the symbols)", INK2, dict(marker="x", markevery=6, ms=4, lw=1.0)),
    "random_push_e0.1": ("random push (η 0.1)", INK2, dict(ls="--")),
    "omniscient_e1":   ("genie flip (upper bound)", INK, dict(lw=2.4)),
    "envelope":        ("classical envelope (min)", INK2, {}),
}

# ---------------------------------------------------------------- the drawn roster
# Four hues re-validated all-pairs (light surface): CVD dE >= 9.2, normal >= 16.3; aqua is
# below 3:1 contrast, so the legend and the tables carry it. The two genies are ink, told
# apart by weight and dash.
PANEL_DETS = ["energy", "kurtosis", "spec_cnn"]
CONDITIONED = {"energy": "energy_b10", "kurtosis": "kurtosis_b10", "spec_cnn": "cnn_b10"}
DRAWN = {   # role -> (label, colour, line kwargs), in drawing order (last on top)
    "omniscient_e1":    ("genie flip (upper bound: knows every symbol)", INK, dict(lw=2.4)),
    "random_push_e0.1": ("random push (knows timing and phase, not the symbols)", INK2, dict(ls="--")),
    "noise":            ("barrage (white noise)", "#4a3aa7", {}),
    "amuru":            ("Amuru pulsed, best p for this detector", "#1baf7a", {}),
    "control":          ("cGAN vanilla (β 0, damage only)", "#eb6834", {}),
    "conditioned":      ("cGAN conditioned on this detector (β 10)", "#2a78d6", {}),
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def new_fig(nrows, ncols, w, h, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, facecolor=SURFACE, **kw)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


# ---------------------------------------------------------------- reading
def load(env):
    with open(E.art(env, "eval.json")) as f:
        return json.load(f)


N_ATTACKERS = 22        # evaluate.py: 12 classical + Zhou + 8 generators + the random push


def complete(env):
    """eval.json exists and holds every attacker (evaluate.py rewrites it after each one)."""
    if not os.path.exists(E.art(env, "eval.json")):
        return False
    return len(load(env)["attackers"]) >= N_ATTACKERS


def seeds(d, key):
    """The point lists of an attacker key: [points] for single entries, one per seed for GAN tasks."""
    if key in SEEDED:
        return [d["attackers"][f"{key}_r{r}"]["points"] for r in range(SEEDED[key])
                if f"{key}_r{r}" in d["attackers"]]
    tag = f"{key}_r0" if key in ("energy_b10", "kurtosis_b10") else key
    return [d["attackers"][tag]["points"]] if tag in d["attackers"] else []


def pd(p, det):
    return p["pdet"][det][A]


def excess(points, key, clean):
    return np.array([p[key] for p in points], float) - clean[key]


def cross_log(x, y, target):
    """First x where y reaches `target`, interpolated in log y (cgan/snr_figures._cross_log)."""
    lt = math.log10(target)
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if np.isnan(b) or b < target:
            continue
        if np.isnan(a) or a <= 0:
            return float(x[i])
        la, lb = math.log10(a), math.log10(b)
        if la < lt:
            return float(x[i - 1] if lb == la else x[i - 1] + (lt - la) * (x[i] - x[i - 1]) / (lb - la))
    return float(x[0]) if len(y) and y[0] >= target else None


def matched(d, points, det, measure, ref):
    """(P_det, JSR) at the first JSR where the excess `measure` ('per'/'ber') reaches ref."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = excess(points, measure, d["clean"])
    if measure == "ber":
        exc = np.where(exc >= BER_FLOOR, exc, 0.0)
    else:
        exc = np.where(exc > 0, exc, 0.0)
    x = cross_log(jsr, exc, ref)
    if x is None:
        return np.nan, None
    return float(np.interp(x, jsr, [pd(p, det) for p in points])), x


def onoff(d, det):
    """Analytic on/off from matched QPSK at +15 dB: dict of arrays over f."""
    p15 = d["attackers"]["pulsed_p1"]["points"][-1]
    assert d["meta"]["jsr_db"][-1] == 15.0
    c = d["clean"]
    far = pd(c, det)
    return dict(f=ONOFF_F, jsr=15.0 + 10 * np.log10(ONOFF_F), pdet=ONOFF_F * pd(p15, det) + (1 - ONOFF_F) * far,
                ber=ONOFF_F * (p15["ber"] - c["ber"]), ser=ONOFF_F * (p15["ser"] - c["ser"]),
                per=ONOFF_F * (p15["per"] - c["per"]), p15=p15, far=far)


def onoff_matched(d, det, measure, ref):
    o = onoff(d, det)
    exc15 = o["p15"][measure] - d["clean"][measure]
    if exc15 < ref:
        return np.nan, None
    f = ref / exc15
    return float(f * pd(o["p15"], det) + (1 - f) * o["far"]), 15.0 + 10 * math.log10(f)


def amuru_keys(d):
    return [k for k in d["attackers"] if k.startswith("pulsed_p")]


def matched_entry(d, key, det, measure, ref):
    """[P_det per seed] at matched damage for one attacker key (incl. 'amuru', 'onoff', 'envelope')."""
    if key == "onoff":
        return [onoff_matched(d, det, measure, ref)[0]]
    if key == "amuru":
        v = [matched(d, d["attackers"][k]["points"], det, measure, ref)[0] for k in amuru_keys(d)]
        v = [x for x in v if not np.isnan(x)]
        return [min(v) if v else np.nan]
    if key == "envelope":
        v = [matched_entry(d, k, det, measure, ref)[0] for k in ("noise", "amuru", "onoff", "tone", "pulse_comb")]
        v = [x for x in v if not np.isnan(x)]
        return [min(v) if v else np.nan]
    return [matched(d, pts, det, measure, ref)[0] for pts in seeds(d, key)]


def ms(v):
    v = [x for x in v if not np.isnan(x)]
    if not v:
        return np.nan, np.nan, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else np.nan), len(v)


def fmt_ms(v):
    m, s, n = ms(v)
    if n == 0:
        return "—"
    return f"{m:.3f}" + (f" ± {s:.3f}" if n > 1 else "")


# ---------------------------------------------------------------- per environment
def best_p(d, det):
    """The Amuru key with the lowest P_det at excess PER 0.1 against det (the table's 'amuru' row)."""
    v = [(matched(d, d["attackers"][k]["points"], det, "per", PER_REF)[0], k) for k in amuru_keys(d)]
    v = [x for x in v if not np.isnan(x[0])]
    return min(v)[1] if v else "pulsed_p1"


def role_seeds(d, role, det):
    """The point lists a drawn role stands for in detector det's column."""
    if role == "conditioned":
        return seeds(d, CONDITIONED[det])
    if role == "amuru":
        return [d["attackers"][best_p(d, det)]["points"]]
    return seeds(d, role)


def column_title(d, det):
    return f"{DET_LABEL[det]}\nblue = {STYLE[CONDITIONED[det]][0]} · Amuru p = {best_p(d, det)[8:]}"


def draw_role(ax, x, ys, col, lw, kw, label, logy=False):
    """Seed mean (thick) over the seeds (thin); x and ys are [seeds, points]. logy hides y <= 0."""
    x, ys = np.atleast_2d(x), np.atleast_2d(ys)
    f = (lambda y: np.where(y > 0, y, np.nan)) if logy else (lambda y: y)
    if len(ys) > 1:
        for xi, yi in zip(x, ys):
            ax.plot(xi, f(yi), color=col, lw=0.8, alpha=0.4, **kw)
    ax.plot(x.mean(0), f(ys.mean(0)), color=col, lw=lw, label=label, **kw)


def fig_vs_jsr(env, d, out):
    """Excess BER, excess PER and P_det stacked on one JSR axis, one column per detector."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    fig, ax = new_fig(3, len(PANEL_DETS), 3.8 * len(PANEL_DETS), 9.0, sharex=True,
                      gridspec_kw=dict(height_ratios=[1.15, 1, 1]))
    for j, det in enumerate(PANEL_DETS):
        for role, (lab, col, kw) in DRAWN.items():
            S = role_seeds(d, role, det)
            if not S:
                continue
            kw = dict(kw)
            lw = kw.pop("lw", 1.6)
            ber = np.array([excess(p, "ber", d["clean"]) for p in S])
            per = np.array([excess(p, "per", d["clean"]) for p in S])
            pdt = np.array([[pd(q, det) for q in p] for p in S])
            X = np.tile(jsr, (len(S), 1))
            draw_role(ax[0, j], X, ber, col, lw, kw, lab if j == 0 else None, logy=True)
            draw_role(ax[1, j], X, per, col, lw, kw, None)
            draw_role(ax[2, j], X, pdt, col, lw, kw, None)
        ax[0, j].set_yscale("log")
        ax[0, j].set_ylim(1e-6, 1.2)
        ax[0, j].axhline(BER_REF, color=AXIS, lw=0.9)
        ax[1, j].set_ylim(-0.02, 1.02)
        ax[1, j].axhline(PER_REF, color=AXIS, lw=0.9)
        ax[2, j].set_ylim(-0.02, 1.02)
        ax[2, j].axhline(E.ALPHA, color=MUTED, ls=":", lw=1.2)
        ax[2, j].text(0.02, 0.96, f"α {E.ALPHA} dotted · realised FAR {pd(d['clean'], det):.3f}", transform=ax[2, j].transAxes,
                      fontsize=7.5, color=INK2, va="top")
        ax[0, j].set_title(column_title(d, det), fontsize=8.5, loc="left")
        ax[2, j].set_xlabel("received JSR [dB]")
        ax[2, j].set_xlim(-45, 15)
    ax[0, 0].set_ylabel("excess BER (log)")
    ax[1, 0].set_ylabel("excess PER")
    ax[2, 0].set_ylabel("P_det")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"(1) Damage and detection vs JSR — env {env} ({E.LABEL[env]}), SNR 15 dB = Es/N0 24 dB\n"
                 f"Read down a column: at one JSR, how much damage and how often flagged. Thick: seed mean; thin: seeds; "
                 f"grey rules: BER 3·10⁻⁴, PER 0.1", fontsize=9.5, color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.subplots_adjust(top=0.9)                  # tight_layout leaves dead space under a two-line suptitle
    fig.savefig(os.path.join(out, "fig1_vs_jsr.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_damage_vs_pdet(env, d, out):
    """The same points with JSR hidden: damage against P_det, where equal damage is compared."""
    fig, ax = new_fig(2, len(PANEL_DETS), 3.8 * len(PANEL_DETS), 7.4)
    for j, det in enumerate(PANEL_DETS):
        for role, (lab, col, kw) in DRAWN.items():
            S = role_seeds(d, role, det)
            if not S:
                continue
            kw = dict(kw)
            lw = kw.pop("lw", 1.6)
            X = np.array([[pd(q, det) for q in p] for p in S])
            draw_role(ax[0, j], X, np.array([excess(p, "ber", d["clean"]) for p in S]), col, lw, kw,
                      lab if j == 0 else None, logy=True)
            draw_role(ax[1, j], X, np.array([excess(p, "per", d["clean"]) for p in S]), col, lw, kw, None)
        for i in range(2):
            ax[i, j].axvline(E.ALPHA, color=MUTED, ls=":", lw=1.2)
            ax[i, j].set_xlim(-0.02, 1.02)
        ax[1, j].set_xlabel(f"P_det — {DET_LABEL[det]}")
        ax[0, j].set_yscale("log")
        ax[0, j].set_ylim(1e-6, 1.2)
        ax[0, j].axhline(BER_REF, color=AXIS, lw=0.9)
        ax[1, j].set_ylim(-0.02, 1.02)
        ax[1, j].axhline(PER_REF, color=AXIS, lw=0.9)
        ax[0, j].set_title(column_title(d, det), fontsize=8.5, loc="left")
    ax[0, 0].set_ylabel("excess BER (log)")
    ax[1, 0].set_ylabel("excess PER (linear)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"(2) Damage vs P_det, JSR hidden — env {env} ({E.LABEL[env]}). Up and left is better for the attacker\n"
                 f"α dotted; grey rules: BER 3·10⁻⁴, PER 0.1. Thick: seed mean at each JSR; thin: seeds",
                 fontsize=9.5, color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.065, 1, 0.965))
    fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(out, "fig2_damage_vs_pdet.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


TABLE_KEYS = ["omniscient_e1", "random_push_e0.1", "cnn_b10", "control", "energy_b10", "kurtosis_b10", "zhou_cgan", "envelope", "amuru",
              "pulsed_p1", "pulsed_p0.5", "pulsed_p0.25", "pulsed_p0.1", "pulsed_p0.05", "pulsed_p0.02", "pulsed_p0.01",
              "noise", "tone", "pulse_comb", "onoff", "omniscient_e0.1"]


def label_of(key):
    return STYLE[key][0] if key in STYLE else (f"Amuru pulsed p {key[8:]}" if key.startswith("pulsed_p") else key)


def summary(env, d):
    """Every matched-damage number of one environment, per seed, as a dict."""
    dets = d["meta"]["dets"]
    out = dict(env=env, far={det: pd(d["clean"], det) for det in dets}, clean=d["clean"], dets=dets, rows={})
    for key in TABLE_KEYS:
        if key not in ("envelope", "amuru", "onoff") and not seeds(d, key):
            continue
        row = {}
        for det in dets:
            row[det] = dict(per01=matched_entry(d, key, det, "per", PER_REF),
                            ber3e4=matched_entry(d, key, det, "ber", BER_REF))
        pts = seeds(d, key) if key not in ("envelope", "amuru", "onoff") else []
        row["dc_share"] = [float(np.mean([p["dc_share"] for p in s if "dc_share" in p])) for s in pts] if pts else []
        row["jsr_per01"] = [matched(d, s, dets[0], "per", PER_REF)[1] for s in pts]
        row["jsr_ber3e4"] = [matched(d, s, dets[0], "ber", BER_REF)[1] for s in pts]
        out["rows"][key] = row
    return out


def write_table(env, s, out):
    dets = s["dets"]
    lines = [f"# env {env} ({E.LABEL[env]}): P_det at matched excess damage (alpha {E.ALPHA})", "",
             "Realised FAR on 4096 clean frames: " + ", ".join(f"{det} {s['far'][det]:.3f}" for det in dets), "",
             "P_det at excess PER 0.1 / excess BER 3e-4; mean ± std over seeds (3 for the CNN-targeted GAN and the "
             "control, else 1). '—' = the attacker never reaches that damage on the grid (JSR ≤ +15 dB).", "",
             "| attacker | JSR@PER0.1 | " + " | ".join(DET_LABEL[det] for det in dets) + " |",
             "|---|---|" + "---|" * len(dets)]
    for key, row in s["rows"].items():
        j = [x for x in row["jsr_per01"] if x is not None]
        cells = [f"{fmt_ms(row[det]['per01'])} / {fmt_ms(row[det]['ber3e4'])}" for det in dets]
        lines.append(f"| {label_of(key)} | {np.mean(j):+.1f} |" if j else f"| {label_of(key)} | — |")
        lines[-1] += " " + " | ".join(cells) + " |"
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def per_env(env):
    d = load(env)
    out = E.art(env)
    fig_vs_jsr(env, d, out)
    fig_damage_vs_pdet(env, d, out)
    s = summary(env, d)
    write_table(env, s, out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, default=float)
    print(f"{env}: fig1-2, table.md, summary.json -> {os.path.relpath(out)}")
    return s


# ---------------------------------------------------------------- across environments
CMP_MARK = {"omniscient_e1": "*", "random_push_e0.1": "X", "noise": "s", "amuru": "D", "control": "o", "conditioned": "o"}
GAIN_OVER = ["amuru", "control", "envelope", "zhou_cgan"]      # the first two are drawn, all four tabled


def cmp_row(s, role, det):
    """The summary row a drawn role stands for in detector det's column."""
    return s["rows"].get(CONDITIONED[det] if role == "conditioned" else role)


def gain(s, det, lvl, other):
    """P_det(CNN GAN) - P_det(other) at matched damage: (mean, SE of the difference)."""
    m1, s1, n1 = ms(s["rows"]["cnn_b10"][det][lvl])
    m2, s2, n2 = ms(s["rows"][other][det][lvl]) if other in s["rows"] else (np.nan, np.nan, 0)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    se = math.sqrt((0 if np.isnan(s1) else s1 ** 2 / n1) + (0 if np.isnan(s2) else s2 ** 2 / n2))
    return m1 - m2, se


def compare():
    envs = [e for e in E.ENVS if complete(e)]
    S = {e: summary(e, load(e)) for e in envs}
    out = os.path.join(E.ART_ROOT, "compare")
    os.makedirs(out, exist_ok=True)
    dets = ["energy", "energy_2s", "kurtosis", "spec_cnn", "energy_csi"]
    x = np.arange(len(envs))
    # --- P_det at matched damage, per detector, per env: the drawn roster
    fig, ax = new_fig(2, len(PANEL_DETS), 3.8 * len(PANEL_DETS), 7.2, sharey=True)
    off = np.linspace(-0.3, 0.3, len(DRAWN))
    for j, det in enumerate(PANEL_DETS):
        for i, (lvl, name) in enumerate((("per01", "excess PER 0.1"), ("ber3e4", "excess BER 3e-4"))):
            a = ax[i, j]
            for k, (role, (lab, col, _)) in enumerate(DRAWN.items()):
                mk = CMP_MARK[role]
                m = [ms(r[det][lvl]) if (r := cmp_row(S[e], role, det)) and det in r else (np.nan, np.nan, 0)
                     for e in envs]
                a.errorbar(x + off[k], [v[0] for v in m], yerr=[0 if np.isnan(v[1]) else v[1] for v in m], fmt=mk,
                           color=col, ms=9 if mk == "*" else 6, lw=1.2, capsize=2,
                           mec=SURFACE, mew=1.0, label=lab if (i == 0 and j == 0) else None)
            a.axhline(E.ALPHA, color=MUTED, ls=":", lw=1.2)
            a.set_xticks(x, [e for e in envs], rotation=30, fontsize=7.5)
            a.set_ylim(-0.02, 1.02)
            a.set_title(f"{DET_LABEL[det]} — at {name}" + (f"\ncGAN conditioned = {STYLE[CONDITIONED[det]][0]}"
                                                           if i == 0 else ""), fontsize=8.5, loc="left")
        ax[0, j].set_xticklabels([])
    ax[0, 0].set_ylabel("P_det at excess PER 0.1")
    ax[1, 0].set_ylabel("P_det at excess BER 3e-4")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False,
               labelcolor=INK2)
    fig.suptitle("P_det at matched damage across the six environments (mean ± std over seeds; α dotted). "
                 "Lower = the attacker hides better\nAmuru: the best p of each environment and detector",
                 fontsize=10, color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "fig_compare_pdet.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # --- the CNN-conditioned cGAN's gains against the CNN (negative = less detectable = a gain),
    # over Amuru's best p and over the vanilla cGAN; the envelope and Zhou gains are tabled only
    G = {}
    for e in envs:
        for lvl in ("per01", "ber3e4"):
            for other in GAIN_OVER:
                G.setdefault(e, {}).setdefault(lvl, {})[other] = gain(S[e], "spec_cnn", lvl, other)
    fig, ax = new_fig(1, 2, 11.5, 4.4, sharey=True)
    series = [("amuru", "over Amuru pulsed, best p", DRAWN["amuru"][1], "D"),
              ("control", "over the vanilla cGAN (= what conditioning on the CNN buys)", DRAWN["control"][1], "o")]
    lim = 5.0
    for i, (lvl, name) in enumerate((("per01", "excess PER 0.1"), ("ber3e4", "excess BER 3e-4"))):
        a = ax[0, i]
        for k, (other, lab, col, mk) in enumerate(series):
            g = [G[e][lvl][other] for e in envs]
            y, err = [100 * v[0] for v in g], [100 * (0 if np.isnan(v[1]) else v[1]) for v in g]
            lim = max(lim, *[abs(v) + ee for v, ee in zip(y, err) if not np.isnan(v)])
            a.errorbar(x + (k - 0.5) * 0.2, y, yerr=err, fmt=mk, color=col, ms=6, capsize=2, lw=1.2, mec=SURFACE,
                       mew=1.0, label=lab if i == 0 else None)
        room = [100 * (ms(S[e]["rows"]["omniscient_e1"]["spec_cnn"][lvl])[0] - ms(S[e]["rows"]["amuru"]["spec_cnn"][lvl])[0])
                for e in envs]
        lim = max(lim, *[abs(v) for v in room if not np.isnan(v)])
        a.plot(x + 0.3, room, "*", color=INK, ms=9, mec=SURFACE,
               label="genie flip over Amuru's best p (the most any attacker can gain)" if i == 0 else None)
        a.axhline(0, color=AXIS, lw=1)
        a.set_xticks(x, envs, rotation=20, fontsize=8)
        a.set_title(f"CNN P_det: CNN-conditioned cGAN minus the other, at {name}", fontsize=9, loc="left")
    lim = 2 * math.ceil(1.1 * lim / 2)
    ax[0, 0].set_ylim(-lim, lim)
    ax[0, 0].set_ylabel("Δ P_det [pp]")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               labelcolor=INK2)
    fig.suptitle("The CNN-conditioned cGAN's gains against the CNN, per environment (± SE over seeds); "
                 "negative = the cGAN hides better", fontsize=10, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(os.path.join(out, "fig_compare_gains.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # --- DC share diagnostic
    dkeys = ["tone", "cnn_b10", "control", "energy_b10", "kurtosis_b10", "zhou_cgan", "pulsed_p1", "pulsed_p0.1", "noise",
             "pulse_comb", "omniscient_e1"]
    fig, ax = new_fig(1, 1, 9.5, 4.0)
    a = ax[0, 0]
    for k, key in enumerate(dkeys):
        lab, col, _ = STYLE[key]
        for i, e in enumerate(envs):
            v = S[e]["rows"].get(key, {}).get("dc_share", [])
            a.scatter(np.full(len(v), k) + (i - 2.5) * 0.08, v, s=22, color=col, edgecolor=SURFACE, lw=0.8, zorder=3)
    a.set_yscale("log")
    a.set_ylim(1e-5, 2)
    a.axhline(1.0 / 1024, color=MUTED, ls=":", lw=1)
    a.text(len(dkeys) - 0.6, 1.3 / 1024, "white noise ≈ 1/1024", fontsize=7, color=MUTED, ha="right")
    a.set_xticks(range(len(dkeys)), [STYLE[k][0] for k in dkeys], rotation=30, ha="right", fontsize=7.5)
    a.set_ylabel("DC share |mean j|² / mean |j|²")
    a.set_title("DC-share diagnostic: is a learner just the supervisor's constant vector (tone = 1)? "
                "Dots = seeds × the six environments", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dc_share.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    # --- tables
    lines = ["# Cross-environment comparison", "", "P_det at matched damage: excess PER 0.1 / excess BER 3e-4, "
             "mean ± std over seeds", ""]
    for det in dets[:4]:                     # energy_csi has its own section below
        lines += [f"## {DET_LABEL[det]}", "", "| attacker | " + " | ".join(envs) + " |", "|---|" + "---|" * len(envs)]
        for key in TABLE_KEYS:
            if all(key not in S[e]["rows"] for e in envs):
                continue
            lines.append(f"| {label_of(key)} | " + " | ".join(
                (f"{fmt_ms(S[e]['rows'][key][det]['per01'])} / {fmt_ms(S[e]['rows'][key][det]['ber3e4'])}"
                 if key in S[e]["rows"] else "") for e in envs) + " |")
        lines.append("")
    lines += ["## energy with the gain known (fading environments only)", ""]
    fenv = [e for e in envs if E.faded(e)]
    if fenv:
        lines += ["| attacker | " + " | ".join(fenv) + " |", "|---|" + "---|" * len(fenv)]
        for key in TABLE_KEYS:
            if all(key in S[e]["rows"] for e in fenv):
                lines.append(f"| {label_of(key)} | " + " | ".join(
                    f"{fmt_ms(S[e]['rows'][key]['energy_csi']['per01'])} / {fmt_ms(S[e]['rows'][key]['energy_csi']['ber3e4'])}"
                    for e in fenv) + " |")
    lines += ["", "## The CNN-conditioned cGAN's gains against the CNN (Δ P_det, pp; ± SE)", "",
              "| env | " + " | ".join(f"over {label_of(o) if o != 'amuru' else 'Amuru, best p'} {lv}"
                                     for lv in ("PER0.1", "BER3e-4") for o in GAIN_OVER) + " |",
              "|---|" + "---|" * (2 * len(GAIN_OVER))]
    for e in envs:
        cells = [f"{100 * G[e][lvl][o][0]:+.1f} ± {100 * (0 if np.isnan(G[e][lvl][o][1]) else G[e][lvl][o][1]):.1f}"
                 for lvl in ("per01", "ber3e4") for o in GAIN_OVER]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines += ["", "## DC share (mean over JSR points; per seed)", "", "| attacker | " + " | ".join(envs) + " |",
              "|---|" + "---|" * len(envs)]
    for key in dkeys:
        lines.append(f"| {label_of(key)} | " + " | ".join(
            ", ".join(f"{v:.2g}" for v in S[e]["rows"].get(key, {}).get("dc_share", [])) for e in envs) + " |")
    with open(os.path.join(out, "compare.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "compare.json"), "w") as f:
        json.dump(dict(envs=envs, summaries=S, gains=G), f, indent=1, default=float)
    print(f"compare over {envs} -> {os.path.relpath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(E.ENVS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.env:
        per_env(args.env)
    if args.all:
        for e in E.ENVS:
            if complete(e):
                per_env(e)
            elif os.path.exists(E.art(e, "eval.json")):
                print(f"{e}: eval.json incomplete, skipped")
    if args.compare:
        compare()


if __name__ == "__main__":
    main()
