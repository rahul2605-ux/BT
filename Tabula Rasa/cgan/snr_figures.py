"""
E2 noise-ablation figures (README §3.3g). Reads snr_ablation's JSONs only (no
Sionna), so this runs on the login node.

    python snr_figures.py

Eight figures into artifacts/cgan/snr_ablation/run001/:
  fig_headline_gain_vs_snr.png   THE RESULT: the CNN row of the matched-BER table, both
                                 halves (detectability, power), with Amuru for reference
  fig_mechanism.png              why: the four stealth edges and the BER onset on one axis
  fig_regression_30db.png        validity: the 30 dB level against the deployed artifacts
  fig_spectrograms_by_snr.png    what the CNN sees (needs examples.npz, snr_examples.py)
  fig_gap_vs_snr.png         THE HYPOTHESIS TEST: the stealth edge vs BOTH onsets
                             (noise parity and a measured excess-BER target), and the
                             two gaps against the gap = SNR diagonal
  fig_matched_ber_vs_snr.png the §3.3f headline's robustness row: P(det) at matched
                             BER, and the JSR needed to reach it, vs SNR
  fig_far_vs_snr.png         clean FAR (held at alpha by re-calibration) and the hit
                             rate on a fixed jammer -- what noise moves on THIS stack
  fig_frontier_by_snr.png    BER vs P(det) over ALL budgets, one panel per SNR

TWO ONSETS, AND THE GAP DEPENDS ON WHICH
----------------------------------------
"Where the jammer starts to bite" has two defensible readings and they do NOT behave
the same way with SNR. Noise parity (JSR = −SNR, where the jammer's power equals the
noise floor) moves 1 dB per dB by construction, so a gap measured against it tracks
the SNR trivially. A measured excess-BER onset need not move at all: above ~10 dB the
clean floor is negligible, so the jammer must supply essentially the whole variance
the target BER needs, which pins the onset near a fixed JSR. fig_gap draws both, and
`--onset` varies the target, because quoting one number here without its definition is
exactly how a threshold artifact becomes a headline (§2.7).

EXCESS BER, NOT RAW BER
-----------------------
§3.3f quotes raw BER because at SNR 30 dB the clean floor is ~0. Across the grid it
is not: at 0 dB the clean floor is 2.3e-3 on its own, so a raw-BER threshold would
mark every point "effective" before the jammer did anything -- the error §3.3c #2
warns about ("below ~12.5 dB the best stealthy jammer is within x1.2 of the floor,
i.e. it does nothing"). Every effectiveness quantity here is therefore BER minus the
clean floor at the same level, which reduces to §3.3f's numbers at 30 dB.

COLOUR
------
Series are METHODS or DETECTORS -- identity -- in the Okabe-Ito order shared with
gan_figures.py, each with its own marker as well, and both palettes pass
validate_palette (light, --pairs all). SNR is never a colour: it is the x-axis, or a
labelled panel. A ten-step one-hue ramp was tried for the levels and FAILED the
ordinal check (adjacent steps dL 0.047 apart) -- ten levels are not ten readable
shades, so the text label carries the level.
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

AD = "../artifacts/cgan/snr_ablation/run001"
DETS = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn"]
DET_LABEL = {"power_one_sided": "power (1-sided)", "power_two_sided": "power (2-sided)",
             "kurtosis": "kurtosis", "spec_cnn": "spectrogram CNN"}
A = "0.05"
ALPHA = 0.05
# The matched-BER operating point. 3e-4 is where §3.3f quotes its headline, so the
# 30 dB row here is directly comparable to it -- and the gain is strongly dependent
# on this value (§3.3f limit (i): it is gone by BER 3e-3), so it is a CLI arg.
# NOTE gan_figures.py uses 1e-3 for the same constant while §3.3f's prose quotes
# 3e-4; that inconsistency is in the existing code, not introduced here.
BER_REF = 3e-4
BER_ONSET = 1e-4      # "the jammer starts to bite": ~13 errors at 512 frames, above the floor
# 512 frames x 128 symbols x 2 bits = 131 072 bits per sweep point, so anything below
# a few 1e-6 is one error or none. Excess BER under this is not a measurement and is
# not drawn or interpolated through. Same floor as gan_figures.py.
BER_FLOOR = 3e-6
B_MIN = 5e-3          # smallest detection budget fig_frontier draws (see there)

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3de"
# categorical (identity) -- same assignment as gan_figures.py so the two read as one set
# Amuru is ORANGE here, not gan_figures' #7a5195: that purple sits dE 12.6 from the
# CNN-targeted maroon (validate_palette: below the 15 normal-vision floor), and those are
# the two series the CNN story compares. Orange keeps "pulsed QPSK = orange", as for p=1.
C = {"noise": "#0072B2", "optimal": "#E69F00", "amuru": "#E69F00", "genie": "#333333",
     "plain": "#56B4E9", "eff": "#009E73", "g_cnn": "#882255", "g_kurt": "#CC79A7",
     "edge": "#0072B2", "onset": "#D55E00"}
DET_C = {"power_one_sided": "#0072B2", "power_two_sided": "#E69F00",
         "kurtosis": "#CC79A7", "spec_cnn": "#882255"}
# Secondary encoding. Kurtosis-pink vs eff-green is dE 7.6 for deutan viewers -- legal
# only when something besides hue tells them apart, so every method has its own marker.
MARK = {"noise": "o", "optimal": "P", "amuru": "D", "plain": "o", "eff": "^",
        "g_kurt": "v", "g_cnn": "s", "genie": "X"}
DET_MARK = {"power_one_sided": "o", "power_two_sided": "s", "kurtosis": "v", "spec_cnn": "D"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


# ---------------------------------------------------------------- loading
def load():
    """[level] sorted by SNR, noiseless last. Each holds the level's whole sweep."""
    levels = []
    for f in glob.glob(os.path.join(AD, "snr_*.json")):
        if f.endswith("_smoke.json"):
            continue
        d = json.load(open(f))
        d["snr"] = d["meta"]["snr_db"]
        d["clean_ber"] = d["clean"]["ber"]
        levels.append(d)
    levels.sort(key=lambda d: (d["snr"] is None, d["snr"] if d["snr"] is not None else 0))
    return levels


def label_of(d):
    return "noiseless" if d["snr"] is None else f"{d['snr']:g} dB"


def excess(pts, clean_ber):
    """BER above the clean floor, with sub-measurement values zeroed (BER_FLOOR)."""
    e = np.array([p["ber"] for p in pts], float) - clean_ber
    return np.where(e >= BER_FLOOR, e, 0.0)


def series(d, name):
    """(jsr, excess BER, {det: P(det) at alpha}) for one jammer at one level."""
    pts = d["attacks"].get(name) or d["generators"].get(name)
    if not pts:
        return None
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = excess(pts, d["clean_ber"])
    pdet = {det: np.array([p["pdet"].get(det, {}).get(A, np.nan) for p in pts], float)
            for det in DETS}
    return jsr, exc, pdet


def _cross(x, y, target, rising=True):
    """First x where y crosses `target`, linear in x. None if it never does."""
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if np.isnan(a) or np.isnan(b):
            continue
        if (a < target <= b) if rising else (a > target >= b):
            return x[i - 1] if b == a else x[i - 1] + (target - a) * (x[i] - x[i - 1]) / (b - a)
    return None


def _cross_log(x, y, target):
    """
    First x where y crosses `target`, interpolated in log y -- BER spans decades.

    A ZERO (or sub-floor) left bracket cannot be interpolated through, so the
    crossing is reported AT the first grid point above the target. Whether a
    one-error point happens to land just below it is measurement-floor luck, and
    letting that decide silently drops whole levels: at 35 dB the noise row goes
    0 -> 2.1e-4 between adjacent JSRs and the onset vanished from the figure.
    Same convention as gan_figures._at_matched_ber.
    """
    lt = math.log10(target)
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if b is None or np.isnan(b) or b < target:
            continue
        if np.isnan(a) or a <= 0:
            return float(x[i])
        la, lb = math.log10(a), math.log10(b)
        if la < lt:
            return float(x[i - 1] if lb == la else
                         x[i - 1] + (lt - la) * (x[i] - x[i - 1]) / (lb - la))
    return None


def stealth_edge(jsr, pdet, alpha=ALPHA, frames=512):
    """
    Loudest JSR still inside the detector's own false-alarm rate: the LAST grid point
    with P(det) <= alpha + 2 sigma, not the first crossing. At low JSR P(det) sits AT
    the realised FAR, which is alpha by construction, so a first-crossing rule fires
    on the first noisy point instead of on the edge. Same 2-sigma budget rule as
    baselines.confirm, so the two agree.
    """
    tol = 2 * math.sqrt(alpha * (1 - alpha) / max(frames, 1))
    ok = [i for i, p in enumerate(pdet) if not np.isnan(p) and p <= alpha + tol]
    return None if not ok else float(jsr[max(ok)])


def ber_onset(jsr, exc, target=BER_ONSET):
    """Quietest JSR at which the jammer costs `target` excess BER."""
    return _cross_log(jsr, exc, target)


def noise_parity(snr_db):
    """
    The other onset, and it needs no measurement: the JSR at which the jammer's power
    equals the noise floor is exactly -SNR, because JSR is signal-relative and
    N0 = P_s/SNR. This is the onset that moves 1 dB per dB by construction; the
    measured BER onset does not have to, and whether it does is the question.
    """
    return -snr_db


def at_matched_ber(d, name, det, target=BER_REF):
    """(P(det), JSR) where this jammer first reaches `target` EXCESS BER."""
    s = series(d, name)
    if s is None:
        return None, None
    jsr, exc, pdet = s
    x = _cross_log(jsr, exc, target)
    if x is None:
        return None, None
    p = np.interp(x, jsr, pdet[det], left=np.nan, right=np.nan)
    return (None if np.isnan(p) else float(p)), float(x)


# ---------------------------------------------------------------- 1. the hypothesis test
def edges_of(levels, det, onset_target=BER_ONSET, jammer="noise"):
    """(SNR, stealth edge, measured BER onset, noise-parity onset) over the grid."""
    xs, edge, onset = [], [], []
    for d in levels:
        if d["snr"] is None:
            continue
        xs.append(d["snr"])
        s = series(d, jammer)
        if s is None:
            edge.append(np.nan); onset.append(np.nan); continue
        jsr, exc, pdet = s
        e = stealth_edge(jsr, pdet[det], frames=d["meta"].get("n_frames", 512))
        o = ber_onset(jsr, exc, onset_target)
        edge.append(np.nan if e is None else e)
        onset.append(np.nan if o is None else o)
    xs = np.array(xs, float)
    return xs, np.array(edge, float), np.array(onset, float), noise_parity(xs)


def fig_gap(levels, onset_target=BER_ONSET):
    """
    THE HYPOTHESIS TEST, and it needs two onsets, not one -- which is the trap here.

    The stealth edge is set by the fluctuation of the CLEAN statistic, dominated by
    the signal, so it should be roughly SNR-independent in JSR. What "effective"
    means is the part that is not obvious:

      * NOISE PARITY, JSR = -SNR: the jammer's power equals the noise floor. Moves
        1 dB per dB BY CONSTRUCTION, so a gap measured against it tracks the SNR
        trivially and proves nothing.
      * The MEASURED BER onset at a stated excess-BER target: this one need not move
        at all. Above ~10 dB the clean floor is negligible, so the jammer has to
        supply essentially the whole variance the target BER needs, which pins the
        onset near a fixed JSR regardless of SNR.

    Both are drawn. If the two gaps behave differently -- and the arithmetic above
    says they should -- then "the 30 dB gap IS the 30 dB SNR" is true only in the
    noise-parity sense, and the operational gap is SNR-independent. Quoting one
    without the other is how a threshold artifact becomes a headline (§2.7).
    Reference jammer: barrage noise, present at every level.
    """
    fig = plt.figure(figsize=(12.5, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.1], hspace=0.42, wspace=0.3)
    gaps, gaps_np = {}, {}
    xs = None
    for k, det in enumerate(DETS):
        ax = fig.add_subplot(gs[0, k])
        style(ax)
        xs, edge, onset, par = edges_of(levels, det, onset_target)
        gaps[det], gaps_np[det] = onset - edge, par - edge
        ax.fill_between(xs, edge, onset, where=~np.isnan(edge + onset),
                        color=C["edge"], alpha=0.10, lw=0)
        ax.plot(xs, edge, color=C["edge"], lw=2.0, marker="o", ms=4, label="stealth edge")
        ax.plot(xs, onset, color=C["onset"], lw=2.0, marker="s", ms=4,
                label=f"BER onset ({onset_target:g})")
        ax.plot(xs, par, color=INK2, lw=1.2, ls=(0, (4, 3)), label="noise parity (−SNR)")
        ax.set_title(DET_LABEL[det], color=INK, fontsize=10)
        ax.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
        if k == 0:
            ax.set_ylabel("received JSR [dB]", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")

    lo, hi = (float(xs.min()), float(xs.max())) if xs is not None and len(xs) else (0.0, 40.0)
    # both bottom panels plot the same quantity (a gap, in dB), so they share a
    # y-scale -- otherwise the flat curve and the falling one look equally steep.
    vals = np.concatenate([v for g in (gaps, gaps_np) for v in g.values()] + [np.array([lo, hi])])
    vals = vals[np.isfinite(vals)]
    pad = 0.08 * (vals.max() - vals.min()) if len(vals) else 1.0
    for col, (g, name) in enumerate([(gaps, f"measured BER onset ({onset_target:g})"),
                                     (gaps_np, "noise parity (−SNR)")]):
        ax = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
        style(ax)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls=(0, (4, 3)),
                zorder=1, label="gap = SNR")
        for det in DETS:
            ax.plot(xs, g[det], color=DET_C[det], lw=2.0, marker=DET_MARK[det], ms=5,
                    label=DET_LABEL[det], zorder=2)
        ax.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
        ax.set_title(f"gap to {name}", color=INK, fontsize=10, loc="left")
        if len(vals):
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
        if col == 0:
            ax.set_ylabel("gap − stealth edge [dB]", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2, loc="lower right")
    fig.suptitle("E2 — the gap between where a jammer is seen and where it bites "
                 "(barrage noise, α = 0.05)", color=INK, fontsize=12, x=0.012, ha="left")
    save(fig, "fig_gap_vs_snr.png")


# ---------------------------------------------------------------- 2. the headline's robustness
def fig_matched(levels, rows, ber_ref=BER_REF):
    """
    §3.3f's headline at one noise level: at matched BER the CNN-targeted generator is
    −19.8 pp less detectable, at 12.8 dB less power. Both halves, swept over SNR.
    """
    xs = [d["snr"] for d in levels if d["snr"] is not None]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), sharex=True)
    handles = {}
    for k, det in enumerate(DETS):
        top, bot = axes[0][k], axes[1][k]
        style(top); style(bot)
        for key, tag, lab in rows:
            p_v, j_v = [], []
            for d in levels:
                if d["snr"] is None:
                    continue
                p, j = at_matched_ber(d, tag, det, ber_ref)
                p_v.append(np.nan if p is None else p)
                j_v.append(np.nan if j is None else j)
            ln, = top.plot(xs, p_v, color=C[key], lw=2.0, marker=MARK[key], ms=5, label=lab)
            bot.plot(xs, j_v, color=C[key], lw=2.0, marker=MARK[key], ms=5, label=lab)
            handles.setdefault(lab, ln)
        top.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
        top.set_ylim(-0.03, 1.03)
        top.set_title(DET_LABEL[det], color=INK, fontsize=10)
        bot.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
        if k == 0:
            top.set_ylabel(f"P(det) at excess BER {ber_ref:g}", color=INK2, fontsize=9)
            bot.set_ylabel("JSR needed [dB]", color=INK2, fontsize=9)
    fig.suptitle("E2 — does the matched-BER stealth gain survive off 30 dB?  "
                 "(lower is stealthier; α dotted)", color=INK, fontsize=12, x=0.012, ha="left")
    if handles:
        fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=9,
                   labelcolor=INK2, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig_matched_ber_vs_snr.png")


# ---------------------------------------------------------------- 3. FAR and discrimination
HIT_JSR = (-40.0, -30.0)     # inside every detector's transition somewhere on the grid


def fig_far(levels):
    """
    What the noise level moves, on THIS stack. §3.3c found noise moved the frozen
    sim08 CNN's FALSE-ALARM rate. Here every detector is re-calibrated per level, so
    the false-alarm rate is alpha BY CONSTRUCTION (left: the calibration check that
    keeps the comparison matched) -- and what moves instead is the HIT RATE on a fixed
    jammer (right): the same barrage noise at the same JSR is caught more often as the
    link gets cleaner. That is §3.3g finding 2 read at a fixed operating point.
    """
    xs = [d["snr"] for d in levels if d["snr"] is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    style(ax1); style(ax2)
    all_far = []
    for det in DETS:
        far, hits = [], {j: [] for j in HIT_JSR}
        for d in levels:
            if d["snr"] is None:
                continue
            far.append(d["clean"]["pdet"].get(det, {}).get(A, np.nan))
            s = series(d, "noise")
            for j in HIT_JSR:
                hits[j].append(np.nan if s is None else
                               float(np.interp(j, s[0], s[2][det], left=np.nan, right=np.nan)))
        all_far += far
        ax1.plot(xs, far, color=DET_C[det], lw=2.0, marker=DET_MARK[det], ms=5,
                 label=DET_LABEL[det])
        ax2.plot(xs, hits[HIT_JSR[1]], color=DET_C[det], lw=2.0, marker=DET_MARK[det], ms=5,
                 label=f"{DET_LABEL[det]}")
        ax2.plot(xs, hits[HIT_JSR[0]], color=DET_C[det], lw=1.3, ls=(0, (3, 2)),
                 marker=DET_MARK[det], ms=3.5, alpha=0.8)
    ax1.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax2.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    fv = np.array([v for v in all_far if np.isfinite(v)], float)
    ax1.set_ylim(0.0, max(2 * ALPHA, float(fv.max()) * 1.4 if len(fv) else ALPHA))
    ax2.set_ylim(-0.03, 1.03)
    ax2.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax1.set_ylabel("clean false-alarm rate", color=INK2, fontsize=9)
    ax1.set_title("false alarms: held at α at every level (re-calibrated)",
                  color=INK, fontsize=10, loc="left")
    ax2.set_ylabel("P(det) of barrage noise", color=INK2, fontsize=9)
    ax2.set_title(f"hit rate at fixed JSR: {HIT_JSR[1]:g} dB solid, {HIT_JSR[0]:g} dB dashed",
                  color=INK, fontsize=10, loc="left")
    for ax in (ax1, ax2):
        ax.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("E2 — what the noise level moves: not the false alarms, the hit rate  (α dotted)",
                 color=INK, fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_far_vs_snr.png")


# ---------------------------------------------------------------- 4. the all-budget frontier
def roc_pdet(clean_q, jam_q, budgets):
    """
    P(det) at every budget, from the two statistics' stored quantiles: the budget
    picks the clean threshold, the jammed CDF reads P(det) off it. This is what lets
    the frontier be drawn over ALL budgets instead of only at alpha (§3.3f caveat).
    """
    cq, jq = np.asarray(clean_q, float), np.asarray(jam_q, float)
    n = len(cq)
    thr = np.interp(1.0 - np.asarray(budgets, float), np.linspace(0, 1, n), cq)
    return np.clip(1.0 - np.searchsorted(jq, thr, side="left") / (n - 1), 0.0, 1.0)


def fig_frontier(levels, rows, det="spec_cnn"):
    """
    BER vs P(det) parametric in JSR, every budget rather than only alpha. Only the
    tags snr_ablation stored statistic quantiles for can be drawn this way; the rest
    would be a single point per curve.

    Solid = the achievable frontier. Dotted horizontal = the same jammer with the
    budget removed ("when loud"). A panel with only the dotted line is the §3.3f
    result restated at that noise level: the jammer works, but not while stealthy.
    """
    # The smallest budget the data can resolve: the stored CDFs have N_Q = 257 points
    # (a 1/256 step in the tail) and the clean side comes from 4 x 512 = 2048 frames, so a
    # budget below ~10/2048 rests on a handful of samples. Drawing down to 1e-3 produced
    # a "frontier at tight budgets only" at 15 dB that was that artifact.
    budgets = np.logspace(np.log10(B_MIN), np.log10(0.5), 40)
    n = len(levels)
    cols = min(5, max(n, 1))
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.0 * cols, 3.0 * rws + 0.6),
                             squeeze=False, sharey=True)
    handles = {}
    for i, d in enumerate(levels):
        ax = axes[i // cols][i % cols]
        style(ax)
        # the clean side of the ROC. Taken from the level's own clean measurement,
        # which snr_ablation always records with keep_stats: the 30 dB level reuses
        # the DEPLOYED thresholds.json, which predates clean_stat_q and has none, and
        # mixing a 20k-frame clean CDF at other levels with a 2k one here would make
        # the panels incomparable. Fall back to the calibration's copy if absent.
        cq = (d["clean"].get("stat_q", {}).get(det)
              or d["meta"].get("thresholds", {}).get("clean_stat_q", {}).get(det))
        empty = []
        for key, tag, lab in rows:
            pts = d["attacks"].get(tag) or d["generators"].get(tag)
            if not pts or cq is None or "stat_q" not in pts[0]:
                continue
            exc = excess(pts, d["clean_ber"])
            # the achievable frontier: at each budget, the most damage any JSR on the
            # grid does while its P(det) still fits inside that budget.
            best = np.full(len(budgets), -np.inf)
            for p, e in zip(pts, exc):
                q = p["stat_q"].get(det)
                if q is None:
                    continue
                best = np.fmax(best, np.where(roc_pdet(cq, q, budgets) <= budgets, e, -np.inf))
            best = np.where(best >= BER_FLOOR, best, np.nan)
            ln, = ax.plot(budgets, best, color=C[key], lw=2.0, label=lab,
                          marker=MARK[key], ms=3.5, markevery=6)
            handles.setdefault(lab, ln)
            # what the same jammer does with the budget removed -- §3.3f quotes the
            # "max BER when loud" beside the stealthy zero, and without it a panel
            # where nothing is stealthy is indistinguishable from a missing file.
            loud = float(np.nanmax(exc)) if np.isfinite(exc).any() else 0.0
            if loud >= BER_FLOOR:
                ax.axhline(loud, color=C[key], lw=1.0, ls=(0, (1, 2)), alpha=0.8)
            empty.append(not np.isfinite(best).any())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(budgets[0], budgets[-1])
        ax.set_ylim(BER_FLOOR, 1.0)
        ax.axvline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
        if empty and all(empty):
            ax.text(0.5, 0.5, "no stealthy point\nabove the measurement floor",
                    transform=ax.transAxes, ha="center", va="center",
                    color=INK2, fontsize=8.5, style="italic")
        ax.set_title(label_of(d), color=INK, fontsize=10)
        if d["snr"] is None and det == "spec_cnn":
            # the CNN is blind here (its edge reads +15 dB): whatever sits in this panel
            # is the out-of-distribution detector, not the jammer. Say so on the figure,
            # which may travel without the report around it.
            ax.text(0.5, 0.12, "CNN out of distribution —\nnot a result",
                    transform=ax.transAxes, ha="center", va="bottom", color=INK2,
                    fontsize=8.5, style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
        ax.set_xlabel("detection budget", color=INK2, fontsize=8)
        if i % cols == 0:
            ax.set_ylabel("best excess BER", color=INK2, fontsize=8)
    for j in range(n, rws * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"E2 — achievable excess BER at every detection budget, "
                 f"{DET_LABEL[det]} (α dotted)", color=INK, fontsize=12, x=0.012, ha="left")
    if handles:
        fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=9,
                   labelcolor=INK2, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "fig_frontier_by_snr.png")


# ---------------------------------------------------------------- 5. the headline, alone
def _curve(levels, tag, det, ber_ref):
    xs, p, j = [], [], []
    for d in levels:
        if d["snr"] is None:
            continue
        pv, jv = at_matched_ber(d, tag, det, ber_ref)
        xs.append(d["snr"]); p.append(np.nan if pv is None else pv); j.append(np.nan if jv is None else jv)
    return np.array(xs), np.array(p), np.array(j)


def fig_headline(levels, ber_ref=BER_REF):
    """
    §3.3f's headline, both halves, swept over SNR: at matched BER the CNN-targeted
    generator is less detectable than the plain GAN (left) at ~12.5 dB less power
    (right). The left panel carries the finding: the gain peaks at 15 dB, and 30 dB --
    where the D-series measured it, because Zhou's protocol fixes it -- sits near the
    weak end. Amuru pulsed is drawn as the classical reference because §3.3f limit (ii)
    ("a classical jammer still evades the CNN better") turns out to hold only above
    ~22 dB.
    """
    rows = [("plain", "plain_run001", "plain GAN — D1"),
            ("g_cnn", "spec_cnn_b10", "GAN trained against the CNN, β=10 — D2"),
            ("amuru", "pulsed_p0.1", "Amuru pulsed (p=0.1), classical")]
    cur = {k: _curve(levels, t, "spec_cnn", ber_ref) for k, t, _ in rows}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    style(ax1); style(ax2)
    x, p1, _ = cur["plain"]; _, p2, _ = cur["g_cnn"]
    ok = ~np.isnan(p1) & ~np.isnan(p2)
    ax1.fill_between(x[ok], p2[ok], p1[ok], color=C["g_cnn"], alpha=0.10, lw=0)
    for key, tag, lab in rows:
        xs, p, j = cur[key]
        ax1.plot(xs, p, color=C[key], lw=2.6 if key != "amuru" else 1.6, marker=MARK[key], ms=6,
                 label=lab, zorder=3, ls="-" if key != "amuru" else (0, (4, 2)))
        ax2.plot(xs, j, color=C[key], lw=2.6 if key != "amuru" else 1.6, marker=MARK[key], ms=6,
                 label=lab, zorder=3, ls="-" if key != "amuru" else (0, (4, 2)))
    ax1.axhline(ALPHA, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax1.text(40.5, ALPHA, "α", color=INK2, fontsize=9, va="center")

    def note(ax, snr, y, text, dy, ha="center"):
        i = int(np.argmin(np.abs(x - snr)))
        ax.annotate(text, xy=(x[i], y[i]), xytext=(x[i], y[i] + dy), ha=ha, fontsize=8.8,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    i15, i30 = int(np.argmin(np.abs(x - 15))), int(np.argmin(np.abs(x - 30)))
    mid15 = (p1[i15] + p2[i15]) / 2
    ax1.annotate(f"peak gain {100 * (p2[i15] - p1[i15]):+.1f} pp\nat 15 dB",
                 xy=(15, mid15), xytext=(0.6, 0.80), fontsize=9, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax1.annotate(f"{100 * (p2[i30] - p1[i30]):+.1f} pp at 30 dB —\nwhere the D-series measured it",
                 xy=(30, p2[i30]), xytext=(20.5, 0.60), fontsize=9, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax1.axvline(30, color=GRID, lw=6, zorder=0)
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax1.set_ylabel(f"P(det) of the spectrogram CNN at matched excess BER {ber_ref:g}",
                   color=INK2, fontsize=9)
    ax1.set_title("detectability at matched damage (lower is stealthier)",
                  color=INK, fontsize=10.5, loc="left")

    _, _, j1 = cur["plain"]; _, _, j2 = cur["g_cnn"]
    sav = j1 - j2
    sv = sav[(x >= 10) & ~np.isnan(sav)]
    ax2.fill_between(x, j2, j1, where=~np.isnan(sav) & (x >= 5), color=C["g_cnn"], alpha=0.10, lw=0)
    if len(sv):
        # below every curve, where the panel is empty; the arrow lands inside the band
        # between D1 and the Amuru line, so the label never sits on data
        ax2.annotate(f"{sv.min():.1f}–{sv.max():.1f} dB less power\nat every SNR ≥ 10 dB",
                     xy=(30, -11.5), xytext=(28, -31), ha="center", va="center", fontsize=9,
                     color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID),
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax2.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax2.set_ylabel("received JSR needed for that damage [dB]", color=INK2, fontsize=9)
    ax2.set_title("the power it takes", color=INK, fontsize=10.5, loc="left")
    fig.suptitle("E2 — the stealth gain of detector-aware training, over the whole SNR range",
                 color=INK, fontsize=12, x=0.012, ha="left")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save(fig, "fig_headline_gain_vs_snr.png")


# ---------------------------------------------------------------- 6. the mechanism, in one panel
def fig_mechanism(levels, onset_target=BER_ONSET):
    """
    Why the gain moves with SNR, on one axis instead of fig_gap's four. Three of the
    four stealth edges FALL as the link gets cleaner -- the clean statistic's variance
    is set by N0, so a cleaner link buys the defender precision -- while the measured
    BER onset stays put. Kurtosis, a shape statistic, is the one that does not sharpen.
    """
    fig, ax = plt.subplots(figsize=(11, 6.6))
    style(ax)
    xs = onset = None
    for det in DETS:
        xs, edge, onset, par = edges_of(levels, det, onset_target)
        ax.plot(xs, edge, color=DET_C[det], lw=2.2, marker=DET_MARK[det], ms=6,
                label=f"stealth edge — {DET_LABEL[det]}", zorder=3)
        if det == "spec_cnn":
            ok = ~np.isnan(edge) & ~np.isnan(onset) & (onset > edge)
            ax.fill_between(xs, edge, onset, where=ok, color=DET_C[det], alpha=0.08, lw=0)
    ax.plot(xs, onset, color=INK, lw=3.0, marker="o", ms=5, zorder=4,
            label=f"BER onset — barrage costs {onset_target:g} excess BER")
    ax.plot(xs, noise_parity(xs), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2,
            label="noise parity (JSR = −SNR)")
    ax.set_xlabel("SNR [dB]", color=INK2, fontsize=9)
    ax.set_ylabel("received JSR [dB]", color=INK2, fontsize=9)
    ax.set_ylim(-52, 4)
    # every label sits in a region the layout audit found empty of data
    ax.annotate("power and CNN edges fall ≈ 0.7 dB per dB: the detector sharpens",
                xy=(27, -37), xytext=(3, -42.5), fontsize=9, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.annotate("BER onset: flat from 10 dB up", xy=(24, -2.6), xytext=(24, -9.5),
                fontsize=9, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.annotate("kurtosis (a shape statistic) does not sharpen", xy=(26, -20), xytext=(22.5, -15.5),
                fontsize=9, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.text(3, -49.5, "at 0 dB the jammer bites before it is seen — but the clean link already errs at 2.3e-3",
            fontsize=8.3, color=INK2, ha="left", va="bottom", style="italic")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.11))
    fig.suptitle("E2 — why the gain moves: the defender improves with SNR, the damage threshold does not",
                 color=INK, fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_mechanism.png")


# ---------------------------------------------------------------- 7. the regression gate, drawn
def fig_regression():
    """
    The 30 dB level re-measured against the deployed §3.3d/§3.3f artifacts. Left: every
    P(det) pair (28 rows x 66 JSRs x 4 detectors) on the identity line, with the 4-sigma
    binomial band. Right: the BER deviations in band-units -- a regression is one-signed,
    Monte-Carlo scatter is not.
    """
    import regress_snr30 as R
    if not os.path.exists(R.AB):
        print("  (no 30 dB level yet -- fig_regression skipped)")
        return
    rows, pd, dev = R.collect()
    tol = rows[0][3]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1, 1.1]))
    style(ax1); style(ax2)
    g = np.linspace(0, 1, 50)
    ax1.fill_between(g, g - tol, g + tol, color=GRID, alpha=0.9, lw=0, label=f"±4σ ({tol:.3f})")
    ax1.plot(g, g, color=INK2, lw=1.0)
    for det in DETS:
        o = [x for d, x, _ in pd if d == det]; n = [y for d, _, y in pd if d == det]
        ax1.scatter(o, n, s=9, color=DET_C[det], marker=DET_MARK[det], alpha=0.35,
                    lw=0, label=DET_LABEL[det])
    out = sum(abs(n - o) > tol for _, o, n in pd)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02); ax1.set_aspect("equal")
    ax1.set_xlabel("P(det), deployed (§3.3d / §3.3f)", color=INK2, fontsize=9)
    ax1.set_ylabel("P(det), E2 at 30 dB (fresh seeds)", color=INK2, fontsize=9)
    ax1.set_title(f"{len(pd)} detection rates — {out} outside 4σ", color=INK, fontsize=10.5, loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", markerscale=1.8)

    dev = np.array(dev)
    bins = np.linspace(-1.75, 1.75, 29)
    ax2.hist(np.clip(dev, -1.75, 1.75), bins=bins, color=C["noise"], alpha=0.85, edgecolor="white", lw=0.8)
    for v in (-1, 1):
        ax2.axvline(v, color=INK2, lw=1.0, ls=(0, (3, 2)))
    ax2.axvline(dev.mean(), color=INK, lw=2.0)
    pos = int((dev > 0).sum())
    ax2.text(0.98, 0.95, f"{pos} above / {len(dev) - pos} below\nmean {dev.mean():+.3f}\n"
             f"{int((np.abs(dev) > 1).sum())} beyond ±1", transform=ax2.transAxes,
             ha="right", va="top", fontsize=9, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    ax2.set_xlabel("BER deviation, E2 vs deployed [4σ-band units; |x| > 1 is outside]",
                   color=INK2, fontsize=9)
    ax2.set_ylabel("sweep points", color=INK2, fontsize=9)
    ax2.set_title(f"{len(dev)} BER comparisons — scattered both ways, no bias",
                  color=INK, fontsize=10.5, loc="left")
    fig.suptitle("E2 — the 30 dB level reproduces the deployed D-series (regress_snr30.py)",
                 color=INK, fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_regression_30db.png")


# ---------------------------------------------------------------- 8. what the CNN sees
def fig_spectrograms():
    """
    The images the spectrogram CNN receives (snr_examples.py): clean, the plain GAN
    and the CNN-targeted GAN, each at the JSR where it reaches matched BER, at four
    noise levels. Read the 15 dB and 30 dB columns together: D2 sits at the same JSR in
    both, but at 15 dB it is buried in 15 dB more noise and the CNN mostly misses it.
    The colour-scale range printed above each column is the off-design caveat.
    """
    path = os.path.join(AD, "examples.npz")
    if not os.path.exists(path):
        print("  (no examples.npz -- run `sbatch submit_snr_examples.sh`; fig_spectrograms skipped)")
        return
    z = np.load(path)
    imgs, jsr, pdet, scales = z["images"], z["jsr"], z["pdet"], z["scales"]
    levels, rows = z["levels"], [str(r) for r in z["rows"]]
    names = {"clean": "clean", "plain_run001": "plain GAN — D1", "spec_cnn_b10": "D2 vs CNN, β=10"}
    nl, nr = len(levels), len(rows)
    fig, axes = plt.subplots(nr, nl, figsize=(3.25 * nl, 2.35 * nr + 0.9), squeeze=False)
    for c in range(nl):
        lab = "noiseless" if np.isnan(levels[c]) else f"SNR {levels[c]:g} dB"
        axes[0][c].set_title(f"{lab}\ncolour scale {scales[c][0]:.0f} … {scales[c][1]:.0f} dB",
                             color=INK, fontsize=9.5)
        for r in range(nr):
            ax = axes[r][c]
            ax.imshow(imgs[c][r], aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            cap = (f"FAR {pdet[c][r]:.2f}" if rows[r] == "clean"
                   else f"JSR {jsr[c][r]:+.1f} dB · P(det) {pdet[c][r]:.2f}")
            ax.text(0.03, 0.04, cap, transform=ax.transAxes, fontsize=8.3, color=INK,
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
            if c == 0:
                ax.set_ylabel(names.get(rows[r], rows[r]), color=INK, fontsize=9.5)
    fig.text(0.5, 0.005, "frequency →   (each panel: one frame's spectrogram, time ↓, as the CNN receives it)",
             ha="center", fontsize=8.5, color=INK2)
    fig.suptitle(f"E2 — what the CNN sees: each jammer at matched excess BER {float(z['ber_ref']):g}",
                 color=INK, fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig_spectrograms_by_snr.png")


# ---------------------------------------------------------------- summary + entry point
def save(fig, name):
    path = os.path.join(AD, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


SHORT = {"power_one_sided": "pow1s", "power_two_sided": "pow2s",
         "kurtosis": "kurt", "spec_cnn": "cnn"}


def summary(levels, rows, onset_target=BER_ONSET, ber_ref=BER_REF):
    """The numbers the write-up quotes, printed so they are never re-typed by hand."""
    print(f"\nedges in received JSR [dB], barrage noise, alpha = {ALPHA:g}, "
          f"onset at excess BER {onset_target:g}")
    print(f"{'SNR':>10} {'cleanBER':>11} {'FAR_cnn':>8} "
          + " ".join(f"{'edge_' + SHORT[d]:>9}" for d in DETS)
          + f" {'onset':>8} {'−SNR':>8} {'gap_cnn':>8} {'gapNP_cnn':>10}")
    for d in levels:
        s = series(d, "noise")
        if s is None:
            continue
        jsr, exc, pdet = s
        nf = d["meta"].get("n_frames", 512)
        edges = [stealth_edge(jsr, pdet[det], frames=nf) for det in DETS]
        o = ber_onset(jsr, exc, onset_target)
        par = None if d["snr"] is None else noise_parity(d["snr"])
        e_cnn = edges[DETS.index("spec_cnn")]
        gap = None if (o is None or e_cnn is None) else o - e_cnn
        gap_np = None if (par is None or e_cnn is None) else par - e_cnn
        f = lambda v, w, p=1: f"{'--' if v is None else f'{v:+.{p}f}':>{w}}"   # noqa: E731
        print(f"{label_of(d):>10} {d['clean_ber']:>11.3e} "
              f"{d['clean']['pdet'].get('spec_cnn', {}).get(A, float('nan')):>8.4f} "
              + " ".join(f(e, 9) for e in edges)
              + f" {f(o, 8)} {f(par, 8)} {f(gap, 8)} {f(gap_np, 10)}")

    print(f"\nmatched excess BER {ber_ref:g} — P(det) [JSR dB]")
    for det in DETS:
        print(f"  {DET_LABEL[det]}")
        for key, tag, lab in rows:
            cells = []
            for d in levels:
                p, j = at_matched_ber(d, tag, det, ber_ref)
                cells.append(f"{label_of(d)}: {'--' if p is None else f'{p:.3f}'}"
                             f"{'' if j is None else f' [{j:+.1f}]'}")
            print(f"    {lab:<28} " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset", type=float, default=BER_ONSET,
                    help="excess BER that defines 'the jammer bites'. The gap result is "
                         "sensitive to it by construction -- vary it before quoting one.")
    ap.add_argument("--ber-ref", type=float, default=BER_REF,
                    help="excess BER the matched-BER comparison is taken at. §3.3f quotes "
                         "its headline at 3e-4, where the clean floor is 0 so excess == raw; "
                         "the gain is strongly dependent on it (§3.3f limit (i)).")
    ap.add_argument("--frontier-det", default="spec_cnn", choices=DETS)
    args = ap.parse_args()

    levels = load()
    if not levels:
        print(f"no sweeps in {AD} — run `sbatch --array=0-9 submit_snr_ablation.sh` first")
        return
    print(f"levels: {', '.join(label_of(d) for d in levels)}")
    have = set()
    for d in levels:
        have |= set(d["attacks"]) | set(d["generators"])
    rows = [(k, t, l) for k, t, l in
            [("noise", "noise", "noise (barrage)"),
             ("optimal", "pulsed_p1", "matched QPSK (p=1)"),
             ("amuru", "pulsed_p0.1", "Amuru pulsed (p=0.1)"),
             ("plain", "plain_run001", "plain GAN — D1"),
             ("eff", "eff", "GAN eff, β=0 — D2"),
             ("g_kurt", "kurtosis_b1", "GAN vs kurtosis, β=1 — D2"),
             ("g_cnn", "spec_cnn_b10", "GAN vs CNN, β=10 — D2")]
            if t in have]
    print(f"jammers drawn: {', '.join(t for _, t, _ in rows)}")
    matched = [r for r in rows if r[0] in ("plain", "eff", "g_cnn", "g_kurt", "amuru")]
    fig_gap(levels, args.onset)
    fig_matched(levels, matched, args.ber_ref)
    fig_far(levels)
    fig_frontier(levels, [r for r in rows if r[0] in ("plain", "g_cnn")], args.frontier_det)
    fig_headline(levels, args.ber_ref)
    fig_mechanism(levels, args.onset)
    fig_regression()
    fig_spectrograms()
    summary(levels, matched, args.onset, args.ber_ref)


if __name__ == "__main__":
    main()
