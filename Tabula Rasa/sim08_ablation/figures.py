"""
sim08 ablations -- figures and summary (README §3.3c). CPU-only post-processing of
../artifacts/sim08_ablation/<run>/sweep_*.json (one per noise level, ablation.py).

    sbatch submit_figures.sh             # after the sweep array has finished

Writes to ../artifacts/sim08_ablation/<run>/:
  noise_ablation.png     fixed jammer's P(detect) and BER vs Eb/N0; best stealthy BER
  power_ablation.png     P(detect) and BER vs jammer power at four noise levels
  stealth_map.png        (power x n_active): who catches each config, BER where nobody does
  njammers_frontier.png  achievable frontier BER*(beta) per N_J, CNN alone and suite
  njammers_summary.png   N_J at matched detectability and at matched total power
  summary.json           every number the figures and README quote

Reading rules (CLAUDE.md): the stealth budget is each detector's own clean false-alarm
rate; "stealthy" numbers are the CONFIRMED frontier picks (fresh frames), drawn hollow
when the confirmation put them > 2 sigma above the budget. The stealth map classifies
each sweep config by a 2-sigma test against the clean FAR instead, because it shows
every cell rather than one max-selected pick.

Colour (dataviz skill; palettes checked with its validator): detectors take categorical
slots 1-3 (CNN blue, energy orange, suite aqua, direct-labelled because aqua is below
3:1 contrast). N_J is ordered, so it takes the one-hue blue ordinal ramp -- used only
in the N_J figures, where no line is a detector. Grey bands mark what the frozen CNN
never saw in training (Eb/N0 outside 5-30 dB, power outside 0.3-8).
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm            # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "sim08_ablation")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND, DETECTED = "#e1e0d9", "#c3c2b7", "#f0efec", "#e1e0d9"
C_DET = {"cnn": "#2a78d6", "energy": "#eb6834", "suite": "#1baf7a"}
L_DET = {"cnn": "CNN", "energy": "Energy", "suite": "Suite (CNN or energy)"}
C_NJ = {1: "#86b6ef", 2: "#2a78d6", 4: "#104281"}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
NOISELESS_X = 45.0
BER_MIN = 1e-6                     # zero-error points are drawn here, as a down-triangle
REF = dict(n_active=16, power=1.0)  # the fixed reference jammer: sim08's residual stealthy config
LEVELS4 = [5.0, 15.0, 30.0, None]
LEVELS3 = [10.0, 20.0, 30.0]
CNN_TRAIN_EBNO = (5.0, 30.0)
CNN_TRAIN_POWER = (10 ** -0.5, 10 ** 0.9)


# ---------------------------------------------------------------- data
def load(run):
    levels = []
    for f in sorted(glob.glob(os.path.join(ART, run, "sweep_*.json"))):
        d = json.load(open(f))
        if "confirm" not in d:
            print(f"WARNING: {os.path.basename(f)} has no confirmation pass (task incomplete); skipped")
            continue
        levels.append(d)
    levels.sort(key=lambda d: (d["meta"]["noiseless"], d["meta"]["ebno_db"] or 0.0))
    return levels


def xpos(d):
    return NOISELESS_X if d["meta"]["noiseless"] else d["meta"]["ebno_db"]


def level(levels, ebno):
    return next(d for d in levels if d["meta"]["ebno_db"] == ebno
                and d["meta"]["noiseless"] == (ebno is None))


def level_name(ebno):
    return "no noise" if ebno is None else f"Eb/N0 {ebno:g} dB"


def row(d, nj, na, p):
    return next(r for r in d["rows"] if r["n_jammers"] == nj and r["n_active"] == na
                and abs(r["power"] - p) < 1e-9 * max(p, 1))


def pick(d, nj, det, budget="far"):
    return next(e for e in d["confirm"] if e["n_jammers"] == nj and e["detector"] == det
                and e["budget"] == str(budget))


def sigma_above(p, n, far, far_n):
    var = p * (1 - p) / n + far * (1 - far) / far_n
    return (p - far) / math.sqrt(var) if var > 0 else (math.inf if p > far else 0.0)


def ber_plot(v):
    return max(v, BER_MIN)


# ---------------------------------------------------------------- style
def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
    })


def clean_axes(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y" if grid_y else "both", color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)


def noise_axis(ax, show_labels=True):
    """Eb/N0 axis with the noiseless anchor as its own tick and the CNN's OOD bands."""
    ax.axvspan(-1.5, CNN_TRAIN_EBNO[0], color=BAND, lw=0, zorder=0)
    ax.axvspan(CNN_TRAIN_EBNO[1], NOISELESS_X + 2.5, color=BAND, lw=0, zorder=0)
    ax.set_xlim(-1.5, NOISELESS_X + 2.5)
    ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, NOISELESS_X]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks[:-1]] + ["no\nnoise"] if show_labels else [])


def series(levels, fn):
    """(x, y) over noise levels; the noiseless anchor returned separately (not joined)."""
    xs, ys, x0, y0 = [], [], None, None
    for d in levels:
        v = fn(d)
        if d["meta"]["noiseless"]:
            x0, y0 = xpos(d), v
        else:
            xs.append(xpos(d)); ys.append(v)
    return np.array(xs), np.array(ys, dtype=float), x0, y0


def line_with_anchor(ax, levels, fn, color, ls="-", marker="o", label=None, lw=1.6, mfc=None):
    xs, ys, x0, y0 = series(levels, fn)
    ok = np.isfinite(ys)
    ax.plot(xs[ok], ys[ok], color=color, ls=ls, lw=lw, marker=marker,
            mfc=mfc or color, mec=SURFACE if mfc is None else color, mew=0.8, label=label)
    if y0 is not None and np.isfinite(y0):
        ax.plot([x0], [y0], color=color, marker=marker, ls="none",
                mfc=mfc or color, mec=SURFACE if mfc is None else color, mew=0.8)
    return xs, ys


def end_label(ax, x, y, text, color=INK2, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center",
                fontsize=8, color=color, **kw)


# ---------------------------------------------------------------- figure 1: noise
def fig_noise(levels, out):
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1, 1.1]))
    a, b, c = axes
    na, p = REF["n_active"], REF["power"]

    # (a) the fixed jammer's P(detect) next to each detector's clean false-alarm rate
    for det in ("cnn", "energy", "suite"):
        col = C_DET[det]
        xs, ys = line_with_anchor(a, levels, lambda d: row(d, 1, na, p)[f"p_{det}"], col)
        line_with_anchor(a, levels, lambda d: d["clean"][f"p_{det}"], col, ls=(0, (3, 2)),
                         marker="", lw=1.1)
        end_label(a, xs[-1], ys[-1], L_DET[det].split(" (")[0], color=INK2,
                  dy={"cnn": 6, "energy": 0, "suite": -6}[det])
    a.set_ylim(-0.02, 1.02); a.set_ylabel("P(detect)")
    a.set_title(f"(a) One fixed jammer (n = {na} subcarriers, power {p:g}; JSR ≈ −5 dB):"
                " detection vs noise\nsolid = jammed frames, dashed = the same detector on clean"
                " frames (its false-alarm rate)")
    a.legend(handles=[Line2D([], [], color=C_DET[k], label=L_DET[k]) for k in C_DET],
             loc="upper right", fontsize=8, ncol=3)

    # (b) the same jammer's BER vs the clean floor
    line_with_anchor(b, levels, lambda d: ber_plot(row(d, 1, na, p)["ber_mean"]), INK)
    line_with_anchor(b, levels, lambda d: ber_plot(d["clean"]["ber_mean"]), MUTED,
                     ls=(0, (3, 2)), marker="", lw=1.1)
    d_last = [d for d in levels if not d["meta"]["noiseless"]][-1]
    end_label(b, 40, row(d_last, 1, na, p)["ber_mean"], "jammed", dy=6)
    end_label(b, 40, ber_plot(d_last["clean"]["ber_mean"]), "clean floor", color=MUTED)
    b.set_yscale("log"); b.set_ylim(BER_MIN * 0.6, 0.6); b.set_ylabel("BER")
    b.set_title("(b) The same jammer's BER (clean floor dashed; zero-error points drawn at 1e-6)")

    # (c) best confirmed stealthy BER at each detector's own FAR budget. The two
    # curves nearly coincide (the suite is the CNN here), so CNN is drawn wide
    # underneath and the suite thin on top rather than hidden.
    for det, lw in (("cnn", 3.4), ("suite", 1.5)):
        col = C_DET[det]
        xs, ys, x0, y0 = series(levels, lambda d: pick(d, 1, det)["confirmed"]["ber_mean"]
                                if pick(d, 1, det)["pick"] else np.nan)
        _, fl, _, fl0 = series(levels, lambda d: bool(pick(d, 1, det).get("flagged")))
        c.plot(xs, [ber_plot(v) for v in ys], color=col, lw=lw, zorder=3,
               alpha=0.55 if det == "cnn" else 1.0,
               label=f"best stealthy vs {L_DET[det].split(' (')[0]}")
        for x, y, f in list(zip(xs, ys, fl)) + [(x0, y0, fl0)]:
            if x is None or y is None or not np.isfinite(y):
                continue
            c.plot([x], [ber_plot(y)], marker="o", ms=5, color=col, zorder=4,
                   mfc=SURFACE if f else col, mec=col if f else SURFACE, mew=1.2 if f else 0.8)
    line_with_anchor(c, levels, lambda d: ber_plot(d["clean_confirmed"]["ber_mean"]), MUTED,
                     ls=(0, (3, 2)), marker="", lw=1.1)
    c.set_yscale("log"); c.set_ylim(BER_MIN * 0.6, 0.6); c.set_ylabel("BER")
    c.set_xlabel("Eb/N0 (dB)  —  grey bands: outside the CNN's training range")
    c.set_title("(c) Strongest jammer that stays within each detector's own false-alarm rate"
                "\n(N_J = 1; confirmed on fresh frames; hollow = confirmation put it > 2σ above"
                " the budget)")
    c.legend(handles=[Line2D([], [], color=C_DET["cnn"], marker="o", lw=3.4, alpha=0.55,
                             label="vs CNN alone"),
                      Line2D([], [], color=C_DET["suite"], marker="o", label="vs suite"),
                      Line2D([], [], color=MUTED, ls=(0, (3, 2)), label="clean floor")],
             loc="upper right", fontsize=8, ncol=3)
    for ax in axes:
        noise_axis(ax)
        clean_axes(ax)
    fig.suptitle("Noise ablation — sim08 realistic channel, frozen channel-valid CNN",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- figure 2: power
def fig_power(levels, out):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey="row")
    na = REF["n_active"]
    for j, eb in enumerate(LEVELS4):
        d = level(levels, eb)
        rows = sorted([r for r in d["rows"] if r["n_jammers"] == 1 and r["n_active"] == na],
                      key=lambda r: r["power"])
        pw = np.array([r["power"] for r in rows])
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            ax.axvspan(pw[0] / 1.5, CNN_TRAIN_POWER[0], color=BAND, lw=0, zorder=0)
            ax.axvspan(CNN_TRAIN_POWER[1], pw[-1] * 1.5, color=BAND, lw=0, zorder=0)
        for det in ("cnn", "energy", "suite"):
            col = C_DET[det]
            top.plot(pw, [r[f"p_{det}"] for r in rows], color=col, marker="o", mec=SURFACE,
                     mew=0.8, label=L_DET[det])
            top.axhline(d["clean"][f"p_{det}"], color=col, ls=(0, (3, 2)), lw=1.0)
        bot.plot(pw, [ber_plot(r["ber_mean"]) for r in rows], color=INK, marker="o", mec=SURFACE,
                 mew=0.8)
        bot.axhline(ber_plot(d["clean"]["ber_mean"]), color=MUTED, ls=(0, (3, 2)), lw=1.1)
        top.set_title(level_name(eb) + ("" if eb is None or CNN_TRAIN_EBNO[0] <= eb <= CNN_TRAIN_EBNO[1]
                                        else "  (outside CNN training)"))
        top.set_ylim(-0.02, 1.02); bot.set_yscale("log"); bot.set_ylim(BER_MIN * 0.6, 0.6)
        bot.set_xscale("log"); bot.set_xlim(pw[0] / 1.5, pw[-1] * 1.5)
        bot.set_xlabel("jammer power per active subcarrier")
        for ax in (top, bot):
            clean_axes(ax)
    axes[0, 0].set_ylabel("P(detect)")
    axes[1, 0].set_ylabel("BER")
    last = level(levels, LEVELS4[-1])
    end_label(axes[1, 3], axes[1, 3].get_xlim()[0] * 1.6, ber_plot(last["clean"]["ber_mean"]),
              "clean floor" + (" (0)" if last["clean"]["ber_mean"] == 0 else ""), color=MUTED, dy=7)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Power ablation — one jammer on n = {na} subcarriers; dashed = the detector's"
                 " clean false-alarm rate (top) / clean BER floor (bottom); grey = outside the"
                 " CNN's training power", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- figure 3: stealth map
def classify(d, r):
    """'' = indistinguishable from clean for the suite (2 sigma), else who catches it."""
    cf, fn = d["clean"], d["clean"]["frames"]
    hit = {k: sigma_above(r[f"p_{k}"], r["frames"], cf[f"p_{k}"], fn) > 2.0
           for k in ("cnn", "energy", "suite")}
    if not hit["suite"]:
        return ""
    if hit["cnn"] and hit["energy"]:
        return "B"
    if hit["cnn"]:
        return "C"
    if hit["energy"]:
        return "E"
    return "S"


def fig_stealth_map(levels, out):
    d0 = levels[0]
    powers, nas = d0["meta"]["powers"], d0["meta"]["n_active"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    # Colour the EXCESS over the clean floor, not the BER: at low Eb/N0 the floor
    # is 0.18 and would saturate every cell, hiding what the jammer actually adds.
    norm = LogNorm(vmin=1e-5, vmax=0.5)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    counts = {}
    for ax, eb in zip(axes.flat, LEVELS4):
        d = level(levels, eb)
        floor = d["clean"]["ber_mean"]
        grid = np.full((len(nas), len(powers)), np.nan)
        tally = dict(stealthy=0, C=0, E=0, B=0, S=0)
        for i, na in enumerate(nas):
            for j, p in enumerate(powers):
                r = row(d, 1, na, p)
                k = classify(d, r)
                if k:
                    tally[k] += 1
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=DETECTED, lw=0))
                    ax.text(j, i, k, ha="center", va="center", fontsize=7, color=INK2)
                else:
                    tally["stealthy"] += 1
                    # Only colour damage the sweep can actually measure: at 0 dB the
                    # per-config BER error is ~2e-3, which would otherwise paint noise.
                    sem = math.hypot(r["ber_sem"], d["clean"]["ber_sem"])
                    excess = r["ber_mean"] - floor
                    grid[i, j] = max(excess, 1e-5) if excess > 2 * sem else 1e-5
        counts[level_name(eb)] = tally
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="lower",
                       aspect="auto", extent=(-0.5, len(powers) - 0.5, -0.5, len(nas) - 0.5))
        for x in range(len(powers) + 1):                   # 2px-surface-gap look between cells
            ax.axvline(x - 0.5, color=SURFACE, lw=1.2)
        for y in range(len(nas) + 1):
            ax.axhline(y - 0.5, color=SURFACE, lw=1.2)
        for bound in CNN_TRAIN_POWER:                      # snapped to a cell boundary
            ax.axvline(round(8 + 4 * math.log10(bound)) - 0.5, color=INK2, lw=0.9)
        e = pick(d, 1, "suite")
        if e["pick"]:
            j = powers.index(e["pick"]["power"]); i = nas.index(e["pick"]["n_active"])
            ax.plot([j], [i], marker="o", ms=13, mfc="none", mec=MUTED if e["flagged"] else INK,
                    mew=1.4)
        ax.set_xticks([0, 4, 8, 12]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
        ax.set_yticks(range(len(nas))); ax.set_yticklabels(nas)
        ax.set_xlim(-0.5, len(powers) - 0.5); ax.set_ylim(-0.5, len(nas) - 0.5)
        ax.set_xlabel("jammer power per active subcarrier"); ax.set_ylabel("active subcarriers")
        ax.set_title(f"{level_name(eb)} — suite FAR {d['clean']['p_suite']:.3f}, "
                     f"clean BER {d['clean']['ber_mean']:.1e}")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for ax in axes.flat[len(LEVELS4):]:                    # unused panel (smoke runs)
        ax.axis("off")
    fig.suptitle("Where the detectors are evaded — one jammer\nColoured = no detector can tell it"
                 " from clean (2σ); grey = caught, by C: CNN only · E: energy only · B: both ·"
                 " S: only their OR\nVertical lines: the CNN's training power range.  Ring: the"
                 " confirmed strongest stealthy config (grey = failed confirmation)",
                 x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.86, top=0.86, bottom=0.07, hspace=0.32, wspace=0.16)
    cb = fig.colorbar(im, cax=fig.add_axes([0.89, 0.1, 0.016, 0.66]))
    cb.set_label("BER above the clean floor (palest = none measurable)")
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)
    return counts


# ---------------------------------------------------------------- figure 4: N_J frontier
def frontier(rows, det):
    pts = sorted((r[f"p_{det}"], r["ber_mean"]) for r in rows)
    x = np.array([p for p, _ in pts]); y = np.maximum.accumulate([b for _, b in pts])
    return np.append(x, 1.0), np.append(y, y[-1])


def fig_nj_frontier(levels, out):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharex=True)
    for i, det in enumerate(("cnn", "suite")):
        for j, eb in enumerate(LEVELS3):
            ax = axes[i, j]
            d = level(levels, eb)
            floor = d["clean"]["ber_mean"]
            for nj, lw in ((1, 3.4), (2, 2.0), (4, 1.2)):   # tapered: the curves coincide
                x, y = frontier([r for r in d["rows"] if r["n_jammers"] == nj], det)
                ax.step(x, y, where="post", color=C_NJ[nj], lw=lw, label=f"N_J = {nj}")
            far = d["clean"][f"p_{det}"]
            ax.axvline(far, color=INK2, lw=1.0, ls=(0, (1, 2)))
            ax.axhline(floor, color=MUTED, lw=1.1, ls=(0, (3, 2)))
            ax.annotate(f"clean FAR {far:.3f}", (far, 0.97), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5, color=INK2,
                        xycoords=("data", "axes fraction"), va="top")
            ax.annotate("clean floor", (0.99, floor), xytext=(0, 4), fontsize=7.5, color=MUTED,
                        textcoords="offset points", xycoords=("axes fraction", "data"), ha="right")
            ax.set_yscale("log"); ax.set_ylim(max(floor * 0.4, 1e-5), 0.6); ax.set_xlim(0, 1)
            ax.set_title(f"{level_name(eb)} — vs {L_DET[det]}")
            if i == 1:
                ax.set_xlabel(f"P(detect) by {L_DET[det].split(' (')[0]}  (budget β)")
            if j == 0:
                ax.set_ylabel("max BER within budget")
            clean_axes(ax, grid_y=False)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Number of independent jammers at equal total power — achievable frontier"
                 " BER*(β) = max BER over configs with P(detect) ≤ β (sweep data)\ndotted:"
                 " the detector's clean false-alarm rate (the stealth budget); dashed: clean BER"
                 " floor", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- figure 5: N_J summary
def fig_nj_summary(levels, out):
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.2), sharex=True)
    a, b, c = axes
    na, p = REF["n_active"], REF["power"]
    for nj, lw in ((1, 3.4), (2, 2.0), (4, 1.2)):          # tapered: the curves coincide
        col = C_NJ[nj]
        xs, ys, x0, y0 = series(levels, lambda d: pick(d, nj, "suite")["confirmed"]["ber_mean"]
                                if pick(d, nj, "suite")["pick"] else np.nan)
        _, fl, _, fl0 = series(levels, lambda d: bool(pick(d, nj, "suite").get("flagged")))
        a.plot(xs, [ber_plot(v) for v in ys], color=col, lw=lw, label=f"N_J = {nj}")
        for x, y, f in list(zip(xs, ys, fl)) + [(x0, y0, fl0)]:
            if x is not None and y is not None and np.isfinite(y):
                a.plot([x], [ber_plot(y)], marker="o", ms=5, color=col,
                       mfc=SURFACE if f else col, mec=col if f else SURFACE, mew=1.2 if f else 0.8)
        line_with_anchor(b, levels, lambda d: ber_plot(row(d, nj, na, p)["ber_mean"]), col,
                         label=f"N_J = {nj}", lw=lw)
        line_with_anchor(c, levels, lambda d: row(d, nj, na, p)["p_suite"], col,
                         label=f"N_J = {nj}", lw=lw)
    for ax in (a, b):
        line_with_anchor(ax, levels, lambda d: ber_plot(d["clean"]["ber_mean"]), MUTED,
                         ls=(0, (3, 2)), marker="", lw=1.1)
        ax.set_yscale("log"); ax.set_ylim(BER_MIN * 0.6, 0.6); ax.set_ylabel("BER")
    line_with_anchor(c, levels, lambda d: d["clean"]["p_suite"], MUTED, ls=(0, (3, 2)),
                     marker="", lw=1.1)
    c.set_ylim(-0.02, 1.02); c.set_ylabel("P(detect) by suite")
    a.set_title("(a) Matched detectability: strongest confirmed jammer within the suite's own"
                " false-alarm rate\n(hollow = confirmation put it > 2σ above the budget; dashed ="
                " clean floor)")
    b.set_title(f"(b) Matched total power: BER with each jammer on n = {na} subcarriers, total"
                f" power {na * p:g} split evenly")
    c.set_title("(c) Matched total power: the same configs' suite detection (dashed = clean FAR)")
    c.set_xlabel("Eb/N0 (dB)  —  grey bands: outside the CNN's training range")
    for ax in axes:
        noise_axis(ax); clean_axes(ax)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
    fig.suptitle("Number of independent jammers — reported both ways (CLAUDE.md)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- summary
def summarise(levels, counts):
    out = dict(levels=[], stealth_map_counts=counts, reference_jammer=REF)
    print("\nStrongest confirmed stealthy jammer at each detector's own FAR budget "
          "(* = failed confirmation)")
    print(f"{'level':>9} | {'floor':>8} | {'FAR cnn':>7} {'FAR suite':>9} | "
          + " | ".join(f"vs {det}: N_J n p -> BER (xfloor)" for det in DETECTORS))
    for d in levels:
        cc = d["clean_confirmed"]
        entry = dict(ebno_db=d["meta"]["ebno_db"], noiseless=d["meta"]["noiseless"],
                     clean_sweep=d["clean"], clean_confirmed=cc, picks=d["confirm"],
                     reference={nj: row(d, nj, REF["n_active"], REF["power"]) for nj in (1, 2, 4)},
                     strongest={})
        cells = []
        for det in DETECTORS:
            cands = [e for e in d["confirm"] if e["detector"] == det and e["budget"] == "far"
                     and e["pick"]]
            good = [e for e in cands if not e["flagged"]] or cands
            if not good:
                entry["strongest"][det] = None; cells.append("none"); continue
            best = max(good, key=lambda e: e["confirmed"]["ber_mean"])
            ber, floor = best["confirmed"]["ber_mean"], cc["ber_mean"]
            z = (ber - floor) / math.sqrt(best["confirmed"]["ber_sem"] ** 2 + cc["ber_sem"] ** 2) \
                if (best["confirmed"]["ber_sem"] or cc["ber_sem"]) else math.inf
            entry["strongest"][det] = dict(n_jammers=best["n_jammers"], **best["pick"], ber=ber,
                                           p_det=best["confirmed"][f"p_{det}"],
                                           budget=best["budget_confirmed"],
                                           flagged=best["flagged"], ber_over_floor=ber - floor,
                                           ber_over_floor_sigma=z,
                                           all_confirmations_flagged=not [e for e in cands
                                                                          if not e["flagged"]])
            ratio = f"x{ber / floor:.1f}" if floor > 0 else "floor=0"
            cells.append(f"{best['n_jammers']} {best['pick']['n_active']:>2} "
                         f"{best['pick']['power']:<7.3g} -> {ber:.4f} ({ratio}, {z:+.1f}σ)"
                         + ("*" if best["flagged"] else ""))
        name = "noiseless" if d["meta"]["noiseless"] else f"{d['meta']['ebno_db']:g} dB"
        print(f"{name:>9} | {cc['ber_mean']:8.2e} | {cc['p_cnn']:7.3f} {cc['p_suite']:9.3f} | "
              + " | ".join(cells))
        out["levels"].append(entry)
    return out


DETECTORS = ("cnn", "suite")


def main():
    global LEVELS4, LEVELS3
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run001")
    ap.add_argument("--smoke", action="store_true",
                    help="use whichever noise levels exist, to shake out the plotting code "
                         "before the array has finished")
    args = ap.parse_args()
    levels = load(args.run)
    if not levels:
        raise SystemExit("no completed sweep files")
    if args.smoke:
        have = [d["meta"]["ebno_db"] if not d["meta"]["noiseless"] else None for d in levels]
        LEVELS4, LEVELS3 = have[:4], have[:3]
        print(f"SMOKE: panels use {LEVELS4} / {LEVELS3}")
    print(f"{len(levels)} noise levels: {[level_name(d['meta']['ebno_db']) for d in levels]}")
    style()
    od = os.path.join(ART, args.run)
    fig_noise(levels, os.path.join(od, "noise_ablation.png"))
    fig_power(levels, os.path.join(od, "power_ablation.png"))
    counts = fig_stealth_map(levels, os.path.join(od, "stealth_map.png"))
    fig_nj_frontier(levels, os.path.join(od, "njammers_frontier.png"))
    fig_nj_summary(levels, os.path.join(od, "njammers_summary.png"))
    summary = summarise(levels, counts)
    print("\nstealth-map cell counts (N_J = 1, 91 configs each):")
    for k, v in counts.items():
        print(f"  {k:>16}: {v}")
    with open(os.path.join(od, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote 5 figures + summary.json to {od}")


if __name__ == "__main__":
    main()
