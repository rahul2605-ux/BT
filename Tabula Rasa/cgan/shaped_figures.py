"""
D2a -- figures and summary.json for the learned shaped-noise control tier
(train_shaped.py), with the fixed run001 baselines overlaid (README §3.3d, §3.4).

    sbatch submit_shaped_figures.sh

Reads artifacts/cgan/learned/run001/task*.json and ../baselines/run001/sweep_K1.json;
writes into artifacts/cgan/learned/run001/:

    fig1_frontier.png     matched detectability at EVERY budget: per evaluating
                          detector, the most BER each attack reaches while flagged in
                          at most a fraction x of frames, x = 0..1 (FAR dotted; raw
                          points faint underneath). Fixed rows, shaped beta = 0, and
                          the shaped jammers trained against that detector.
    fig2_transfer.png     trained-against x evaluated-by P(det) at three JSRs.
    fig3_shapes.png       what CMA-ES found: PSD (victim band shaded) and envelope.
    fig4_convergence.png  best fitness per generation, per task and JSR, and the
                          confirmed gain over theta = 0.
    fig5_ber_vs_jsr.png   BER vs JSR: fixed rows vs shaped at beta = 0 (prediction P2).
    summary.json          predictions P1-P3, the theta = 0 check, the seed check, and
                          BER at matched P(det) budgets 0.05 / 0.1 / 0.25 / 0.5.
"""

import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import attacks
import baselines_figures as bf
import link as lk
import train_shaped as ts

OUT = ts.OUT
SA = bf.SA
TGT = ts.TARGETS
DETS = bf.DET_ORDER
FIXED = ["noise", "pulsed_p1", "pulsed_p0.1"]
BUDGETS = [0.05, 0.1, 0.25, 0.5]           # P(det) budgets for the matched-BER table
# single-hue blue ramp (dataviz palette): 100..700 for magnitude, 250..700 for ordered marks
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", BLUE)
CLIP_DB = -40.0
ORD = matplotlib.colors.LinearSegmentedColormap.from_list("blue_ord", BLUE[3:])
DLABEL = dict(bf.DET_LABEL, lrt_noise="Noise LRT (mismatched)")   # optimal only for white noise
EDGE_MHZ = (1 + 0.35) / 2                                           # RRC 0.35 band edge at 1 MBd
BETA_STYLE = {0.0: dict(ls="-", marker="s"), 0.01: dict(ls=(0, (4, 2)), marker="^"),
              1.0: dict(ls="-", marker="o")}


def load_tasks():
    tasks = {}
    for p in sorted(glob.glob(os.path.join(OUT, "task*.json"))):
        t = int(os.path.basename(p)[4:-5])
        tasks[t] = json.load(open(p))["result"]
    return tasks


def task_of(target, beta):
    return ts.TASKS.index(dict(target=target, beta=beta))


def row(runs, key="confirm"):
    jsr = np.array([r["jsr_db"] for r in runs])
    return jsr, [r[key] for r in runs]


def pdet(points, det):
    return np.array([p["pdet"].get(det, {}).get(SA, np.nan) for p in points])


def ber(points):
    return np.clip(np.array([p["ber"] for p in points]), 1e-6, 1)


def shape_metrics(theta, n=4096):
    """In-band share of the jammer's power (|f| <= RRC band edge) and the envelope's
    effective duty cycle (sum e)^2 / (16 sum e^2), 1/16 = all power in one slot."""
    th = np.asarray(theta, dtype=float)
    g = 10 ** (attacks.shaped_psd_db(th, n).numpy() / 10)
    f_mhz = np.fft.fftfreq(n) * lk.LINK["sps"]
    e = np.exp(th[attacks.SHAPED_BINS:])
    return dict(inband=float(g[np.abs(f_mhz) <= EDGE_MHZ].sum() / g.sum()),
                duty=float(e.sum() ** 2 / (attacks.SHAPED_PERIOD * (e ** 2).sum())))


def _legend_union(fig, axes, **kw):
    seen = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen.keys()), **kw)


# ---------------------------------------------------------------- fig 1
def frontier_rows(tasks, k1, det):
    """(label, colour, style, [(P(det), BER, JSR)]) for every attack shown against `det`."""
    rows = [(bf.ATTACK_LABEL[n], bf.ACOLOR[n], dict(ls="-", lw=1.4),
             list(zip(pdet(k1["attacks"][n], det), [p["ber"] for p in k1["attacks"][n]], k1["jsr_db"])))
            for n in FIXED]
    learned = [(0, "Shaped, β=0 (learned)", bf.INK, dict(ls="-", lw=2.0))]
    if det in TGT:
        learned += [(task_of(det, b), f"Shaped, trained vs this detector, β={b:g}", bf.PAL[6],
                     dict(ls=(0, (4, 2)) if b < 1 else "-", lw=2.0)) for b in ts.BETAS]
    for t, lab, col, sty in learned:
        if t in tasks:
            j, pts = row(tasks[t]["runs"])
            rows.append((lab, col, sty, list(zip(pdet(pts, det), [p["ber"] for p in pts], j))))
    # a detector that was not evaluated on an attack (the noise LRT on pulsed) has no row, not a zero row
    return [r for r in rows if not all(np.isnan(p) for p, _, _ in r[3])]


def matched_ber(points, budget):
    """Most BER over the points with P(det) <= budget, and its JSR; (0, None) if none."""
    ok = [(b, j) for p, b, j in points if not np.isnan(p) and p <= budget]
    return max(ok) if ok else (0.0, None)


def fig_frontier(tasks, k1):
    x = np.linspace(0, 1, 1001)
    fig, axes = plt.subplots(1, len(DETS), figsize=(4.2 * len(DETS), 4.8), sharey=True)
    for ax, det in zip(axes, DETS):
        for lab, col, sty, pts in frontier_rows(tasks, k1, det):
            env = np.array([matched_ber(pts, v)[0] for v in x])
            ax.plot(x, np.clip(env, 1e-6, 1), color=col, label=lab, **sty)
            ax.scatter([p for p, _, _ in pts], np.clip([b for _, b, _ in pts], 1e-6, 1), s=7,
                       color=col, alpha=0.25, linewidths=0)
        ax.axvline(bf.ALPHA, color=bf.INK2, lw=1.0, ls=":", label=f"FAR α = {bf.ALPHA:g} (clean frames)")
        ax.set_yscale("log"); ax.set_ylim(8e-7, 1.4); ax.set_xlim(0, 1.0)
        ax.set_xlabel(f"P(detect) budget, {DLABEL[det]}"); ax.set_title(DLABEL[det], loc="left")
        bf._clean(ax)
    axes[0].set_ylabel("Most BER reachable within the budget\n(1e-6 = no errors)")
    _legend_union(fig, axes, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Matched detectability at every budget: most damage while flagged in ≤ x of frames "
                 "(faint dots = raw sweep points; learned rows on a 4 dB grid)", x=0.01, y=1.18, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_frontier.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_transfer(tasks, jsrs=(-32.0, -20.0, -16.0)):
    rows = [("White noise (θ=0)", 0, "confirm_theta0"), ("Shaped, β=0", 0, "confirm")] + \
           [(f"vs {DLABEL[t]}, β=1", task_of(t, 1.0), "confirm") for t in TGT]
    rows = [r for r in rows if r[1] in tasks]
    fig, axes = plt.subplots(1, len(jsrs), figsize=(5.6 * len(jsrs), 0.55 * len(rows) + 2.4), sharey=True)
    for ax, j in zip(axes, jsrs):
        M = np.full((len(rows), len(DETS)), np.nan)
        for i, (_, t, key) in enumerate(rows):
            r = next((r for r in tasks[t]["runs"] if r["jsr_db"] == j), None)
            if r is None:
                continue
            M[i] = [r[key]["pdet"].get(d, {}).get(SA, np.nan) for d in DETS]
            b = r[key]["ber"]
            ax.text(len(DETS) - 0.35, i, f"BER {b:.1e}" if b > 0 else "BER 0", ha="left", va="center",
                    fontsize=8, color=bf.INK2)
        ax.imshow(M, vmin=0, vmax=1, cmap=SEQ, aspect="auto")
        for (i, k), v in np.ndenumerate(M):
            if not np.isnan(v):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color=bf.SURFACE if v > 0.6 else bf.INK)
        ax.set_xlim(-0.5, len(DETS) + 1.2)
        ax.set_xticks(range(len(DETS)), [DLABEL[d] for d in DETS], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"JSR {j:+.0f} dB", loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)), [n for n, _, _ in rows], fontsize=8)
    fig.suptitle("P(detect) at α = 0.05: trained against (rows) × evaluated by (columns); "
                 "BER of that jammer at right", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_transfer.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_shapes(tasks, jsrs=(-20.0, -8.0)):
    n = 4096
    f_mhz = np.fft.fftshift(np.fft.fftfreq(n)) * lk.LINK["sps"]          # 1 MBd -> fs = sps MHz
    edge = EDGE_MHZ
    # a jammer trained against a detector wears that detector's colour (baselines_figures.DCOLOR)
    show = [(0, bf.INK, "β=0 (effectiveness only)")] + \
           [(task_of(t, 1.0), bf.DCOLOR[t], f"vs {DLABEL[t]}, β=1") for t in TGT]
    show = [s for s in show if s[0] in tasks]
    fig, axes = plt.subplots(len(jsrs), 2, figsize=(13, 4.2 * len(jsrs)), gridspec_kw=dict(width_ratios=[2, 1]))
    axes = np.atleast_2d(axes)
    for (ax_p, ax_e), j in zip(axes, jsrs):
        ax_p.axvspan(-edge, edge, color=bf.PAL[2], alpha=0.08, label="victim band (RRC 0.35)")
        for t, color, lab in show:
            r = next((r for r in tasks[t]["runs"] if r["jsr_db"] == j), None)
            if r is None:
                continue
            th = np.array(r["theta"])
            psd = attacks.shaped_psd_db(th, n).numpy()
            psd = np.fft.fftshift(psd - 10 * np.log10(np.mean(10 ** (psd / 10))))     # 0 dB = white at same power
            m = shape_metrics(th)
            ax_p.plot(f_mhz, np.maximum(psd, CLIP_DB), color=color, lw=1.6,
                      label=f"{lab}: {100 * m['inband']:.0f} % in band, duty {m['duty']:.2f}")
            env = th[attacks.SHAPED_BINS:]
            env_db = 10 / math.log(10) * env - 10 * np.log10(np.mean(np.exp(env)))
            ax_e.step(np.arange(attacks.SHAPED_PERIOD), np.maximum(env_db, CLIP_DB), where="mid",
                      color=color, lw=1.6)
        ax_p.set_xlabel("Frequency (MHz, baseband)"); ax_p.set_ylabel("PSD rel. to white (dB)")
        ax_p.set_title(f"Learned spectrum at JSR {j:+.0f} dB (floored at {CLIP_DB:.0f} dB: bins below "
                       "carry < 0.1 % of the power and are not identified)", loc="left", fontsize=9)
        ax_p.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
        bf._clean(ax_p)
        ax_e.set_xlabel("Symbol slot (period 16, own clock)"); ax_e.set_ylabel("Power rel. to mean (dB)")
        ax_e.set_title("Learned envelope", loc="left"); bf._clean(ax_e)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_shapes.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_convergence(tasks):
    ids = [t for t in range(len(ts.TASKS) - 1) if t in tasks]
    cols = 3
    rows_ = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 3.4 * rows_), squeeze=False,
                             constrained_layout=True)
    cmap = ORD
    for ax, t in zip(axes.flat, ids):
        runs = tasks[t]["runs"]
        for k, r in enumerate(runs):
            b = np.array([h["best"] for h in r["history"]])
            ax.plot(b - b[0], color=cmap(k / max(1, len(runs) - 1)), lw=1.0)
        task = ts.TASKS[t]
        name = "β=0 (ranked by log E[BER])" if task["target"] is None \
            else f"{DLABEL[task['target']]}, β={task['beta']:g}"
        ax.set_title(name, loc="left", fontsize=9); ax.set_xlabel("generation")
        ax.set_ylabel("Δ log E[BER] (nats)" if task["target"] is None else "best fitness − gen 0")
        bf._clean(ax)
    for ax in list(axes.flat)[len(ids):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(ts.JSR_GRID[0], ts.JSR_GRID[-1]))
    fig.colorbar(sm, ax=axes, label="JSR (dB)", shrink=0.6)
    fig.suptitle("CMA-ES convergence: best of each generation, on that generation's frames", x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "fig4_convergence.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_ber_vs_jsr(tasks, k1):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    jk = np.array(k1["jsr_db"])
    for name in FIXED:
        ax.plot(jk, ber(k1["attacks"][name]), color=bf.ACOLOR[name], lw=1.6, label=bf.ATTACK_LABEL[name])
    if 0 in tasks:
        j, pts = row(tasks[0]["runs"])
        ax.plot(j, ber(pts), color=bf.INK, lw=2, marker="s", ms=4, label="Shaped, β=0 (learned)")
        j, pts = row(tasks[0]["runs"], "confirm_theta0")
        ax.plot(j, ber(pts), color=bf.INK2, lw=1.2, ls=":", marker="o", ms=3, label="Shaped θ=0 (white)")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2); ax.set_xlabel("Received JSR (dB)"); ax.set_ylabel("BER")
    ax.set_title("What shaping buys in effectiveness alone (β = 0)", loc="left"); bf._clean(ax)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_ber_vs_jsr.png"), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- summary
def _cross(jsr, b):
    """JSR at BER 1e-3 by log-linear interpolation on the grid, or None."""
    b = np.maximum(np.asarray(b, dtype=float), 1e-12)
    for i in range(1, len(b)):
        if b[i - 1] < 1e-3 <= b[i]:
            x0, x1, y0, y1 = jsr[i - 1], jsr[i], math.log10(b[i - 1]), math.log10(b[i])
            return float(x0 + (-3 - y0) * (x1 - x0) / (y1 - y0))
    return None


def summarise(tasks, k1):
    s = dict(run=ts.RUN, tasks={str(t): ts.TASKS[t] for t in tasks})
    sig = lambda n: math.sqrt(bf.ALPHA * (1 - bf.ALPHA) / n)       # P(det) std at the FAR
    diff_tol = lambda n: 2 * math.sqrt(2) * sig(n)                  # 2 sigma of a difference of two
    # fitness(theta*) >= fitness(theta = 0): white noise is in the family. Tolerance: the
    # P(det) term's noise, plus a 4-sigma BER term (bits per frame treated as independent).
    viol = []
    for t, res in tasks.items():
        for r in res["runs"]:
            n, b0 = r["confirm"]["frames"], r["confirm_theta0"]["ber"]
            tol = r["beta"] * diff_tol(n) + 4 * math.sqrt(max(b0, 1e-9) / (n * 2 * 128))
            if r["fitness_confirm"] < r["fitness_confirm_theta0"] - tol:
                viol.append(dict(task=t, jsr=r["jsr_db"], f=r["fitness_confirm"],
                                 f0=r["fitness_confirm_theta0"], tol=tol))
    s["learned_not_worse_than_white"] = dict(violations=viol, holds=not viol)
    # P1: trained against one-sided power, no P(det) reduction beyond noise at any JSR
    t = task_of("power_one_sided", 1.0)
    if t in tasks:
        runs = tasks[t]["runs"]
        d = [(r["jsr_db"], r["confirm_theta0"]["pdet"]["power_one_sided"][SA]
              - r["confirm"]["pdet"]["power_one_sided"][SA]) for r in runs]
        jsr, worst = max(d, key=lambda x: x[1])
        tol = diff_tol(runs[0]["confirm"]["frames"])
        s["P1_power_unmovable"] = dict(max_pdet_reduction=worst, at_jsr=jsr, tol=tol, holds=worst <= tol)
    # P2: beta = 0 gain at BER 1e-3 over white noise (sps processing gain = the ceiling)
    if 0 in tasks:
        j, pts = row(tasks[0]["runs"])
        _, pts0 = row(tasks[0]["runs"], "confirm_theta0")
        cl, c0 = _cross(j, [p["ber"] for p in pts]), _cross(j, [p["ber"] for p in pts0])
        s["P2_inband_gain"] = dict(jsr_at_1e3_learned=cl, jsr_at_1e3_white=c0,
                                   gain_db=(c0 - cl) if (cl is not None and c0 is not None) else None,
                                   predicted_up_to_db=10 * math.log10(lk.LINK["sps"]))
    # P3: confirmed stealthy BER of each task against every detector
    s["P3_stealthy_ber"] = {}
    for t, res in tasks.items():
        if "confirmed" not in res:
            continue
        task = ts.TASKS[t]
        name = "beta0" if task["target"] is None else f"{task['target']}_b{task['beta']:g}"
        s["P3_stealthy_ber"][name] = {d: (res["confirmed"].get(d) or {}).get(SA) for d in DETS}
    # seed check: same target/beta/JSR, another seed -> same P(det) on the target
    last = len(ts.TASKS) - 1
    if last in tasks:
        chk = []
        for r in tasks[last]["runs"]:
            ref = next((x for x in tasks.get(task_of(r["target"], r["beta"]), {}).get("runs", [])
                        if x["jsr_db"] == r["jsr_db"]), None)
            if ref is None:
                continue
            dp = r["confirm"]["pdet"][r["target"]][SA] - ref["confirm"]["pdet"][r["target"]][SA]
            chk.append(dict(target=r["target"], jsr=r["jsr_db"], d_pdet_target=dp,
                            d_ber=r["confirm"]["ber"] - ref["confirm"]["ber"],
                            agrees=abs(dp) <= diff_tol(r["confirm"]["frames"])))
        s["seed_check"] = dict(runs=chk, holds=all(c["agrees"] for c in chk) if chk else None)
    s["matched_ber"] = {det: {lab: {str(b): dict(zip(("ber", "jsr"), matched_ber(pts, b))) for b in BUDGETS}
                              for lab, _, _, pts in frontier_rows(tasks, k1, det)} for det in DETS}
    s["shapes"] = {str(t): [dict(jsr=r["jsr_db"], **shape_metrics(r["theta"])) for r in res["runs"]]
                   for t, res in tasks.items()}
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    return s


def main():
    bf._style()
    tasks = load_tasks()
    if not tasks:
        raise SystemExit(f"no task*.json in {OUT} -- run train_shaped.py first")
    k1 = bf.load(1)
    fig_frontier(tasks, k1)
    fig_transfer(tasks)
    fig_shapes(tasks)
    fig_convergence(tasks)
    fig_ber_vs_jsr(tasks, k1)
    s = summarise(tasks, k1)
    print(json.dumps({k: v for k, v in s.items() if k not in ("tasks", "shapes", "matched_ber")}, indent=2))
    for det, rows in s["matched_ber"].items():
        print(f"== {det}: most BER at P(det) <= " + " / ".join(map(str, BUDGETS)))
        for lab, d in rows.items():
            print(f"  {lab:<44} " + "  ".join(f"{d[str(b)]['ber']:.1e}" for b in BUDGETS))


if __name__ == "__main__":
    main()
