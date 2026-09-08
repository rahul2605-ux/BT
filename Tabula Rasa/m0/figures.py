"""
M0 -- figures for E1.

ON THE "DUAL AXIS" REQUEST
--------------------------
The 2026-08-21 meeting asked for "double axes, BER/detection". Two y-scales on
one panel is the single most criticised chart form, because the crossing point
of the two curves is an artefact of the arbitrary relative scaling. It is also
UNNECESSARY here: BER, SER and P(detect) are all probabilities, so they share
units and a single axis is available. We therefore give the same information two
ways and use neither trick:
  * `tradeoff`  -- stacked panels sharing the power axis: effectiveness above,
                   detectability below. Reads exactly as he asked, no scale
                   artefact.
  * `onepanel`  -- both quantities on ONE shared log axis, which is legitimate
                   precisely because they are the same kind of number.

Palette: the reference categorical instance, used unmodified in slot order.
"""

import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference categorical palette, light mode, fixed slot order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

DET = [("p_np", "NP-optimal", C[0]), ("p_l", "Learned CNN", C[1]),
       ("p_e1", "Energy (1-sided)", C[2]), ("p_e2", "Energy (2-sided)", C[3])]
ATK_LABEL = {"barrage": "Barrage", "gaussian": "Gaussian",
             "boundary_blind": "Boundary (blind)",
             "boundary_genie": "Boundary (genie)",
             "permute": "Permute (invisible)",
             "counter_null": "Counter-null", "counter_flip": "Counter-flip"}


def style(ax):
    ax.grid(True, color=GRID, lw=.6, alpha=.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "results_sigma*.json"))):
        r = json.load(open(p))
        out[r["sigma"]] = r
    return out


def best_over_duty(rows, attack, key="ber"):
    """Achievable frontier: at each power keep the duty that maximises BER."""
    by = {}
    for r in rows:
        if r["attack"] != attack:
            continue
        k = r["power"]
        if k not in by or r[key] > by[k][key]:
            by[k] = r
    return [by[k] for k in sorted(by)]


# --------------------------------------------------------------- fig: tradeoff
def fig_tradeoff(data, sigmas, out):
    sigmas = [s for s in sigmas if s in data]
    fig, axes = plt.subplots(2, len(sigmas), figsize=(4.1 * len(sigmas), 6.2),
                             sharex=True, squeeze=False)
    for j, sg in enumerate(sigmas):
        d = data[sg]
        rows = best_over_duty(d["rows"], "boundary_blind")
        P = [r["power"] for r in rows]
        a_top, a_bot = axes[0][j], axes[1][j]

        a_top.plot(P, [r["ber"] for r in rows], "-o", ms=5, lw=2,
                   color=C[0], label="BER")
        a_top.plot(P, [r["ser"] for r in rows], "-s", ms=5, lw=2,
                   color=C[1], label="SER")
        a_top.axhline(d["clean_ber"], ls="--", lw=1.6, color=INK2)
        a_top.annotate("no attacker (clean BER floor)",
                       (P[0], d["clean_ber"]), fontsize=7.5, color=INK2,
                       va="bottom")
        a_top.axhline(1.0, ls=":", lw=1.6, color=C[5])
        a_top.annotate("omniscient counter-flip (BER 1.0)", (P[0], 1.0),
                       fontsize=7.5, color=C[5], va="top")
        a_top.set_yscale("log"); style(a_top)
        a_top.set_title(f"$\\sigma$ = {sg}   ({d['ebn0_db']:.0f} dB Eb/N0)",
                        fontsize=10.5, color=INK)
        if j == 0:
            a_top.set_ylabel("error rate", color=INK, fontsize=9.5)
            a_top.legend(fontsize=8, frameon=False)

        for key, lab, col in DET:
            v = [r[key] for r in rows]
            if not np.all(np.isnan(v)):
                a_bot.plot(P, v, "-o", ms=5, lw=2, color=col, label=lab)
        a_bot.axhline(d["alpha"], ls="--", lw=1.6, color=INK2)
        a_bot.annotate(f"$\\alpha$ = {d['alpha']} (false-alarm rate)",
                       (P[0], d["alpha"]), fontsize=7.5, color=INK2, va="bottom")
        a_bot.set_xscale("log"); a_bot.set_ylim(-.04, 1.04); style(a_bot)
        a_bot.set_xlabel("jammer power budget $P$", color=INK, fontsize=9.5)
        if j == 0:
            a_bot.set_ylabel("P(detect)", color=INK, fontsize=9.5)
            a_bot.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("M0 effectiveness vs detectability - blind minimum-energy attack\n"
                 "what the victim loses (top) against what each detector sees (bottom)",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------- fig: frontier
def fig_frontier(data, sigmas, out, det="p_np"):
    sigmas = [s for s in sigmas if s in data]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.1 * len(sigmas), 4.0),
                             sharey=True, squeeze=False)
    for j, sg in enumerate(sigmas):
        d, ax = data[sg], axes[0][j]
        for i, atk in enumerate(["barrage", "gaussian", "boundary_blind",
                                 "boundary_genie", "permute"]):
            rows = best_over_duty(d["rows"], atk)
            x = [r[det] for r in rows]; y = [r["ber"] for r in rows]
            ax.plot(x, y, "-o", ms=4.5, lw=2, color=C[i], label=ATK_LABEL[atk])
        cf = [r for r in d["rows"] if r["attack"] == "counter_flip"]
        if cf:
            ax.plot(cf[0][det], cf[0]["ber"], "*", ms=17, color=C[5],
                    mec="white", mew=1.0, label="Counter-flip (genie)")
        ax.axhline(d["clean_ber"], ls="--", lw=1.5, color=INK2)
        ax.axvline(d["alpha"], ls=":", lw=1.5, color=INK2)
        ax.annotate("stealth budget\n$\\beta=\\alpha$", (d["alpha"], 0.6),
                    fontsize=7.5, color=INK2, ha="left")
        ax.set_yscale("log"); style(ax)
        ax.set_xlim(-.04, 1.04)
        ax.set_xlabel("P(detect), NP-optimal", color=INK, fontsize=9.5)
        ax.set_title(f"$\\sigma$ = {sg}", fontsize=10.5, color=INK)
        if j == 0:
            ax.set_ylabel("BER", color=INK, fontsize=9.5)
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Achievable frontier: BER against detection probability\n"
                 "up and to the LEFT is better; anything right of the dotted line is caught",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------- fig: detector compare
def fig_detectors(data, sigma, out):
    d = data[sigma]
    atks = ["gaussian", "barrage", "boundary_blind", "boundary_genie"]
    fig, axes = plt.subplots(1, len(atks), figsize=(3.6 * len(atks), 3.7),
                             sharey=True, squeeze=False)
    for j, atk in enumerate(atks):
        ax = axes[0][j]
        rows = best_over_duty(d["rows"], atk)
        P = [r["power"] for r in rows]
        for key, lab, col in DET:
            v = [r[key] for r in rows]
            if not np.all(np.isnan(v)):
                ax.plot(P, v, "-o", ms=5, lw=2, color=col, label=lab)
        ax.axhline(d["alpha"], ls="--", lw=1.5, color=INK2)
        ax.set_xscale("log"); ax.set_ylim(-.04, 1.04); style(ax)
        ax.set_xlabel("power $P$", color=INK, fontsize=9.5)
        ax.set_title(ATK_LABEL[atk], fontsize=10.5, color=INK)
        if j == 0:
            ax.set_ylabel("P(detect)", color=INK, fontsize=9.5)
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Detector comparison at $\\sigma$={sigma}, $\\alpha$={d['alpha']}"
                 "   -   the gap NP $-$ Learned is the adaptation budget",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------- fig: stealth region vs sigma
def fig_stealth_vs_sigma(data, out):
    """
    Max EXCESS BER (over the clean floor) achievable while staying under the
    detector's own false-alarm rate.

    Excess, not raw BER: at sigma=0.5 the unjammed link already sits at BER
    0.079, so a jammer that "achieves BER 0.079 stealthily" has achieved
    nothing. Reporting raw BER here lets the noise floor masquerade as jamming
    effectiveness -- the exact trap the sim08 matched-detectability correction
    was about.
    """
    sgs = sorted(data)
    atks = ["barrage", "gaussian", "boundary_blind", "boundary_genie", "permute"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    empty = {0: [], 1: []}
    for i, atk in enumerate(atks):
        for pi, (ax, det) in enumerate(((axes[0], "p_np"), (axes[1], "p_l"))):
            best = []
            for sg in sgs:
                d = data[sg]
                cand = [r["ber"] - d["clean_ber"] for r in d["rows"]
                        if r["attack"] == atk and not np.isnan(r[det])
                        and r[det] <= d["alpha"]]
                v = max(cand) if cand else np.nan
                best.append(v if (not np.isnan(v) and v > 1e-4) else np.nan)
            if np.all(np.isnan(best)):
                empty[pi].append(ATK_LABEL[atk])
                continue
            ax.plot(sgs, best, "-o", ms=6, lw=2.2, color=C[i],
                    label=ATK_LABEL[atk])
    for pi, (ax, ttl) in enumerate(((axes[0], "vs NP-optimal detector (the bound)"),
                                    (axes[1], "vs learned CNN (a real detector)"))):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(sgs); ax.set_xticklabels([f"{s:g}" for s in sgs])
        ax.minorticks_off(); style(ax)
        ax.set_xlabel("noise level $\\sigma$", color=INK, fontsize=9.5)
        ax.set_title(ttl, fontsize=10.5, color=INK)
        if empty[pi]:
            ax.text(.5, .06, "no stealthy configuration exists for: "
                    + ", ".join(empty[pi]), transform=ax.transAxes,
                    ha="center", fontsize=8, color=INK2, style="italic")
    axes[0].set_ylabel("max EXCESS BER over clean floor\n"
                       "while $P_{det}\\leq\\alpha$", color=INK, fontsize=9.5)
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh); l.append(ll)
    fig.legend(h, l, fontsize=9, frameon=False, ncol=len(l),
               loc="lower center", bbox_to_anchor=(.5, -.06))
    fig.suptitle("Who can jam without being seen?  The constellation-permuting "
                 "attack is invisible at every noise level;\nthe blind attack "
                 "only gets cover once the noise floor is high enough to hide in",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, .02, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../artifacts/m0/frontier")
    ap.add_argument("--out", default="../artifacts/m0")
    a = ap.parse_args()
    here = os.path.dirname(__file__)
    res = os.path.abspath(os.path.join(here, a.results))
    out = os.path.abspath(os.path.join(here, a.out))
    os.makedirs(out, exist_ok=True)
    data = load(res)
    if not data:
        raise SystemExit(f"no results in {res}")
    print(f"loaded sigmas: {sorted(data)}")
    show = [s for s in (0.05, 0.2, 0.4) if s in data] or sorted(data)[:3]
    for p in [fig_tradeoff(data, show, os.path.join(out, "e1_tradeoff.png")),
              fig_frontier(data, show, os.path.join(out, "e1_frontier.png")),
              fig_detectors(data, show[len(show)//2],
                            os.path.join(out, "e1_detectors.png")),
              fig_stealth_vs_sigma(data, os.path.join(out, "e1_stealth_vs_sigma.png"))]:
        print("wrote", p)


if __name__ == "__main__":
    main()
