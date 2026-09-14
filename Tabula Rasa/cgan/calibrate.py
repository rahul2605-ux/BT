"""
C2 -- calibrate the link parameters Zhou et al. 2025 leave unstated, against
their own Fig. 6 (digitised in C0).

    sbatch submit_calibrate.sh      # -> ../artifacts/cgan/calibration.json, c2_calibration.png

Two stages, in the order the information allows, each scored TWO ways:

    rms      RMS error in log10(BER) over the usable digitised points (present,
             not occluded, BER >= 1e-6) -- how well the whole curve is matched
    cross    |JSR at BER 1e-3 (ours) - JSR at BER 1e-3 (paper)| -- how well the
             point the success bar is read at (README §2.10) is matched

The first run (job 2259150) showed the two disagree. The paper's Noise curve is
not a single Q-function (C0), so no link matches it everywhere: the rms-best
oversampling factor misses the 1e-3 crossing by 2.4 dB. This script therefore
does not pick a winner. It carries every rms-best and cross-best candidate
through both stages and reports them side by side for the user to choose at the
C2 checkpoint.

1. NOISE CURVE -> (sps, band). Closed form (link.py). Under white noise the
   BER does not depend on the pulse at all, so the pulse is NOT identified here.
   The best constant-processing-gain fit is reported as the floor any link
   choice can reach, and so is the gain that reproduces the paper's crossing
   exactly.

2. OPTIMAL CURVE -> (pulse, sync), per candidate sps. Monte Carlo of the
   matched-modulation jammer for every pulse x synchronisation variant.
   Zero-error points enter at their 95% upper bound 3/bits; a crossing computed
   through one is a lower bound and flagged.
"""

import argparse
import json
import os
import time

import numpy as np

import link as lk
import jammers

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "cgan")

SPS_GRID = [2, 4, 8, 16]
PULSES = ["rect", "rrc0.25", "rrc0.35", "rrc0.5"]
BANDS = jammers.BANDS
SYNCS = jammers.SYNC_VARIANTS
TARGET = 1e-3

# Categorical slots 1-3 of the dataviz reference palette, fixed per entity across
# every cgan figure: Optimal = blue, Noise = orange, GAN = aqua.
C_OPT, C_NOISE, C_GAN = "#2a78d6", "#eb6834", "#1baf7a"
C_OTHER, INK, INK2, GRID = "#8a8984", "#1f1f1e", "#5f5e5a", "#e4e3de"


def rms_log(model, paper, mask):
    m = np.log10(np.maximum(np.asarray(model, float)[mask], 1e-300))
    p = np.log10(np.asarray([paper[i] for i in np.where(mask)[0]], float))
    return float(np.sqrt(np.mean((m - p) ** 2))), (m - p).tolist()


def cross_err(ours, paper_cross):
    return None if ours is None else abs(ours - paper_cross)


def stage1_noise(jsr, paper, flags, paper_cross):
    mask = lk.usable_paper_points(paper["noise"], flags["noise"])
    rows = []
    for sps in SPS_GRID:
        for pulse in PULSES:
            L = lk.Link(sps=sps, pulse=pulse)
            for band in BANDS:
                ber = L.ber_noise_jammer(jsr, band=band)
                rms, res = rms_log(ber, paper["noise"], mask)
                cross = lk.jsr_at_ber(jsr, ber.tolist(), TARGET)
                rows.append(dict(sps=sps, pulse=pulse, band=band, kappa=L.kappa(band),
                                 rms_log10=rms, residuals_log10=res, ber=ber.tolist(),
                                 jsr_at_1e3=cross, cross_err_db=cross_err(cross, paper_cross)))

    # Constant processing gain g, free: BER = Q(sqrt(g / (1/SNR + JSR))).
    snr = 10 ** (lk.SNR_DB / 10)
    j_lin = 10 ** (jsr / 10)
    fine = np.arange(-10.0, 10.001, 0.05)
    best_g = cross_g = None
    for g_db in np.arange(-5.0, 20.0, 0.01):
        ber = lk.q_function(np.sqrt(10 ** (g_db / 10) / (1 / snr + j_lin)))
        rms, res = rms_log(ber, paper["noise"], mask)
        if best_g is None or rms < best_g["rms_log10"]:
            best_g = dict(g_db=float(g_db), rms_log10=rms, residuals_log10=res, ber=ber.tolist())
        # The gain whose curve crosses 1e-3 exactly where the paper's does.
        ber_f = lk.q_function(np.sqrt(10 ** (g_db / 10) / (1 / snr + 10 ** (fine / 10))))
        c = lk.jsr_at_ber(fine, ber_f.tolist(), TARGET)
        if c is not None and (cross_g is None or abs(c - paper_cross) < cross_g["cross_err_db"]):
            cross_g = dict(g_db=float(g_db), equivalent_sps=float(10 ** (g_db / 10)),
                           rms_log10=rms, cross_err_db=abs(c - paper_cross), ber=ber.tolist())
    return rows, best_g, cross_g, mask


def candidates(noise_rows):
    """Every (sps, band) that is best on rms or best on the crossing."""
    by_rms = min(noise_rows, key=lambda r: r["rms_log10"])
    by_cross = min((r for r in noise_rows if r["cross_err_db"] is not None),
                   key=lambda r: r["cross_err_db"])
    out = []
    for why, r in (("rms-best", by_rms), ("cross-best", by_cross)):
        key = (r["sps"], r["band"])
        found = next((c for c in out if (c["sps"], c["band"]) == key), None)
        if found:
            found["why"] += " + " + why
        else:
            out.append(dict(sps=r["sps"], band=r["band"], why=why, noise=r))
    return out


def stage2_optimal(jsr, paper, flags, paper_cross, sps, band, max_bits):
    mask = lk.usable_paper_points(paper["optimal"], flags["optimal"])
    rows = []
    for pulse in PULSES:
        L = lk.Link(sps=sps, pulse=pulse)
        for sync in SYNCS:
            t0 = time.time()
            pts = [lk.measure_ber(L, jammers.make("optimal", float(j), sync=sync),
                                  max_bits=max_bits) for j in jsr]
            zero = [p["errors"] == 0 for p in pts]
            ber = [p["upper"] if z else p["ber"] for p, z in zip(pts, zero)]
            rms, res = rms_log(ber, paper["optimal"], mask)
            cross = lk.jsr_at_ber(jsr, ber, TARGET)
            # Is the left bracket of the crossing a zero-error point?
            lower_bound = False
            if cross is not None:
                i = int(np.searchsorted(jsr, cross)) - 1
                lower_bound = bool(0 <= i < len(zero) and zero[i])
            rows.append(dict(sps=sps, pulse=pulse, band=band, sync=sync, rms_log10=rms,
                             residuals_log10=res, ber=ber,
                             errors=[p["errors"] for p in pts], bits=[p["bits"] for p in pts],
                             zero_error=zero, jsr_at_1e3=cross,
                             jsr_at_1e3_is_lower_bound=lower_bound,
                             cross_err_db=cross_err(cross, paper_cross)))
            c = "never" if cross is None else f"{'>=' if lower_bound else ''}{cross:.2f}"
            print(f"  sps {sps:>2} {pulse:<8} {sync:<13} rms {rms:.3f}  JSR@1e-3 {c}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return rows, mask


def figure(jsr, paper, noise_rows, best_g, cross_g, cands, noise_mask, opt_mask, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    n = 1 + len(cands)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)

    def paper_curve(ax, key, colour, mask, label):
        b = np.array([np.nan if v is None else v for v in paper[key]], float)
        ax.plot(jsr, b, ls="--", lw=1.2, color=colour, alpha=0.6)
        ax.plot(jsr[mask], b[mask], ls="none", marker="o", ms=7, mfc="white", mec=colour,
                mew=1.8, label=label)
        ax.plot(jsr[~mask], b[~mask], ls="none", marker="o", ms=7, mfc="white",
                mec=C_OTHER, mew=1.2, label="paper point, not fitted")

    ax = axes[0]
    paper_curve(ax, "noise", C_NOISE, noise_mask, "Zhou Fig. 6 Noise")
    styles = ["-", (0, (5, 2)), "-."]
    for c, ls in zip(cands, styles):
        r = c["noise"]
        ax.plot(jsr, r["ber"], color=C_NOISE, lw=2, ls=ls,
                label=f"sps={r['sps']} {r['band']}: rms {r['rms_log10']:.2f} dec, "
                      f"1e-3 off {r['cross_err_db']:.2f} dB")
    ax.plot(jsr, best_g["ber"], color=INK2, lw=1.4, ls=":",
            label=f"any Q-function, best: g={best_g['g_db']:.1f} dB, rms {best_g['rms_log10']:.2f}")
    ax.set_title("Noise: closed form vs paper", loc="left", fontsize=11)

    sync_ls = {"locked": "-", "random_phase": "-.", "async": (0, (5, 2))}
    for ax, c in zip(axes[1:], cands):
        paper_curve(ax, "optimal", C_OPT, opt_mask, "Zhou Fig. 6 Optimal")
        rows = c["optimal"]
        best_by_sync = {s: min((r for r in rows if r["sync"] == s), key=lambda r: r["rms_log10"])
                        for s in SYNCS}
        best = min(best_by_sync.values(), key=lambda r: r["rms_log10"])
        for s, r in best_by_sync.items():
            hi = r is best
            ce = "never" if r["cross_err_db"] is None else f"{r['cross_err_db']:.2f} dB"
            ax.plot(jsr, r["ber"], color=C_OPT if hi else C_OTHER, lw=2 if hi else 1.4,
                    ls=sync_ls[s], label=f"{s}, {r['pulse']}: rms {r['rms_log10']:.2f}, 1e-3 off {ce}")
            z = np.array(r["zero_error"])
            if z.any():
                ax.plot(jsr[z], np.array(r["ber"])[z], ls="none", marker="v", ms=6,
                        color=C_OPT if hi else C_OTHER)
        ax.set_title(f"Optimal, Monte Carlo at sps={c['sps']} ({c['why']})", loc="left",
                     fontsize=11)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(1e-11, 1)
        ax.set_xlabel("Jamming-to-signal ratio (dB)")
        ax.axhline(TARGET, color=INK2, lw=0.8, ls="--")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("Bit error rate   (▼ = zero errors, plotted at 95% bound)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bits", type=float, default=2e7,
                    help="Monte-Carlo cap per point in stage 2")
    args = ap.parse_args()

    lk.setup(seed=2026)
    os.makedirs(ART, exist_ok=True)
    jsr, paper, flags = lk.load_paper_fig6()
    paper_cross = {k: lk.jsr_at_ber(jsr, paper[k], TARGET) for k in ("optimal", "gan", "noise")}
    print("paper JSR@1e-3:", {k: round(v, 2) for k, v in paper_cross.items()})

    print("\nstage 1: noise curve, closed form", flush=True)
    noise_rows, best_g, cross_g, noise_mask = stage1_noise(jsr, paper, flags, paper_cross["noise"])
    print(f"  best constant gain on rms:     g = {best_g['g_db']:.2f} dB, rms {best_g['rms_log10']:.3f}")
    print(f"  gain matching the 1e-3 point:  g = {cross_g['g_db']:.2f} dB "
          f"(= sps {cross_g['equivalent_sps']:.2f} under white noise), rms {cross_g['rms_log10']:.3f}")
    for r in sorted(noise_rows, key=lambda r: r["rms_log10"])[:10]:
        ce = "never" if r["cross_err_db"] is None else f"{r['cross_err_db']:.2f}"
        print(f"  sps {r['sps']:>2} {r['pulse']:<8} {r['band']:<6} kappa {r['kappa']:6.3f}  "
              f"rms {r['rms_log10']:.3f}  1e-3 off {ce} dB")
    cands = candidates(noise_rows)
    for c in cands:
        same = [r for r in noise_rows if (r["sps"], r["band"]) == (c["sps"], c["band"])]
        c["pulse_identified_by_noise_curve"] = (
            max(r["rms_log10"] for r in same) - min(r["rms_log10"] for r in same) > 1e-3)

    for c in cands:
        print(f"\nstage 2: optimal curve, Monte Carlo at sps={c['sps']}, band={c['band']} "
              f"({c['why']})", flush=True)
        c["optimal"], opt_mask = stage2_optimal(jsr, paper, flags, paper_cross["optimal"],
                                                c["sps"], c["band"], args.max_bits)

    out = dict(
        paper_jsr_at_1e3=paper_cross,
        noise_fit=dict(best_constant_gain_rms=best_g, constant_gain_matching_crossing=cross_g,
                       ranking=sorted(noise_rows, key=lambda r: r["rms_log10"]),
                       fitted_jsr=jsr[noise_mask].tolist()),
        candidates=[dict(sps=c["sps"], band=c["band"], why=c["why"],
                         pulse_identified_by_noise_curve=c["pulse_identified_by_noise_curve"],
                         noise=c["noise"],
                         optimal_ranking=sorted(c["optimal"], key=lambda r: r["rms_log10"]))
                    for c in cands],
        optimal_fitted_jsr=jsr[opt_mask].tolist(),
        max_bits=args.max_bits, jsr_db=jsr.tolist(), snr_db=lk.SNR_DB,
        selected=None,   # decided with the user at the C2 checkpoint, then recorded here
    )
    with open(os.path.join(ART, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    figure(jsr, paper, noise_rows, best_g, cross_g, cands, noise_mask, opt_mask,
           os.path.join(ART, "c2_calibration.png"))

    print("\nSUMMARY per candidate (best optimal variant by rms, and by crossing):")
    for c in cands:
        n = c["noise"]
        by_rms = min(c["optimal"], key=lambda r: r["rms_log10"])
        crossing = [r for r in c["optimal"] if r["cross_err_db"] is not None]
        by_cross = min(crossing, key=lambda r: r["cross_err_db"]) if crossing else None
        print(f"  sps={c['sps']} {c['band']} [{c['why']}]: noise rms {n['rms_log10']:.3f} dec, "
              f"noise 1e-3 at {n['jsr_at_1e3']:.2f} dB (paper {paper_cross['noise']:.2f})")
        print(f"    optimal by rms:   {by_rms['pulse']} {by_rms['sync']}  rms {by_rms['rms_log10']:.3f}  "
              f"1e-3 at {by_rms['jsr_at_1e3']} (paper {paper_cross['optimal']:.2f})")
        if by_cross:
            print(f"    optimal by cross: {by_cross['pulse']} {by_cross['sync']}  rms "
                  f"{by_cross['rms_log10']:.3f}  1e-3 at {by_cross['jsr_at_1e3']:.2f}"
                  f"{' (lower bound)' if by_cross['jsr_at_1e3_is_lower_bound'] else ''}")
    print(f"wrote {os.path.join(ART, 'calibration.json')}")


if __name__ == "__main__":
    main()
