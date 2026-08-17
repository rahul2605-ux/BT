"""
sim08 milestone-2 follow-up — the MATCHED-DETECTABILITY frontier (step 1).

Pure post-processing of the existing full-suite frontier sweep
(artifacts/sim08/frontier/results.json — no GPU / no cluster). It answers the
question the m1 "channel-aware +70%" metric could NOT: at MATCHED suite
detectability P(det), does a channel-aware jammer actually beat a blind one?

Why this is the honest metric. The m1 headline compared BER at matched jammer
*config* (same n_active, power). But concentrating power on high-gain
subcarriers raises BER AND the spectrogram signature together — channel-aware
is louder to the CNN. The only fair comparison holds the thing the defender
sees (P(suite)) fixed and asks who causes more BER underneath it.

For each strategy and SNR we build the ACHIEVABLE FRONTIER
    BER*(beta) = max BER over that strategy's configs with p_suite <= beta,
a monotone step function of the detectability budget beta. Comparing the two
frontiers = "at matched detectability, who wins."

Two reference operating points are drawn instead of the single p<=0.5 convention:
  * FAR-matched: beta = the suite's false-alarm rate on CLEAN traffic at that
    SNR (a jammer this quiet is statistically indistinguishable from the
    detector's own baseline — the only defensible "stealthy" point).
  * p<=0.5: the frontier-characterization convention (NOT operational stealth;
    0.5^k -> detected over k frames), drawn for continuity with the m2 figure.

Outputs:
  artifacts/sim08/frontier/matched_detectability.png  — per-SNR BER-vs-P(suite)
      scatter + achievable-frontier step curves (blind vs channel-aware),
      clean floor, FAR-matched and 0.5 reference lines.
  artifacts/sim08/frontier/matched_summary.json       — the numbers.
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATS = [("sparse_blind", "blind", "steelblue"),
          ("sparse_channelaware", "channel-aware", "crimson")]
# jammer classes that are in-scope for "can a sparse jammer stay stealthy":
# broadband is included in the envelope too (it is a valid jammer), but the
# blind-vs-channel-aware comparison is the sparse pair above.


def achievable_frontier(pts):
    """pts: list of (p_suite, ber). Returns (beta_grid, best_ber) step curve =
    max BER achievable at detectability budget <= beta, evaluated at each
    distinct p_suite (monotone non-decreasing)."""
    if not pts:
        return np.array([]), np.array([])
    pts = sorted(pts, key=lambda t: t[0])
    betas = np.array([p for p, _ in pts])
    bers = np.array([b for _, b in pts])
    best = np.maximum.accumulate(bers)   # cumulative max BER as budget grows
    return betas, best


def max_ber_within(pts, budget):
    c = [b for p, b in pts if p <= budget + 1e-12]
    return max(c) if c else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="../artifacts/sim08/frontier/results.json")
    ap.add_argument("--out", default="../artifacts/sim08/frontier")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    rows = json.load(open(os.path.join(here, args.results)))["rows"]
    out_dir = os.path.join(here, args.out)

    ebnos = sorted({r["ebno_db"] for r in rows})

    def strat_pts(strat, e):
        return [(r["p_suite"], r["ber_mean"]) for r in rows
                if r["strategy"] == strat and abs(r["ebno_db"] - e) < 1e-9]

    def clean_row(e):
        return next(r for r in rows if r["strategy"] == "clean"
                    and abs(r["ebno_db"] - e) < 1e-9)

    # ---- figure: one panel per SNR -----------------------------------------
    n = len(ebnos)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow),
                             squeeze=False)
    summary = {}

    for i, e in enumerate(ebnos):
        ax = axes[i // ncol][i % ncol]
        cr = clean_row(e)
        floor = cr["ber_mean"]
        far = cr["p_suite"]          # suite false-alarm rate on clean = honest budget
        summary[str(e)] = {"clean_floor": floor, "suite_far": far,
                           "strategies": {}}

        for strat, lbl, col in STRATS:
            pts = strat_pts(strat, e)
            ax.scatter([p for p, _ in pts], [b for _, b in pts],
                       c=col, s=40, edgecolor="k", linewidth=0.3, alpha=0.85,
                       label=f"{lbl} (configs)", zorder=3)
            bx, by = achievable_frontier(pts)
            if len(bx):
                ax.step(bx, by, where="post", color=col, lw=1.8, alpha=0.9,
                        zorder=2)
            summary[str(e)]["strategies"][lbl] = {
                "ber_at_far": max_ber_within(pts, far),
                "ber_at_0.1": max_ber_within(pts, 0.1),
                "ber_at_0.5": max_ber_within(pts, 0.5)}

        ax.axhline(floor, color="black", ls="--", lw=1,
                   label=f"clean floor ({floor:.4f})")
        ax.axvline(far, color="dimgray", ls=":", lw=1.4,
                   label=f"FAR-matched ({far:.2f})")
        ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7,
                   label="p<=0.5 convention")
        ax.set_yscale("log")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("P(detect) — CNN+energy suite")
        ax.set_ylabel("BER")
        ax.set_title(f"Eb/N0 = {e:g} dB")
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    # hide any unused panels
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("sim08 — matched-detectability frontier: BER a jammer causes "
                 "vs suite P(detect)\n(achievable frontier = max BER at "
                 "detectability budget; upper-LEFT = stealthy AND effective)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "matched_detectability.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "matched_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console headline ---------------------------------------------------
    print("\n=== MATCHED-DETECTABILITY: max BER at equal suite P(detect) ===")
    print(f"{'Eb/N0':>6} | {'floor':>7} | {'FAR':>5} | "
          f"{'blind@FAR':>9} {'ca@FAR':>7} | "
          f"{'blind@.5':>8} {'ca@.5':>7} | ca-gain@.5")
    for e in ebnos:
        s = summary[str(e)]["strategies"]
        far = summary[str(e)]["suite_far"]
        b5, c5 = s["blind"]["ber_at_0.5"], s["channel-aware"]["ber_at_0.5"]
        gain = (c5 - b5) / b5 * 100 if b5 == b5 and b5 > 0 else float("nan")
        print(f"{e:>6g} | {summary[str(e)]['clean_floor']:>7.4f} | {far:>5.2f} | "
              f"{s['blind']['ber_at_far']:>9.4f} {s['channel-aware']['ber_at_far']:>7.4f} | "
              f"{b5:>8.4f} {c5:>7.4f} | {gain:>+6.1f}%")
    print(f"\nWrote matched_detectability.png + matched_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
