"""
C6 -- evaluate a trained CGAN against Zhou et al. 2025, Fig. 6 (README §2.10, §3.4).

    sbatch submit_eval.sh                       # run001_G.pt
    sbatch submit_eval.sh --run run002 --max-bits 1e9

Zhou's protocol: SNR 30 dB, JSR -10:2:10 dB, jammers noise / optimal / gan, 1e4
symbols x 100 trials. EXTENDED by an error stopping rule (>= 100 bit errors or
max-bits) so the low-BER points are real; zero-error points are plotted at their
95% upper bound. The link is the calibrated one (link.LINK); the optimal jammer
uses the calibrated sync (async).

Reports JSR at BER 1e-3 for each jammer, and the two gaps -- Noise - GAN and
GAN - Optimal -- next to the paper's own (4.44 and 1.31 dB). This is REPORT-ONLY:
there is no pass/fail bar (§2.10, decided at the C2 checkpoint).
"""

import argparse
import json
import os

import numpy as np
import torch

import link as lk
import jammers
import models

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "cgan")

C_OPT, C_NOISE, C_GAN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3de"


def load_generator(run, device):
    ckpt = torch.load(os.path.join(ART, f"{run}_G.pt"), map_location=device, weights_only=False)
    G = models.Generator(n_classes=ckpt["n_classes"], seg_len=ckpt["seg_len"]).to(device)
    G.load_state_dict(ckpt["state_dict"]); G.eval()
    return G, ckpt


def sweep(L, jsr_db, jammer_fn, snr_db, min_errors, max_bits):
    rows = []
    for j in jsr_db:
        r = lk.measure_ber(L, jammer_fn(float(j)), snr_db=snr_db,
                           min_errors=min_errors, max_bits=max_bits)
        r["zero"] = r["errors"] == 0
        r["plot"] = r["upper"] if r["zero"] else r["ber"]
        rows.append(r)
        tag = f">={r['upper']:.1e} (0 err)" if r["zero"] else f"{r['ber']:.2e}"
        print(f"  JSR {j:+5.1f} dB  BER {tag:>16}  ({r['errors']} err / {r['bits']:.1e} bits)",
              flush=True)
    return rows


def figure(jsr, ours, paper, pflags, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    order = [("optimal", C_OPT, "Optimal"), ("noise", C_NOISE, "Noise"), ("gan", C_GAN, "GAN")]
    for key, c, lab in order:
        b = np.array([r["plot"] for r in ours[key]])
        zero = np.array([r["zero"] for r in ours[key]])
        ax.plot(jsr, b, color=c, lw=2, label=f"{lab} (ours)")
        if zero.any():
            ax.plot(jsr[zero], b[zero], ls="none", marker="v", ms=6, color=c)
        pb = np.array([np.nan if v is None else v for v in paper[key]], float)
        vis = np.array(["occluded" not in fl for fl in pflags[key]])
        ax.plot(jsr[vis], pb[vis], ls="none", marker="o", ms=6, mfc="white", mec=c, mew=1.5,
                label=f"{lab} (Zhou Fig. 6)")
    ax.axhline(1e-3, color=INK2, lw=0.8, ls="--")
    ax.text(9.5, 1.3e-3, "BER = 1e-3", ha="right", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-11, 1)
    ax.set_xlabel("Jamming-to-signal ratio (dB)"); ax.set_ylabel("Bit error rate")
    ax.set_title("CGAN reproduction vs Zhou 2025 Fig. 6 (QPSK, SNR 30 dB)", loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run001")
    ap.add_argument("--min-errors", type=int, default=100)
    ap.add_argument("--max-bits", type=float, default=1e9)
    args = ap.parse_args()

    device = lk.setup(seed=7)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    G, ckpt = load_generator(args.run, device)
    jsr, paper, pflags = lk.load_paper_fig6()

    jammer_fns = {
        "noise": lambda j: jammers.make("noise", j, band=lk.LINK["noise_band"]),
        "optimal": lambda j: jammers.make("optimal", j, sync=lk.LINK["optimal_sync"]),
        "gan": lambda j: jammers.make("gan", j, G=G, scale=ckpt["scale"],
                                      z_dim=models.Z_DIM, seg_len=ckpt["seg_len"]),
    }
    ours = {}
    for name, fn in jammer_fns.items():
        print(f"\n{name}:", flush=True)
        ours[name] = sweep(L, jsr, fn, lk.SNR_DB, args.min_errors, args.max_bits)

    # Crossing uses the plotted values, so a zero-error point enters at its 95%
    # upper bound. Per link.jsr_at_ber that makes the result a LOWER BOUND on the
    # true crossing JSR when the left bracket had zero errors; `cross_is_bound`
    # flags those. (Excluding zero-error points instead returns None whenever the
    # crossing sits just past one, which hid the Optimal crossing in run001.)
    cross, cross_is_bound = {}, {}
    for k in jammer_fns:
        cross[k] = lk.jsr_at_ber(jsr, [r["plot"] for r in ours[k]])
        i = None if cross[k] is None else int(np.searchsorted(jsr, cross[k])) - 1
        cross_is_bound[k] = bool(i is not None and 0 <= i < len(ours[k]) and ours[k][i]["zero"])
    paper_cross = {k: lk.jsr_at_ber(jsr, paper[k]) for k in jammer_fns}

    def gap(a, b, d):
        return None if (d[a] is None or d[b] is None) else round(d[a] - d[b], 2)

    report = dict(
        run=args.run, snr_db=lk.SNR_DB, jsr_db=jsr.tolist(), link=lk.LINK,
        ber={k: [r["plot"] for r in ours[k]] for k in jammer_fns},
        zero_error={k: [r["zero"] for r in ours[k]] for k in jammer_fns},
        errors={k: [r["errors"] for r in ours[k]] for k in jammer_fns},
        bits={k: [r["bits"] for r in ours[k]] for k in jammer_fns},
        jsr_at_1e3={k: (None if v is None else round(v, 2)) for k, v in cross.items()},
        jsr_at_1e3_is_lower_bound=cross_is_bound,
        paper_jsr_at_1e3={k: round(v, 2) for k, v in paper_cross.items()},
        gaps_ours=dict(noise_minus_gan=gap("noise", "gan", cross),
                       gan_minus_optimal=gap("gan", "optimal", cross)),
        gaps_paper=dict(noise_minus_gan=gap("noise", "gan", paper_cross),
                        gan_minus_optimal=gap("gan", "optimal", paper_cross)),
    )
    with open(os.path.join(ART, f"{args.run}_ber_vs_jsr.json"), "w") as f:
        json.dump(report, f, indent=2)
    figure(jsr, ours, paper, pflags, os.path.join(ART, f"{args.run}_ber_vs_jsr.png"))

    print("\n--- REPORT (no pass/fail; §2.10) ---")
    print(f"JSR at BER 1e-3:  ours {report['jsr_at_1e3']}")
    print(f"                 paper {report['paper_jsr_at_1e3']}")
    print(f"Noise - GAN:      ours {report['gaps_ours']['noise_minus_gan']} dB  "
          f"(paper {report['gaps_paper']['noise_minus_gan']} dB)")
    print(f"GAN - Optimal:    ours {report['gaps_ours']['gan_minus_optimal']} dB  "
          f"(paper {report['gaps_paper']['gan_minus_optimal']} dB)")
    print(f"wrote {args.run}_ber_vs_jsr.json, {args.run}_ber_vs_jsr.png to {ART}")


if __name__ == "__main__":
    main()
