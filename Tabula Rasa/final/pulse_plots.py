"""
Final experiment -- where in time the jammers put their energy (user, 2026-09-30). New file.

    sbatch submit_pulse.sh base                    # draws frames (Sionna) and plots
    python pulse_plots.py --env base --plot-only   # login node: replot from the saved samples

Every jammer is drawn at the JSR where it breaks 50 % of frames (li_figures.matched on
eval_li.json, argmax decision), on fresh frames of the environment. For each, e_k = |MF(j)_k|^2 is
the jammer's energy at symbol k's decision point, and

    p_eff = (mean e_k)^2 / mean(e_k^2)

its effective duty cycle: exactly p for an on/off jammer with constant on-energy, so it puts a
learned jammer on Amuru's p axis (calibrated here on the pulsed family itself). A "hard push" is a
symbol with |MF(j)_k| > 0.5, against 0.71 to the decision boundary.

Figure (li/fig_pulses.png, and Li_<env>/ with --clean): rasters of |MF(j)_k| over frames x symbols
for cnnGAN1 (grey-box, seed r0), Amuru p 0.01 and Li's protocol-aware (p 0.25); hard pushes per frame
against a Poisson count with the same mean; how concentrated each jammer's energy is; and |j[n]|^2
over four frames of cnnGAN1. Numbers: li/pulses.json.
"""

import argparse
import json
import os

import numpy as np

import env as E
import li_figures as LF

N_FRAMES = 512
N_RASTER = 48
HARD = 0.5
CALIB = ["pulsed_p1", "pulsed_p0.5", "pulsed_p0.25", "pulsed_p0.1", "pulsed_p0.05", "pulsed_p0.02", "pulsed_p0.01",
         "noise", "tone", "omniscient_e1"] + [f"cnn_b10_grey_r{r}" for r in range(3)] + [f"cnn_b10_r{r}" for r in range(3)]
SHOW = [("cnn_b10_grey_r0", "cnnGAN1 (grey-box, seed r0)", "#2a78d6"),
        ("pulsed_p0.01", "Amuru pulsed, p 0.01", "#eb6834"),
        ("pulsed_p0.25", "Li: protocol-aware (Amuru p 0.25)", "#1baf7a")]


def draw(env, out_npz):
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")          # before any Sionna import
    import torch
    import attacks
    import evaluate
    import link as lk
    import models

    d = LF.load(env)
    device = lk.setup(seed=778)
    L = E.apply(lk.Link(**lk.LINK), env)
    specs = dict(evaluate.classical())
    save, stats = {}, {}
    for tag in CALIB:
        if tag not in d["attackers"]:
            continue
        _, jsr = LF.matched(d, d["attackers"][tag]["points"], "argmax", 0.5)
        if np.isnan(jsr):
            continue
        if tag in specs:
            spec = specs[tag]
        else:
            G, scale, _ = models.load_generator(E.art(env, "gan", f"{tag}_G.pt"), device)
            spec = dict(name="gan", G=G, scale=scale, tag=tag)
        with torch.no_grad():
            out = attacks.frames(L, N_FRAMES, spec, [jsr], E.SNR_DB, keep_jammer=True)
            zj = L.matched_filter(out["j"], E.N_SYM).cpu().numpy()          # [F, N]
        e = np.abs(zj) ** 2
        pushes = (np.abs(zj) > HARD).sum(axis=1)
        broken = (out["bits"] != out["bits_hat"]).any(dim=-1).cpu().numpy()
        es = np.sort(e.ravel())[::-1]
        stats[tag] = dict(jsr=float(jsr), p_eff=float(e.mean() ** 2 / (e ** 2).mean()),
                          share_90=float(np.searchsorted(np.cumsum(es) / es.sum(), 0.9) / es.size),
                          pushes_mean=float(pushes.mean()), frames_without_push=float((pushes == 0).mean()),
                          per=float(broken.mean()))
        save[f"{tag}__e"] = e.astype(np.float32)
        save[f"{tag}__pushes"] = pushes.astype(np.int16)
        if tag == SHOW[0][0] or tag == "pulsed_p0.01":
            j = out["j"][:4, L.active(E.N_SYM)].cpu().numpy()
            save[f"{tag}__trace"] = (np.abs(j) ** 2 / L.p_s).astype(np.float32)
        print(f"  {tag:<18} JSR {jsr:+6.1f} dB: p_eff {stats[tag]['p_eff']:.4f}, 90 % of energy on "
              f"{100 * stats[tag]['share_90']:.2f} % of symbols, hard pushes/frame {stats[tag]['pushes_mean']:.2f}, "
              f"frames with none {stats[tag]['frames_without_push']:.3f}, PER {stats[tag]['per']:.3f}", flush=True)
    np.savez_compressed(out_npz, **save, stats=json.dumps(stats))
    with open(os.path.join(os.path.dirname(out_npz), "pulses.json"), "w") as f:
        json.dump(dict(env=env, level="PER 0.5", hard=HARD, n_frames=N_FRAMES, jammers=stats), f, indent=1)


def plot(env, npz, out_png):
    plt = LF.F.plt
    from matplotlib.colors import LinearSegmentedColormap
    S = np.load(npz)
    st = json.loads(str(S["stats"]))
    cmap = LinearSegmentedColormap.from_list("ramp", [LF.F.SURFACE, "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
    fig = plt.figure(figsize=(11.5, 11.2), facecolor=LF.F.SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 0.75], hspace=0.55, wspace=0.28)
    b = 1 / np.sqrt(2)
    for i, (tag, lab, _) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, i])
        LF.F.style_axes(ax)
        ax.grid(False)
        a = np.sqrt(S[f"{tag}__e"][:N_RASTER]) / b
        im = ax.imshow(np.clip(a, 0, 2), aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        s = st[tag]
        ax.set_title(f"{lab}\nJSR {s['jsr']:+.1f} dB · p_eff {s['p_eff']:.3f} · {s['pushes_mean']:.2f} hard pushes/frame",
                     fontsize=8, loc="left")
        ax.set_xlabel("symbol in frame")
        if i == 0:
            ax.set_ylabel(f"frame (first {N_RASTER} of {N_FRAMES})")
    cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.01)
    cb.set_label("|jammer at decision point| / distance to boundary\n(≥ 1: flips the bit on its own)", fontsize=7.5,
                 color=LF.F.INK2)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 0:2])
    LF.F.style_axes(ax)
    k = np.arange(0, 9)
    w = 0.26
    for i, (tag, lab, col) in enumerate(SHOW):
        c = np.bincount(np.minimum(S[f"{tag}__pushes"], 8), minlength=9)[:9] / len(S[f"{tag}__pushes"])
        ax.bar(k + (i - 1) * w, c, width=w * 0.92, color=col, label=lab)
    m = st[SHOW[0][0]]["pushes_mean"]
    from math import exp, factorial
    ax.plot(k - w, [exp(-m) * m ** n / factorial(n) for n in k], "o", color=LF.F.INK, ms=4,
            label=f"Poisson with cnnGAN1's mean ({m:.2f}): bursts at random times")
    ax.set_xticks(k)
    ax.set_xticklabels([str(n) for n in k[:-1]] + ["8+"])
    ax.set_xlabel(f"hard pushes in a frame (|jammer at decision point| > {HARD}; boundary at 0.71)")
    ax.set_ylabel("share of frames")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=LF.F.INK2)
    ax.set_title("How many symbols per frame it hits hard (all at the JSR where each breaks 50 % of frames)",
                 fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[1, 2])
    LF.F.style_axes(ax)
    amuru = sorted([(float(t[8:]), s["p_eff"]) for t, s in st.items() if t.startswith("pulsed_p")])
    ax.plot([p for p, _ in amuru], [q for _, q in amuru], "o-", color="#eb6834", ms=4, lw=1.4, label="Amuru pulsed, nominal p")
    for tag, s in st.items():
        if tag.startswith("cnn_b10_grey"):
            ax.axhline(s["p_eff"], color="#2a78d6", lw=1.0, alpha=0.8)
    ax.axhline(np.nan, color="#2a78d6", lw=1.0, label="cnnGAN1, grey-box seeds")
    ax.plot([0.005, 1], [0.005, 1], ls=":", color=LF.F.MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Amuru's p (share of symbols on)")
    ax.set_ylabel("measured p_eff")
    ax.legend(fontsize=7, frameon=False, labelcolor=LF.F.INK2, loc="upper left")
    ax.set_title("cnnGAN1 on Amuru's p axis", fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[2, :])
    LF.F.style_axes(ax)
    tr = S[f"{SHOW[0][0]}__trace"]
    n = tr.shape[1]
    for f in range(tr.shape[0]):
        x = np.arange(n) + f * (n + 64)
        ax.plot(x, tr[f], color="#2a78d6", lw=0.7)
        ax.axvspan(f * (n + 64) - 64, f * (n + 64), color=LF.F.GRID, alpha=0.6, lw=0)
    ax.set_xlim(0, tr.shape[0] * (n + 64) - 64)
    ax.set_xticks([f * (n + 64) + n / 2 for f in range(tr.shape[0])])
    ax.set_xticklabels([f"frame {f + 1}" for f in range(tr.shape[0])])
    ax.set_ylabel("|j[n]|² / signal power")
    ax.set_title(f"cnnGAN1's waveform, four frames of {n} samples (8 per symbol): power per sample", fontsize=8.5, loc="left")
    fig.suptitle(f"Where cnnGAN1 puts its energy, next to Amuru's pulsed jammer — env {env} ({E.LABEL[env]}), "
                 f"SNR 15 dB = Es/N0 24 dB", fontsize=10, color=LF.F.INK, x=0.01, ha="left", y=0.995)
    fig.savefig(out_png, dpi=140, facecolor=LF.F.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out_png)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--plot-only", action="store_true", help="replot from pulse_samples.npz (login node)")
    args = ap.parse_args()
    out = E.art(args.env, "li")
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, "pulse_samples.npz")
    if not args.plot_only:
        draw(args.env, npz)
    plot(args.env, npz, os.path.join(out, "fig_pulses.png"))
    clean = os.path.join(E.ART_ROOT, f"Li_{args.env}")
    os.makedirs(clean, exist_ok=True)
    plot(args.env, npz, os.path.join(clean, "fig_pulses_cnnGAN1.png"))


if __name__ == "__main__":
    main()
