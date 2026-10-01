"""
Final experiment -- IQ pictures of the jammers (user, 2026-09-30). New file.

    sbatch submit_iq.sh base                     # draws frames (Sionna) and plots
    python iq_plots.py --env base --plot-only    # login node: replot from the saved samples

Each jammer is drawn at the JSR where it breaks 50 % and 90 % of frames (li_figures.matched on
eval_li.json, argmax decision; seed r0 for the generators), on fresh frames of the environment.
One row per jammer, three panels:
  1. its waveform at the receiver, j[n] over the victim's active window, one dot per sample,
     scaled to its own RMS: the SHAPE of the signal (the row states the power);
  2. what it adds at the victim's decision points: the matched-filter output of j alone at the
     symbol instants, in units of the symbol amplitude. The QPSK points sit at (+-0.71, +-0.71)
     and the decision boundaries are the axes, 0.71 away;
  3. the victim's received constellation z_k (signal + noise + jammer), symbol errors in black.
A clean row comes first. The CNN sees none of this directly: it sees the spectrogram of r.

Outputs, artifacts/final/<env>/li/: iq_samples.npz, fig_iq_per50.png, fig_iq_per90.png
"""

import argparse
import json
import os

import numpy as np

import env as E
import li_figures as LF

LEVELS = (0.5, 0.9)
N_FRAMES = 64
N_WAVE = 6000                          # waveform dots per panel (subsampled)
ROWS = [   # tag in eval_li.json -> roster key for label/colour
    ("clean", None), ("noise", "noise"), ("tone", "tone"), ("pulse_comb", "pulse_comb"),
    ("pulsed_p0.25", "pulsed_p0.25"), ("omniscient_e1", "omniscient_e1"),
    ("cnn_b10_r0", "cnn_b10"), ("cnn_b10_grey_r0", "cnn_b10_grey"),
    ("cnn_li_r0", "cnn_li"), ("cnn_li_grey_r0", "cnn_li_grey"),
]


def operating_points(env):
    """{tag: {level: (jsr, recall, excess PER)}} at matched damage, from eval_li.json."""
    d = LF.load(env)
    jsr_grid = np.array(d["meta"]["jsr_db"], float)
    ops = {}
    for tag, _ in ROWS[1:]:
        if tag not in d["attackers"]:
            continue
        pts = d["attackers"][tag]["points"]
        per = LF.F.excess(pts, "per", d["clean"])
        ops[tag] = {}
        for l in LEVELS:
            rec, jsr = LF.matched(d, pts, "argmax", l)
            if not np.isnan(jsr):
                ops[tag][l] = (float(jsr), float(rec), float(np.interp(jsr, jsr_grid, per)))
    return ops


def draw(env, ops, out_npz):
    """Sionna side: frames at each operating point -> iq_samples.npz."""
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")          # before any Sionna import
    import torch
    import attacks
    import evaluate
    import link as lk
    import models

    device = lk.setup(seed=777)
    L = E.apply(lk.Link(**lk.LINK), env)
    specs = dict(evaluate.classical())
    n0 = L.noise_var(E.SNR_DB)
    act = L.active(E.N_SYM)
    rng = np.random.default_rng(0)
    save = {}

    def one(key, spec, jsr):
        out = attacks.frames(L, N_FRAMES, spec, None if spec is None else [jsr], E.SNR_DB, keep_jammer=True)
        z = out["z"].cpu().numpy().ravel()
        wrong = (out["bits"] != out["bits_hat"]).reshape(N_FRAMES, -1, 2).any(-1).cpu().numpy().ravel()
        save[f"{key}__z"], save[f"{key}__wrong"] = z.astype(np.complex64), wrong
        if "j" in out:
            j = out["j"]
            ja = j[:, act].cpu().numpy().ravel()
            ja = ja / np.sqrt(np.mean(np.abs(ja) ** 2))
            save[f"{key}__wave"] = ja[rng.choice(ja.size, min(N_WAVE, ja.size), replace=False)].astype(np.complex64)
            save[f"{key}__zj"] = L.matched_filter(j, E.N_SYM).cpu().numpy().ravel().astype(np.complex64)

    one("clean", None, None)
    for tag, _ in ROWS[1:]:
        if tag not in ops:
            continue
        if tag in specs:
            spec = specs[tag]
        else:
            G, scale, _ = models.load_generator(E.art(env, "gan", f"{tag}_G.pt"), device)
            spec = dict(name="gan", G=G, scale=scale, tag=tag)
        for l, (jsr, _, _) in ops[tag].items():
            with torch.no_grad():
                one(f"{tag}@{l}", spec, jsr)
    np.savez_compressed(out_npz, **save, ops=json.dumps({t: {str(l): v for l, v in o.items()} for t, o in ops.items()}),
                        n0=n0)
    print(f"wrote {os.path.relpath(out_npz)} ({len(save)} arrays)", flush=True)


def plot(env, npz, level, out_png):
    plt = LF.F.plt
    S = np.load(npz)
    ops = {t: {float(l): v for l, v in o.items()} for t, o in json.loads(str(S["ops"])).items()}
    rows = [(t, k) for t, k in ROWS if t == "clean" or (t in ops and level in ops[t] and k in LF.ROSTER)]
    fig = plt.figure(figsize=(8.4, 2.75 * len(rows) + 0.6), facecolor=LF.F.SURFACE)
    subs = fig.subfigures(len(rows) + 1, 1, height_ratios=[0.22] + [1] * len(rows))
    subs[0].set_facecolor(LF.F.SURFACE)
    subs[0].text(0.01, 0.5, f"IQ of the jammers at matched damage: each where it breaks {100 * level:.0f} % of frames — "
                 f"env {env} ({E.LABEL[env]}), SNR 15 dB = Es/N0 24 dB", fontsize=9.5, color=LF.F.INK, va="center")
    subs = subs[1:]
    q = 1 / np.sqrt(2)
    for i, (sf, (tag, key)) in enumerate(zip(subs, rows)):
        sf.set_facecolor(LF.F.SURFACE)
        ax = sf.subplots(1, 3)
        for a in ax:
            LF.F.style_axes(a)
            a.set_aspect("equal")
            a.grid(False)
        if tag == "clean":
            k = "clean"
            label, col = "no jammer (clean)", LF.F.INK2
            head = None
            clean_note = f"no jammer (clean)\nEs/N0 24 dB\nsymbol errors: {int(S['clean__wrong'].sum())} of {S['clean__wrong'].size}"
        else:
            k = f"{tag}@{level}"
            label, col, _ = LF.ROSTER[key]
            jsr, rec, per = ops[tag][level]
            head = (f"{label}{' · seed r0' if tag.endswith('_r0') else ''}\n"
                    f"JSR {jsr:+.1f} dB · excess PER {per:.2f} · CNN flags {rec:.2f} of samples (argmax)")
        if head:
            sf.suptitle(head, fontsize=8.5, color=LF.F.INK, x=0.02, ha="left")
        z, wrong = S[f"{k}__z"], S[f"{k}__wrong"]
        lim = max(1.6, float(np.quantile(np.abs(np.concatenate([z.real, z.imag])), 0.998)) * 1.08)
        if f"{k}__wave" in S:
            w, zj = S[f"{k}__wave"], S[f"{k}__zj"]
            wl = max(2.5, float(np.quantile(np.abs(np.concatenate([w.real, w.imag])), 0.998)) * 1.08)
            ax[0].scatter(w.real, w.imag, s=1.2, color=col, alpha=0.35, lw=0, rasterized=True)
            ax[0].add_patch(plt.Circle((0, 0), 1.0, fill=False, ec=LF.F.AXIS, lw=0.8, ls="--"))
            ax[0].set_xlim(-wl, wl)
            ax[0].set_ylim(-wl, wl)
            lim = max(lim, float(np.quantile(np.abs(np.concatenate([zj.real, zj.imag])), 0.998)) * 1.08)
            ax[1].scatter(zj.real, zj.imag, s=2.0, color=col, alpha=0.45, lw=0, rasterized=True)
        else:
            for a, t in zip(ax[:2], (clean_note, "no jammer")):
                a.set_axis_off()
                a.text(0.5, 0.45, t, transform=a.transAxes, ha="center", va="center", color=LF.F.INK2, fontsize=8)
        ax[2].scatter(z[~wrong].real, z[~wrong].imag, s=1.6, color=LF.F.MUTED, alpha=0.35, lw=0, rasterized=True)
        ax[2].scatter(z[wrong].real, z[wrong].imag, s=4.0, color=LF.F.INK, alpha=0.9, lw=0,
                      label="symbol error", rasterized=True)
        for a in (ax[1:] if f"{k}__wave" in S else ax[2:]):
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.axhline(0, color=LF.F.AXIS, lw=0.8)
            a.axvline(0, color=LF.F.AXIS, lw=0.8)
            a.scatter([q, q, -q, -q], [q, -q, q, -q], marker="+", s=40, color=LF.F.INK2, lw=0.9, zorder=3)
        for a in ax:
            a.tick_params(labelsize=6.5)
        if i == 0:
            ax[0].set_title("1 · jammer waveform j[n]\n(per sample, / its RMS; dashed = RMS)", fontsize=7.5)
            ax[1].set_title("2 · jammer at the decision points\n(MF output of j; + = QPSK points)", fontsize=7.5)
            ax[2].set_title("3 · received constellation z_k\n(grey correct, black = symbol error)", fontsize=7.5)
    fig.savefig(out_png, dpi=130, facecolor=LF.F.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out_png)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--plot-only", action="store_true", help="replot from iq_samples.npz (login node)")
    ap.add_argument("--clean", action="store_true", help="li_figures.CLEAN roster only, into artifacts/final/Li_<env>/")
    args = ap.parse_args()
    out = E.art(args.env, "li")
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, "iq_samples.npz")
    if not args.plot_only:
        draw(args.env, operating_points(args.env), npz)
    if args.clean:
        LF.clean_mode()
        out = os.path.join(E.ART_ROOT, f"Li_{args.env}")
        os.makedirs(out, exist_ok=True)
    for l in LEVELS:
        plot(args.env, npz, l, os.path.join(out, f"fig_iq_per{100 * l:.0f}.png"))


if __name__ == "__main__":
    main()
