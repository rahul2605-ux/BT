"""
Final experiment, chain step 4 -- every attacker against every detector of one
environment (README §3.4). New file; the sweep point is defender.measure.

    sbatch submit_evaluate.sh base           # usually via submit_env.sh base (pinned titan_rtx)

Per attacker, JSR -50 ... +15 dB in 1 dB steps, 512 frames per point; per point BER, SER,
PER with their counts, P_det of each detector at alpha 0.01 / 0.05, the mean statistic
and the jammer's DC share. The clean reference (4096 frames of the same environment) gives
the realised FAR and the clean damage floor that "excess" damage is measured against. No
confirmation pass: no pick at alpha is reported.

Attackers (README §3.4 table):
    omniscient_e1     the genie flip (upper bound, in every figure)
    omniscient_e0.1   the genie push (reference)
    random_push_e0.1  the genie's timing and phase without its data (appended last, so the
                      attackers before it keep their seeds)
    noise             barrage (= Li's barrage)
    pulsed_p<p>       Amuru pulsed, p in {1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01}; p 1 = matched
                      QPSK (Zhou's "optimal"), p 0.25 = Li's protocol-aware
    tone              Li's tone = the supervisor's constant-vector baseline
    pulse_comb        Li's pulse comb
    zhou_cgan         Zhou's plain CGAN, artifacts/cgan/run001_G.pt, evaluated, not retrained
    control_r*, cnn_b10_r*, energy_b10_r0, kurtosis_b10_r0     this environment's generators
    cnn_li_r*         the cnnGAN continued at Li's damage level, where trained (appended last)
    cnn_b10_grey_r*, cnn_li_grey_r*   the same two, grey-box (trained on the surrogate cnn_s12),
                      evaluated like every attacker against the DEPLOYED CNN
The on/off baseline is analytic (figures.py), from pulsed_p1 at +15 dB.

Each attacker is re-seeded by position, so every attacker sees the same bits, noise and
channel draws in the first draws of each sweep (common random numbers within a job).

Output: artifacts/final/<env>/eval.json, rewritten after every attacker. --out eval_li.json
writes the Li-style rerun (the CNN's argmax decision added, README §3.3q) beside it and leaves
eval.json, which §3.3q cites, untouched.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import

import argparse
import json
import os
import time

import numpy as np
import torch

import attacks
import defender
import env as E
import link as lk
import models
import train_gan

JSR_GRID = [float(v) for v in np.arange(-50.0, 15.5, 1.0)]
N_FRAMES = 512
N_CLEAN = 4096
SEED = 3000
ZHOU_G = os.path.join(E.CGAN_ART, "run001_G.pt")


def classical():
    return ([("omniscient_e1", dict(name="omniscient", eta=1.0)),
             ("omniscient_e0.1", dict(name="omniscient", eta=0.1)),
             ("noise", dict(name="noise"))]
            + [(f"pulsed_p{p:g}", dict(name="pulsed", p=p)) for p in attacks.PULSED_P]
            + [("tone", dict(name="tone")), ("pulse_comb", dict(name="pulse_comb"))])


def added():
    """Attackers added after the run of 2026-09-30, last in the list so no earlier seed moves."""
    return [("random_push_e0.1", dict(name="random_push", eta=0.1))]


def generators(env):
    return [("zhou_cgan", ZHOU_G)] + [(train_gan.tag(t, r), E.art(env, "gan", f"{train_gan.tag(t, r)}_G.pt"))
                                      for t, r in train_gan.ARRAY]


def extra_generators(env):
    """train_gan.ARRAY_EXTRA (cnn_li, then the grey-box cnn_b10_grey / cnn_li_grey), where trained;
    appended last in that order, so no earlier attacker's seed moves."""
    tags = [train_gan.tag(t, r) for t, r in train_gan.ARRAY_EXTRA]
    return [(t, E.art(env, "gan", f"{t}_G.pt")) for t in tags if os.path.exists(E.art(env, "gan", f"{t}_G.pt"))]


def suite_generators(env):
    """train_gan.ARRAY_SUITE (exploratory suite-trained generators), appended last so no earlier seed
    moves. Intended for --out eval_suite.json, not the cited eval.json."""
    tags = [train_gan.tag(t, r) for t, r in train_gan.ARRAY_SUITE]
    return [(t, E.art(env, "gan", f"{t}_G.pt")) for t in tags if os.path.exists(E.art(env, "gan", f"{t}_G.pt"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--only", default=None, help="comma list of attacker tags to (re)measure into eval.json")
    ap.add_argument("--out", default="eval.json", help="file under artifacts/final/<env>/ (eval_li.json: the Li-style rerun)")
    args = ap.parse_args()
    grid = [-30.0, -10.0, 0.0, 15.0] if E.SMOKE else JSR_GRID
    n = 64 if E.SMOKE else N_FRAMES
    t0 = time.time()
    device = lk.setup(seed=SEED)
    L = E.apply(lk.Link(**lk.LINK), args.env)
    dfd = defender.Defender.load(L, E.SNR_DB, E.art(args.env, "cnn"))
    with open(E.art(args.env, "bands.json")) as f:
        bands = json.load(f)["bands"]
    out_path = E.art(args.env, args.out)
    only = None if args.only is None else set(args.only.split(","))
    if only and os.path.exists(out_path):
        with open(out_path) as f:
            res = json.load(f)
    else:
        res = dict(meta=dict(env=args.env, env_config=E.ENVS[args.env], snr_db=E.SNR_DB,
                             esn0_db=E.SNR_DB + E.ESN0_OFFSET_DB, jsr_db=grid, n_frames=n, n_clean=N_CLEAN,
                             alphas=[0.01, 0.05], alpha=E.ALPHA, dets=dfd.dets, seed=SEED,
                             n_sym=E.N_SYM, thresholds={k: v for k, v in dfd.thr.items() if k != "clean_stat_q"},
                             bands=bands, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                   clean=None, attackers={})
    print(f"env {args.env} = {E.ENVS[args.env]} | detectors {dfd.dets} | {len(grid)} JSR points x {n} frames "
          f"| {res['meta']['gpu']}", flush=True)

    def dump():
        with open(out_path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(out_path + ".tmp", out_path)

    if res["clean"] is None:
        lk.setup(device, seed=SEED - 1)
        res["clean"] = defender.measure(L, dfd, None, None, 1024 if E.SMOKE else N_CLEAN)
        c = res["clean"]
        print(f"clean: BER {c['ber']:.2e} PER {c['per']:.4f} | realised FAR "
              + " ".join(f"{d} {c['pdet'][d]['0.05']:.4f}" for d in c["pdet"]), flush=True)
        dump()

    entries = ([(t, "classical", s) for t, s in classical()] + [(t, "generator", p) for t, p in generators(args.env)]
               + [(t, "classical", s) for t, s in added()]
               + [(t, "generator", p) for t, p in extra_generators(args.env)]
               + [(t, "generator", p) for t, p in suite_generators(args.env)])
    for i, (tag, group, what) in enumerate(entries):
        if only is not None and tag not in only:
            continue
        lk.setup(device, seed=SEED + 100 * (i + 1))
        if group == "generator":
            G, scale, ckpt = models.load_generator(what, device)
            spec = dict(name="gan", G=G, scale=scale, tag=tag)
            info = dict(path=os.path.relpath(what), recipe=ckpt.get("recipe"))
        else:
            spec, info = what, dict(spec=what)
        omni = spec["name"] in ("omniscient", "random_push")
        pts, t1 = [], time.time()
        for jsr in grid:
            pts.append(defender.measure(L, dfd, spec, jsr if omni else [jsr], n))
        res["attackers"][tag] = dict(group=group, info=info, points=pts)
        dump()
        best = max(pts, key=lambda p: p["ber"])
        print(f"  {tag:<18} done ({time.time() - t1:.0f}s): max BER {best['ber']:.3e}, PER at +15 dB "
              f"{pts[-1]['per']:.3f}, P_det at +15 dB " + " ".join(f"{d} {pts[-1]['pdet'][d]['0.05']:.2f}"
                                                                    for d in dfd.dets)
              + (f", DC share {pts[-1]['dc_share']:.3f}" if "dc_share" in pts[-1] else ""), flush=True)
    res["runtime_s"] = time.time() - t0
    dump()
    print(f"wrote {os.path.relpath(out_path)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
