"""
E2b -- the outlier alarm: P(det) at false-alarm rates far below alpha (README §3.3g).

Why. Once the jammer chooses its power per frame, switching on and off beats every
shaped waveform on average BER at a given per-frame P(det) at alpha = 0.05: a few loud
frames carry all the damage, and a defender that only sees flags at alpha cannot tell
them from false alarms (README §3.3g, 2026-09-27). A loud frame is, however, an
extreme outlier. A defender that also alarms far out in the clean tail (alpha' =
1e-3, 1e-4) catches it the first time; a waveform whose frames look nearly clean is
not caught that way. This measures both alarms on the same frames.

    sbatch --array=0,1 submit_outlier_alarm.sh     # 15 and 30 dB

Per SNR level (one task):
  1. calibrate every detector's threshold at ALPHAS on N_CAL fresh clean frames
     (one calibration for all alphas, so they are mutually consistent; 1e-4 on
     200 000 frames rests on ~20 exceedances);
  2. check the realised false-alarm rate on N_VAL further clean frames;
  3. sweep the jammers over baselines.JSR_GRID_K1: BER and P(det) at every alpha
     for every detector, on fresh frames per point.
No confirmation pass: this is a characterisation of the alarm, not a frontier pick.

Output: artifacts/cgan/outlier_alarm/<run>/snr_<snr>.json.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import argparse
import json
import os
import time

import numpy as np
import torch

import attacks
import baselines
import calibrate_snr
import detectors
import link as lk
import models
import scene
import snr_ablation
import train_gan

LEVELS = [15.0, 30.0]
ALPHAS = [0.05, 0.01, 1e-3, 1e-4]
N_CAL, N_VAL = 200_000, 50_000
CLASSICAL = ["noise", "pulsed_p1", "pulsed_p0.1"]
GENS = ["eff", "spec_cnn_b10", "spec_cnn_b1"]
ART = os.path.join(scene.ART, "..", "outlier_alarm")


def clean_stats(L, dfd, n, batch=512):
    """Per-detector statistics of n clean frames, concatenated (float64, on CPU)."""
    acc = {}
    for i in range(0, n, batch):
        out = attacks.frames(L, min(batch, n - i), None, None, dfd.snr_db)
        s, _ = dfd.statistics(out["r"], out["z"])
        for k, v in s.items():
            acc.setdefault(k, []).append(v.double().cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def point(L, dfd, spec, jsr, thr, n_frames, batch=512):
    """BER and P(det) at every alpha for one jammer at one JSR."""
    counts = np.zeros(4, dtype=np.int64)
    frame_err = 0                                # frames with >= 1 bit error (uncoded PER)
    hits = {d: {a: 0 for a in ALPHAS} for d in thr}
    for i in range(0, n_frames, batch):
        out = attacks.frames(L, min(batch, n_frames - i), spec, [jsr], dfd.snr_db)
        counts += np.array(attacks.error_counts(out))
        wrong = out["bits"] != out["bits_hat"]
        frame_err += int(wrong.reshape(wrong.shape[0], -1).any(dim=-1).sum())
        s, _ = dfd.statistics(out["r"], out["z"])
        for d in thr:
            v = s[d].double().cpu()
            for a in ALPHAS:
                hits[d][a] += int((v > thr[d][a]).sum())
    e, b = int(counts[0]), int(counts[1])
    return dict(jsr=jsr, ber=e / b, errors=e, bits=b, per=frame_err / n_frames, frame_errors=frame_err,
                pdet={d: {str(a): hits[d][a] / n_frames for a in ALPHAS} for d in thr})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help="index into LEVELS")
    ap.add_argument("--run", default="run003")
    ap.add_argument("--gan-run", default="run003")
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--points", default=None,
                    help="targeted mode: 'tag:jsr,tag:jsr' measured with --frames each (e.g. 100000), "
                         "reusing the thresholds of the grid run (--thr-from) so the two agree")
    ap.add_argument("--thr-from", default=None, help="a grid-run JSON whose thresholds to reuse")
    ap.add_argument("--gens", default=None, help="comma list of generator tags (default GENS)")
    ap.add_argument("--classical", default=None, help="comma list of classical attacks (default CLASSICAL)")
    ap.add_argument("--own-frames", type=int, default=8192,
                    help="frames for a learned-power generator's single own-power point")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    snr = LEVELS[args.task]
    n_cal, n_val = (4096, 2048) if args.smoke else (args.n_cal, args.n_val)
    jsr_grid = [-30.0, -10.0, 0.0] if args.smoke else baselines.JSR_GRID_K1
    frames = 128 if args.smoke else args.frames
    out_dir = os.path.join(ART, args.run)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"snr_{snr:g}{'_points' if args.points else ''}"
                                 f"{'_smoke' if args.smoke else ''}.json")

    device = lk.setup(seed=5000 + args.task)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = calibrate_snr.defender_at(L, snr)
    dets = [d for d in dfd.dets if d != "lrt_noise"]
    t0 = time.time()

    if args.thr_from:
        prev = json.load(open(args.thr_from.replace("{snr}", f"{snr:g}")))
        thr = {d: {a: prev["thresholds"][d][str(a)] for a in ALPHAS} for d in dets}
        far, n_cal, n_val = prev["far"], prev["meta"]["n_cal"], prev["meta"]["n_val"]
    else:
        cal = clean_stats(L, dfd, n_cal)
        thr = {d: {a: detectors.calibrate(cal[d], a) for a in ALPHAS} for d in dets}
        val = clean_stats(L, dfd, n_val)
        far = {d: {str(a): float((val[d] > thr[d][a]).double().mean()) for a in ALPHAS} for d in dets}
    print(f"SNR {snr:g} dB | {n_cal} calibration + {n_val} validation clean frames "
          f"({time.time() - t0:.0f}s) | realised FAR {json.dumps(far)}", flush=True)
    res = dict(meta=dict(snr_db=snr, alphas=ALPHAS, n_cal=n_cal, n_val=n_val, frames=frames,
                         jsr_db=jsr_grid, gan_run=args.gan_run, dets=dets),
               thresholds={d: {str(a): thr[d][a] for a in ALPHAS} for d in dets},
               far=far, jammers={})

    classical = CLASSICAL if args.classical is None else [c for c in args.classical.split(",") if c]
    specs = [(attacks.spec_name(s), s) for s in attacks.all_specs() if attacks.spec_name(s) in classical]
    gdir = os.path.join(scene.ART, "..", "gan", args.gan_run)
    own = {}                                                   # learned-power generators: tag -> cap
    gens = GENS if args.gens is None else [g for g in args.gens.split(",") if g]
    for tag, gpath in snr_ablation.generators(gdir):          # tag from each checkpoint
        if tag in gens:
            G, scale, ckpt = models.load_generator(gpath, device)
            spec = dict(name="gan", G=G, scale=scale, tag=tag)
            if ckpt.get("power") == "learned":
                # it chose its own power: evaluate it THERE (one point, the cap as the
                # JSR argument), not over a forced JSR sweep
                spec.update(power="learned", log_gain=torch.tensor(ckpt["log_gain"], device=device))
                own[tag] = ckpt["jsr_cap"]
            specs.append((tag, spec))

    want = None
    if args.points:
        want = {}
        for item in args.points.split(","):
            tag, x = item.split(":")
            want.setdefault(tag, []).append(float(x))
    for name, spec in specs:
        if want is not None and name not in want:
            continue
        grid = want[name] if want is not None else ([own[name]] if name in own else jsr_grid)
        n_f = args.own_frames if name in own else frames
        res["jammers"][name] = [point(L, dfd, spec, x, thr, n_f) for x in grid]
        if name in own:            # the power it actually chose, as JSR
            res["jammers"][name][0]["jsr_realised"] = train_gan.jammer_jsr_db(L, spec, own[name])
        with open(path, "w") as f:
            json.dump(res, f)
        print(f"  {name} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {os.path.relpath(path)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
