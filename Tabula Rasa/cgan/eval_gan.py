"""
D1 -- put a *plain* (reproduced) GAN jammer on the D0 BER-P(det) plane
(README §3.4). No detector-aware training: this measures the detectability of
Zhou's reproduced generator (run001_G, async) as it is, which is expected to land
on the floor with noise/pulsed. It is the honest baseline D2 must then beat.

    sbatch submit_eval_gan.sh                          # run001_G on the K = 1 plane
    sbatch submit_eval_gan.sh --gen ../artifacts/cgan/gan/run001/task3_G.pt --tag d2_cnn_b1

K = 1 only: with T power-controlled and every jammer asynchronous a single
jammer's BER and P(det) depend on its received JSR alone (scene.py, checked in
verify.py §10), so the whole result is one sweep over received JSR, reusing
baselines.measure / baselines.confirm and the same four calibrated detectors
(the noise NP row is not produced for a non-Gaussian jammer; see baselines.measure).

Outputs: artifacts/cgan/gan/<run>/<tag>.json.
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
import detectors
import link as lk
import models
import scene

RUN = "run001"
OUT = os.path.join(scene.ART, "..", "gan", RUN)


def sweep(L, dfd, spec, jsr_grid, n_frames, n_confirm):
    pts = []
    t0 = time.time()
    for jsr in jsr_grid:
        pts.append(baselines.measure(L, dfd, spec, [jsr], n_frames))
        p = pts[-1]
        print(f"  JSR {jsr:+6.1f} dB  BER {p['ber']:.2e}  P(det)@0.05 " +
              " ".join(f"{d} {p['pdet'][d]['0.05']:.3f}" for d in p["pdet"]) +
              f"  ({time.time() - t0:.0f}s)", flush=True)
    jsr_of = lambda i: [jsr_grid[i]]
    confirmed = baselines.confirm(L, dfd, spec, pts, jsr_of, n_confirm,
                                  baselines.DETS, detectors.ALPHAS)
    return pts, confirmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=os.path.join(scene.ART, "..", "run001_G.pt"))
    ap.add_argument("--tag", default="plain_run001")
    ap.add_argument("--task", type=int, default=None,
                    help="evaluate a trained generator gan/<run>/task{T}_G.pt (tag from its checkpoint)")
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--confirm", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.task is not None:
        args.gen = os.path.join(OUT, f"task{args.task}_G.pt")
        import torch as _t
        args.tag = _t.load(args.gen, map_location="cpu", weights_only=False).get("tag", f"task{args.task}")

    device = lk.setup(seed=3000)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = baselines.Defender(L)
    G, scale, ckpt = models.load_generator(args.gen, device)
    print(f"device {device} | generator {os.path.relpath(args.gen)} | scale {scale:.4f} "
          f"| n_classes {ckpt.get('n_classes', 1)}", flush=True)

    spec = dict(name="gan", G=G, scale=scale, tag=args.tag)
    jsr_grid = ([-30.0, -10.0, 0.0] if args.smoke else baselines.JSR_GRID_K1)
    frames = 64 if args.smoke else args.frames
    confirm = 256 if args.smoke else args.confirm

    clean = baselines.measure(L, dfd, None, None, 4 * frames)
    print(f"clean: BER {clean['ber']:.2e}, FAR {json.dumps(clean['pdet'])}", flush=True)
    t0 = time.time()
    pts, confirmed = sweep(L, dfd, spec, jsr_grid, frames, confirm)

    res = dict(run=RUN, tag=args.tag, generator=os.path.relpath(args.gen), scale=scale,
               K=1, jsr_db=jsr_grid, n_frames=frames, n_confirm=confirm,
               alphas=detectors.ALPHAS, clean=clean, points=pts, confirmed=confirmed,
               runtime_s=time.time() - t0)
    path = os.path.join(OUT, f"{args.tag}{'_smoke' if args.smoke else ''}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    print(f"confirmed picks: {json.dumps(confirmed)}")
    print(f"wrote {os.path.relpath(path)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
