# copied from cgan/train_gan.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment, chain step 3 -- one generator, trained WHITE-BOX by direct gradient
against one environment's frozen detector (README §3.4).

    sbatch --array=0-7 submit_train_gan.sh base          # usually via submit_env.sh base

Array index -> (task, seed replicate), ARRAY below. Tasks are named; `cgan_task` is the
cgan/train_gan.py task id they correspond to, and it sets the seed as it did there
(4000 + id + 1000 rep), so rep 0 starts from run003's / S4's seeds:

    control        beta 0, no detector term (cgan task 0)              3 seeds
    energy_b10     one-sided `energy` after the matched filter, beta 10
                   (cgan task 2, retargeted from full-band power)       1 seed
    kurtosis_b10   two-sided kurtosis, beta 10 (cgan task 10)          1 seed
    cnn_b10        the environment's retrained CNN, beta 10 (task 14)  3 seeds

Recipe (run003's, as S4): random init (Zhou's architecture, no imitation stage), 4000
Adam steps (lr 2e-4, betas 0.5/0.999), 128 frames per step (32 for the CNN), one JSR per
step drawn uniformly from the task's band in bands.json, alpha 0.05, and

    loss = -( log E[BER] - beta * P_det_soft ),

README §2.8's reward. log E[BER] is exact over the AWGN (attacks.log_expected_ber), with
each frame's own noise variance under noise-level uncertainty. P_det_soft is
detectors.soft_pdet at the environment's CFAR threshold, width = the statistic's clean
std on this environment's clean frames. Power is NOT in the loss: channel.receive scales
every frame to exactly its JSR (hard equality projection). The training frames carry the
environment's noise-level factor and fading, like the defender's.

Pruned from cgan/train_gan.py: warm start, learned power, PER damage, shadowing, sync,
surrogate/E2 defenders, numbered tasks and the beta grid.

Outputs: artifacts/final/<env>/gan/<task>_r<rep>_G.pt (net_scratch, symlinked) and
<task>_r<rep>.json (recipe + per-step history).
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import

import argparse
import json
import os
import time

import torch

import attacks
import defender
import detectors
import env as E
import link as lk
import models

TASKS = {
    "control":      dict(target=None,       beta=0.0,  cgan_task=0),
    "energy_b10":   dict(target="energy",   beta=10.0, cgan_task=2),
    "kurtosis_b10": dict(target="kurtosis", beta=10.0, cgan_task=10),
    "cnn_b10":      dict(target="spec_cnn", beta=10.0, cgan_task=14),
}
ARRAY = [("control", 0), ("control", 1), ("control", 2), ("cnn_b10", 0), ("cnn_b10", 1),
         ("cnn_b10", 2), ("energy_b10", 0), ("kurtosis_b10", 0)]
FRAMES = {None: 128, "energy": 128, "kurtosis": 128, "spec_cnn": 32}
STEPS = 4000
LR = 2e-4


def tag(task, rep):
    return f"{task}_r{rep}"


def clean_scale(L, dfd, target, snr_db, n=4096, seed=1):
    """Std of the target's statistic on clean frames of this environment: the soft width."""
    lk.setup(L.device, seed=seed)
    with torch.no_grad():
        out = attacks.frames(L, n, None, None, snr_db)
        s = dfd.statistics(out["r"], out["z"], dets=[target], gain=out.get("gain"))
    return float(s[target].std())


def train(L, dfd, G, target, beta, steps, frames, band, seed, snr_db):
    """One generator; returns the trained G and a per-step history."""
    n0 = L.noise_var(snr_db)
    thr = dfd.threshold(target, E.ALPHA) if target else None
    sc = clean_scale(L, dfd, target, snr_db) if target else None
    if target is not None:
        print(f"  target {target}: threshold {thr:.6g}, clean-frame std {sc:.6g} (soft_pdet width)", flush=True)
    spec = dict(name="gan", G=G, scale=1.0, grad=True)
    opt = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    lo, hi = band
    hist, t0 = [], time.time()
    for step in range(steps):
        lk.setup(L.device, seed=seed * 1_000_000 + step)     # fresh randomness per step
        jsr = lo + (hi - lo) * float(torch.rand(1, device=L.device))
        out = attacks.frames(L, frames, spec, [jsr], snr_db, noiseless=True)
        log_ber = attacks.log_expected_ber(out, attacks.frame_noise_var(out, n0))
        if target is None:
            soft = torch.zeros((), device=L.device)
        else:
            s = dfd.statistics(out["r"], out["z"], dets=[target], grad=True, gain=out.get("gain"))
            soft = detectors.soft_pdet(s[target], thr, sc)
        loss = -(log_ber - beta * soft)
        if not torch.isfinite(loss):           # fail fast instead of 4000 steps of NaN
            raise RuntimeError(f"non-finite loss at step {step} (JSR {jsr}): {float(loss)}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            hist.append(dict(step=step, jsr_db=jsr, loss=float(loss.detach()),
                             log_ber=float(log_ber.detach()), soft_pdet=float(soft.detach())))
            print(f"  step {step:4d}  JSR {jsr:+6.1f}  loss {float(loss.detach()):.4f}  "
                  f"logE[BER] {float(log_ber):.3f}  P_det_soft {float(soft):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return G, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--index", type=int, default=None,
                    help="position in ARRAY (default SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--task", choices=list(TASKS), default=None)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    if args.task is None:
        idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        args.task, args.rep = ARRAY[idx]
    if E.SMOKE:
        args.steps = 20
    task = TASKS[args.task]
    seed = 4000 + task["cgan_task"] + 1000 * args.rep

    with open(E.art(args.env, "bands.json")) as f:
        bands = json.load(f)["bands"]
    band = tuple(bands["control" if task["target"] is None else task["target"]]["band"])
    frames = FRAMES[task["target"]]

    device = lk.setup(seed=seed)
    L = E.apply(lk.Link(**lk.LINK), args.env)
    dfd = defender.Defender.load(L, E.SNR_DB, E.art(args.env, "cnn"))
    G = models.Generator(n_classes=1).to(device)            # random init: run003's recipe
    G.train()
    name = tag(args.task, args.rep)
    print(f"env {args.env} = {E.ENVS[args.env]} | task {name} = {task} | seed {seed} | band {band} | "
          f"{frames} frames/step | {args.steps} steps | device {device}", flush=True)
    t0 = time.time()
    G, hist = train(L, dfd, G, task["target"], task["beta"], args.steps, frames, band, seed, E.SNR_DB)
    recipe = dict(env=args.env, env_config=E.ENVS[args.env], task=args.task, rep=args.rep,
                  target=task["target"], beta=task["beta"], cgan_task=task["cgan_task"], seed=seed,
                  steps=args.steps, frames=frames, lr=LR, jsr_band=list(band), alpha=E.ALPHA,
                  init="random", snr_db=E.SNR_DB, defender=os.path.relpath(E.art(args.env, "cnn")),
                  runtime_s=time.time() - t0)
    G.eval()
    E.save_big({"state_dict": G.state_dict(), "n_classes": 1, "seg_len": models.SEG_LEN, "scale": 1.0,
                "target": task["target"], "beta": task["beta"], "tag": name, "recipe": recipe},
               args.env, os.path.join("gan", f"{name}_G.pt"))
    with open(E.art(args.env, "gan", f"{name}.json"), "w") as f:
        json.dump(dict(tag=name, recipe=recipe, history=hist), f)
    print(f"wrote gan/{name}_G.pt, gan/{name}.json in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
