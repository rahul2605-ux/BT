"""
D2 -- a detector-aware generative jammer, trained WHITE-BOX by direct gradient
against a frozen detector (README §3.4, threat model in the plan).

    sbatch --array=0-16 submit_train_gan.sh           # one task per (target, beta)
    sbatch submit_train_gan.sh --smoke                # task 0 + one power task, few steps

The neural counterpart of the shaped-noise control D2a (train_shaped.py): the SAME
objective and the SAME detectors, but an 8.6 M-parameter generator over raw IQ
instead of a 48-parameter family, so GAN vs shaped isolates the hypothesis class.

Per task (target detector, beta):
  * the generator is warm-started from the reproduced Zhou generator (run001_G) and
    trained by Adam to minimise, over JSR sampled in the active band each step,
        loss = -( log E[BER] - beta * P_det_soft ),
    README §2.8's reward. log E[BER] is the grad-carrying tensor
    attacks.log_expected_ber (logsumexp; never underflows -- the A.5 / D2a wall).
    P_det_soft is detectors.soft_pdet at the detector's calibrated alpha threshold,
    with the clean-frame std as its width. beta = 0 (task 0) is effectiveness only;
  * power is NOT in the loss: channel.receive scales every frame to exactly its JSR
    (the hard equality projection), so the projection removes the generator's overall
    gain and the gradient reaches only the waveform SHAPE (verify.py §14b/§14c);
  * the frozen detector is the deployed one (white-box: the attacker trains against
    the actual weights -- the attacker upper bound, plan threat model). Only the CNN
    has secret weights; power/kurtosis are analytic.

This script trains and saves each generator. eval_gan.py then puts it on the D0/D2a
BER-P(det) plane against all detectors, with the confirmation pass -- the same code
path as D1, so the numbers are directly comparable.

Every TARGET is differentiated white-box: power and kurtosis analytically, and
spec_cnn through the straight-through colour LUT of detectors.spectrogram_image
(grad=True), which leaves the deployed CNN's forward statistic untouched. Outputs:
artifacts/cgan/gan/run001/task{T}_G.pt (+ scale, target, beta, recipe) and task{T}.json.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import argparse
import json
import math
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
# spec_cnn is appended, never inserted: task ids 0-12 must keep meaning what the
# 2026-09-20 artifacts say they mean (§3.3f). The CNN is white-box like the rest --
# detectors.cnn_statistic(grad=True) differentiates the DEPLOYED weights through a
# straight-through colour LUT, so no surrogate is trained and no transfer gap is priced.
TARGETS = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn"]
BETAS = [1.0, 10.0, 100.0, 1000.0]   # geometric: log E[BER] runs ~1e3, so beta spans orders
# The CNN needs a wider beta range than the analytic detectors, and the reason is
# measured (job 2267938 §14d): at equal beta the detector term reaches G with
# ||grad|| ~ 4e-2 while the full loss carries ~4e4 -- a CNN logit depends on the
# waveform far less directly than mean power does, so beta must climb further before
# stealth bites. Appended, so tasks 0-16 keep their meaning.
EXTRA_CNN_BETAS = [1e4, 1e5, 1e6]
TASKS = ([dict(target=None, beta=0.0)]
         + [dict(target=t, beta=b) for t in TARGETS for b in BETAS]
         + [dict(target="spec_cnn", beta=b) for b in EXTRA_CNN_BETAS])
ALPHA = detectors.HEADLINE_ALPHA
# The band JSR is sampled from during training. The rule is "straddle the target
# detector's own transition, starting ~5 dB below where it begins to flag": power
# flags from ~-27 dB and BER crosses ~-16 (§3.3e), hence (-32, 0). The CNN's
# transition sits 12-16 dB LOWER -- measured on the D0 sweep, it already flags noise
# at P(det) 0.28 at -32 dB and 0.84 at -28, and only falls to alpha near -40 -- so on
# (-32, 0) every sampled JSR would be saturated-detected, sigmoid'((stat-thr)/scale)
# would be ~1e-11, and no beta could recover a gradient. Per-target band instead.
JSR_BAND = (-32.0, 0.0)
JSR_BANDS = {"spec_cnn": (-48.0, -16.0)}
# Frames per gradient step. The CNN keeps a full EfficientNet-B0 graph per frame, so
# its batch is smaller; the analytic detectors keep the 2026-09-20 value.
FRAMES = {None: 128, "power_one_sided": 128, "power_two_sided": 128, "kurtosis": 128,
          "spec_cnn": 32}


def clean_scale(L, dfd, target, n=4096, seed=1):
    """Std of the target's statistic on clean frames -- the soft-detector's width."""
    lk.setup(L.device, seed=seed)
    out = attacks.frames(L, n, None, None, lk.SNR_DB)
    s, _ = dfd.statistics(out["r"], out["z"], dets=[target], gain=out.get("gain"))
    return float(s[target].std())


def jammer_jsr_db(L, spec, cap_db, n=64):
    """Mean received jammer power over the active window, as JSR [dB] (no grad)."""
    with torch.no_grad():
        _, sym, _ = L.modulate(n, scene.N_SYM)
        j = attacks.jammer_at_rx(L, spec, sym, [cap_db])
        a = L.active(scene.N_SYM)
        return float(10 * torch.log10(j[..., a].abs().pow(2).mean() / L.p_s))


def train(L, dfd, G, scale, target, beta, steps, frames, jsr_band, seed, mode=None):
    """
    One generator; returns the trained G, a per-step history and the learned log-gain
    (None unless mode["power"] == "learned"). mode (README §3.3f, 2026-09-27):
      power  "fixed"   -- every frame forced to a JSR sampled from jsr_band (all runs <= run004);
             "learned" -- a learned global gain under a hard per-frame cap (attacks.learned_power),
                          so the generator CHOOSES its power; cap = mode["cap"] dB JSR,
                          started at mode["init"] dB so the detection gradient is alive;
      damage "ber" (log E[BER], all runs <= run004) or "per" (log E[frame error rate]).
    """
    mode = {**dict(power="fixed", damage="ber", cap=0.0, init=-40.0, gain_lr=1e-2), **(mode or {})}
    n0 = L.noise_var(lk.SNR_DB)
    thr = dfd.threshold(target, ALPHA) if target else None
    sc = clean_scale(L, dfd, target) if target else None
    if target is not None:
        print(f"  target {target}: threshold {thr:.6g}, clean-frame std {sc:.6g} "
              f"(soft_pdet width)", flush=True)
    spec = dict(name="gan", G=G, scale=scale, grad=True)
    groups = [dict(params=G.parameters(), lr=2e-4)]
    log_gain = None
    if mode["power"] == "learned":
        log_gain = torch.zeros((), device=L.device, requires_grad=True)
        spec.update(power="learned", log_gain=log_gain)
        big = mode["cap"] + 200.0                            # a cap that never binds
        with torch.no_grad():
            log_gain.fill_(0.5 * math.log(10 ** ((mode["init"] - jammer_jsr_db(L, spec, big)) / 10)))
        print(f"  learned power: cap {mode['cap']:+.1f} dB, start {jammer_jsr_db(L, spec, big):+.1f} dB, "
              f"gain lr {mode['gain_lr']:g}, damage {mode['damage']}", flush=True)
        groups.append(dict(params=[log_gain], lr=mode["gain_lr"]))
    opt = torch.optim.Adam(groups, betas=(0.5, 0.999))
    damage = attacks.log_expected_per if mode["damage"] == "per" else attacks.log_expected_ber
    lo, hi = jsr_band
    hist, t0 = [], time.time()
    for step in range(steps):
        lk.setup(L.device, seed=seed * 1_000_000 + step)     # fresh randomness per step
        jsr = mode["cap"] if log_gain is not None else lo + (hi - lo) * float(torch.rand(1, device=L.device))
        out = attacks.frames(L, frames, spec, [jsr], lk.SNR_DB, noiseless=True)
        log_ber = damage(out, n0)          # log E[BER] or log E[PER] (name kept for the history keys)
        if target is None:
            soft = torch.zeros((), device=L.device)
        else:
            s, _ = dfd.statistics(out["r"], out["z"], dets=[target], grad=True, gain=out.get("gain"))
            soft = detectors.soft_pdet(s[target], thr, sc)
        loss = -(log_ber - beta * soft)
        if not torch.isfinite(loss):           # fail fast instead of 4000 steps of NaN
            raise RuntimeError(f"non-finite loss at step {step} (JSR {jsr}): {float(loss)}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            eff = jammer_jsr_db(L, spec, jsr) if log_gain is not None else jsr
            hist.append(dict(step=step, jsr_db=eff, loss=float(loss.detach()), damage=mode["damage"],
                             log_ber=float(log_ber.detach()), soft_pdet=float(soft.detach())))
            print(f"  step {step:4d}  JSR {eff:+6.1f}  loss {float(loss.detach()):.4f}  "
                  f"logE[{mode['damage'].upper()}] {float(log_ber):.3f}  P_det_soft {float(soft):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return G, hist, (None if log_gain is None else float(log_gain.detach()))


def run_task(L, dfd, task, gen0, steps, frames, seed, band=JSR_BAND, init="warm", mode=None):
    if init == "warm":
        G, scale, _ = models.load_generator(gen0, L.device)  # warm start from run001_G
    else:
        # Cold start (README §3.3f ablation, 2026-09-26): Zhou's architecture with
        # fresh weights, no imitation stage. The scale is irrelevant: channel.receive
        # rescales every frame to its JSR.
        G, scale = models.Generator(n_classes=1).to(L.device), 1.0
    G.train()
    G, hist, log_gain = train(L, dfd, G, scale, task["target"], task["beta"], steps, frames, band, seed, mode)
    return G, scale, hist, log_gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--gen0", default=os.path.join(scene.ART, "..", "run001_G.pt"))
    ap.add_argument("--init", choices=["warm", "random"], default="warm",
                    help="warm = start from --gen0 (every reported D2 result); random = "
                         "fresh weights, the no-imitation ablation")
    ap.add_argument("--run", default=RUN, help="output folder under artifacts/cgan/gan/")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="override the per-target JSR band [dB] (default JSR_BANDS / JSR_BAND); "
                         "the wide-band ablation for damage when loud, README §4.3")
    ap.add_argument("--detector-dir", default=None,
                    help="train against a SURROGATE detector (grey-box: the attacker's own CNN, "
                         "trained with train_spectrogram_cnn.py --out-dir); evaluation always uses "
                         "the deployed one")
    ap.add_argument("--power", choices=["fixed", "learned"], default="fixed",
                    help="learned: the generator chooses its power under a hard cap (--jsr-cap)")
    ap.add_argument("--jsr-cap", type=float, default=0.0, help="learned power: per-frame cap, JSR dB")
    ap.add_argument("--jsr-init", type=float, default=-40.0, help="learned power: starting JSR, dB")
    ap.add_argument("--gain-lr", type=float, default=1e-2, help="learned power: Adam lr of the log-gain")
    ap.add_argument("--damage", choices=["ber", "per"], default="ber",
                    help="per: train on log E[frame error rate] instead of log E[BER]")
    ap.add_argument("--shadow-db", type=float, default=0.0,
                    help="log-normal shadowing on the victim's link [dB] (README §3.3k); pair it "
                         "with --detector-dir pointing at that level's retrained defender")
    ap.add_argument("--sync", action="store_true",
                    help="the listening jammer (README §3.3l): lands on R's symbol grid, phase still random")
    ap.add_argument("--rep", type=int, default=0,
                    help="seed replicate: 0 = the original seeds; r > 0 shifts every seed by 1000 r "
                         "(fresh init for --init random, fresh frames either way)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--frames", type=int, default=None,
                    help="frames per gradient step (default: FRAMES[target])")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = os.path.join(scene.ART, "..", "gan", args.run)
    if args.run == RUN and (args.init != "warm" or args.rep != 0 or args.steps != 400 or args.band
                            or args.detector_dir or args.power != "fixed" or args.damage != "ber"
                            or args.shadow_db or args.sync):
        raise SystemExit("run001 holds the original recipe (warm, 400 steps, rep 0); "
                         "write any variant to its own --run folder")
    os.makedirs(out, exist_ok=True)
    device = lk.setup(seed=4000 + args.task + 1000 * args.rep)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    L.shadow_db = args.shadow_db
    L.jammer_sync = args.sync
    dfd = (baselines.Defender(L) if args.detector_dir is None else
           baselines.Defender(L, thr_dir=args.detector_dir,
                              cnn_path=os.path.join(args.detector_dir, "detector_spec.pt")))

    if args.smoke:
        cnn0 = next(i for i, t in enumerate(TASKS) if t["target"] == "spec_cnn")
        tasks = [(0, TASKS[0]), (1, TASKS[1]), (cnn0, TASKS[cnn0])]
        steps = 20
    else:
        tasks = [(args.task, TASKS[args.task])]
        steps = args.steps

    sfx = "_smoke" if args.smoke else ""       # never overwrite a real generator
    for tid, task in tasks:
        tag = "eff" if task["target"] is None else f"{task['target']}_b{task['beta']:g}"
        frames = args.frames if args.frames is not None else FRAMES[task["target"]]
        if args.smoke:
            frames = min(frames, 32)
        print(f"task {tid} = {task}  (tag {tag}, {frames} frames/step, "
              f"JSR band {tuple(args.band) if args.band else JSR_BANDS.get(task['target'], JSR_BAND)})",
              flush=True)
        band = tuple(args.band) if args.band else JSR_BANDS.get(task["target"], JSR_BAND)
        mode = dict(power=args.power, damage=args.damage, cap=args.jsr_cap, init=args.jsr_init,
                    gain_lr=args.gain_lr)
        G, scale, hist, log_gain = run_task(L, dfd, task, args.gen0, steps, frames,
                                            seed=4000 + tid + 1000 * args.rep, band=band, init=args.init,
                                            mode=mode)
        recipe = dict(target=task["target"], beta=task["beta"], steps=steps, frames=frames,
                      lr=2e-4, jsr_band=band, alpha=ALPHA, init=args.init, rep=args.rep,
                      detector=args.detector_dir or "deployed", power=args.power, damage=args.damage,
                      shadow_db=args.shadow_db, sync=args.sync,
                      jsr_cap=args.jsr_cap if args.power == "learned" else None,
                      warm_start=os.path.relpath(args.gen0) if args.init == "warm" else None)
        torch.save({"state_dict": G.state_dict(), "n_classes": 1, "seg_len": models.SEG_LEN,
                    "scale": scale, "target": task["target"], "beta": task["beta"],
                    "tag": tag, "recipe": recipe, "power": args.power, "log_gain": log_gain,
                    "jsr_cap": args.jsr_cap if args.power == "learned" else None},
                   os.path.join(out, f"task{tid}{sfx}_G.pt"))
        with open(os.path.join(out, f"task{tid}{sfx}.json"), "w") as f:
            json.dump(dict(run=args.run, task=task, tid=tid, tag=tag, recipe=recipe, history=hist), f)
        print(f"wrote task{tid}{sfx}_G.pt, task{tid}{sfx}.json  (tag {tag})", flush=True)


if __name__ == "__main__":
    main()
