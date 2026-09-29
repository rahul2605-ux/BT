"""
D6 -- the arms race, round 1: the DEFENDER retrains, the attacker stays frozen
(README §3.3o).

    sbatch --array=0-4 submit_arms_cnn.sh     # the five retrained CNNs (train_spectrogram_cnn.py)
    sbatch --array=0-5 submit_arms_eval.sh    # one (SNR, defender) per task -> this script

The supervisor's mandate is that adaptation cost is the headline, not "the jammer
evades the CNN" (README §2.9). Round 1 asks what one retraining of Li et al.'s CNN
buys against the run003 attackers, and what it costs on the jammers Li et al.
trained it for. Three defenders per SNR, all the same recipe and seed:
  r0  the round-0 CNN. At 30 dB the deployed detector the attackers were trained
      against. At 15 dB the same recipe retrained at 15 dB: every CNN is trained
      at its own SNR (user, 2026-09-28), so 15 dB compares like with like;
  A   attacker-AGNOSTIC: Li's four types with the jammed frames' JSR widened from
      U[-20, +10] to U[-35, +10] dB, no generator frames. The attackers work at
      ~-23 dB, below everything r0 was shown: A closing the gap would make it a
      training-RANGE gap, fixed without ever seeing the attacker;
  B   attacker-AWARE: A + the round-0 generators' frames (run003 CNN beta 1 / 10)
      as a fifth jammed type of 204 frames, same widened range.

Each defender is scored on the same frames: every jammer is re-seeded from its
position in the full list, so the JSONs at one SNR are paired draw for draw
(identical BER and PER; only the detector statistics differ) -- on ONE GPU model:
CUDA's random streams differ between card types (measured 2026-09-28: the same
seed-11 path gives -44.25 dB on an RTX 3090 and -44.28 dB on a 2080 Ti as the
colour scale's floor), so submit_arms_eval.sh pins titan_rtx. Per task:
  1. clean frames: BER floor and each detector's realised FAR;
  2. Li et al.'s own classes on fixed frames -- P(det) at alpha vs JSR per type,
     and argmax accuracy / FAR + AUC on Li's class mixture at the original and the
     widened range. This is the COST side: Phase 0.5 (README §A.6) lost 99.8 -> 90.5 %
     accuracy by retraining a CNN on in-band jammers;
  3. the jammers over baselines.JSR_GRID_K1: the classical references and the
     generators, by role --
        control   run003 beta = 0 (no detection term; CNN-visible at r0),
        seen      the round-0 attackers, i.e. arm B's training set,
        twin      trained from the SAME init seed as a round-0 attacker (relative
                  weight distance 0.12-0.15 to it, vs 0.22-0.32 between independent
                  seeds; measured 2026-09-28, asserted in verify.py §23),
        held-out  the independent beta = 10 seeds: the generalisation number.
No confirmation pass: every D6 read-out -- P(det) at matched BER, damage at
P(det) <= 0.5, frames per extra alarm -- comes from the sweep (arms_figures.py).

Output: artifacts/cgan/baselines/arms/eval/<run>/snr<snr>_<arm>.json.
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
import train_spectrogram_cnn as tsc

ARMS = os.path.join(scene.ART, "arms")
GAN = os.path.join(scene.ART, "..", "gan")
TASKS = [(30.0, "r0"), (30.0, "A"), (30.0, "B"), (15.0, "r0"), (15.0, "A"), (15.0, "B")]
WIDE_DB = (-35.0, 10.0)                        # arm A/B's jammed-frame JSR range (submit_arms_cnn.sh)
CLASSICAL = ["noise", "pulsed_p1", "pulsed_p0.1"]
GENS = [("eff", "run003/task0_G.pt", "control"),
        ("r0_b1", "run003/task13_G.pt", "seen"),
        ("r0_b10", "run003/task14_G.pt", "seen"),
        ("twin_cold_b10", "cold002_4k/task14_G.pt", "twin"),
        ("twin_grey_b1", "run004_grey/task13_G.pt", "twin"),
        ("twin_grey_b10", "run004_grey/task14_G.pt", "twin"),
        ("held_r1_b10", "cold002_4k_r1/task14_G.pt", "held-out"),
        ("held_r2_b10", "cold002_4k_r2/task14_G.pt", "held-out"),
        ("held_r3_b10", "cold002_4k_r3/task14_G.pt", "held-out")]
LI_GRID = [float(v) for v in np.arange(-50.0, 10.5, 2.0)]
N_LI, N_MIX = 256, 4               # frames per (type, JSR) point; repeats of Li's 1578-frame mixture


def cnn_dir(snr, arm):
    """The defender's folder; None = the deployed detector (30 dB round 0)."""
    return None if (snr == lk.SNR_DB and arm == "r0") else os.path.join(ARMS, f"snr{snr:g}_{arm}")


def defender(L, snr, arm):
    d = cnn_dir(snr, arm)
    if d is None:
        return baselines.Defender(L)
    return baselines.Defender(L, snr_db=snr, thr_dir=d, cnn_path=os.path.join(d, "detector_spec.pt"))


@torch.no_grad()
def li_test(L, dfd, seed):
    """The CNN on Li et al.'s own classes, on frames every defender at this SNR sees identically."""
    a = detectors.HEADLINE_ALPHA
    thr = dfd.threshold("spec_cnn", a)
    cnn = lambda r: detectors.cnn_statistic(dfd.net, r, dfd.scale)
    lk.setup(L.device, seed=seed)
    curves = {k: [detectors.p_detect(cnn(tsc.batched_frames(
        L, N_LI, k, torch.full((N_LI,), g, device=L.device), dfd.snr_db)), thr) for g in LI_GRID]
        for k in attacks.LI_TYPES}
    mix = {}
    for name, rng in (("li_range", tsc.JSR_RANGE_DB), ("wide_range", WIDE_DB)):
        gen = torch.Generator(device=L.device).manual_seed(seed)
        s, y, kinds = [], [], []
        for _ in range(N_MIX):
            r, yy, kk, _ = tsc.dataset(L, gen, rng, snr_db=dfd.snr_db)
            s.append(cnn(r)); y.append(yy); kinds.append(kk)
        s, y, kinds = torch.cat(s), torch.cat(y), np.concatenate(kinds)
        pred, clean = (s > 0).long(), y == 0
        by_kind = lambda k: torch.as_tensor(kinds == k, device=s.device)
        mix[name] = dict(
            jsr_range_db=list(rng), n=int(len(y)), accuracy=float((pred == y).double().mean()),
            far_argmax=float(pred[clean].double().mean()), dr_argmax=float(pred[~clean].double().mean()),
            far_alpha=detectors.p_detect(s[clean], thr), dr_alpha=detectors.p_detect(s[~clean], thr),
            auc=tsc.roc(s, y)[2],
            per_type_argmax={k: float((pred[by_kind(k)] == (0 if k == "clean" else 1)).double().mean())
                             for k in ["clean"] + attacks.LI_TYPES},
            per_type_alpha={k: detectors.p_detect(s[by_kind(k)], thr) for k in attacks.LI_TYPES})
    return dict(alpha=a, jsr_db=LI_GRID, n_per_point=N_LI, pdet_vs_jsr=curves, mixture=mix)


def jammers(device):
    """[(name, spec, role)], classical first; the order fixes each jammer's seed."""
    specs = {attacks.spec_name(s): s for s in attacks.all_specs()}
    out = [(n, specs[n], "classical") for n in CLASSICAL]
    for tag, rel, role in GENS:
        G, scale, _ = models.load_generator(os.path.join(GAN, rel), device)
        out.append((tag, dict(name="gan", G=G, scale=scale, tag=tag), role))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help=f"0..{len(TASKS) - 1}: {TASKS}")
    ap.add_argument("--run", default="round1")
    ap.add_argument("--frames", type=int, default=512, help="frames per sweep point")
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    snr, arm = TASKS[args.task]
    out_dir = os.path.join(ARMS, "eval", args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"snr{snr:g}_{arm}{'_smoke' if args.smoke else ''}.json")

    device = lk.setup(seed=args.seed)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = defender(L, snr, arm)
    jam = jammers(device)
    grid = [-30.0, -10.0, 0.0] if args.smoke else baselines.JSR_GRID_K1
    frames = 64 if args.smoke else args.frames
    t0 = time.time()
    print(f"task {args.task} | SNR {snr:g} dB | defender {arm} ({cnn_dir(snr, arm) or 'deployed'}) | "
          f"detectors {', '.join(dfd.dets)} | device {device}", flush=True)

    meta = dict(run=args.run, task=args.task, snr_db=snr, arm=arm, cnn_dir=cnn_dir(snr, arm),
                dets=dfd.dets, jsr_db=grid, n_frames=frames, alphas=detectors.ALPHAS, seed=args.seed,
                n_sym=scene.N_SYM, K=1, shadow_db=0.0, wide_jsr_db=list(WIDE_DB),
                roles={name: role for name, _, role in jam},
                gens={tag: rel for tag, rel, _ in GENS}, thresholds=dfd.thr)
    lk.setup(device, seed=args.seed)
    clean = baselines.measure(L, dfd, None, None, 4 * frames, keep_stats=True)
    print(f"clean: BER {clean['ber']:.3e}, FAR "
          f"{json.dumps({d: clean['pdet'][d]['0.05'] for d in clean['pdet']})}", flush=True)
    res = dict(meta=meta, clean=clean, li=None, attacks={}, generators={})

    def dump():
        with open(out_path, "w") as f:
            json.dump(res, f)

    res["li"] = li_test(L, dfd, args.seed + 500)
    for name, m in res["li"]["mixture"].items():
        print(f"Li classes, {name} {m['jsr_range_db']}: accuracy {m['accuracy']:.4f}, FAR (argmax) "
              f"{m['far_argmax']:.4f}, DR at alpha {m['dr_alpha']:.4f}, AUC {m['auc']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    dump()

    for i, (name, spec, role) in enumerate(jam):
        lk.setup(device, seed=args.seed + 1 + i)     # paired across defenders at this SNR
        pts = []
        for jsr in grid:
            pts.append(baselines.measure(L, dfd, spec, [jsr], frames))
            p = pts[-1]
            print(f"  {name:<14} JSR {jsr:+6.1f} dB  BER {p['ber']:.2e}  PER {p['per']:.3f}  P(det)@0.05 "
                  + " ".join(f"{d} {p['pdet'][d]['0.05']:.3f}" for d in p["pdet"]), flush=True)
        res["attacks" if role == "classical" else "generators"][name] = pts
        dump()
        print(f"  [{arm} @ {snr:g} dB] {name} ({role}) done ({time.time() - t0:.0f}s)", flush=True)

    res["runtime_s"] = time.time() - t0
    dump()
    print(f"wrote {os.path.relpath(out_path)} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
