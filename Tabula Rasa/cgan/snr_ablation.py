"""
E2 -- the noise ablation on the LIVE models (README §3.3g).

The supervisor's mandated primary ablation (README §B.2, verbatim: "Ablation:
parameter study, increase noise and see what happens (less detection e.g.)" +
"Noise level, eps, change exponentially") was discharged once, on the FROZEN sim08
OFDM stack (§3.3c). The live waveform track pinned its noise instead: link.SNR_DB
= 30 dB, because Zhou's protocol says 30 dB and §3.3d power-controls T to 30 dB at
R so one detector calibration serves every drop. The swept axis became JSR.

So every D-series number -- D1, the 20 D2 generators, all four detectors, and the
matched-BER headline of §3.3f -- sits at exactly ONE noise level. This sweeps it.

WHAT IT TESTS
-------------
§3.3f reports a ~30 dB gap between the stealth edge (max JSR with P(det) <= alpha,
~-45 dB) and BER onset (~-16 dB), at SNR 30 dB. Those edges are set by different
quantities: detectability by the fluctuation of the CLEAN statistic, which at 30 dB
is dominated by the signal and should be roughly SNR-independent in JSR; damage by
the NOISE FLOOR, which moves 1 dB per dB of SNR. The hypothesis is therefore that
the 30 dB gap IS the 30 dB SNR, and it predicts a critical SNR where the wall
closes. §3.3c predicts a partial counter-result from the other stack (absolute
stealthy BER was SNR-independent there; below ~12.5 dB the rising clean floor
swallowed the attacker's gain), and the two stacks differ structurally -- this one
is LOS, power-controlled, with near-deterministic clean received power.

TIER 1 ONLY (user decision, 2026-09-23). Generators are EVALUATED at each noise
level, not retrained: this answers "does the 30 dB-trained attack transfer", not
"is the gain achievable at each SNR". The latter is Tier 2 (train_gan.py with
snr_db as an argument) and is gated on what this shows. The CNN's weights are
likewise frozen at 30 dB and only re-calibrated (calibrate_snr.py).

    sbatch --array=6 submit_snr_ablation.sh    # 30 dB alone: MUST reproduce §3.3f
    sbatch --array=0-9 submit_snr_ablation.sh  # the full grid, one level per task

Per noise level (one task):
  1. re-calibrate every detector on clean frames at this SNR (calibrate_snr), since
     every threshold is a quantile of clean frames and alpha stops being alpha
     otherwise -- which is the whole reason the BER-P(det) comparison is matched;
  2. the clean reference: BER floor and each detector's clean FAR (the stealth
     budget, CLAUDE.md);
  3. the classical envelope over JSR -- noise (barrage), the pulsed/Amuru family,
     the omniscient genie -- reusing baselines.measure/confirm unchanged;
  4. all 21 generators (D1 run001_G + the 20 D2 checkpoints) over the same JSR grid,
     reusing eval_gan.sweep unchanged;
  5. confirmation on fresh frames for every frontier pick, as everywhere else.

Output: artifacts/cgan/snr_ablation/<run>/snr_<tag>.json, one per level.

--axis shadow (README §3.3k, 2026-09-27) sweeps the second channel knob instead:
per-frame log-normal shadowing on the victim's link, SHADOW_GRID_DB, at 30 dB. The
defender at each level is the CNN RETRAINED on shadowed frames plus its own
calibration (train_spectrogram_cnn.py --shadow-db, under SHADOW_ROOT/shadow_<s>/),
which adds the gain-aware power detector power_csi. sigma 0 is the deployed
defender, so that level reproduces §3.3f/E2's 30 dB row. Output
<run>/shadow_<s>.json.

    for s in 0.01 0.03 0.1 0.3 1 3; do sbatch submit_train_spectrogram_cnn.sh \\
        --shadow-db $s --out-dir ../artifacts/cgan/baselines/shadow/shadow_$s; done
    sbatch --array=0-6 submit_snr_ablation.sh --axis shadow --run shadow_run003 --gan-run run003
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import argparse
import glob
import json
import os
import re
import time

import torch

import attacks
import baselines
import calibrate_snr
import detectors
import eval_gan
import link as lk
import models
import scene

# Linear in dB IS exponential in noise power, which is the mandate's literal wording.
# The range overlaps sim08's 5-30 dB from below so §3.3c stays comparable (§3.2's
# own argument). 30 dB is IN the grid and must reproduce §3.3f -- the regression
# check, as sim08_ablation re-measures frozen job 102390.
SNR_GRID_DB = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
N_TASKS = len(SNR_GRID_DB) + 1                 # last task = the noiseless anchor

ART = os.path.join(scene.ART, "..", "snr_ablation")
GAN_DIR = os.path.join(scene.ART, "..", "gan", "run001")
PLAIN_G = os.path.join(scene.ART, "..", "run001_G.pt")

# Sweep points whose statistic quantiles are stored, so the frontier can be drawn
# over ALL budgets and not only at alpha (detectors.stat_quantiles; §3.3f caveat).
# Kept to two tags: ~1.5 MB per tag per level, and the home quota is small (§C.4).
STATS_FOR = ("plain_run001", "spec_cnn_b10")


# std of the victim link's per-frame power gain [dB]. The naive energy detector's
# margin at 30 dB is +0.24 % of frame power (-26 dB JSR); with unit-mean shadowing a
# +1.64 sigma fade is exp(1.645 s - s^2/2) - 1, s = sigma ln10/10: +0.4 % at 0.01 dB,
# +3.8 % at 0.1 dB, +142 % at 3 dB -- the grid spans "invisible to the detector" to
# sim01's hand-set +200 % budget (README §3.3k).
SHADOW_GRID_DB = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SHADOW_ROOT = os.path.join(scene.ART, "shadow")


def task_level(task):
    """Array index -> SNR in dB, or None for the noiseless anchor."""
    return SNR_GRID_DB[task] if task < len(SNR_GRID_DB) else None


def shadow_tag(sigma_db):
    return f"shadow_{sigma_db:g}"


def shadow_defender(L, sigma_db):
    """The deployed defender at sigma 0; else the CNN retrained at sigma + its calibration."""
    if sigma_db == 0:
        return baselines.Defender(L)
    d = os.path.join(SHADOW_ROOT, shadow_tag(sigma_db))
    return baselines.Defender(L, thr_dir=d, cnn_path=os.path.join(d, "detector_spec.pt"))


def generators(gan_dir=GAN_DIR):
    """[(tag, path)] for D1 and every D2 checkpoint, D1 first, D2 in task order."""
    out = [("plain_run001", PLAIN_G)]
    paths = sorted((p for p in glob.glob(os.path.join(gan_dir, "task*_G.pt"))
                    if re.search(r"task\d+_G\.pt$", p)),            # skips *_smoke_G.pt
                   key=lambda p: int(re.search(r"task(\d+)_G", p).group(1)))
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        out.append((ckpt.get("tag", os.path.basename(p)[:-5]), p))
    return out


def sweep_classical(L, dfd, spec, jsr_grid, n_frames, n_confirm, keep_stats=False):
    """One classical attack over the JSR grid. baselines.measure/confirm, unchanged."""
    omni = spec["name"] == "omniscient"
    jsr_of = (lambda i: jsr_grid[i]) if omni else (lambda i: [jsr_grid[i]])
    pts = []
    t0 = time.time()
    for i, jsr in enumerate(jsr_grid):
        pts.append(baselines.measure(L, dfd, spec, jsr_of(i), n_frames, keep_stats=keep_stats))
        p = pts[-1]
        print(f"    JSR {jsr:+6.1f} dB  BER {p['ber']:.2e}  P(det)@0.05 " +
              " ".join(f"{d} {p['pdet'][d]['0.05']:.3f}" for d in p["pdet"]) +
              f"  ({time.time() - t0:.0f}s)", flush=True)
    confirmed = baselines.confirm(L, dfd, spec, pts, jsr_of, n_confirm,
                                  dfd.dets, detectors.ALPHAS)
    return pts, confirmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help=f"0..{N_TASKS - 1}")
    ap.add_argument("--run", default="run001")
    ap.add_argument("--gan-run", default="run001",
                    help="D2 generator folder under artifacts/cgan/gan/ (run002 = 4000 steps)")
    ap.add_argument("--frames", type=int, default=512, help="frames per sweep point")
    ap.add_argument("--confirm", type=int, default=4096, help="frames for the confirmation pass")
    ap.add_argument("--n-cal", type=int, default=calibrate_snr.N_CALIBRATION,
                    help="clean frames the detectors are re-calibrated on")
    ap.add_argument("--attacks", default=None,
                    help="comma list of classical attacks (default all); '' to skip them")
    ap.add_argument("--gens", default=None,
                    help="comma list of generator tags (default all 21); '' to skip them")
    ap.add_argument("--stats-for", default=",".join(STATS_FOR),
                    help="tags whose statistic quantiles are stored for the all-budget frontier")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--axis", choices=["snr", "shadow"], default="snr",
                    help="shadow: the task indexes SHADOW_GRID_DB at 30 dB (README §3.3k)")
    ap.add_argument("--sync", action="store_true",
                    help="every jammer lands on R's symbol grid, the listening jammer (README §3.3l); "
                         "same seed as an async run = paired draws (same bits, noise, phases)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    shadow = SHADOW_GRID_DB[args.task] if args.axis == "shadow" else 0.0
    snr = lk.SNR_DB if args.axis == "shadow" else task_level(args.task)
    tag = shadow_tag(shadow) if args.axis == "shadow" else calibrate_snr.level_tag(snr)
    out_dir = os.path.join(ART, args.run)
    os.makedirs(out_dir, exist_ok=True)
    stem = tag if args.axis == "shadow" else f"snr_{tag}"
    out_path = os.path.join(out_dir, f"{stem}{'_smoke' if args.smoke else ''}.json")

    device = lk.setup(seed=args.seed + 100 * args.task)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    L.shadow_db = shadow
    L.jammer_sync = args.sync
    t0 = time.time()

    jsr_grid = [-30.0, -10.0, 0.0] if args.smoke else baselines.JSR_GRID_K1
    frames = 64 if args.smoke else args.frames
    confirm = 256 if args.smoke else args.confirm
    keep = set(t for t in args.stats_for.split(",") if t)

    print(f"task {args.task} | {tag} | SNR {'noiseless' if snr is None else f'{snr:g} dB'} | "
          f"device {device}", flush=True)
    if args.axis == "shadow":
        dfd = shadow_defender(L, shadow)
        print(f"shadowing {shadow:g} dB | detectors: {', '.join(dfd.dets)}  "
              f"(CNN retrained + calibrated on shadowed frames)", flush=True)
    else:
        dfd = calibrate_snr.defender_at(L, snr, n_cal=(1024 if args.smoke else args.n_cal))
        print(f"detectors: {', '.join(dfd.dets)}  (thresholds re-calibrated at this level)", flush=True)

    clean = baselines.measure(L, dfd, None, None, 4 * frames, keep_stats=True)
    print(f"clean: BER {clean['ber']:.3e} (closed form {L.ber_clean(dfd.snr_db):.3e}), "
          f"FAR {json.dumps({d: clean['pdet'][d]['0.05'] for d in clean['pdet']})}", flush=True)

    meta = dict(run=args.run, task=args.task, snr_db=snr, noiseless=snr is None, tag=tag,
                n0=dfd.n0, p_s=dfd.p_s, jsr_db=jsr_grid, n_frames=frames, n_confirm=confirm,
                alphas=detectors.ALPHAS, dets=dfd.dets, seed=args.seed + 100 * args.task,
                n_sym=scene.N_SYM, K=1, tier=1, axis=args.axis, shadow_db=shadow, sync=args.sync,
                cnn=("retrained on shadowed frames (train_spectrogram_cnn --shadow-db)"
                     if shadow else "frozen 30 dB weights, re-calibrated scale + threshold"),
                thresholds=dfd.thr)
    res = dict(meta=meta, clean=clean, attacks={}, generators={},
               confirmed_attacks={}, confirmed_generators={})

    def dump():
        with open(out_path, "w") as f:
            json.dump(res, f)

    dump()

    # ---- 1. the classical envelope ----------------------------------------------
    want = None if args.attacks is None else set(s for s in args.attacks.split(",") if s)
    specs = [s for s in attacks.all_specs()
             if want is None or attacks.spec_name(s) in want]
    for spec in specs:
        name = attacks.spec_name(spec)
        print(f"  [{tag}] classical {name}", flush=True)
        pts, conf = sweep_classical(L, dfd, spec, jsr_grid, frames, confirm,
                                    keep_stats=name in keep)
        res["attacks"][name] = pts
        res["confirmed_attacks"][name] = conf
        dump()                                        # checkpoint per attack
        print(f"  [{tag}] {name} confirmed: {json.dumps(conf)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. the 21 generators ----------------------------------------------------
    gwant = None if args.gens is None else set(g for g in args.gens.split(",") if g)
    gens = [(t, p) for t, p in generators(os.path.join(scene.ART, "..", "gan", args.gan_run)) if gwant is None or t in gwant]
    for gtag, gpath in gens:
        G, scale, _ = models.load_generator(gpath, device)
        spec = dict(name="gan", G=G, scale=scale, tag=gtag)
        print(f"  [{tag}] generator {gtag} (scale {scale:.4f})", flush=True)
        if gtag in keep:
            pts, conf = sweep_classical(L, dfd, spec, jsr_grid, frames, confirm, keep_stats=True)
        else:
            pts, conf = eval_gan.sweep(L, dfd, spec, jsr_grid, frames, confirm)
        res["generators"][gtag] = pts
        res["confirmed_generators"][gtag] = conf
        dump()                                        # checkpoint per generator
        print(f"  [{tag}] {gtag} confirmed: {json.dumps(conf)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    res["runtime_s"] = time.time() - t0
    dump()
    print(f"wrote {os.path.relpath(out_path)} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
