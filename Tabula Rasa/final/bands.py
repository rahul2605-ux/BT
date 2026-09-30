"""
Final experiment, chain step 2 -- the JSR band each generator is trained on (README §3.4,
§3.1 trap 1). New file.

    sbatch submit_bands.sh base              # usually via submit_env.sh base

Why it matters: training samples one JSR per step from the band, and the detector term is
soft P_det = mean sigmoid((stat - threshold)/scale). A band that lies above the target's
transition has every sampled frame saturated-detected, sigmoid' ~ 1e-11, and no beta
recovers a gradient; below it the term is flat at 0 (trap 1 of §3.1).

The rule is S4's (§3.3m), made automatic. Start from the 30 dB band (cgan/train_gan.py
JSR_BANDS: CNN (-48, -16), every other target (-32, 0)) and shift it by how far that
target's transition moved: the JSR where WHITE NOISE reaches P_det 0.5, measured against
this environment's defender minus the same against the deployed 30 dB one. The upper edge
of the analytic targets (energy, kurtosis) is capped at 0 dB. The control keeps (-32, 0).

Deviation from §3.4 (recorded in README §3.3q): the spec says energy and full-band power
transition within ~1 dB at 30 dB, "so power's reference serves for energy". For WHITE
noise they do not: the matched filter passes only 1/10 of a white jammer's frame energy
(128 symbol-energy units at the MF vs 1280 samples of the full band) while the clean
signal x noise spread is the same, so energy's white-noise transition sits 10 dB above
power's (verify.py checks this). For an in-band jammer they agree to ~1.4 dB. The shift is
therefore measured on energy's OWN statistic at both ends: the energy detector calibrated
on 20k clean 30 dB frames. Its 30 dB transition is ~10 dB above power's, and so is its
15 dB one, so the shift -- the only thing the rule uses -- is what S4 measured for power.

Outputs: artifacts/final/<env>/bands.json; artifacts/final/bands_ref30.json (the 30 dB
reference transitions, shared by all environments, computed once and cached).
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
import detectors
import env as E
import link as lk

SNR30_DB = 30.0
BAND30 = {"energy": (-32.0, 0.0), "kurtosis": (-32.0, 0.0), "spec_cnn": (-48.0, -16.0)}
CONTROL_BAND = (-32.0, 0.0)
ANALYTIC = ("energy", "kurtosis")
LEVEL = 0.5
GRID = [float(v) for v in np.arange(-60.0, 15.5, 1.0)]
N_FRAMES = 512
REF_SEED = 7000
MIN_WIDTH_DB = 8.0


def crossing(grid, p, level=LEVEL):
    """First JSR where P_det reaches `level`, linearly interpolated; None if never."""
    for i in range(1, len(grid)):
        if p[i] >= level > p[i - 1]:
            return grid[i - 1] + (level - p[i - 1]) * (grid[i] - grid[i - 1]) / (p[i] - p[i - 1])
    return grid[0] if p and p[0] >= level else None


@torch.no_grad()
def noise_curve(L, dfd, dets, snr_db, grid=GRID, n=N_FRAMES):
    """P_det at alpha 0.05 of white noise vs JSR, for each detector in `dets`."""
    curves = {d: [] for d in dets}
    for jsr in grid:
        out = attacks.frames(L, n, dict(name="noise"), [jsr], snr_db)
        s = dfd.statistics(out["r"], out["z"], dets=dets, gain=out.get("gain"))
        for d in dets:
            curves[d].append(detectors.p_detect(s[d], dfd.threshold(d, E.ALPHA)))
    return curves


def energy30_defender(L30, n_cal=defender.N_CALIBRATION):
    """The one-sided energy detector at 30 dB on the ideal link, calibrated on n_cal clean frames."""
    e = torch.cat([detectors.energy(attacks.frames(L30, min(1024, n_cal - i), None, None, SNR30_DB)["z"])
                   for i in range(0, n_cal, 1024)])
    thr = dict(energy_clean_mean=float(e.mean()),
               energy={str(a): detectors.calibrate(e, a) for a in detectors.ALPHAS})
    return defender.Defender(L30, SNR30_DB, thr, None, None, ["energy"])


def reference30(device, n=N_FRAMES, n_cal=defender.N_CALIBRATION, grid=GRID):
    """
    White-noise transitions at 30 dB against the deployed defender (cgan/'s deployed CNN,
    kurtosis and full-band power thresholds) and the 30 dB energy detector.
    """
    lk.setup(device, seed=REF_SEED)
    L30 = lk.Link(**lk.LINK)                               # the ideal link: no noise factor, no fading
    dep = defender.Defender.load(L30, SNR30_DB, os.path.join(E.CGAN_ART, "baselines"),
                                 dets=["power_one_sided", "kurtosis", "spec_cnn"])
    en = energy30_defender(L30, n_cal)
    curves = noise_curve(L30, dep, ["power_one_sided", "kurtosis", "spec_cnn"], SNR30_DB, grid, n)
    curves.update(noise_curve(L30, en, ["energy"], SNR30_DB, grid, n))
    return dict(snr_db=SNR30_DB, grid=grid, n_frames=n, level=LEVEL, curves=curves,
                transition={d: crossing(grid, c) for d, c in curves.items()},
                energy_threshold=en.thr, defender="artifacts/cgan/baselines (deployed, 30 dB)")


def bands_from(ref, trans_env):
    """{task target: band} and the shift per target, from the two transition dicts."""
    out, notes = {"control": dict(band=list(CONTROL_BAND), shift=None, target=None)}, []
    for tgt, (lo, hi) in BAND30.items():
        t30, te = ref["transition"].get(tgt), trans_env.get(tgt)
        if t30 is None or te is None:
            raise SystemExit(f"{tgt}: no P_det {LEVEL} transition (30 dB {t30}, env {te}) -- cannot place its band")
        shift = te - t30
        b = [lo + shift, hi + shift]
        if tgt in ANALYTIC:
            b[1] = min(b[1], 0.0)
        if b[1] - b[0] < MIN_WIDTH_DB:
            notes.append(f"{tgt}: band {b} narrower than {MIN_WIDTH_DB} dB, lower edge moved down")
            b[0] = b[1] - MIN_WIDTH_DB
        out[tgt] = dict(band=[round(b[0], 1), round(b[1], 1)], shift=shift, target=tgt,
                        transition_30=t30, transition_env=te)
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    args = ap.parse_args()
    t0 = time.time()
    n = 64 if E.SMOKE else N_FRAMES
    n_cal = 2048 if E.SMOKE else defender.N_CALIBRATION
    device = lk.setup(seed=REF_SEED)

    ref_path = os.path.join(E.ART_ROOT, "bands_ref30.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)
        print(f"30 dB reference from {os.path.relpath(ref_path)}", flush=True)
    else:
        ref = reference30(device, n, n_cal)
        os.makedirs(E.ART_ROOT, exist_ok=True)
        with open(ref_path + f".{os.getpid()}", "w") as f:
            json.dump(ref, f, indent=2)
        os.replace(ref_path + f".{os.getpid()}", ref_path)
        print(f"30 dB reference computed ({time.time() - t0:.0f}s)", flush=True)
    print("  30 dB white-noise transitions (P_det 0.5): "
          + "  ".join(f"{d} {v:+.2f}" for d, v in ref["transition"].items() if v is not None), flush=True)

    lk.setup(device, seed=REF_SEED + 1)
    L = E.apply(lk.Link(**lk.LINK), args.env)
    dfd = defender.Defender.load(L, E.SNR_DB, E.art(args.env, "cnn"))
    dets = ["energy", "kurtosis", "spec_cnn"]
    curves = noise_curve(L, dfd, dets, E.SNR_DB, GRID, n)
    trans = {d: crossing(GRID, c) for d, c in curves.items()}
    print(f"  env {args.env} white-noise transitions: "
          + "  ".join(f"{d} {v:+.2f}" if v is not None else f"{d} none" for d, v in trans.items()), flush=True)
    bands, notes = bands_from(ref, trans)
    for k, v in bands.items():
        print(f"  band[{k}] = {v['band']}" + (f"  (shift {v['shift']:+.2f} dB)" if v["shift"] is not None else ""),
              flush=True)
    for s in notes:
        print(f"  NOTE {s}", flush=True)
    res = dict(env=args.env, env_config=E.ENVS[args.env], snr_db=E.SNR_DB, grid=GRID, n_frames=n,
               level=LEVEL, curves=curves, transition=trans, bands=bands, notes=notes,
               band30=BAND30, control_band=CONTROL_BAND, runtime_s=time.time() - t0)
    with open(E.art(args.env, "bands.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {os.path.relpath(E.art(args.env, 'bands.json'))} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
