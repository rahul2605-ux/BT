"""
Final experiment -- the regression gate: final/'s copied code vs S4's JSON, before any
environment runs (README §3.4). New file. Needs Sionna: runs inside submit_verify.sh.

    python regress.py            # exit 0 = the copy reproduces S4

The copied code (link, channel, attacks, detectors with full-band power, defender.measure)
re-measures S4's evaluation (§3.3m): S4's four generators (artifacts/cgan/gan/snr15_003:
eff, power_one_sided_b10, spec_cnn_b1, spec_cnn_b10) and the classical attacks of that run
(noise, pulsed p 1 / 0.5 / 0.25 / 0.1, omniscient eta 0.1 / 1), against S4's defender
(artifacts/cgan/baselines/arms/snr15_r0: the CNN retrained at 15 dB + its calibration), at
15 dB, JSR -50..+15 dB, 512 frames per point, on one-sided full-band power, kurtosis and the
CNN. Reference: artifacts/cgan/snr_ablation/snr15_run003/snr_snr_15.json.

The gate is statistical (bit-exact only on the same GPU model and draw order, §3.3n; S4
also drew confirmation passes between attackers). Tests, as cgan/regress_snr30.py with its
known bug fixed:
  * P(det): per point |new - old| <= 4 sigma of a DIFFERENCE of two binomial estimates,
    sigma = sqrt(2 p(1-p)/n) at the pooled p (floored at 1/n) -- the √2 correction;
  * BER: compared as ERROR COUNTS, only where both have >= 25 errors, capped at the frame
    count (bursty jammers concentrate errors in frames), 4-sigma Poisson band in log10;
  * no global bias: mean P(det) z within +-0.15 and BER deviations 35-65 % positive with
    mean within +-0.15 band units;
  * the clean realised FAR within the same P(det) band.
Output: artifacts/final/regress/s4_gate.json (the new measurements + the verdict).
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import

import json
import math
import os
import sys
import time

import numpy as np

import defender
import env as E
import link as lk
import models

S4_JSON = os.path.join(E.CGAN_ART, "snr_ablation", "snr15_run003", "snr_snr_15.json")
S4_DEFENDER = os.path.join(E.CGAN_ART, "baselines", "arms", "snr15_r0")
S4_GANS = os.path.join(E.CGAN_ART, "gan", "snr15_003")
GENS = [("eff", "task0_G.pt"), ("power_one_sided_b10", "task2_G.pt"), ("spec_cnn_b1", "task13_G.pt"),
        ("spec_cnn_b10", "task14_G.pt")]
CLASSICAL = [("noise", dict(name="noise"))] + \
            [(f"pulsed_p{p:g}", dict(name="pulsed", p=p)) for p in (1.0, 0.5, 0.25, 0.1)] + \
            [(f"omniscient_e{e:g}", dict(name="omniscient", eta=e)) for e in (0.1, 1.0)]
A = "0.05"
K_SIGMA, MIN_ERRORS = 4.0, 25
SEED = 3300                      # S4's (snr_ablation task 3: 3000 + 100 * 3)


def pdet_z(new, old, n):
    p = min(max(0.5 * (new + old), 1.0 / n), 1.0 - 1.0 / n)
    return (new - old) / math.sqrt(2.0 * p * (1.0 - p) / n)


def ber_dev(a, b, frames):
    """Deviation of log10 BER in units of the 4-sigma frame-capped Poisson band; None if too few errors."""
    if a["errors"] < MIN_ERRORS or b["errors"] < MIN_ERRORS:
        return None
    n1, n2 = min(a["errors"], frames), min(b["errors"], frames)
    return (math.log10(a["ber"]) - math.log10(b["ber"])) / (K_SIGMA * math.sqrt(1.0 / n1 + 1.0 / n2) / math.log(10))


def compare(new_pts, old_pts, n):
    zs, devs = [], []
    for a, b in zip(new_pts, old_pts):
        for d in defender.GATE_DETS:
            zs.append((d, pdet_z(a["pdet"][d][A], b["pdet"][d][A], n)))
        v = ber_dev(a, b, n)
        if v is not None:
            devs.append(v)
    return zs, devs


def main():
    t0 = time.time()
    with open(S4_JSON) as f:
        s4 = json.load(f)
    grid, n = s4["meta"]["jsr_db"], s4["meta"]["n_frames"]
    device = lk.setup(seed=SEED)
    L = lk.Link(**lk.LINK)                                  # the ideal link, as S4
    snr = s4["meta"]["snr_db"]
    dfd = defender.Defender.load(L, snr, S4_DEFENDER, dets=defender.GATE_DETS)
    print(f"regression gate vs S4 ({os.path.relpath(S4_JSON)}): {len(grid)} JSR points x {n} frames, "
          f"SNR {snr:g} dB, detectors {dfd.dets}", flush=True)
    new = dict(clean=defender.measure(L, dfd, None, None, 4 * n), attacks={}, generators={})
    rows = []
    for tag, spec in CLASSICAL:
        omni = spec["name"] == "omniscient"
        new["attacks"][tag] = [defender.measure(L, dfd, spec, j if omni else [j], n) for j in grid]
        rows.append((f"classical {tag}", new["attacks"][tag], s4["attacks"][tag]))
        print(f"  measured {tag} ({time.time() - t0:.0f}s)", flush=True)
    for tag, fn in GENS:
        G, scale, _ = models.load_generator(os.path.join(S4_GANS, fn), device)
        spec = dict(name="gan", G=G, scale=scale, tag=tag)
        new["generators"][tag] = [defender.measure(L, dfd, spec, [j], n) for j in grid]
        rows.append((f"generator {tag}", new["generators"][tag], s4["generators"][tag]))
        print(f"  measured {tag} ({time.time() - t0:.0f}s)", flush=True)

    ok, all_z, all_dev, report = True, [], [], []
    for label, a, b in rows:
        zs, devs = compare(a, b, n)
        bad_p = [(d, z) for d, z in zs if abs(z) > K_SIGMA]
        bad_b = [v for v in devs if abs(v) > 1]
        good = not bad_p and not bad_b
        ok &= good
        all_z += [z for _, z in zs]
        all_dev += devs
        report.append(dict(label=label, pdet_out=len(bad_p), pdet_n=len(zs), ber_out=len(bad_b), ber_n=len(devs),
                           max_z=max(abs(z) for _, z in zs), max_ber_dev=max((abs(v) for v in devs), default=0.0)))
        print(f"  [{'OK ' if good else 'FAIL'}] {label:<30} P(det): {len(bad_p)}/{len(zs)} beyond {K_SIGMA:g} sigma "
              f"(max |z| {max(abs(z) for _, z in zs):.2f}) | BER: {len(bad_b)}/{len(devs)} outside the band "
              f"(max {max((abs(v) for v in devs), default=0):.2f} band units)", flush=True)
    far_z = {d: pdet_z(new["clean"]["pdet"][d][A], s4["clean"]["pdet"][d][A], 4 * n) for d in defender.GATE_DETS}
    far_ok = all(abs(z) <= K_SIGMA for z in far_z.values())
    print("  clean FAR new vs S4: " + "  ".join(
        f"{d} {new['clean']['pdet'][d][A]:.4f} vs {s4['clean']['pdet'][d][A]:.4f} (z {far_z[d]:+.2f})"
        for d in defender.GATE_DETS), flush=True)
    mean_z = float(np.mean(all_z))
    pos = sum(v > 0 for v in all_dev) / max(len(all_dev), 1)
    mean_dev = float(np.mean(all_dev)) if all_dev else 0.0
    unbiased = abs(mean_z) <= 0.15 and 0.35 <= pos <= 0.65 and abs(mean_dev) <= 0.15
    print(f"  global: P(det) mean z {mean_z:+.3f} over {len(all_z)} pairs ({sum(abs(z) > 3 for z in all_z)} beyond "
          f"3 sigma, {len(all_z) * 0.0027:.1f} expected by chance) | BER {pos:.2f} positive, mean {mean_dev:+.3f} "
          f"band units over {len(all_dev)} points -> {'unbiased' if unbiased else 'BIASED'}", flush=True)
    ok = ok and far_ok and unbiased
    os.makedirs(os.path.join(E.ART_ROOT, "regress"), exist_ok=True)
    with open(os.path.join(E.ART_ROOT, "regress", "s4_gate.json"), "w") as f:
        json.dump(dict(passed=bool(ok), rows=report, far_z=far_z, mean_z=mean_z, ber_pos=pos, ber_mean_dev=mean_dev,
                       n_pdet_pairs=len(all_z), n_ber_points=len(all_dev), reference=os.path.relpath(S4_JSON),
                       defender=os.path.relpath(S4_DEFENDER), seed=SEED, measured=new,
                       runtime_s=time.time() - t0), f)
    print("\n" + ("GATE PASSES -- the copy reproduces S4" if ok else "GATE FAILS -- no environment runs on this copy"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
