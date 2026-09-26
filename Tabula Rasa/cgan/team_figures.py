"""
E3 summary (login node; reads the team_fading.py JSONs, no Sionna).

    python team_figures.py              # matched-BER tables
    python team_figures.py --smoke      # the same on the *_smoke.json files
    python team_figures.py --timing     # D4a: the delay-decay tables (team_timing/run001)

Per (SNR, channel) and series (jammer | K | timing): the nominal total JSR the
series needs for EXCESS BER 3e-4 (README §3.3f's operating point) and each
detector's P(det) at alpha = 0.05 there, interpolated exactly as snr_figures.py
does for E2, so the lossless K = 1 rows are directly comparable to §3.3g. The
per-drone transmit power is that JSR minus 10 log10 K (equal split).

Then the two numbers the check exists for, per jammer and detector:
  gap       P(det) faded K=1 - P(det) lossless K=1     (what the channel costs)
  recovery  (faded K=1 - faded K) / gap                (share a team wins back)
A recovery is only printed where the gap exceeds ~2 binomial SE (512 frames).

--timing (D4a, README §4.2 Q12): per SNR, generator and K, the same matched-BER
cell for every inter-jammer timing error sigma, then 'shifted' (no coordination,
the floor) -- with the single jammer (K = 1, the containment ceiling) on top.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

import snr_figures as sf

AD = "../artifacts/cgan/team_fading/run001"
AD_TIMING = "../artifacts/cgan/team_timing/run001"
DETS = ["spec_cnn", "power_one_sided", "kurtosis"]
CHANS = ["lossless", "rician10", "rayleigh"]


def load(smoke=False, ad=AD):
    out = {}
    for f in sorted(glob.glob(os.path.join(ad, "*.json"))):
        if f.endswith("_smoke.json") != smoke:
            continue
        d = json.load(open(f))
        out[(d["meta"]["snr_db"], d["meta"]["channel"])] = d
    return out


def matched(d, key, det, target=sf.BER_REF):
    """(P(det), total JSR) where series `key` first reaches `target` excess BER."""
    pts = d["series"].get(key)
    if not pts:
        return None, None
    jsr = np.array(d["meta"]["jsr_db"], float)
    x = sf._cross_log(jsr, sf.excess(pts, d["clean"]["ber"]), target)
    if x is None:
        return None, None
    p = np.interp(x, jsr, np.array([q["pdet"][det]["0.05"] for q in pts], float))
    return float(p), float(x)


def fmt(p, x):
    return "      --      " if p is None else f"{p:5.3f} @{x:+6.1f}"


def timing_tables(data, ber):
    """D4a: rows = timing (sigma ascending, then 'shifted'), columns = detectors."""
    for (snr, _), d in sorted(data.items()):
        keys = list(d["series"])
        for jam in dict.fromkeys(k.split("|")[0] for k in keys):
            for K in sorted({k.split("|")[1] for k in keys if k.startswith(jam + "|")} - {"K1"}):
                print(f"\nSNR {snr:g} dB | {jam} | {K} | P(det) @ total JSR [dB] at excess BER {ber:g}")
                print(f"  {'timing':<14}" + "".join(f"{det:>18}" for det in DETS))
                rows = [(f"{jam}|K1|aligned", "single (K=1)")] + \
                    [(k, k.split("|")[2]) for k in keys if k.startswith(f"{jam}|{K}|")]
                for key, label in rows:
                    print(f"  {label:<14}" + "".join(f"{fmt(*matched(d, key, det, ber)):>18}"
                                                     for det in DETS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timing", action="store_true", help="D4a delay-decay tables")
    ap.add_argument("--ber", type=float, default=sf.BER_REF)
    args = ap.parse_args()
    if args.timing:
        data = load(args.smoke, AD_TIMING)
        if not data:
            print("no team_timing JSONs yet")
        timing_tables(data, args.ber)
        return
    data = load(args.smoke)
    if not data:
        print("no team_fading JSONs yet")
        return
    se = 2 * math.sqrt(0.25 / 512)                      # ~2 binomial SE at the worst p
    for snr in sorted({k[0] for k in data}):
        chans = [c for c in CHANS if (snr, c) in data]
        keys = list(dict.fromkeys(k for c in chans for k in data[(snr, c)]["series"]))
        for det in DETS:
            print(f"\nSNR {snr:g} dB | {det} | P(det) @ total JSR [dB] at excess BER {args.ber:g}")
            print(f"  {'series':<28}" + "".join(f"{c:>16}" for c in chans))
            for key in keys:
                cells = [fmt(*matched(data[(snr, c)], key, det, args.ber)) for c in chans]
                print(f"  {key:<28}" + "".join(f"{s:>16}" for s in cells))
            # gap and recovery, per jammer
            for jam in dict.fromkeys(k.split("|")[0] for k in keys):
                ref, _ = matched(data[(snr, "lossless")], f"{jam}|K1|aligned", det, args.ber) \
                    if (snr, "lossless") in data else (None, None)
                for c in chans[1:]:
                    p1, _ = matched(data[(snr, c)], f"{jam}|K1|aligned", det, args.ber)
                    if ref is None or p1 is None:
                        continue
                    gap = p1 - ref
                    line = f"    {jam:<14} {c:<9} gap {gap:+.3f}"
                    for key in keys:
                        j, K, t = key.split("|")
                        if j != jam or K == "K1":
                            continue
                        pk, _ = matched(data[(snr, c)], key, det, args.ber)
                        if pk is None:
                            continue
                        rec = f"{(p1 - pk) / gap:+.2f}" if abs(gap) > se else "n/a"
                        line += f" | {K} {t[:5]} P {pk:.3f} rec {rec}"
                    print(line)


if __name__ == "__main__":
    main()
