"""
S1 -- the shadowing ablation's read-out (README §3.3k). Login node, no Sionna.

    python shadow_figures.py                  # tables from snr_ablation/shadow_run003/

Per shadowing level: each detector's clean FAR, P(det) at matched excess BER 3e-4
(the E2 convention, snr_figures._cross_log), frames broken per extra alarm (PER over
P(det) - FAR, best over JSR, excess >= 2 sigma -- the E2b power-free unit), and the
best confirmed stealthy BER inside the budget. At sigma 0 there is no power_csi row:
it equals power_one_sided there by construction (detectors.power_csi).
"""

import argparse
import glob
import json
import math
import os

import numpy as np

from snr_figures import BER_FLOOR, _cross_log

AD = "../artifacts/cgan/snr_ablation"
DETS = ["power_one_sided", "power_csi", "kurtosis", "spec_cnn"]
SHORT = {"power_one_sided": "pow", "power_csi": "csi", "power_two_sided": "pow2",
         "kurtosis": "kurt", "spec_cnn": "cnn", "lrt_noise": "lrt"}
A, BER_REF = "0.05", 3e-4
JAM = [("noise", "noise"), ("pulsed_p1", "matched QPSK"), ("pulsed_p0.1", "Amuru p0.1"),
       ("eff", "GAN beta0"), ("power_one_sided_b10", "GAN pow1s b10"),
       ("spec_cnn_b1", "GAN cnn b1"), ("spec_cnn_b10", "GAN cnn b10"), ("omniscient_e1", "genie")]


def pdet(p, det):
    if det == "power_csi" and det not in p["pdet"]:
        det = "power_one_sided"          # sigma 0: identical by construction
    return p["pdet"].get(det, {}).get(A, np.nan)


def load(run):
    fs = [f for f in glob.glob(os.path.join(AD, run, "shadow_*.json")) if not f.endswith("_smoke.json")]
    return sorted((json.load(open(f)) for f in fs), key=lambda d: d["meta"]["shadow_db"])


def matched(d, pts, det):
    """(P(det), JSR) at the first JSR reaching BER_REF excess BER."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = np.array([p["ber"] for p in pts]) - d["clean"]["ber"]
    x = _cross_log(jsr, np.where(exc >= BER_FLOOR, exc, 0.0), BER_REF)
    if x is None:
        return np.nan, None
    return float(np.interp(x, jsr, [pdet(p, det) for p in pts])), x


def frames_per_alarm(d, pts, det):
    """Best over JSR of excess PER / (P(det) - FAR), extra alarms >= 2 binomial sigma."""
    far, per0 = pdet(d["clean"], det), d["clean"].get("per", 0.0)
    best = np.nan
    for p in pts:
        extra = pdet(p, det) - far
        if "per" in p and extra >= 2 * math.sqrt(far * (1 - far) / p["frames"]):
            v = (p["per"] - per0) / extra
            best = v if np.isnan(best) else max(best, v)
    return best


def summary(levels):
    for d in levels:
        s, c = d["meta"]["shadow_db"], d["clean"]
        print(f"\n=== shadowing {s:g} dB | clean BER {c['ber']:.1e} PER {c.get('per', 0):.4f} | FAR "
              + " ".join(f"{SHORT[k]} {pdet(c, k):.3f}" for k in DETS))
        print(f"  {'jammer':<15}{'JSR@3e-4':>9}  P(det): " + " ".join(f"{SHORT[k]:>6}" for k in DETS)
              + "   frames/extra alarm: " + " ".join(f"{SHORT[k]:>6}" for k in DETS))
        for key, lab in JAM:
            pts = d["attacks"].get(key) or d["generators"].get(key)
            if not pts:
                continue
            row = [matched(d, pts, det) for det in DETS]
            x = row[0][1]
            print(f"  {lab:<15}{('%+.1f' % x) if x is not None else '--':>9}          "
                  + " ".join(f"{p:6.3f}" for p, _ in row) + "                       "
                  + " ".join(f"{frames_per_alarm(d, pts, det):6.2f}" for det in DETS))
        best = {}
        for grp in ("confirmed_attacks", "confirmed_generators"):
            for name, conf in d.get(grp, {}).items():
                if name.startswith("omniscient"):
                    continue                          # the genie is the reference, not a result
                for det, by_a in conf.items():
                    pick = by_a.get(A)
                    if pick and pick.get("passed") and pick["confirmed_ber"] > best.get(det, (0.0, ""))[0]:
                        best[det] = (pick["confirmed_ber"], name)
        print("  confirmed stealthy BER @ alpha 0.05 (non-genie): "
              + "  ".join(f"{SHORT.get(k, k)} {v[0]:.1e} ({v[1]})" for k, v in best.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shadow_run003")
    summary(load(ap.parse_args().run))


if __name__ == "__main__":
    main()
