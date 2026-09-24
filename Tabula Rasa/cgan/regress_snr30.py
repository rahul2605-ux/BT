"""E2 regression gate: the ablation's 30 dB level vs the deployed §3.3d/§3.3f artifacts.

Task 6 of snr_ablation re-measures the SAME configuration the D-series was reported
at, with fresh seeds, so it must reproduce it. Run this before spending the array --
it is the cgan/ analogue of sim08_ablation re-measuring frozen job 102390.

    python regress_snr30.py        # login node: reads JSONs only, no Sionna
    exit 0 = reproduces

Every quantity should agree to MONTE-CARLO noise -- and the test has to use the
right noise model for EACH quantity, or it flags the measurement floor instead of
a regression. Both traps below were live: a first version of this gate reported
MISMATCH on 6 rows purely because of them.

  P(det): binomial over n_frames. 4 sigma at 512 frames is 0.088 near p = 0.5.
  BER:    Poisson in the ERROR COUNT, not in the BER, AND capped at the number of
          FRAMES. Two separate traps. (i) At 131 072 bits per point a BER of 7.6e-6
          is ONE error, and 1 vs 3 errors is a factor of 3 with no significance --
          so points below MIN_ERRORS are skipped. (ii) Bits are not independent:
          each frame draws its own jammer symbols, delay and phase, so a pulsed or
          bursty jammer concentrates its errors in few frames and the independent
          unit is the FRAME. verify.py says the same thing in mc_tol's `n_eff`.
          The effective count is therefore min(errors, frames).

Bias is what a regression would look like, so the run is also scored globally: a
real shift is one-signed, noise is not.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(HERE, "..", "artifacts/cgan/snr_ablation/run001/snr_snr_30.json")
BD = os.path.join(HERE, "..", "artifacts/cgan/baselines/run001/sweep_K1.json")
GD = os.path.join(HERE, "..", "artifacts/cgan/gan/run001")
A = "0.05"
DETS = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn"]
MIN_ERRORS = 25          # below this the count is not a BER measurement
K_SIGMA = 4.0


def pairs():
    """(label, E2 30 dB points, deployed points, jsr grid, frames) for every comparable row."""
    ab = json.load(open(AB))
    jsr, frames = ab["meta"]["jsr_db"], ab["meta"]["n_frames"]
    old = json.load(open(BD))
    assert old["jsr_db"] == jsr, "JSR grid differs from sweep_K1"
    for name, pts in old["attacks"].items():
        if name in ab["attacks"]:
            yield f"classical {name}", ab["attacks"][name], pts, jsr, frames
    for tag in ab["generators"]:
        f = os.path.join(GD, f"{tag}.json")
        if os.path.exists(f):
            g = json.load(open(f))
            if g["jsr_db"] == jsr:
                yield f"generator {tag}", ab["generators"][tag], g["points"], jsr, frames


def compare(new_pts, old_pts, frames):
    """
    -> (P(det) pairs [(det, old, new)], BER deviations in band-units, P(det) tolerance).
    A band-unit is the 4-sigma Poisson band on the frame-capped error count, so |x| > 1
    is outside it and the SIGN is what a bias would show.
    """
    p_tol = K_SIGMA * math.sqrt(0.25 / frames)
    pd, dev = [], []
    for a, b in zip(new_pts, old_pts):
        for d in DETS:
            x = a["pdet"].get(d, {}).get(A); y = b["pdet"].get(d, {}).get(A)
            if x is not None and y is not None:
                pd.append((d, y, x))
        e1, e2 = a["errors"], b["errors"]
        if e1 >= MIN_ERRORS and e2 >= MIN_ERRORS:
            d_log = math.log10(a["ber"]) - math.log10(b["ber"])
            n1, n2 = min(e1, frames), min(e2, frames)
            dev.append(d_log / (K_SIGMA * math.sqrt(1.0 / n1 + 1.0 / n2) / math.log(10)))
    return pd, dev, p_tol


def collect():
    """Everything the gate scores, pooled -- for snr_figures.fig_regression."""
    rows, pd, dev = [], [], []
    for label, new, old, jsr, frames in pairs():
        p, dv, tol = compare(new, old, frames)
        rows.append((label, p, dv, tol))
        pd += p
        dev += dv
    return rows, pd, dev


def main():
    if not os.path.exists(AB):
        sys.exit(f"no 30 dB ablation yet: {AB}")
    ab = json.load(open(AB))
    print(f"30 dB regression vs §3.3d/§3.3f  ({ab['meta']['n_frames']} frames/point; BER compared "
          f"only where both have >= {MIN_ERRORS} errors, against a {K_SIGMA:g}-sigma Poisson band)")
    rows, _, dev = collect()
    ok = True
    for label, p, dv, tol in rows:
        bad_p = sum(abs(n - o) > tol for _, o, n in p)
        bad_b = sum(abs(v) > 1 for v in dv)
        good = bad_p == 0 and bad_b == 0
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {label:<26} "
              f"P(det): {bad_p}/{len(p)} out, max {max(abs(n - o) for _, o, n in p):.3f} "
              f"(tol {tol:.3f}) | BER: {bad_b}/{len(dv)} out, "
              f"max {max((abs(v) for v in dv), default=0):.2f} band-units")
    cn = ab["clean"]["pdet"]; co = json.load(open(BD))["clean"]["pdet"]
    print("  clean FAR  " + "  ".join(
        f"{d.replace('_', ''):<12} {cn.get(d, {}).get(A, float('nan')):.4f} vs "
        f"{co.get(d, {}).get(A, float('nan')):.4f}" for d in DETS))
    pos = sum(1 for v in dev if v > 0)
    mean = sum(dev) / len(dev) if dev else 0.0
    biased = abs(mean) > 0.15 or not (0.35 <= pos / max(len(dev), 1) <= 0.65)
    print(f"  global BER bias: {pos}/{len(dev)} positive, mean {mean:+.3f} band-units "
          f"-> {'BIASED' if biased else 'unbiased scatter'}")
    ok = ok and not biased
    print("\n" + ("30 dB REPRODUCES §3.3f -- safe to run the array" if ok
                  else "MISMATCH -- do not run the array"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
