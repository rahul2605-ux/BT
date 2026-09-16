"""
sim08 ablations -- correctness suite. Exit 0 before any sweep is trusted.

The ablation re-implements one thing the frozen sim08 code cannot express
(evaluate() with N_J > 1 jammers, a noiseless anchor, the received JSR), so this
checks that path against exact facts and against the frozen dense sweep it must
reproduce at N_J = 1 (job 102390, artifacts/sim08/frontier_dense/results.json).
Sionna cannot be imported on the login node, so it runs as a job:

    sbatch submit_verify.sh          # -> runs/verify_<JOBID>.out

Tolerances are 4-sigma Monte-Carlo bands: frame-level standard errors for BER
(errors are correlated within a faded frame), binomial over frames for P(detect).
"""

import json
import math
import os
import sys

import torch

import ablation as ab
from frontier_channel import calibrate_energy

FAILURES = []
DENSE = os.path.join(ab.HERE, "..", "artifacts", "sim08", "frontier_dense", "results.json")


def check(label, got, want, tol, note=""):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<60} got={got:<11.5g} want={want:<11.5g} "
          f"tol={tol:.3g}" + (f"   {note}" if note else ""), flush=True)
    if not ok:
        FAILURES.append(label)
    return ok


def binom_tol(p1, n1, p2, n2, k=4.0):
    p = max(0.5 * (p1 + p2), 1e-3)
    return k * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def ref_row(rows, strategy, ebno, n_active=0, power=0.0):
    return next(r for r in rows if r["strategy"] == strategy and r["ebno_db"] == ebno
                and r["n_active"] == n_active and abs(r["power"] - power) < 1e-9)


def main():
    device, detector, eff_full, chan = ab.setup(seed=4242)
    B = 512
    dense = json.load(open(DENSE))["rows"]
    thresholds = {}

    def ctx(ebno):
        if ebno not in thresholds:
            no, _ = ab.noise_variances(ebno)
            thresholds[ebno] = calibrate_energy(chan, no, B, device, far=ab.ENERGY_FAR, n_batches=8)
        e = thresholds[ebno]
        return lambda nj, na, p, nb: ab.evaluate(chan, detector, eff_full, e, ebno, nj, na, p,
                                                 B, nb, device)

    print("\n[1] noiseless anchor: ZF with perfect CSI and no noise makes zero errors")
    clean0 = ctx(None)(0, 0, 0.0, 2)
    check("noiseless clean BER", clean0["ber_mean"], 0.0, 0.0)

    print("\n[2] equal total transmit power across N_J (exact, by construction)")
    _, _, tx_grid, _ = ab.ofdm.generate_ofdm_frame(B)
    h_tx, h_jam = chan.sample(B)
    for nj in ab.N_JAMMERS:
        jams = ab.build_jams(nj, 8, 2.0, tx_grid, h_tx, h_jam, eff_full, device)
        tot = float(sum(j.abs().pow(2).sum(-1).mean() for j in jams))
        check(f"N_J={nj}: sum_k sum_SC |jam_k|^2 per OFDM symbol", tot, 16.0, 1e-6)

    print("\n[3] received JSR = n_active * power / 52 in expectation (unit-power QPSK, "
          "0 dB path gains, independent fading)")
    ev30 = ctx(30.0)
    for nj, na, want in [(1, 52, 0.0), (4, 52, 0.0), (1, 13, 10 * math.log10(13 / 52))]:
        r = ev30(nj, na, 1.0, 2)
        check(f"N_J={nj} n={na} p=1: JSR_rx (dB)", r["jsr_rx_db"], want, 0.5)

    print("\n[4] N_J = 1 reproduces the frozen dense sweep (job 102390)")
    for ebno in (10.0, 30.0):
        ev = ctx(ebno)
        for na, p in [(16, 1.0), (8, 4.0)]:
            ref = ref_row(dense, "sparse_blind", ebno, na, p)
            r = ev(1, na, p, 4)
            tag = f"{ebno:g} dB n={na} p={p:g}"
            sem = math.sqrt(r["ber_sem"] ** 2 + (ref["ber_std"] / math.sqrt(B)) ** 2)
            check(f"{tag}: BER", r["ber_mean"], ref["ber_mean"], 4 * sem)
            for mine, theirs in [("p_cnn", "p_cnn"), ("p_energy", "p_energy_mean"),
                                 ("p_suite", "p_suite")]:
                check(f"{tag}: {mine}", r[mine], ref[theirs],
                      binom_tol(r[mine], r["frames"], ref[theirs], B))

    print("\n[5] clean BER floor and CNN clean FAR match the dense sweep")
    for ebno in (5.0, 10.0, 15.0, 20.0, 30.0):
        ref = ref_row(dense, "clean", ebno)
        r = ctx(ebno)(0, 0, 0.0, 4)
        sem = math.sqrt(r["ber_sem"] ** 2 + (ref["ber_std"] / math.sqrt(B)) ** 2)
        check(f"{ebno:g} dB: clean BER floor", r["ber_mean"], ref["ber_mean"], 4 * sem)
        check(f"{ebno:g} dB: clean p_cnn", r["p_cnn"], ref["p_cnn"],
              binom_tol(r["p_cnn"], r["frames"], ref["p_cnn"], B))

    print("\n[6] the energy threshold honours its 1% FAR on fresh clean frames")
    for ebno in (10.0, None):
        r = ctx(ebno)(0, 0, 0.0, 8)
        n = r["frames"]
        check(f"{ab.level_tag(ebno)}: fresh clean p_energy", r["p_energy"], ab.ENERGY_FAR,
              4 * math.sqrt(2 * ab.ENERGY_FAR * (1 - ab.ENERGY_FAR) / n))

    print("\n[7] a vanishing jammer is indistinguishable from clean")
    ev20 = ctx(20.0)
    clean20, faint = ev20(0, 0, 0.0, 4), ev20(1, 52, 1e-5, 4)
    check("20 dB: p_suite(p=1e-5) vs clean", faint["p_suite"], clean20["p_suite"],
          binom_tol(faint["p_suite"], faint["frames"], clean20["p_suite"], clean20["frames"]))
    check("20 dB: BER(p=1e-5) vs clean", faint["ber_mean"], clean20["ber_mean"],
          4 * math.sqrt(faint["ber_sem"] ** 2 + clean20["ber_sem"] ** 2))

    print("\n[8] frontier selection picks the max-BER config under the budget")
    rows = [dict(n_jammers=1, ber_mean=b, p_cnn=p, p_suite=p) for b, p in
            [(0.1, 0.02), (0.3, 0.05), (0.2, 0.01), (0.5, 0.6)]]
    check("pick at budget 0.03", ab.frontier_pick(rows, "cnn", 0.03)["ber_mean"], 0.2, 0.0)
    check("pick at budget 0.05 (inclusive)", ab.frontier_pick(rows, "cnn", 0.05)["ber_mean"],
          0.3, 0.0)
    check("pick at budget 0.001 is None", float(ab.frontier_pick(rows, "cnn", 0.001) is None),
          1.0, 0.0)

    print("\n" + ("ALL CHECKS PASS" if not FAILURES else
                  f"{len(FAILURES)} FAILED:\n  " + "\n  ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
