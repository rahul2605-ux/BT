"""
Final experiment -- the correctness suite (README §3.4 "final/verify.py"). New file; the
S5 checks are ported from cgan/verify.py §22. Needs Sionna, so it runs via sbatch:

    sbatch submit_verify.sh          # verify.py, then the regression gate regress.py
    python verify.py [-v]            # inside a job only

Exit code 0 iff every check passes. Each check states its prediction, then tests it.

Two predictions of §3.4 are tested in a corrected form, because the README's
back-of-envelope missed the jammer side of the comparison (recorded in README §3.3q):
  * "energy's noise share ~ 1.35/8 of full-band power's": the matched filter's noise-
    equivalent bandwidth is the symbol rate, 1/8 of the band, and the full-band frame also
    carries the pulse tails (1280 samples for 128 symbols), so relative to the signal the
    energy statistic holds (0.1 + N0)/(c0^2 + N0) ~ 1/10 of power's noise, not 1.35/8.
  * "at 30 dB, energy's white-noise transition within ~1 dB of power's": the clean spreads
    agree (the signal x noise cross term), but the MF passes only 1/10 of a WHITE jammer's
    frame energy, so energy's white-noise transition is 10 log10(1280/128) = 10 dB above
    power's. For an in-band jammer (async matched QPSK) the two agree to ~1.4 dB (the MF
    loses 1 - beta/4 of an asynchronous QPSK jammer's energy). Both are checked.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import

import argparse
import math
import os
import sys

import numpy as np
import torch

import attacks
import bands
import defender
import detectors
import env as E
import link as lk
import models

FAILURES = []
VERBOSE = False
S4_CNN = os.path.join(E.CGAN_ART, "baselines", "arms", "snr15_r0", "detector_spec.pt")


def check(label, got, want, tol, note=""):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<66} got={got:<11.5g} want={want:<11.5g} tol={tol:.3g}"
          + (f"   {note}" if note else ""), flush=True)
    if not ok:
        FAILURES.append(label)
    return ok


def mc_tol(p, n):
    return 4 * math.sqrt(max(p, 1e-9) * (1 - p) / n) + 1e-6


def batches(L, n, spec, jsr, snr_db, batch=1024, noiseless=False):
    outs = [attacks.frames(L, min(batch, n - i), spec, jsr, snr_db, noiseless=noiseless) for i in range(0, n, batch)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def ber_of(out):
    e, b, _, _ = attacks.error_counts(out)
    return e / b, b


def fresh_link(env=None):
    L = lk.Link(**lk.LINK)
    return E.apply(L, env) if env else L


# ---------------------------------------------------------------- 1. closed forms
def test_closed_forms():
    print("\n1. Closed forms: Es/N0 = SNR + 9.03 dB, the clean BER and the white-noise BER")
    L = fresh_link()
    check("Es/N0 offset = 10 log10(sps) [dB]", E.ESN0_OFFSET_DB, 10 * math.log10(8), 1e-12)
    for snr in (-6.0, 15.0):
        esn0 = 10 ** ((snr + E.ESN0_OFFSET_DB) / 10)
        want = float(lk.q_function(math.sqrt(esn0)))
        check(f"ber_clean({snr:g} dB) == Q(sqrt(Es/N0)), Es/N0 = SNR + 9.03 dB",
              float(L.ber_clean(snr)) / max(want, 1e-300), 1.0, 1e-3, note=f"Q = {want:.3g}")
    for snr in (-6.0, -3.0):
        out = batches(L, 4096, None, None, snr)
        b, nb = ber_of(out)
        want = float(L.ber_clean(snr))
        check(f"clean BER at SNR {snr:g} dB (Es/N0 {snr + E.ESN0_OFFSET_DB:.2f} dB) vs Q(sqrt(Es/N0))", b, want,
              mc_tol(want, nb))
    out = batches(L, 4096, None, None, E.SNR_DB)
    b, _ = ber_of(out)
    check("clean BER at our 15 dB (Es/N0 24 dB) is 0 in 1 Mbit (closed form ~1e-56)", b, 0.0, 0.0)
    for jsr in (0.0, 5.0):
        out = batches(L, 4096, dict(name="noise"), [jsr], E.SNR_DB)
        b, nb = ber_of(out)
        want = float(L.ber_noise_jammer(jsr, E.SNR_DB))
        check(f"white noise JSR {jsr:+g} dB at 15 dB: BER vs Q(c0/sqrt(N0 + JSR P_s))", b, want, mc_tol(want, nb))


# ---------------------------------------------------------------- 2. energy after the matched filter
def transition(L, det_fn, thr, spec, snr_db, grid, n=512, omni=False):
    p = []
    with torch.no_grad():
        for j in grid:
            out = attacks.frames(L, n, spec, j if omni else [j], snr_db)
            p.append(detectors.p_detect(det_fn(out), thr))
    return bands.crossing(grid, p)


def test_energy():
    print("\n2. energy = mean |z_k|^2: clean mean, noise share, transitions vs full-band power at 30 dB")
    L = fresh_link()
    n0 = L.noise_var(E.SNR_DB)
    out = batches(L, 20000, None, None, E.SNR_DB, noiseless=True)
    e = detectors.energy(out["z"])
    e_nf = detectors.energy(out["z_nf"])
    p = detectors.power(out["r"])
    n = e.numel()
    check("15 dB: clean energy mean == c0^2 + N0", float(e.mean()), L.c0 ** 2 + n0,
          4 * float(e.std()) / math.sqrt(n) + 2e-4)
    check("15 dB: the noise adds N0 to the energy (noisy - noiseless mean)", float((e - e_nf).mean()), n0,
          4 * float((e - e_nf).std()) / math.sqrt(n) + 1e-5)
    frame_len = L.waveform_length(E.N_SYM)
    share_e = n0 / float(e.mean())
    share_p = n0 / float(p.mean())
    want = (E.N_SYM / frame_len + n0) / (L.c0 ** 2 + n0)
    check("energy's noise share / full-band power's == (N/len + N0)/(c0^2 + N0)", share_e / share_p, want, 0.01,
          note=f"README §3.4 said ~1.35/8 = {1.35 / 8:.3f}: corrected, see the docstring")
    # transitions at 30 dB: calibrate both statistics on clean 30 dB frames
    L30, snr30 = fresh_link(), 30.0
    c = batches(L30, 20000, None, None, snr30)
    thr_e = detectors.calibrate(detectors.energy(c["z"]), E.ALPHA)
    thr_p = detectors.calibrate(detectors.power(c["r"]), E.ALPHA)
    grid = [float(v) for v in np.arange(-40.0, 0.5, 1.0)]
    fe = lambda o: detectors.energy(o["z"])
    fp = lambda o: detectors.power(o["r"])
    te = transition(L30, fe, thr_e, dict(name="noise"), snr30, grid)
    tp = transition(L30, fp, thr_p, dict(name="noise"), snr30, grid)
    check("30 dB white noise: energy's P_det-0.5 transition - power's [dB]", te - tp,
          10 * math.log10(frame_len / E.N_SYM), 1.0,
          note=f"energy {te:+.2f}, power {tp:+.2f}; README §3.4 expected ~0: corrected")
    te = transition(L30, fe, thr_e, dict(name="pulsed", p=1.0), snr30, grid)
    tp = transition(L30, fp, thr_p, dict(name="pulsed", p=1.0), snr30, grid)
    # power sees the jammer's P_s per sample over the whole frame; the MF sees (1 - beta/4)
    # of an async QPSK symbol's energy: detection-SNR ratio len P_s / ((1 - beta/4) N)
    check("30 dB async matched QPSK: energy's transition - power's [dB]", te - tp,
          10 * math.log10(frame_len / (L.sps * E.N_SYM * (1.0 - 0.35 / 4))), 1.0,
          note=f"energy {te:+.2f}, power {tp:+.2f}: an in-band jammer is seen alike")


# ---------------------------------------------------------------- 3. Rician fading
def test_rician():
    print("\n3. Rician block fading: unit mean power, K -> inf, empirical K, clean BER, plumbing")
    L = fresh_link()
    for k_db in (12.0, 28.0):
        L.rician_k_db = k_db
        g2 = L.fading_gain(400_000).double().pow(2).flatten()
        k = 10 ** (k_db / 10)
        check(f"K {k_db:g} dB: mean power gain E[g^2]", float(g2.mean()), 1.0, 4 * float(g2.std()) / math.sqrt(g2.numel()))
        gamma = float(g2.var() / g2.mean() ** 2)
        k_hat = math.sqrt(1 - gamma) / (1 - math.sqrt(1 - gamma))
        check(f"K {k_db:g} dB: empirical K (moment estimator) [dB]", 10 * math.log10(k_hat), k_db, 0.2)
        check(f"K {k_db:g} dB: std of the frame power gain == sqrt(1 + 2K)/(K + 1)", float(g2.std()),
              math.sqrt(1 + 2 * k) / (k + 1), 0.01 * math.sqrt(1 + 2 * k) / (k + 1),
              note=f"= {float((10 * g2.log10()).std()):.2f} dB std")
    L.rician_k_db = 300.0
    g = L.fading_gain(10000)
    check("K -> inf (300 dB): g == 1", float((g - 1).abs().max()), 0.0, 1e-5)
    L.rician_k_db = None
    check("no fading: fading_gain is None and frames carry no gain",
          float(L.fading_gain(4) is None and "gain" not in attacks.frames(L, 4, None, None, E.SNR_DB)), 1.0, 0)
    L.rician_k_db = 12.0
    out = batches(L, 4096, None, None, E.SNR_DB)
    b, _ = ber_of(out)
    want = float(np.mean(lk.q_function((out["gain"].flatten() * L.c0 / math.sqrt(L.noise_var(E.SNR_DB))).cpu().numpy())))
    check("K 12 dB at 15 dB: clean BER ~ 0 (the receiver knows its phase)", b, want, 1e-5, note=f"closed form {want:.1e}")
    snr = -3.0
    out = batches(L, 8192, None, None, snr)
    fb = (out["bits"] != out["bits_hat"]).double().mean(-1)
    want = float(np.mean(lk.q_function((out["gain"].flatten() * L.c0 / math.sqrt(L.noise_var(snr))).cpu().numpy())))
    check("K 12 dB at -3 dB: BER == mean_f Q(g_f c0 / sqrt(N0)) (gain on the measurement path)",
          float(fb.mean()), want, 4 * float(fb.std()) / math.sqrt(fb.numel()) + 1e-6)
    g2 = out["gain"].flatten().double() ** 2
    zs = out["z"] - L.mapper(out["bits"]).reshape(out["z"].shape) * out["gain"]
    ratio = zs.abs().pow(2).mean(-1).double() / L.noise_var(snr)
    check("K 12 dB: decision-point noise stays N0 (only the signal is faded)", float(ratio.mean()), 1.0,
          4 * float(ratio.std()) / math.sqrt(ratio.numel()) + 0.01)
    e = detectors.energy(batches(L, 8192, None, None, E.SNR_DB)["z"])
    csi_out = batches(L, 8192, None, None, E.SNR_DB)
    ecsi = detectors.energy_csi(csi_out["z"], csi_out["gain"], L.c0)
    L0 = fresh_link()
    e0 = detectors.energy(batches(L0, 8192, None, None, E.SNR_DB)["z"])
    check("K 12 dB: energy_csi removes the fading (its clean std / base energy's)", float(ecsi.std() / e0.std()), 1.0, 0.1,
          note=f"naive energy std is {float(e.std() / e0.std()):.1f}x base")
    del g2


# ---------------------------------------------------------------- 4. noise-level uncertainty
def test_noise_uncertainty():
    print("\n4. sigma_N: S5's factor law, sigma_N = 0 draws nothing, the damage term per frame")
    L = fresh_link()
    L.noise_unc_db = 1.0
    u = L.noise_scale(200_000).double()
    check("sigma_N 1 dB: mean variance factor", float(u.mean()), 1.0, 0.005)
    check("sigma_N 1 dB: std of the factor [dB]", float((10 * u.log10()).std()), 1.0, 0.01)
    L.noise_unc_db = 0.0
    check("sigma_N 0: noise_scale is None and frames carry no factor",
          float(L.noise_scale(4) is None and "noise_scale" not in attacks.frames(L, 4, None, None, E.SNR_DB)), 1.0, 0)
    L.noise_unc_db, snr = 3.0, 0.0
    out = batches(L, 8192, dict(name="noise"), [-10.0], snr, noiseless=True)
    fb = (out["bits"] != out["bits_hat"]).double().mean(-1)
    n0 = L.noise_var(snr)
    ebar = attacks.expected_ber(out, attacks.frame_noise_var(out, n0))
    tol = 4 * float(fb.std()) / math.sqrt(fb.numel()) + 1e-6
    check("sigma_N 3 dB: exact E[BER] with the per-frame noise == the measured BER", ebar, float(fb.mean()), tol)
    ebar_scalar = attacks.expected_ber(out, n0)
    check("... and with the nominal N0 it would be biased (control: |bias| > tol)",
          float(abs(ebar_scalar - float(fb.mean())) > tol), 1.0, 0, note=f"nominal-N0 E[BER] {ebar_scalar:.4g}")


# ---------------------------------------------------------------- 5. CFAR per environment
def proxy_defender(L, env_name, n_cal):
    """This environment's calibration with S4's 15 dB CNN weights (the base recipe): the
    calibration code path of train_cnn.py, before any environment's CNN exists."""
    net, _, _ = detectors.load_cnn(S4_CNN, L.device)
    with torch.no_grad():
        scale = detectors.SpecScale.fit(batches(L, 2048, None, None, E.SNR_DB)["r"])
    dets = defender.env_dets(E.faded(env_name))
    thr = defender.calibrate_env(L, net, scale, E.SNR_DB, dets, n_cal=n_cal, verbose=VERBOSE)
    return defender.Defender(L, E.SNR_DB, thr, net, scale, dets)


def test_cfar(n_cal, n_test=8192):
    print(f"\n5. CFAR: every detector's realised FAR on fresh clean frames of each environment "
          f"({n_cal} calibration + {n_test} test frames; S4's CNN weights as the proxy)")
    a = E.ALPHA
    tol = 4 * math.sqrt(a * (1 - a) * (1 / n_test + 1 / n_cal))
    dfds = {}
    for env_name in E.ENVS:
        L = fresh_link(env_name)
        dfd = proxy_defender(L, env_name, n_cal)
        dfds[env_name] = dfd
        with torch.no_grad():
            out = batches(L, n_test, None, None, E.SNR_DB)
            s = dfd.statistics(out["r"], out["z"], gain=out.get("gain"))
        for d in dfd.dets:
            check(f"{env_name}: {d} realised FAR at alpha {a}", detectors.p_detect(s[d], dfd.threshold(d, a)), a, tol)
        marg = 100 * (dfd.threshold("energy", a) / dfd.thr["energy_clean_mean"] - 1)
        print(f"         {env_name}: energy threshold margin over the clean mean {marg:+.2f} %", flush=True)
    return dfds


# ---------------------------------------------------------------- 6. bands at base
def test_bands(dfd_base):
    print("\n6. bands.py at base ~ S4's bands (CNN (-39, -7), energy vs S4's power (-24, 0)); S4's CNN as base proxy")
    ref = bands.reference30(dfd_base.L.device)
    print("         30 dB transitions: " + "  ".join(f"{d} {v:+.2f}" for d, v in ref["transition"].items()), flush=True)
    L = dfd_base.L
    curves = bands.noise_curve(L, dfd_base, ["energy", "kurtosis", "spec_cnn"], E.SNR_DB)
    trans = {d: bands.crossing(bands.GRID, c) for d, c in curves.items()}
    b, notes = bands.bands_from(ref, trans)
    print("         base bands: " + "  ".join(f"{k} {v['band']}" for k, v in b.items()) + (f"  notes {notes}" if notes else ""),
          flush=True)
    check("base: CNN band lower edge vs S4's -39 dB", b["spec_cnn"]["band"][0], -39.0, 2.0)
    check("base: CNN band upper edge vs S4's -7 dB", b["spec_cnn"]["band"][1], -7.0, 2.0)
    check("base: energy band lower edge vs S4's power band -24 dB", b["energy"]["band"][0], -24.0, 2.0)
    check("base: energy band upper edge (capped) vs S4's 0 dB", b["energy"]["band"][1], 0.0, 0.0)


# ---------------------------------------------------------------- 7. baselines and references
def test_baselines(dfds):
    print("\n7. tone's DC share, on/off analytic vs simulated, the genie flip and push, the random push")
    dfd = dfds["base"]
    L = dfd.L
    a = E.ALPHA
    with torch.no_grad():
        out = attacks.frames(L, 1024, dict(name="tone"), [-10.0], E.SNR_DB)
        check("tone: DC share == 1", float(out["dc_share"].mean()), 1.0, 1e-4)
        out = attacks.frames(L, 1024, dict(name="noise"), [-10.0], E.SNR_DB)
        check("white noise: DC share ~ 1/len(active window)", float(out["dc_share"].mean()), 1.0 / (E.N_SYM * L.sps),
              3.0 / (E.N_SYM * L.sps))
        # on/off at f = 0.3, +15 dB: simulated vs the analytic formula of figures.py
        f, n = 0.3, 4096
        on = defender.measure(L, dfd, dict(name="pulsed", p=1.0), [15.0], n)
        clean = defender.measure(L, dfd, None, None, n)
        sim = defender.measure(L, dfd, dict(name="onoff", f=f), [15.0], n)
        for d in dfd.dets:
            want = f * on["pdet"][d][str(a)] + (1 - f) * clean["pdet"][d][str(a)]
            check(f"on/off f {f}: P_det {d} == f P_det(+15) + (1-f) FAR", sim["pdet"][d][str(a)], want,
                  4 * math.sqrt(want * (1 - want) / n) + 0.01)
        check(f"on/off f {f}: PER == f PER(+15)", sim["per"], f * on["per"], 4 * math.sqrt(f * (1 - f) / n))
        check(f"on/off f {f}: BER == f BER(+15)", sim["ber"], f * on["ber"], 4 * math.sqrt(f * (1 - f) / n) * on["ber"] + 1e-4)
        for env_name in ("base", "both"):
            dd = dfds[env_name]
            for jsr in (-6.0, 0.0):
                m = defender.measure(dd.L, dd, dict(name="omniscient", eta=1.0), jsr, 4096)
                delta = min(1.0, 10 ** (jsr / 10) / 4.0)
                want = math.floor(delta * E.N_SYM) / E.N_SYM
                check(f"{env_name}: genie flip JSR {jsr:+g} dB: BER == floor(delta N)/N", m["ber"], want, 2e-3)
                for d in dd.dets:
                    check(f"{env_name}: genie flip JSR {jsr:+g} dB: P_det {d} ~ alpha", m["pdet"][d][str(a)], a,
                          4 * math.sqrt(a * (1 - a) / 4096) + 0.01)
        m = defender.measure(L, dfd, dict(name="omniscient", eta=0.1), -10.0, 2048)
        check("genie push -10 dB: two-sided energy flags it (P_det >= 0.9)", float(m["pdet"]["energy_2s"][str(a)] >= 0.9),
              1.0, 0, note=f"P_det {m['pdet']['energy_2s'][str(a)]:.3f}")
        check("genie push -10 dB: one-sided energy does not (P_det <= alpha + 4 sigma)",
              float(m["pdet"]["energy"][str(a)] <= a + 4 * math.sqrt(a * (1 - a) / 2048)), 1.0, 0,
              note=f"P_det {m['pdet']['energy'][str(a)]:.3f}")
        # random push -10 dB: each attacked axis is pushed against its bit w.p. 1/2 and then sits
        # eta/sqrt(2) past the boundary, so per attacked symbol E[bit errors]/2 = (1 - Q(eta/(sqrt2 sigma)))/2
        n = 2048
        m = defender.measure(L, dfd, dict(name="random_push", eta=0.1), -10.0, n)
        frac = math.floor(10 ** (-10 / 10) / 1.1 ** 2 * E.N_SYM) / E.N_SYM
        sig = math.sqrt(L.noise_var(E.SNR_DB) / 2)
        q = 0.5 * math.erfc(0.1 * L.c0 / math.sqrt(2) / sig / math.sqrt(2))
        want = frac * 0.5 * (1 - q)
        check("random push -10 dB: BER == floor(delta N)/N (1 - Q(eta c0/(sqrt2 sigma)))/2", m["ber"], want,
              4 * math.sqrt(want / (n * 2 * E.N_SYM)) + 1e-3)
        want = frac * (0.25 * (1 - q ** 2) + 0.5 * (1 - q))       # u = -s: both axes; u orthogonal: one
        check("random push -10 dB: SER == floor(delta N)/N (1/4 (1-q^2) + 1/2 (1-q))", m["ser"], want,
              4 * math.sqrt(want / (n * E.N_SYM)) + 1e-3)
        check("random push -10 dB: one-sided energy flags it (P_det >= 0.9)", float(m["pdet"]["energy"][str(a)] >= 0.9),
              1.0, 0, note=f"P_det {m['pdet']['energy'][str(a)]:.3f}")


# ---------------------------------------------------------------- 8. the training path
def test_training_step(dfds):
    print("\n8. One training step per target in env `both` (noise factor + fading): finite loss, gradient reaches G")
    import train_gan
    dfd = dfds["both"]
    L = dfd.L
    n0 = L.noise_var(E.SNR_DB)
    for target in (None, "energy", "kurtosis", "spec_cnn"):
        G = models.Generator(n_classes=1).to(L.device)
        G.train()
        spec = dict(name="gan", G=G, scale=1.0, grad=True)
        out = attacks.frames(L, train_gan.FRAMES[target], spec, [-20.0], E.SNR_DB, noiseless=True)
        loss = -attacks.log_expected_ber(out, attacks.frame_noise_var(out, n0))
        if target is not None:
            s = dfd.statistics(out["r"], out["z"], dets=[target], grad=True, gain=out.get("gain"))
            loss = loss + 10.0 * detectors.soft_pdet(s[target], dfd.threshold(target, E.ALPHA), 1.0)
        loss.backward()
        gn = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in G.parameters() if p.grad is not None))
        check(f"target {target}: loss finite and ||grad G|| > 0", float(math.isfinite(float(loss)) and gn > 0), 1.0, 0,
              note=f"loss {float(loss):.3f}, ||grad|| {gn:.3g}")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--n-cal", type=int, default=defender.N_CALIBRATION)
    args = ap.parse_args()
    VERBOSE = args.v
    device = lk.setup(seed=1234)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    test_closed_forms()
    test_energy()
    test_rician()
    test_noise_uncertainty()
    dfds = test_cfar(args.n_cal)
    test_bands(dfds["base"])
    test_baselines(dfds)
    test_training_step(dfds)
    print("\n" + ("ALL CHECKS PASS" if not FAILURES else f"{len(FAILURES)} FAILED:\n  " + "\n  ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
