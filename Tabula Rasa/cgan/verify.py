"""
CGAN track -- correctness suite.

Every claim the link and the jammers rest on, checked against an analytic
prediction, before any calibration (C2), training (C5) or evaluation (C6) job is
trusted. It needs Sionna, so it runs through sbatch, never on the login node:

    sbatch submit_verify.sh          # -> runs/verify_<JOBID>.out
    python verify.py [-v]            # inside a job / on a compute node only

Exit code 0 iff every check passes. Written to be readable as documentation:
each check states the prediction, then tests it.
"""

import argparse
import math
import sys

import numpy as np
import torch

import link as lk
import jammers
import models
import losses

FAILURES = []
VERBOSE = False
NO_NOISE_DB = 300.0     # SNR that makes the AWGN term numerically zero


def check(label, got, want, tol, note=""):
    ok = abs(got - want) <= tol
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<58} got={got:<11.5g} want={want:<11.5g} tol={tol:g}"
          + (f"   {note}" if note else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def mc_tol(p, n_bits, n_eff=None):
    """4-sigma binomial band. `n_eff` < n_bits when errors are correlated within a frame."""
    n = n_bits if n_eff is None else n_eff
    return 4 * math.sqrt(max(p, 1e-9) * (1 - p) / n) + 1e-6


def ber(L, n_frames, n_sym, snr_db, jammer=None, batches=1):
    err = bits = 0
    for _ in range(batches):
        e, b = L.run(n_frames, n_sym, snr_db, jammer)
        err, bits = err + e, bits + b
    return err / bits, bits


class PerfectG:
    """
    A stand-in for a perfect generator: ignores z and returns clean QPSK segments
    from the link itself, peak-normalised exactly as training data is. Anything
    the GAN evaluation path does to a generator's output must leave this one
    indistinguishable from the `optimal` jammer.
    """

    def __init__(self, L):
        self.L = L
        self.scaler = models.PeakScaler.fit(L.real_segments(256, models.SEG_LEN))

    def __call__(self, z, labels):
        return self.scaler.normalize(self.L.real_segments(z.shape[0], models.SEG_LEN))


# ---------------------------------------------------------------- 1. waveform
def test_waveform(L):
    """
    Unit-energy Gray QPSK with m0's labelling; unit-energy pulse, so the matched
    filter cascade peaks at c0 = 1 and the mean power per sample is P_s = 1/sps.
    With no noise and no jammer the loopback must be error-free (the cascade is
    Nyquist; RRC truncation ISI is ~1e-3, far below the 1/sqrt(2) margin).
    """
    print(f"\n1. Waveform [{L.pulse}, sps={L.sps}]")
    pts = L.mapper(torch.tensor([[0., 0., 0., 1., 1., 0., 1., 1.]], device=L.device))[0]
    want = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=pts.dtype,
                        device=L.device) / math.sqrt(2)
    check("Gray labels: bit0 -> sign I, bit1 -> sign Q (0 -> +)",
          (pts - want).abs().max().item(), 0.0, 1e-6)
    _, sym, x = L.modulate(64, 4096)
    check("E|s|^2", sym.abs().pow(2).mean().item(), 1.0, 5e-3)
    k = L.filt.length
    steady = x[:, k:-k]
    check("mean power per sample == 1/sps", steady.abs().pow(2).mean().item(), L.p_s,
          0.01 * L.p_s)
    check("cascade peak c0", L.c0, 1.0, 1e-3 if L.pulse == "rect" else 1e-2)
    # The geometric checks in section 5 need margins of ~4% of the amplitude;
    # residual ISI (signal + jammer) must stay an order of magnitude below that.
    check("worst-case residual ISI sum_k|c_k|/c0", L.isi_sum(), 0.0, 5e-3)
    b, _ = ber(L, 16, 4096, NO_NOISE_DB)
    check("noiseless loopback BER", b, 0.0, 0.0)


# ---------------------------------------------------------------- 2. clean link
def test_clean_vs_theory(L):
    """
    Unjammed BER = Q(c0 * sqrt(sps * SNR)): per-sample SNR, full processing gain.
    Tested only where the BER is large enough to measure.
    """
    print(f"\n2. Clean link vs Q(sqrt(sps*SNR)) [{L.pulse}, sps={L.sps}]")
    for snr_db in [-12.0, -9.0, -6.0]:
        th = float(L.ber_clean(snr_db))
        b, n = ber(L, 64, 8192, snr_db, batches=2)
        check(f"BER(SNR={snr_db:+.0f} dB)", b, th, mc_tol(th, n))


# ---------------------------------------------------------------- 3. noise jammer
def test_noise_jammer_vs_closed_form(L):
    """
    A Gaussian jammer at per-sample JSR reaches the decision with variance
    JSR*P_s*kappa: kappa = 1 for white ("full") noise, kappa = sum|h*h|^2 for
    noise shaped like the signal ("inband"). This validates the processing-gain
    formula the C2 calibration fits to Zhou's Fig. 6.
    """
    print(f"\n3. Gaussian jammer vs closed form [{L.pulse}, sps={L.sps}]")
    grid = {"full": [0.0, 5.0, 10.0], "inband": [-8.0, -5.0, 0.0]}
    for band, jsrs in grid.items():
        for jsr in jsrs:
            th = float(L.ber_noise_jammer(jsr, band=band))
            b, n = ber(L, 64, 8192, lk.SNR_DB, jammers.make("noise", jsr, band=band), 2)
            check(f"{band:<6} noise BER(JSR={jsr:+.0f} dB)  kappa={L.kappa(band):.3f}",
                  b, th, mc_tol(th, n))


# ---------------------------------------------------------------- 4. JSR scaling
def test_jsr_is_exact(L):
    """
    Every jammer frame is scaled to exactly JSR * P_s (a hard constraint), over the
    window the symbols occupy. A short frame (64 symbols against a 32-symbol
    filter) is used on purpose: it is where a full-frame average would be wrong.
    """
    print(f"\n4. Realised JSR [{L.pulse}, sps={L.sps}]")
    n_sym = 64
    G = PerfectG(L)
    cases = [("noise", dict(band="full")), ("noise", dict(band="inband"))] + \
            [("optimal", dict(sync=s)) for s in jammers.SYNC_VARIANTS] + \
            [("gan", dict(G=G, scale=G.scaler.scale, sync=s)) for s in jammers.SYNC_VARIANTS]
    _, _, x = L.modulate(256, n_sym)
    ps = x[..., L.active(n_sym)].abs().pow(2).mean().item()
    check("signal power over the active window == 1/sps", ps, L.p_s, 0.02 * L.p_s)
    for name, kw in cases:
        tag = {k: v for k, v in kw.items() if k in ("band", "sync")}
        for jsr in [-10.0, 3.0]:
            j = jammers.make(name, jsr, **kw)(L, 32, n_sym)
            p = j[..., L.active(n_sym)].abs().pow(2).mean(dim=-1) / L.p_s
            got = (10 * torch.log10(p) - jsr).abs().max().item()
            check(f"{name} {tag} JSR={jsr:+.0f} dB: max |realised - target| dB", got, 0.0, 0.05)
        want_len = L.waveform_length(n_sym)
        check(f"{name} {tag} length", j.shape[-1], want_len, 0)


# ---------------------------------------------------------------- 5. geometry
def test_optimal_geometry(L):
    """
    With no noise, a symbol-synchronous QPSK jammer lands exactly one
    constellation point sqrt(JSR) away on the victim's decision variable:
      locked:        an axis flips iff sqrt(JSR) > 1 and the jammer's bit on that
                     axis differs (prob 1/2)  ->  BER = 0 below 0 dB, 0.5 above.
      random_phase:  the jammer point is uniformly rotated, so its I component is
                     sqrt(JSR)*cos(U), U uniform, and a flip needs it to exceed
                     1/sqrt(2) with the opposite sign
                     ->  BER = arccos(1/sqrt(2*JSR)) / pi   for JSR >= 1/2, else 0.
    """
    print(f"\n5. Optimal jammer geometry at zero noise [{L.pulse}, sps={L.sps}]")
    for jsr, want in [(-0.5, 0.0), (0.5, 0.5)]:
        b, n = ber(L, 32, 4096, NO_NOISE_DB, jammers.make("optimal", jsr, sync="locked"))
        tol = 0.0 if want == 0.0 else mc_tol(want, n)
        check(f"locked BER(JSR={jsr:+.1f} dB)", b, want, tol)
    for jsr in [-3.5, 0.0, 3.0, 6.0]:
        j_lin = 10 ** (jsr / 10)
        want = math.acos(1 / math.sqrt(2 * j_lin)) / math.pi if j_lin >= 0.5 else 0.0
        n_frames = 4000   # one phase per frame: the frame count is the sample size
        b, n = ber(L, n_frames, 64, NO_NOISE_DB, jammers.make("optimal", jsr, sync="random_phase"))
        tol = 0.0 if want == 0.0 else 4 * math.sqrt(want * (1 - want) / n_frames) + 1e-3
        check(f"random_phase BER(JSR={jsr:+.1f} dB)", b, want, tol)


# ---------------------------------------------------------------- 5b. GAN path == optimal
def test_perfect_generator(L):
    """
    The GAN jammer path (tile -> align -> desync -> JSR projection) fed by a
    perfect generator must produce the same BER as `optimal` under the same sync.
    locked vs async differ by ~0.06 at 0 dB and ~0.2 at +3 dB, so a path that
    silently changes the synchronisation (the pre-2026-09-15 bug) fails here.

    Tolerance: 4 sigma on a difference of two frame means, with the per-frame
    variance bounded by p(1-p) (conservative: async draws one offset and phase per
    frame, so errors are correlated within a frame), plus 0.01 for the tile
    boundaries, where a perfect generator's independent segments lose their
    neighbours' pulse tails.
    """
    print(f"\n5b. GAN path with a perfect generator == optimal jammer [{L.pulse}, sps={L.sps}]")
    G = PerfectG(L)
    n_frames, n_sym = 8000, 128
    for sync in ("locked", "async"):
        for jsr in (0.0, 3.0):
            b_opt, _ = ber(L, n_frames, n_sym, lk.SNR_DB, jammers.make("optimal", jsr, sync=sync))
            b_gan, _ = ber(L, n_frames, n_sym, lk.SNR_DB,
                           jammers.make("gan", jsr, G=G, scale=G.scaler.scale, sync=sync))
            p = max(b_opt, b_gan)
            tol = 4 * math.sqrt(2 * p * (1 - p) / n_frames) + 0.01
            check(f"{sync:<6} JSR={jsr:+.0f} dB: BER gan(perfect G) - BER optimal", b_gan - b_opt,
                  0.0, tol, note=f"optimal {b_opt:.4f}")


# ---------------------------------------------------------------- 6. GAN models
def test_models():
    """
    Shapes Zhou's Figs. 2-3 fix, and the loss zero-points the training loop relies
    on. Pure torch, no link -- runs anywhere. n_classes=3 exercises the conditioning
    path step 2 reuses; n_classes=1 must make the auxiliary classifier a no-op.
    """
    print("\n6. GAN models and losses")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    B, seg = 8, models.SEG_LEN
    for n_cls in (1, 3):
        G = models.Generator(n_classes=n_cls).to(device)
        D = models.Discriminator(n_classes=n_cls).to(device)
        z = torch.randn(B, models.Z_DIM, device=device)
        labels = torch.randint(0, n_cls, (B,), device=device)
        fake = G(z, labels)
        check(f"[n_cls={n_cls}] G output shape (B,2,seg)", float(fake.shape == (B, 2, seg)), 1.0, 0)
        check(f"[n_cls={n_cls}] G output in [-1,1]", float(fake.abs().max() <= 1.0), 1.0, 0)
        score, clogits, feats = D(fake)
        check(f"[n_cls={n_cls}] D feature length == D_CH*FEAT_LEN",
              feats.shape[1], models.D_CH * models.FEAT_LEN, 0)
        check(f"[n_cls={n_cls}] D score shape (B,1)", float(score.shape == (B, 1)), 1.0, 0)
        check(f"[n_cls={n_cls}] D class logits shape (B,n_cls)",
              float(clogits.shape == (B, n_cls)), 1.0, 0)
        closs = float(losses.classification_loss(clogits, labels).detach())
        if n_cls == 1:
            check("[n_cls=1] classification loss is exactly 0", closs, 0.0, 0.0)
        else:
            check("[n_cls=3] classification loss is finite and > 0",
                  float(math.isfinite(closs) and closs > 0), 1.0, 0)
    # Loss zero-points and finiteness, on an identical real/fake pair.
    real = torch.randn(B, 2, seg, device=device).clamp(-1, 1)
    check("STFT loss on identical batches", float(losses.stft_loss(real, real.clone())), 0.0, 1e-6)
    check("I/Q loss on identical batches", float(losses.iq_distribution_loss(real, real.clone())),
          0.0, 1e-6)
    f = torch.randn(B, models.D_CH * models.FEAT_LEN, device=device)
    check("feature-matching loss on identical features",
          float(losses.feature_matching_loss(f, f.clone())), 0.0, 1e-6)
    check("STFT loss > 0 on different batches",
          float(losses.stft_loss(real, torch.randn_like(real).clamp(-1, 1)) > 0), 1.0, 0)
    gp = losses.gradient_penalty(D, real, torch.randn_like(real).clamp(-1, 1))
    check("gradient penalty is finite and >= 0", float(torch.isfinite(gp) & (gp >= 0)), 1.0, 0)


# ---------------------------------------------------------------- 7. normalisation
def test_normalisation(L):
    """PeakScaler round trip is exact, and the largest real sample maps to HEADROOM < 1."""
    print(f"\n7. Peak normalisation [{L.pulse}, sps={L.sps}]")
    seg = L.real_segments(64, models.SEG_LEN)
    sc = models.PeakScaler.fit(seg)
    check("normalize -> max|.| == HEADROOM", sc.normalize(seg).abs().max().item(),
          models.HEADROOM, 1e-6)
    rt = sc.denormalize(sc.normalize(seg))
    check("denormalize(normalize(x)) == x", (rt - seg).abs().max().item(), 0.0, 1e-6)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    VERBOSE = ap.parse_args().v

    device = lk.setup(seed=1234)
    print(f"device: {device}")
    for sps, pulse in [(8, "rrc0.35"), (4, "rect")]:
        L = lk.Link(sps=sps, pulse=pulse)
        if VERBOSE:
            print(f"\n[{pulse}, sps={sps}] filter length {L.filt.length}, delay {L.delay}, "
                  f"c0 {L.c0:.6f}, kappa(inband) {L.kappa('inband'):.4f}")
        test_waveform(L)
        test_clean_vs_theory(L)
        test_noise_jammer_vs_closed_form(L)
        test_jsr_is_exact(L)
        test_optimal_geometry(L)
        test_perfect_generator(L)
        test_normalisation(L)
    test_models()

    print("\n" + ("ALL CHECKS PASS" if not FAILURES else
                  f"{len(FAILURES)} FAILED:\n  " + "\n  ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
