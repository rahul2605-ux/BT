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

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import: no OptiX on the cluster (scene.py)

import argparse
import math
import sys

import numpy as np
import torch

import link as lk
import jammers
import models
import losses
import attacks
import channel
import detectors
import scene

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


# ================================================================ BASELINES (2026-09-17)
# The 3-D scene, the Sionna channel, the attacks and the detectors of the classical
# baseline set (README §2.10). Only on the decided link (link.LINK).

def _frame_ber(out):
    """Per-frame BER [F] (errors are correlated within a frame: one offset/phase each)."""
    return (out["bits"] != out["bits_hat"]).double().mean(dim=-1)


def _same_ber(label, a, b, floor=2e-4):
    """Two Monte-Carlo BER estimates agree within 4 sigma of their per-frame spread."""
    fa, fb = _frame_ber(a), _frame_ber(b)
    tol = 4 * math.sqrt(fa.var() / fa.numel() + fb.var() / fb.numel()) + floor
    check(label, float(fa.mean() - fb.mean()), 0.0, float(tol))


def _batches(L, n, spec, jsr, snr_db, batch=1024):
    outs = [attacks.frames(L, min(batch, n - i), spec, jsr, snr_db) for i in range(0, n, batch)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def test_scene():
    """
    Sionna RT in an empty scene must reproduce free space exactly; T's power control
    must land every drop at SNR 30 dB with the cap never binding.
    """
    print("\n8. Scene: Sionna RT path gains, noise floor, power control")
    pos = np.stack([scene.draw_positions(7000 + i) for i in range(12)])
    seps = [np.linalg.norm(p[:, None] - p[None], axis=-1)[np.triu_indices(p.shape[0], 1)].min()
            for p in pos]
    check("min pairwise separation >= 50 m (12 drops)", float(min(seps) >= scene.MIN_SEPARATION),
          1.0, 0)
    check("all nodes inside the box", float(np.all((pos >= 0) & (pos <= np.array(scene.BOX)))),
          1.0, 0)
    g, tau = scene.path_gains(pos, return_delays=True)
    d = np.linalg.norm(pos[:, [0] + list(range(2, pos.shape[1]))] - pos[:, [1]], axis=-1)
    err_db = np.abs(10 * np.log10(g) - 10 * np.log10(scene.free_space_gain(d))).max()
    check("RT LOS gain == (lambda/4 pi d)^2, max |dB error|", float(err_db), 0.0, 0.01)
    check("RT LOS delay == d/c, max |error| [ns]", float(np.abs(tau - d / 299_792_458.0).max() * 1e9),
          0.0, 0.01)
    n_w = scene.noise_power_w()
    want = 10 * math.log10(1.380649e-23 * scene.TEMPERATURE * scene.BANDWIDTH * 1e3) \
        + scene.NOISE_FIGURE_DB
    check("noise floor k T B NF [dBm]", float(scene.w_to_dbm(n_w)), want, 0.05)
    p_t = scene.tx_power_w(g[:, 0], n_w)
    snr = 10 * np.log10(p_t * g[:, 0] / n_w)
    check("power control: realised SNR at R, max |dev from 30 dB|", float(np.abs(snr - 30).max()),
          0.0, 0.05)
    # the cap: the farthest possible T-R pair in the box
    far = scene.free_space_gain(np.linalg.norm(scene.BOX))
    check("power control cap never binds (farthest pair) [dB margin > 0]",
          float(scene.TX_MAX_DBM - scene.w_to_dbm(n_w * 10 ** 3 / far) > 0), 1.0, 0)
    jsr = scene.received_jsr(g[:, 1:], 20.0, 3, n_w * 1e3)
    back = jsr * n_w * 1e3 * 3 / g[:, 1:4]
    check("received_jsr inverts to the budget (equal split)",
          float(np.abs(scene.w_to_dbm(back) - 20.0).max()), 0.0, 1e-6)


def test_channel(L):
    """
    The Sionna one-path channel: an integer delay is an exact shift; a fractional
    delay matches an independent FFT delay of the band-limited waveform; every
    jammer lands at exactly its JSR; independent jammers add in power.
    """
    print("\n9. Channel: Sionna cir_to_time_channel + ApplyTimeChannel")
    F, N = 64, scene.N_SYM
    x = attacks.pulsed_tx(L, F, N, 1.0)
    act = L.active(N)
    zero = torch.zeros(F, device=L.device)
    ref = channel.receive(L, x, 0.0, zero, zero)
    y3 = channel.receive(L, x, 0.0, zero + 3, zero)
    rel = ((y3[:, act.start + 3:act.stop] - ref[:, act.start:act.stop - 3]).abs().pow(2).mean()
           / ref[:, act].abs().pow(2).mean()).sqrt()
    check("integer delay 3 == 3-sample shift (relative RMS error)", float(rel), 0.0, 5e-3,
          note="each output is re-scaled over its own window")
    delta = 2.37
    yd = channel.receive(L, x, 0.0, zero + delta, zero)
    X = torch.fft.fft(x.to(torch.complex128), dim=-1)
    f = torch.fft.fftfreq(x.shape[-1], device=L.device).double()
    xf = torch.fft.ifft(X * torch.exp(-2j * math.pi * f * delta), dim=-1)
    xf = xf[:, channel.PAD:channel.PAD + ref.shape[-1]]
    xf = jammers.scale_to_jsr(xf.to(torch.complex64), L, 0.0)
    rel = ((yd[:, act] - xf[:, act]).abs().pow(2).mean() / xf[:, act].abs().pow(2).mean()).sqrt()
    check("fractional delay 2.37 == FFT delay (relative RMS error)", float(rel), 0.0, 0.01)
    for jsr_db in (-17.0, -3.0, 6.0):
        delay, phase = channel.async_draw(L, F)
        j = channel.receive(L, x, jsr_db, delay, phase)
        p = 10 * math.log10(float(j[:, act].abs().pow(2).mean(-1).max()) / L.p_s)
        q = 10 * math.log10(float(j[:, act].abs().pow(2).mean(-1).min()) / L.p_s)
        check(f"JSR {jsr_db:+.0f} dB exact per frame (worst frame, dB)",
              max(abs(p - jsr_db), abs(q - jsr_db)), 0.0, 0.01)
    bits, sym, _ = L.modulate(1024, N)
    jsrs = [-3.0, -6.0, -10.0]
    j = attacks.jammer_at_rx(L, dict(name="pulsed", p=1.0), sym, jsrs)
    tot = 10 * math.log10(float(j[:, act].abs().pow(2).mean()) / L.p_s)
    want = 10 * math.log10(sum(10 ** (v / 10) for v in jsrs))
    check("3 async jammers add in power (dB)", tot, want, 0.05)


def test_attacks_vs_references(L):
    """
    Noise against its closed form; async matched QPSK against step 1's
    jammers.optimal(async); the pulsed duty cycle; the omniscient genie's exact
    BER and its budget; and the two SIMPLIFICATIONS the model rests on.
    """
    print("\n10. Attacks vs references, and the simplifications")
    snr, N = lk.SNR_DB, scene.N_SYM
    for jsr_db in (0.0, 3.0):
        out = _batches(L, 4096, dict(name="noise"), [jsr_db], snr)
        e, b, _, _ = attacks.error_counts(out)
        want = float(L.ber_noise_jammer(jsr_db, snr, band="full"))
        check(f"noise JSR {jsr_db:+.0f} dB: BER vs closed form", e / b, want, mc_tol(want, b))
    # async matched QPSK with INTEGER offsets == step 1's optimal(async), same law
    for jsr_db in (-4.0, 0.0):
        ours = []
        for _ in range(2):
            bits, sym, x = L.modulate(2048, N)
            r = L.awgn(x, L.noise_var(snr))
            delay = torch.randint(0, L.sps, (2048,), device=L.device).float()
            _, phase = channel.async_draw(L, 2048)
            r = r + channel.receive(L, attacks.pulsed_tx(L, 2048, N, 1.0), jsr_db, delay, phase)
            ours.append(dict(bits=bits, bits_hat=L.decide(L.matched_filter(r, N))))
        ours = {k: torch.cat([o[k] for o in ours]) for k in ours[0]}
        ref = []
        for _ in range(2):
            bits, sym, x = L.modulate(2048, N)
            r = L.awgn(x, L.noise_var(snr)) + jammers.make("optimal", jsr_db, sync="async")(L, 2048, N)
            ref.append(dict(bits=bits, bits_hat=L.decide(L.matched_filter(r, N))))
        ref = {k: torch.cat([o[k] for o in ref]) for k in ref[0]}
        _same_ber(f"pulsed(1) integer-async == step-1 optimal(async), JSR {jsr_db:+.0f} dB",
                  ours, ref)
    # duty cycle: matched-filter the undelayed pulsed waveform at the symbol instants
    for p in (0.5, 0.1):
        x = attacks.pulsed_tx(L, 512, N, p)[:, channel.PAD:channel.PAD + L.waveform_length(N)]
        z = L.matched_filter(x, N)
        on = float((z.abs() > 0.5 / math.sqrt(p)).double().mean())
        check(f"pulsed(p={p}) realised ON fraction", on, p, 4 * math.sqrt(p * (1 - p) / z.numel()))
    # omniscient: exact BER = floor(delta N)/N, never over budget
    act = L.active(N)
    for eta, jsr_db in ((1.0, 10 * math.log10(4.0)), (1.0, 0.0), (0.1, -3.0)):
        out = _batches(L, 1024, dict(name="omniscient", eta=eta), jsr_db, snr)
        e, b, _, _ = attacks.error_counts(out)
        delta = min(1.0, 10 ** (jsr_db / 10) / (1 + eta) ** 2)
        want = math.floor(delta * N) / N
        check(f"omniscient(eta={eta}) JSR {jsr_db:+.1f} dB: BER == floor(delta N)/N", e / b, want,
              2e-3)
        bits, sym, _ = L.modulate(256, N)
        d = attacks.omniscient_rx(L, sym, jsr_db, eta)
        used = float(d[:, act].abs().pow(2).mean()) / L.p_s
        check(f"omniscient(eta={eta}) JSR {jsr_db:+.1f} dB spends <= budget (ratio)",
              float(used <= 10 ** (jsr_db / 10) * 1.02), 1.0, 0)
    # SIMPLIFICATION 1: a geometric delay and carrier phase on top of the async draw
    # change nothing (law-invariance). 17.3 samples ~ 650 m of extra path.
    F = 4096
    outs = {}
    for tag, extra_delay, extra_phase in (("async only", 0.0, 0.0), ("async + geometry", 17.3, 1.1)):
        bits, sym, x = L.modulate(F, N)
        r = L.awgn(x, L.noise_var(snr))
        delay, phase = channel.async_draw(L, F)
        r = r + channel.receive(L, attacks.pulsed_tx(L, F, N, 1.0), -3.0, delay + extra_delay,
                                phase + extra_phase, l_min=-32, l_max=64)
        outs[tag] = dict(bits=bits, bits_hat=L.decide(L.matched_filter(r, N)), r=r)
    _same_ber("geometric delay/phase on vs off: same BER (pulsed(1), -3 dB)",
              outs["async + geometry"], outs["async only"])
    for name, fn in (("power", detectors.power), ("kurtosis", detectors.kurtosis)):
        a_, b_ = fn(outs["async + geometry"]["r"]), fn(outs["async only"]["r"])
        tol = 4 * math.sqrt(a_.var() / F + b_.var() / F)
        check(f"geometric delay/phase on vs off: same mean {name} statistic",
              float(a_.mean() - b_.mean()), 0.0, float(tol))
    # SIMPLIFICATION 2: K white-noise jammers == one at the total JSR
    two = _batches(L, 4096, dict(name="noise"), [-3.0, -6.0], snr)
    one = _batches(L, 4096, dict(name="noise"), [10 * math.log10(10 ** -0.3 + 10 ** -0.6)], snr)
    _same_ber("2 noise jammers == 1 at the total JSR: BER", two, one)


def test_detectors(L):
    """
    Every detector honours its false-alarm rate on fresh clean frames; the
    omniscient flip is invisible to power and kurtosis; and on the noise jammer
    the likelihood-ratio test is at least as good as every other detector.
    """
    print("\n11. Detectors: calibration, invisibility of the flip, LRT optimality")
    snr, N = lk.SNR_DB, scene.N_SYM
    n0 = L.noise_var(snr)
    n_cal, n_test = 16384, 16384

    def stats(spec, jsr, n):
        out = _batches(L, n, spec, jsr, snr)
        return detectors.statistics(out["r"], out["z"])

    cal = stats(None, None, n_cal)
    test = stats(None, None, n_test)
    pm, km = float(cal["power"].mean()), float(cal["kurtosis"].mean())
    stat = {
        "power (one-sided)": lambda s: s["power"],
        "power (two-sided)": lambda s: detectors.two_sided(s["power"], pm),
        "kurtosis (two-sided)": lambda s: detectors.two_sided(s["kurtosis"], km),
    }
    thr = {}
    for a in detectors.ALPHAS:
        for name, fn in stat.items():
            thr[(name, a)] = detectors.calibrate(fn(cal), a)
            far = detectors.p_detect(fn(test), thr[(name, a)])
            check(f"{name}: realised FAR at alpha={a}", far, a, 4 * math.sqrt(a * (1 - a) / n_test))
    flip = stats(dict(name="omniscient", eta=1.0), 10 * math.log10(4.0), 8192)
    for name in ("power (two-sided)", "kurtosis (two-sided)"):
        p = detectors.p_detect(stat[name](flip), thr[(name, 0.05)])
        check(f"omniscient flip (BER 1) is invisible to {name}: P(det) == FAR", p, 0.05,
              4 * math.sqrt(0.05 * 0.95 / 8192))
    for jsr_db in (-45.0, -35.0, -20.0):
        jam = stats(dict(name="noise"), [jsr_db], 4096)
        jsr = 10 ** (jsr_db / 10)
        llr_cal = detectors.lrt_noise(cal["lrt"], n0, jsr, L.p_s)
        llr_jam = detectors.lrt_noise(jam["lrt"], n0, jsr, L.p_s)
        p_lrt = detectors.p_detect(llr_jam, detectors.calibrate(llr_cal, 0.05))
        others = {name: detectors.p_detect(fn(jam), thr[(name, 0.05)]) for name, fn in stat.items()}
        best = max(others.values())
        if VERBOSE:
            print(f"      JSR {jsr_db:+.0f} dB: P_lrt {p_lrt:.4f}  " +
                  "  ".join(f"{k} {v:.4f}" for k, v in others.items()))
        check(f"noise JSR {jsr_db:+.0f} dB: LRT >= best other detector (P_lrt - best)",
              float(p_lrt - best >= -4 * math.sqrt(0.25 / 4096)), 1.0, 0)
    far_lrt = detectors.p_detect(detectors.lrt_noise(test["lrt"], n0, 10 ** -4.5, L.p_s),
                                 detectors.calibrate(detectors.lrt_noise(cal["lrt"], n0, 10 ** -4.5,
                                                                         L.p_s), 0.05))
    check("LRT: realised FAR at alpha=0.05", far_lrt, 0.05, 4 * math.sqrt(0.05 * 0.95 / n_test))


def test_spectrogram_cnn(L):
    """Only once train_spectrogram_cnn.py has run: FAR on fresh clean frames, and it learned."""
    import os
    path = os.path.join(scene.ART, "detector_spec.pt")
    print("\n12. Spectrogram CNN (Li et al., retrained)")
    if not os.path.exists(path):
        print("  [SKIP] no checkpoint yet (artifacts/cgan/baselines/detector_spec.pt) -- "
              "run train_spectrogram_cnn.py, then re-run verify")
        return
    net, sc, ckpt = detectors.load_cnn(path, L.device)
    snr, n = lk.SNR_DB, 8192
    clean = _batches(L, n, None, None, snr)
    s = detectors.cnn_statistic(net, clean["r"], sc)
    for a in detectors.ALPHAS:
        far = detectors.p_detect(s, ckpt["thresholds"]["cnn"][str(a)])
        check(f"spec_cnn: realised FAR at alpha={a} (fresh clean)", far, a,
              4 * math.sqrt(a * (1 - a) / n))
    jam = _batches(L, 1024, dict(name="barrage"), [10.0], snr)
    p = detectors.p_detect(detectors.cnn_statistic(net, jam["r"], sc), ckpt["thresholds"]["cnn"]["0.05"])
    check("spec_cnn detects barrage at +10 dB (P(det) > 0.5)", float(p > 0.5), 1.0, 0,
          note=f"P(det) = {p:.4f}")


# ================================================================ D2a (2026-09-19)
def _frames_nf(L, n, spec, jsr, snr_db, batch=1024):
    outs = [attacks.frames(L, min(batch, n - i), spec, jsr, snr_db, noiseless=True)
            for i in range(0, n, batch)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def test_shaped(L):
    """
    The learned control tier (train_shaped.py): theta = 0 of the shaped family is
    the barrage jammer to every detector; any theta lands at exactly its JSR; the
    FFT shaping realises the PSD it is given; and the exact expected BER the
    optimiser scores with agrees with the closed form and with Monte-Carlo BER on
    the same frames, for white and for in-band-concentrated noise.
    """
    import os
    print("\n13. Shaped-noise jammer (D2a) and the exact expected BER")
    snr, N = lk.SNR_DB, scene.N_SYM
    n0 = L.noise_var(snr)
    shaped0 = dict(name="shaped", theta=[0.0] * attacks.SHAPED_DIM)
    noise = dict(name="noise")
    for jsr_db in (0.0, 3.0):
        _same_ber(f"shaped(theta=0) == noise: BER at JSR {jsr_db:+.0f} dB",
                  _batches(L, 4096, shaped0, [jsr_db], snr), _batches(L, 4096, noise, [jsr_db], snr))
    cal = _batches(L, 8192, None, None, snr)["r"]
    km = float(detectors.kurtosis(cal).mean())
    stat = {"power (one-sided)": detectors.power,
            "kurtosis (two-sided)": lambda r: detectors.two_sided(detectors.kurtosis(r), km)}
    at = {"power (one-sided)": -27.0, "kurtosis (two-sided)": -13.0}     # mid-transition (§3.3d)
    path = os.path.join(scene.ART, "detector_spec.pt")
    if os.path.exists(path):
        net, sc, ckpt = detectors.load_cnn(path, L.device)
        stat["spec_cnn"] = lambda r: detectors.cnn_statistic(net, r, sc)
        at["spec_cnn"] = -30.0
    for name, fn in stat.items():
        thr = detectors.calibrate(fn(cal), 0.05)
        a = detectors.p_detect(fn(_batches(L, 4096, shaped0, [at[name]], snr)["r"]), thr)
        b = detectors.p_detect(fn(_batches(L, 4096, noise, [at[name]], snr)["r"]), thr)
        check(f"shaped(theta=0) == noise: {name} P(det) at {at[name]:+.0f} dB", a - b, 0.0,
              4 * math.sqrt(2 * 0.25 / 4096), note=f"{a:.3f} vs {b:.3f}")
    g = torch.Generator().manual_seed(5)
    theta = (2.0 * torch.randn(attacks.SHAPED_DIM, generator=g, dtype=torch.float64)).tolist()
    _, sym, _ = L.modulate(256, N)
    act = L.active(N)
    for jsr_db in (-20.0, 0.0):
        j = attacks.jammer_at_rx(L, dict(name="shaped", theta=theta), sym, [jsr_db])
        pw = 10 * torch.log10(j[:, act].abs().pow(2).mean(-1).double() / L.p_s)
        check(f"shaped(random theta) JSR {jsr_db:+.0f} dB exact per frame (worst, dB)",
              float((pw - jsr_db).abs().max()), 0.0, 0.01)
    th = theta[:attacks.SHAPED_BINS] + [0.0] * attacks.SHAPED_PERIOD
    x = attacks.shaped_tx(L, 512, N, th).to(torch.complex128)
    n = x.shape[-1]
    ratio = torch.fft.fft(x, dim=-1).abs().pow(2).mean(0).cpu() / n \
        / torch.pow(10.0, attacks.shaped_psd_db(th, n) / 10.0)
    b = ((torch.fft.fftfreq(n, dtype=torch.float64) + 0.5) * attacks.SHAPED_BINS).floor().long() \
        .clamp(0, attacks.SHAPED_BINS - 1)
    per_bin = torch.zeros(attacks.SHAPED_BINS, dtype=torch.float64).index_add_(0, b, ratio) \
        / torch.bincount(b, minlength=attacks.SHAPED_BINS)
    check("shaped: realised PSD == requested, worst of 32 bins (dB)",
          float((10 * torch.log10(per_bin)).abs().max()), 0.0, 0.2)
    out = _frames_nf(L, 4096, noise, [0.0], snr)
    e, bits, _, _ = attacks.error_counts(out)
    eb = attacks.expected_ber(out, n0)
    want = float(L.ber_noise_jammer(0.0, snr, band="full"))
    check("E[BER] (exact over AWGN) vs closed form, noise 0 dB", eb, want, mc_tol(want, bits))
    check("E[BER] vs Monte-Carlo BER, same frames, noise 0 dB", eb, e / bits, mc_tol(eb, bits))
    centres = (torch.arange(attacks.SHAPED_BINS) + 0.5) / attacks.SHAPED_BINS - 0.5
    inband = torch.where(centres.abs() < 0.09, 4.0, -4.0).tolist() + [0.0] * attacks.SHAPED_PERIOD
    out = _frames_nf(L, 4096, dict(name="shaped", theta=inband), [-6.0], snr)
    fe, fh = attacks.expected_ber(out, n0, per_frame=True), _frame_ber(out)
    tol = 4 * float((fh - fe).std()) / math.sqrt(fh.numel()) + 1e-6
    check("E[BER] vs Monte-Carlo BER, same frames, in-band shaped -6 dB", float(fe.mean()),
          float(fh.mean()), tol, note=f"BER {float(fh.mean()):.2e}")



# ================================================================ 14. GAN jammer (D1/D2)
def test_gan_jammer(L):
    """
    The GAN jammer on the 3-D attacks/channel path (attacks.gan_tx + channel.receive
    async), for D1 (put run001_G on the plane) and D2 (train against a detector).

      14a a perfect generator through this path == pulsed(1) async (which verify
          already ties to the step-1 optimal jammer). Tolerance carries the same
          +0.01 tile-boundary slack as section 5b: independent 1024-sample segments
          lose their neighbours' pulse tails at the join.
      14b every gan frame lands at exactly its requested JSR (hard projection).
      14c THE D2 checkpoint: the full training loss -(log E[BER] - beta*P_det_soft)
          reaches the generator's parameters, i.e. autograd flows through
          channel.receive (Sionna ApplyTimeChannel), the matched filter and the
          detector. A finite, nonzero gradient on G's weights de-risks the method.
      14d THE CNN checkpoint: the straight-through colour LUT does not change the
          deployed CNN's statistic -- exactly, at matched precision -- and the loss
          built on it reaches G. Without it the CNN has zero gradient everywhere and
          cannot be a D2 target at all. The deployed path autocasts to fp16 on CUDA
          while training runs in fp32, so that separate effect is measured on its own
          and only has to leave the DECISIONS alone.
    """
    print(f"\n14. GAN jammer on the 3-D link [{L.pulse}, sps={L.sps}]")
    snr, N = lk.SNR_DB, scene.N_SYM
    n0 = L.noise_var(snr)
    PG = PerfectG(L)
    gan = dict(name="gan", G=PG, scale=PG.scaler.scale)
    pulsed = dict(name="pulsed", p=1.0)

    # 14a: perfect-G gan path == pulsed(1), both async
    for jsr_db in (0.0, 3.0):
        _same_ber(f"gan(perfect G) == pulsed(1) async, JSR {jsr_db:+.0f} dB",
                  _batches(L, 8000, gan, [jsr_db], snr),
                  _batches(L, 8000, pulsed, [jsr_db], snr), floor=0.01)

    # 14b: realised JSR exact per frame
    _, sym, _ = L.modulate(256, N)
    act = L.active(N)
    for jsr_db in (-20.0, 0.0):
        j = attacks.jammer_at_rx(L, gan, sym, [jsr_db])
        pw = 10 * torch.log10(j[:, act].abs().pow(2).mean(-1).double() / L.p_s)
        check(f"gan JSR {jsr_db:+.0f} dB exact per frame (worst, dB)",
              float((pw - jsr_db).abs().max()), 0.0, 0.01)

    # 14c: differentiability of the D2 loss w.r.t. G's parameters
    device = L.device
    G = models.Generator(n_classes=1).to(device)
    for p in G.parameters():
        p.grad = None
    spec = dict(name="gan", G=G, scale=1.0, grad=True)
    out = attacks.frames(L, 64, spec, [-6.0], snr, noiseless=True)
    log_ber = attacks.log_expected_ber(out, n0)
    stat = detectors.power(out["r"])
    thr = float(stat.detach().mean())
    sc = float(stat.detach().std()) + 1e-12
    soft = detectors.soft_pdet(stat, thr, sc)
    loss = -(log_ber - 1.0 * soft)
    loss.backward()
    gnorm = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in G.parameters()
                          if p.grad is not None))
    n_with_grad = sum(1 for p in G.parameters() if p.grad is not None)
    n_params = sum(1 for _ in G.parameters())
    check("D2 loss is finite", float(math.isfinite(float(loss.detach()))), 1.0, 0.0,
          note=f"loss={float(loss.detach()):.4g}")
    check("D2 gradient reaches all of G's parameter tensors", float(n_with_grad), float(n_params), 0.0)
    check("D2 gradient on G is finite and nonzero", float(math.isfinite(gnorm) and gnorm > 0), 1.0,
          0.0, note=f"||grad|| = {gnorm:.4g}")

    # 14d: the CNN as a D2 target -- straight-through LUT, forward unchanged
    import os
    path = os.path.join(scene.ART, "detector_spec.pt")
    if not os.path.exists(path):
        print("  [SKIP] 14d: no detector_spec.pt -- run train_spectrogram_cnn.py first")
        return
    net, sc, ckpt = detectors.load_cnn(path, device)
    thr_cnn = ckpt["thresholds"]["cnn"][str(detectors.HEADLINE_ALPHA)]
    clean = _batches(L, 1024, None, None, snr)["r"]
    s_dep = detectors.cnn_statistic(net, clean, sc)                            # deployed path
    spread = float(s_dep.std())
    with torch.no_grad():
        s_ste = torch.cat([detectors.cnn_statistic(net, clean[i:i + 64], sc, grad=True)
                           for i in range(0, clean.shape[0], 64)])

    # (i) the LUT change ALONE, at matched precision. The deployed statistic runs under
    # autocast(fp16) on CUDA and the training path in fp32 (fp16 grads underflow), so the
    # two must be separated: the image is built identically either way, and on CPU -- where
    # neither path autocasts -- the whole statistic is identical too. Job 2267938 conflated
    # the two and charged fp16 rounding to the LUT.
    with torch.no_grad():
        img_h = detectors.spectrogram_image(clean[:32], sc)
        img_s = detectors.spectrogram_image(clean[:32], sc, grad=True)
    check("spec_cnn: straight-through LUT reproduces the deployed image exactly",
          float((img_h - img_s).abs().max()), 0.0, 1e-6)
    net_cpu, sc_cpu, _ = detectors.load_cnn(path, "cpu")
    r_cpu = clean[:32].cpu()
    with torch.no_grad():
        c_dep = detectors.cnn_statistic(net_cpu, r_cpu, sc_cpu)
        c_ste = detectors.cnn_statistic(net_cpu, r_cpu, sc_cpu, grad=True)
    check("spec_cnn: STE == deployed statistic at matched precision (fp32, CPU)",
          float((c_dep - c_ste).abs().max()), 0.0, 1e-3)

    # (ii) what fp16 autocast alone costs, reported: it must not move the DECISIONS.
    d = (s_ste - s_dep) / spread
    print(f"  [INFO] fp32 vs deployed fp16 statistic: bias {float(d.mean()):+.4f}, "
          f"sd {float(d.std()):.4f}, max |.| {float(d.abs().max()):.4f} (units of clean std)")
    # reported, not asserted: no measurement exists yet to set a tolerance from, and the
    # scientific requirement is the decision-level agreement checked next, not the raw value.
    check("spec_cnn: straight-through LUT gives the same decisions at alpha=0.05",
          float(((s_ste > thr_cnn) != (s_dep > thr_cnn)).double().mean()), 0.0, 0.01)

    jam = _batches(L, 512, dict(name="barrage"), [-5.0], snr)["r"]
    s_dep_j = detectors.cnn_statistic(net, jam, sc)
    with torch.no_grad():
        s_ste_j = torch.cat([detectors.cnn_statistic(net, jam[i:i + 64], sc, grad=True)
                             for i in range(0, jam.shape[0], 64)])
    check("spec_cnn: same P(det) on jammed frames, STE vs deployed",
          detectors.p_detect(s_ste_j, thr_cnn), detectors.p_detect(s_dep_j, thr_cnn), 0.01,
          note=f"deployed {detectors.p_detect(s_dep_j, thr_cnn):.3f}")

    for p in G.parameters():
        p.grad = None
    out = attacks.frames(L, 16, spec, [-6.0], snr, noiseless=True)
    stat = detectors.cnn_statistic(net, out["r"], sc, grad=True)
    soft = detectors.soft_pdet(stat, thr_cnn, spread)
    loss = -(attacks.log_expected_ber(out, n0) - 1.0 * soft)
    loss.backward()
    gnorm = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in G.parameters()
                          if p.grad is not None))
    check("spec_cnn: D2 loss gradient on G is finite and nonzero",
          float(math.isfinite(gnorm) and gnorm > 0), 1.0, 0.0, note=f"||grad|| = {gnorm:.4g}")
    dg = torch.autograd.grad(detectors.soft_pdet(
        detectors.cnn_statistic(net, attacks.frames(L, 16, spec, [-6.0], snr)["r"], sc, grad=True),
        thr_cnn, spread), list(G.parameters()), allow_unused=True)
    gn_det = math.sqrt(sum(float(g.pow(2).sum()) for g in dg if g is not None))
    check("spec_cnn: the DETECTOR term alone reaches G (not just the BER term)",
          float(math.isfinite(gn_det) and gn_det > 0), 1.0, 0.0, note=f"||grad|| = {gn_det:.4g}")

def test_baselines():
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    test_scene()
    test_channel(L)
    test_attacks_vs_references(L)
    test_detectors(L)
    test_spectrogram_cnn(L)
    test_shaped(L)
    test_gan_jammer(L)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--baselines-only", action="store_true",
                    help="run only sections 8-14 (scene, channel, attacks, detectors, CNN, shaped, GAN)")
    args = ap.parse_args()
    VERBOSE = args.v

    device = lk.setup(seed=1234)
    print(f"device: {device}")
    if not args.baselines_only:
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
    test_baselines()

    print("\n" + ("ALL CHECKS PASS" if not FAILURES else
                  f"{len(FAILURES)} FAILED:\n  " + "\n  ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
