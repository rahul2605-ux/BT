"""
CGAN track -- the waveform-level QPSK link Zhou et al. 2025 jam, built from Sionna.

M0 works on one symbol at a time. Zhou's jammer is a *waveform* (1024 I/Q
samples), so this link keeps exactly the extra layers a waveform needs --
oversampling, a pulse-shaping filter, a matched filter, symbol-time sampling --
and nothing else: one channel (h = 1), AWGN, no fading, no synchronisation
errors on the victim's side.

SIGNAL MODEL
------------
Per frame of N symbols s[k], iid uniform Gray QPSK, unit energy:

    x[n] = sum_k s[k] h[n - k*sps]                       transmitted waveform
    r[n] = x[n] + w[n] + j[n]                            w ~ CN(0, N0), j = jammer
    z[k] = (r * h)[delay + k*sps]                        matched filter, sampled
    bits = per-axis sign of z[k]                         ML for QPSK in Gaussian noise

`h` is unit-energy (Sionna `normalize=True`), so the cascade h*h peaks at 1 and
is Nyquist: z[k] = s[k] + (filtered noise and jammer), no ISI for the RRC family
up to filter truncation, none at all for rect.

CONVENTIONS -- the definitions Zhou leaves unstated (README §4.2 Q7)
--------------------------------------------------------------------
* Gray labelling from Sionna's "qam" constellation: bit0 -> sign of I, bit1 ->
  sign of Q, bit 0 -> positive. Same map as m0/link.py.
* Average transmitted power per sample is P_s = ||h||^2 / sps = 1/sps.
* SNR and JSR are ratios of MEAN POWER PER SAMPLE at the receiver input, over
  the full simulated bandwidth (sps x symbol rate):
      N0         = P_s / SNR
      E|j[n]|^2  = JSR * P_s        (imposed on each frame by jammers.scale_to_jsr)
* Pulse names: "rect" (sps taps + one zero tap, because Sionna filters need an odd
  length; the zero only delays the cascade, which `delay` absorbs) or
  "rrc<beta>", e.g. "rrc0.35", with span RRC_SPAN symbols.
* Decisions are hard sign tests on z[k] directly, not Sionna's Demapper: the
  Demapper needs a noise variance `no` (README §C.2), which for a jammed link is
  exactly the quantity nobody knows.

CLOSED FORM FOR A GAUSSIAN JAMMER (what C2 fits and verify.py checks)
---------------------------------------------------------------------
A jammer that is complex Gaussian with per-sample variance JSR*P_s, shaped by a
unit-energy filter f (f = delta for "full"-band white noise, f = h for "inband"),
reaches the decision with variance (JSR*P_s) * kappa, kappa = sum_n |(f*h)[n]|^2.
With c0 = (h*h)[peak] (= 1 up to truncation), per axis:

    BER = Q( c0 / sqrt(N0 + JSR*P_s*kappa) )
        = Q( c0 * sqrt( sps / (1/SNR + JSR*kappa) ) )

For white noise kappa = 1, so the jammer is suppressed by the full processing
gain sps. For in-band noise kappa > 1 and the gain shrinks to about 1.8 dB for
rect and about 0.5 dB for RRC. That kappa is what decides C2's `band` choice.
"""

import math

import numpy as np
import torch

import sionna.phy as sn
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.signal import CustomFilter, RootRaisedCosineFilter, Upsampling
from sionna.phy.channel import AWGN

SNR_DB = 30.0      # Zhou, §III

# The link step 1 is built on -- decided with the user at the C2 checkpoint
# (2026-09-14) as STATED ASSUMPTIONS, not as fitted values. Each has a reason
# independent of Zhou's figure; calibrate.py (job 2259157) only shows where they
# land against it: Noise crosses BER 1e-3 at -0.67 dB (paper -1.32), Optimal at
# -6.00 dB (paper -7.06). README §2.10.
#   sps 8            common simulation choice, well above Nyquist (sps > 1 + beta);
#                    > 1 is REQUIRED once positions/delays enter the channel
#   rrc0.35          the classic default roll-off (e.g. DVB-S)
#   noise "full"     a classical barrage jammer, white over the simulated band
#   optimal "async"  a jammer is not synchronised to its victim unless assumed;
#                    it is also the timing a spatial channel produces
LINK = dict(sps=8, pulse="rrc0.35", noise_band="full", optimal_sync="async")
# Symbols. Zhou states no filter at all, so the link must not add errors of its
# own. Span 8 left summed truncation ISI of 3.6% of the symbol amplitude
# (beta 0.35), enough to flip symbols whose margin is a few percent (verify job
# 2259141); span 32 brings it to 0.3%, below every margin the checks rely on.
RRC_SPAN = 32


def setup(device=None, seed=None):
    """Sionna reads device/seed at block construction (README §C.2): call first."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sn.config.device = device
    if seed is not None:
        sn.config.seed = seed       # Sionna blocks: BinarySource, AWGN
        torch.manual_seed(seed)     # plain-torch draws: noise jammer, phases, offsets
    return device


def q_function(z):
    """Gaussian tail Q(z) = 0.5*erfc(z/sqrt(2)), elementwise on floats/arrays."""
    return 0.5 * np.vectorize(math.erfc)(np.asarray(z, dtype=float) / math.sqrt(2.0))


def make_pulse(pulse, sps, device):
    """Unit-energy pulse-shaping filter as a Sionna block."""
    if pulse == "rect":
        coeffs = np.concatenate([np.ones(sps), np.zeros(1)]).astype(np.float32)
        return CustomFilter(samples_per_symbol=sps, coefficients=coeffs, device=device)
    if pulse.startswith("rrc"):
        beta = float(pulse[3:])
        return RootRaisedCosineFilter(span_in_symbols=RRC_SPAN, samples_per_symbol=sps,
                                      beta=beta, device=device)
    raise ValueError(f"unknown pulse {pulse!r}")


class Link:
    """One QPSK link at a fixed (sps, pulse). All tensors live on `device`."""

    def __init__(self, sps=8, pulse="rrc0.35", device=None):
        self.device = setup(device)
        self.sps, self.pulse = sps, pulse
        self.source = BinarySource(device=self.device)
        self.mapper = Mapper("qam", 2, device=self.device)
        self.upsample = Upsampling(sps, device=self.device)
        self.filt = make_pulse(pulse, sps, self.device)
        self.awgn = AWGN(device=self.device)
        self.p_s = 1.0 / sps

        # Cascade response measured through the actual Sionna blocks, so the
        # sampling delay does not depend on whether Sionna convolves or correlates.
        imp = torch.zeros(1, 4 * self.filt.length, dtype=torch.complex64, device=self.device)
        imp[0, 0] = 1.0
        self._casc = self.filt(self.filt(imp, padding="full"), padding="full")[0]
        self.delay = int(torch.argmax(self._casc.abs()).item())
        self.c0 = float(self._casc[self.delay].abs().item())

    # ---------------------------------------------------------------- TX side
    def modulate(self, n_frames, n_sym):
        """-> (bits [F, 2N], symbols [F, N], waveform x [F, N*sps + K - 1])."""
        bits = self.source([n_frames, 2 * n_sym])
        sym = self.mapper(bits)
        x = self.filt(self.upsample(sym), padding="full")
        return bits, sym, x

    def real_segments(self, n_frames, seg_len):
        """
        Clean QPSK waveform segments as GAN training data (Zhou's "target signal"):
        (n_frames, 2, seg_len) real I/Q, cropped from the steady middle of a longer
        frame so no filter tail is included. Enough symbols are generated to fill
        seg_len at this sps.
        """
        n_sym = seg_len // self.sps + 2 * (self.filt.length // self.sps + 1)
        _, _, x = self.modulate(n_frames, n_sym)
        start = self.active(n_sym).start
        seg = x[:, start:start + seg_len]
        return torch.stack([seg.real, seg.imag], dim=1)

    def isi_sum(self):
        """sum_k |(h*h)[delay + k*sps]| over k != 0: worst-case residual ISI, relative to c0."""
        taps = self._casc[self.delay % self.sps::self.sps].abs()
        return float((taps.sum() - taps.max()).item() / self.c0)

    def waveform_length(self, n_sym):
        return n_sym * self.sps + self.filt.length - 1

    def active(self, n_sym):
        """
        The n_sym*sps samples the symbols occupy, without the filter's tapered
        tails. Power ratios (SNR, JSR) are defined over this window: averaging over
        the full frame would dilute a short frame by the tails -- 1.76 dB for
        64 symbols with a 32-symbol RRC (verify job 2259142).
        """
        half = (self.filt.length - 1) // 2
        return slice(half, half + n_sym * self.sps)

    def n_sym_of(self, length):
        return (length - self.filt.length + 1) // self.sps

    # ---------------------------------------------------------------- RX side
    def matched_filter(self, r, n_sym):
        """Matched-filter the received frame and sample at the symbol instants."""
        z = self.filt(r, padding="full")
        return z[..., self.delay:self.delay + n_sym * self.sps:self.sps]

    @staticmethod
    def decide(z):
        """Per-axis sign test -> bits [F, 2N], Sionna's bit order (b0 = I, b1 = Q)."""
        b0 = (z.real < 0).to(torch.float32)
        b1 = (z.imag < 0).to(torch.float32)
        return torch.stack([b0, b1], dim=-1).flatten(-2)

    # ---------------------------------------------------------------- the link
    def noise_var(self, snr_db):
        return self.p_s / 10.0 ** (snr_db / 10.0)

    def run(self, n_frames, n_sym, snr_db=SNR_DB, jammer=None):
        """
        One batch through the link. `jammer` is None or a callable
        (link, n_frames, n_sym) -> j [F, waveform_length(n_sym)], already scaled to
        its JSR (see jammers.py). Returns (bit_errors, n_bits) as Python ints, so
        callers can accumulate until an error-count stopping rule is met.
        """
        bits, _, x = self.modulate(n_frames, n_sym)
        r = self.awgn(x, self.noise_var(snr_db))
        if jammer is not None:
            r = r + jammer(self, n_frames, n_sym)
        bits_hat = self.decide(self.matched_filter(r, n_sym))
        return int((bits != bits_hat).sum().item()), bits.numel()

    # ---------------------------------------------------------------- closed form
    def kappa(self, band):
        """Decision-point variance gain of a unit-variance Gaussian jammer."""
        if band == "full":
            return 1.0      # white noise through a unit-energy filter keeps its variance
        if band == "inband":
            return float((self._casc.abs() ** 2).sum().item())
        raise ValueError(f"unknown band {band!r}")

    def ber_noise_jammer(self, jsr_db, snr_db=SNR_DB, band="full"):
        """Closed-form BER under a Gaussian jammer (module docstring)."""
        jsr = 10.0 ** (np.asarray(jsr_db, dtype=float) / 10.0)
        var = self.noise_var(snr_db) + jsr * self.p_s * self.kappa(band)
        return q_function(self.c0 / np.sqrt(var))

    def ber_clean(self, snr_db):
        """Closed-form BER with no jammer: Q(sqrt(Es/N0)) with Es/N0 = sps*SNR."""
        return q_function(self.c0 / math.sqrt(self.noise_var(snr_db)))


# -------------------------------------------------------------------- measurement
def measure_ber(L, jammer, snr_db=SNR_DB, n_sym=10_000, min_errors=100, max_bits=2e7,
                samples_per_batch=4e6):
    """
    Monte-Carlo BER with Zhou's frame length (1e4 symbols), extended by an error
    stopping rule: keep adding frames until `min_errors` bit errors or `max_bits`.
    Returns dict(ber, errors, bits, upper) where `upper` is the 95% one-sided
    bound 3/bits used when no error was seen at all.
    """
    frames = max(1, int(samples_per_batch // (n_sym * L.sps)))
    errors = bits = 0
    while errors < min_errors and bits < max_bits:
        e, b = L.run(frames, n_sym, snr_db, jammer)
        errors, bits = errors + e, bits + b
    return dict(ber=errors / bits, errors=errors, bits=bits, upper=3.0 / bits)


# -------------------------------------------------------------------- the paper's curves
def load_paper_fig6(path=None):
    """Digitised Zhou Fig. 6 (C0). Returns (jsr_db array, {curve: [ber|None]}, flags)."""
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_fig6.json")
    with open(path) as f:
        d = json.load(f)
    return np.array(d["jsr_db"]), d["ber"], d["flags"]


def usable_paper_points(ber, flags, floor=1e-6):
    """Mask of digitised points fit to compare against: present, visible, measurable."""
    return np.array([b is not None and b >= floor and "occluded" not in fl
                     for b, fl in zip(ber, flags)])


def jsr_at_ber(jsr_db, ber, target=1e-3):
    """
    First JSR at which a rising BER curve crosses `target`, by linear
    interpolation in log10(BER) between the bracketing points. None if it never
    crosses. Points given as None or <= 0 are skipped.

    For Monte-Carlo curves pass zero-error points at their upper bound 3/bits:
    a lower BER at the left bracket only moves a log-linear crossing to the
    right, so the result is then a LOWER BOUND on the true crossing JSR.
    """
    pts = [(j, b) for j, b in zip(jsr_db, ber) if b is not None and b > 0]
    lt = math.log10(target)
    for (j0, b0), (j1, b1) in zip(pts, pts[1:]):
        l0, l1 = math.log10(b0), math.log10(b1)
        if (l0 - lt) * (l1 - lt) <= 0 and l1 != l0:
            return j0 + (lt - l0) / (l1 - l0) * (j1 - j0)
    return None
