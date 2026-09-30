# copied from cgan/link.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment -- the waveform-level QPSK link, built from Sionna.

Pruned from cgan/link.py: the Zhou-calibration helpers (measure_ber, the Fig. 6 loaders,
jsr_at_ber, Link.run, real_segments) are gone; S1's log-normal `shadow_gain` is replaced by
`fading_gain`, a per-frame Rician amplitude (README §3.4 decision 4); `scale_to_jsr` moved
here from cgan/jammers.py, which is not copied. Everything else is unchanged.

SIGNAL MODEL
------------
Per frame of N symbols s[k], iid uniform Gray QPSK, unit energy:

    x[n] = g * sum_k s[k] h[n - k*sps]                   transmitted waveform, faded
    r[n] = x[n] + w[n] + j[n]                            w ~ CN(0, N0 u), j = jammer
    z[k] = (r * h)[delay + k*sps]                        matched filter, sampled
    bits = per-axis sign of z[k]                         ML for QPSK in Gaussian noise

`h` is unit-energy (Sionna `normalize=True`), so the cascade h*h peaks at c0 = 1 and is
Nyquist: z[k] = g s[k] + (filtered noise and jammer), no ISI up to filter truncation.
g = 1 and u = 1 on the ideal link; `Link.rician_k_db` draws g per frame (Rician
amplitude, unit mean power, real and positive: the receiver knows its phase, so there is
no equaliser), `Link.noise_unc_db` draws u per frame (log-normal, unit mean, S5 §3.3n).
Both only on the measurement path (attacks.frames).

CONVENTIONS (README §4.2 Q7)
----------------------------
* Gray labelling from Sionna's "qam" constellation: bit0 -> sign of I, bit1 -> sign of Q.
* Average transmitted power per sample is P_s = ||h||^2 / sps = 1/sps.
* SNR and JSR are ratios of MEAN POWER PER SAMPLE at the receiver input, over the full
  simulated bandwidth (sps x symbol rate):  N0 = P_s / SNR,  E|j[n]|^2 = JSR * P_s.
  So Es/N0 = sps * SNR: +9.03 dB at sps 8 (env.ESN0_OFFSET_DB).
* Pulse "rrc0.35", span RRC_SPAN symbols.
* Decisions are hard sign tests on z[k] directly (no Demapper: it needs a noise variance).

CLOSED FORM FOR A GAUSSIAN JAMMER
---------------------------------
A complex Gaussian jammer with per-sample variance JSR*P_s, white over the band, reaches
the decision with variance JSR*P_s (kappa = 1), so per axis

    BER = Q( c0 / sqrt(N0 + JSR*P_s) ).
"""

import math

import numpy as np
import torch

import sionna.phy as sn
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.signal import CustomFilter, RootRaisedCosineFilter, Upsampling
from sionna.phy.channel import AWGN

# The link cgan/ step 1 decided (README §2.10): sps 8, RRC roll-off 0.35.
LINK = dict(sps=8, pulse="rrc0.35")
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


def scale_to_jsr(j, link, jsr_db):
    """
    Scale each frame to mean per-sample power JSR * P_s (hard, exact), measured over the
    window the victim's symbols occupy (link.active), not the filter tails.
    """
    target = link.p_s * 10.0 ** (jsr_db / 10.0)
    a = link.active(link.n_sym_of(j.shape[-1]))
    p = j[..., a].abs().pow(2).mean(dim=-1, keepdim=True)
    return j * torch.sqrt(target / (p + 1e-30))


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
        self.rician_k_db = None     # Rician K of the victim's per-frame amplitude [dB]; None = no fading
        self.noise_unc_db = 0.0     # std of the per-frame noise-variance factor [dB] (§3.3n)
        self.jammer_sync = False    # read by channel.async_draw; always asynchronous here (§3.3l)

        # Cascade response measured through the actual Sionna blocks.
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

    def fading_gain(self, n_frames):
        """
        Per-frame amplitude gain [F, 1] of the victim's link: |h| with h Rician,
        h = sqrt(K/(K+1)) + sqrt(1/(K+1)) CN(0, 1), so E|h|^2 = 1 and SNR, JSR keep their
        meaning on average. Amplitude only (real, positive): the receiver knows its phase
        (System Model), so sign decisions need no equaliser. Frame power std is
        sqrt(1 + 2K)/(K + 1): 1.5 dB at K = 12 dB, 0.24 dB at 28 dB. None when rician_k_db
        is None -- nothing is drawn, so the ideal link stays bit-identical.
        """
        if self.rician_k_db is None:
            return None
        k = 10.0 ** (self.rician_k_db / 10.0)
        re = torch.randn(n_frames, 1, device=self.device)
        im = torch.randn(n_frames, 1, device=self.device)
        s = math.sqrt(1.0 / (2.0 * (k + 1.0)))           # per-axis std of the scattered part
        return torch.sqrt((math.sqrt(k / (k + 1.0)) + s * re) ** 2 + (s * im) ** 2)

    def noise_scale(self, n_frames):
        """
        Per-frame factor [F] on the noise VARIANCE: log-normal with std noise_unc_db in dB,
        normalised to unit mean so the SNR keeps its meaning on average (README §3.3n, S5).
        The receiver does not know it. None when noise_unc_db == 0 -- nothing is drawn.
        """
        if self.noise_unc_db <= 0:
            return None
        s = self.noise_unc_db * math.log(10.0) / 10.0       # std of ln(variance factor)
        return torch.exp(s * torch.randn(n_frames, device=self.device) - s * s / 2)

    def waveform_length(self, n_sym):
        return n_sym * self.sps + self.filt.length - 1

    def active(self, n_sym):
        """
        The n_sym*sps samples the symbols occupy, without the filter's tapered tails.
        Power ratios (SNR, JSR) are defined over this window.
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

    # ---------------------------------------------------------------- closed form
    def noise_var(self, snr_db):
        return self.p_s / 10.0 ** (snr_db / 10.0)

    def ber_noise_jammer(self, jsr_db, snr_db):
        """Closed-form BER under a white Gaussian jammer (module docstring)."""
        jsr = 10.0 ** (np.asarray(jsr_db, dtype=float) / 10.0)
        return q_function(self.c0 / np.sqrt(self.noise_var(snr_db) + jsr * self.p_s))

    def ber_clean(self, snr_db):
        """Closed-form BER with no jammer: Q(sqrt(Es/N0)) with Es/N0 = sps*SNR."""
        return q_function(self.c0 / math.sqrt(self.noise_var(snr_db)))
