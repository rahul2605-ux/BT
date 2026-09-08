"""
M0 — the minimal model: physical layer, metrics.

One QPSK link, one channel coefficient, AWGN, K interferers. No OFDM grid, no
IDFT, no guard/DC/pilot bins, no fading. This deletes exactly the machinery that
produced and then invalidated earlier results (out-of-band leakage, the
real-vs-complex STFT bug, the ambiguous channel-aware criterion) and leaves a
2-D constellation -- the picture the whole problem is actually about.

SIGNAL MODEL
------------
Per frame of N symbols, with s[n] drawn iid uniform from the QPSK alphabet:

    y[n] = h0*s[n] + sum_k g_k * x_k[n] + w[n],      w ~ CN(0, 2*sigma^2)

`y` is the COMPOSITE received frame, before equalisation. It is what the
detector sees. Equalising by h0 gives what the victim decides on:

    yhat[n] = y[n]/h0 = s[n] + d[n] + (sigma/|h0|)*wtilde[n]
    d[n]    = sum_k (g_k/h0) * x_k[n]

The split matters and is the source of the whole problem's tension: the victim
is harmed by `d` (equalised domain), the detector observes `y` (composite
domain). An attacker wanting effect `d` must transmit x_k = d*h0/g_k, so the
detector sees that perturbation scaled by g_k, not by g_k/h0.

CONVENTIONS
-----------
* Unit symbol energy: E|s|^2 = 1, constellation {(+-1 +- 1j)/sqrt(2)}.
* sigma is the noise std per REAL dimension, so N0 = 2*sigma^2.
* Gray mapping: bit0 -> sign of I, bit1 -> sign of Q, with bit 0 -> positive.
  Under this map a perturbation along -I flips bit0 only, which is what makes
  the minimum-energy boundary attack a one-bit-per-symbol operation.
* Everything is torch, real dtype float32 / complex64, so the whole chain stays
  differentiable for the generator experiments later.

SNR RELATIONS (used for cross-referencing the sim08 appendix results)
    Es = 1, N0 = 2*sigma^2
    Es/N0 = 1/(2*sigma^2);  QPSK carries 2 bits, so Eb/N0 = 1/(4*sigma^2)
    sigma = 0.5 -> Eb/N0 =  0.0 dB
    sigma = 0.1 -> Eb/N0 = 14.0 dB
    sigma -> 0  -> Eb/N0 -> inf
So the planned sweep sigma in [0, 0.5] overlaps sim08's 5-30 dB from below.
"""

import math
import torch

# QPSK alphabet, unit energy.
INV_SQRT2 = 1.0 / math.sqrt(2.0)


def random_bits(n_frames, n_sym, device="cpu", generator=None):
    """Uniform iid payload bits, shape (n_frames, n_sym, 2)."""
    return torch.randint(0, 2, (n_frames, n_sym, 2), device=device,
                         generator=generator, dtype=torch.int64)


def bits_to_symbols(bits):
    """Gray QPSK map. bit==0 -> +1/sqrt(2) on that axis, bit==1 -> -1/sqrt(2)."""
    i = (1.0 - 2.0 * bits[..., 0].to(torch.float32)) * INV_SQRT2
    q = (1.0 - 2.0 * bits[..., 1].to(torch.float32)) * INV_SQRT2
    return torch.complex(i, q)


def symbols_to_bits(sym):
    """Maximum-likelihood QPSK decision: per-axis sign test."""
    b0 = (sym.real < 0).to(torch.int64)
    b1 = (sym.imag < 0).to(torch.int64)
    return torch.stack([b0, b1], dim=-1)


def awgn(shape, sigma, device="cpu", generator=None):
    """CN(0, 2*sigma^2): independent N(0, sigma^2) on each real dimension."""
    re = torch.randn(shape, device=device, generator=generator) * sigma
    im = torch.randn(shape, device=device, generator=generator) * sigma
    return torch.complex(re, im)


def receive(sym, d, sigma, h0=1.0 + 0j, g=None, x=None, device="cpu",
            generator=None):
    """
    Build the composite and equalised frames.

    Pass EITHER `d` (the aggregate perturbation already referred to the
    equalised domain) or `x` together with `g` (per-jammer transmitted signals
    and link gains). `d` is the convenient form for closed-form attacks; `x`/`g`
    is the honest form once several jammers with distinct links are present.

    Returns (y, yhat) with y the composite pre-equalisation frame.
    """
    h0 = torch.as_tensor(h0, dtype=torch.complex64, device=device)

    if d is None:
        # x: (..., K, n_sym), g: (K,) -> aggregate in the equalised domain
        g = torch.as_tensor(g, dtype=torch.complex64, device=device)
        d = (x * g.reshape(*([1] * (x.dim() - 2)), -1, 1)).sum(dim=-2) / h0

    w = awgn(sym.shape, sigma, device=device, generator=generator)
    y = h0 * (sym + d) + w          # composite: what the detector sees
    yhat = y / h0                   # equalised: what the victim decides on
    return y, yhat


def ber_ser(bits_tx, bits_rx):
    """Bit error rate and symbol error rate (a symbol errs if either bit does)."""
    bit_err = (bits_tx != bits_rx)
    ber = bit_err.to(torch.float32).mean().item()
    ser = bit_err.any(dim=-1).to(torch.float32).mean().item()
    return ber, ser


def theoretical_ber(sigma):
    """
    Analytic QPSK-over-AWGN bit error rate, for verification.

    Per real dimension the signal amplitude is 1/sqrt(2) and the noise std is
    sigma, so P(bit error) = Q( (1/sqrt(2)) / sigma ) = Q( 1/(sigma*sqrt(2)) ).
    Equivalently Q(sqrt(2*Eb/N0)) with Eb/N0 = 1/(4*sigma^2).
    """
    if sigma <= 0:
        return 0.0
    # Q(z) = 0.5*erfc(z/sqrt(2))
    z = 1.0 / (sigma * math.sqrt(2.0))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ebn0_db(sigma):
    """Eb/N0 in dB for unit-energy QPSK with noise std sigma per real dim."""
    if sigma <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / (4.0 * sigma ** 2))


def sigma_for_ebn0_db(db):
    """Inverse of ebn0_db -- lets the sweep be quoted in dB against sim08."""
    return math.sqrt(1.0 / (4.0 * 10.0 ** (db / 10.0)))
