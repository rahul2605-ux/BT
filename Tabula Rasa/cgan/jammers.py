"""
CGAN track -- the three jammers of Zhou et al. 2025, Fig. 4-6.

    name       what it transmits                                  role in Zhou
    ---------------------------------------------------------------------------
    noise      complex Gaussian, white over the simulated band    "conventional"
               ("full") or shaped like the signal ("inband")      baseline
    optimal    a QPSK waveform with the target's own pulse,       the "theoretical
               random symbols; three synchronisation variants     optimal" reference
    gan        generator output, denormalised and tiled           the method

Zhou's "optimal" is "synthesized using a modulation system with identical
filtering and modulation parameters as the target signal". The paper says
nothing about timing or carrier phase, and those decide the curve, so all three
readings are implemented and the C2 checkpoint fixed one (async, README §2.10):

    locked        symbol-synchronous and carrier-phase-locked to the victim
    random_phase  symbol-synchronous, one uniform carrier phase per frame
    async         random phase and a uniform integer timing offset in [0, sps)

This is NOT the BER-maximising jammer under a power constraint (Amuru & Buehrer
2015 -- generally pulsed at low JSR); README §4.2 Q9.

JSR IS AN EQUALITY HERE
-----------------------
Zhou sweeps the jammer AT a JSR, so every frame is scaled to exactly
JSR * P_s mean power per sample over the victim's active window (`scale_to_jsr`).
It is still a hard constraint applied to the transmitted waveform, never a loss
term. m0/attacks.py `project_power` is the inequality (<= budget) form. That is
the right contract once a learned jammer may choose to spend less (step 2), and
it is not what Zhou's figures measure.

Every jammer has the signature  jammer(link, n_frames, n_sym) -> j  with
j of shape [F, link.waveform_length(n_sym)], so link.run() can take any of them.
`make(name, jsr_db, **kw)` binds the parameters.
"""

import math

import torch

JAMMERS = ["noise", "optimal", "gan"]
SYNC_VARIANTS = ["locked", "random_phase", "async"]
BANDS = ["full", "inband"]


def scale_to_jsr(j, link, jsr_db):
    """
    Scale each frame to mean per-sample power JSR * P_s (hard, exact), measured
    over the window the victim's symbols occupy (link.active), not the filter tails.
    """
    target = link.p_s * 10.0 ** (jsr_db / 10.0)
    a = link.active(link.n_sym_of(j.shape[-1]))
    p = j[..., a].abs().pow(2).mean(dim=-1, keepdim=True)
    return j * torch.sqrt(target / (p + 1e-30))


def noise(link, n_frames, n_sym, jsr_db, band="full"):
    length = link.waveform_length(n_sym)
    re = torch.randn(n_frames, length, device=link.device)
    im = torch.randn(n_frames, length, device=link.device)
    j = torch.complex(re, im) / math.sqrt(2.0)
    if band == "inband":
        j = link.filt(j, padding="same")
    elif band != "full":
        raise ValueError(f"unknown band {band!r}")
    return scale_to_jsr(j, link, jsr_db)


def optimal(link, n_frames, n_sym, jsr_db, sync="async"):
    length = link.waveform_length(n_sym)
    if sync == "async":
        # One spare symbol, then a per-frame integer advance of 0..sps-1 samples.
        _, _, x = link.modulate(n_frames, n_sym + 1)
        offset = torch.randint(0, link.sps, (n_frames, 1), device=link.device)
        idx = offset + torch.arange(length, device=link.device)
        j = torch.gather(x, 1, idx)
    else:
        _, _, j = link.modulate(n_frames, n_sym)
    if sync in ("random_phase", "async"):
        theta = 2 * math.pi * torch.rand(n_frames, 1, device=link.device)
        j = j * torch.exp(1j * theta).to(j.dtype)
    elif sync != "locked":
        raise ValueError(f"unknown sync {sync!r}")
    return scale_to_jsr(j, link, jsr_db)


def gan(link, n_frames, n_sym, jsr_db, G=None, scale=1.0, z_dim=400, seg_len=1024):
    """
    Generator jammer: draw (n_frames, 2, seg_len) segments, undo the peak
    normalisation, read as complex I+jQ, and tile/crop to the frame length. The
    generator is evaluated with no grad (it is frozen at evaluation time).
    """
    length = link.waveform_length(n_sym)
    n_seg = -(-length // seg_len)                       # ceil
    labels = torch.zeros(n_frames * n_seg, dtype=torch.long, device=link.device)
    z = torch.randn(n_frames * n_seg, z_dim, device=link.device)
    with torch.no_grad():
        seg = G(z, labels) * scale                      # (F*n_seg, 2, seg_len)
    j = torch.complex(seg[:, 0], seg[:, 1]).reshape(n_frames, n_seg * seg_len)[:, :length]
    return scale_to_jsr(j, link, jsr_db)


def make(name, jsr_db, **kw):
    """Bind a jammer to its JSR (and variant) -> callable(link, n_frames, n_sym)."""
    fn = {"noise": noise, "optimal": optimal, "gan": gan}[name]
    return lambda link, n_frames, n_sym: fn(link, n_frames, n_sym, jsr_db, **kw)
