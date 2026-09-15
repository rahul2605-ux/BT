"""
CGAN track -- the three jammers of Zhou et al. 2025, Fig. 4-6.

    name       what it transmits                                  role in Zhou
    ---------------------------------------------------------------------------
    noise      complex Gaussian, white over the simulated band    "conventional"
               ("full") or shaped like the signal ("inband")      baseline
    optimal    a QPSK waveform with the target's own pulse,       the "theoretical
               random symbols; three synchronisation variants     optimal" reference
    gan        generator output, denormalised and tiled;          the method
               the same three synchronisation variants

Zhou's "optimal" is "synthesized using a modulation system with identical
filtering and modulation parameters as the target signal". The paper says
nothing about timing or carrier phase, and those decide the curve, so all three
readings are implemented and the C2 checkpoint fixed one (async, README §2.10):

    locked        symbol-synchronous and carrier-phase-locked to the victim
    random_phase  symbol-synchronous, one uniform carrier phase per frame
    async         random phase and a uniform integer timing offset in [0, sps)

SYNCHRONISATION IS ONE MODEL FOR BOTH STRUCTURED JAMMERS (`desync`). The GAN is
trained on clean segments cropped on the victim's symbol grid at carrier phase
0, so tiling its output unchanged scores it as a LOCKED jammer. Until
2026-09-15 that is what happened while `optimal` was async -- the GAN-vs-Optimal
gap mixed in a synchronisation assumption. Both now go through `desync`.

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


def spare(link, sync):
    """Extra samples `desync` needs beyond the frame length: sps for async, else none."""
    return link.sps if sync == "async" else 0


def desync(j, link, length, sync):
    """
    Apply a synchronisation model to a structured jammer j [F, >= length + spare].
    The jammer's samples are assumed aligned to the victim's (locked) on input.
      locked        crop to length
      random_phase  crop, then one uniform carrier phase per frame
      async         per-frame integer advance of 0..sps-1 samples, then random phase
    """
    if sync == "async":
        offset = torch.randint(0, link.sps, (j.shape[0], 1), device=link.device)
        j = torch.gather(j, 1, offset + torch.arange(length, device=link.device))
    elif sync in ("locked", "random_phase"):
        j = j[:, :length]
    else:
        raise ValueError(f"unknown sync {sync!r}")
    if sync != "locked":
        theta = 2 * math.pi * torch.rand(j.shape[0], 1, device=link.device)
        j = j * torch.exp(1j * theta).to(j.dtype)
    return j


def optimal(link, n_frames, n_sym, jsr_db, sync="async"):
    length = link.waveform_length(n_sym)
    # async shifts into one spare symbol (exactly `spare` = sps samples)
    _, _, x = link.modulate(n_frames, n_sym + (1 if sync == "async" else 0))
    return scale_to_jsr(desync(x, link, length, sync), link, jsr_db)


def gan(link, n_frames, n_sym, jsr_db, G=None, scale=1.0, z_dim=400, seg_len=1024, sync="async"):
    """
    Generator jammer: draw (n_frames, 2, seg_len) segments, undo the peak
    normalisation, read as complex I+jQ, tile, then apply the same `desync` as
    `optimal`. The generator is evaluated with no grad (frozen at evaluation).

    G learned segments cropped at link.active().start (Link.real_segments), so a
    tile is aligned with the victim when it starts at an index congruent to that
    start modulo sps; `lead` drops the first samples to put it there. For the
    RRC links used here the start is a multiple of sps and lead = 0.
    """
    length = link.waveform_length(n_sym)
    lead = (-link.active(n_sym).start) % link.sps
    n_seg = -(-(lead + length + spare(link, sync)) // seg_len)     # ceil
    labels = torch.zeros(n_frames * n_seg, dtype=torch.long, device=link.device)
    z = torch.randn(n_frames * n_seg, z_dim, device=link.device)
    with torch.no_grad():
        seg = G(z, labels) * scale                      # (F*n_seg, 2, seg_len)
    j = torch.complex(seg[:, 0], seg[:, 1]).reshape(n_frames, n_seg * seg_len)[:, lead:]
    return scale_to_jsr(desync(j, link, length, sync), link, jsr_db)


def make(name, jsr_db, **kw):
    """Bind a jammer to its JSR (and variant) -> callable(link, n_frames, n_sym)."""
    fn = {"noise": noise, "optimal": optimal, "gan": gan}[name]
    return lambda link, n_frames, n_sym: fn(link, n_frames, n_sym, jsr_db, **kw)
