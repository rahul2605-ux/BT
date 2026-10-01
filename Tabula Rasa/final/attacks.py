# copied from cgan/attacks.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment -- the attacks, and one received frame at the victim.

    name          what reaches R                                     sync
    ---------------------------------------------------------------------------
    none          nothing: the clean floor and each detector's FAR   --
    noise         white complex Gaussian (barrage = Li's barrage)     law is delay-invariant
    pulsed(p)     the victim's own RRC-QPSK waveform with random      async
                  symbols, each symbol ON with probability p at
                  amplitude 1/sqrt(p). p = 1 is Zhou's "optimal"
                  (matched QPSK); p < 1 is Amuru & Buehrer's pulsed
                  family; p = 0.25 is Li's protocol-aware type
    omniscient(e) a genie: -(1+e) * s on a fraction                  synchronised (by
                  delta = min(1, JSR/(1+e)^2) of the victim's         definition)
                  symbols, pulse-shaped on the victim's own grid
    random_push(e) the genie's timing and phase without its data:     synchronised
                  (1+e) * u on the same fraction delta, u a random
                  QPSK point drawn independently of s
    tone          Li's single tone: a constant complex baseband       delay 0, phase uniform
                  value, phase uniform per frame (the supervisor's
                  naive constant-vector baseline)
    pulse_comb    Li's successive pulse: impulses every 64 samples    async
    gan           a generator's waveform (gan_tx)                     async
    onoff(f)      matched QPSK on a random fraction f of frames,      async
                  silent otherwise. verify.py only: the reported
                  on/off baseline is analytic (figures.py)

omniscient(1) flips the attacked symbols, so R receives -s: still i.i.d. uniform QPSK,
i.e. the clean law exactly -- undetectable by ANY test, BER = delta. omniscient(0.1)
pushes each attacked symbol just past both decision boundaries: the cheapest per bit, but
the received power drops, so only a two-sided test sees it (README §3.3d). random_push(0.1)
is the push for an attacker that knows the victim's symbol timing and carrier phase but
neither its symbols nor its fading gain: each attacked axis is pushed against its bit with
probability 1/2, so BER ~ delta/2 and SER ~ 3 delta/4, and the received power RISES
(E|s + (1+e)u|^2 = 1 + (1+e)^2), so a one-sided test sees it.

Pruned from cgan/attacks.py: the shaped control (D2a), the learned-power projection and
the team `rx` hook are gone. Added: `onoff`, the per-frame DC share of the jammer
(`out["dc_share"]`), the Rician gain in place of shadowing in `frames`, and a per-frame
noise variance in `ber_logprob` (the damage term under noise-level uncertainty).
"""

import math

import torch

import channel
from env import N_SYM

PULSED_P = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
OMNI_ETA = [0.1, 1.0]
LI_TYPES = ["barrage", "tone", "pulse_comb", "protocol_aware"]
COMB_SPACING = 64           # samples between impulses (64 lines over the simulated band)
PROTOCOL_AWARE_P = 0.25


def spec_name(spec):
    """'noise', 'pulsed_p0.25', 'omniscient_e1', ... from an attack spec dict."""
    n = spec["name"]
    if n == "pulsed":
        return f"pulsed_p{spec['p']:g}"
    if n in ("omniscient", "random_push"):
        return f"{n}_e{spec['eta']:g}"
    if n == "onoff":
        return f"onoff_f{spec['f']:g}"
    if n == "gan":
        return spec.get("tag", "gan")
    return n


# ---------------------------------------------------------------- transmit side (padded)
def _padded_length(link, n_sym):
    return link.waveform_length(n_sym) + 2 * channel.PAD


def noise_tx(link, n_frames, n_sym):
    n = _padded_length(link, n_sym)
    re = torch.randn(n_frames, n, device=link.device)
    im = torch.randn(n_frames, n, device=link.device)
    return torch.complex(re, im) / math.sqrt(2.0)


def pulsed_tx(link, n_frames, n_sym, p):
    """
    RRC-QPSK with random symbols, each ON with probability p at amplitude 1/sqrt(p)
    (so the mean power does not depend on p). Aligned so that padded sample PAD
    is on the victim's symbol grid before the async delay.
    """
    extra = channel.PAD // link.sps + 1
    n_tot = n_sym + 2 * extra
    bits = link.source([n_frames, 2 * n_tot])
    sym = link.mapper(bits).reshape(n_frames, n_tot)
    if p < 1.0:
        on = (torch.rand(n_frames, n_tot, device=link.device) < p).to(sym.dtype)
        sym = sym * on / math.sqrt(p)
    x = link.filt(link.upsample(sym), padding="full")
    start = extra * link.sps - channel.PAD
    return x[:, start:start + _padded_length(link, n_sym)]


def gan_tx(link, n_frames, n_sym, G, scale, z=None, grad=False, z_dim=400, seg_len=1024):
    """
    A generator jammer's transmitted (padded) waveform, aligned like pulsed_tx so
    that padded sample PAD sits on the victim's symbol grid before the async delay.
    Tile seg_len-sample segments to fill the padded frame. channel.receive applies the
    async delay and phase, so this waveform is 'locked' -- do NOT desync it here.

    grad=False (evaluation): G runs under no_grad. grad=True (training): the graph is
    kept so the loss reaches G's parameters.
    """
    dev = link.device
    n = _padded_length(link, n_sym)
    lead = channel.PAD + link.active(n_sym).start
    c = (-lead) % link.sps
    n_seg = -(-(c + n) // seg_len)                    # ceil
    if z is None:
        z = torch.randn(n_frames * n_seg, z_dim, device=dev)
    labels = torch.zeros(n_frames * n_seg, dtype=torch.long, device=dev)
    if grad:
        seg = G(z, labels) * scale
    else:
        with torch.no_grad():
            seg = G(z, labels) * scale
    stream = torch.complex(seg[:, 0], seg[:, 1]).reshape(n_frames, n_seg * seg_len)
    return stream[:, c:c + n]


def li_tx(link, n_frames, n_sym, kind):
    """Li et al.'s jammer types, single-carrier adaptation (cgan/attacks.py docstring)."""
    n = _padded_length(link, n_sym)
    dev = link.device
    if kind == "barrage":
        return noise_tx(link, n_frames, n_sym)
    if kind == "tone":
        theta = 2 * math.pi * torch.rand(n_frames, 1, device=dev)
        return torch.exp(1j * theta).to(torch.complex64).expand(n_frames, n).clone()
    if kind == "pulse_comb":
        x = torch.zeros(n_frames, n, dtype=torch.complex64, device=dev)
        start = torch.randint(0, COMB_SPACING, (n_frames,), device=dev)
        idx = start[:, None] + COMB_SPACING * torch.arange(n // COMB_SPACING, device=dev)
        theta = 2 * math.pi * torch.rand(n_frames, 1, device=dev)
        x.scatter_(1, idx.clamp(max=n - 1), torch.exp(1j * theta).to(torch.complex64)
                   .expand(-1, idx.shape[1]).contiguous())
        return x
    if kind == "protocol_aware":
        return pulsed_tx(link, n_frames, n_sym, PROTOCOL_AWARE_P)
    raise ValueError(f"unknown Li et al. jammer type {kind!r}")


# ---------------------------------------------------------------- at the receiver
def omniscient_rx(link, sym, jsr_db, eta):
    """
    The genie's received perturbation on the victim's grid (no channel delay,
    phase 0): -(1+eta) * s on exactly floor(delta * N) random symbols per frame.
    """
    F, N = sym.shape
    delta = min(1.0, 10.0 ** (jsr_db / 10.0) / (1.0 + eta) ** 2)
    m = int(math.floor(delta * N))
    rank = torch.rand(F, N, device=link.device).argsort(dim=1).argsort(dim=1)
    mask = (rank < m).to(sym.dtype)
    return link.filt(link.upsample(-(1.0 + eta) * sym * mask), padding="full")


def random_push_rx(link, sym, jsr_db, eta):
    """
    omniscient_rx without the data: (1+eta) * u on exactly floor(delta * N) random symbols
    per frame, u uniform unit-energy QPSK independent of the victim's symbols, and not
    scaled by the victim's fading gain (the attacker does not know it). Same power as the
    genie push at the same JSR.
    """
    F, N = sym.shape
    delta = min(1.0, 10.0 ** (jsr_db / 10.0) / (1.0 + eta) ** 2)
    m = int(math.floor(delta * N))
    rank = torch.rand(F, N, device=link.device).argsort(dim=1).argsort(dim=1)
    mask = (rank < m).to(sym.dtype)
    u = link.mapper(link.source([F, 2 * N])).reshape(F, N)
    return link.filt(link.upsample((1.0 + eta) * u * mask), padding="full")


def jammer_at_rx(link, spec, sym, jsr_db_k):
    """
    Sum of K jammers at R for one attack spec. jsr_db_k: list of K scalars, or a
    [F, K] tensor. The genie and the random push are one perturbation and take a single JSR.
    """
    F, N = sym.shape
    name = spec["name"]
    if name in ("omniscient", "random_push"):
        jsr = jsr_db_k if isinstance(jsr_db_k, (int, float)) else float(jsr_db_k[0])
        rx = omniscient_rx if name == "omniscient" else random_push_rx
        return rx(link, sym, jsr, spec["eta"])
    jsr = torch.as_tensor(jsr_db_k, dtype=torch.float32, device=link.device)
    if jsr.dim() == 1:
        jsr = jsr.expand(F, -1)
    total = None
    for k in range(jsr.shape[1]):
        if name == "noise":
            j = channel.receive(link, noise_tx(link, F, N), jsr[:, k])
        elif name == "pulsed":
            delay, phase = channel.async_draw(link, F)
            j = channel.receive(link, pulsed_tx(link, F, N, spec["p"]), jsr[:, k], delay, phase)
        elif name == "onoff":
            delay, phase = channel.async_draw(link, F)
            j = channel.receive(link, pulsed_tx(link, F, N, 1.0), jsr[:, k], delay, phase)
            j = j * (torch.rand(F, 1, device=link.device) < spec["f"]).to(j.dtype)
        elif name == "gan":
            delay, phase = channel.async_draw(link, F)
            j = channel.receive(link, gan_tx(link, F, N, spec["G"], spec["scale"],
                                             grad=spec.get("grad", False)), jsr[:, k], delay, phase)
        elif name in LI_TYPES:
            delay, phase = channel.async_draw(link, F)
            if name in ("barrage", "tone"):
                delay = torch.zeros_like(delay)
            j = channel.receive(link, li_tx(link, F, N, name), jsr[:, k], delay, phase)
        else:
            raise ValueError(f"unknown attack {name!r}")
        total = j if total is None else total + j
    return total


def dc_share(link, j, n_sym):
    """Per-frame |mean_n j|^2 / mean_n |j|^2 over the active window [F]: 1 for Li's tone."""
    a = j[:, link.active(n_sym)].detach()
    return (a.mean(-1).abs().pow(2) / a.abs().pow(2).mean(-1).clamp_min(1e-30)).double()


def frames(link, n_frames, spec, jsr_db_k, snr_db, n_sym=None, noiseless=False, keep_jammer=False):
    """
    One batch of received frames: the victim's burst of n_sym symbols + AWGN at
    snr_db + the jammers at R. spec None or name 'none' -> clean.
    Returns dict(bits, bits_hat, r [F, frame length], z [F, N] matched-filter samples),
    plus z_nf, the matched-filter samples WITHOUT the AWGN, if noiseless (for the damage
    term), and dc_share [F] when a jammer is present.

    With link.rician_k_db set, the victim's waveform is scaled per frame by
    link.fading_gain and out["gain"] [F, 1] holds it. Jammers keep their JSR against the
    MEAN signal power; the genie is handed the faded symbols, because it knows the
    victim's channel by definition.

    With link.noise_unc_db > 0 each frame's noise variance is noise_var(snr_db) times
    link.noise_scale and out["noise_scale"] [F] holds the factor.

    keep_jammer=True also returns out["j"], the jammer's waveform at R (iq_plots.py).
    """
    n_sym = N_SYM if n_sym is None else n_sym
    bits, sym, x = link.modulate(n_frames, n_sym)
    g = link.fading_gain(n_frames)
    if g is not None:
        x, sym = x * g, sym * g
    u = link.noise_scale(n_frames)
    n0 = link.noise_var(snr_db)
    r = link.awgn(x, n0 if u is None else n0 * u)
    r_nf = x
    j = None
    if spec is not None and spec["name"] != "none":
        j = jammer_at_rx(link, spec, sym, jsr_db_k)
        r, r_nf = r + j, x + j
    z = link.matched_filter(r, n_sym)
    out = dict(bits=bits, bits_hat=link.decide(z), r=r, z=z)
    if j is not None:
        out["dc_share"] = dc_share(link, j, n_sym)
        if keep_jammer:
            out["j"] = j
    if g is not None:
        out["gain"] = g
    if u is not None:
        out["noise_scale"] = u
    if noiseless:
        out["z_nf"] = link.matched_filter(r_nf, n_sym)
    return out


def error_counts(out):
    """(bit errors, bits, symbol errors, symbols) as ints."""
    wrong = (out["bits"] != out["bits_hat"])
    F = wrong.shape[0]
    sym_wrong = wrong.reshape(F, -1, 2).any(dim=-1)
    return (int(wrong.sum()), wrong.numel(), int(sym_wrong.sum()), sym_wrong.numel())


def frame_noise_var(out, n0):
    """The noise variance each frame was drawn with: n0, or n0 * u [F] under noise uncertainty."""
    u = out.get("noise_scale")
    return n0 if u is None else n0 * u.double()


def ber_logprob(out, noise_var):
    """
    Per-bit log P(error) over the AWGN, [F, N, 2] float64: each bit errs with
    Q(margin / sigma) at the noiseless matched-filter sample, sigma^2 = noise_var/2
    per axis (unit-energy pulse). noise_var: a float, or [F] per frame
    (frame_noise_var). Needs frames(..., noiseless=True). Differentiable in z_nf.
    """
    z = out["z_nf"].to(torch.complex128)
    b = out["bits"].reshape(z.shape[0], -1, 2)
    m = torch.stack([torch.where(b[..., 0] == 0, z.real, -z.real),
                     torch.where(b[..., 1] == 0, z.imag, -z.imag)], dim=-1)
    nv = torch.as_tensor(noise_var, dtype=torch.float64, device=z.device)
    if nv.dim() == 1:
        nv = nv.reshape(-1, 1, 1)
    return torch.special.log_ndtr(-m / torch.sqrt(nv / 2.0))


def log_expected_per(out, noise_var):
    """log E[frame error rate] as a scalar TENSOR, exact over the AWGN (cgan/attacks.py)."""
    lp = ber_logprob(out, noise_var).flatten(1)                  # [F, bits]
    return _log_mean_per(lp)


def _log_mean_per(lp):
    """log of the mean over frames of PER_f, from per-bit log error probs lp [F, bits].
    The exact branch is fed a dummy where it is not used: torch.where differentiates
    BOTH branches (job 2274904, 2026-09-27)."""
    s = torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-12)).sum(-1)  # log P(frame survives)
    use_exact = s < -1e-12
    s_safe = torch.where(use_exact, s, torch.full_like(s, -1.0))
    exact = torch.log(-torch.expm1(s_safe))
    first_order = torch.logsumexp(lp, dim=-1)
    log_per = torch.where(use_exact, exact, first_order)
    return torch.logsumexp(log_per, 0) - math.log(log_per.numel())


def log_expected_ber(out, noise_var):
    """log E[BER] as a scalar TENSOR (logsumexp; never underflows). The damage term."""
    lp = ber_logprob(out, noise_var)
    return torch.logsumexp(lp.flatten(), 0) - math.log(lp.numel())


def expected_ber(out, noise_var):
    """Exact E[BER] over the AWGN, given everything else in the frames, as a float."""
    return float(ber_logprob(out, noise_var).exp().mean())
