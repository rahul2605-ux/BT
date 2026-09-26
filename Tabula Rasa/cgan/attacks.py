"""
Baselines -- the attacks, and one received frame at the victim.

    name          what reaches R                                     sync
    ---------------------------------------------------------------------------
    none          nothing: the clean floor and each detector's FAR   --
    noise         white complex Gaussian (barrage)                    law is delay-invariant
    pulsed(p)     the victim's own RRC-QPSK waveform with random      async
                  symbols, each symbol ON with probability p at
                  amplitude 1/sqrt(p). p = 1 is Zhou's "optimal"
                  jammer; p < 1 is Amuru & Buehrer's pulsed family
                  (swept, no closed-form p is trusted)
    omniscient(e) a genie: -(1+e) * s on a fraction                  synchronised (by
                  delta = min(1, JSR/(1+e)^2) of the victim's         definition)
                  symbols, pulse-shaped on the victim's own grid

    shaped(theta) D2a, the learned control tier (README §3.4): white   async
                  Gaussian noise shaped by 32 log-PSD gains over the
                  simulated band and a periodic 16-symbol envelope of
                  log-power gains (own symbol clock, random offset per
                  frame). theta = 0 is the barrage jammer. Optimised
                  black-box by train_shaped.py.

omniscient(1) flips the attacked symbols, so R receives -s: still i.i.d.
uniform QPSK, i.e. the clean law exactly -- undetectable by ANY test, BER =
delta. omniscient(0.1) pushes each attacked symbol just past both decision
boundaries: the cheapest per bit, but the received power drops. It is the
mandated "impossible to beat" reference and needs what no causal jammer has
(the current symbol, and a signal arriving through T->J->R no later than T->R).
It spends at most its budget: exactly floor(delta * N) symbols per frame.

Every realisable jammer is asynchronous (channel.py). For K > 1 each jammer
draws its own offset, phase and symbols, and is scaled to its own JSR.

The four jammer types of Li et al. (IEEE Access 2022, Sec. III-A) are here too,
adapted to this single-carrier link, because the spectrogram CNN is retrained on
them (README §2.10): barrage = noise; single tone at the carrier = a constant
complex exponential at baseband; successive pulse = impulses every 64 samples,
a comb of 64 lines over the simulated band (Li: Nj = 64 tones across theirs);
protocol-aware ("simulates the transmitter of the targeted protocol" with
low-power shot-noise pulses) = pulsed(0.25).
"""

import math

import torch

import channel

ATTACKS = ["noise", "pulsed", "omniscient"]
PULSED_P = [1.0, 0.5, 0.25, 0.1]
OMNI_ETA = [0.1, 1.0]
LI_TYPES = ["barrage", "tone", "pulse_comb", "protocol_aware"]
SHAPED_BINS = 32            # log-PSD gains, evenly over the simulated band [-fs/2, fs/2)
SHAPED_PERIOD = 16          # envelope period [symbols of the jammer's own clock]
SHAPED_DIM = SHAPED_BINS + SHAPED_PERIOD
COMB_SPACING = 64           # samples between impulses (64 lines over the simulated band)
PROTOCOL_AWARE_P = 0.25


def spec_name(spec):
    """'noise', 'pulsed_p0.25', 'omniscient_e1', ... from an attack spec dict."""
    n = spec["name"]
    if n == "pulsed":
        return f"pulsed_p{spec['p']:g}"
    if n == "omniscient":
        return f"omniscient_e{spec['eta']:g}"
    if n == "shaped":
        return spec.get("tag", "shaped")
    if n == "gan":
        return spec.get("tag", "gan")
    return n


def all_specs():
    return ([dict(name="noise")] + [dict(name="pulsed", p=p) for p in PULSED_P]
            + [dict(name="omniscient", eta=e) for e in OMNI_ETA])


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


def shaped_psd_db(theta, n):
    """The shaped jammer's power gain [dB] on an n-point FFT grid (fftfreq order)."""
    theta = torch.as_tensor(theta, dtype=torch.float64)
    g = theta[:SHAPED_BINS]
    centres = (torch.arange(SHAPED_BINS, dtype=torch.float64) + 0.5) / SHAPED_BINS - 0.5
    f = torch.fft.fftfreq(n, dtype=torch.float64)
    # piecewise-linear in log power between bin centres, flat beyond the outermost two
    x = ((f - centres[0]) * SHAPED_BINS).clamp(0, SHAPED_BINS - 1)
    i0 = x.floor().long().clamp(max=SHAPED_BINS - 2)
    w = x - i0
    return 10.0 / math.log(10.0) * ((1 - w) * g[i0] + w * g[i0 + 1])


def shaped_tx(link, n_frames, n_sym, theta):
    """
    White complex Gaussian -> FFT -> x sqrt(PSD gain) -> IFFT (circular: the
    input is stationary, so there is no edge effect), then x a periodic per-symbol
    amplitude sqrt(exp(theta_env)) with a random circular offset per frame --
    the jammer does not know where the victim's frame starts. The absolute level
    is irrelevant: channel.receive scales every frame to its JSR.
    """
    dev = link.device
    theta = torch.as_tensor(theta, dtype=torch.float64)
    w = noise_tx(link, n_frames, n_sym)
    n = w.shape[-1]
    gain = torch.pow(10.0, shaped_psd_db(theta, n) / 20.0).to(dev)
    x = torch.fft.ifft(torch.fft.fft(w.to(torch.complex128), dim=-1) * gain, dim=-1)
    env = torch.exp(0.5 * theta[SHAPED_BINS:]).to(dev)
    offset = torch.randint(0, SHAPED_PERIOD, (n_frames, 1), device=dev)
    slot = (torch.arange(n, device=dev) // link.sps + offset) % SHAPED_PERIOD
    return (x * env[slot]).to(torch.complex64)


def gan_tx(link, n_frames, n_sym, G, scale, z=None, grad=False, z_dim=400, seg_len=1024):
    """
    A generator jammer's transmitted (padded) waveform, aligned like pulsed_tx so
    that padded sample PAD sits on the victim's symbol grid before the async delay.

    G learned segments cropped at link.active().start (Link.real_segments), so its
    sample 0 is on the victim's grid. Tile seg_len-sample segments to fill the
    padded frame; crop so a grid instant lands at padded index `lead = PAD +
    active.start`. For the RRC link `lead` is a multiple of sps (c = 0); the crop
    offset c handles any other pulse. channel.receive applies the async delay and
    phase, so this waveform is 'locked' -- do NOT desync it here.

    grad=False (evaluation): G runs under no_grad. grad=True (D2 training): the
    graph is kept so the loss reaches G's parameters.
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
    """Li et al.'s jammer types, single-carrier adaptation (module docstring)."""
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


def jammer_at_rx(link, spec, sym, jsr_db_k):
    """
    Sum of K jammers at R for one attack spec. jsr_db_k: list of K scalars, or a
    [F, K] tensor. The omniscient team is one aggregate perturbation (scene.py),
    so it takes a single JSR.
    """
    F, N = sym.shape
    if "rx" in spec:            # a spec may bring its own receive model (team_fading.py)
        return spec["rx"](link, sym, jsr_db_k)
    name = spec["name"]
    if name == "omniscient":
        jsr = jsr_db_k if isinstance(jsr_db_k, (int, float)) else float(jsr_db_k[0])
        return omniscient_rx(link, sym, jsr, spec["eta"])
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
        elif name == "shaped":
            delay, phase = channel.async_draw(link, F)
            j = channel.receive(link, shaped_tx(link, F, N, spec["theta"]), jsr[:, k], delay, phase)
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


def frames(link, n_frames, spec, jsr_db_k, snr_db, n_sym=None, noiseless=False):
    """
    One batch of received frames: the victim's burst of n_sym symbols + AWGN at
    snr_db + the jammers at R. spec None or name 'none' -> clean.
    Returns dict(bits, bits_hat, r [F, frame length], z [F, N] matched-filter samples),
    plus z_nf, the matched-filter samples WITHOUT the AWGN, if noiseless (for
    expected_ber). The random draws are the same either way.
    """
    from scene import N_SYM
    n_sym = N_SYM if n_sym is None else n_sym
    bits, sym, x = link.modulate(n_frames, n_sym)
    r = link.awgn(x, link.noise_var(snr_db))
    r_nf = x
    if spec is not None and spec["name"] != "none":
        j = jammer_at_rx(link, spec, sym, jsr_db_k)
        r, r_nf = r + j, x + j
    z = link.matched_filter(r, n_sym)
    out = dict(bits=bits, bits_hat=link.decide(z), r=r, z=z)
    if noiseless:
        out["z_nf"] = link.matched_filter(r_nf, n_sym)
    return out


def error_counts(out):
    """(bit errors, bits, symbol errors, symbols) as ints."""
    wrong = (out["bits"] != out["bits_hat"])
    F = wrong.shape[0]
    sym_wrong = wrong.reshape(F, -1, 2).any(dim=-1)
    return (int(wrong.sum()), wrong.numel(), int(sym_wrong.sum()), sym_wrong.numel())


def ber_logprob(out, noise_var):
    """
    Per-bit log P(error) over the AWGN, [F, N, 2] float64: each bit errs with
    Q(margin / sigma) at the noiseless matched-filter sample, sigma^2 = noise_var/2
    per axis (unit-energy pulse). Needs frames(..., noiseless=True). Differentiable
    in z_nf, hence in a generator that produced the jammer -- the tensor D2 (and the
    log-domain expected_ber) are both built from this.
    """
    z = out["z_nf"].to(torch.complex128)
    b = out["bits"].reshape(z.shape[0], -1, 2)
    m = torch.stack([torch.where(b[..., 0] == 0, z.real, -z.real),
                     torch.where(b[..., 1] == 0, z.imag, -z.imag)], dim=-1)
    return torch.special.log_ndtr(-m / math.sqrt(noise_var / 2.0))


def log_expected_ber(out, noise_var):
    """log E[BER] as a scalar TENSOR (logsumexp; never underflows). D2's effectiveness term."""
    lp = ber_logprob(out, noise_var)
    return torch.logsumexp(lp.flatten(), 0) - math.log(lp.numel())


def expected_ber(out, noise_var, per_frame=False, log=False):
    """
    Exact E[BER] over the AWGN, given everything else in the frames (bits, jammer).
    Float64 via log_ndtr; it underflows to 0 only where the Monte-Carlo BER is 0
    anyway. A float, or [F] if per_frame. log=True returns log E[BER] as a float
    (logsumexp; never underflows, same ordering); use log_expected_ber for the
    grad-carrying tensor.
    """
    lp = ber_logprob(out, noise_var)
    if log:
        return float(torch.logsumexp(lp.flatten(), 0) - math.log(lp.numel()))
    p = lp.exp()
    return p.flatten(1).mean(-1) if per_frame else float(p.mean())
