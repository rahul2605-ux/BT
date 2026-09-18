"""
Baselines -- the jammer -> receiver channel, applied by Sionna.

Each jammer's waveform reaches the victim receiver R through a one-path channel
built by `sionna.phy.channel.cir_to_time_channel` (sinc-interpolated taps, so a
delay may fall between samples) and applied by `ApplyTimeChannel`. The path
carries:

  * the jammer's asynchronous clock offset, uniform in [0, sps) samples, and
    its carrier phase, uniform in [0, 2 pi), both per frame (scene.py: the
    geometric delay and phase are absorbed into these and not applied);
  * its received power, imposed exactly on the frame by `jammers.scale_to_jsr`
    over the victim's active window -- the hard, equality-form power
    constraint the step-1 jammers already use.

Jammers go through the channel one at a time, so each is scaled to its own JSR
before the contributions are summed at R. White noise has a delay-invariant law;
it takes the same path with delay 0 (the sinc taps are then an exact delta).

Waveforms enter PADDED: sample PAD of the input is the victim's sample 0, and
PAD samples of margin on each side cover the delay plus the sinc taps' reach.

DTYPE: torch.complex64 is float32 real + imaginary (Sionna's single precision);
torch.complex32 is HALF precision and must not be used.
"""

import math

import torch
from sionna.phy.channel import ApplyTimeChannel, cir_to_time_channel

import jammers
from scene import SYMBOL_RATE

PAD = 48            # input margin each side [samples]
L_MIN, L_MAX = -32, 40  # sinc tap range: delays up to L_MAX - 32 samples keep +-32 taps


def async_draw(link, n_frames):
    """Per-frame asynchronous timing offset [samples, float] and carrier phase [rad]."""
    delay = torch.rand(n_frames, device=link.device) * link.sps
    phase = 2 * math.pi * torch.rand(n_frames, device=link.device)
    return delay, phase


def taps(link, delay, phase, l_min=L_MIN, l_max=L_MAX):
    """
    Sionna time-channel taps of a single path with unit gain, the given delay
    [samples] and phase [rad]: [F, 1, 1, 1, 1, 1, l_tot].
    """
    F = delay.shape[0]
    fs = link.sps * SYMBOL_RATE
    a = torch.exp(1j * phase.double()).to(torch.complex64).reshape(F, 1, 1, 1, 1, 1, 1)
    tau = (delay.double() / fs).to(torch.float32).reshape(F, 1, 1, 1)          # [s]
    return cir_to_time_channel(fs, a, tau, l_min, l_max)


def receive(link, x_padded, jsr_db, delay=None, phase=None, l_min=L_MIN, l_max=L_MAX):
    """
    One jammer at R: x_padded [F, length + 2*PAD] -> [F, length], delayed and
    phase-rotated by Sionna, then scaled to jsr_db (scalar or [F]) over the
    victim's active window. delay/phase default to 0 (a synchronised arrival).
    """
    F, n_in = x_padded.shape
    length = n_in - 2 * PAD
    dev = link.device
    delay = torch.zeros(F, device=dev) if delay is None else delay
    phase = torch.zeros(F, device=dev) if phase is None else phase
    h = taps(link, delay, phase, l_min, l_max)
    apply = _apply(n_in, l_max - l_min + 1, dev)
    y = apply(x_padded.to(torch.complex64).reshape(F, 1, 1, n_in), h)[:, 0, 0]
    # output index b holds input index b + l_min (before the path delay)
    start = PAD - l_min
    y = y[:, start:start + length]
    jsr = torch.as_tensor(jsr_db, dtype=torch.float32, device=dev).reshape(-1, 1)
    return _scale(y, link, jsr)


_APPLY = {}


def _apply(n_in, l_tot, device):
    """ApplyTimeChannel builds an index matrix on construction: build once per shape."""
    key = (n_in, l_tot, str(device))
    if key not in _APPLY:
        _APPLY[key] = ApplyTimeChannel(num_time_samples=n_in, l_tot=l_tot, device=device)
    return _APPLY[key]


def _scale(j, link, jsr_db):
    """jammers.scale_to_jsr with a per-frame JSR [F, 1] (or scalar)."""
    if jsr_db.numel() == 1:
        return jammers.scale_to_jsr(j, link, float(jsr_db))
    unit = jammers.scale_to_jsr(j, link, 0.0)
    return unit * torch.sqrt(10.0 ** (jsr_db / 10.0))
