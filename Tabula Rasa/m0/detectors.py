"""
M0 -- the defender suite.

All three detectors observe the COMPOSITE pre-equalisation frame y of link.py
and nothing else: no channel state, no side information about the jammers, and
no ground-truth labels. Each produces a per-frame scalar statistic; a threshold
is then calibrated on CLEAN frames at the operating noise level so that
P(detect | clean) = alpha, and P(detect | jammed) is read off at that threshold.
Comparing anything at a common alpha is what makes "matched detectability"
meaningful.

    D_E   energy meter        -- the classical baseline; a power meter
    D_L   learned classifier  -- CNN on the 2-D IQ histogram; the deployed tier
    D_NP  Neyman-Pearson LRT  -- provably optimal; no detector can beat it

WHY D_NP IS THE POINT OF M0
---------------------------
With one carrier, known sigma and a stated perturbation law, the likelihood
ratio is available in closed form. That converts every claim of the form "the
trained detector does not see this attack" into "NO detector can do better than
this", and it gives the adaptation-cost measurement a reference point: the gap
between a retrained detector and D_NP is the adaptation budget that remains.

D_NP is granted knowledge of the exact attack law. That is deliberately the
strongest possible defender -- it is what makes the resulting curve a BOUND
rather than one model's opinion. A realistic mismatched detector can only do
worse, so any attack that evades D_NP evades everything.

DENSITIES
---------
A circularly-symmetric complex normal with E|z-mu|^2 = v has density
    p(z) = 1/(pi v) exp(-|z-mu|^2 / v),   v = 2*sigma^2
Clean, with iid uniform QPSK symbols, the per-sample law is the 4-component
mixture
    p0(y) = (1/4) sum_{s in S} CN(y; h0*s, v)
and p1 replaces h0*s by h0*(s+d), averaged over the attack's law for d. Because
the symbols are iid the frame log-likelihood ratio is just the sum over samples.
"""

import math

import torch

SQRT2_INV = 1.0 / math.sqrt(2.0)
_QPSK = torch.tensor([complex(a * SQRT2_INV, b * SQRT2_INV)
                      for a in (1.0, -1.0) for b in (1.0, -1.0)],
                     dtype=torch.complex64)


# --------------------------------------------------------------- thresholding
def calibrate(stat_clean, alpha):
    """
    Threshold at false-alarm rate `alpha`, from clean-frame statistics.

    Uses the empirical (1-alpha) quantile, so alpha is honoured by construction
    at the sample size used. Statistics are assumed to be "larger = more
    suspicious"; all three below are built that way.
    """
    q = torch.quantile(stat_clean.double(), 1.0 - alpha)
    return q.to(stat_clean.dtype)


def p_detect(stat, threshold):
    """Fraction of frames flagged. On clean frames this is the realised FAR."""
    return (stat > threshold).to(torch.float32).mean().item()


# ------------------------------------------------------------- D_E energy
def energy_statistic(y, clean_mean=None):
    """
    Mean received power per frame. The classical detector, and a pure scalar.

    ONE-SIDED (clean_mean=None) is the textbook form: flag when received power
    exceeds a threshold. It is what the classical literature deploys, because it
    was designed against barrage jammers, which ADD power.

    That convention carries a structural blind spot, and the minimum-energy
    attacks fall straight into it. Pushing a symbol toward its decision boundary
    moves it toward the ORIGIN, so the received power FALLS: at sigma=0.1 the
    clean frame power is 1.020, and boundary_genie at a transmit budget of 0.6
    brings it to 0.525, counter_null to 0.020. Both are then flagged with
    probability 0.0000 -- strictly BELOW the detector's own false-alarm rate.
    The energy detector is an added-power detector, not a jamming detector.

    TWO-SIDED (pass clean_mean) tests |T - E[T|clean]| instead, and catches both
    directions: the same two attacks go to 1.0000. Report both; the gap between
    them is a detector-characterisation result, not a tuning choice.

    Neither form sees counter_flip, whose received power is unchanged (1.0200 vs
    a clean 1.0202) because it REPLACES the signal rather than adding to it.
    """
    t = y.abs().pow(2).mean(dim=-1)
    return t if clean_mean is None else (t - clean_mean).abs()


# ------------------------------------------------------------- D_NP optimal
def _log_cn(y, mu, v):
    """log CN(y; mu, v) for complex y, broadcasting over a leading mixture axis."""
    return -math.log(math.pi) - torch.log(v) - (y - mu).abs().pow(2) / v


def log_p0(y, sigma, h0=1.0 + 0j):
    """Clean per-sample log-density: the 4-component QPSK mixture."""
    v = torch.as_tensor(2.0 * sigma ** 2, dtype=torch.float32, device=y.device)
    h0 = torch.as_tensor(h0, dtype=torch.complex64, device=y.device)
    mu = (h0 * _QPSK.to(y.device)).reshape(-1, *([1] * y.dim()))
    return torch.logsumexp(_log_cn(y.unsqueeze(0), mu, v), dim=0) - math.log(4.0)


def log_p1(y, attack, sigma, power=0.0, duty=1.0, h0=1.0 + 0j):
    """
    Jammed per-sample log-density under the stated attack law.

    Each branch is exact; only `barrage` needs a special function, and it has a
    closed form too -- averaging a Gaussian over a uniform-phase ring of radius
    r gives a modified Bessel factor I_0(2 r |u| / v).
    """
    dev = y.device
    v = torch.as_tensor(2.0 * sigma ** 2, dtype=torch.float32, device=dev)
    h0c = torch.as_tensor(h0, dtype=torch.complex64, device=dev)
    S = _QPSK.to(dev)
    rho = math.sqrt(power / duty) if (power > 0 and duty > 0) else 0.0

    def mix(mus):
        """Uniform mixture over the given means (a leading axis)."""
        mu = mus.reshape(-1, *([1] * y.dim()))
        return (torch.logsumexp(_log_cn(y.unsqueeze(0), mu, v), dim=0)
                - math.log(float(mus.numel())))

    if attack == "none":
        return log_p0(y, sigma, h0)

    if attack == "gaussian":
        # d ~ CN(0, power) adds to the noise: the jammed law is the SAME
        # constellation with inflated variance. This is why a Gaussian
        # perturbation is only ever detectable through its energy.
        return (torch.logsumexp(
            _log_cn(y.unsqueeze(0), (h0c * S).reshape(-1, *([1] * y.dim())),
                    v + power), dim=0) - math.log(4.0))

    if attack == "barrage":
        # Constant modulus r, uniform phase. Averaging over the ring:
        #   E_theta exp(-|u - r e^{j theta}|^2 / v)
        #     = exp(-(|u|^2 + r^2)/v) * I_0(2 r |u| / v)
        r = math.sqrt(power)
        u = (y.unsqueeze(0) - (h0c * S).reshape(-1, *([1] * y.dim()))).abs()
        z = 2.0 * r * u / v
        # i0e(z) = exp(-|z|) I_0(z), so log I_0(z) = log i0e(z) + z
        log_i0 = torch.log(torch.special.i0e(z) + 1e-30) + z
        comp = (-math.log(math.pi) - torch.log(v)
                - (u.pow(2) + r ** 2) / v + log_i0)
        return torch.logsumexp(comp, dim=0) - math.log(4.0)

    if attack == "permute":
        # Permutes the constellation, so the received law is EXACTLY the clean
        # law: p1 == p0 and the likelihood ratio is identically 1. No test can
        # separate them -- this is a theorem, not an approximation.
        return log_p0(y, sigma, h0)

    if attack == "counter_null":
        # d = -s: every symbol lands on the origin. y ~ CN(0, v), independent
        # of what was sent -- a single Gaussian blob, trivially distinguishable
        # from the four-lobed clean law.
        return _log_cn(y, torch.zeros((), dtype=torch.complex64, device=dev), v)

    if attack == "counter_flip":
        # d = -2s: y ~ CN(-h0 s, v). The QPSK alphabet is symmetric under
        # negation, so {-s} = {s} as a set and this law is IDENTICAL to p0.
        # The maximally effective attack is therefore EXACTLY undetectable by
        # any test on the received distribution. See verify_detectors.py.
        return mix(h0c * (-S))

    if attack in ("boundary_blind", "boundary_genie"):
        # Duty cycling mixes an unattacked component in with weight (1-duty).
        if attack == "boundary_blind":
            # d takes the four axis values +-rho, +-j rho, independent of s.
            offs = torch.tensor([rho, -rho, 1j * rho, -1j * rho],
                                dtype=torch.complex64, device=dev)
            means = (S.reshape(-1, 1) + offs.reshape(1, -1)).reshape(-1)
        else:
            # d depends on s: push toward the I or the Q boundary of that symbol.
            di = -rho * torch.sign(S.real).to(torch.complex64)
            dq = -rho * torch.sign(S.imag).to(torch.complex64) * 1j
            means = torch.cat([S + di, S + dq])
        lp_att = mix(h0c * means)
        if duty >= 1.0:
            return lp_att
        lp_cln = log_p0(y, sigma, h0)
        return torch.logaddexp(lp_cln + math.log(1.0 - duty),
                               lp_att + math.log(duty))

    raise ValueError(f"no NP density implemented for attack {attack!r}")


def np_statistic(y, attack, sigma, power=0.0, duty=1.0, h0=1.0 + 0j):
    """
    Frame log-likelihood ratio, the Neyman-Pearson statistic.

    Symbols are iid, so the frame joint factorises and the LLR is the sum of
    per-sample LLRs.
    """
    return (log_p1(y, attack, sigma, power, duty, h0)
            - log_p0(y, sigma, h0)).sum(dim=-1)


# ------------------------------------------------------------- D_L learned
def iq_histogram(y, bins=48, lim=2.5):
    """
    Permutation-invariant frame summary: a 2-D histogram of the received points.

    The received samples are iid given the attack, so their ORDER carries no
    information and a histogram is the right inductive bias -- the single-carrier
    counterpart of the spectrogram image used on the OFDM stack. Returns
    (n_frames, 1, bins, bins), L1-normalised per frame.
    """
    n = y.shape[0]
    e = torch.clamp(((y.real + lim) / (2 * lim) * bins).long(), 0, bins - 1)
    f = torch.clamp(((y.imag + lim) / (2 * lim) * bins).long(), 0, bins - 1)
    flat = (e * bins + f).reshape(n, -1)
    h = torch.zeros(n, bins * bins, device=y.device)
    h.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    h = h / h.sum(dim=1, keepdim=True).clamp(min=1.0)
    return h.reshape(n, 1, bins, bins)


class LearnedDetector(torch.nn.Module):
    """Small CNN on the IQ histogram. Deliberately modest: M0 is a 2-D problem."""

    def __init__(self, bins=48):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(64, 1))

    def forward(self, hist):
        return self.net(hist).squeeze(-1)      # logit; larger = more suspicious
