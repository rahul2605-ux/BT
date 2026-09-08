"""
M0 -- the attacker ladder.

Every attacker is a conditional law pi(d | c): it returns the perturbation `d`
in the EQUALISED domain (what the victim's decision variable is displaced by).
The tiers differ only in what `c` contains, which is what lets closed-form and
learned attacks be compared on one axis.

    name            knows                       role
    ---------------------------------------------------------------------
    none            --                          BER/SER floor + detector FAR
    barrage         nothing                     naive floor (random direction)
    gaussian        nothing                     max-entropy perturbation; the
                                                stealth reference for E3
    boundary        symbol + both channels      closed-form MINIMUM ENERGY,
                                                no training -- the real bar
    counter_null    symbol + both channels      d = -s, received point -> 0
    counter_flip    symbol + both channels      d = -2s, both bits flip; the
                                                omniscient CEILING

POWER IS A HARD CONSTRAINT, NOT A PENALTY
-----------------------------------------
The budget applies to the TRANSMITTED signal x_k = d*h0/g_k, not to `d`. In the
base case h0 = g = 1 the two coincide, but the distinction is what makes the
"where detection happens" question bite once channels differ. `project_power`
enforces it by scaling, so no attacker can buy effectiveness by overspending.

WHY THE BOUNDARY ATTACK IS THE BAR
----------------------------------
For unit-energy QPSK the constellation sits at (+-1 +- 1j)/sqrt(2) and the
decision boundaries are the two axes, so every symbol is at distance 1/sqrt(2)
from its nearest boundary. Displacing by rho along -I therefore flips bit0 (and
only bit0) as soon as rho > 1/sqrt(2) ~ 0.7071, i.e. at a per-symbol energy of
rho^2 ~ 0.5 -- yielding BER 0.5 and SER 1.0. Flipping BOTH bits requires pushing
the point past the origin, energy ~1.0, which is the counter_null regime. A
random-direction shove of the same energy achieves far less, because most of it
slides the symbol ALONG a boundary instead of across it.

Two knobs sweep the frontier:
  power  -- mean transmit power budget P
  duty   -- fraction of symbols attacked. At budget P the attacked symbols get
            rho = sqrt(P/duty), so duty < 1 trades breadth for depth. This is
            the M0 form of the sparse-vs-dense lever that mattered in sim08:
            below P = 0.5 a dense attack cannot reach the boundary on any
            symbol, whereas a sparse one deterministically flips a subset.
"""

import math
import torch

from link import INV_SQRT2

ATTACKS = ["none", "barrage", "gaussian", "boundary_blind", "boundary_genie",
           "permute", "counter_null", "counter_flip"]


def transmit_power(x):
    """Mean per-symbol transmit power, averaged over the frame."""
    return x.abs().pow(2).mean(dim=-1)


def project_power(x, budget):
    """
    Hard power constraint: scale any frame exceeding `budget` back onto the
    boundary of the feasible set. Frames already inside are untouched, so this
    is a projection and not a normalisation.
    """
    if budget is None or budget == float("inf"):
        return x
    p = transmit_power(x)
    scale = torch.clamp(math.sqrt(budget) / (p.sqrt() + 1e-12), max=1.0)
    return x * scale.unsqueeze(-1)


def _duty_mask(shape, duty, device, generator):
    """
    Random subset of EXACTLY round(duty*n_sym) symbols per frame.

    Deliberately not iid Bernoulli. With a Bernoulli mask the number of
    attacked symbols fluctuates (Binomial), so the realised frame power
    fluctuates about the budget; `project_power` then scales the over-budget
    frames down -- which drops rho below 1/sqrt(2) and destroys ALL of their
    bit flips -- while leaving under-budget frames alone. That one-sided
    clipping silently removed roughly half the flips at low duty. Fixing the
    count makes the realised frame power exactly duty*rho^2 = P, so the
    projection never binds and the attacker spends its budget exactly.
    """
    if duty >= 1.0:
        return torch.ones(shape, device=device)
    n_sym = shape[-1]
    k = max(1, int(round(duty * n_sym)))
    u = torch.rand(shape, device=device, generator=generator)
    idx = u.argsort(dim=-1)[..., :k]
    mask = torch.zeros(shape, device=device)
    mask.scatter_(-1, idx, 1.0)
    return mask


def build_attack(name, sym, power, sigma=None, duty=1.0, axis="random",
                 h0=1.0 + 0j, g=1.0 + 0j, device="cpu", generator=None):
    """
    Return (d, x): the equalised-domain perturbation and the transmitted signal.

    `power` is the hard budget on x. `sym` is the victim's symbol frame -- only
    the symbol-aware tiers (boundary, counter_*) may read it; the blind tiers
    use its shape alone.
    """
    h0 = torch.as_tensor(h0, dtype=torch.complex64, device=device)
    g = torch.as_tensor(g, dtype=torch.complex64, device=device)
    shape = sym.shape

    if name == "none":
        d = torch.zeros_like(sym)

    elif name == "barrage":
        # Constant modulus, uniform random phase: "a vector in a random
        # direction in the I/Q plot". Blind to the symbol.
        theta = torch.rand(shape, device=device, generator=generator) * (2 * math.pi)
        mag = math.sqrt(power) if power > 0 else 0.0
        d = mag * torch.complex(torch.cos(theta), torch.sin(theta))

    elif name == "gaussian":
        # CN(0, power): the maximum-entropy perturbation at this power, and the
        # law a large team of independent jammers converges to (CLT). Its whole
        # effect on the received distribution is a noise-variance increase.
        s2 = math.sqrt(power / 2.0) if power > 0 else 0.0
        d = torch.complex(torch.randn(shape, device=device, generator=generator) * s2,
                          torch.randn(shape, device=device, generator=generator) * s2)

    elif name == "boundary_blind":
        # BLIND minimum-energy attack: push along one of the four axis
        # directions {+I,-I,+Q,-Q}, chosen uniformly at random and INDEPENDENTLY
        # of the transmitted symbol. No demodulation of the victim, no
        # within-symbol reaction time -- this is realisable.
        #
        # Half the time the push opposes the symbol's sign on that axis and
        # flips that bit; half the time it drives the point deeper into its own
        # quadrant and does nothing. So an attacked symbol errs with prob 1/2,
        # against prob 1 for the genie, at identical energy: symbol knowledge is
        # worth exactly a factor of two, no more.
        #
        # Randomising over all four directions does not raise BER over using a
        # single fixed direction (which also gives 1/2) -- it symmetrises the
        # signature, which is a STEALTH gain, not an effectiveness one.
        rho = math.sqrt(power / duty) if (power > 0 and duty > 0) else 0.0
        mask = _duty_mask(shape, duty, device, generator)
        use_i = (torch.rand(shape, device=device, generator=generator) < 0.5
                 ).to(torch.float32)
        sgn = torch.where(
            torch.rand(shape, device=device, generator=generator) < 0.5,
            torch.ones(shape, device=device), -torch.ones(shape, device=device))
        d = torch.complex(rho * sgn * use_i, rho * sgn * (1.0 - use_i)) * mask

    elif name == "boundary_genie":
        # Minimum-energy: displace along the axis that flips exactly one bit.
        rho = math.sqrt(power / duty) if (power > 0 and duty > 0) else 0.0
        mask = _duty_mask(shape, duty, device, generator)
        if axis == "i":
            use_i = torch.ones(shape, device=device)
        elif axis == "q":
            use_i = torch.zeros(shape, device=device)
        else:  # "random": pick the I or Q boundary per symbol, equiprobably.
            use_i = (torch.rand(shape, device=device, generator=generator) < 0.5
                     ).to(torch.float32)
        # Push against the sign of the chosen component, i.e. toward its boundary.
        dr = -rho * torch.sign(sym.real) * use_i
        di = -rho * torch.sign(sym.imag) * (1.0 - use_i)
        d = torch.complex(dr, di) * mask

    elif name == "permute":
        # THE INVISIBLE ATTACK. Displace each symbol by exactly the difference
        # vector to an ADJACENT constellation point: rho = sqrt(2) along one
        # axis. The symbol does not land near a boundary, it lands ON another
        # legitimate constellation point, so one bit is flipped with certainty
        # while the received DISTRIBUTION is unchanged.
        #
        # Because the payload is iid uniform, any perturbation that PERMUTES the
        # constellation leaves the received law exactly invariant, and is
        # therefore undetectable by every possible test -- p1 == p0, so the
        # likelihood ratio is identically 1. counter_flip (s -> -s) is the other
        # member of this family, at energy 4 for BER 1.0.
        #
        # Cost: energy |s' - s|^2 = 2 per attacked symbol, against 0.5 for the
        # cheapest DETECTABLE bit flip. Spending budget P at this fixed
        # displacement attacks a fraction duty = P/2, giving
        #       BER = min(P/4, 1/2)
        # against the minimum-energy min(P, 1/2). Invisibility costs exactly 4x.
        #
        # It is knife-edged: rho must be sqrt(2). At rho = 1.2 or 1.6 the points
        # land BETWEEN constellation points and P(detect) jumps to 1.0. That is
        # what makes the amplitude/phase-error realism axis decisive rather than
        # decorative -- and it needs the genie, since choosing the difference
        # vector requires knowing which point was sent.
        rho = math.sqrt(2.0)
        duty_eff = min(1.0, power / 2.0) if power > 0 else 0.0
        mask = _duty_mask(shape, duty_eff, device, generator)
        use_i = (torch.rand(shape, device=device, generator=generator) < 0.5
                 ).to(torch.float32)
        dr = -rho * torch.sign(sym.real) * use_i
        di = -rho * torch.sign(sym.imag) * (1.0 - use_i)
        d = torch.complex(dr, di) * mask

    elif name == "counter_null":
        # d = -s: the received point lands on the origin, so both bits are
        # decided by noise alone -> BER 0.5. Per-symbol energy 1.
        d = -sym

    elif name == "counter_flip":
        # d = -2s: the received point is the exact negation -> both bits flip,
        # BER 1.0. Per-symbol energy 4. The "impossible to beat" ceiling.
        d = -2.0 * sym

    else:
        raise ValueError(f"unknown attack {name!r}; choose from {ATTACKS}")

    # Referred to the transmitter, then constrained. The counter_* tiers are
    # reported as bounds, so they are exempt from the budget by construction.
    x = d * h0 / g
    if name not in ("none", "permute", "counter_null", "counter_flip"):
        x = project_power(x, power)
        d = x * g / h0
    return d, x
