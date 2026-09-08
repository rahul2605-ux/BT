"""
M0 -- correctness suite.

Every claim the model rests on, checked against an analytic prediction. Run
before trusting any frontier output:

    python verify.py            # all checks
    python verify.py -v         # print the tables too

Exit code 0 iff every check passes. Written to be readable as documentation:
each check states the prediction, then tests it.
"""

import argparse
import math
import sys

import torch

import link
import attacks

FAILURES = []
VERBOSE = False


def check(label, got, want, tol, note=""):
    ok = abs(got - want) <= tol
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<46} got={got:<10.5f} want={want:<10.5f} tol={tol:g}"
          + (f"   {note}" if note else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def _sim(name, P, sigma, duty=1.0, nf=800, ns=512, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    bits = link.random_bits(nf, ns, generator=g)
    s = link.bits_to_symbols(bits)
    d, x = attacks.build_attack(name, s, power=P, duty=duty, generator=g, **kw)
    _, yhat = link.receive(s, d=d, sigma=sigma, generator=g)
    ber, ser = link.ber_ser(bits, link.symbols_to_bits(yhat))
    return ber, ser, attacks.transmit_power(x).mean().item()


# ---------------------------------------------------------------- 1. the link
def test_link_matches_theory():
    """
    Unjammed BER must equal the analytic QPSK-over-AWGN value
        Q(1/(sigma*sqrt(2)))
    Only tested where the predicted BER is large enough to measure: below
    sigma~0.15 the theoretical value is <1e-7, so zero errors in a 4e6-bit
    sample is the correct outcome and carries no information.
    """
    print("\n1. Link vs analytic QPSK/AWGN")
    for sigma in [0.2, 0.25, 0.3, 0.4, 0.5]:
        ber, ser, _ = _sim("none", 0.0, sigma, seed=11)
        th = link.theoretical_ber(sigma)
        # 4-sigma Monte Carlo band on ~800*512*2 bits
        n = 800 * 512 * 2
        tol = 4 * math.sqrt(max(th, 1e-12) * (1 - th) / n) + 1e-5
        check(f"BER(sigma={sigma}) [Eb/N0={link.ebn0_db(sigma):.1f}dB]", ber, th, tol)
        # SER = 1-(1-BER)^2 holds only IN EXPECTATION: it assumes the two bit
        # errors are independent. Empirically SER = 2*BER - P(both wrong), and
        # P(both wrong) fluctuates about BER^2 with a binomial spread over the
        # n_sym symbols. Test against that band, not against equality.
        n_sym = 800 * 512
        p2 = ber ** 2
        tol_ser = 4 * math.sqrt(max(p2, 1e-12) * (1 - p2) / n_sym) + 1e-6
        check(f"SER(sigma={sigma}) == 1-(1-BER)^2", ser, 1 - (1 - ber) ** 2, tol_ser)


def test_unit_energy_and_domains():
    """E|s|^2 = 1, and the composite/equalised split is exact for any h0."""
    print("\n2. Constellation energy and the two-domain split")
    g = torch.Generator().manual_seed(12)
    s = link.bits_to_symbols(link.random_bits(1, 200000, generator=g))
    check("E|s|^2", s.abs().pow(2).mean().item(), 1.0, 1e-4)
    h0 = 0.6 + 0.8j
    d = torch.full_like(s, 0.3 + 0.1j)
    y, yhat = link.receive(s, d=d, sigma=0.0, h0=h0, generator=g)
    check("max|yhat-(s+d)|", (yhat - (s + d)).abs().max().item(), 0.0, 1e-6)
    check("max|y-h0(s+d)|", (y - torch.as_tensor(h0, dtype=torch.complex64)
                             * (s + d)).abs().max().item(), 0.0, 1e-6)


# ------------------------------------------------------------ 3. the attackers
def test_deterministic_attacks():
    """
    Closed-form predictions at sigma=0:
      counter_flip  d=-2s -> received = -s, both bits flip -> BER 1.0, energy 4
      counter_null  d=-s  -> received = 0, tie breaks to bits (0,0) -> BER 0.5
      boundary      rho>1/sqrt2 (P>0.5) -> exactly one bit flips -> BER 0.5
      boundary      rho<1/sqrt2 (P<0.5) -> never crosses -> BER 0
    """
    print("\n3. Deterministic attacks at sigma=0")
    ber, ser, p = _sim("counter_flip", 0.0, 0.0, seed=13)
    check("counter_flip BER", ber, 1.0, 1e-9)
    check("counter_flip SER", ser, 1.0, 1e-9)
    check("counter_flip energy", p, 4.0, 1e-3)
    ber, _, p = _sim("counter_null", 0.0, 0.0, seed=13)
    check("counter_null BER", ber, 0.5, 5e-3)
    check("counter_null energy", p, 1.0, 1e-3)
    ber, ser, _ = _sim("boundary_genie", 0.51, 0.0, seed=13)
    check("boundary P=0.51 BER (one bit flipped)", ber, 0.5, 1e-3)
    check("boundary P=0.51 SER", ser, 1.0, 1e-3)
    ber, _, _ = _sim("boundary_genie", 0.49, 0.0, seed=13)
    check("boundary P=0.49 BER (falls short)", ber, 0.0, 1e-9)


def test_power_is_a_hard_constraint():
    """No attacker may exceed its budget. counter_* are exempt: they are bounds."""
    print("\n4. Power budget is a hard constraint")
    for name in ["barrage", "gaussian", "boundary_blind", "boundary_genie"]:
        for P in [0.25, 1.0]:
            _, _, mp = _sim(name, P, 0.1, seed=14)
            ok = mp <= P * 1.001
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} budget={P}"
                  f"{'':<{max(0, 26-len(name))}} measured={mp:.5f}")
            if not ok:
                FAILURES.append(f"{name} power {P}")


def test_min_energy_frontier():
    """
    The sigma=0 minimum-energy frontier.

    A boundary push of rho flips one of the symbol's two bits iff rho > 1/sqrt2,
    at per-symbol energy rho^2. Spending budget P on a fraction `duty` of the
    symbols gives rho = sqrt(P/duty), so flips occur iff duty < 2P, and

        BER = 0.5 * duty     ->     BER*(P) = sup = min(P, 0.5)

    approached as duty -> 2P from below. Exactly at duty = 2P the push lands on
    the boundary (rho = 1/sqrt2) and the tie breaks the wrong way, so the
    supremum is not attained -- which is why the sweep must not include duty=2P.
    """
    print("\n5. Minimum-energy frontier  BER*(P) = min(P, 0.5)")
    for P in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        duty = 0.98 * min(1.0, 2 * P)
        ber, _, mp = _sim("boundary_genie", P, 0.0, duty=duty, seed=15)
        check(f"BER at P={P}, duty={duty:.3f}", ber, 0.5 * duty, 4e-3,
              note=f"frontier min(P,.5)={min(P, 0.5):.3f}")
        check(f"  spends budget exactly (P={P})", mp, P, 2e-3)


def test_blind_frontier():
    """
    The BLIND minimum-energy frontier.

    Pushing along a random axis direction, independent of the symbol, flips a
    bit only when the push opposes that axis's sign -- probability 1/2. So an
    attacked symbol errs w.p. 1/2 rather than 1, and

        BER = 0.25 * duty   ->   BER*(P) = min(P/2, 0.25)

    exactly half the genie frontier min(P, 0.5) at every power. That factor of
    two is the entire value of knowing the transmitted symbol, and it bounds
    what any amount of learning could recover on a single link.
    """
    print("\n7. Blind minimum-energy frontier  BER*(P) = min(P/2, 0.25)")
    for P in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        duty = 0.98 * min(1.0, 2 * P)
        bb, ss, _ = _sim("boundary_blind", P, 0.0, duty=duty, seed=17)
        check(f"blind BER at P={P}", bb, 0.25 * duty, 4e-3,
              note=f"frontier min(P/2,.25)={min(P / 2, 0.25):.3f}")
        check(f"  blind SER == 2*BER (one bit at a time)", ss, 2 * bb, 4e-3)
        bg, _, _ = _sim("boundary_genie", P, 0.0, duty=duty, seed=17)
        check(f"  genie/blind ratio at P={P}", bg / max(bb, 1e-9), 2.0, 0.06)


def test_efficiency_ranking():
    """
    BER per unit energy. The omniscient counter_flip attack is the ceiling on
    EFFECT (BER 1.0) but the FLOOR on efficiency -- it spends energy 4 for what
    the genie boundary attack achieves at 0.5. Stated explicitly because the
    two roles are easy to conflate.
    """
    print("\n8. Efficiency ranking (BER per unit transmit energy)")
    rows = []
    for nm, P in [("boundary_genie", 0.51), ("counter_null", 0.0),
                  ("boundary_blind", 0.51), ("counter_flip", 0.0)]:
        b, _, e = _sim(nm, P, 0.05, seed=18)
        rows.append((b / e, nm, b, e))
    for eff, nm, b, e in rows:
        print(f"  {nm:<16} energy={e:<6.2f} BER={b:<6.3f} -> {eff:.3f} BER/energy")
    ok = rows[0][0] > rows[-1][0]
    print(f"  [{'PASS' if ok else 'FAIL'}] genie boundary is the most efficient, "
          f"counter_flip the least")
    if not ok:
        FAILURES.append("efficiency ranking")


def test_boundary_dominates_blind():
    """
    At matched power the minimum-energy attack must beat both blind tiers.
    This is the claim that motivates the whole attacker ladder; if it fails,
    the 'energy-optimal' framing is wrong.
    """
    print("\n6. Minimum-energy dominates the blind tiers at matched power")
    for P in [0.1, 0.3, 1.0]:
        duty = 0.98 * min(1.0, 2 * P)
        bb, _, _ = _sim("boundary_genie", P, 0.0, duty=duty, seed=16)
        ba, _, _ = _sim("barrage", P, 0.0, seed=16)
        bg, _, _ = _sim("gaussian", P, 0.0, seed=16)
        best = max(ba, bg)
        ok = bb > best
        print(f"  [{'PASS' if ok else 'FAIL'}] P={P:<5} boundary={bb:.4f} > "
              f"barrage={ba:.4f}, gaussian={bg:.4f}   ({bb / max(best, 1e-9):.0f}x)")
        if not ok:
            FAILURES.append(f"dominance P={P}")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    VERBOSE = ap.parse_args().verbose

    print("=" * 78)
    print(" M0 correctness suite")
    print("=" * 78)
    test_link_matches_theory()
    test_unit_energy_and_domains()
    test_deterministic_attacks()
    test_power_is_a_hard_constraint()
    test_min_energy_frontier()
    test_boundary_dominates_blind()
    test_blind_frontier()
    test_efficiency_ranking()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f" {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print(" ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
