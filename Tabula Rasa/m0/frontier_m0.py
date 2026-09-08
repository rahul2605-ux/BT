"""
M0 -- E1: the effectiveness-detectability sweep.

For every (attack, power, duty) at a given noise level, measures what the victim
loses (BER, SER) and what each detector sees (P_detect at a common false-alarm
rate alpha). That pair is the whole experiment: one point per configuration on
the effectiveness-detectability plane.

Detectors, all calibrated on CLEAN frames at this sigma so alpha is honoured:
    P_E1  energy, one-sided (the classical, deployed form)
    P_E2  energy, two-sided (covers the power-REDUCING blind spot)
    P_L   learned CNN on the IQ histogram (per-sigma checkpoint)
    P_NP  Neyman-Pearson optimal -- the bound; no detector beats it

sigma = 0 is NOT simulated. There the clean constellation is four exact points,
the Gaussian densities degenerate, and any perturbation is detected with
probability one; the anchor is analytic (BER*(P) = min(P, 0.5)) and is stated
rather than measured.

    python frontier_m0.py --sigma 0.2
    sbatch submit_frontier.sh          # array job, one sigma per task
"""

import argparse, json, math, os, time

import torch

import link, attacks, detectors as D

ATTACKS = ["none", "barrage", "gaussian", "boundary_blind", "boundary_genie",
           "permute", "counter_null", "counter_flip"]
UNBUDGETED = ("none", "counter_null", "counter_flip")   # reported as bounds


def duty_grid(attack, power):
    """Duty values to try. The optimum sits just below 2P; also probe a coarse
    grid so the achievable frontier is a max over strategies, not one guess."""
    if "boundary" not in attack:
        return [1.0]
    opt = 0.98 * min(1.0, 2.0 * power)
    return sorted({round(min(1.0, max(1e-3, d)), 5)
                   for d in (opt, 1.0, 0.5, 0.25, 0.1)})


def run_sigma(sigma, args, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    NF, NS, ALPHA = args.frames, args.n_sym, args.alpha

    def frames(atk, P, duty):
        bits = link.random_bits(NF, NS, device=device, generator=g)
        s = link.bits_to_symbols(bits)
        d, x = attacks.build_attack(atk, s, power=P, duty=duty,
                                    device=device, generator=g)
        y, yh = link.receive(s, d=d, sigma=sigma, device=device, generator=g)
        ber, ser = link.ber_ser(bits, link.symbols_to_bits(yh))
        return y, ber, ser, attacks.transmit_power(x).mean().item()

    # ---- calibrate every detector on clean frames at this sigma -------------
    yc, ber0, ser0, _ = frames("none", 0.0, 1.0)
    mu = D.energy_statistic(yc).mean()
    t1 = D.calibrate(D.energy_statistic(yc), ALPHA)
    t2 = D.calibrate(D.energy_statistic(yc, mu), ALPHA)

    net = tL = None
    ck_path = os.path.join(args.detector_dir, f"dl_sigma{sigma:g}.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        net = D.LearnedDetector(bins=ck["bins"]).to(device)
        net.load_state_dict(ck["state_dict"]); net.eval()
        with torch.no_grad():
            tL = D.calibrate(net(D.iq_histogram(yc, ck["bins"])), ALPHA)
    else:
        print(f"  [warn] no learned detector at {ck_path}; P_L omitted")

    def p_learned(y):
        if net is None:
            return float("nan")
        with torch.no_grad():
            return D.p_detect(net(D.iq_histogram(y, ck["bins"])), tL)

    rows = []
    # realised false-alarm rates on an INDEPENDENT clean batch
    yc2, _, _, _ = frames("none", 0.0, 1.0)
    far = {"E1": D.p_detect(D.energy_statistic(yc2), t1),
           "E2": D.p_detect(D.energy_statistic(yc2, mu), t2),
           "L": p_learned(yc2)}
    print(f"  sigma={sigma}  clean BER={ber0:.5f}  realised FAR "
          f"E1={far['E1']:.3f} E2={far['E2']:.3f} L={far['L']:.3f}", flush=True)

    for atk in ATTACKS:
        powers = [0.0] if atk in UNBUDGETED else args.powers
        for P in powers:
            for duty in duty_grid(atk, P):
                y, ber, ser, ptx = frames(atk, P, duty)
                try:
                    tN = D.calibrate(D.np_statistic(yc, atk, sigma, P, duty), ALPHA)
                    pnp = D.p_detect(D.np_statistic(y, atk, sigma, P, duty), tN)
                except Exception as e:                     # no density for it
                    pnp = float("nan")
                rows.append(dict(sigma=sigma, attack=atk, power=P, duty=duty,
                                 ber=ber, ser=ser, p_tx=ptx,
                                 p_e1=D.p_detect(D.energy_statistic(y), t1),
                                 p_e2=D.p_detect(D.energy_statistic(y, mu), t2),
                                 p_l=p_learned(y), p_np=pnp))
    return dict(sigma=sigma, ebn0_db=link.ebn0_db(sigma), alpha=ALPHA,
                n_sym=NS, frames=NF, clean_ber=ber0, clean_ser=ser0,
                far=far, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=None,
                    help="single sigma; default runs the whole grid")
    ap.add_argument("--sigmas", default="0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--powers", default="0.001,0.003,0.01,0.03,0.06,0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--n-sym", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--detector-dir", default="../artifacts/m0/detector")
    ap.add_argument("--out", default="../artifacts/m0/frontier")
    args = ap.parse_args()
    args.powers = [float(x) for x in args.powers.split(",")]
    args.detector_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), args.detector_dir))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(out, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = ([args.sigma] if args.sigma is not None
              else [float(x) for x in args.sigmas.split(",")])
    print(f"device={device}  sigmas={sigmas}  powers={args.powers}", flush=True)

    t0 = time.time()
    for sg in sigmas:
        res = run_sigma(sg, args, device)
        tag = f"sigma{sg:g}"
        with open(os.path.join(out, f"results_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"  -> {out}/results_{tag}.json  ({len(res['rows'])} rows)", flush=True)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
