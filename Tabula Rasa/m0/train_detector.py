"""
M0 -- train the learned detector D_L, one per noise level.

WHY ONE PER SIGMA
-----------------
D_NP is granted the noise level AND the exact attack law. To make the
comparison isolate the interesting difference, D_L is granted the noise level
too (a detector per sigma) but NOT the attack law: it is trained on a mixture of
attacks and must generalise. The remaining gap D_NP - D_L is then exactly "what
a realistic learned detector leaves on the table against an optimal one", which
is the reference the adaptation-cost measurement needs.

LABELLING
---------
Frames are labelled by ATTACK PRESENCE, not by induced BER. A very low-power
attack that causes no errors is still labelled jammed, which inflates the
false-alarm rate -- the same caveat the sim08 detector carried. Kept deliberately
so the two are comparable; BER-thresholded labelling is a listed refinement.

    python train_detector.py --sigmas 0.05,0.1,0.2,0.3,0.5 --epochs 30
"""

import argparse, json, math, os, time

import torch

import link, attacks, detectors as D

TRAIN_ATTACKS = ["gaussian", "barrage", "boundary_blind", "boundary_genie",
                 "counter_null"]
# counter_flip is EXCLUDED from training: its received law is provably identical
# to clean (see detectors.log_p1), so labelling it "jammed" teaches the network
# to fit noise and only damages the clean false-alarm rate.


def make_batch(n, sigma, n_sym, bins, device, generator, jam_frac=0.5):
    n_j = int(n * jam_frac)
    bits = link.random_bits(n, n_sym, device=device, generator=generator)
    s = link.bits_to_symbols(bits)
    d = torch.zeros_like(s)
    if n_j:
        per = max(1, n_j // len(TRAIN_ATTACKS))
        i = 0
        for atk in TRAIN_ATTACKS:
            j = min(i + per, n_j)
            if j <= i:
                break
            # log-uniform power over the regime where detection is non-trivial
            P = float(10 ** (torch.empty(1, device=device).uniform_(-3.0, 0.0,
                      generator=generator).item()))
            duty = 0.98 * min(1.0, 2 * P) if "boundary" in atk else 1.0
            duty = max(duty, 1.0 / n_sym)
            dd, _ = attacks.build_attack(atk, s[i:j], power=P, duty=duty,
                                         device=device, generator=generator)
            d[i:j] = dd
            i = j
    y, _ = link.receive(s, d=d, sigma=sigma, device=device, generator=generator)
    lab = torch.zeros(n, device=device)
    lab[:n_j] = 1.0
    return D.iq_histogram(y, bins=bins), lab


def train_one(sigma, args, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    net = D.LearnedDetector(bins=args.bins).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for _ in range(args.steps):
            h, lab = make_batch(args.batch, sigma, args.n_sym, args.bins,
                                device, g)
            loss = lossf(net(h), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"    sigma={sigma:<5} epoch {ep+1:>3}/{args.epochs}  "
                  f"loss={tot/args.steps:.4f}", flush=True)
    # held-out accuracy and the realised clean false-alarm rate at alpha
    net.eval()
    with torch.no_grad():
        h, lab = make_batch(4096, sigma, args.n_sym, args.bins, device, g)
        logit = net(h)
        acc = (((logit > 0).float()) == lab).float().mean().item()
        hc, _ = make_batch(4096, sigma, args.n_sym, args.bins, device, g,
                           jam_frac=0.0)
        thr = D.calibrate(net(hc), args.alpha).item()
    return net, acc, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--n-sym", type=int, default=256)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="../artifacts/m0/detector")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(out, exist_ok=True)
    print(f"device={device}  out={out}", flush=True)

    summary, t0 = {}, time.time()
    for sigma in [float(x) for x in args.sigmas.split(",")]:
        print(f"  training D_L for sigma={sigma} "
              f"(Eb/N0={link.ebn0_db(sigma):.1f} dB)", flush=True)
        net, acc, thr = train_one(sigma, args, device)
        torch.save({"state_dict": net.state_dict(), "bins": args.bins,
                    "sigma": sigma, "threshold": thr, "n_sym": args.n_sym},
                   os.path.join(out, f"dl_sigma{sigma:g}.pt"))
        summary[str(sigma)] = {"acc": acc, "threshold": thr,
                               "ebn0_db": link.ebn0_db(sigma)}
        print(f"    -> held-out acc={acc:.4f}  thr@alpha={thr:.3f}", flush=True)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "per_sigma": summary}, f, indent=2)
    print(f"\ndone in {time.time()-t0:.1f}s -> {out}/summary.json")


if __name__ == "__main__":
    main()
