"""
C5 -- train the CGAN, Zhou et al. 2025 §III (README §2.10, §3.4).

    sbatch submit_train.sh --run run002                 # default recipe below
    sbatch submit_train.sh --run run003 --adv nsgan     # adversarial-loss ablation

Recipe (Zhou states iterations, batch and the loss family; the rest follows the
papers Zhou cites where they are specific, README §4.2 Q7):
  10,000 iterations, batch 128, Adam lr 2e-4.
  wgan-gp (default, [7]): n_critic 5, betas (0.5, 0.9).  nsgan: n_critic 1, betas (0.5, 0.999).
  G loss weights: adv 1, feat 2, stft 45 ([5] Fre-GAN's lambda_fm, lambda_mel), iq 1.
run001 (2026-09-14) used nsgan, n_critic 1, all weights 1, MSE feature matching,
log1p STFT and peak headroom 1.0 -- the code no longer reproduces it.

Real training data is clean QPSK waveform segments from the calibrated link
(link.LINK), peak-normalised into (-1, 1) to match G's Tanh. QPSK is one class, so
labels are all 0 -- the conditioning path is exercised but trivial (step 2 gives it
work). Writes the generator, the scaler, the loss log and a loss figure to
../artifacts/cgan/<run>_*.
"""

import argparse
import json
import os
import time

import torch

import link as lk
import models
import losses

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "cgan")

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3de"
C_G, C_D = "#2a78d6", "#eb6834"          # dataviz slots: generator blue, discriminator orange


def loss_figure(log, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    it = [r["it"] for r in log]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 4.5))
    a0.plot(it, [r["g_total"] for r in log], color=C_G, lw=1.5, label="generator")
    a0.plot(it, [r["d_total"] for r in log], color=C_D, lw=1.5, label="discriminator")
    a0.set_title("Total loss", loc="left")
    a0.set_xlabel("iteration"); a0.set_ylabel("loss"); a0.legend(frameon=False)
    for k, c in [("adv", C_G), ("feat", "#1baf7a"), ("stft", "#eda100"), ("iq", "#e87ba4")]:
        a1.plot(it, [r["g"][k] for r in log], color=c, lw=1.3, label=f"G {k} (unweighted)")
    a1.set_title("Generator loss terms", loc="left")
    a1.set_xlabel("iteration"); a1.legend(frameon=False, fontsize=8)
    a2.plot(it, [r["d"]["score_gap"] for r in log], color=C_D, lw=1.3)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_title("Critic gap  E D(real) − E D(fake)", loc="left")
    a2.set_xlabel("iteration")
    for ax in (a0, a1, a2):
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run001")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv", choices=["nsgan", "wgan-gp"], default="wgan-gp")
    ap.add_argument("--n-critic", type=int, default=None)
    ap.add_argument("--w-adv", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=2.0)     # [5] lambda_fm
    ap.add_argument("--w-stft", type=float, default=45.0)    # [5] lambda_mel
    ap.add_argument("--w-iq", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    wgan = args.adv == "wgan-gp"
    n_critic = args.n_critic if args.n_critic is not None else (5 if wgan else 1)
    betas = (0.5, 0.9) if wgan else (0.5, 0.999)

    device = lk.setup(seed=args.seed)
    os.makedirs(ART, exist_ok=True)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})

    # Fit the peak scaler once on a large sample of clean segments.
    scaler = models.PeakScaler.fit(L.real_segments(4096, models.SEG_LEN))
    print(f"device {device} | adv {args.adv} | n_critic {n_critic} | betas {betas} | "
          f"peak scale {scaler.scale:.4f}", flush=True)

    G = models.Generator(n_classes=1).to(device)
    D = models.Discriminator(n_classes=1).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=betas)
    w_g = {"adv": args.w_adv, "feat": args.w_feat, "stft": args.w_stft, "iq": args.w_iq}
    w_d = {}                     # D weights: adv 1, gp GP_LAMBDA, cls 1 (losses.py)
    recipe = dict(adv=args.adv, n_critic=n_critic, lr=args.lr, betas=betas, g_weights=w_g,
                  gp_lambda=losses.GP_LAMBDA, headroom=models.HEADROOM, stft_clamp=losses.STFT_CLAMP)

    def real_batch():
        return scaler.normalize(L.real_segments(args.batch, models.SEG_LEN)).to(device)

    def fake_batch():
        z = torch.randn(args.batch, models.Z_DIM, device=device)
        labels = torch.zeros(args.batch, dtype=torch.long, device=device)
        return G(z, labels), labels

    log, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        for _ in range(n_critic):
            real = real_batch()
            with torch.no_grad():
                fake, labels = fake_batch()
            d_total, d_terms = losses.discriminator_loss(D, real, fake, labels, w_d, args.adv)
            optD.zero_grad(set_to_none=True); d_total.backward(); optD.step()

        real = real_batch()
        fake, labels = fake_batch()
        g_total, g_terms = losses.generator_loss(D, real, fake, labels, w_g, args.adv)
        optG.zero_grad(set_to_none=True); g_total.backward(); optG.step()

        if it % args.log_every == 0 or it == 1:
            log.append(dict(it=it, g_total=float(g_total.detach()), d_total=float(d_total.detach()),
                            g=g_terms, d=d_terms))
            print(f"it {it:6d}  G {g_total.item():8.3f}  D {d_total.item():8.3f}  "
                  f"gap {d_terms['score_gap']:7.3f} | adv {g_terms['adv']:.3f} "
                  f"feat {g_terms['feat']:.3f} stft {g_terms['stft']:.3f} "
                  f"iq {g_terms['iq']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"state_dict": G.state_dict(), "n_classes": 1, "seg_len": models.SEG_LEN,
                "scale": scaler.scale, "adv": args.adv, "link": lk.LINK, "iters": args.iters,
                "recipe": recipe},
               os.path.join(ART, f"{args.run}_G.pt"))
    with open(os.path.join(ART, f"{args.run}_losses.json"), "w") as f:
        json.dump(dict(args=vars(args), recipe=recipe, scale=scaler.scale, log=log), f, indent=2)
    loss_figure(log, os.path.join(ART, f"{args.run}_losses.png"))
    print(f"wrote {args.run}_G.pt, {args.run}_losses.json, {args.run}_losses.png to {ART}")


if __name__ == "__main__":
    main()
