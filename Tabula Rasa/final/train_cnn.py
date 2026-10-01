# copied from cgan/train_spectrogram_cnn.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment, chain step 1 -- retrain Li et al.'s spectrogram CNN on one
environment's frames and calibrate every detector there (README §3.4).

    sbatch submit_train_cnn.sh base          # usually via submit_env.sh base

Li et al., IEEE Access 10, 2022: EfficientNet-B0, two-class (clean / jammed), SGD lr
1e-3, batch 32, 100 epochs, 762 clean + 204 images per jammer type (barrage, tone, pulse
comb, protocol-aware), 70/30 split, seed 11, received JSR of the jammed frames uniform in
dB over [-20, +10] (Li's own range; at our 15 dB it sits at the noise, §3.3o caveat).
The environment's noise-level factor and fading are on the training frames AND on the
20k calibration frames, so every threshold is CFAR for that environment.

Pruned from cgan/train_spectrogram_cnn.py: the noise LRT, the D6 arms options
(--jsr-range, --extra-gens), shadowing and the refusal to overwrite the deployed
detector (final/ writes only under artifacts/final/<env>/). Step 5 is
defender.calibrate_env, which adds the matched-filter energy detectors.

--seed 12 --name cnn_s12 trains the grey-box attacker's SURROGATE (same recipe, independent
seed, as cgan's run004): the attacker knows the detector's type and recipe, not its weights.

Outputs, artifacts/final/<env>/cnn/ (or --name):
    detector_spec.pt         weights + colour scale (net_scratch, symlinked)
    thresholds.json          every detector's CFAR threshold, clean mean, clean quantiles
    spec_cnn_training.json   loss/accuracy per epoch, validation metrics, ROC,
                             P(det) vs JSR per Li type
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import: no OptiX on the cluster

import argparse
import json
import os
import time

import numpy as np
import torch

import attacks
import defender
import detectors
import env as E
import link as lk

N_CLEAN, N_PER_TYPE, TRAIN_FRACTION = 762, 204, 0.7
JSR_RANGE_DB = (-20.0, 10.0)


def make_frames(L, n, kind, jsr_db, snr_db):
    """n received frames of one kind ('clean' or a Li et al. type) at per-frame JSRs [n]."""
    spec = None if kind == "clean" else dict(name=kind)
    out = attacks.frames(L, n, spec, None if spec is None else jsr_db.reshape(n, 1), snr_db)
    return out["r"]


def batched_frames(L, n, kind, jsr_db, snr_db, batch=512):
    return torch.cat([make_frames(L, min(batch, n - i), kind, jsr_db[i:i + batch], snr_db)
                      for i in range(0, n, batch)])


def dataset(L, gen, snr_db, n_clean=N_CLEAN, n_per_type=N_PER_TYPE):
    """Li et al.'s class balance: 762 clean + 204 per jammer type, JSR uniform in dB."""
    parts = [("clean", n_clean)] + [(t, n_per_type) for t in attacks.LI_TYPES]
    rs, labels, kinds = [], [], []
    for kind, n in parts:
        jsr = torch.empty(n, device=L.device).uniform_(*JSR_RANGE_DB, generator=gen)
        rs.append(batched_frames(L, n, kind, jsr, snr_db))
        labels += [0 if kind == "clean" else 1] * n
        kinds += [kind] * n
    return torch.cat(rs), torch.tensor(labels, device=L.device), np.array(kinds)


def evaluate(net, r, y, scale, batch=128):
    net.eval()
    loss, correct, stats = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, r.shape[0], batch):
            logits = net(detectors.spectrogram_image(r[i:i + batch], scale))
            loss += float(torch.nn.functional.cross_entropy(logits, y[i:i + batch], reduction="sum"))
            correct += int((logits.argmax(1) == y[i:i + batch]).sum())
            stats.append((logits[:, 1] - logits[:, 0]).double())
    return loss / r.shape[0], correct / r.shape[0], torch.cat(stats)


def roc(stat, y, n_points=200):
    """(FPR, TPR) over thresholds at the statistic's quantiles, and the AUC."""
    s, yy = stat.cpu().numpy(), y.cpu().numpy()
    thr = np.unique(np.quantile(s, np.linspace(0, 1, n_points)))[::-1]
    fpr = [float(((s > t) & (yy == 0)).sum() / max(1, (yy == 0).sum())) for t in thr]
    tpr = [float(((s > t) & (yy == 1)).sum() / max(1, (yy == 1).sum())) for t in thr]
    fpr, tpr = [0.0] + fpr + [1.0], [0.0] + tpr + [1.0]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(E.ENVS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-cal", type=int, default=defender.N_CALIBRATION)
    ap.add_argument("--name", default="cnn",
                    help="folder under artifacts/final/<env>/; cnn_s12 = the grey-box attacker's surrogate")
    args = ap.parse_args()
    if args.name == "cnn" and args.seed != 11:
        raise SystemExit("refusing to write a non-default seed over the deployed CNN: pass --name")
    if E.SMOKE:
        args.epochs, args.n_cal = 2, 2048
    out_dir = E.art(args.env, args.name)
    os.makedirs(out_dir, exist_ok=True)

    device = lk.setup(seed=args.seed)
    L = E.apply(lk.Link(**lk.LINK), args.env)
    snr = E.SNR_DB
    gen = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.time()
    print(f"env {args.env} = {E.ENVS[args.env]} | SNR {snr:g} dB (Es/N0 {snr + E.ESN0_OFFSET_DB:.2f} dB) "
          f"| device {device} | {'SMOKE' if E.SMOKE else 'full'}", flush=True)

    # 1. the fixed colour scale, from clean frames of this environment only
    scale = detectors.SpecScale.fit(batched_frames(L, 2048, "clean", torch.zeros(2048, device=device), snr))
    print(f"spectrogram scale: vmin {scale.vmin:.2f} dB, vmax {scale.vmax:.2f} dB", flush=True)

    # 2. Li et al.'s dataset and split
    k = 8 if E.SMOKE else 1
    r, y, kinds = dataset(L, gen, snr, N_CLEAN // k, N_PER_TYPE // k)
    perm = torch.randperm(r.shape[0], generator=gen, device=device)
    n_train = int(round(TRAIN_FRACTION * r.shape[0]))
    tr, va = perm[:n_train], perm[n_train:]
    print(f"dataset: {r.shape[0]} frames ({n_train} train / {len(va)} val), {time.time() - t0:.0f}s", flush=True)

    # 3. train exactly as Li et al. (Table 7, EfficientNet-B0)
    net = detectors.build_cnn(pretrained=True).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.0)
    history = []
    for ep in range(args.epochs):
        net.train()
        order = tr[torch.randperm(len(tr), generator=gen, device=device)]
        tl, tc = 0.0, 0
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            logits = net(detectors.spectrogram_image(r[idx], scale))
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss) * len(idx)
            tc += int((logits.argmax(1) == y[idx]).sum())
        vl, vacc, _ = evaluate(net, r[va], y[va], scale)
        history.append(dict(epoch=ep + 1, train_loss=tl / len(tr), train_acc=tc / len(tr),
                            val_loss=vl, val_acc=vacc))
        print(f"epoch {ep + 1:3d}  train loss {tl / len(tr):.4f} acc {tc / len(tr):.4f} | "
              f"val loss {vl:.4f} acc {vacc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # 4. validation metrics in the paper's terms (argmax decisions)
    _, vacc, vstat = evaluate(net, r[va], y[va], scale)
    pred = (vstat > 0).long()
    yv = y[va]
    far_argmax = float(((pred == 1) & (yv == 0)).sum() / (yv == 0).sum())
    dr_jammed = float(((pred == 1) & (yv == 1)).sum() / (yv == 1).sum())
    fpr, tpr, auc = roc(vstat, yv)
    va_kinds = kinds[va.cpu().numpy()]
    per_type_val = {kk: float((pred[torch.as_tensor(va_kinds == kk, device=device)] ==
                               (0 if kk == "clean" else 1)).double().mean())
                    for kk in ["clean"] + attacks.LI_TYPES}
    del r

    # 5. every detector's CFAR threshold, on fresh clean frames of this environment
    dets = defender.env_dets(E.faded(args.env))
    thr = defender.calibrate_env(L, net, scale, snr, dets, n_cal=args.n_cal)
    thr.update(env=args.env, jsr_range_db=list(JSR_RANGE_DB), seed=args.seed, epochs=args.epochs)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(thr, f, indent=2)

    # 6. P(det) vs JSR per Li type at alpha 0.05 (the transition the bands are read against)
    thr05 = thr["spec_cnn"]["0.05"]
    grid = list(np.arange(-30.0, 10.5, 2.0))
    n_g = 64 if E.SMOKE else 256
    pdet_vs_jsr = {kind: [detectors.p_detect(detectors.cnn_statistic(
        net, batched_frames(L, n_g, kind, torch.full((n_g,), g, device=device), snr), scale), thr05)
        for g in grid] for kind in attacks.LI_TYPES}

    E.save_big(dict(state_dict=net.state_dict(), scale=scale.to_dict(), env=args.env,
                    epochs=args.epochs, seed=args.seed, history=history),
               args.env, os.path.join(args.name, "detector_spec.pt"))
    report = dict(
        env=args.env, env_config=E.ENVS[args.env],
        config=dict(model="EfficientNet-B0 (torchvision, ImageNet init)", optimiser="SGD", lr=1e-3,
                    momentum=0.0, batch=32, epochs=args.epochs, n_clean=N_CLEAN // k,
                    n_per_type=N_PER_TYPE // k, train_fraction=TRAIN_FRACTION,
                    jsr_range_db=list(JSR_RANGE_DB), types=attacks.LI_TYPES, snr_db=snr, seed=args.seed),
        history=history,
        validation=dict(accuracy=vacc, detection_rate_jammed=dr_jammed, far_argmax=far_argmax,
                        auc=auc, per_type_correct=per_type_val, roc_fpr=fpr, roc_tpr=tpr),
        pdet_vs_jsr=dict(jsr_db=grid, alpha=0.05, per_type=pdet_vs_jsr),
        runtime_s=time.time() - t0)
    with open(os.path.join(out_dir, "spec_cnn_training.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Li et al. spectrogram CNN, retrained on env {args.env} ---")
    print(f"validation accuracy {vacc:.4f} (S4/D6 15 dB r0: 0.932); detection rate on jammed "
          f"{dr_jammed:.4f}, FAR (argmax) {far_argmax:.4f}, AUC {auc:.4f} (r0: 0.978)")
    print(f"per type (val): {per_type_val}")
    print(f"P(det) at alpha 0.05 vs JSR {grid}:")
    for kk, v in pdet_vs_jsr.items():
        print(f"  {kk:<15} " + " ".join(f"{p:.2f}" for p in v))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
