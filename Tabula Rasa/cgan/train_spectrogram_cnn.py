"""
Baselines -- retrain Li et al.'s spectrogram CNN on this link, and calibrate every
detector's threshold (README §2.10).

    sbatch submit_train_spectrogram_cnn.sh            # -> runs/spec_cnn_<JOBID>.out

Li et al., IEEE Access 10, 2022 (source_papers/Li 2025.pdf -- the filename's year
is wrong): EfficientNet-B0, two-class (clean / jammed), SGD lr 1e-3, batch 32,
100 epochs, 762 clean + 204 images per jammer type, 70/30 split. What differs,
and why, is the table in README §2.10; in short: single-carrier adaptations of
their four jammer types (attacks.LI_TYPES), a fixed-dB spectrogram of one
received frame instead of a GNU Radio waterfall screenshot, and received JSR
drawn uniformly in dB over [-20, +10] -- their jammer sat 0.5-1.5 m from the
receiver, far above anything a stealth study cares about.

Outputs, in artifacts/cgan/baselines/:
    detector_spec.pt         weights, colour scale, CNN thresholds, training record
    thresholds.json          every detector's clean mean and thresholds at each alpha
    clean_lrt_parts.pt       the noise-LRT's sufficient statistics on the same clean
                             frames (its threshold depends on the jammer power, so it
                             is recomputed per sweep point)
    spec_cnn_training.json   loss/accuracy per epoch, validation metrics vs the paper,
                             ROC, accuracy at high JSR, P(det) vs JSR per jammer type
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import: no OptiX on the cluster (scene.py)

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import attacks
import detectors
import link as lk
import scene

PAPER = dict(two_class_dr=100.0, two_class_va=99.91, five_class_dr=99.79, far_weighted=0.03)
N_CLEAN, N_PER_TYPE, TRAIN_FRACTION = 762, 204, 0.7
JSR_RANGE_DB = (-20.0, 10.0)
N_CALIBRATION = 20_000


def make_frames(L, n, kind, jsr_db):
    """n received frames of one kind ('clean' or a Li et al. type) at per-frame JSRs [n]."""
    spec = None if kind == "clean" else dict(name=kind)
    out = attacks.frames(L, n, spec, None if spec is None else jsr_db.reshape(n, 1), lk.SNR_DB)
    return out["r"]


def batched_frames(L, n, kind, jsr_db, batch=512):
    return torch.cat([make_frames(L, min(batch, n - i), kind, jsr_db[i:i + batch])
                      for i in range(0, n, batch)])


def dataset(L, gen):
    """Li et al.'s class balance: 762 clean + 204 per jammer type, JSR uniform in dB."""
    rs, labels, kinds, jsrs = [], [], [], []
    for kind, n in [("clean", N_CLEAN)] + [(t, N_PER_TYPE) for t in attacks.LI_TYPES]:
        jsr = torch.empty(n, device=L.device).uniform_(*JSR_RANGE_DB, generator=gen)
        rs.append(batched_frames(L, n, kind, jsr))
        labels += [0 if kind == "clean" else 1] * n
        kinds += [kind] * n
        jsrs += (jsr.tolist() if kind != "clean" else [float("nan")] * n)
    return torch.cat(rs), torch.tensor(labels, device=L.device), np.array(kinds), np.array(jsrs)


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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(scene.ART, exist_ok=True)

    device = lk.setup(seed=args.seed)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    gen = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.time()

    # 1. the fixed colour scale, from clean frames only
    scale = detectors.SpecScale.fit(batched_frames(L, 2048, "clean", torch.zeros(2048, device=device)))
    print(f"spectrogram scale: vmin {scale.vmin:.2f} dB, vmax {scale.vmax:.2f} dB", flush=True)

    # 2. Li et al.'s dataset and split
    r, y, kinds, jsrs = dataset(L, gen)
    perm = torch.randperm(r.shape[0], generator=gen, device=device)
    n_train = int(round(TRAIN_FRACTION * r.shape[0]))
    tr, va = perm[:n_train], perm[n_train:]
    print(f"dataset: {r.shape[0]} frames ({n_train} train / {len(va)} val), "
          f"{time.time() - t0:.0f}s", flush=True)

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

    # 4. validation metrics in the paper's terms (argmax decisions)
    _, vacc, vstat = evaluate(net, r[va], y[va], scale)
    pred = (vstat > 0).long()
    yv = y[va]
    far_argmax = float(((pred == 1) & (yv == 0)).sum() / (yv == 0).sum())
    dr_jammed = float(((pred == 1) & (yv == 1)).sum() / (yv == 1).sum())
    fpr, tpr, auc = roc(vstat, yv)
    va_kinds = kinds[va.cpu().numpy()]
    per_type_val = {k: float((pred[torch.as_tensor(va_kinds == k, device=device)] ==
                              (0 if k == "clean" else 1)).double().mean())
                    for k in ["clean"] + attacks.LI_TYPES}

    # 5. every detector's threshold, on N_CALIBRATION fresh clean frames
    stat = dict(power=[], kurtosis=[], cnn=[], z=[], e_perp=[])
    for i in range(0, N_CALIBRATION, 1024):
        n = min(1024, N_CALIBRATION - i)
        out = attacks.frames(L, n, None, None, lk.SNR_DB)
        s = detectors.statistics(out["r"], out["z"], net, scale)
        stat["power"].append(s["power"]); stat["kurtosis"].append(s["kurtosis"])
        stat["cnn"].append(s["cnn"])
        stat["z"].append(s["lrt"][0].to(torch.complex64).cpu()); stat["e_perp"].append(s["lrt"][1].cpu())
    D = s["lrt"][2]
    stat = {k: torch.cat(v) for k, v in stat.items()}
    pm, km = float(stat["power"].mean()), float(stat["kurtosis"].mean())
    thresholds = dict(
        n_clean=N_CALIBRATION, alphas=detectors.ALPHAS, power_clean_mean=pm, kurtosis_clean_mean=km,
        power_one_sided={str(a): detectors.calibrate(stat["power"], a) for a in detectors.ALPHAS},
        power_two_sided={str(a): detectors.calibrate(detectors.two_sided(stat["power"], pm), a)
                         for a in detectors.ALPHAS},
        kurtosis={str(a): detectors.calibrate(detectors.two_sided(stat["kurtosis"], km), a)
                  for a in detectors.ALPHAS},
        cnn={str(a): detectors.calibrate(stat["cnn"], a) for a in detectors.ALPHAS},
        lrt_d=D, noise_var=L.noise_var(lk.SNR_DB), p_s=L.p_s, spec_scale=scale.to_dict(),
    )
    torch.save(dict(z=stat["z"], e_perp=stat["e_perp"], D=D), os.path.join(scene.ART, "clean_lrt_parts.pt"))
    with open(os.path.join(scene.ART, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # 6. reproduction check at high JSR (the paper's regime), and P(det) vs JSR per type
    thr05 = thresholds["cnn"]["0.05"]
    hi = {}
    for kind, n in [("clean", N_CLEAN)] + [(t, N_PER_TYPE) for t in attacks.LI_TYPES]:
        rr = batched_frames(L, n, kind, torch.full((n,), 10.0, device=device))
        s = detectors.cnn_statistic(net, rr, scale)
        hi[kind] = dict(argmax_correct=float(((s > 0) == (kind != "clean")).double().mean()),
                        p_det_at_alpha05=detectors.p_detect(s, thr05))
    n_hi = N_CLEAN + 4 * N_PER_TYPE
    acc_hi = (hi["clean"]["argmax_correct"] * N_CLEAN
              + sum(hi[t]["argmax_correct"] * N_PER_TYPE for t in attacks.LI_TYPES)) / n_hi
    grid = list(np.arange(-30.0, 10.5, 2.0))
    pdet_vs_jsr = {}
    for kind in attacks.LI_TYPES:
        pdet_vs_jsr[kind] = [detectors.p_detect(
            detectors.cnn_statistic(net, batched_frames(L, 256, kind, torch.full((256,), g, device=device)),
                                    scale), thr05) for g in grid]

    torch.save(dict(state_dict=net.state_dict(), scale=scale.to_dict(), thresholds=thresholds,
                    epochs=args.epochs, seed=args.seed, history=history),
               os.path.join(scene.ART, "detector_spec.pt"))
    report = dict(
        paper=PAPER, config=dict(model="EfficientNet-B0 (torchvision, ImageNet init)", optimiser="SGD",
                                 lr=1e-3, momentum=0.0, batch=32, epochs=args.epochs,
                                 n_clean=N_CLEAN, n_per_type=N_PER_TYPE, train_fraction=TRAIN_FRACTION,
                                 jsr_range_db=JSR_RANGE_DB, types=attacks.LI_TYPES,
                                 image=[detectors.IMG_H, detectors.IMG_W], n_fft=detectors.N_FFT,
                                 hop=detectors.HOP, headroom_db=detectors.HEADROOM_DB),
        history=history,
        validation=dict(accuracy=vacc, detection_rate_jammed=dr_jammed, far_argmax=far_argmax,
                        auc=auc, per_type_correct=per_type_val, roc_fpr=fpr, roc_tpr=tpr),
        high_jsr_10db=dict(accuracy=acc_hi, per_type=hi),
        pdet_vs_jsr=dict(jsr_db=grid, alpha=0.05, per_type=pdet_vs_jsr),
        thresholds=thresholds, runtime_s=time.time() - t0,
    )
    with open(os.path.join(scene.ART, "spec_cnn_training.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Li et al. spectrogram CNN, retrained ---")
    print(f"validation accuracy {vacc:.4f} (paper two-class VA {PAPER['two_class_va']} %, "
          f"DR {PAPER['two_class_dr']} %)")
    print(f"validation: detection rate on jammed {dr_jammed:.4f}, FAR (argmax) {far_argmax:.4f}, AUC {auc:.4f}")
    print(f"per type (val): {per_type_val}")
    print(f"at JSR +10 dB: accuracy {acc_hi:.4f}; {hi}")
    print(f"P(det) at alpha 0.05 vs JSR {grid}:")
    for k, v in pdet_vs_jsr.items():
        print(f"  {k:<15} " + " ".join(f"{p:.2f}" for p in v))
    print(f"thresholds: {json.dumps({k: v for k, v in thresholds.items() if k != 'spec_scale'})}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
