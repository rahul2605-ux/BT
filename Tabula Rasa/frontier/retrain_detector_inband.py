"""
Phase 0.5 — Retrain the CNN detector WITH in-band jammer samples.

The sim06 detector was trained on 4 classical jammers, 3 of which are time-domain
broadband operations that spill out of band (barrage, successive-pulse,
protocol-aware); only single-tone is narrowband. Phase 0 showed it therefore
learned an out-of-band-emission detector and is near-blind to spectrally-compliant
in-band interference.

This script augments the training set with in-band jammers (injected in the
frequency domain on the 52 effective subcarriers, then OFDM-modulated) covering
the sparse/broadband/held/hopping/matched families from the frontier sweep, while
KEEPING the 4 classical jammers so the detector doesn't forget them. The question:
does the blind spot close, and at what cost to clean-signal accuracy?

Saves to artifacts/frontier/detector/. Evaluate afterward by re-running
frontier_sweep.py against this checkpoint (see submit_phase05.sh).
"""

import os
import sys
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import ofdm
import train_detector as td
from detector import build_detector, time_signal_to_spectrogram, spectrogram_to_image
from frontier_sweep import build_jam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td.device = device  # classical jammers close over td.device

IN_BAND_STRATEGIES = ["sparse_blind", "sparse_hopping", "broadband", "sparse_matched"]


def to_img(time_sig):
    spec = time_signal_to_spectrogram(time_sig)
    return spectrogram_to_image(spec)


@torch.no_grad()
def make_clean():
    _, _, _, time_sig = ofdm.generate_ofdm_frame(1)
    return to_img(time_sig.squeeze())


@torch.no_grad()
def make_classical():
    _, _, _, time_sig = ofdm.generate_ofdm_frame(1)
    time_sig = time_sig.squeeze()
    jfn = td.JAMMER_TYPES[random.randrange(len(td.JAMMER_TYPES))]
    return to_img(time_sig + jfn(time_sig.shape[0]))


@torch.no_grad()
def make_inband(eff_full):
    _, _, tx_grid, _ = ofdm.generate_ofdm_frame(1)
    strat = random.choice(IN_BAND_STRATEGIES)
    n_active = random.randint(1, eff_full.shape[0])
    power = 10 ** random.uniform(-0.5, 0.9)          # ~0.3 .. 8
    jam_full = build_jam(strat, n_active, power, tx_grid, eff_full, device)
    rx_grid = tx_grid.clone()
    rx_grid[:, 0, 0] = rx_grid[:, 0, 0] + jam_full
    rx_time = ofdm.modulator(rx_grid)
    return to_img(rx_time.squeeze())


class MixedDataset(Dataset):
    """clean (0) + classical-jammed (1) + in-band-jammed (1)."""
    def __init__(self, n_clean, n_classical, n_inband, eff_full):
        self.images, self.labels = [], []

        def add(n, fn, label, tag):
            for i in range(n):
                self.images.append(fn())
                self.labels.append(label)
                if (i + 1) % 100 == 0:
                    print(f"    {tag}: {i+1}/{n}", flush=True)

        print(f"  clean...", flush=True)
        add(n_clean, make_clean, 0, "clean")
        print(f"  classical jammed...", flush=True)
        add(n_classical, make_classical, 1, "classical")
        print(f"  in-band jammed...", flush=True)
        add(n_inband, lambda: make_inband(eff_full), 1, "inband")

        self.images = torch.stack(self.images).cpu()
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


@torch.no_grad()
def eval_inband_dr(model, eff_full, n=128):
    """Detection rate specifically on fresh in-band jammers (the blind spot)."""
    model.eval()
    imgs = torch.stack([make_inband(eff_full) for _ in range(n)]).to(device)
    pred = model(imgs).argmax(1)
    return 100.0 * pred.eq(1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=700)
    ap.add_argument("--n-classical", type=int, default=350)
    ap.add_argument("--n-inband", type=int, default=350)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "frontier", "detector")
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(out_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".pt")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"Phase 0.5 detector retrain — run {run_id} on {device}", flush=True)

    ofdm.init_device(device)
    eff_arr = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(
        eff_arr.cpu().numpy() if hasattr(eff_arr, "cpu") else eff_arr,
        dtype=torch.long, device=device)

    print(f"Building dataset: {args.n_clean} clean + {args.n_classical} classical "
          f"+ {args.n_inband} in-band", flush=True)
    t0 = time.time()
    ds = MixedDataset(args.n_clean, args.n_classical, args.n_inband, eff_full)
    print(f"Dataset built in {time.time()-t0:.1f}s ({len(ds)} samples)", flush=True)

    n_train = int(len(ds) * 0.7)
    train_set, test_set = random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_detector(num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    tl_hist, vl_hist, ta_hist, va_hist = [], [], [], []
    print(f"\nTraining {args.epochs} epochs...\n", flush=True)
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tl, ta = td.train_one_epoch(model, train_loader, criterion, optimizer)
        vm = td.evaluate(model, test_loader, criterion)
        scheduler.step()
        tl_hist.append(tl); ta_hist.append(ta)
        vl_hist.append(vm["loss"]); va_hist.append(vm["accuracy"])
        ib_dr = eval_inband_dr(model, eff_full)
        print(f"Epoch {epoch:3d}/{args.epochs} | val_acc={vm['accuracy']:.1f}% "
              f"DR={vm['dr']:.1f}% FAR={vm['far']:.2f}% | "
              f"in-band DR={ib_dr:.1f}% | {time.time()-t_ep:.1f}s", flush=True)
        if vm["accuracy"] > best_acc:
            best_acc = vm["accuracy"]
            torch.save(model.state_dict(), os.path.join(out_dir, f"run{run_id}_best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, f"run{run_id}_best.pt"),
                                     weights_only=True))
    fm = td.evaluate(model, test_loader, criterion)
    ib_dr = eval_inband_dr(model, eff_full, n=256)
    print(f"\nFinal: acc={fm['accuracy']:.2f}% DR={fm['dr']:.2f}% FAR={fm['far']:.2f}% "
          f"F1={fm['f1']:.4f} | in-band DR={ib_dr:.1f}%", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(va_hist, label="val acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Phase 0.5 detector (clean+classical+in-band) run{run_id}\n"
                 f"final acc={fm['accuracy']:.1f}% FAR={fm['far']:.2f}% in-band DR={ib_dr:.1f}%")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), os.path.join(out_dir, f"run{run_id}_model.pt"))
    print(f"Saved → {out_dir}/run{run_id}_best.pt", flush=True)


if __name__ == "__main__":
    main()
