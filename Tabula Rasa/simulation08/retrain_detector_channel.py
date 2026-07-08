"""
sim08 milestone 2 — channel-valid CNN detector.

Retrains the EfficientNet-B0 complex-STFT spectrogram detector on the REALISTIC
channel (frequency-selective TDL fading + AWGN at a range of Eb/N0), so its
clean-vs-jammed decision boundary is valid on faded signals. The lossless-trained
sim06 detector has ~13% FAR on faded clean frames with no clean/jammed separation
(see frontier_channel.py note) and is invalid here. This produces the detector the
full-suite (CNN + energy) frontier on the realistic channel needs — the paper's
core figure.

Dataset — every frame is passed through MultiLinkChannel BEFORE the spectrogram:
  clean      (0) — faded + noisy, no jammer
  classical  (1) — faded clean + a classical time-domain jammer added at the RX
                   (barrage / single-tone / successive-pulse / protocol-aware =
                   the out-of-band emitters the sim06 detector already caught)
  in-band    (1) — faded clean + an in-band freq-domain jammer through its OWN
                   channel h_jam (sparse blind / sparse channel-aware / broadband
                   in-band) — the spectrally-compliant interference the lossless
                   detector was blind to.

Eb/N0 is drawn uniformly in [--ebno-min, --ebno-max] per mini-batch so a single
CNN is valid across the SNRs the frontier evaluates (5..30 dB).

Caveat (documented refinement, matches Phase 0.5): in-band samples are labeled
"jammed" whenever a jammer is present, even when it barely raises BER — this
inflates FAR relative to a BER-thresholded labeling. Kept for parity with the
Phase 0.5 study; BER-thresholded labels are a follow-up refinement.

Saves to artifacts/sim08/detector/. Evaluate by running frontier_channel.py against
the checkpoint (submit.sh does both in sequence).
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import sionna.phy as sn
import ofdm
import train_detector as td
from detector import (build_detector, time_signal_to_spectrogram,
                      spectrogram_to_image)
from channel import MultiLinkChannel
from frontier_channel import build_jam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td.device = device  # classical jammers close over td.device

IN_BAND_STRATEGIES = ["sparse_blind", "sparse_channelaware", "broadband_inband"]


def ebnodb2no(ebno_db):
    return float(sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=2,
                                    coderate=1.0, resource_grid=ofdm.resource_grid))


@torch.no_grad()
def to_img_batch(time_sig):
    """[B, T] complex time signal -> [B, 3, 224, 224] spectrogram images (on CPU)."""
    spec = time_signal_to_spectrogram(time_sig)
    return spectrogram_to_image(spec).cpu()


@torch.no_grad()
def faded_clean_time(chan, tx_grid, h_tx, h_jam, no):
    """Faded + noisy clean time signal [B, T] (no jammer)."""
    B = tx_grid.shape[0]
    zero = torch.zeros(B, ofdm.N_OFDM_SYMBOLS, ofdm.FFT_SIZE,
                       dtype=torch.complex64, device=device)
    rx_grid = chan.apply(tx_grid, [zero], h_tx, h_jam, no)
    return ofdm.modulator(rx_grid)[:, 0, 0]


@torch.no_grad()
def gen_clean(chan, B, ebno):
    _, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    h_tx, h_jam = chan.sample(B)
    t = faded_clean_time(chan, tx_grid, h_tx, h_jam, ebnodb2no(ebno))
    return to_img_batch(t)


@torch.no_grad()
def gen_classical(chan, B, ebno):
    _, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    h_tx, h_jam = chan.sample(B)
    t = faded_clean_time(chan, tx_grid, h_tx, h_jam, ebnodb2no(ebno))   # [B, T]
    T = t.shape[1]
    jam = torch.stack([td.JAMMER_TYPES[random.randrange(len(td.JAMMER_TYPES))](T)
                       for _ in range(B)])                              # [B, T]
    return to_img_batch(t + jam)


@torch.no_grad()
def gen_inband(chan, B, ebno, eff_full):
    _, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    h_tx, h_jam = chan.sample(B)
    strat = random.choice(IN_BAND_STRATEGIES)
    n_active = random.randint(1, eff_full.shape[0])
    power = 10 ** random.uniform(-0.5, 0.9)                             # ~0.3 .. 8
    jam = build_jam(strat, n_active, power, tx_grid, h_tx, h_jam, eff_full, device)
    rx_grid = chan.apply(tx_grid, [jam], h_tx, h_jam, ebnodb2no(ebno))
    return to_img_batch(ofdm.modulator(rx_grid)[:, 0, 0])


def build_dataset(chan, eff_full, n_clean, n_classical, n_inband,
                  ebno_min, ebno_max):
    imgs, labels = [], []

    def run(n, bs, fn, label, tag):
        done = 0
        while done < n:
            b = min(bs, n - done)
            ebno = random.uniform(ebno_min, ebno_max)
            imgs.append(fn(b, ebno))
            labels.extend([label] * b)
            done += b
            if done % (bs * 4) == 0 or done == n:
                print(f"    {tag}: {done}/{n}", flush=True)

    print("  clean (faded+noise)...", flush=True)
    run(n_clean, 64, lambda b, e: gen_clean(chan, b, e), 0, "clean")
    print("  classical jammed (faded)...", flush=True)
    run(n_classical, 64, lambda b, e: gen_classical(chan, b, e), 1, "classical")
    print("  in-band jammed (faded, through h_jam)...", flush=True)
    run(n_inband, 16, lambda b, e: gen_inband(chan, b, e, eff_full), 1, "inband")

    images = torch.cat(imgs, 0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


@torch.no_grad()
def eval_faded(model, chan, eff_full, ebno, n=128):
    """Fresh faded samples at one SNR: clean FAR and in-band DR."""
    model.eval()
    clean, inband = [], []
    d = 0
    while d < n:
        b = min(64, n - d); clean.append(gen_clean(chan, b, ebno)); d += b
    d = 0
    while d < n:
        b = min(16, n - d); inband.append(gen_inband(chan, b, ebno, eff_full)); d += b
    ci = torch.cat(clean).to(device)
    ii = torch.cat(inband).to(device)
    far = 100.0 * model(ci).argmax(1).eq(1).float().mean().item()
    dr = 100.0 * model(ii).argmax(1).eq(1).float().mean().item()
    return far, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=1200)
    ap.add_argument("--n-classical", type=int, default=400)
    ap.add_argument("--n-inband", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--ebno-min", type=float, default=5.0)
    ap.add_argument("--ebno-max", type=float, default=30.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_clean, args.n_classical, args.n_inband = 48, 24, 24
        args.epochs = 2

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim08", "detector")
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(out_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".pt")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"sim08 m2 channel-valid detector — run {run_id} on {device}", flush=True)

    ofdm.init_device(device)
    eff_arr = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(
        eff_arr.cpu().numpy() if hasattr(eff_arr, "cpu") else eff_arr,
        dtype=torch.long, device=device)

    chan = MultiLinkChannel(ofdm.resource_grid, n_jammers=1, device=device,
                            model="C", delay_spread=100e-9, carrier_freq=5.2e9)

    print(f"Building dataset: {args.n_clean} clean + {args.n_classical} classical "
          f"+ {args.n_inband} in-band, Eb/N0∈[{args.ebno_min:g},{args.ebno_max:g}] dB",
          flush=True)
    t0 = time.time()
    images, labels = build_dataset(chan, eff_full, args.n_clean, args.n_classical,
                                   args.n_inband, args.ebno_min, args.ebno_max)
    print(f"Dataset built in {time.time()-t0:.1f}s "
          f"({len(labels)} samples, {int(labels.eq(0).sum())} clean / "
          f"{int(labels.eq(1).sum())} jammed)", flush=True)

    ds = TensorDataset(images, labels)
    n_train = int(len(ds) * 0.7)
    train_set, test_set = random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_detector(num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ebno_grid = [args.ebno_min, (args.ebno_min + args.ebno_max) / 2, args.ebno_max]
    best_acc, va_hist = 0.0, []
    print(f"\nTraining {args.epochs} epochs...\n", flush=True)
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tl, ta = td.train_one_epoch(model, train_loader, criterion, optimizer)
        vm = td.evaluate(model, test_loader, criterion)
        scheduler.step()
        va_hist.append(vm["accuracy"])
        print(f"Epoch {epoch:3d}/{args.epochs} | val_acc={vm['accuracy']:.1f}% "
              f"DR={vm['dr']:.1f}% FAR={vm['far']:.2f}% | {time.time()-t_ep:.1f}s",
              flush=True)
        if vm["accuracy"] > best_acc:
            best_acc = vm["accuracy"]
            torch.save(model.state_dict(), os.path.join(out_dir, f"run{run_id}_best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, f"run{run_id}_best.pt"),
                                     weights_only=True))
    fm = td.evaluate(model, test_loader, criterion)
    print(f"\nFinal (held-out mix): acc={fm['accuracy']:.2f}% DR={fm['dr']:.2f}% "
          f"FAR={fm['far']:.2f}% F1={fm['f1']:.4f}", flush=True)

    print("\n=== per-SNR on FRESH faded samples (clean FAR / in-band DR) ===", flush=True)
    snr_far, snr_dr = [], []
    for e in ebno_grid:
        far, dr = eval_faded(model, chan, eff_full, e, n=192 if not args.smoke else 32)
        snr_far.append(far); snr_dr.append(dr)
        print(f"  Eb/N0={e:>4g} dB | clean FAR={far:5.1f}% | in-band DR={dr:5.1f}%",
              flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, len(va_hist) + 1), va_hist, "-o", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val accuracy (%)")
    axes[0].set_title(f"channel-valid detector run{run_id}\n"
                      f"held-out mix acc={fm['accuracy']:.1f}% FAR={fm['far']:.2f}%")
    axes[0].grid(alpha=0.3)
    x = np.arange(len(ebno_grid))
    axes[1].bar(x - 0.2, snr_far, 0.4, label="clean FAR", color="steelblue")
    axes[1].bar(x + 0.2, snr_dr, 0.4, label="in-band DR", color="crimson")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{e:g}" for e in ebno_grid])
    axes[1].set_xlabel("Eb/N0 (dB)"); axes[1].set_ylabel("%")
    axes[1].set_title("faded generalization (fresh samples)")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), os.path.join(out_dir, f"run{run_id}_model.pt"))
    print(f"\nSaved → {out_dir}/run{run_id}_best.pt", flush=True)


if __name__ == "__main__":
    main()
