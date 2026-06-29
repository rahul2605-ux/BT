"""
sim06 Phase 1 — Train CNN spectrogram detector on classical jammers over OFDM.

Replicates Li et al. (IEEE Access 2022) methodology:
  - EfficientNet-B0, SGD lr=0.001, batch=32, 100 epochs
  - Binary classification: clean (0) vs jammed (1)
  - Spectrograms from OFDM time-domain signals

The trained detector is FROZEN during Phase 2 (MAPPO jammer training).
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ofdm
from detector import (build_detector, time_signal_to_spectrogram,
                       spectrogram_to_image, IMG_SIZE)

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

# ── Hyperparameters (Li et al. Table 7, EfficientNet-B0) ────────────────────
BATCH_SIZE  = 32
LR          = 0.001
EPOCHS      = 100
MOMENTUM    = 0.9
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.7

N_CLEAN       = 762
N_PER_JAMMER  = 204

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Classical jammer generators (applied in time domain) ────────────────────

def jam_barrage(n_samples, power_range=(0.5, 5.0)):
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    return torch.randn(n_samples, dtype=torch.complex64, device=device) * np.sqrt(power / 2)


def jam_single_tone(n_samples, power_range=(1.0, 8.0)):
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    t = torch.arange(n_samples, device=device, dtype=torch.float32)
    freq = torch.empty(1, device=device).uniform_(-0.1, 0.1).item()
    phase = torch.empty(1, device=device).uniform_(0, 2 * np.pi).item()
    return np.sqrt(power) * torch.exp(1j * (2 * np.pi * freq * t + phase))


def jam_successive_pulse(n_samples, power_range=(1.0, 6.0), n_pulses=64):
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    jam = torch.zeros(n_samples, dtype=torch.complex64, device=device)
    spacing = max(1, n_samples // n_pulses)
    indices = torch.arange(0, n_samples, spacing, device=device)
    phase = torch.empty(len(indices), device=device).uniform_(0, 2 * np.pi)
    jam[indices] = np.sqrt(power * n_pulses) * torch.exp(1j * phase)
    return jam


def jam_protocol_aware(n_samples, power_range=(0.3, 2.0)):
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    mask = torch.rand(n_samples, device=device) < 0.15
    n_active = max(mask.sum().item(), 1)
    noise = torch.randn(n_samples, dtype=torch.complex64, device=device) * np.sqrt(power / n_active)
    jam = torch.zeros(n_samples, dtype=torch.complex64, device=device)
    jam[mask] = noise[mask]
    return jam


JAMMER_TYPES = [jam_barrage, jam_single_tone, jam_successive_pulse, jam_protocol_aware]
JAMMER_NAMES = ['Barrage', 'Single-tone', 'Successive-pulse', 'Protocol-aware']


# ── Dataset ──────────────────────────────────────────────────────────────────

def generate_sample(is_jammed):
    """Generate one spectrogram sample from OFDM signal."""
    with torch.no_grad():
        _, _, _, time_sig = ofdm.generate_ofdm_frame(1)
        time_sig = time_sig.squeeze()  # [T] complex

        if is_jammed:
            jfn = JAMMER_TYPES[torch.randint(0, len(JAMMER_TYPES), (1,)).item()]
            jam = jfn(time_sig.shape[0])
            time_sig = time_sig + jam

        spec_db = time_signal_to_spectrogram(time_sig)
        img = spectrogram_to_image(spec_db)
    return img


class OFDMSpectrogramDataset(Dataset):
    def __init__(self, n_clean, n_jammed):
        self.images = []
        self.labels = []

        print(f"  Generating {n_clean} clean OFDM samples...", flush=True)
        for i in range(n_clean):
            self.images.append(generate_sample(is_jammed=False))
            self.labels.append(0)
            if (i + 1) % 100 == 0:
                print(f"    clean: {i+1}/{n_clean}", flush=True)

        print(f"  Generating {n_jammed} jammed OFDM samples...", flush=True)
        for i in range(n_jammed):
            self.images.append(generate_sample(is_jammed=True))
            self.labels.append(1)
            if (i + 1) % 100 == 0:
                print(f"    jammed: {i+1}/{n_jammed}", flush=True)

        self.images = torch.stack(self.images).cpu()
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ── Training / evaluation ───────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = tn = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        pred = out.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()

    acc = 100.0 * correct / total
    dr = 100.0 * tp / max(tp + fn, 1)
    far = 100.0 * fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {'loss': total_loss / total, 'accuracy': acc, 'dr': dr, 'far': far,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_training_plots(tl, ta, vl, va, run_id, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(tl, label='Train'); axes[0].plot(vl, label='Val')
    axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'sim06 CNN Detector — run{run_id}\n'
                      f'EfficientNet-B0 binary, SGD lr={LR}, OFDM 64-SC')
    axes[1].plot(ta, label='Train'); axes[1].plot(va, label='Val')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close()


def save_confusion_matrix(m, run_id, out_dir):
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Greens')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Clean','Jammed']); ax.set_yticklabels(['Clean','Jammed'])
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=14)
    ax.set_title(f'sim06 Confusion Matrix — run{run_id}\n'
                 f'DR={m["dr"]:.1f}% FAR={m["far"]:.2f}% F1={m["f1"]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}_cm.png"), dpi=150)
    plt.close()


def save_sample_spectrograms(dataset, run_id, out_dir, n=8):
    fig, axes = plt.subplots(2, n // 2, figsize=(16, 6))
    labels_text = {0: 'Clean', 1: 'Jammed'}
    indices = torch.randperm(len(dataset))[:n]
    for i, idx in enumerate(indices):
        img, label = dataset[idx.item()]
        ax = axes[i // (n // 2), i % (n // 2)]
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.set_title(labels_text[int(label)], fontsize=10)
        ax.axis('off')
    plt.suptitle(f'sim06 OFDM Spectrograms — run{run_id}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}_samples.png"), dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clean', type=int, default=N_CLEAN)
    parser.add_argument('--n-per-jammer', type=int, default=N_PER_JAMMER)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim06", "detector")
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(out_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"Run {run_id}", flush=True)

    ofdm.init_device(device)

    # Generate dataset
    n_jammed = args.n_per_jammer * len(JAMMER_TYPES)
    print(f"Generating dataset: {args.n_clean} clean + {n_jammed} jammed = "
          f"{args.n_clean + n_jammed} total", flush=True)
    t0 = time.time()
    dataset = OFDMSpectrogramDataset(args.n_clean, n_jammed)
    print(f"Dataset generated in {time.time() - t0:.1f}s", flush=True)

    n_train = int(len(dataset) * TRAIN_RATIO)
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    print(f"Train: {n_train}, Test: {n_test}", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    save_sample_spectrograms(dataset, run_id, out_dir)

    model = build_detector(num_classes=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0: {params:,} params", flush=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_acc = 0.0

    print(f"\nTraining {args.epochs} epochs...\n", flush=True)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vm = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(tl); train_accs.append(ta)
        val_losses.append(vm['loss']); val_accs.append(vm['accuracy'])

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={tl:.4f} train_acc={ta:.1f}% | "
              f"val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.1f}% "
              f"DR={vm['dr']:.1f}% FAR={vm['far']:.2f}% | "
              f"{time.time()-t_ep:.1f}s", flush=True)

        if vm['accuracy'] > best_val_acc:
            best_val_acc = vm['accuracy']
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"run{run_id}_best.pt"))

    print(f"\nDone in {time.time()-t_start:.1f}s", flush=True)

    model.load_state_dict(torch.load(os.path.join(out_dir, f"run{run_id}_best.pt"),
                                      weights_only=True))
    fm = evaluate(model, test_loader, criterion)
    print(f"\nFinal: acc={fm['accuracy']:.2f}% DR={fm['dr']:.2f}% "
          f"FAR={fm['far']:.2f}% F1={fm['f1']:.4f}", flush=True)

    save_training_plots(train_losses, train_accs, val_losses, val_accs, run_id, out_dir)
    save_confusion_matrix(fm, run_id, out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, f"run{run_id}_model.pt"))
    print(f"Artifacts → {out_dir}/run{run_id}_*", flush=True)


if __name__ == "__main__":
    main()
