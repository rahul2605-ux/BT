"""
sim05 — CNN spectrogram detector (EfficientNet-B0)

Replicates Li et al. (IEEE Access 2022) detector methodology:
  - EfficientNet-B0 on spectrogram images (their best: 99.79% DR, 0.03% FAR)
  - SGD optimizer, LR=0.001, batch_size=32, 100 epochs
  - Image size: 224x224x3 (EfficientNet-B0 native input size)

Adaptations for our setup:
  - Binary classification (clean vs jammed) instead of 5-class
  - Synthetic data generated from our QPSK channel (not SDR captures)
  - 4 classical jammer types collapsed into "jammed" label
  - PyTorch (our stack) instead of TensorFlow/Keras
  - Spectrograms generated via torch.stft from raw IQ samples

Reference: Li et al., "Jamming Detection and Classification in OFDM-Based
UAVs via Feature- and Spectrogram-Tailored Machine Learning", IEEE Access,
vol. 10, pp. 16859-16870, 2022. DOI: 10.1109/ACCESS.2022.3150020
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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

# ── Hyperparameters (matching Li et al. Table 7 for EfficientNet-B0) ─────────
IMG_SIZE       = 224          # EfficientNet-B0 native input
N_SUBCARRIERS  = 64           # 802.11a OFDM
N_SYMBOLS_TIME = 128          # OFDM symbols per spectrogram window
SAMPLE_RATE    = 20e6         # 20 MHz (802.11a)
N_FFT          = 128          # STFT FFT size
HOP_LENGTH     = 32           # STFT hop
BATCH_SIZE     = 32           # Li et al. Table 7
LR             = 0.001        # Li et al. Table 7
EPOCHS         = 100          # Li et al. Table 7
MOMENTUM       = 0.9          # standard SGD
WEIGHT_DECAY   = 1e-4

# Dataset generation
N_CLEAN        = 762          # matching Li et al. clean count
N_PER_JAMMER   = 204          # matching Li et al. per-jammer count (×4 types)
TRAIN_RATIO    = 0.7          # Li et al.: 70/30 split

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── QPSK constellation ──────────────────────────────────────────────────────
QPSK_POINTS = torch.tensor(
    [1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64
) / np.sqrt(2)
QPSK_POINTS = QPSK_POINTS.to(device)


def generate_qpsk_signal(n_symbols):
    """Generate a sequence of n_symbols random QPSK symbols."""
    idx = torch.randint(0, 4, (n_symbols,), device=device)
    return QPSK_POINTS[idx]


# ── Jamming signal generators ────────────────────────────────────────────────

def jam_barrage(n_symbols, power_range=(0.5, 5.0)):
    """Broadband Gaussian noise jamming."""
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    noise = torch.randn(n_symbols, dtype=torch.complex64, device=device)
    return noise * np.sqrt(power / 2)


def jam_single_tone(n_symbols, power_range=(1.0, 8.0)):
    """Single-tone CW interference at center frequency."""
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    amp = np.sqrt(power)
    t = torch.arange(n_symbols, device=device, dtype=torch.float32)
    freq_offset = torch.empty(1, device=device).uniform_(-0.1, 0.1).item()
    phase = torch.empty(1, device=device).uniform_(0, 2 * np.pi).item()
    return amp * torch.exp(1j * (2 * np.pi * freq_offset * t + phase))


def jam_successive_pulse(n_symbols, power_range=(1.0, 6.0), n_pulses=64):
    """Successive pulse jamming: periodic pulses at subcarrier spacing."""
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    jam = torch.zeros(n_symbols, dtype=torch.complex64, device=device)
    pulse_spacing = max(1, n_symbols // n_pulses)
    pulse_indices = torch.arange(0, n_symbols, pulse_spacing, device=device)
    amp = np.sqrt(power * n_pulses)
    phase = torch.empty(len(pulse_indices), device=device).uniform_(0, 2 * np.pi)
    jam[pulse_indices] = amp * torch.exp(1j * phase)
    return jam


def jam_protocol_aware(n_symbols, power_range=(0.3, 2.0)):
    """Protocol-aware: low-power shot-noise pulses (hardest to detect)."""
    power = torch.empty(1, device=device).uniform_(*power_range).item()
    mask = torch.rand(n_symbols, device=device) < 0.15
    amp = np.sqrt(power / max(mask.sum().item(), 1))
    noise = torch.randn(n_symbols, dtype=torch.complex64, device=device) * amp
    jam = torch.zeros(n_symbols, dtype=torch.complex64, device=device)
    jam[mask] = noise[mask]
    return jam


JAMMER_TYPES = [jam_barrage, jam_single_tone, jam_successive_pulse, jam_protocol_aware]


# ── Spectrogram generation ──────────────────────────────────────────────────

def iq_to_spectrogram(iq_signal, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert complex IQ signal to power spectrogram (magnitude squared).
    Computes STFT on the real (I) component — standard for baseband IQ.
    Returns a real-valued 2D tensor [freq_bins, time_frames] in dB."""
    signal_real = iq_signal.real.float()
    window = torch.hann_window(n_fft, device=iq_signal.device)
    spec = torch.stft(
        signal_real,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=False,
    )
    power = spec.abs().pow(2)
    power = torch.clamp(power, min=1e-10)
    return 10 * torch.log10(power)  # dB scale


def spectrogram_to_rgb(spec_db, img_size=IMG_SIZE):
    """Convert dB spectrogram to 3-channel RGB image tensor [3, H, W].
    Uses a colormap-like mapping analogous to GNURadio's waterfall display."""
    spec_np = spec_db.cpu().numpy()
    vmin, vmax = np.percentile(spec_np, [2, 98])
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    normed = np.clip((spec_np - vmin) / (vmax - vmin), 0, 1)

    cmap = plt.cm.viridis
    rgb = cmap(normed)[:, :, :3]  # [H, W, 3]
    rgb = rgb.transpose(2, 0, 1)  # [3, H, W]
    img = torch.tensor(rgb, dtype=torch.float32)

    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(img_size, img_size), mode='bilinear',
        align_corners=False
    ).squeeze(0)
    return img


def generate_sample(is_jammed, n_symbols=N_SYMBOLS_TIME * N_SUBCARRIERS):
    """Generate one spectrogram sample (clean or jammed)."""
    tx = generate_qpsk_signal(n_symbols)

    if is_jammed:
        jammer_fn = JAMMER_TYPES[torch.randint(0, len(JAMMER_TYPES), (1,)).item()]
        jam = jammer_fn(n_symbols)
        rx = tx + jam
    else:
        rx = tx.clone()

    spec_db = iq_to_spectrogram(rx)
    img = spectrogram_to_rgb(spec_db)
    return img


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """Pre-generated spectrogram dataset stored in memory."""

    def __init__(self, n_clean, n_jammed):
        self.images = []
        self.labels = []

        print(f"  Generating {n_clean} clean samples...", flush=True)
        for i in range(n_clean):
            self.images.append(generate_sample(is_jammed=False))
            self.labels.append(0)
            if (i + 1) % 100 == 0:
                print(f"    clean: {i+1}/{n_clean}", flush=True)

        print(f"  Generating {n_jammed} jammed samples...", flush=True)
        for i in range(n_jammed):
            self.images.append(generate_sample(is_jammed=True))
            self.labels.append(1)
            if (i + 1) % 100 == 0:
                print(f"    jammed: {i+1}/{n_jammed}", flush=True)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes=2):
    """EfficientNet-B0 adapted for binary spectrogram classification.
    Li et al. used EfficientNet-B0 (their best: 99.79% DR, 0.03% FAR).
    We load ImageNet-pretrained weights and replace the classifier head."""
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception:
        model = efficientnet_b0(weights=None)
        print("WARNING: pretrained weights unavailable, training from scratch", flush=True)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        tp += ((predicted == 1) & (labels == 1)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()

    acc = 100.0 * correct / total
    dr = 100.0 * tp / max(tp + fn, 1)
    far = 100.0 * fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'loss': running_loss / total,
        'accuracy': acc,
        'dr': dr,
        'far': far,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_training_plots(train_losses, train_accs, val_losses, val_accs, run_id, runs_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'sim05 CNN Detector Training — run{run_id}\n'
                      f'EfficientNet-B0 binary (clean vs jammed), '
                      f'SGD lr={LR}, batch={BATCH_SIZE}, epochs={EPOCHS}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Val')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, f"run{run_id}.png"), dpi=150)
    plt.close()


def save_sample_spectrograms(dataset, run_id, runs_dir, n_samples=8):
    """Save a grid of sample spectrograms for qualitative inspection."""
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(16, 6))
    labels_text = {0: 'Clean', 1: 'Jammed'}

    indices = torch.randperm(len(dataset))[:n_samples]
    for i, idx in enumerate(indices):
        img, label = dataset[idx.item()]
        ax = axes[i // (n_samples // 2), i % (n_samples // 2)]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(labels_text[int(label)], fontsize=10)
        ax.axis('off')

    plt.suptitle(f'sim05 Sample Spectrograms — run{run_id}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, f"run{run_id}_samples.png"), dpi=150)
    plt.close()


def save_confusion_matrix(metrics, run_id, runs_dir):
    """Save confusion matrix plot."""
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Clean', 'Jammed'])
    ax.set_yticklabels(['Clean', 'Jammed'])
    ax.set_xlabel('Actual Label')
    ax.set_ylabel('Predicted Label')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=14)
    ax.set_title(f'sim05 Confusion Matrix — run{run_id}\n'
                 f'DR={metrics["dr"]:.2f}%  FAR={metrics["far"]:.2f}%  '
                 f'F1={metrics["f1"]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, f"run{run_id}_cm.png"), dpi=150)
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

    runs_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim05", "detector")
    os.makedirs(runs_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(runs_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"Run {run_id}", flush=True)

    # Generate dataset
    n_jammed = args.n_per_jammer * len(JAMMER_TYPES)
    print(f"Generating dataset: {args.n_clean} clean + {n_jammed} jammed "
          f"= {args.n_clean + n_jammed} total", flush=True)
    t0 = time.time()
    dataset = SpectrogramDataset(args.n_clean, n_jammed)
    print(f"Dataset generated in {time.time() - t0:.1f}s", flush=True)

    # Train/test split (70/30, matching Li et al.)
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_test = n_total - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    print(f"Train: {n_train}, Test: {n_test}", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Save sample spectrograms for qualitative evaluation
    save_sample_spectrograms(dataset, run_id, runs_dir)

    # Build model
    model = build_model(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet-B0: {total_params:,} params ({trainable_params:,} trainable)", flush=True)

    # Li et al.: SGD with 100 epochs
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler (cosine annealing — standard for fine-tuning)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...\n", flush=True)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, epoch)
        val_metrics = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])

        dt = time.time() - t_epoch
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.1f}% "
              f"DR={val_metrics['dr']:.1f}% FAR={val_metrics['far']:.2f}% | "
              f"{dt:.1f}s", flush=True)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(),
                       os.path.join(runs_dir, f"run{run_id}_best.pt"))

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)

    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(runs_dir, f"run{run_id}_best.pt"),
                                      weights_only=True))
    final_metrics = evaluate(model, test_loader, criterion)
    print(f"\nFinal test results (best checkpoint):", flush=True)
    print(f"  Accuracy:  {final_metrics['accuracy']:.2f}%", flush=True)
    print(f"  DR:        {final_metrics['dr']:.2f}%", flush=True)
    print(f"  FAR:       {final_metrics['far']:.2f}%", flush=True)
    print(f"  F1:        {final_metrics['f1']:.4f}", flush=True)
    print(f"  Precision: {final_metrics['precision']:.4f}", flush=True)
    print(f"  Recall:    {final_metrics['recall']:.4f}", flush=True)

    # Save plots
    save_training_plots(train_losses, train_accs, val_losses, val_accs, run_id, runs_dir)
    save_confusion_matrix(final_metrics, run_id, runs_dir)

    # Save final model
    torch.save(model.state_dict(), os.path.join(runs_dir, f"run{run_id}_model.pt"))
    print(f"\nArtifacts saved to {runs_dir}/run{run_id}_*", flush=True)


if __name__ == "__main__":
    main()
