"""
CNN detector — EfficientNet-B0 binary classifier on OFDM spectrograms.

Replicates Li et al. (IEEE Access 2022) detector methodology:
  - EfficientNet-B0 with ImageNet pretrained weights
  - Binary classification: clean (0) vs jammed (1)
  - Input: 224x224x3 RGB spectrogram images

Provides both model definition and spectrogram conversion utilities,
reused by train_detector.py, train_jammer.py, and eval.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Spectrogram parameters ──────────────────────────────────────────────────
SPEC_N_FFT = 128
SPEC_HOP = 32
IMG_SIZE = 224

# ── Viridis colormap LUT (avoids matplotlib dependency in hot loop) ─────────
# 256-entry RGB table, precomputed from plt.cm.viridis
import matplotlib.pyplot as plt
_VIRIDIS_LUT = torch.tensor(
    [plt.cm.viridis(i / 255.0)[:3] for i in range(256)],
    dtype=torch.float32
)  # [256, 3]


# ── Model ────────────────────────────────────────────────────────────────────

def build_detector(num_classes=2):
    """EfficientNet-B0 with ImageNet weights and binary classifier head."""
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception:
        model = efficientnet_b0(weights=None)
        print("WARNING: pretrained weights unavailable, training from scratch",
              flush=True)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_detector(path, device='cpu'):
    """Load a trained detector from checkpoint."""
    model = build_detector(num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ── Spectrogram generation ──────────────────────────────────────────────────

def time_signal_to_spectrogram(iq_signal, n_fft=SPEC_N_FFT, hop=SPEC_HOP):
    """Convert complex time-domain signal to dB power spectrogram.

    Args:
        iq_signal: [B, T] complex or [T] complex
        n_fft: STFT FFT size
        hop: STFT hop length

    Returns:
        spec_db: [B, freq_bins, time_frames] or [freq_bins, time_frames] float
    """
    squeezed = iq_signal.ndim == 1
    if squeezed:
        iq_signal = iq_signal.unsqueeze(0)

    signal_real = iq_signal.real.float()
    window = torch.hann_window(n_fft, device=iq_signal.device)

    specs = []
    for i in range(signal_real.shape[0]):
        spec = torch.stft(
            signal_real[i], n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, return_complex=True, center=False,
        )
        specs.append(spec.abs().pow(2))

    power = torch.stack(specs)  # [B, freq, time]
    power = torch.clamp(power, min=1e-10)
    spec_db = 10 * torch.log10(power)

    if squeezed:
        return spec_db.squeeze(0)
    return spec_db


def spectrogram_to_image(spec_db, img_size=IMG_SIZE):
    """Convert dB spectrogram(s) to RGB image tensor(s).

    Args:
        spec_db: [B, H, W] or [H, W] float

    Returns:
        images: [B, 3, img_size, img_size] or [3, img_size, img_size] float
    """
    squeezed = spec_db.ndim == 2
    if squeezed:
        spec_db = spec_db.unsqueeze(0)

    B = spec_db.shape[0]
    images = []

    lut = _VIRIDIS_LUT.to(spec_db.device)

    for i in range(B):
        s = spec_db[i]
        vmin = s.quantile(0.02)
        vmax = s.quantile(0.98)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        normed = ((s - vmin) / (vmax - vmin)).clamp(0, 1)

        # Map to viridis LUT indices
        idx = (normed * 255).long().clamp(0, 255)
        rgb = lut[idx]  # [H, W, 3]
        img = rgb.permute(2, 0, 1)  # [3, H, W]

        img = F.interpolate(
            img.unsqueeze(0), size=(img_size, img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        images.append(img)

    result = torch.stack(images)  # [B, 3, H, W]
    if squeezed:
        return result.squeeze(0)
    return result


# ── End-to-end detection ────────────────────────────────────────────────────

@torch.no_grad()
def detect(model, time_signal):
    """End-to-end: time-domain signal → P(jammed).

    Args:
        model: frozen EfficientNet-B0 detector
        time_signal: [B, T] complex time-domain signal

    Returns:
        p_jammed: [B] float — probability of jamming per sample
    """
    spec_db = time_signal_to_spectrogram(time_signal)
    images = spectrogram_to_image(spec_db).to(time_signal.device)
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 1]  # P(class=1=jammed)
