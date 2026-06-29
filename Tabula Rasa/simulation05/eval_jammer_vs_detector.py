"""
sim05 — Evaluate a trained jammer (sim04) against the CNN detector.

Loads a sim04 2-agent NSF jammer and the sim05 CNN detector, then:
1. Generates clean QPSK signals
2. Generates jammed signals using the trained jammer
3. Generates classically jammed signals (barrage, tone, pulse, protocol-aware)
4. Runs all through the CNN detector
5. Reports detection rates for each — does the kurtosis-trained jammer
   accidentally evade the CNN detector too?
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zuko
from torchvision.models import efficientnet_b0

# Import spectrogram helpers from the detector trainer
from train_detector import (
    iq_to_spectrogram, spectrogram_to_rgb, IMG_SIZE,
    generate_qpsk_signal, jam_barrage, jam_single_tone,
    jam_successive_pulse, jam_protocol_aware,
    QPSK_POINTS, N_SYMBOLS_TIME, N_SUBCARRIERS,
)

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── sim04 jammer architecture (must match sim04/train.py) ────────────────────
N_SYMBOLS_JAM = 128
OBS_DIM = 2 * N_SYMBOLS_JAM
CTX_DIM = 64


def make_agent():
    encoder = nn.Sequential(
        nn.Linear(OBS_DIM, 64), nn.ReLU(),
        nn.Linear(64, CTX_DIM), nn.ReLU(),
    )
    flow = zuko.flows.NSF(
        features=OBS_DIM,
        context=CTX_DIM,
        transforms=3,
        hidden_features=[64, 64],
        randperm=True,
        passes=2,
    )
    encoder.to(device)
    flow.to(device)
    return encoder, flow


def load_jammer(model_path):
    """Load sim04 2-agent jammer from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    n_jammers = ckpt["N_JAMMERS"]
    agents = []
    for i in range(n_jammers):
        enc, flow = make_agent()
        enc.load_state_dict(ckpt["agents"][i]["encoder"])
        flow.load_state_dict(ckpt["agents"][i]["flow"])
        enc.eval()
        flow.eval()
        agents.append((enc, flow))
    print(f"Loaded {n_jammers}-agent jammer from {model_path} "
          f"(step {ckpt['step']})", flush=True)
    return agents


def sample_jammer(encoder, flow, obs):
    """Generate jam symbols from one agent."""
    ctx = encoder(obs)
    dist = flow(ctx)
    jam_flat = dist.rsample()
    jam_flat = torch.nan_to_num(jam_flat, nan=0.0, posinf=20.0, neginf=-20.0)
    jam_flat = jam_flat.clamp(-20, 20)
    return torch.complex(jam_flat[:, :N_SYMBOLS_JAM], jam_flat[:, N_SYMBOLS_JAM:])


# ── CNN detector ─────────────────────────────────────────────────────────────

def load_detector(model_path):
    """Load sim05 EfficientNet-B0 binary detector."""
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2),
    )
    model.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded CNN detector from {model_path}", flush=True)
    return model


# ── Signal generation ────────────────────────────────────────────────────────

def generate_clean_spectrogram():
    """Generate one clean QPSK spectrogram."""
    n = N_SYMBOLS_TIME * N_SUBCARRIERS
    tx = generate_qpsk_signal(n)
    spec = iq_to_spectrogram(tx)
    return spectrogram_to_rgb(spec)


def generate_classical_jammed_spectrogram(jammer_fn):
    """Generate one classically jammed spectrogram."""
    n = N_SYMBOLS_TIME * N_SUBCARRIERS
    tx = generate_qpsk_signal(n)
    jam = jammer_fn(n)
    rx = tx + jam
    spec = iq_to_spectrogram(rx)
    return spectrogram_to_rgb(spec)


@torch.no_grad()
def generate_marl_jammed_spectrogram(agents):
    """Generate one spectrogram jammed by the sim04 MARL jammer.

    The sim04 jammer operates on 128-symbol windows with observation of tx.
    We tile this across the full spectrogram signal length."""
    n_total = N_SYMBOLS_TIME * N_SUBCARRIERS
    tx_full = generate_qpsk_signal(n_total)

    # Process in 128-symbol chunks (matching sim04's N_SYMBOLS)
    rx_full = tx_full.clone()
    n_chunks = n_total // N_SYMBOLS_JAM

    for chunk_idx in range(n_chunks):
        start = chunk_idx * N_SYMBOLS_JAM
        end = start + N_SYMBOLS_JAM
        tx_chunk = tx_full[start:end]

        # Observation: flattened [I, Q] of overheard tx
        obs = torch.cat([tx_chunk.real, tx_chunk.imag]).unsqueeze(0)  # [1, 256]

        # Sum contributions from all agents
        jam_total = torch.zeros(N_SYMBOLS_JAM, dtype=torch.complex64, device=device)
        for enc, flow in agents:
            jam = sample_jammer(enc, flow, obs)  # [1, 128]
            jam_total += jam.squeeze(0)

        rx_full[start:end] = tx_chunk + jam_total

    spec = iq_to_spectrogram(rx_full.detach())
    return spectrogram_to_rgb(spec)


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_detection(detector, spectrograms):
    """Run detector on a batch of spectrograms, return detection rate."""
    images = torch.stack(spectrograms).to(device)
    outputs = detector(images)
    predicted = outputs.argmax(dim=1)
    # label 1 = "jammed" → detection
    detection_rate = (predicted == 1).float().mean().item()
    return detection_rate, predicted.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jammer-model', type=str, required=True,
                        help='Path to sim04 jammer model .pt file')
    parser.add_argument('--detector-model', type=str, required=True,
                        help='Path to sim05 detector model .pt file')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Samples per category')
    args = parser.parse_args()

    runs_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim05", "jammer")
    os.makedirs(runs_dir, exist_ok=True)

    # Load models
    agents = load_jammer(args.jammer_model)
    detector = load_detector(args.detector_model)

    N = args.n_samples
    results = {}

    # 1. Clean signals (should NOT be detected → low detection rate = good)
    print(f"\nGenerating {N} clean samples...", flush=True)
    clean_specs = [generate_clean_spectrogram() for _ in range(N)]
    dr, _ = evaluate_detection(detector, clean_specs)
    results['Clean'] = dr
    print(f"  Clean:           detection rate = {dr*100:.1f}% "
          f"(want ~0%, FAR)", flush=True)

    # 2. Classical jammers (should be detected → high detection rate = good)
    classical = {
        'Barrage': jam_barrage,
        'Single-tone': jam_single_tone,
        'Successive-pulse': jam_successive_pulse,
        'Protocol-aware': jam_protocol_aware,
    }
    for name, jammer_fn in classical.items():
        print(f"Generating {N} {name} jammed samples...", flush=True)
        specs = [generate_classical_jammed_spectrogram(jammer_fn) for _ in range(N)]
        dr, _ = evaluate_detection(detector, specs)
        results[name] = dr
        print(f"  {name:20s} detection rate = {dr*100:.1f}%", flush=True)

    # 3. MARL jammer (the interesting one — does the kurtosis-trained jammer
    #    also evade the CNN detector it was never trained against?)
    print(f"Generating {N} MARL-jammed samples...", flush=True)
    marl_specs = [generate_marl_jammed_spectrogram(agents) for _ in range(N)]
    dr, _ = evaluate_detection(detector, marl_specs)
    results['MARL (sim04)'] = dr
    print(f"  {'MARL (sim04)':20s} detection rate = {dr*100:.1f}% "
          f"(kurtosis-trained jammer vs CNN detector)", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("DETECTION RATE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'Category':20s} {'Det. Rate':>10s}  {'Interpretation'}", flush=True)
    print("-" * 60, flush=True)
    for name, dr in results.items():
        if name == 'Clean':
            interp = "← FAR (want low)" if dr < 0.1 else "← HIGH FAR!"
        elif name == 'MARL (sim04)':
            interp = "← EVADES detector!" if dr < 0.3 else "← detected"
        else:
            interp = "← detected" if dr > 0.7 else "← evades!"
        print(f"  {name:20s} {dr*100:9.1f}%  {interp}", flush=True)
    print("=" * 60, flush=True)

    # ── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    rates = [results[n] * 100 for n in names]
    colors = ['#2ecc71' if n == 'Clean' else
              '#e74c3c' if n == 'MARL (sim04)' else '#3498db'
              for n in names]

    bars = ax.bar(names, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('sim05 — CNN Detector vs Different Jammer Types\n'
                 '(MARL jammer trained on kurtosis, never saw CNN detector)')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random guess')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(runs_dir, "eval_jammer_vs_detector.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
