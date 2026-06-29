"""
sim06 — Cross-evaluation: trained MAPPO jammer vs frozen CNN detector.

Generates OFDM signals under different conditions and measures detection
rate and BER for each. Compares classical jammers against the MARL jammer.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ofdm
from jammer import load_agents, OBS_DIM
from detector import load_detector, detect
from train_detector import (jam_barrage, jam_single_tone,
                             jam_successive_pulse, jam_protocol_aware)

print(f"CUDA: {torch.cuda.is_available()}", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freq_grid_to_obs(freq_grid):
    g = freq_grid[:, 0, 0]
    return torch.cat([g.real, g.imag], dim=-1)


@torch.no_grad()
def evaluate_condition(detector, name, n_samples, generate_fn):
    """Evaluate detection rate and BER for a signal generation function.

    Args:
        detector: frozen CNN model
        name: condition name for logging
        n_samples: number of frames to evaluate
        generate_fn: callable(batch_size) → (rx_time_flat [B,T], ber [B])

    Returns:
        dict with detection_rate, mean_ber
    """
    all_det = []
    all_ber = []

    batch = min(n_samples, 32)
    for start in range(0, n_samples, batch):
        bs = min(batch, n_samples - start)
        rx_time, ber = generate_fn(bs)
        p_jammed = detect(detector, rx_time)
        all_det.append((p_jammed > 0.5).float())
        all_ber.append(ber)

    det_rate = torch.cat(all_det).mean().item()
    mean_ber = torch.cat(all_ber).mean().item()
    return {'detection_rate': det_rate, 'ber': mean_ber}


# ── Signal generators ───────────────────────────────────────────────────────

def gen_clean(batch_size):
    tx_bits, _, _, time_sig = ofdm.generate_ofdm_frame(batch_size)
    rx_time = time_sig[:, 0, 0]
    _, rx_eff = ofdm.demodulate_frame(time_sig)
    ber = ofdm.compute_ber(rx_eff, tx_bits, no=1e-10)
    return rx_time, ber


def make_classical_gen(jammer_fn):
    def gen(batch_size):
        tx_bits, _, _, time_sig = ofdm.generate_ofdm_frame(batch_size)
        time_flat = time_sig[:, 0, 0]
        T = time_flat.shape[-1]
        jammed = torch.stack([time_flat[i] + jammer_fn(T) for i in range(batch_size)])
        jammed_4d = jammed.unsqueeze(1).unsqueeze(1)
        _, rx_eff = ofdm.demodulate_frame(jammed_4d)
        ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)
        return jammed, ber
    return gen


def make_marl_gen(agents):
    def gen(batch_size):
        tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(batch_size)
        n_ofdm = ofdm.N_OFDM_SYMBOLS

        rx_grid = tx_grid.clone()
        for t in range(n_ofdm):
            tx_sym = tx_grid[:, :, :, t:t+1, :]
            obs = freq_grid_to_obs(tx_sym.squeeze(3))

            jam_total = torch.zeros(batch_size, ofdm.FFT_SIZE,
                                    dtype=torch.complex64, device=device)
            for agent in agents:
                jam_c, _, _ = agent.sample(obs)
                jam_total += jam_c

            rx_grid[:, 0, 0, t] = tx_grid[:, 0, 0, t] + jam_total

        rx_time = ofdm.modulator(rx_grid)
        rx_time_flat = rx_time[:, 0, 0]
        _, rx_eff = ofdm.demodulate_frame(rx_time)
        ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)
        return rx_time_flat, ber
    return gen


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jammer-model', type=str, required=True)
    parser.add_argument('--detector-model', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=200)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim06", "jammer")
    os.makedirs(out_dir, exist_ok=True)

    ofdm.init_device(device)

    agents, step = load_agents(args.jammer_model, device=device)
    print(f"Loaded {len(agents)}-agent jammer (step {step})", flush=True)

    detector = load_detector(args.detector_model, device=device)
    print(f"Loaded detector", flush=True)

    N = args.n_samples
    conditions = [
        ('Clean', gen_clean),
        ('Barrage', make_classical_gen(jam_barrage)),
        ('Single-tone', make_classical_gen(jam_single_tone)),
        ('Successive-pulse', make_classical_gen(jam_successive_pulse)),
        ('Protocol-aware', make_classical_gen(jam_protocol_aware)),
        ('MARL (sim06)', make_marl_gen(agents)),
    ]

    results = {}
    for name, gen_fn in conditions:
        print(f"Evaluating {name} ({N} samples)...", flush=True)
        r = evaluate_condition(detector, name, N, gen_fn)
        results[name] = r
        print(f"  {name:20s} DR={r['detection_rate']*100:5.1f}%  "
              f"BER={r['ber']:.4f}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65, flush=True)
    print(f"{'Category':20s} {'Det.Rate':>10s} {'BER':>8s}  Interpretation", flush=True)
    print("-" * 65, flush=True)
    for name, r in results.items():
        dr = r['detection_rate']
        if name == 'Clean':
            interp = "← FAR" if dr < 0.1 else "← HIGH FAR!"
        elif name == 'MARL (sim06)':
            interp = "← EVADES!" if dr < 0.3 else "← detected"
        else:
            interp = "← detected" if dr > 0.7 else "← evades!"
        print(f"  {name:20s} {dr*100:9.1f}%  {r['ber']:.4f}  {interp}", flush=True)
    print("=" * 65, flush=True)

    # ── Bar chart ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    names = list(results.keys())
    drs = [results[n]['detection_rate'] * 100 for n in names]
    bers = [results[n]['ber'] for n in names]
    colors = ['#2ecc71' if n == 'Clean' else
              '#e74c3c' if 'MARL' in n else '#3498db' for n in names]

    bars1 = ax1.bar(names, drs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('sim06 — CNN Detector vs Jammer Types (OFDM)')
    ax1.set_ylim(0, 105)
    ax1.axhline(50, color='gray', ls='--', alpha=0.5, label='Random')
    for b, v in zip(bars1, drs):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height()+1,
                 f'{v:.1f}%', ha='center', fontsize=9)
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(names, bers, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('BER')
    ax2.set_title('BER per Jammer Type')
    for b, v in zip(bars2, bers):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height()+0.005,
                 f'{v:.3f}', ha='center', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_jammer_vs_detector.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_dir}/eval_jammer_vs_detector.png", flush=True)


if __name__ == "__main__":
    main()
