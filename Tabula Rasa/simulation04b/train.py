"""
sim04b — Same as sim04 but uses Sionna source/mapper/demapper on GPU
(via sionna.phy.config.device) instead of handcrafted pure-PyTorch ops.

Goal: verify that Sionna on GPU + torch.compile is viable for sim06/07
where we'll need Sionna's channel models. Compare sps and RSS with sim04.
"""
import os
import gc
import time
try:
    import psutil
except ImportError:
    psutil = None
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
print("imports starting...", flush=True)
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import zuko
import sionna.phy as sn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
print(f"imports done — CUDA: {torch.cuda.is_available()}", flush=True)

gc.set_threshold(100, 5, 5)
_proc = psutil.Process() if psutil else None

# ── Device setup (BEFORE creating Sionna modules) ───────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sn.config.device = str(device)
print(f"Using device: {device} (sionna.phy.config.device={sn.config.device})", flush=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_JAMMERS      = 2
N_SYMBOLS      = 128
KURT_THRESH    = -1.0
LAMBDA         = 2.0
GAMMA          = 0.02
BATCH_SIZE     = 2048
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TOTAL_STEPS    = 20_000
LOG_EVERY      = 50
CHECKPOINT_EVERY = 500

# ── Sionna blocks (on GPU via config.device) ─────────────────────────────────
source   = sn.mapping.BinarySource()
constel  = sn.mapping.Constellation("qam", num_bits_per_symbol=2)
mapper   = sn.mapping.Mapper(constellation=constel)
demapper = sn.mapping.Demapper("app", constellation=constel)

# ── Run bookkeeping ──────────────────────────────────────────────────────────
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim04b")
os.makedirs(RUNS_DIR, exist_ok=True)
existing = [int(f[3:6]) for f in os.listdir(RUNS_DIR)
            if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
run_id = f"{max(existing, default=0) + 1:03d}"
print(f"Run {run_id}", flush=True)

writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, "tb", f"run{run_id}"))

# ── Agents ───────────────────────────────────────────────────────────────────
OBS_DIM = 2 * N_SYMBOLS
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

agents = [make_agent() for _ in range(N_JAMMERS)]

all_params = [p for enc, fl in agents for p in list(enc.parameters()) + list(fl.parameters())]
optimizer = torch.optim.Adam(all_params, lr=LR, weight_decay=WEIGHT_DECAY)

# ── Helpers ──────────────────────────────────────────────────────────────────
def excess_kurtosis_batch(x: torch.Tensor) -> torch.Tensor:
    diff = x - x.mean(dim=-1, keepdim=True)
    m2   = (diff ** 2).mean(dim=-1)
    m4   = (diff ** 4).mean(dim=-1)
    return m4 / (m2 ** 2 + 1e-12) - 3


def _sample_jammer(encoder, flow, obs):
    ctx      = encoder(obs)
    dist     = flow(ctx)
    jam_flat = dist.rsample()
    del dist
    jam_flat = torch.nan_to_num(jam_flat, nan=0.0, posinf=20.0, neginf=-20.0)
    jam_flat = jam_flat.clamp(-20, 20)
    return torch.complex(jam_flat[:, :N_SYMBOLS], jam_flat[:, N_SYMBOLS:])

try:
    sample_jammer = torch.compile(_sample_jammer)
    print("torch.compile: enabled", flush=True)
except Exception as e:
    sample_jammer = _sample_jammer
    print(f"torch.compile: unavailable ({e}), using eager mode", flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────
QPSK_I = np.array([1, 1, -1, -1]) / np.sqrt(2)
QPSK_Q = np.array([1, -1, 1, -1]) / np.sqrt(2)

meta_str = (f"run={run_id}  |  N_JAMMERS={N_JAMMERS}  |  N_SYMBOLS={N_SYMBOLS}  |  "
            f"LAMBDA={LAMBDA}  |  GAMMA={GAMMA}  |  KURT_THRESH={KURT_THRESH}  |  "
            f"BATCH={BATCH_SIZE}  |  sionna=GPU")


def save_plot(steps_done):
    steps_logged = [i * LOG_EVERY for i in range(len(log_ber))]
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(steps_logged, log_ber)
    axes[0].set_ylabel("BER")
    axes[0].set_title(f"sim04b — Sionna on GPU — 2-agent cooperative jammer\n"
                      f"{meta_str}  |  steps={steps_done:,}/{TOTAL_STEPS:,}", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps_logged, log_detection, color='red')
    axes[1].set_ylabel("Detection rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps_logged, log_kurtosis, color='green')
    axes[2].axhline(KURT_THRESH, color='orange', ls='--', label=f'KURT_THRESH ({KURT_THRESH})')
    axes[2].axhline(0,           color='gray',   ls=':',  label='Gaussian reference (0)')
    axes[2].axhline(-2,          color='blue',   ls=':',  label='QPSK reference (-2)')
    axes[2].set_ylabel("Mean kurtosis (rx)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    for i, (color, label) in enumerate(zip(['purple', 'darkorange'], ['Jammer 1', 'Jammer 2'])):
        axes[3].plot(steps_logged, log_power[i], color=color, label=label)
    axes[3].set_ylabel("Jammer power (per agent)")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.3)

    axes[4].plot(steps_logged, log_total_power, color='black')
    axes[4].set_ylabel("Total jam power")
    axes[4].set_xlabel("Step")
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RUNS_DIR, f"run{run_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def save_iq_plot():
    if not iq_snapshots:
        return
    times  = [l for l in ("early", "mid", "late") if l in iq_snapshots]
    row_labels = [f"Jammer {i+1}" for i in range(N_JAMMERS)] + ["rx = tx + Σjam"]
    n_rows, n_cols = len(row_labels), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"IQ scatter — [{meta_str}]", fontsize=7)

    for col, t in enumerate(times):
        snaps = iq_snapshots[t]
        for row, (label, (I, Q)) in enumerate(zip(row_labels, snaps)):
            ax = axes[row, col]
            color = ['steelblue', 'darkorange', 'green'][row % 3]
            ax.scatter(I, Q, s=1, alpha=0.05, color=color)
            ax.scatter(QPSK_I, QPSK_Q, s=60, color='red', marker='x',
                       label='QPSK ref', zorder=5)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f"{label} — {t}", fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    iq_path = os.path.join(RUNS_DIR, f"run{run_id}_iq.png")
    plt.savefig(iq_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {iq_path}", flush=True)


def save_model(steps_done):
    model_path = os.path.join(RUNS_DIR, f"run{run_id}_model.pt")
    torch.save({
        "agents": [{"encoder": enc.state_dict(), "flow": fl.state_dict()}
                   for enc, fl in agents],
        "step": steps_done,
        "N_JAMMERS": N_JAMMERS,
    }, model_path)
    print(f"Saved model → {model_path}", flush=True)


def take_snapshot(jam_syms_list, rx_syms):
    snaps = []
    for j in jam_syms_list:
        snaps.append((j.real.detach().cpu().numpy().flatten(),
                      j.imag.detach().cpu().numpy().flatten()))
    snaps.append((rx_syms.real.detach().cpu().numpy().flatten(),
                  rx_syms.imag.detach().cpu().numpy().flatten()))
    return snaps


# ── Training loop ────────────────────────────────────────────────────────────
log_ber        = []
log_detection  = []
log_kurtosis   = []
log_power      = [[] for _ in range(N_JAMMERS)]
log_total_power = []
iq_snapshots   = {}

print("training started...", flush=True)
t_start = time.time()
t_last_log = t_start

for step in range(TOTAL_STEPS):

    # ── Sionna source + mapper (on GPU via config.device) ────────────────────
    with torch.no_grad():
        tx_bits_raw = source([BATCH_SIZE * N_SYMBOLS, 2])
        tx_syms_raw = mapper(tx_bits_raw).squeeze(-1)

    tx_syms = tx_syms_raw.reshape(BATCH_SIZE, N_SYMBOLS)
    tx_bits = tx_bits_raw.reshape(BATCH_SIZE, N_SYMBOLS, 2)

    obs = torch.stack([tx_syms.real, tx_syms.imag], dim=-1).reshape(BATCH_SIZE, OBS_DIM)

    jam_syms_list = [sample_jammer(enc, fl, obs) for enc, fl in agents]

    rx_syms = tx_syms + sum(jam_syms_list)

    # ── IQ snapshots ─────────────────────────────────────────────────────────
    if step == 0:
        iq_snapshots["early"] = take_snapshot(jam_syms_list, rx_syms)
    if step == TOTAL_STEPS // 2:
        iq_snapshots["mid"] = take_snapshot(jam_syms_list, rx_syms)
    if step % LOG_EVERY == 0:
        iq_snapshots["late"] = take_snapshot(jam_syms_list, rx_syms)

    # ── Kurtosis constraint ──────────────────────────────────────────────────
    kurt_i = excess_kurtosis_batch(rx_syms.real)
    kurt_q = excess_kurtosis_batch(rx_syms.imag)
    kurt   = (kurt_i + kurt_q) / 2
    loss_k = F.relu(kurt - KURT_THRESH).mean()

    # ── Soft BER loss (Sionna demapper on GPU) ───────────────────────────────
    rx_for_demap = rx_syms.reshape(-1).unsqueeze(-1)
    llr          = demapper(rx_for_demap, 1.0).clamp(-10, 10)
    wrong_labels = 1.0 - tx_bits.reshape(-1, 2).float()
    loss_ber     = F.binary_cross_entropy_with_logits(llr, wrong_labels)

    # ── Power penalty ────────────────────────────────────────────────────────
    powers    = [j.abs().pow(2).mean() for j in jam_syms_list]
    loss_pow  = GAMMA * sum(powers)

    loss = loss_ber + LAMBDA * loss_k + loss_pow

    optimizer.zero_grad()

    if not torch.isfinite(loss):
        print(f"step {step:5d} | non-finite loss ({float(loss)}), skipping update", flush=True)
        continue

    loss.backward()

    for enc, fl in agents:
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(fl.parameters()), max_norm=1.0
        )

    optimizer.step()

    # ── Logging ──────────────────────────────────────────────────────────────
    if step % LOG_EVERY == 0:
        with torch.no_grad():
            rx_bits  = sn.utils.hard_decisions(llr.detach())
            ber      = float((tx_bits.reshape(-1, 2) != rx_bits).float().mean())
            detected = float((kurt > KURT_THRESH).float().mean())
            kurt_val = float(kurt.mean())
            pow_vals = [float(p) for p in powers]
            total_pow = sum(pow_vals)
            loss_val     = float(loss)
            loss_ber_val = float(loss_ber)
            loss_k_val   = float(loss_k)

        log_ber.append(ber)
        log_detection.append(detected)
        log_kurtosis.append(kurt_val)
        for i, pv in enumerate(pow_vals):
            log_power[i].append(pv)
        log_total_power.append(total_pow)

        now     = time.time()
        elapsed = now - t_start
        sps     = (step + 1) / elapsed
        inst_sps = LOG_EVERY / (now - t_last_log) if step > 0 else sps
        t_last_log = now
        rss_mb = _proc.memory_info().rss / 1024**2 if _proc else 0
        power_str = "  ".join(f"p{i+1}={pv:.3f}" for i, pv in enumerate(pow_vals))
        print(f"step {step:5d} | {inst_sps:5.2f} sps (avg {sps:.2f}) | RSS {rss_mb:.0f}MB "
              f"| loss {loss_val:.4f} | BER {ber:.3f} | det {detected:.3f} "
              f"| kurt {kurt_val:.3f} | {power_str}", flush=True)

        writer.add_scalar("loss/total", loss_val, step)
        writer.add_scalar("loss/ber_term", loss_ber_val, step)
        writer.add_scalar("loss/kurtosis_term", loss_k_val, step)
        writer.add_scalar("ber", ber, step)
        writer.add_scalar("detection_rate", detected, step)
        writer.add_scalar("kurtosis", kurt_val, step)
        writer.add_scalar("jam_power/total", total_pow, step)
        for i, pv in enumerate(pow_vals):
            writer.add_scalar(f"jam_power/agent_{i+1}", pv, step)
        writer.add_scalar("steps_per_sec/avg", sps, step)
        writer.add_scalar("steps_per_sec/inst", inst_sps, step)

    if (step + 1) % CHECKPOINT_EVERY == 0:
        save_plot(step + 1)
        save_iq_plot()
        save_model(step + 1)

    del jam_syms_list, rx_syms, loss, loss_ber, loss_k, loss_pow, powers
    del kurt, kurt_i, kurt_q, rx_for_demap, llr, wrong_labels

# ── Final save ───────────────────────────────────────────────────────────────
save_plot(TOTAL_STEPS)
save_iq_plot()
save_model(TOTAL_STEPS)
writer.close()
