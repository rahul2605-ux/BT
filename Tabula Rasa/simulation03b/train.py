import os
import gc
import time
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

# Disable automatic GC: zuko's flow(ctx).rsample() creates autograd-graph
# reference cycles each step, which the cyclic collector can only reclaim via
# full gen-2 sweeps. As cyclic garbage accumulates, those sweeps get scanned
# over more and more tracked objects, causing per-step time to grow ~5x over
# a run. Collect manually on a fixed schedule instead (bounded, predictable cost).
gc.disable()
GC_EVERY = 500

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_SYMBOLS      = 128
KURT_THRESH    = -1.0
LAMBDA         = 2.0    # kurtosis penalty weight
GAMMA          = 0.02   # power penalty weight
BATCH_SIZE     = 64
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TOTAL_STEPS    = 20_000
LOG_EVERY      = 50
CHECKPOINT_EVERY = 500   # save plot + model every N steps so a timeout doesn't lose everything

# ── Sionna blocks ─────────────────────────────────────────────────────────────
source   = sn.mapping.BinarySource()
constel  = sn.mapping.Constellation("qam", num_bits_per_symbol=2)
mapper   = sn.mapping.Mapper(constellation=constel)
demapper = sn.mapping.Demapper("app", constellation=constel)

# ── Run bookkeeping ───────────────────────────────────────────────────────────
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim03b")
os.makedirs(RUNS_DIR, exist_ok=True)
existing = [int(f[3:6]) for f in os.listdir(RUNS_DIR)
            if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
run_id = f"{max(existing, default=0) + 1:03d}"
print(f"Run {run_id}", flush=True)

writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, "tb", f"run{run_id}"))

# ── Model ─────────────────────────────────────────────────────────────────────
OBS_DIM = 2 * N_SYMBOLS
CTX_DIM = 64

# Small MLP encodes tx_syms observation into a context vector for the flow
encoder = nn.Sequential(
    nn.Linear(OBS_DIM, 64), nn.ReLU(),
    nn.Linear(64, CTX_DIM), nn.ReLU(),
)

# NSF conditioned on context — same architecture as sim03 FlowPolicy
flow = zuko.flows.NSF(
    features=OBS_DIM,
    context=CTX_DIM,
    transforms=3,
    hidden_features=[64, 64],
    randperm=True,
    passes=2,  # coupling-style (2 sequential passes) instead of fully autoregressive
              # (passes=None) — fully autoregressive needs `features` sequential
              # hypernetwork calls per .rsample(), ~768 calls/step here (~20s/step on CPU)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
flow.to(device)
print(f"Using device: {device}", flush=True)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(flow.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

# ── Helpers ───────────────────────────────────────────────────────────────────
def excess_kurtosis_batch(x: torch.Tensor) -> torch.Tensor:
    # x: [B, N] — Fisher excess kurtosis per sample over N dimension
    diff = x - x.mean(dim=-1, keepdim=True)
    m2   = (diff ** 2).mean(dim=-1)
    m4   = (diff ** 4).mean(dim=-1)
    return m4 / (m2 ** 2 + 1e-12) - 3  # [B]

# ── Plotting helper (used for checkpoints and final save) ────────────────────
def save_plot(steps_done):
    steps_logged = [i * LOG_EVERY for i in range(len(log_ber))]
    meta = (f"run={run_id}  |  steps={steps_done:,}/{TOTAL_STEPS:,}  |  batch={BATCH_SIZE}  |  "
            f"N_SYMBOLS={N_SYMBOLS}  |  LAMBDA={LAMBDA}  |  GAMMA={GAMMA}  |  "
            f"KURT_THRESH={KURT_THRESH}")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(steps_logged, log_ber)
    axes[0].set_ylabel("BER")
    axes[0].set_title(f"Generative (NSF direct-gradient) — Kurtosis Detector\n{meta}", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps_logged, log_detection, color='red')
    axes[1].set_ylabel("Detection rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps_logged, log_kurtosis, color='green')
    axes[2].axhline(KURT_THRESH, color='orange', ls='--', label=f'KURT_THRESH ({KURT_THRESH})')
    axes[2].axhline(0,           color='gray',   ls=':',  label='Gaussian reference (0)')
    axes[2].axhline(-2,          color='blue',   ls=':',  label='QPSK reference (-2)')
    axes[2].set_ylabel("Mean kurtosis (batch)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(steps_logged, log_power, color='purple')
    axes[3].set_ylabel("Jammer power")
    axes[3].set_xlabel("Step")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RUNS_DIR, f"run{run_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


QPSK_I = np.array([1, 1, -1, -1]) / np.sqrt(2)
QPSK_Q = np.array([1, -1, 1, -1]) / np.sqrt(2)


def save_iq_plot():
    if not iq_snapshots:
        return
    order   = [l for l in ("early", "mid", "late") if l in iq_snapshots]
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
    if len(order) == 1:
        axes = [axes]
    fig.suptitle(f"IQ scatter — generated jam symbols  [run={run_id}]", fontsize=8)

    for ax, label in zip(axes, order):
        I, Q = iq_snapshots[label]
        ax.scatter(I, Q, s=1, alpha=0.05, color='steelblue')
        ax.scatter(QPSK_I, QPSK_Q, s=60, color='red', marker='x', label='QPSK ref', zorder=5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    iq_path = os.path.join(RUNS_DIR, f"run{run_id}_iq.png")
    plt.savefig(iq_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {iq_path}", flush=True)


def save_model(steps_done):
    model_path = os.path.join(RUNS_DIR, f"run{run_id}_model.pt")
    torch.save({
        "encoder": encoder.state_dict(),
        "flow": flow.state_dict(),
        "step": steps_done,
    }, model_path)
    print(f"Saved model → {model_path}", flush=True)


# ── Training loop ─────────────────────────────────────────────────────────────
log_ber       = []
log_detection = []
log_kurtosis  = []
log_power     = []
log_loss      = []
iq_snapshots  = {}  # label -> (I, Q) numpy arrays, captured at early/mid/late training

print("training started...", flush=True)
t_start = time.time()

for step in range(TOTAL_STEPS):

    # Fresh batch of tx symbols — no gradients needed here
    with torch.no_grad():
        tx_bits_raw = source([BATCH_SIZE * N_SYMBOLS, 2])           # [B*N, 2]
        tx_syms_raw = mapper(tx_bits_raw).squeeze(-1)               # [B*N]

    tx_syms = tx_syms_raw.reshape(BATCH_SIZE, N_SYMBOLS).to(device) # [B, N]
    tx_bits = tx_bits_raw.reshape(BATCH_SIZE, N_SYMBOLS, 2).to(device)

    # Observation: flattened [I, Q] of tx symbols
    obs = torch.stack([tx_syms.real, tx_syms.imag], dim=-1).reshape(BATCH_SIZE, OBS_DIM)

    # Generate jam symbols via flow (reparameterized — gradients flow back)
    ctx      = encoder(obs)                                         # [B, CTX_DIM]
    jam_flat = flow(ctx).rsample()                                  # [B, 2*N]
    # NSF's rational-quadratic spline can occasionally produce actual NaN/Inf
    # entries (e.g. a near-zero bin width in the spline's hypernetwork output
    # causes a 0/0 or x/0 inside the rational-quadratic transform), which then
    # poisons the kurtosis ratio (m4/m2^2 -> nan) and, via "skip update", can
    # freeze the weights at a permanently-broken point for the rest of training.
    # `clamp` alone does NOT fix this — clamp(nan, ...) == nan. Replace any
    # nan/inf with finite values first (zero-gradient at those entries, so it
    # doesn't poison the update), then clamp well above the |~1.4| needed for
    # the jam=-2*tx optimum, just to keep things bounded.
    jam_flat = torch.nan_to_num(jam_flat, nan=0.0, posinf=20.0, neginf=-20.0)
    jam_flat = jam_flat.clamp(-20, 20)
    jam_syms = torch.complex(jam_flat[:, :N_SYMBOLS],
                             jam_flat[:, N_SYMBOLS:])               # [B, N]

    # Lossless channel
    rx_syms = tx_syms + jam_syms                                    # [B, N]

    # ── IQ snapshots (early / mid / late) for distribution-shape diagnostics ──
    if step == 0:
        iq_snapshots["early"] = (jam_syms.real.detach().cpu().numpy().flatten(),
                                  jam_syms.imag.detach().cpu().numpy().flatten())
    if step == TOTAL_STEPS // 2:
        iq_snapshots["mid"] = (jam_syms.real.detach().cpu().numpy().flatten(),
                                jam_syms.imag.detach().cpu().numpy().flatten())
    if step % LOG_EVERY == 0:
        iq_snapshots["late"] = (jam_syms.real.detach().cpu().numpy().flatten(),
                                 jam_syms.imag.detach().cpu().numpy().flatten())

    # ── Kurtosis constraint loss ──────────────────────────────────────────────
    kurt_i = excess_kurtosis_batch(rx_syms.real)                    # [B]
    kurt_q = excess_kurtosis_batch(rx_syms.imag)                    # [B]
    kurt   = (kurt_i + kurt_q) / 2                                  # [B]
    loss_k = F.relu(kurt - KURT_THRESH).mean()

    # ── Soft BER loss (maximize BER = minimise CE with flipped labels) ────────
    # NOTE: `no` must be O(1), not ~0 — with no~1e-10 the demapper's LLRs blow up
    # to +-inf for any nonzero rx-tx deviation, saturate the clamp, and
    # binary_cross_entropy_with_logits has ~zero gradient there. That left the
    # optimizer stuck at jam_power~0 (loss pinned at the clamp ceiling, ~20).
    rx_for_demap = rx_syms.reshape(-1).unsqueeze(-1)                # [B*N, 1]
    llr          = demapper(rx_for_demap, 1.0).clamp(-10, 10)      # [B*N, 2]
    wrong_labels = 1.0 - tx_bits.reshape(-1, 2).float()            # [B*N, 2]
    loss_ber     = F.binary_cross_entropy_with_logits(llr, wrong_labels)

    # ── Power penalty ─────────────────────────────────────────────────────────
    jam_power = jam_syms.abs().pow(2).mean()

    loss = loss_ber + LAMBDA * loss_k + GAMMA * jam_power

    optimizer.zero_grad()

    if not torch.isfinite(loss):
        # NaN/Inf loss (e.g. from a flow sample landing far outside the spline's
        # support, blowing up kurtosis) — skip this step entirely so the bad
        # gradient never reaches the weights. Without this, one bad step
        # permanently poisons the model with NaN parameters for the rest of training.
        print(f"step {step:5d} | non-finite loss ({float(loss)}), skipping update", flush=True)
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(flow.parameters()), max_norm=1.0
    )
    optimizer.step()

    # ── Logging ───────────────────────────────────────────────────────────────
    if step % LOG_EVERY == 0:
        with torch.no_grad():
            rx_bits  = sn.utils.hard_decisions(llr.detach())
            ber      = float((tx_bits.reshape(-1, 2) != rx_bits).float().mean())
            detected = float((kurt > KURT_THRESH).float().mean())

        log_ber.append(ber)
        log_detection.append(detected)
        log_kurtosis.append(float(kurt.mean()))
        log_power.append(float(jam_power))
        log_loss.append(float(loss))

        elapsed = time.time() - t_start
        sps     = (step + 1) / elapsed
        print(f"step {step:5d} | {sps:6.2f} steps/s | loss {loss:.4f} | BER {ber:.3f} "
              f"| det {detected:.3f} | kurt {kurt.mean():.3f} "
              f"| power {float(jam_power):.3f}", flush=True)

        writer.add_scalar("loss/total", float(loss), step)
        writer.add_scalar("loss/ber_term", float(loss_ber), step)
        writer.add_scalar("loss/kurtosis_term", float(loss_k), step)
        writer.add_scalar("ber", ber, step)
        writer.add_scalar("detection_rate", detected, step)
        writer.add_scalar("kurtosis", float(kurt.mean()), step)
        writer.add_scalar("jam_power", float(jam_power), step)
        writer.add_scalar("steps_per_sec", sps, step)

    if (step + 1) % CHECKPOINT_EVERY == 0:
        save_plot(step + 1)
        save_iq_plot()
        save_model(step + 1)

    if (step + 1) % GC_EVERY == 0:
        gc.collect()

# ── Final save ────────────────────────────────────────────────────────────────
save_plot(TOTAL_STEPS)
save_iq_plot()
save_model(TOTAL_STEPS)
writer.close()
