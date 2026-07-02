"""
sim07 — Blind cooperative MAPPO jammer with causal observation.

Threat model: BLACK-BOX, score-based.
  The jammer accesses the frozen CNN detector ONLY through P(jammed), a scalar
  detection confidence score. No gradients flow through the detector — ever.
  This forces MAPPO (RL), not direct-gradient. The black-box constraint holds
  during training as well as evaluation.

Observation model: CAUSAL DELAY.
  The jammer observes tx[t-1], not tx[t]. At t=0, zeros.
  With i.i.d. QPSK, tx[t-1] is uninformative about tx[t], so the jam=-2*tx
  cancellation shortcut is mathematically unreachable. The jammer must learn
  a blind waveform distribution — a genuinely non-trivial learning problem.

  EXCEPTION — pilots: OFDM symbols 2 and 11 carry deterministic pilot values.
  When the jammer observes tx[t-1] and that happens to be a pilot (at t=3 or
  t=12), it can recognize the known pattern. Any concentration of energy on
  pilot-adjacent symbols is protocol-aware jamming discovered through learning.

Generative model: the NSF learns a largely UNCONDITIONAL stealthy waveform
  distribution. The observation is uninformative for data symbols, so the flow
  is NOT learning a mapping obs→jam — it is learning the shape of a distribution
  to sample from. This is why a normalizing flow fits the blind setting: it can
  represent complex, non-Gaussian waveform distributions with exact log_prob
  for PPO's importance ratio.

Reward: BER - β·(-log(1 - P(jam) + ε)) - γ·power
  Log-shaped detection penalty with linear β warmup.

Supports checkpoint-and-resume for multi-job training.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation06'))
import ofdm
from jammer import JammerAgent, OBS_DIM, ACTION_DIM, save_agents
from detector import load_detector, detect

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
N_JAMMERS       = 2
BATCH_SIZE      = 256
TOTAL_FRAMES    = 100_000
LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3
GAMMA_RL        = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEFF   = 0.0
VALUE_COEFF     = 0.5
MAX_GRAD_NORM   = 1.0
N_PPO_EPOCHS    = 4
N_MINIBATCHES   = 4
BETA_DETECT     = 0.3
GAMMA_POWER     = 0.05
WARMUP_ITERS    = 200
MAX_POWER_PER_AGENT = 1.0
K_ACTIVE_SC     = 1
LOG_EVERY       = 10
CHECKPOINT_EVERY = 25

PILOT_OFDM_INDICES = [2, 11]
PILOT_REACT_TIMESTEPS = {idx + 1 for idx in PILOT_OFDM_INDICES}  # {3, 12}: tx[t-1] is a pilot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Centralized critic (CTDE) ───────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim):
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


# ── Rollout buffer ──────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_agents, batch_size, n_steps, obs_dim, action_dim):
        self.obs = torch.zeros(n_agents, batch_size, n_steps, obs_dim)
        self.actions = torch.zeros(n_agents, batch_size, n_steps, action_dim)
        self.log_probs = torch.zeros(n_agents, batch_size, n_steps)
        self.values = torch.zeros(batch_size, n_steps)
        self.rewards = torch.zeros(batch_size, n_steps)
        self.advantages = torch.zeros(batch_size, n_steps)
        self.returns = torch.zeros(batch_size, n_steps)

    def to(self, device):
        for attr in ['obs', 'actions', 'log_probs', 'values',
                      'rewards', 'advantages', 'returns']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def compute_gae(self, last_value):
        gae = 0.0
        for t in reversed(range(self.rewards.shape[1])):
            if t == self.rewards.shape[1] - 1:
                next_value = last_value
            else:
                next_value = self.values[:, t + 1]
            delta = self.rewards[:, t] + GAMMA_RL * next_value - self.values[:, t]
            gae = delta + GAMMA_RL * GAE_LAMBDA * gae
            self.advantages[:, t] = gae
        self.returns = self.advantages + self.values


# ── Observation helper ──────────────────────────────────────────────────────

def apply_sparsity_and_power_cap(jam_complex):
    """Zero all but the K highest-magnitude subcarriers, then cap mean power.

    Forces the agent into the sparse/low-power regime where the CNN detector's
    P(jam) actually varies (broadband output is detected at any power level).
    """
    mags = jam_complex.abs()
    _, top_idx = mags.topk(K_ACTIVE_SC, dim=-1)
    sparse_mask = torch.zeros_like(mags)
    sparse_mask.scatter_(1, top_idx, 1.0)
    jam_complex = jam_complex * sparse_mask

    agent_power = jam_complex.abs().pow(2).mean(dim=-1, keepdim=True)
    scale = torch.where(
        agent_power > MAX_POWER_PER_AGENT,
        (MAX_POWER_PER_AGENT / agent_power).sqrt(),
        torch.ones_like(agent_power),
    )
    return jam_complex * scale


def freq_grid_to_obs(freq_grid):
    """Convert OFDM freq grid to jammer observation.
    Args:
        freq_grid: [B, 1, 1, FFT_SIZE] complex
    Returns:
        obs: [B, OBS_DIM] real (flattened I/Q)
    """
    g = freq_grid[:, 0, 0]
    return torch.cat([g.real, g.imag], dim=-1)


# ── Plotting ────────────────────────────────────────────────────────────────

def save_plot(logs, run_id, out_dir):
    fig, axes = plt.subplots(8, 1, figsize=(10, 24), sharex=True)
    iters = list(range(len(logs['ber'])))

    axes[0].plot(iters, logs['ber'])
    axes[0].set_ylabel('BER'); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'sim07 Blind MAPPO — run{run_id}\n'
                      f'N_JAM={N_JAMMERS} BATCH={BATCH_SIZE} '
                      f'β={BETA_DETECT} γ={GAMMA_POWER} (causal obs)')

    axes[1].plot(iters, logs['p_jammed'], color='red')
    axes[1].set_ylabel('P(jammed)'); axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, logs['det_penalty'], color='darkred')
    axes[2].set_ylabel('-log(1-P(jam))'); axes[2].grid(alpha=0.3)

    axes[3].plot(iters, logs['reward'], color='green')
    axes[3].set_ylabel('Mean reward'); axes[3].grid(alpha=0.3)

    axes[4].plot(iters, logs['power'], color='purple')
    axes[4].set_ylabel('Total jam power'); axes[4].grid(alpha=0.3)

    axes[5].plot(iters, logs['pilot_power_ratio'], color='magenta')
    axes[5].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[5].set_ylabel('Pilot/Data power'); axes[5].grid(alpha=0.3)

    axes[6].plot(iters, logs['critic_loss'], color='orange')
    axes[6].set_ylabel('Critic loss'); axes[6].grid(alpha=0.3)

    axes[7].plot(iters, logs['entropy'], color='teal')
    axes[7].set_ylabel('Policy entropy'); axes[7].set_xlabel('Iteration')
    axes[7].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close()


def save_iq_plot(agents, device, run_id, out_dir, step_label=""):
    """IQ scatter of jammer outputs on a clean OFDM frame."""
    n_samples = 4
    with torch.no_grad():
        tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(n_samples)
        tx_sym_grid = tx_grid[:, :, :, 0:1, :]
        # Causal: at t=0, obs is zeros (blind)
        obs = torch.zeros(n_samples, OBS_DIM, device=device)

        fig, axes = plt.subplots(1, N_JAMMERS + 2, figsize=(5 * (N_JAMMERS + 2), 4))
        tx_flat = tx_sym_grid.squeeze(3)[:, 0, 0]
        ax = axes[0]
        ax.scatter(tx_flat.real.cpu().flatten(), tx_flat.imag.cpu().flatten(),
                   s=2, alpha=0.3)
        ax.set_title('TX (clean)'); ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        jam_total = torch.zeros(n_samples, ofdm.FFT_SIZE,
                                dtype=torch.complex64, device=device)
        for i, agent in enumerate(agents):
            agent.eval()
            jam_complex, jam_flat, _ = agent.sample(obs)
            jam_complex = apply_sparsity_and_power_cap(jam_complex)
            jam_total += jam_complex
            ax = axes[i + 1]
            ax.scatter(jam_complex.real.cpu().flatten(),
                       jam_complex.imag.cpu().flatten(), s=2, alpha=0.3)
            ax.set_title(f'Jammer {i} (blind)'); ax.set_xlabel('I')
            ax.set_aspect('equal'); ax.grid(alpha=0.3)

        rx = tx_flat + jam_total
        ax = axes[N_JAMMERS + 1]
        ax.scatter(rx.real.cpu().flatten(), rx.imag.cpu().flatten(),
                   s=2, alpha=0.3, c='red')
        ax.set_title('RX (jammed)'); ax.set_xlabel('I')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        plt.suptitle(f'sim07 IQ — run{run_id} {step_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"run{run_id}_iq.png"), dpi=150)
        plt.close()


def save_per_symbol_power_plot(logs, run_id, out_dir):
    """Bar chart of mean jam power per OFDM symbol index."""
    if not logs['per_sym_power']:
        return
    per_sym = np.array(logs['per_sym_power'])
    mean_per_sym = per_sym.mean(axis=0)  # [N_OFDM]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['magenta' if i in PILOT_OFDM_INDICES else 'steelblue'
              for i in range(len(mean_per_sym))]
    ax.bar(range(len(mean_per_sym)), mean_per_sym, color=colors)
    ax.set_xlabel('OFDM symbol index')
    ax.set_ylabel('Mean jam power')
    ax.set_title(f'sim07 run{run_id} — per-symbol jam power '
                 f'(magenta = pilot symbols)')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}_persym.png"), dpi=150)
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector-model', type=str, required=True)
    parser.add_argument('--total-frames', type=int, default=TOTAL_FRAMES)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim07", "jammer")
    os.makedirs(out_dir, exist_ok=True)

    # ── Run ID ─────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        run_id = ckpt['run_id']
        start_iter = ckpt['iteration'] + 1
        print(f"Resuming run {run_id} from iteration {start_iter}", flush=True)
    else:
        existing = [int(f[3:6]) for f in os.listdir(out_dir)
                    if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
        run_id = f"{max(existing, default=0) + 1:03d}"
        start_iter = 0
        print(f"Run {run_id}", flush=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb", f"run{run_id}"))

    # ── Init OFDM + detector ───────────────────────────────────────────
    ofdm.init_device(device)
    n_ofdm = ofdm.N_OFDM_SYMBOLS

    detector = load_detector(args.detector_model, device=device)
    for p in detector.parameters():
        p.requires_grad = False
    print(f"Frozen detector loaded (black-box: score only, no gradients)",
          flush=True)

    # ── Create / restore agents + critic ───────────────────────────────
    agents = [JammerAgent().to(device) for _ in range(N_JAMMERS)]
    critic = Critic(N_JAMMERS, OBS_DIM, ACTION_DIM).to(device)

    actor_params = [p for a in agents for p in a.parameters()]
    optimizer_actor = torch.optim.Adam(actor_params, lr=LR_ACTOR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    if args.resume:
        for i, sd in enumerate(ckpt['agents']):
            agents[i].load_state_dict(sd)
        critic.load_state_dict(ckpt['critic'])
        optimizer_actor.load_state_dict(ckpt['optimizer_actor'])
        optimizer_critic.load_state_dict(ckpt['optimizer_critic'])
        logs = ckpt['logs']
        print(f"Restored agents, critic, optimizers, logs", flush=True)
    else:
        logs = {k: [] for k in [
            'ber', 'p_jammed', 'det_penalty', 'reward', 'power',
            'pilot_power_ratio', 'critic_loss', 'entropy', 'actor_loss',
            'per_sym_power',
        ]}

    total_params = sum(p.numel() for p in actor_params) + \
                   sum(p.numel() for p in critic.parameters())
    print(f"Agents: {N_JAMMERS}x JammerAgent (NSF, blind/causal), "
          f"Critic: CTDE MLP, total params: {total_params:,}", flush=True)

    n_iterations = args.total_frames // args.batch_size
    B = args.batch_size

    print(f"\nTraining: iters {start_iter}..{n_iterations} "
          f"({args.total_frames} frames, batch={B})\n"
          f"Observation: CAUSAL tx[t-1] (blind for data, pilot-aware at t=3,12)\n"
          f"Threat model: black-box (score-based, no detector gradients)\n",
          flush=True)
    t_start = time.time()

    for iteration in range(start_iter, n_iterations):
        t_iter = time.time()

        # ── ROLLOUT COLLECTION ─────────────────────────────────────────
        buf = RolloutBuffer(N_JAMMERS, B, n_ofdm, OBS_DIM, ACTION_DIM).to(device)
        all_rx_grids = []
        all_tx_bits = None
        all_jam_powers = []
        per_sym_jam_power = []

        with torch.no_grad():
            tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(B)
            all_tx_bits = tx_bits

            # ── HELD DATA-SYMBOL ACTION ─────────────────────────────────
            # Sample ONE jam waveform per agent, used at every timestep where
            # tx[t-1] is i.i.d. data (uninformative for tx[t]). Resampling
            # fresh every OFDM symbol — even from an uninformative obs — made
            # each agent hop to a different subcarrier at every symbol; over
            # a 14-symbol frame that reconstructs a multi-tone/broadband-like
            # spectrogram footprint and gets caught, even though only one SC
            # is active at any single instant (confirmed via direct probe,
            # see Tabula Rasa/README.md sim07 run history). Holding the
            # action constant across data symbols keeps the spectral
            # footprint a single, temporally-stable tone.
            obs0 = torch.zeros(B, OBS_DIM, device=device)
            data_jam_complex, data_jam_flat, data_lp = [], [], []
            for agent in agents:
                jc, jf, lp = agent.sample(obs0)
                jc = apply_sparsity_and_power_cap(jc)
                data_jam_complex.append(jc)
                data_jam_flat.append(jf)
                data_lp.append(lp)

            for t in range(n_ofdm):
                # ── CAUSAL OBSERVATION: tx[t-1], not tx[t] ─────────
                # At t=0: no history → zeros (fully blind) → held data action
                # At t=3,12: tx[t-1] is a pilot symbol (t-1 ∈ {2,11})
                #   → recognizable known pattern → fresh, pilot-conditioned
                #   sample, enabling protocol-aware jamming
                # All other t: tx[t-1] is i.i.d. QPSK data → held data action
                is_pilot_reactive = t in PILOT_REACT_TIMESTEPS
                if is_pilot_reactive:
                    prev_grid = tx_grid[:, :, :, t-1:t, :]
                    obs = freq_grid_to_obs(prev_grid.squeeze(3))
                else:
                    obs = obs0

                jam_total = torch.zeros(B, ofdm.FFT_SIZE,
                                        dtype=torch.complex64, device=device)
                global_obs_actions = []

                for i, agent in enumerate(agents):
                    if is_pilot_reactive:
                        jam_complex, jam_flat, lp = agent.sample(obs)
                        jam_complex = apply_sparsity_and_power_cap(jam_complex)
                    else:
                        jam_complex = data_jam_complex[i]
                        jam_flat = data_jam_flat[i]
                        lp = data_lp[i]
                    buf.obs[i, :, t] = obs
                    buf.actions[i, :, t] = jam_flat
                    buf.log_probs[i, :, t] = lp
                    jam_total += jam_complex
                    global_obs_actions.extend([obs, jam_flat])

                rx_sym = tx_grid[:, 0, 0, t] + jam_total
                all_rx_grids.append(rx_sym)

                sym_power = jam_total.abs().pow(2).mean(dim=-1)  # [B]
                all_jam_powers.append(sym_power)
                per_sym_jam_power.append(sym_power.mean().item())

                global_state = torch.cat(global_obs_actions, dim=-1)
                buf.values[:, t] = critic(global_state)

            # ── FRAME-LEVEL REWARD ─────────────────────────────────
            rx_grid_full = tx_grid.clone()
            for t in range(n_ofdm):
                rx_grid_full[:, 0, 0, t] = all_rx_grids[t]

            rx_time = ofdm.modulator(rx_grid_full)
            rx_time_flat = rx_time[:, 0, 0]

            # Black-box: scalar score only, @torch.no_grad enforced in detect()
            p_jammed = detect(detector, rx_time_flat)

            _, rx_eff = ofdm.demodulate_frame(rx_time)
            ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)

            total_power = torch.stack(all_jam_powers).mean(dim=0)

            det_penalty = -torch.log(1.0 - p_jammed + 1e-6)
            beta_eff = BETA_DETECT * min(1.0, iteration / WARMUP_ITERS)
            reward = ber - beta_eff * det_penalty - GAMMA_POWER * total_power
            for t in range(n_ofdm):
                buf.rewards[:, t] = reward

            last_value = torch.zeros(B, device=device)
            buf.compute_gae(last_value)

        # ── PPO UPDATE ─────────────────────────────────────────────────
        total_steps = B * n_ofdm
        mb_size = total_steps // N_MINIBATCHES

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_entropy = 0.0
        n_updates = 0

        for _ in range(N_PPO_EPOCHS):
            indices = torch.randperm(total_steps, device=device)

            for mb_start in range(0, total_steps, mb_size):
                mb_idx = indices[mb_start:mb_start + mb_size]
                b_idx = mb_idx // n_ofdm
                t_idx = mb_idx % n_ofdm

                total_actor_loss = torch.tensor(0.0, device=device)
                total_entropy = torch.tensor(0.0, device=device)

                for i, agent in enumerate(agents):
                    obs_mb = buf.obs[i][b_idx, t_idx]
                    act_mb = buf.actions[i][b_idx, t_idx]
                    old_lp = buf.log_probs[i][b_idx, t_idx]
                    adv_mb = buf.advantages[b_idx, t_idx]

                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    new_lp = agent.log_prob(obs_mb, act_mb)
                    ratio = torch.exp(new_lp - old_lp).clamp(1e-5, 10.0)

                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
                    actor_loss = -torch.min(surr1, surr2).mean()

                    ent = agent.entropy(obs_mb).mean()

                    total_actor_loss += actor_loss - ENTROPY_COEFF * ent
                    total_entropy += ent.detach()

                global_states = []
                for i in range(N_JAMMERS):
                    global_states.extend([
                        buf.obs[i][b_idx, t_idx],
                        buf.actions[i][b_idx, t_idx]
                    ])
                global_state_mb = torch.cat(global_states, dim=-1)
                value_pred = critic(global_state_mb)
                returns_mb = buf.returns[b_idx, t_idx]
                critic_loss = F.mse_loss(value_pred, returns_mb)

                loss = total_actor_loss + VALUE_COEFF * critic_loss

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                optimizer_actor.step()
                optimizer_critic.step()

                epoch_actor_loss += total_actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += total_entropy.item() / N_JAMMERS
                n_updates += 1

        # ── LOGGING ────────────────────────────────────────────────────
        mean_ber = ber.mean().item()
        mean_pj = p_jammed.mean().item()
        mean_dp = det_penalty.mean().item()
        mean_rew = reward.mean().item()
        mean_pow = total_power.mean().item()
        mean_cl = epoch_critic_loss / max(n_updates, 1)
        mean_ent = epoch_entropy / max(n_updates, 1)
        mean_al = epoch_actor_loss / max(n_updates, 1)

        # Pilot vs data power ratio
        pilot_power = np.mean([per_sym_jam_power[i] for i in PILOT_OFDM_INDICES])
        data_indices = [i for i in range(n_ofdm) if i not in PILOT_OFDM_INDICES]
        data_power = np.mean([per_sym_jam_power[i] for i in data_indices])
        pilot_ratio = pilot_power / max(data_power, 1e-8)

        logs['ber'].append(mean_ber)
        logs['p_jammed'].append(mean_pj)
        logs['det_penalty'].append(mean_dp)
        logs['reward'].append(mean_rew)
        logs['power'].append(mean_pow)
        logs['pilot_power_ratio'].append(pilot_ratio)
        logs['critic_loss'].append(mean_cl)
        logs['entropy'].append(mean_ent)
        logs['actor_loss'].append(mean_al)
        logs['per_sym_power'].append(per_sym_jam_power)

        for k, v in [('ber', mean_ber), ('p_jammed', mean_pj),
                      ('det_penalty', mean_dp), ('reward', mean_rew),
                      ('power', mean_pow), ('pilot_power_ratio', pilot_ratio),
                      ('critic_loss', mean_cl), ('entropy', mean_ent)]:
            writer.add_scalar(k, v, iteration)

        dt = time.time() - t_iter
        fps = B / dt

        if iteration % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            print(f"iter {iteration:4d}/{n_iterations} | "
                  f"BER={mean_ber:.3f} P(jam)={mean_pj:.3f} "
                  f"β_eff={beta_eff:.3f} rew={mean_rew:.3f} "
                  f"pow={mean_pow:.2f} pilot_r={pilot_ratio:.2f} | "
                  f"ent={mean_ent:.1f} c_loss={mean_cl:.3f} | "
                  f"{fps:.1f} fps | {elapsed:.0f}s", flush=True)

        if iteration % CHECKPOINT_EVERY == 0 and iteration > 0:
            ckpt = {
                'run_id': run_id,
                'iteration': iteration,
                'agents': [a.state_dict() for a in agents],
                'critic': critic.state_dict(),
                'optimizer_actor': optimizer_actor.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
                'logs': logs,
                'step': iteration * B,
            }
            ckpt_path = os.path.join(out_dir, f"run{run_id}_ckpt.pt")
            torch.save(ckpt, ckpt_path)
            save_plot(logs, run_id, out_dir)
            save_iq_plot(agents, device, run_id, out_dir,
                         step_label=f"iter {iteration}")
            save_per_symbol_power_plot(logs, run_id, out_dir)

    # ── Final save ─────────────────────────────────────────────────────
    ckpt = {
        'run_id': run_id,
        'iteration': n_iterations - 1,
        'agents': [a.state_dict() for a in agents],
        'critic': critic.state_dict(),
        'optimizer_actor': optimizer_actor.state_dict(),
        'optimizer_critic': optimizer_critic.state_dict(),
        'logs': logs,
        'step': args.total_frames,
    }
    torch.save(ckpt, os.path.join(out_dir, f"run{run_id}_ckpt.pt"))
    save_agents(agents, os.path.join(out_dir, f"run{run_id}_model.pt"),
                step=args.total_frames,
                extra={"critic": critic.state_dict()})
    save_plot(logs, run_id, out_dir)
    save_iq_plot(agents, device, run_id, out_dir, step_label="final")
    save_per_symbol_power_plot(logs, run_id, out_dir)

    print(f"\nTraining complete in {time.time()-t_start:.1f}s", flush=True)
    print(f"Final: BER={logs['ber'][-1]:.3f} P(jam)={logs['p_jammed'][-1]:.3f} "
          f"reward={logs['reward'][-1]:.3f} pilot_ratio={logs['pilot_power_ratio'][-1]:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
