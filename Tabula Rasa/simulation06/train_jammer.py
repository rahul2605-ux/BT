"""
sim06 Phase 2 — MAPPO jammer training against frozen CNN detector.

Two cooperative NSF jammer agents learn to produce OFDM-domain jam signals
that maximize BER while evading a pretrained, frozen EfficientNet-B0 detector.

MAPPO (Yu et al., NeurIPS 2022):
  - On-policy, clipped surrogate objective
  - Centralized critic (CTDE): sees both agents' obs + actions during training
  - Each actor only sees its own observation at execution
  - NSF flow provides exact log_prob (no approximation)

Reward per frame:
  reward = BER - β·(-log(1 - P(jam) + ε)) - γ·total_jam_power
  Log-shaped detection penalty amplifies gradient near P(jam)≈1.
"""

import os
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

import ofdm
from jammer import JammerAgent, OBS_DIM, ACTION_DIM, save_agents
from detector import load_detector, detect

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
N_JAMMERS       = 2
BATCH_SIZE      = 256       # frames per rollout
TOTAL_FRAMES    = 100_000
LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3
GAMMA_RL        = 0.99      # discount (within episode)
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEFF   = 0.0
VALUE_COEFF     = 0.5
MAX_GRAD_NORM   = 1.0
N_PPO_EPOCHS    = 4
N_MINIBATCHES   = 4
BETA_DETECT     = 0.3       # detection penalty weight (log-scaled)
GAMMA_POWER     = 0.1       # power penalty weight
WARMUP_ITERS    = 100       # linearly ramp β from 0 over this many iterations
LOG_EVERY       = 10        # log every N iterations
CHECKPOINT_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Centralized critic (CTDE) ───────────────────────────────────────────────

class Critic(nn.Module):
    """Centralized value function: sees all agents' obs + actions."""
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
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.values = self.values.to(device)
        self.rewards = self.rewards.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        return self

    def compute_gae(self, last_value):
        """Compute GAE advantages and returns."""
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

def freq_grid_to_obs(freq_grid):
    """Convert OFDM freq grid to jammer observation.

    Args:
        freq_grid: [B, 1, 1, FFT_SIZE] complex (single OFDM symbol)

    Returns:
        obs: [B, OBS_DIM] real (flattened I/Q)
    """
    g = freq_grid[:, 0, 0]  # [B, FFT_SIZE] complex
    return torch.cat([g.real, g.imag], dim=-1)  # [B, 2*FFT_SIZE]


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_plot(logs, run_id, out_dir, total_iters):
    fig, axes = plt.subplots(7, 1, figsize=(10, 21), sharex=True)
    iters = list(range(len(logs['ber'])))

    axes[0].plot(iters, logs['ber'])
    axes[0].set_ylabel('BER'); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'sim06 MAPPO Jammer — run{run_id}\n'
                      f'N_JAM={N_JAMMERS} BATCH={BATCH_SIZE} '
                      f'β={BETA_DETECT} γ_pow={GAMMA_POWER} (log penalty)')

    axes[1].plot(iters, logs['p_jammed'], color='red')
    axes[1].set_ylabel('P(jammed)'); axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, logs['det_penalty'], color='darkred')
    axes[2].set_ylabel('-log(1-P(jam))'); axes[2].grid(alpha=0.3)

    axes[3].plot(iters, logs['reward'], color='green')
    axes[3].set_ylabel('Mean reward'); axes[3].grid(alpha=0.3)

    axes[4].plot(iters, logs['power'], color='purple')
    axes[4].set_ylabel('Total jam power'); axes[4].grid(alpha=0.3)

    axes[5].plot(iters, logs['critic_loss'], color='orange')
    axes[5].set_ylabel('Critic loss'); axes[5].grid(alpha=0.3)

    axes[6].plot(iters, logs['entropy'], color='teal')
    axes[6].set_ylabel('Policy entropy'); axes[6].set_xlabel('Iteration')
    axes[6].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close()


def save_iq_plot(agents, device, run_id, out_dir, step_label=""):
    """Save IQ scatter of jammer outputs on a clean OFDM frame."""
    n_samples = 4
    with torch.no_grad():
        tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(n_samples)
        tx_sym_grid = tx_grid[:, :, :, 0:1, :]  # first OFDM symbol
        obs = freq_grid_to_obs(tx_sym_grid.squeeze(3))

        fig, axes = plt.subplots(1, N_JAMMERS + 2, figsize=(5 * (N_JAMMERS + 2), 4))
        # TX reference
        tx_flat = tx_sym_grid.squeeze(3)[:, 0, 0]  # [n, FFT]
        ax = axes[0]
        ax.scatter(tx_flat.real.cpu().flatten(), tx_flat.imag.cpu().flatten(),
                   s=2, alpha=0.3)
        ax.set_title('TX (clean)'); ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        jam_total = torch.zeros(n_samples, ofdm.FFT_SIZE,
                                dtype=torch.complex64, device=device)
        for i, agent in enumerate(agents):
            agent.eval()
            jam_complex, _, _ = agent.sample(obs)
            jam_total += jam_complex
            ax = axes[i + 1]
            ax.scatter(jam_complex.real.cpu().flatten(),
                       jam_complex.imag.cpu().flatten(), s=2, alpha=0.3)
            ax.set_title(f'Jammer {i}'); ax.set_xlabel('I')
            ax.set_aspect('equal'); ax.grid(alpha=0.3)

        # RX = TX + jam
        rx = tx_flat + jam_total
        ax = axes[N_JAMMERS + 1]
        ax.scatter(rx.real.cpu().flatten(), rx.imag.cpu().flatten(),
                   s=2, alpha=0.3, c='red')
        ax.set_title('RX (jammed)'); ax.set_xlabel('I')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        plt.suptitle(f'sim06 IQ — run{run_id} {step_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"run{run_id}_iq.png"), dpi=150)
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector-model', type=str, required=True)
    parser.add_argument('--total-frames', type=int, default=TOTAL_FRAMES)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim06", "jammer")
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(out_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"Run {run_id}", flush=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb", f"run{run_id}"))

    # Initialize OFDM
    ofdm.init_device(device)
    n_ofdm = ofdm.N_OFDM_SYMBOLS  # 14 steps per episode

    # Load frozen detector
    detector = load_detector(args.detector_model, device=device)
    for p in detector.parameters():
        p.requires_grad = False
    print(f"Frozen detector loaded from {args.detector_model}", flush=True)

    # Create agents + critic
    agents = [JammerAgent().to(device) for _ in range(N_JAMMERS)]
    critic = Critic(N_JAMMERS, OBS_DIM, ACTION_DIM).to(device)

    actor_params = [p for a in agents for p in a.parameters()]
    optimizer_actor = torch.optim.Adam(actor_params, lr=LR_ACTOR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    total_params = sum(p.numel() for p in actor_params) + sum(p.numel() for p in critic.parameters())
    print(f"Agents: {N_JAMMERS}x JammerAgent, Critic: CTDE MLP, "
          f"total params: {total_params:,}", flush=True)

    n_iterations = args.total_frames // args.batch_size
    B = args.batch_size

    logs = {k: [] for k in ['ber', 'p_jammed', 'det_penalty', 'reward', 'power',
                              'critic_loss', 'entropy', 'actor_loss']}

    print(f"\nTraining: {n_iterations} iterations "
          f"({args.total_frames} frames, batch={B})...\n", flush=True)
    t_start = time.time()

    for iteration in range(n_iterations):
        t_iter = time.time()

        # ── ROLLOUT COLLECTION ──────────────────────────────────────────
        buf = RolloutBuffer(N_JAMMERS, B, n_ofdm, OBS_DIM, ACTION_DIM).to(device)
        all_rx_grids = []
        all_tx_bits = None
        all_jam_powers = []

        with torch.no_grad():
            # Generate full OFDM frame
            tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(B)
            all_tx_bits = tx_bits

            # Process per OFDM symbol
            for t in range(n_ofdm):
                tx_sym_grid = tx_grid[:, :, :, t:t+1, :]  # [B,1,1,1,FFT]
                obs = freq_grid_to_obs(tx_sym_grid.squeeze(3))  # [B, OBS_DIM]

                # Each agent samples
                jam_total = torch.zeros(B, ofdm.FFT_SIZE,
                                        dtype=torch.complex64, device=device)
                global_obs_actions = []

                for i, agent in enumerate(agents):
                    jam_complex, jam_flat, lp = agent.sample(obs)
                    buf.obs[i, :, t] = obs
                    buf.actions[i, :, t] = jam_flat
                    buf.log_probs[i, :, t] = lp
                    jam_total += jam_complex
                    global_obs_actions.extend([obs, jam_flat])

                # Apply jamming in frequency domain
                rx_sym = tx_sym_grid.squeeze(3)[:, 0, 0] + jam_total  # [B, FFT]
                all_rx_grids.append(rx_sym)

                # Jam power tracking
                all_jam_powers.append(jam_total.abs().pow(2).mean(dim=-1))  # [B]

                # Critic value
                global_state = torch.cat(global_obs_actions, dim=-1)  # [B, N*(OBS+ACT)]
                buf.values[:, t] = critic(global_state)

            # ── FRAME-LEVEL REWARD ──────────────────────────────────────
            # Reconstruct full rx grid
            rx_grid_full = tx_grid.clone()
            for t in range(n_ofdm):
                rx_grid_full[:, 0, 0, t] = all_rx_grids[t]

            # Modulate to time domain for CNN detector
            rx_time = ofdm.modulator(rx_grid_full)  # [B,1,1,T]
            rx_time_flat = rx_time[:, 0, 0]  # [B, T]

            # CNN detection probability
            p_jammed = detect(detector, rx_time_flat)  # [B]

            # BER from received grid
            _, rx_eff = ofdm.demodulate_frame(rx_time)
            ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)  # [B]

            # Total jam power
            total_power = torch.stack(all_jam_powers).mean(dim=0)  # [B]

            # Reward: log-shaped detection penalty with linear warmup
            det_penalty = -torch.log(1.0 - p_jammed + 1e-6)
            beta_eff = BETA_DETECT * min(1.0, iteration / WARMUP_ITERS)
            reward = ber - beta_eff * det_penalty - GAMMA_POWER * total_power
            for t in range(n_ofdm):
                buf.rewards[:, t] = reward

            # GAE
            last_value = torch.zeros(B, device=device)
            buf.compute_gae(last_value)

        # ── PPO UPDATE ──────────────────────────────────────────────────
        # Flatten time dimension for minibatch sampling
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

                # Actor update (per agent)
                total_actor_loss = torch.tensor(0.0, device=device)
                total_entropy = torch.tensor(0.0, device=device)

                for i, agent in enumerate(agents):
                    obs_mb = buf.obs[i][b_idx, t_idx]
                    act_mb = buf.actions[i][b_idx, t_idx]
                    old_lp = buf.log_probs[i][b_idx, t_idx]
                    adv_mb = buf.advantages[b_idx, t_idx]

                    # Normalize advantages
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    new_lp = agent.log_prob(obs_mb, act_mb)
                    ratio = torch.exp(new_lp - old_lp).clamp(1e-5, 10.0)

                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
                    actor_loss = -torch.min(surr1, surr2).mean()

                    ent = agent.entropy(obs_mb).mean()

                    total_actor_loss += actor_loss - ENTROPY_COEFF * ent
                    total_entropy += ent.detach()

                # Critic update
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

                # Combined backward
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

        # ── LOGGING ─────────────────────────────────────────────────────
        mean_ber = ber.mean().item()
        mean_pj = p_jammed.mean().item()
        mean_dp = det_penalty.mean().item()
        mean_rew = reward.mean().item()
        mean_pow = total_power.mean().item()
        mean_cl = epoch_critic_loss / max(n_updates, 1)
        mean_ent = epoch_entropy / max(n_updates, 1)
        mean_al = epoch_actor_loss / max(n_updates, 1)

        logs['ber'].append(mean_ber)
        logs['p_jammed'].append(mean_pj)
        logs['det_penalty'].append(mean_dp)
        logs['reward'].append(mean_rew)
        logs['power'].append(mean_pow)
        logs['critic_loss'].append(mean_cl)
        logs['entropy'].append(mean_ent)
        logs['actor_loss'].append(mean_al)

        writer.add_scalar('ber', mean_ber, iteration)
        writer.add_scalar('p_jammed', mean_pj, iteration)
        writer.add_scalar('det_penalty', mean_dp, iteration)
        writer.add_scalar('reward', mean_rew, iteration)
        writer.add_scalar('power', mean_pow, iteration)
        writer.add_scalar('critic_loss', mean_cl, iteration)
        writer.add_scalar('entropy', mean_ent, iteration)

        dt = time.time() - t_iter
        fps = B / dt

        if iteration % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            print(f"iter {iteration:4d}/{n_iterations} | "
                  f"BER={mean_ber:.3f} P(jam)={mean_pj:.3f} "
                  f"det={mean_dp:.2f} β_eff={beta_eff:.3f} "
                  f"rew={mean_rew:.3f} pow={mean_pow:.3f} | "
                  f"ent={mean_ent:.1f} c_loss={mean_cl:.4f} | "
                  f"{fps:.1f} fps | {elapsed:.0f}s", flush=True)

        if iteration % CHECKPOINT_EVERY == 0 and iteration > 0:
            save_agents(agents, os.path.join(out_dir, f"run{run_id}_model.pt"),
                        step=iteration * B,
                        extra={"critic": critic.state_dict()})
            save_plot(logs, run_id, out_dir, iteration)
            save_iq_plot(agents, device, run_id, out_dir,
                         step_label=f"iter {iteration}")

    # ── Final save ──────────────────────────────────────────────────────
    save_agents(agents, os.path.join(out_dir, f"run{run_id}_model.pt"),
                step=args.total_frames,
                extra={"critic": critic.state_dict()})
    save_plot(logs, run_id, out_dir, n_iterations)
    save_iq_plot(agents, device, run_id, out_dir, step_label="final")

    print(f"\nTraining complete in {time.time()-t_start:.1f}s", flush=True)
    print(f"Final: BER={logs['ber'][-1]:.3f} P(jam)={logs['p_jammed'][-1]:.3f} "
          f"reward={logs['reward'][-1]:.3f}", flush=True)
    print(f"Artifacts → {out_dir}/run{run_id}_*", flush=True)


if __name__ == "__main__":
    main()
