"""
sim06b — Single-subcarrier MAPPO jammer (diagnostic).

Tests whether MAPPO can learn jam=-2*tx and find the detection-avoidance
sweet spot when the action space is tractable (2 real dims instead of 128).

Probe result (sim06/probe_1sc.py) showed:
  - 1SC jamming at scale≤2.0: P(jam)≈0.003 (undetected)
  - 1SC jamming at scale≥5.0: P(jam)≈0.99  (detected)
  - ALL SC random noise at ANY scale: P(jam)≈1.0 (always detected)

This confirms MAPPO's failure in sim06 was due to broadband noise outputs
being instantly detected, not action-space dimensionality per se.

Architecture: simple Gaussian MLP policy (2D doesn't need NSF).
  obs:    [I_tx, Q_tx] of target subcarrier (2 real dims)
  action: [I_jam, Q_jam] of target subcarrier (2 real dims)
  reward: per-SC BER - β·det_penalty - γ·power

Same OFDM chain and frozen CNN detector as sim06.
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
ENTROPY_COEFF   = 0.01       # OK for 2D — doesn't blow up power like 128D
VALUE_COEFF     = 0.5
MAX_GRAD_NORM   = 1.0
N_PPO_EPOCHS    = 4
N_MINIBATCHES   = 4
BETA_DETECT     = 2.0        # linear P(jam) — gradient exists in 1SC regime
GAMMA_POWER     = 0.02
TARGET_SC       = 20         # FFT index of target subcarrier
LOG_EVERY       = 10
CHECKPOINT_EVERY = 50

OBS_DIM    = 2   # [I_tx, Q_tx]
ACTION_DIM = 2   # [I_jam, Q_jam]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Simple Gaussian policy (2D action space) ───────────────────────────────

class GaussianAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def _get_dist(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs):
        dist = self._get_dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs, action):
        dist = self._get_dist(obs)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, obs):
        dist = self._get_dist(obs)
        return dist.entropy().sum(dim=-1)


# ── Centralized critic ─────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, n_agents):
        super().__init__()
        input_dim = n_agents * (OBS_DIM + ACTION_DIM)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Rollout buffer ─────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_agents, batch_size, n_steps):
        self.obs = torch.zeros(n_agents, batch_size, n_steps, OBS_DIM)
        self.actions = torch.zeros(n_agents, batch_size, n_steps, ACTION_DIM)
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


# ── Plotting ───────────────────────────────────────────────────────────────

def save_plot(logs, run_id, out_dir):
    fig, axes = plt.subplots(7, 1, figsize=(10, 21), sharex=True)
    iters = list(range(len(logs['ber'])))

    axes[0].plot(iters, logs['ber'])
    axes[0].set_ylabel('Per-SC BER'); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'sim06b 1-SC MAPPO — run{run_id}\n'
                      f'N_JAM={N_JAMMERS} SC={TARGET_SC} '
                      f'β={BETA_DETECT} γ={GAMMA_POWER}')

    axes[1].plot(iters, logs['full_ber'], color='blue', alpha=0.7)
    axes[1].set_ylabel('Full-frame BER'); axes[1].grid(alpha=0.3)

    axes[2].plot(iters, logs['p_jammed'], color='red')
    axes[2].set_ylabel('P(jammed)'); axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.3)

    axes[3].plot(iters, logs['reward'], color='green')
    axes[3].set_ylabel('Mean reward'); axes[3].grid(alpha=0.3)

    axes[4].plot(iters, logs['power'], color='purple')
    axes[4].set_ylabel('Jam power (1 SC)'); axes[4].grid(alpha=0.3)

    axes[5].plot(iters, logs['critic_loss'], color='orange')
    axes[5].set_ylabel('Critic loss'); axes[5].grid(alpha=0.3)

    axes[6].plot(iters, logs['entropy'], color='teal')
    axes[6].set_ylabel('Policy entropy'); axes[6].set_xlabel('Iteration')
    axes[6].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"run{run_id}.png"), dpi=150)
    plt.close()


def save_iq_plot(agents, device, run_id, out_dir, step_label=""):
    """IQ scatter of jammer outputs on the target subcarrier."""
    n_frames = 64
    with torch.no_grad():
        tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(n_frames)

        fig, axes = plt.subplots(1, N_JAMMERS + 2, figsize=(5 * (N_JAMMERS + 2), 4))

        # TX reference (target SC across all OFDM symbols)
        tx_sc = tx_grid[:, 0, 0, :, TARGET_SC]  # [n, N_OFDM] complex
        ax = axes[0]
        ax.scatter(tx_sc.real.cpu().flatten(), tx_sc.imag.cpu().flatten(),
                   s=8, alpha=0.5)
        ax.set_title('TX (target SC)'); ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        jam_total = torch.zeros_like(tx_sc)
        for i, agent in enumerate(agents):
            agent.eval()
            obs = torch.stack([tx_sc.real, tx_sc.imag], dim=-1)  # [n, N_OFDM, 2]
            obs_flat = obs.reshape(-1, 2)
            action, _ = agent.sample(obs_flat)
            jam_complex = torch.complex(action[:, 0], action[:, 1])
            jam_complex = jam_complex.reshape(tx_sc.shape)
            jam_total += jam_complex

            ax = axes[i + 1]
            ax.scatter(action[:, 0].cpu(), action[:, 1].cpu(), s=2, alpha=0.3)
            ax.set_title(f'Jammer {i}'); ax.set_xlabel('I')
            ax.set_aspect('equal'); ax.grid(alpha=0.3)

        rx_sc = tx_sc + jam_total
        ax = axes[N_JAMMERS + 1]
        ax.scatter(rx_sc.real.cpu().flatten(), rx_sc.imag.cpu().flatten(),
                   s=8, alpha=0.5, c='red')
        ax.set_title('RX (target SC)'); ax.set_xlabel('I')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        plt.suptitle(f'sim06b IQ — run{run_id} {step_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"run{run_id}_iq.png"), dpi=150)
        plt.close()


# ── Per-subcarrier BER ─────────────────────────────────────────────────────

def compute_target_sc_ber(rx_grid, tx_bits, target_eff_idx):
    """BER on only the target subcarrier's data symbols."""
    _, rx_eff = ofdm.demodulate_frame(ofdm.modulator(rx_grid))
    pp = ofdm.resource_grid.pilot_pattern
    data_mask = (pp.mask[0, 0] == 0)  # [N_OFDM, N_EFF_SC]

    # Which OFDM symbols carry data on the target SC?
    target_data_mask = data_mask[:, target_eff_idx]  # [N_OFDM] bool
    n_data_ofdm = target_data_mask.sum().item()

    # Extract target SC across data OFDM symbols
    rx_target = rx_eff[:, 0, 0, target_data_mask, target_eff_idx]  # [B, n_data_ofdm]
    rx_target = rx_target.reshape(rx_target.shape[0], 1, 1, -1)

    llr = ofdm.demapper(rx_target, 1.0)
    rx_bits_target = (llr > 0).float()

    # Find corresponding tx bits
    # tx_bits are ordered by the resource grid data mapping: row-major over
    # (OFDM symbol, effective SC) for data positions only.
    # We need the bit indices for (target_eff_idx) across data OFDM symbols.
    n_eff = ofdm.get_num_effective_subcarriers()
    bits_per_sym = ofdm.BITS_PER_SYMBOL

    # Build flat index mapping: for each data position in (ofdm, eff_sc) order,
    # compute its index in the flat tx_bits array
    data_positions = data_mask.nonzero()  # [num_data, 2] — (ofdm_idx, eff_sc_idx)
    target_bit_indices = []
    for pos_idx in range(data_positions.shape[0]):
        if data_positions[pos_idx, 1].item() == target_eff_idx:
            start = pos_idx * bits_per_sym
            target_bit_indices.extend(range(start, start + bits_per_sym))

    target_bit_indices = torch.tensor(target_bit_indices, device=tx_bits.device)
    tx_bits_target = tx_bits[:, 0, 0, target_bit_indices]  # [B, n_data_ofdm * bps]
    rx_bits_flat = rx_bits_target.reshape(rx_bits_target.shape[0], -1)

    errors = (tx_bits_target != rx_bits_flat).float()
    return errors.mean(dim=-1)  # [B]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector-model', type=str, required=True)
    parser.add_argument('--total-frames', type=int, default=TOTAL_FRAMES)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sim06b", "jammer")
    os.makedirs(out_dir, exist_ok=True)
    existing = [int(f[3:6]) for f in os.listdir(out_dir)
                if f.startswith("run") and f[3:6].isdigit() and f.endswith(".png")]
    run_id = f"{max(existing, default=0) + 1:03d}"
    print(f"Run {run_id}", flush=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb", f"run{run_id}"))

    ofdm.init_device(device)
    n_ofdm = ofdm.N_OFDM_SYMBOLS

    # Find effective SC index for TARGET_SC
    rg = ofdm.resource_grid
    eff_arr = rg.effective_subcarrier_ind
    eff_indices = eff_arr.cpu().numpy() if hasattr(eff_arr, 'cpu') else eff_arr
    target_eff_idx = list(eff_indices).index(TARGET_SC)
    print(f"Target FFT SC: {TARGET_SC}, effective SC idx: {target_eff_idx}", flush=True)

    detector = load_detector(args.detector_model, device=device)
    for p in detector.parameters():
        p.requires_grad = False
    print(f"Frozen detector loaded from {args.detector_model}", flush=True)

    agents = [GaussianAgent().to(device) for _ in range(N_JAMMERS)]
    critic = Critic(N_JAMMERS).to(device)

    actor_params = [p for a in agents for p in a.parameters()]
    optimizer_actor = torch.optim.Adam(actor_params, lr=LR_ACTOR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    total_params = sum(p.numel() for p in actor_params) + sum(p.numel() for p in critic.parameters())
    print(f"Agents: {N_JAMMERS}x GaussianAgent (2D), Critic: CTDE MLP, "
          f"total params: {total_params:,}", flush=True)

    n_iterations = args.total_frames // args.batch_size
    B = args.batch_size

    logs = {k: [] for k in ['ber', 'full_ber', 'p_jammed', 'reward', 'power',
                              'critic_loss', 'entropy', 'actor_loss']}

    print(f"\nTraining: {n_iterations} iterations "
          f"({args.total_frames} frames, batch={B})...\n", flush=True)
    t_start = time.time()

    for iteration in range(n_iterations):
        t_iter = time.time()

        # ── ROLLOUT ────────────────────────────────────────────────────
        buf = RolloutBuffer(N_JAMMERS, B, n_ofdm).to(device)
        all_jam_powers = []

        with torch.no_grad():
            tx_bits, tx_syms, tx_grid, _ = ofdm.generate_ofdm_frame(B)
            rx_grid = tx_grid.clone()

            for t in range(n_ofdm):
                # Observe target SC only
                tx_sc = tx_grid[:, 0, 0, t, TARGET_SC]  # [B] complex
                obs = torch.stack([tx_sc.real, tx_sc.imag], dim=-1)  # [B, 2]

                jam_sc_total = torch.zeros(B, dtype=torch.complex64, device=device)
                global_obs_actions = []

                for i, agent in enumerate(agents):
                    action, lp = agent.sample(obs)
                    buf.obs[i, :, t] = obs
                    buf.actions[i, :, t] = action
                    buf.log_probs[i, :, t] = lp

                    jam_sc = torch.complex(action[:, 0], action[:, 1])
                    jam_sc_total += jam_sc
                    global_obs_actions.extend([obs, action])

                # Inject jam on target SC only
                rx_grid[:, 0, 0, t, TARGET_SC] = tx_sc + jam_sc_total
                all_jam_powers.append(jam_sc_total.abs().pow(2))  # [B]

                # Critic
                global_state = torch.cat(global_obs_actions, dim=-1)
                buf.values[:, t] = critic(global_state)

            # ── FRAME-LEVEL REWARD ─────────────────────────────────────
            rx_time = ofdm.modulator(rx_grid)
            rx_time_flat = rx_time[:, 0, 0]

            p_jammed = detect(detector, rx_time_flat)

            # Per-SC BER (target subcarrier only)
            sc_ber = compute_target_sc_ber(rx_grid, tx_bits, target_eff_idx)

            # Full-frame BER (for logging, not reward)
            _, rx_eff = ofdm.demodulate_frame(rx_time)
            full_ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)

            total_power = torch.stack(all_jam_powers).mean(dim=0)

            reward = sc_ber - BETA_DETECT * p_jammed - GAMMA_POWER * total_power
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

                # Critic
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
        mean_ber = sc_ber.mean().item()
        mean_fber = full_ber.mean().item()
        mean_pj = p_jammed.mean().item()
        mean_rew = reward.mean().item()
        mean_pow = total_power.mean().item()
        mean_cl = epoch_critic_loss / max(n_updates, 1)
        mean_ent = epoch_entropy / max(n_updates, 1)
        mean_al = epoch_actor_loss / max(n_updates, 1)

        logs['ber'].append(mean_ber)
        logs['full_ber'].append(mean_fber)
        logs['p_jammed'].append(mean_pj)
        logs['reward'].append(mean_rew)
        logs['power'].append(mean_pow)
        logs['critic_loss'].append(mean_cl)
        logs['entropy'].append(mean_ent)
        logs['actor_loss'].append(mean_al)

        for k, v in [('sc_ber', mean_ber), ('full_ber', mean_fber),
                      ('p_jammed', mean_pj), ('reward', mean_rew),
                      ('power', mean_pow), ('critic_loss', mean_cl),
                      ('entropy', mean_ent)]:
            writer.add_scalar(k, v, iteration)

        dt = time.time() - t_iter
        fps = B / dt

        if iteration % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            print(f"iter {iteration:4d}/{n_iterations} | "
                  f"scBER={mean_ber:.3f} fBER={mean_fber:.4f} "
                  f"P(jam)={mean_pj:.4f} rew={mean_rew:.3f} pow={mean_pow:.3f} | "
                  f"ent={mean_ent:.2f} c_loss={mean_cl:.4f} | "
                  f"{fps:.1f} fps | {elapsed:.0f}s", flush=True)

        if iteration % CHECKPOINT_EVERY == 0 and iteration > 0:
            ckpt = {"agents": [a.state_dict() for a in agents],
                    "critic": critic.state_dict(), "step": iteration * B}
            torch.save(ckpt, os.path.join(out_dir, f"run{run_id}_model.pt"))
            save_plot(logs, run_id, out_dir)
            save_iq_plot(agents, device, run_id, out_dir,
                         step_label=f"iter {iteration}")

    # ── Final save ─────────────────────────────────────────────────────
    ckpt = {"agents": [a.state_dict() for a in agents],
            "critic": critic.state_dict(), "step": args.total_frames}
    torch.save(ckpt, os.path.join(out_dir, f"run{run_id}_model.pt"))
    save_plot(logs, run_id, out_dir)
    save_iq_plot(agents, device, run_id, out_dir, step_label="final")

    print(f"\nTraining complete in {time.time()-t_start:.1f}s", flush=True)
    print(f"Final: scBER={logs['ber'][-1]:.3f} P(jam)={logs['p_jammed'][-1]:.4f} "
          f"reward={logs['reward'][-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
