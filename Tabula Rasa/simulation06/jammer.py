"""
NSF jammer agent — Neural Spline Flow policy for MAPPO.

Wraps the sim04 (encoder, flow) pattern in a class with sample() and
log_prob() methods needed for PPO's importance ratio and entropy bonus.

Architecture per agent:
  encoder: MLP [OBS_DIM → 64 → CTX_DIM] with ReLU
  flow:    NSF (3 transforms, hidden=[64,64], passes=2)
  output:  ACTION_DIM real values → FFT_SIZE complex jam symbols
"""

import torch
import torch.nn as nn
import zuko

from ofdm import FFT_SIZE

OBS_DIM = 2 * FFT_SIZE     # 128: observe freq-domain tx (64 complex → 128 real)
ACTION_DIM = 2 * FFT_SIZE  # 128: output jam in freq domain (64 complex → 128 real)
CTX_DIM = 64


class JammerAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.ReLU(),
            nn.Linear(64, CTX_DIM), nn.ReLU(),
        )
        self.flow = zuko.flows.NSF(
            features=ACTION_DIM,
            context=CTX_DIM,
            transforms=3,
            hidden_features=[64, 64],
            randperm=True,
            passes=2,
        )

    def _get_dist(self, obs):
        """Get the conditional flow distribution for given observations."""
        ctx = self.encoder(obs)
        return self.flow(ctx)

    def sample(self, obs):
        """Sample jam signal and compute log_prob.

        Args:
            obs: [B, OBS_DIM] real — flattened I/Q of observed tx

        Returns:
            jam_complex: [B, FFT_SIZE] complex — jam signal per subcarrier
            jam_flat:    [B, ACTION_DIM] real — raw action (for log_prob later)
            log_prob:    [B] — log probability of the sampled action
        """
        dist = self._get_dist(obs)
        jam_flat = dist.rsample()
        jam_flat = torch.nan_to_num(jam_flat, nan=0.0, posinf=20.0, neginf=-20.0)
        jam_flat = jam_flat.clamp(-20, 20)

        log_prob = dist.log_prob(jam_flat)
        log_prob = log_prob.clamp(-50, 50)

        jam_complex = torch.complex(
            jam_flat[:, :FFT_SIZE], jam_flat[:, FFT_SIZE:]
        )
        return jam_complex, jam_flat, log_prob

    def log_prob(self, obs, jam_flat):
        """Compute log_prob for a previously sampled action (PPO update).

        Args:
            obs:      [B, OBS_DIM] real
            jam_flat: [B, ACTION_DIM] real — raw action from rollout

        Returns:
            log_prob: [B]
        """
        dist = self._get_dist(obs)
        lp = dist.log_prob(jam_flat)
        return lp.clamp(-50, 50)

    def entropy(self, obs, n_samples=8):
        """Monte Carlo entropy estimate (same as sim03's FlowDist).

        Args:
            obs: [B, OBS_DIM] real

        Returns:
            entropy: [B]
        """
        dist = self._get_dist(obs)
        samples = dist.rsample((n_samples,))        # [n_samples, B, ACTION_DIM]
        return -dist.log_prob(samples).mean(dim=0)   # [B]


# ── Checkpoint I/O ──────────────────────────────────────────────────────────

def save_agents(agents, path, step, extra=None):
    """Save agent checkpoints (compatible with sim04 format).

    Args:
        agents: list of JammerAgent
        path: file path
        step: training step
        extra: optional dict of additional state to save
    """
    state = {
        "agents": [agent.state_dict() for agent in agents],
        "step": step,
        "N_JAMMERS": len(agents),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_agents(path, device='cpu'):
    """Load agents from checkpoint.

    Returns:
        agents: list of JammerAgent (eval mode, on device)
        step: training step from checkpoint
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agents = []
    for state_dict in ckpt["agents"]:
        agent = JammerAgent()
        agent.load_state_dict(state_dict)
        agent.to(device)
        agent.eval()
        agents.append(agent)
    return agents, ckpt["step"]
