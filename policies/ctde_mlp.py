"""
policies/ctde_mlp.py
---------------------
Stage 1 CTDE baseline: parameter-sharing actor + centralized critic.

Architecture
------------
                    ┌─────────────────────────────────────┐
  Global obs        │        ParameterSharingExtractor      │
  (N_jam*(S+3)+1)   │                                       │
  ─────────────────►│  Actor path (decentralized execution) │
                    │    split obs → N_jam local tokens     │
                    │    (B*N_jam, S+4) ─► shared MLP       │
                    │    reshape → (B, N_jam*hidden)         │
                    │         │                             │  latent_dim_pi
                    │         ▼                             │
                    │    action_net (SB3 built-in)          │────► actions (N_jam*N_sub,)
                    │                                       │
                    │  Critic path (centralized training)   │
                    │    full global obs ─► critic MLP      │
                    │         │                             │  latent_dim_vf
                    │         ▼                             │
                    │    value_net (SB3 built-in)           │────► V(s)
                    └─────────────────────────────────────┘

Local obs for jammer i = [SINR(N_sub) | pos_i(3) | step_frac(1)]
  extracted from the flat global obs layout:
    [SINR * N_jam | pos * N_jam | step_frac]

Why this is the Stage 1 weak baseline
--------------------------------------
The actor maps  (N_jam * hidden, )  →  (N_jam * N_sub, )  using a linear
action_net. Internally the hidden state is the CONCATENATION of all per-jammer
features, so the mapping is sensitive to the ORDER of jammers in the obs vector.
Permuting jammer indices produces a different (and incorrect) mapping — that is
the permutation sensitivity we expose in Stage 2.
Stage 3 replaces the concatenation+linear with a Set Transformer or GNN that is
order-invariant by construction.

Usage
-----
    from policies.ctde_mlp import CTDEMlpPolicy
    model = PPO(
        policy=CTDEMlpPolicy,
        env=env,
        policy_kwargs={"n_jammers": N_jam, "n_subcarriers": N_sub},
        ...
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class ParameterSharingExtractor(nn.Module):
    """
    MLP extractor with parameter-sharing actor and centralized critic.

    Parameters
    ----------
    obs_dim       : full global obs dimension = N_jam*(N_sub+3)+1
    n_jammers     : number of jammer agents
    n_subcarriers : OFDM subcarriers (N_sub)
    hidden_size   : hidden layer width for both actor and critic MLPs
    """

    def __init__(
        self,
        obs_dim:       int,
        n_jammers:     int,
        n_subcarriers: int,
        hidden_size:   int = 256,
    ):
        super().__init__()
        self.n_jam     = n_jammers
        self.n_sub     = n_subcarriers
        self.local_dim = n_subcarriers + 3 + 1   # SINR + pos + step_frac = (N_sub+4)

        # Actor: shared weights — applied independently to each jammer's local obs.
        # Two hidden layers; same MLP for all N_jammers.
        self.actor_net = nn.Sequential(
            nn.Linear(self.local_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),    nn.ReLU(),
        )

        # Critic: centralized — sees the full global state.
        # Separate weights from actor; richer view of the environment.
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        # SB3 reads these to determine the input dims of action_net and value_net.
        self.latent_dim_pi = n_jammers * hidden_size
        self.latent_dim_vf = hidden_size

    # ------------------------------------------------------------------

    def _extract_local_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Split the flat global obs into per-jammer local tokens.

        Global obs layout:
            [SINR_0(N_sub) | SINR_1 | ... | pos_0(3) | pos_1 | ... | step_frac(1)]

        Local obs for jammer i:
            [SINR_i(N_sub) | pos_i(3) | step_frac(1)]   shape = (N_sub+4,)
        """
        B    = obs.shape[0]
        N, S = self.n_jam, self.n_sub

        sinr_block = obs[:, :N*S].view(B, N, S)                   # (B, N, S)
        pos_block  = obs[:, N*S : N*(S+3)].view(B, N, 3)          # (B, N, 3)
        step_frac  = obs[:, -1:].unsqueeze(1).expand(B, N, 1)     # (B, N, 1)

        local_obs  = torch.cat([sinr_block, pos_block, step_frac], dim=2)  # (B, N, S+4)
        return local_obs.reshape(B * N, self.local_dim)                    # (B*N, S+4)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        B         = obs.shape[0]
        local_obs = self._extract_local_obs(obs)          # (B*N_jam, local_dim)
        features  = self.actor_net(local_obs)             # (B*N_jam, hidden)
        return features.view(B, self.latent_dim_pi)       # (B, N_jam*hidden)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(obs)                        # (B, hidden)

    def forward(self, obs: torch.Tensor):
        return self.forward_actor(obs), self.forward_critic(obs)


# ---------------------------------------------------------------------------

class CTDEMlpPolicy(ActorCriticPolicy):
    """
    SB3-compatible CTDE policy for N jammers.

    Pass to PPO via:
        PPO(
            policy=CTDEMlpPolicy,
            env=env,
            policy_kwargs=dict(n_jammers=N, n_subcarriers=S),
        )

    n_jammers and n_subcarriers MUST match the environment config.
    hidden_size defaults to 256.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_jammers:     int,
        n_subcarriers: int,
        hidden_size:   int = 256,
        **kwargs,
    ):
        # Store before super().__init__() calls _build() → _build_mlp_extractor()
        self._n_jammers     = n_jammers
        self._n_subcarriers = n_subcarriers
        self._hidden_size   = hidden_size
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ParameterSharingExtractor(
            obs_dim       = self.observation_space.shape[0],
            n_jammers     = self._n_jammers,
            n_subcarriers = self._n_subcarriers,
            hidden_size   = self._hidden_size,
        )
