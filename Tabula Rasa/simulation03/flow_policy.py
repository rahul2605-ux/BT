import torch
import torch.nn as nn
import numpy as np
import zuko
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from typing import Optional, Tuple

class FlowDist(Distribution):
    """
    SB3-compatible action distribution backed by a conditioned zuko flow.

    SB3's PPO expects a Distribution with:
        .log_prob(actions)  — for policy gradient
        .entropy()          — for entropy bonus
        .sample()           — for rollout collection
        .mode()             — for deterministic eval

    A normalizing flow gives us all of these exactly (no approximation on log_prob).
    """

    def __init__(self):
        super().__init__()
        self._dist = None   # filled by proba_distribution() each forward pass

    # ── SB3 required: called by ActorCriticPolicy._build() ────────────────────
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        # SB3 uses the returned module as self.action_net (Gaussian mean layer).
        # We return a dummy Identity — the real network lives in FlowPolicy.flow.
        # The dummy Parameter satisfies SB3's expected (module, parameter) return.
        return nn.Identity(), nn.Parameter(torch.zeros(1), requires_grad=False)

    # ── Called once per forward pass with the conditioned zuko distribution ───
    def proba_distribution(self, conditioned_flow) -> "FlowDist":
        """Receive an already-conditioned zuko distribution and store it."""
        self._dist = conditioned_flow
        return self

    # ── The four methods PPO actually uses ────────────────────────────────────
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # Exact log-prob via change-of-variables — this is the flow's superpower
        return self._dist.log_prob(x)

    def entropy(self) -> Optional[torch.Tensor]:
        # Flows have no closed-form entropy → Monte Carlo estimate:
        # H(p) ≈ -E_p[log p(x)]  via 8 samples per batch element
        with torch.no_grad():
            samples = self._dist.rsample((8,))      # (8, batch, action_dim)
        return -self._dist.log_prob(samples).mean(dim=0)   # (batch,)

    def sample(self) -> torch.Tensor:
        # rsample = reparameterised sample (differentiable, needed for PPO)
        return self._dist.rsample()                 # (batch, action_dim)

    def mode(self) -> torch.Tensor:
        # Flows have no analytical mode → return a sample
        # Used only during deterministic evaluation, not training
        return self._dist.rsample()

    # ── SB3 boilerplate — not called directly by PPO but required by ABC ──────
    def actions_from_params(self, conditioned_flow, deterministic=False):
        self.proba_distribution(conditioned_flow)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, conditioned_flow) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(conditioned_flow)
        return actions, self.log_prob(actions)


class FlowPolicy(ActorCriticPolicy):
    """
    PPO policy with a Neural Spline Flow action head.

    Architecture:
        obs (256)
          └─ FlattenExtractor
               └─ MLP extractor  [64, 64]  ← shared with MlpPolicy
                    ├─ latent_pi (64) ──→ NSF flow (3 transforms) → action (256)
                    └─ latent_vf (64) ──→ value_net              → V(s)

    Only the output distribution changes vs MlpPolicy.
    The MLP trunk, value head, and optimizer are all standard SB3.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            **kwargs
        )
        # _build() is called by super().__init__(), which triggers our override below

    def _build(self, lr_schedule):
        # 1. Build the standard MLP extractor + value net + (dummy) Gaussian head
        super()._build(lr_schedule)

        latent_dim = self.mlp_extractor.latent_dim_pi
        action_dim  = int(np.prod(self.action_space.shape))   # 2 * N_SYMBOLS = 256

        # 2. Build the normalizing flow conditioned on MLP features
        self.flow = zuko.flows.NSF(
            features=action_dim,        # output dimension (action space)
            context=latent_dim,         # conditioning dimension (MLP latent)
            transforms=3,              # number of spline coupling layers
            hidden_features=[64, 64],  # width of each coupling network
            randperm=True,             # randomly permute dims between transforms
            passes=2,                  # coupling-style sampling (2 sequential passes)
                                        # instead of fully autoregressive (passes=None,
                                        # which needs `action_dim` sequential calls per
                                        # rsample/log_prob — this was the ~1fps bottleneck)
        )
        # randperm=True: each coupling layer sees a different ordering of dims,
        # ensuring all dimensions interact across the 3 transforms.

        # 3. Rebuild the optimizer so it includes the flow's parameters
        # (super()._build created the optimizer before self.flow existed)
        self.optimizer = self.optimizer_class(
            self.parameters(),          # now includes self.flow
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> FlowDist:
        """
        Override SB3's Gaussian default.
        Called every forward pass — conditions the flow on current MLP features.
        """
        conditioned = self.flow(latent_pi)      # zuko: (batch, latent) → Distribution
        return FlowDist().proba_distribution(conditioned)
