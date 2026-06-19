import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Independent, Categorical, MixtureSameFamily
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from typing import Optional, Tuple

N_SYMBOLS = 128   # must match jammer_env.py — number of (I,Q) symbols per action
K         = 8     # mixture components PER SYMBOL (deliberately > QPSK's 4 points,
                   # so "4 active modes" would be a discovered result, not a ceiling)


class GMMDist(Distribution):
    """
    SB3-compatible action distribution: a per-symbol Gaussian Mixture Model.

    The action vector has shape (2*N_SYMBOLS,) = 128 (I,Q) pairs. We treat each
    symbol's (I,Q) as an independent 2D random variable drawn from its OWN
    K-component Gaussian mixture (diagonal covariance per component). Symbols
    are independent of each other — same factorisation as sim01/02's diagonal
    Gaussian, just with K Gaussians instead of 1 per symbol.

    SB3's PPO expects a Distribution with:
        .log_prob(actions)  — for policy gradient
        .entropy()          — for entropy bonus
        .sample()           — for rollout collection
        .mode()             — for deterministic eval
    """

    def __init__(self):
        super().__init__()
        self._dist = None   # filled by proba_distribution() each forward pass

    # ── SB3 required: called by ActorCriticPolicy._build() ────────────────────
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        # Not used — GMMPolicy builds its own head (self.gmm_head) and bypasses
        # the default action_net entirely, same pattern as FlowPolicy/FlowDist.
        return nn.Identity(), nn.Parameter(torch.zeros(1), requires_grad=False)

    # ── Called once per forward pass with the raw GMM parameters ──────────────
    def proba_distribution(self, params: torch.Tensor) -> "GMMDist":
        """
        params: (batch, N_SYMBOLS * K * 5) — raw output of GMMPolicy.gmm_head.
        The 5 numbers per (symbol, component) are:
            [0]   mixture logit        (unnormalised log-weight)
            [1:3] mean   (mu_I, mu_Q)
            [3:5] log-std (logsig_I, logsig_Q)
        """
        batch = params.shape[0]
        p = params.view(batch, N_SYMBOLS, K, 5)

        logits   = p[..., 0]                       # (batch, N_SYMBOLS, K)
        means    = p[..., 1:3]                      # (batch, N_SYMBOLS, K, 2)
        log_stds = p[..., 3:5].clamp(-2.0, 0.5)     # keep std in ~[0.14, 1.65] — avoid
                                                     # pathologically sharp components, which
                                                     # made log_prob (summed over 128 symbols)
                                                     # swing wildly between PPO updates (run001:
                                                     # approx_kl 0.1-1.8, clip_fraction ~0.85)
        stds     = log_stds.exp()

        # Categorical over the K components, one per symbol.
        mixture = Categorical(logits=logits)                       # batch_shape (batch, N_SYMBOLS)
        # Each component is a diagonal 2D Gaussian over (I,Q).
        component = Independent(Normal(means, stds), 1)            # batch_shape (batch, N_SYMBOLS, K), event_shape (2,)

        # MixtureSameFamily combines them: batch_shape (batch, N_SYMBOLS), event_shape (2,)
        self._dist = MixtureSameFamily(mixture, component)
        return self

    # ── The four methods PPO actually uses ────────────────────────────────────
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (batch, 2*N_SYMBOLS) -> reshape to (batch, N_SYMBOLS, 2)
        batch = actions.shape[0]
        a = actions.view(batch, N_SYMBOLS, 2)
        # self._dist.log_prob(a) -> (batch, N_SYMBOLS) — one log-density per symbol.
        # Symbols are independent, so total log_prob = sum over symbols.
        return self._dist.log_prob(a).sum(dim=-1)   # (batch,)

    def entropy(self) -> Optional[torch.Tensor]:
        # GMMs have no closed-form entropy -> Monte Carlo estimate:
        # H(p) ~= -E_p[log p(x)]  via 8 samples per batch element
        with torch.no_grad():
            samples = self._dist.sample((8,))               # (8, batch, N_SYMBOLS, 2)
        logp = self._dist.log_prob(samples).sum(dim=-1)      # (8, batch)
        return -logp.mean(dim=0)                             # (batch,)

    def sample(self) -> torch.Tensor:
        # Discrete component choice + Gaussian sample — not reparameterised,
        # but PPO doesn't need rsample (unlike sim03b's direct-gradient setup).
        s = self._dist.sample()              # (batch, N_SYMBOLS, 2)
        return s.reshape(s.shape[0], -1)      # (batch, 2*N_SYMBOLS)

    def mode(self) -> torch.Tensor:
        # Deterministic eval: for each symbol, pick the highest-weight
        # component and return its mean (no sampling noise).
        probs = self._dist.mixture_distribution.probs         # (batch, N_SYMBOLS, K)
        means = self._dist.component_distribution.mean        # (batch, N_SYMBOLS, K, 2)
        best  = probs.argmax(dim=-1)                           # (batch, N_SYMBOLS)
        idx   = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)  # (batch, N_SYMBOLS, 1, 2)
        chosen = torch.gather(means, 2, idx).squeeze(2)        # (batch, N_SYMBOLS, 2)
        return chosen.reshape(chosen.shape[0], -1)             # (batch, 2*N_SYMBOLS)

    # ── SB3 boilerplate — not called directly by PPO but required by ABC ──────
    def actions_from_params(self, params, deterministic=False):
        self.proba_distribution(params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, params) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(params)
        return actions, self.log_prob(actions)


class GMMPolicy(ActorCriticPolicy):
    """
    PPO policy with a per-symbol Gaussian-Mixture action head.

    Architecture:
        obs (256)
          └─ FlattenExtractor
               └─ MLP extractor  [64, 64]  ← shared with MlpPolicy
                    ├─ latent_pi (64) ──→ gmm_head (Linear) → (N_SYMBOLS*K*5) GMM params
                    └─ latent_vf (64) ──→ value_net          → V(s)

    Single feedforward pass — no sequential coupling layers like NSF.
    Only the output distribution changes vs MlpPolicy / FlowPolicy.
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

        latent_dim = self.mlp_extractor.latent_dim_pi   # 64
        out_dim    = N_SYMBOLS * K * 5                  # 128 * 8 * 5 = 5120

        # 2. Build the GMM parameter head conditioned on MLP features
        self.gmm_head = nn.Linear(latent_dim, out_dim)

        # Bias-init the log_std outputs toward the clamp floor (-2.0), giving
        # an initial std ~ 0.22 instead of the default ~1.0. With default init,
        # jam power starts ~2 (std~1, comparable to QPSK's spacing) and kurtosis
        # is already > KURT_THRESH from step one — i.e. the policy starts on the
        # wrong side of the kurtosis penalty cliff and runs001-003 show it never
        # climbs back. Starting near-silent (kurt~-2, matching QPSK's natural
        # kurtosis, reward~-0.05) puts the policy on the SAME side of the cliff
        # as the optimum, so it only needs to learn to dial power UP where it pays.
        with torch.no_grad():
            self.gmm_head.bias.data.view(N_SYMBOLS, K, 5)[..., 3:5].fill_(-1.5)

        # 3. Rebuild the optimizer so it includes gmm_head's parameters
        # (super()._build created the optimizer before self.gmm_head existed)
        self.optimizer = self.optimizer_class(
            self.parameters(),          # now includes self.gmm_head
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> GMMDist:
        """
        Override SB3's Gaussian default.
        Called every forward pass — produces GMM params from current MLP features.
        """
        params = self.gmm_head(latent_pi)        # (batch, N_SYMBOLS*K*5)
        return GMMDist().proba_distribution(params)
