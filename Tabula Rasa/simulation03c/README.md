# Simulation 03c — PPO Jammer with GMM Policy (closed, see verdict)

## What this is
Same lossless QPSK channel + kurtosis detector as sim03/03b. Replaces sim03's NSF (normalizing
flow) action head with a **per-symbol Gaussian Mixture Model (GMM)** action head, trained with
PPO. Built as a candidate "PPO-compatible generative head" for sim04's multi-agent setup, since
NSF's sequential coupling layers are slow and a GMM is a single feedforward pass (K=8 components
per symbol — deliberately > QPSK's 4 points, so "4 active modes" would be a discovered result,
not an architectural ceiling).

## Files
| File | Purpose |
|---|---|
| `jammer_env.py` | identical to sim03's env — N_SYMBOLS=128, KURT_THRESH=-1.0, BETA=2.0, GAMMA=0.02 |
| `gmm_policy.py` | `GMMDist` (per-symbol `MixtureSameFamily` of `Independent(Normal)`) + `GMMPolicy` (SB3 `ActorCriticPolicy` with custom `gmm_head`) |
| `train_ppo.py` | PPO training loop, checkpointing every 10k steps (plots + model) |
| `submit.sh` | sbatch script, CPU-only (`CUDA_VISIBLE_DEVICES=""`) |

## Verdict (2026-06-13): GMM+PPO does not work for this problem — closed

Eight runs (see `artifacts/RUNS.md` for full table) systematically ruled out every PPO-mechanics
hyperparameter:

| Run | Change | Outcome |
|---|---|---|
| 001 | baseline GMM+PPO | unstable: `approx_kl` 0.1-1.8, `clip_fraction`~0.85, kurt~+0.4-0.6, det=1.0 |
| 002 | tighter `log_std` clamp `(-2,0.5)` + `target_kl=0.02` | PPO stable (approx_kl~0.05-0.16) but kurt~0-0.3, det=1.0, unchanged outcome |
| 003 | + `ent_coef=0.01` | no visible change — `entropy_loss` frozen at -390 |
| 004 | bias-init `log_std` to -1.5 (jam starts near-silent) | **big jump**: det 1.0→0.3-0.4, BER 0.25→0.05, kurt pinned at -1.0 (the cliff edge), power 2.6→0.375 |
| 005 | `lr` 1e-4→3e-5 (compensate for ~20x higher gradient sensitivity at smaller std) | `approx_kl` dropped to 0.08-0.14, but still 5-7x over `target_kl`; outcome identical to 004 |
| 006 | `target_kl` 0.02→0.15 | `approx_kl` pinned at ~0.15, still ~2 updates/iteration; outcome identical |
| 007 | `ent_coef` 0.01→0.1 | `entropy_loss` still frozen at -105; outcome identical |
| 008 | `target_kl=None` | `n_updates` jumped 10x (110-140), `approx_kl`~1.5, `clip_fraction`~0.87 — **massive raw parameter movement, zero behavioral change** |

**run008 is the decisive result.** Despite ~10x more gradient updates with huge KL/clip_fraction
(parameters clearly moving a lot), `entropy_loss` and all macro stats (BER, power, kurtosis,
detection) stayed bit-for-bit identical to run004. Diagnosis: **GMM permutation symmetry** — with
K=8 components per symbol, gradient steps can substantially relabel/reshuffle individual
component parameters without changing the *marginal mixture distribution* that's actually
sampled. Combined with `MixtureSameFamily`'s non-reparameterized (score-function) `log_prob`/
`entropy` gradients being high-variance for overlapping components, the optimizer's movement
budget gets absorbed by this symmetry instead of reshaping the output distribution.

**Why sim03b (NSF + direct-gradient) did better** (kurt~-1.30, BER~0.17, best result across all
of sim02/03/03b/03c): a normalizing flow is a bijective transform — no permutation symmetry/
label-switching — and direct-gradient training uses **reparameterized sampling**, giving
low-variance pathwise gradients straight from the loss to the distribution parameters. PPO's
score-function gradient on a GMM has neither property.

## Useful side-finding (independent of GMM): the reward "cliff"
Worked out from `reward = ber - BETA*max(0, kurt-KURT_THRESH) - 0.05 - GAMMA*power`:
- `jam=0` (trivial): `kurt=-2` (QPSK's natural kurtosis) → `kurtosis_excess=0`, `ber=0` →
  **reward = -0.05**.
- Default SB3 init gives jam power≈2, std≈1 (comparable to QPSK's symbol spacing ~1.4) → kurt≈0
  from step one → **already on the wrong side of the `kurt>-1` penalty cliff** (reward≈-2.25 in
  runs 001-003).
- As power increases from 0, reward *increases* (ber rises, kurtosis_excess stays 0) until
  `kurt` crosses -1, then falls off a cliff into the penalty zone. **The true optimum is at the
  cliff edge** (kurt≈-1, higher ber/power than `jam=0`, reward likely +0.3 to +0.5) — run004-008
  sit *at* this edge (kurt pinned ~-1.0) but with BER only ~0.05 (barely better than `jam=0`'s
  BER=0), suggesting they're at the near edge of the cliff, not the actual optimum further along
  it. This cliff-landscape framing is architecture-independent and worth carrying into sim04's
  reward design.

## Next step
Pivot to sim04 (MARL, 2 jammers) using **sim03b's NSF + direct-gradient** approach as the
per-agent policy basis, not GMM+PPO. If PPO/RL is needed for multi-agent credit assignment,
consider NSF with reparameterized policy gradients rather than a GMM action head.
