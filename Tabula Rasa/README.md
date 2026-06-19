# Tabula Rasa — Jamming Simulations

Fresh restart of baseline experiments. Goal: build intuition step by step before reintroducing RL agents, fading channels, and multi-antenna setups. Each simulation adds exactly one layer of complexity.

**Hard rule:** only library code (Sionna, SB3, gymnasium, scipy). No reuse from the old project.

**Stack:** Python, Sionna 2.x (`sionna.phy`, PyTorch backend), stable-baselines3, gymnasium, numpy, matplotlib.

---

## Simulation 00 — Lossless channel, no learning

**File:** `simulation00/baseline_lossless.py`

**What it is:** Purely observational. No RL, no training. Measures what happens when a fixed max-power jammer turns on mid-episode.

**Scenario:**
- 2 legitimate QPSK users, independent TX→RX pairs
- 1 jammer: silent for t=0..4, max-power Gaussian noise for t=5..9 (10 timesteps total)
- Channel: lossless — `rx = tx + jam`, no noise, gain = 1
- Detection: power threshold — flag if `mean|rx|² > 3.0`

**Key parameters:**
| Parameter | Value |
|---|---|
| N_TIMESTEPS | 10 |
| JAMMER_START | 5 |
| N_LEGIT | 2 |
| N_SYMBOLS | 512 |
| JAMMER_POWER | 50.0 |
| DET_THRESH | 3.0 |

**Results:**
- t=0..4: BER=0, power≈1.0, 0 users detect
- t=5..9: BER≈0.5, power≈51.0, 2 users detect

**Output:** `simulation00/baseline_lossless.png` — 3-panel: BER, received power, # users detected

**How to run:**
```bash
cd "tabula rasa/simulation00"
python baseline_lossless.py
```

---

## Simulation 01 — PPO jammer, power threshold detector

**Files:** `simulation01/jammer_env.py`, `simulation01/train_ppo.py`

**What it is:** First trainable scenario. A PPO agent learns to jam a single TX→RX pair while staying undetected by a power threshold detector.

**Scenario:**
- 1 TX, 1 RX, 1 PPO-trained jammer
- Channel: lossless — `rx = tx + jam`
- Detection: power threshold (`mean|rx|² > DET_THRESH`)
- Jammer policy: SB3 `MlpPolicy` — diagonal Gaussian over the action space (this IS the generative model for now)

**Key parameters:**
| Parameter | Value | Notes |
|---|---|---|
| N_SYMBOLS | 16 | symbols per step, action space = 32 dims |
| DET_THRESH | 3.0 | linear power |
| BETA | 0.5 | detection penalty weight |
| idle penalty | 0.05 | per-step cost to prevent all-zeros policy |
| TOTAL_STEPS | 200,000 | |
| action space | Box(-10, 10, (32,)) | SB3 requires finite bounds |

**Reward:**
```
r = BER  −  BETA · detected  −  0.05
```
- `BER`: jamming effectiveness (want high)
- `BETA · detected`: stealth penalty (want low)
- `0.05`: idle cost — forces agent away from zero-power trivial solution

**Observation:** flattened TX symbols overheard by jammer → `[I₀, Q₀, ..., I₁₅, Q₁₅]` shape `(32,)`

**Action:** jammer's transmitted signal → `(32,)` → reshaped to complex `(16,)` in step()

**Signal chain per step:**
```
tx_bits  = BinarySource([N_SYMBOLS, 2])
tx_syms  = Mapper(tx_bits).squeeze()
rx_syms  = tx_syms + jam_syms            ← lossless channel
detected = mean|rx|² > DET_THRESH
llr      = Demapper(rx_syms.unsqueeze(-1), 1e-10)
rx_bits  = hard_decisions(llr)
BER      = mean(tx_bits ≠ rx_bits)
```

**Results (run004, 200k steps):**
- BER converged to ~0.19 (started ~0.24)
- Detection rate dropped from ~45% → ~10%
- Jammer power settled at ~2.2 (just below threshold)
- IQ scatter: Gaussian cloud throughout — no structure discovered

**Why BER ~0.19 is near-optimal:**
- TX signal contributes ~1.0 to received power
- Detection threshold = 3.0 → jammer budget ≈ 2.0
- Theoretical max BER at jammer power 2.0 on QPSK ≈ Q(√0.5) ≈ 0.24
- Agent found the power sweet spot: near-optimal BER with only ~10% detection rate

**Known limitation:** power threshold detection is too simple — only strategy available is power tuning.
The diagonal Gaussian policy can only produce Gaussian clouds — no structure will emerge.

**Outputs:** `simulation01/runs/run00N.png` (training curves), `run00N_iq.png` (IQ scatter snapshots)

**How to run:**
```bash
cd "tabula rasa/simulation01"
python train_ppo.py
```

---

## Simulation 02 — Kurtosis detector

**Files:** `simulation02/jammer_env.py`, `simulation02/train_ppo.py`

**What it is:** Same lossless TX→RX→jammer setup as sim01, but the power-threshold detector is
replaced with a kurtosis-based detector (`scipy.stats.kurtosis`). QPSK has excess kurtosis ≈ -2;
Gaussian noise has excess kurtosis = 0. The detector flags `kurtosis(rx) > KURT_THRESH`.

**Key parameters:** N_SYMBOLS=128, BETA=0.5, KURT_THRESH=-1.0, GAMMA=0.02 (binary detection penalty)

**Policy:** SB3 `MlpPolicy` (diagonal Gaussian) — same as sim01.

**Results (run001, run002):**
- Detection rate flat at 100% regardless of training
- Kurtosis stuck around -0.25 (Gaussian-ish), never approaches QPSK's -2
- A diagonal Gaussian policy can only ever produce Gaussian-shaped IQ clouds — it is
  *structurally incapable* of producing QPSK-like (sub-Gaussian) statistics, no matter how
  training proceeds. The agent gives up on stealth (run002: jam power rises 1.0→2.5).

**Conclusion:** the kurtosis detector is unbeatable by a Gaussian policy. This motivates the
normalizing-flow policy upgrade in sim03 — the action distribution itself needs to be able to
represent non-Gaussian (e.g. bimodal/QPSK-like) shapes.

**Outputs:** `artifacts/sim02/run00N.png`

---

## Simulation 03 — Normalizing flow (NSF) policy

**Files:** `simulation03/jammer_env.py`, `simulation03/train_ppo.py`, `simulation03/flow_policy.py`

**What it is:** Replaces SB3's diagonal-Gaussian action head with a Neural Spline Flow (NSF, via
`zuko`) conditioned on the PPO MLP latent (`FlowPolicy` / `FlowDist` in `flow_policy.py`). The MLP
trunk, value head, and PPO optimizer are otherwise standard SB3. `FlowDist` provides exact
`log_prob` (change-of-variables) and a Monte-Carlo `entropy()` estimate (8 rsamples), so it's a
drop-in for everything PPO needs.

**Key parameters:** N_SYMBOLS=128, BETA=2.0, GAMMA=0.02, continuous kurtosis penalty,
KURT_THRESH=-1.0, NSF: 3 transforms, hidden=[64,64], `passes=2`, total_steps=50k.

**Run history (run001-004):** all four runs used `FlowPolicy`. Detection stayed flat (~85-100%),
kurtosis stuck near -0.25 (same Gaussian-shaped wall as sim02) — N=16 (run001-003) was also too
noisy for the kurtosis estimate to give a useful gradient/reward signal; N=128 (run004) fixed the
estimator noise but kurtosis still didn't move.

**Performance bottleneck found and fixed:** `zuko.flows.NSF` defaults to `passes=None`
(fully autoregressive MAF) — sampling/log_prob requires `action_dim` (=256) sequential
hypernetwork calls per transform × 3 transforms ≈ 768 sequential calls per step, ~20s/step
on CPU (~1fps). This made FlowPolicy too slow/costly to train for useful step counts (jobs
93396/93407 hit the time limit after ~500 steps). **Fix:** added `passes=2` (coupling-style,
RealNVP-like — 2 sequential passes per transform instead of 256) — ~44x speedup
(~20s/step → ~0.45s/step), while still an exact-likelihood flow.

**Current status:** the working tree currently has `train_ppo.py` reverted to plain
`PPO("MlpPolicy", ...)` as a stopgap (job 93423, produced run004, still Gaussian-shaped IQ —
not representative of NSF). With `passes=2` now applied to `flow_policy.py`, **re-running sim03
with `FlowPolicy` is the natural next step** — it should now be cheap enough to actually test
whether NSF can escape the kurtosis wall that a Gaussian policy can't.

**Outputs:** `artifacts/sim03/run00N.png`, `run00N_iq.png`

---

## Simulation 03b — Direct-gradient generative jammer (no RL)

**Files:** `simulation03b/train.py`, `simulation03b/submit.sh`

**What it is:** Same lossless QPSK channel and kurtosis detector as sim03, but trained with
**direct backprop** (no PPO/RL). A small MLP encoder + the same NSF flow architecture
(3 transforms, hidden=[64,64], `passes=2`) generate jam symbols directly; the loss is
fully differentiable end-to-end:

```
loss = soft_BER (BCE with flipped labels) + LAMBDA * relu(kurtosis(rx) - KURT_THRESH) + GAMMA * jam_power
```

Since the channel is deterministic and kurtosis is differentiable, no RL is needed — this is the
cleanest possible test of "can a generative model alone push `kurtosis(rx)` toward QPSK's -2
while keeping BER high."

**Fair comparison constraint:** kurtosis must NOT be in the observation (only `tx_syms` is) —
the generative model's only structural advantage over PPO is that its loss is differentiable,
not extra information.

**Key parameters:** N_SYMBOLS=128, BATCH_SIZE=64, LR=1e-3, LAMBDA=2.0, GAMMA=0.02,
KURT_THRESH=-1.0, TOTAL_STEPS=5000.

**run001 result + bug found:** with `demapper(rx, no=1e-10)`, the system collapsed to the trivial
"do nothing" solution — `jam_power → 0` by step 1250, `BER → 0`, loss pinned at the
`clamp(-20,20)` ceiling for the remaining 3750 steps. **Root cause:** `no=1e-10` makes the
app-demapper's LLRs (`exp(-|y-s|²/no)`) saturate to ±∞ for any nonzero `rx-tx` deviation, so
`binary_cross_entropy_with_logits` has ~zero gradient once `jam_power` shrinks even slightly —
nothing can pull the optimizer back out, while the kurtosis and power penalty terms keep
rewarding `jam_power → 0`. **Fix:** `no=1.0`, clamp tightened to `(-10,10)`.

**Diagnostics added:** IQ scatter (`run00N_iq.png`, early/mid/late snapshots vs QPSK reference)
and TensorBoard logging (`artifacts/sim03b/tb/run00N/`) — loss terms, BER, detection, kurtosis,
power, steps/s. TensorBoard requires the `tensorboard` package in `bt_env`
(`pip install tensorboard`); resolved after job 98440 failed on this. View via VSCode
Remote-SSH: run `tensorboard --logdir artifacts/sim03b/tb --port 6006` on the cluster, then
open the auto-forwarded port from the Ports tab — no manual `ssh -L`/fingerprint needed.

**run002 result (job 98441, `no=1.0` fix, 5000 steps, ~7.7 steps/s):** fix worked — no collapse
to the clamp ceiling. Found a non-trivial local optimum: `jam_power` settled ~0.3-0.6 (down from
random-init ~2.75), `kurtosis` ~-1.25 to -1.5 (under `KURT_THRESH=-1.0` → ~0% detection),
`BER` oscillating 0.05-0.15 with a slow upward drift in the last ~1500-2000 steps (not yet
plateaued). IQ scatter: collapsed to a single small unimodal blob near the origin
(kurtosis≈-1.3, consistent with a uniform-like/platykurtic shape) — not QPSK's 4-cluster
structure, but already enough to evade the kurtosis detector.

**run003 (job 98470, `TOTAL_STEPS` bumped to 20000, ~7.46 steps/s):** early portion (steps
0-1500) mirrored run002 — BER spike to ~0.43 then drop near 0, kurtosis dipping to ~-1.75 then
recovering to ~-1.25, jam_power dropping to ~0.1 then slowly rising. By step 10950: `loss
1.8152 | BER 0.156 | kurt -1.301 | power 0.864` — i.e. *better* than run002 (higher power,
similar kurtosis/BER), still trending. **Then it diverged to NaN** between step 10950 and
11000 (`loss nan | kurt nan | power nan`, BER settling at ~0.5 = random-guessing level,
consistent with NaN jam symbols) and stayed NaN for the rest of the run.

**Root cause + fix (job 98470 cancelled, fixed in `train.py`):** NSF's rational-quadratic
splines can occasionally extrapolate to huge values outside their support; once `jam_flat`
gets large enough, `excess_kurtosis_batch`'s `m4/m2²` ratio overflows to `inf/inf = nan`. One
bad step then permanently poisons the weights with NaN (NaN propagates forever once it's in
the parameters). **Fix applied:**
1. `jam_flat = jam_flat.clamp(-20, 20)` right after sampling — bounds kurtosis inputs while
   leaving plenty of headroom above the `|jam|≈1.4` needed for the `jam=-2*tx` optimum.
2. `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` before `optimizer.step()`.
3. Skip the optimizer step entirely (`continue`) if `loss` is non-finite, so a rare bad batch
   can never poison the weights.

**run004 (job 98478, `TOTAL_STEPS=20000`) — froze again at step 7664:** same symptom as run003
— NaN right as `power` crosses ~0.85-0.92 and `kurt` ~-1.27 to -1.32 (both runs hit this exact
"edge" region). Root cause refined: `flow(ctx).rsample()` can return literal `NaN` entries
(likely a near-zero spline-bin-width in NSF's hypernetwork causing a `0/0`/`x/0` inside the
rational-quadratic transform) — **`.clamp(-20,20)` does NOT fix `NaN`** (`clamp(nan,...)==nan`
in PyTorch), only `Inf`. So the loss went non-finite, the "skip update" guard froze the weights
at that exact broken point, and all ~12,000 remaining steps were wasted on
`non-finite loss (nan), skipping update`.

**Fix applied (job 98478 cancelled → run005, job 98483):** added
`jam_flat = torch.nan_to_num(jam_flat, nan=0.0, posinf=20.0, neginf=-20.0)` *before* the clamp.
This replaces any stray NaN/Inf entries with finite values (zero-gradient at those entries, so
they don't poison the update) while the rest of the batch still provides a valid gradient —
should let training push through the `power≈0.9` instability region instead of freezing there.
Re-running as run005 (job 98483); if it still freezes past `power≈0.9`, the next step is
lowering LR (currently 1e-3) and/or adding `weight_decay` to Adam, since two independent runs
hitting the *same* power/kurtosis region suggests the flow's hypernetwork weights are drifting
toward a structurally degenerate spline configuration around there, not just a one-off rare
sample.

**Theoretical BER ceiling (derived, not yet reached):** the global optimum of the loss is
`jam = -2 * tx_syms` → `rx = tx + jam = -tx`. This gives:
- `BER = 1.0` — negating a QPSK symbol flips both bits under Gray mapping, so every bit is wrong.
- `kurtosis(rx) = kurtosis(-tx) = kurtosis(tx) ≈ -2` — `-tx` has *exactly* the same distribution
  as `tx` (QPSK is symmetric under negation), so `rx` is statistically indistinguishable from a
  clean signal → 0% detection, not just "below threshold."
- `jam_power = |{-2·tx}|² = 4` (vs `GAMMA=0.02` → cost `0.08`).
- → `loss_ber → 0`, `loss_k = relu(-2-(-1)) = 0`, total `loss ≈ 0.08` — the global minimum.

Current runs (`loss≈1.8-1.9`) are far from this — there's a large basin-of-attraction gap
between the "small low-power blob" local optimum found so far and the "full-power 180°
rotation" global optimum, likely because both the kurtosis-relu term and `GAMMA*power` create
gradient pressure toward small `jam_power` early on, and a big coordinated jump to `power≈4`
is needed to escape.

**Caveat on the BER=1.0 optimum:** this relies on the jammer's loss treating BER=1.0 (perfect
bit-flip) as the target. From a strict information-theory standpoint, a *deterministic*
full-inversion (`rx=-tx`) is informationally equivalent to BER=0 for an adversary that knows
the pattern (just invert all received bits) — so "BER=1" here is a property of this specific
loss formulation, not necessarily a "win" against an adaptive receiver. Relevant to the
single-jammer-vs-detector skepticism below.

**Outputs:** `artifacts/sim03b/run00N.png`, `run00N_iq.png`, `run00N_model.pt`,
`artifacts/sim03b/tb/run00N/`. **Note:** these files are overwritten in place at every
`CHECKPOINT_EVERY=500` checkpoint (and at the end) — only the latest snapshot is ever on disk,
there is no per-checkpoint history.

---

## Simulation 03c — GMM policy + PPO (closed, negative result)

**Files:** `simulation03c/jammer_env.py`, `gmm_policy.py`, `train_ppo.py`, `submit.sh`, `README.md`

**What it was:** explored a per-symbol Gaussian Mixture Model (K=8 components) as a
PPO-compatible action head — a single-feedforward alternative to sim03's NSF, motivated as a
candidate building block for sim04's multi-agent PPO.

**Verdict (2026-06-13): closed, does not work.** Nine runs (full table in `artifacts/RUNS.md`)
systematically tried every PPO-mechanics fix — std clamp, target_kl, learning rate, entropy
coefficient, removing target_kl entirely. run004 (bias-initializing `log_std` so the jammer
starts near-silent) produced a large one-time jump (det 1.0→0.3, kurt pinned at the
`KURT_THRESH=-1.0` cliff edge), but runs005-008 showed this point is a **dead local optimum**:
run008 (`target_kl=None`) produced 10x more gradient updates with `approx_kl`~1.5 and
`clip_fraction`~0.87 — massive raw parameter movement — yet `entropy_loss` and all macro stats
(BER, power, kurtosis, detection) stayed bit-for-bit identical to run004.

**run009 (2026-06-16, 1M steps) — definitive confirmation:** longest run by 5–20×. BER ≈ 0.05
and *declining*, detection flat at ~80%, kurtosis pinned at exactly −1.0 (the cliff edge),
jammer power slowly drifting down. IQ scatter: symmetric Gaussian blob unchanged across early/
mid/late snapshots — all K=8 components collapsed to a single isotropic Gaussian. After 1M steps
the agent is slowly drifting toward jam=0. No path forward with GMM+PPO.

**Diagnosis: GMM permutation symmetry.** With K=8 components per symbol, gradient steps can
substantially relabel/reshuffle individual mixture components without changing the *marginal
distribution* that's actually sampled — the optimizer's movement budget gets absorbed by this
symmetry instead of reshaping the output. Combined with `MixtureSameFamily`'s non-reparameterized
(score-function) `log_prob`/`entropy` gradients being high-variance for overlapping components,
PPO+GMM cannot make directed progress here.

**Why NSF + direct-gradient (sim03b) did better** (best result across sim02/03/03b/03c: kurt~-1.30,
BER~0.17): a normalizing flow is a bijective transform (no permutation symmetry) and
direct-gradient training uses reparameterized sampling — low-variance pathwise gradients from
loss to distribution parameters. Neither property holds for GMM+PPO.

**Useful side-finding (architecture-independent): the reward "cliff".** From
`reward = ber - BETA*max(0, kurt-KURT_THRESH) - 0.05 - GAMMA*power`: `jam=0` gives `kurt=-2`
(QPSK's natural kurtosis) → `kurtosis_excess=0` → `reward=-0.05`. Default init (std≈1, power≈2)
starts with `kurt≈0`, i.e. **already past the `kurt>-1` penalty cliff** (reward≈-2.25 in
runs001-003) — worse than doing nothing. As power increases from 0, reward *increases* (kurt
stays ≤-1, ber rises) until the cliff at `kurt=-1`, beyond which it falls off sharply. The true
optimum sits at this cliff edge with higher BER/power than `jam=0`. Worth carrying into sim04's
reward design regardless of architecture.

**Next:** pivot to sim04 (MARL, 2 jammers) using sim03b's NSF + direct-gradient approach as the
per-agent policy basis.

---

## Simulation 04 — 2 cooperative jammers, direct-gradient NSF, kurtosis detector

**Files:** `simulation04/train.py`, `simulation04/submit.sh`

**What it is:** Extends sim03b from one jammer to two. Both agents share the same lossless QPSK
channel (`rx = tx + jam₁ + jam₂`) and are trained jointly from a single shared differentiable
loss — centralized direct-gradient, not yet MARL/PPO. Each agent has its own NSF encoder+flow;
the combined optimizer backpropagates through both simultaneously, so each agent's gradient
already accounts for the other's contribution to `rx`.

**Why two agents over one:**  the global optimum for a single jammer is `jam = -2·tx` (power=4).
With two agents the equivalent optimum is `jam₁ = jam₂ = -tx` (power=1 each) — the same `rx=-tx`
result at half the per-agent power, which is easier for the optimizer to find and avoids the
NaN instability region that plagued sim03b at power≈0.9.

**Architecture (per agent):** same as sim03b — MLP encoder `[OBS→64→CTX_DIM=64]` + NSF flow
(3 transforms, hidden=[64,64], `passes=2`, `randperm=True`).

**Key parameters:**
| Parameter | Value | Notes |
|---|---|---|
| N_JAMMERS | 2 | |
| N_SYMBOLS | 128 | |
| KURT_THRESH | −1.0 | |
| LAMBDA (kurtosis weight) | 2.0 | |
| GAMMA (per-agent power weight) | 0.02 | |
| BATCH_SIZE | 64 → **2048** | bumped in run005+ for GPU |
| LR | 3e-4 | |
| WEIGHT_DECAY | 1e-4 | |
| TOTAL_STEPS | 100,000 | |

**Loss:**
```
loss = soft_BER + LAMBDA · relu(kurt(rx) − KURT_THRESH) + GAMMA · (power₁ + power₂)
```

**run001 (job 99211, cancelled at 12k/20k steps by 3h wall time):**
- BER ≈ 0.35 and still rising — already 2× better than sim03b's best (0.17)
- Detection ≈ 5–10%, kurtosis ≈ −1.2
- Per-agent power ≈ 0.4–0.5 each (total ≈ 0.9)
- **Key finding:** both agents independently converged to a **4-cluster QPSK-like IQ structure**.
  The received signal `rx = tx + jam₁ + jam₂` is statistically indistinguishable from clean QPSK
  (kurtosis ≈ −2), evading the kurtosis detector — a more sophisticated emergent strategy than
  sim03b's unimodal blob.
- Time limit bug: at ~1 step/s, 20k steps requires ~5.5h; 3h limit killed the run early.

**run002 (job 99245, 5h limit, 17.5k steps):**
- Fixes from run001: all tensor→scalar logging conversions moved inside `torch.no_grad()`
  (eliminates `requires_grad=True` UserWarning); `--time` bumped to 5h.
- BER ≈ 0.28 at step 17k, detection ≈ 3%, kurt ≈ −1.25, power ≈ 0.8 per agent.

**Performance crisis (runs 001–004, batch=64, CPU):** training suffered a **5× slowdown**
over the course of a run — instantaneous sps dropped from 3.87 to 0.76 by step 10k and
plateaued there. Three root causes investigated:

1. *Autograd reference cycles from zuko `rsample()`.* Disabling GC + manual `gc.collect()`
   every 500 steps (runs 001–002) still let cycles accumulate between collections,
   causing RSS to grow from 894 MB → 2.2 GB and cache-thrashing the CPU.
2. *Re-enabling generational GC* with aggressive thresholds (`gc.set_threshold(100,5,5)`,
   run004) did not help — RSS still grew, sps still declined.
3. *Sionna per-step overhead.* `BinarySource()`, `Mapper()`, `Demapper()` called every step
   on CPU added Python-level overhead and likely contributed to RSS growth.

**Fix (run005+):** three changes eliminated the slowdown:

1. **Removed Sionna from the training loop.** Constellation points extracted from Sionna
   once at init; training uses pure-PyTorch `generate_qpsk()` (random index into 4
   constellation points) and `qpsk_demapper()` (APP demapper via `logsumexp`). Both are
   mathematically identical to Sionna's ops — just fewer Python calls and GPU-native.
2. **Moved to GPU** (`--gpus=1`, removed `CUDA_VISIBLE_DEVICES=""`). With `BATCH_SIZE=2048`,
   GPU parallelism dominates kernel-launch overhead.
3. **`torch.compile`** on `sample_jammer` — fuses the many small sequential NSF coupling-layer
   ops into fewer kernels.

Result: **19–20 sps (instantaneous, constant)** with RSS flat at ~2.5 GB. No degradation.
Throughput: 19 sps × 2048 batch = ~39k samples/s vs old peak 3.87 × 64 = 248 samples/s
(**157× throughput improvement**).

**run005 (job 100037, batch=2048, GPU): broken — LLR sign bug.**
The handcrafted `qpsk_demapper` used `log P(bit=0) − log P(bit=1)` instead of Sionna's
convention `log P(bit=1) − log P(bit=0)`. With the wrong sign, `wrong_labels = 1 − tx_bits`
rewarded *correct* decoding → optimizer drove `jam_power → 0`, `BER → 0`. Confirmed by
`loss → 0.14` (should be ~2.0 when jammer is active). **Fix:** swapped `mask0`/`mask1` in
the `logsumexp` terms. Also fixed `hard_decisions`: `(llr > 0)` to match the new convention
(was `(llr < 0)` from Sionna's opposite sign).

**run006 (job 100040, LLR fix applied):** BER metric still inverted (`llr < 0` not yet
fixed in this run). Shown BER went 1.0 → 0.67, i.e. **true BER 0.0 → 0.33** — already the
best result across all sims. 19–20 sps, RSS flat. Confirmed the training itself was correct;
only the logged BER was `1 − actual`.

**run007 (job TBD, both fixes applied):** first clean run with correct loss AND correct BER
logging. Running at 100k steps.

**Outputs:** `artifacts/sim04/run00N.png`, `run00N_iq.png`, `run00N_model.pt`

**How to run:**
```bash
cd "Tabula Rasa/simulation04"
sbatch submit.sh
```

---

## Simulation 04b — Sionna on GPU (validation run)

**Files:** `simulation04b/train.py`, `simulation04b/submit.sh`

**What it is:** Identical to sim04 in architecture and loss, but uses Sionna's
`BinarySource`, `Mapper`, and `Demapper` on GPU (via `sn.config.device = "cuda:0"`)
instead of the handcrafted pure-PyTorch replacements. This is a **validation experiment**
to confirm that Sionna on GPU is viable for sim06/07, where Sionna's channel models will
be needed and can't easily be replaced with hand-written PyTorch.

**Key difference:** `sn.config.device` is set before creating any Sionna modules, so all
Sionna ops run on GPU with automatic input casting. `torch.compile` is still used for the
NSF sampling. `TOTAL_STEPS=20,000` (enough to compare sps/RSS, not a full training run).

**What we're measuring:**
1. **sps** — sim04 gets 19–20 with pure PyTorch. How close can Sionna on GPU get?
2. **RSS** — sim04 is flat at 2.5 GB. Does Sionna leak memory on GPU?
3. **Correctness** — same BER/kurtosis trajectory confirms the demappers are equivalent.

**Outputs:** `artifacts/sim04b/run00N.png`, `run00N_iq.png`

**How to run:**
```bash
cd "Tabula Rasa/simulation04b"
sbatch submit.sh
```

---

## Planned roadmap (sim05–08)

**Ordering rationale (2026-06-19):** upgrade the **detector first**, then the **channel**.
The kurtosis detector is already solved (BER 0.65, <0.1% detection in sim04). A CNN detector
is the critical milestone: it breaks differentiability, forcing the switch from direct gradient
to MARL (PPO/SAC). This transition should happen on the simplest channel so we only debug one
thing at a time. Once MARL works against a learned detector, adding channel effects is a
controlled experiment.

**Channel model rationale:** plain AWGN (just adding noise) doesn't meaningfully change the
jamming dynamics — the jammer just needs slightly more power. The physically interesting
effect for the UAV scenario is **per-link path loss + phase rotation**: each jammer→target
pair has a different complex channel gain `h = (d_ref/d) · exp(jφ(d))`, so signal strength
and phase depend on distance. This fundamentally changes cooperation — jammers at different
positions must coordinate *who targets whom* and account for *how their signals arrive* at
each target. That's where multi-agent becomes essential.

**IQ-level waveform generation is the novel contribution.** Unlike PyJama (power allocation)
or standard RL jammers (channel selection / power levels), this project generates *raw IQ
waveforms* cooperatively. Even on a simplified channel, demonstrating that cooperative MARL
can produce deceptive waveforms that evade learned detection on OFDM would be a genuine
research contribution — no existing paper combines cooperative RL + IQ-level generation +
OFDM. Full 5G NR scale (1200+ subcarriers) is not needed: 802.11a uses 64 subcarriers with
4 pilots = 128 real dims, exactly matching the current NSF action space.

```
sim05   upgrade detector → CNN (frozen, pretrained on classical jammers)
        CNN is non-differentiable → direct-gradient breaks
        switch to MAPPO/MASAC + NSF policy
        novel contribution: cooperative MARL beats a learned detector

sim06   upgrade channel → per-link path loss + phase + noise
        rx = h_tx · tx + Σ_i h_jam_i · jam_i + noise
        h = (d_ref/d) · exp(jφ(d)) per jammer→target link
        different distances → different gains → spatial cooperation
        jammers observe: overheard tx (with own channel), own position, target positions
        Sionna channel models on GPU (validated in sim04b)

sim07   upgrade channel → 64-subcarrier OFDM + 4 pilots (802.11a-scale)
        adds frequency structure; pilots give detector a clean reference
        action space stays 128 real dims (64 complex subcarriers)
        jammer must learn to handle pilot positions without being told which they are
        detector: CNN on received resource grid (2D time-frequency image)

sim08   (stretch) add frequency-selective fading + UAV mobility
        channel changes per coherence interval → policy must generalize
        randomized UAV positions each episode → spatial coordination under uncertainty
```

**Future vision (beyond thesis scope):** multiple UAV jammers with randomized positions
coordinate IQ-level waveforms targeting multiple receivers. Each jammer's signal travels a
different distance to each target, arriving with different power and phase. The MARL policy
must learn spatial resource allocation (who jams whom) and waveform shaping (what signal to
send) jointly, while remaining undetectable to each target's local detector.

---

## Detector roadmap

| Detector | Used for | Source |
|---|---|---|
| Power threshold | sim00, sim01 (done) | scratch — 2 lines |
| Kurtosis test | sim02–04 training reward (done) | `scipy.stats.kurtosis` / PyTorch |
| GLRT | evaluation only | `scipy.stats` + ~20 lines custom |
| Pilot variance | sim07 evaluation | scratch ~10 lines |
| CNN on spectrogram | sim05+, training reward | `torchvision` resnet18 fine-tuned |
| CNN on resource grid | sim07+, training reward | 2D CNN on time-frequency grid |
| VAE anomaly detector | evaluation only | PyTorch (ref: arXiv:2410.01632) |
| PyJama detectors | citation / reference only | see note below |

---

## Research notes (open — pending supervisor discussion)

### Thesis endgame: SOTA detection vs SOTA jamming

The goal is a final comparison: *classical SOTA jammer* and *independent RL jammer* both lose to
the *novel cooperative MARL jammer*, all facing the same strong detector. The exact baselines
and detector are TBD. Notes from literature survey below.

### PyJama (arXiv:2407.15473, SPAWC 2024, ETH Zurich IIP)

PyJama is a differentiable jamming library on Sionna that uses SGD to optimise power allocation
over an OFDM resource grid. It's the closest published work to what this project does.

**Compatibility issue:** PyJama is built on Sionna 0.x + TensorFlow. This project uses
Sionna 2.x (PyTorch backend). Porting is non-trivial. **Use as citation and results reference,
not as a code dependency.** The pilot nulling strategy (Clancy 2011, 7.5 dB more efficient than
barrage) can be re-implemented cleanly from scratch in ~20 lines.

### On stealthy/undetectable jamming — a genuine research gap

Almost no published work studies a jammer that hides its *signal statistics* to defeat
a statistical detector. "Stealthy jamming" in the literature almost always means timing stealth
(sense-then-jam, only transmit when channel is active), not waveform-level stealth.

**Why the gap exists:** in practice a jammer is caught *physically* before signal statistics
matter — direction finding (AoA/TDOA), path loss anomaly (RSS), and channel reciprocity
violations all reveal a jammer regardless of IQ distribution.

**Why it's still valid scope for this thesis:** physical detection requires multi-antenna
infrastructure. In a simulation study, only baseband samples are available, so the relevant
threat model is the statistical detector. This is also the natural threat model for
cognitive radio / spectrum sharing scenarios where the jammer looks like another user.

**Thesis framing to discuss with supervisor:**
> "Can cooperative MARL agents learn to generate deceptive IQ-level waveforms that are both
> effective (high BER) and undetectable by learned detectors, on OFDM channels with pilots?"
> Physical detection (AoA/TDOA, path loss anomaly) is explicitly out of scope.
> Demonstrated on a 64-subcarrier system (802.11a-scale); architecturally compatible with
> larger systems via weight-sharing extensions.

**What makes this novel (no existing paper combines all three):**
1. **IQ-level waveform generation** — not power allocation (PyJama) or channel selection
   (standard RL jammers), but raw complex-valued signal synthesis
2. **Cooperative MARL** — multiple agents coordinate waveforms, enabling spatial strategies
   impossible for a single jammer (e.g. distributing power across agents to stay below
   per-link detection thresholds)
3. **Learned stealth** — evading a neural-network detector by shaping the jam signal's
   statistics, not just its timing or power level

The closest literature neighbours are adversarial-ML attacks on modulation classifiers —
the jammer crafts a signal that fools a neural-network detector. No existing paper does this
with cooperative RL jamming. PyJama (ETH Zurich) is closest in setup but uses SGD-based
power allocation, not RL and not IQ-level. Sagduyu et al. use GANs for IQ spoofing but not
cooperative MARL or OFDM.

### Jammer realism: omniscient vs causal/blind observation (deferred)

Currently the jammer observes `tx_syms` for the *same* timestep it jams — a **genie-aided /
omniscient jammer**. This is what makes `jam=-2*tx_syms` (the BER=1.0/kurtosis=-2 theoretical
optimum above) computable. Physically this requires the jammer's sense→process→transmit
latency to be shorter than one symbol period, which is generally unrealistic — a real reactive
jammer would at best act on `tx[t-1]`/`rx[t-1]` to produce `jam[t]` (one-symbol causal delay).

For i.i.d. symbols (no memory across symbols), `tx[t-1]` carries zero information about
`tx[t]`, so a causal jammer collapses to a **blind jammer**: `jam[t]` must be statistically
independent of `tx[t]`, and the `jam=-2*tx` trick becomes unreachable. The achievable-BER
ceiling under that constraint is a genuinely different (and likely much lower) number — the
classic jamming-vs-statistical-detector tradeoff.

**Decision (2026-06-11):** keep the omniscient observation for now. Priority is to get results
that beat SOTA with the current (simpler) formulation first; the causal/blind variant is
flagged as a future "abstraction" step (candidate for sim04+) rather than something to build
now. Don't let this complicate the current iteration.

### On GANs vs normalizing flows for waveform synthesis

GANs are the dominant approach for adversarial waveform synthesis in the literature:
- Sagduyu et al. (ACM WiSec 2019, IEEE TCCN 2021): GAN generates spoofing IQ signals
  over-the-air; generator produces synthetic IQ samples, discriminator distinguishes spoofed
  from legitimate signals.
- IEEE 2024: GAN-based radar jamming waveform generation from signal header snippets.
- Sagduyu et al. (arXiv 2018): GAN for data augmentation in jammer training.

**Decision (2026-06-18): NSF over GAN.** Three reasons:

1. **log_prob requirement.** sim05 switches to MAPPO (CNN detector is non-differentiable →
   direct-gradient breaks). PPO needs `log_prob(action|state)` — GANs fundamentally cannot
   provide this. NSF gives exact log_prob via change-of-variables. A GAN generator would be
   a dead end at the MARL transition.

2. **Low-dimensional action space favors flows.** Grover et al. (2020) found normalizing flows
   outperform GANs on low-dimensional density modeling. Our action space is 256 real dims
   (128 complex symbols) — firmly in the regime where flows excel.

3. **Mode collapse ≈ the GMM failure.** GAN mode collapse (generator converges to a narrow
   waveform subset) is structurally the same failure as sim03c's GMM component collapse.
   NSF's bijective transform is immune to this.

**Thesis framing:** the sim05 MAPPO setup is conceptually GAN-like (jammer policy = generator,
frozen CNN detector = discriminator, reward = `BER - β·D(rx)`). This connection is worth noting
in the related work section without actually using GAN training mechanics. Cite the Sagduyu
papers as the closest GAN-based prior work.

### On RL baselines

Most "RL jammer" papers in the literature are actually *anti-jamming* (a defender RL agent
avoids a fixed jammer). True offensive RL jammers that generate arbitrary waveforms are rare.
What exists operates mostly on discrete channel-selection or power-level actions, not IQ output.

The most natural RL baseline for a cooperative MARL thesis is therefore *internal*:
independent multi-agent PPO with no coordination (same architecture, no CTDE). This is the
standard MARL ablation and requires no external paper.

---

## Key design principles

- **No inductive bias:** never tell the jammer pilot positions, modulation scheme, or channel info.
  Let it discover strategies from raw observations.
- **Detection in reward, not hard clip:** power constraints come from the penalty term, not from
  clipping the action space.
- **Generative model upgrade path:** diagonal Gaussian (sim01) → normalizing flow (sim03/03b) → NSF carry-forward.
  GMM action head (sim03c) tried and abandoned — PPO+GMM permutation-symmetry degeneracy.
  GAN considered and rejected — no log_prob for PPO, mode collapse risk, flows outperform on
  low-dim distributions (see research note). NSF + direct-gradient (sim03b) is the carry-forward
  basis for sim04; NSF + MAPPO for sim05+.
- **Detector pretrained and frozen** during jammer training. Gradients never flow into detector.
- **Episode = 1 step for now.** No memory, no multi-step dependencies. Add episode structure with OFDM.

---

## Artifacts convention

All training outputs go to `artifacts/simXX/` (one folder per simulation, including `sim03b`, `sim04`, etc.),
**not** `simulationXX/runs/`. Each `train_*.py` script should:

- write `runNNN.png` (training curves) and `runNNN_iq.png` (IQ scatter, if applicable) to `artifacts/simXX/`
- save the trained model with `model.save(os.path.join(RUNS_DIR, f"run{run_id}_model"))` whenever the run
  is good enough to reuse (e.g. for cross-simulation evaluation/transfer)
- get a new row in `artifacts/RUNS.md` documenting: sim, run id, policy, key hyperparams, steps, result
  summary, whether a model was saved, and any notes

This keeps every run's plots, model, and hyperparameters discoverable in one place, and makes it possible
to load a model trained in one simulation and evaluate it in another.

---

## Sionna-specific notes (apply to all simulations)

- Sionna returns PyTorch tensors — use `.abs().pow(2).mean()`, not `np.mean(np.abs(...))`
- Call `.numpy()` before passing to numpy ops
- `Demapper` needs noise variance `no` as second arg — pass `1e-10` for lossless case (not 0),
  **but only when the LLR is used non-differentiably** (e.g. just for `hard_decisions`/BER
  bookkeeping, as in sim00/01/02/03). With `no=1e-10`, LLRs blow up to ±∞ for any nonzero
  rx-tx deviation.
- **If the LLR feeds a differentiable loss** (e.g. sim03b's soft-BER `binary_cross_entropy_with_logits`),
  use `no=1.0` (O(1)) and a tighter clamp (e.g. `(-10,10)`) — `no≈0` saturates the LLR/clamp and
  kills the gradient, which can trap an optimizer at a local optimum it can't escape.
- `Mapper` output shape is `(N, 1)` — always `.squeeze()` to `(N,)` before arithmetic
- `sn.utils.PlotBER.simulate()` is for EbNo sweeps only — not used in timestep loops

---

## GPU vs CPU on the ETH student cluster

GPU: RTX 5060 Ti, sm_120 (Blackwell), nightly cu130.

**sim00–03c (CPU):** GPU was slower than CPU — tiny networks ([64,64] MLPs), kernel-launch
overhead dominated, Sionna/scipy/gym env forced CPU round trips. `submit.sh` set
`CUDA_VISIBLE_DEVICES=""`.

**sim04+ (GPU):** GPU became viable after three changes:
1. **Large batch** (`BATCH_SIZE=2048`) amortises kernel-launch overhead.
2. **Sionna removed from training loop** — pure-PyTorch ops run natively on GPU with no
   CPU↔GPU transfers. (sim04b validates whether `sn.config.device="cuda:0"` can achieve
   comparable performance with Sionna in the loop.)
3. **`torch.compile`** fuses NSF's many small sequential coupling-layer ops into fewer kernels.

Result: **19–20 sps on GPU** (constant, no degradation) vs 3.87→0.76 sps on CPU.
Throughput: ~39k samples/s (GPU) vs ~248 samples/s peak (CPU) = **157× improvement**.

**Sionna on GPU (sim04b, pending):** Sionna 2.x modules inherit from `torch.nn.Module` and
support GPU via `sn.config.device = "cuda:0"` (set before module creation). They also have
explicit `torch.compile` compatibility (`torch.compiler.is_compiling()` guards in
`Block.__call__`). sim04b tests whether this eliminates the need for handcrafted replacements.
