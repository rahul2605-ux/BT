# Tabula Rasa — Jamming Simulations

Fresh restart of baseline experiments. Goal: build intuition step by step before reintroducing RL agents, fading channels, and multi-antenna setups. Each simulation adds exactly one layer of complexity.

**Hard rule:** only library code (Sionna, SB3, gymnasium, scipy). No reuse from the old project.

**Stack:** Python, Sionna 2.x (`sionna.phy`, PyTorch backend), stable-baselines3, gymnasium, numpy, matplotlib.

---

## Current status (2026-07-02)

**Active:** sim07 (job 101817, run005) — blind causal MAPPO jammer, black-box threat model,
top-K=1 subcarrier sparsity masking + temporally-stable held-frame action (frequency-hopping fix).

**System as implemented:** 1 TX → 1 RX (64-SC OFDM, QPSK, 802.11a-like) on a lossless
channel. 2 cooperative NSF jammer agents trained with MAPPO (CTDE). Frozen EfficientNet-B0
CNN detector (99.79% accuracy) provides P(jammed) as a black-box scalar score — no
gradients through the detector, ever. Jammer observes tx[t-1] (causal delay), making it
effectively blind for data symbols. Reward: `BER - β·log(1-P(jam)+ε) - γ·power`. Each
agent's output is now forced sparse (only the highest-magnitude subcarrier survives, the
other 63 are zeroed) before a hard per-agent power cap — see sim07 run history below for
why this was necessary.

**What's been established:**
- sim00–04: progressive complexity from lossless to 2-jammer NSF (direct-gradient found
  QPSK-like structure, BER=0.33+, kurtosis evasion)
- sim05: CNN detector needs OFDM structure (flat QPSK failed, 78.9% accuracy)
- sim06 Phase 1: CNN detector on OFDM = 99.79% accuracy ✓
- sim06 Phase 2: MAPPO failed on 128-dim omniscient setting (3 runs: broadband NSF output
  always detected P(jam)≈0.999, no reward gradient). Probe showed CNN fooled by 1-SC
  jamming at moderate power but catches ANY broadband noise even at power=0.01.
- sim06b: MAPPO failed on 2-dim omniscient setting too (stealth OK at P(jam)≈0.003, but
  BER=0.35 vs optimal 1.0 — Gaussian blobs, no input correlation). Confirms scalar reward
  cannot teach input-correlated waveforms at ANY dimensionality.
- sim06 audit: omniscient observation makes the problem trivially solvable (jam=-2*tx).
  De-trivialized by switching to causal observation in sim07.
- sim07 run001–003: causal observation alone does NOT fix the broadband-detection wall.
  Power capping and removing the entropy bonus only reduce P(jam) marginally (0.999→0.99);
  the real fix is forcing structural sparsity (top-K subcarrier masking). See sim07 section
  below for full diagnosis, including a direct spectrogram probe of the detector showing a
  hard cliff at exactly 4 simultaneously-active subcarriers.

**Key open question:** with sparsity now enforced (K=1 active subcarrier per agent, ≤2
active in the frame), can MAPPO learn a non-trivial blind waveform/SC-selection policy
that's both stealthy (P(jam)≈0, validated achievable by direct probe) and produces
meaningful BER? Or is BER from only 1–2 active subcarriers per frame too low to matter,
in which case the next pivot is to drop the "fool the CNN outright" framing in favor of
reporting jammer effectiveness/evasion across the full detector suite (power threshold,
kurtosis, GLRT, pilot variance, CNN) rather than a single binary pass/fail against the
hardest detector.

**Paper status:** related works section in progress in a separate session — see
`paper/README.md` for full reference triage, structure decisions, and open questions
(Hameed/Ziemann inclusion, PyJama dropped). System model and methodology sections are
stubs. No experimental results from lossless channels will appear in the paper — sim08
(realistic channel) is where paper claims begin.

**Cluster/compute notes (2026-06-30):** confirmed via `sacct` that sim07 jobs land on
`studgpu-node01`, a 5060ti node — already the fastest standard GPU on this cluster (other
options: 1080ti ×24 nodes, 2080ti ×4 nodes, gb10 [DGX Spark-style] ×6 nodes/1 GPU each).
No GPU-hour budget is enforced on the `projects` account (only a concurrency cap of 1 GPU
job at a time via `MaxJobsPU=1`); usage this month was 41 GPU-hours. The `projects_4gpus`
account exists but has `GrpTRESMins=gres/gpu=0` (provisioned, unfunded — confirmed via a
rejected dry-run submission) — likely what the supervisor's "100 exceptional GPU hours"
request targets. Note even if funded, `projects_4gpus`' `MaxJobsPU=1` means it enables one
job using up to 4 GPUs at once (multi-GPU data-parallel), NOT multiple parallel single-GPU
sweep jobs — for that, `MaxJobsPU` on the regular `projects` QOS would need to be raised.

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

## Simulation 05 — CNN spectrogram detector on flat QPSK (negative result)

**Files:** `simulation05/train_detector.py`, `simulation05/eval_jammer_vs_detector.py`

**What it was:** Attempted to train a CNN spectrogram detector (EfficientNet-B0, replicating
Li et al. IEEE Access 2022) on the flat QPSK channel from sim04. Binary classification:
clean vs jammed (4 classical jammer types collapsed into one label).

**Result: detector failed.** Best validation accuracy 78.9% (vs Li et al.'s 99.79%).
Massive overfitting: train accuracy hit 99.7% while val accuracy stalled at ~75%.

**Cross-evaluation (sim04 jammer vs CNN detector):**
| Jammer | Detection Rate | Verdict |
|---|---|---|
| Clean | 2.5% | FAR — low, good |
| Barrage | 2.5% | Undetected — same as clean |
| Single-tone | 100.0% | Detected (spectral spike) |
| Successive-pulse | 97.0% | Detected (periodic pattern) |
| Protocol-aware | 1.0% | Undetected |
| MARL (sim04) | 1.0% | Undetected |

**Root cause:** flat QPSK has no time-frequency structure. Spectrograms of "QPSK + Gaussian
noise" are indistinguishable from "QPSK at different SNR." The CNN only learned to detect
spectral lines (single-tone) and periodic impulses (successive-pulse). Broadband/noise-like
jammers are invisible in spectrogram space without OFDM structure to disrupt.

**Conclusion:** spectrograms require OFDM for the CNN detector to be meaningful. This
motivated merging the original sim05/06/07 roadmap into a single sim06 that combines OFDM
channel + CNN detector + MAPPO.

**Outputs:** `artifacts/sim05/detector/`, `artifacts/sim05/jammer/`

---

## Simulation 06 — OFDM + CNN detector + MAPPO jammers

**Files:** `simulation06/ofdm.py`, `simulation06/detector.py`, `simulation06/jammer.py`,
`simulation06/train_detector.py`, `simulation06/train_jammer.py`, `simulation06/eval.py`

**What it is:** The core thesis contribution. Merges the original sim05/06/07 plan into one
simulation with three simultaneous upgrades from sim04:

| Component | sim04 | sim06 |
|---|---|---|
| Channel | Lossless flat QPSK | 64-subcarrier OFDM (802.11a-like, Sionna) |
| Detector | Kurtosis (differentiable) | CNN EfficientNet-B0 (non-differentiable, frozen) |
| Training | Direct-gradient | MAPPO (Yu et al. NeurIPS 2022) |
| Reward | `soft_BER + λ·relu(kurt-thresh) + γ·power` | `BER - β·P(jammed) - γ·power` |

**OFDM parameters (802.11a-like, via Sionna `ResourceGrid`):**
| Parameter | Value |
|---|---|
| FFT size | 64 |
| Cyclic prefix | 16 samples |
| Effective subcarriers | 52 (48 data + 4 pilot) |
| Guard carriers | 6 left + 5 right + DC null |
| Pilot pattern | Kronecker (OFDM symbols 2, 11 all-pilot) |
| Subcarrier spacing | 312.5 kHz |
| OFDM symbols per frame | 14 |
| Samples per frame | 1120 |

**Phase 1 — Train CNN detector** (`train_detector.py`):
EfficientNet-B0 on OFDM spectrograms, binary (clean vs jammed). Trained on 4 classical
jammer types (barrage, single-tone, successive-pulse, protocol-aware). Li et al. hyperparams:
SGD lr=0.001, batch=32, 100 epochs. Frozen after training.

**Phase 2 — Train MAPPO jammers** (`train_jammer.py`):
2 cooperative NSF jammer agents trained with MAPPO against the frozen CNN detector.
CTDE pattern: centralized critic sees both agents' obs+actions, actors use local obs only.
NSF flow provides exact `log_prob` for PPO importance ratio; Monte Carlo entropy estimate
(8 rsamples). Jammers inject in the frequency domain (per subcarrier), before OFDM
modulation. Each OFDM frame (14 symbols) = one episode.

**Architecture:**
- Per agent: MLP encoder [128→64→64] + NSF flow (3 transforms, hidden=[64,64], passes=2)
- OBS_DIM = 128 (64 complex subcarriers → 128 real)
- ACTION_DIM = 128 (output: 64 complex jam values per subcarrier)
- Critic: MLP [512→256→128→1] (centralized, sees both agents)

**Reward:** `BER - β·P(jammed) - γ·total_jam_power`
- BER: hard-decision from Sionna demapper on data subcarriers
- P(jammed): `softmax(CNN_logits)[1]` from frozen detector on full-frame spectrogram
- Power: mean `|jam|²` across both agents

**Design principle:** use Sionna wherever possible. The OFDM chain (`ResourceGrid`,
`OFDMModulator`, `OFDMDemodulator`, `BinarySource`, `Mapper`, `Demapper`) runs on GPU
via `sn.config.device`. Only the CNN detector and NSF jammer are custom PyTorch.

**Phase 1 result (run002, 100 epochs, 312s):**
| Metric | sim05 (flat QPSK) | sim06 (OFDM) | Li et al. (real SDR) |
|---|---|---|---|
| Accuracy | 78.9% | **99.79%** | 99.79% |
| DR | 59.1% | **99.59%** | 99.79% |
| FAR | 2.07% | **0.00%** | 0.03% |
| F1 | 0.73 | **0.998** | 1.00 |

Matches Li et al.'s paper results exactly — confirms OFDM structure was the missing piece.
No overfitting: train and val both converge to 99%+ by epoch 5 and stay stable. Detector
checkpoint: `artifacts/sim06/detector/run002_best.pt`.

**Phase 2 — MAPPO jammer results (3 runs, all failed):**

All three runs failed to learn detection avoidance. P(jam)≈0.999 throughout.

**run001 (β=2.0 linear, γ=0.02, entropy=0.01):** power diverged 4→33, BER=0.43,
entropy→294 (maximum). Agent learned "more noise = more BER" with no detection gradient.
The entropy bonus actively pushed the policy toward high-variance random noise.

**run002 (β=0.3 log-shaped, γ=0.1, entropy=0.01):** log reward shaping
`-β·log(1 - P(jam) + ε)` amplified the penalty magnitude (det≈7.4 vs 1.0) but it was
still a **constant** across all batch elements — P(jam)≈0.999 for every action, so no
differentiation. Same power/entropy blowup as run001.

**run003 (β=0.3 log + warmup, γ=0.1, entropy=0.0):** entropy=0 and β warmup (ramp over
100 iters) prevented the power blowup (power stable at 4.4). But the policy completely
froze — BER=0.318, entropy=179.7, power=4.4 all unchanged for 290 iterations. Without
entropy bonus, PPO had no exploration mechanism. With constant P(jam)=0.999, no gradient
to follow.

**Diagnostic: why does the CNN always detect?** (`probe_1sc.py`)

Tested P(jam) for different jamming strategies at various power levels:

| Strategy | Scale | Power | P(jam) |
|---|---|---|---|
| 1SC: jam=-2*tx | 0.5 | 1.0 | **0.004** |
| 1SC: jam=-2*tx | 2.0 | 16.0 | **0.007** |
| 1SC: jam=-2*tx | 5.0 | 100.0 | 0.993 |
| 1SC: random noise | 2.0 | 3.9 | **0.003** |
| 1SC: random noise | 5.0 | 25.3 | 0.266 |
| ALL SC: random noise | 0.1 | **0.01** | **1.000** |
| ALL SC: random noise | 0.5 | 0.25 | 1.000 |
| ALL SC: random noise | 1.0 | 1.0 | 0.999 |

**Key finding:** the CNN is trivially fooled by single-subcarrier jamming at moderate
power (P(jam)≈0.003), but detects **all-subcarrier broadband noise instantly — even at
power=0.01 per SC**. The NSF flow initializes to ~N(0,1) on all 128 dims, producing
textbook broadband noise. Every action the agent ever tried was immediately detected with
P(jam)≈0.999. There was never any reward differentiation, making MAPPO structurally unable
to learn stealth.

**Root cause analysis:** MAPPO fails on this problem due to the combination of:
1. **128-dim continuous action space** where the optimal action is a structured function
   of the input (`jam[k] ∝ -tx[k]` per subcarrier)
2. **Holistic scalar reward** (frame-level BER) that doesn't decompose per subcarrier
3. **NSF initialization** producing broadband noise that's instantly detected
4. PPO's isotropic noise exploration cannot discover input-correlated strategies

MAPPO works well for the cooperative multi-agent jamming papers in the literature because
they use **low-dimensional or discrete** action spaces (channel selection, discrete power
levels, 2D position). IQ-level waveform synthesis is a different class of problem.

**Where MAPPO/SAC remain relevant:**
- Spatial coordination in sim07+ (which jammer attacks which target)
- Discrete decisions (subcarrier selection, resource allocation)
- Non-differentiable environments (real channels, sim-to-real transfer)
- The MAPPO negative result is itself publishable as an ablation

**Next steps (sim06b):** single-subcarrier MAPPO to validate that the algorithm works
when the action space is tractable (2D) and P(jam) varies with power level. Then either:
(a) scale up subcarrier count with curriculum, or (b) make the CNN pipeline differentiable
and use direct-gradient for the full 128-dim waveform (the entire path
`jam → OFDM → spectrogram → CNN` is differentiable except for one integer LUT lookup in
the viridis colormap, fixable with linear interpolation).

**Outputs:** `artifacts/sim06/detector/`, `artifacts/sim06/jammer/`

**How to run:**
```bash
# Phase 1: train detector
cd "Tabula Rasa/simulation06"
sbatch submit_detector.sh

# Phase 2: train MAPPO jammers (after detector is trained)
sbatch submit_jammer.sh
```

---

## Simulation 06b — Single-subcarrier MAPPO (diagnostic)

**Files:** `simulation06b/train_jammer_1sc.py`, `simulation06b/submit.sh`

**What it is:** Diagnostic experiment to confirm MAPPO can learn `jam = -2·tx` and find
the detection-avoidance sweet spot when the action space is tractable (2 real dims instead
of 128). Uses the same OFDM chain and frozen CNN detector as sim06.

**Motivation:** sim06's probe showed P(jam)≈0.003 for 1-SC jamming at moderate power,
proving the CNN can be fooled. This experiment tests whether MAPPO discovers the optimal
strategy when the exploration problem is tractable.

**Architecture:**
- Per agent: simple Gaussian MLP [2→64→64→2] (no NSF needed for 2D)
- OBS_DIM = 2 (I/Q of target subcarrier)
- ACTION_DIM = 2 (I/Q of jam signal on target subcarrier)
- Critic: CTDE MLP [8→128→64→1]
- TARGET_SC = 20 (FFT index, effective SC index 14)

**Reward:** `per_SC_BER - β·P(jammed) - γ·power`
- per_SC_BER: BER computed only on the target subcarrier's data symbols
- P(jammed): full-frame CNN detection (same detector as sim06)
- β=2.0 (linear — gradient exists in the 1-SC P(jam) range)
- γ=0.02

**What we're testing:**
1. Does MAPPO discover `jam = -2·tx` through 2D exploration? (BER side)
2. Does it find the power sweet spot where P(jam) transitions 0→1? (stealth side)
3. Does the IQ scatter show structured output (rotated QPSK) vs random blob?

**run001 result (390 iters, 261s, ~400 fps):**
Stealth solved: P(jam)≈0.003 throughout — completely undetected. But waveform learning
failed: per-SC BER plateaued at 0.35 (theoretical optimum = 1.0 for jam=-2*tx). IQ
scatter shows Gaussian blobs in both jammers, no input correlation — identical to sim06.
Power drifted 4.0→6.0 (entropy bonus pushing variance up).

**Conclusion:** even in 2D with a working stealth gradient (P(jam) varies meaningfully),
MAPPO converges to random noise rather than structured jam=-2*tx. The problem is NOT
dimensionality — it's that a scalar reward fundamentally cannot teach input-output
correlation. PPO would need to randomly sample an action near -2*tx, get a high reward,
and reinforce that specific input-dependent direction.

Combined with sim06's results, this gives a clean negative result: MAPPO fails for
IQ-level waveform synthesis regardless of action-space dimensionality, because the
scalar frame-level reward carries no per-dimension structural information.

**However:** this negative result applies to the OMNISCIENT setting where the optimal
strategy requires input-correlated output. In the BLIND setting (sim07), the jammer
learns a fixed waveform distribution, not a mapping obs→jam — a fundamentally different
and potentially more RL-tractable problem.

**Outputs:** `artifacts/sim06b/jammer/`

**How to run:**
```bash
cd "Tabula Rasa/simulation06b"
sbatch submit.sh
```

---

## Simulation 07 — Blind causal MAPPO jammer (black-box threat model)

**Files:** `simulation07/train_jammer.py`, `simulation07/submit.sh`

**What it is:** Changes exactly ONE axis from sim06 — the observation model — plus
locks in the black-box threat model. No channel, fading, noise, or SINR changes.
One axis at a time so that if results misbehave, the cause is unambiguous.

### Threat model: black-box, score-based

The jammer accesses the frozen CNN detector ONLY through a scalar detection score
`P(jammed)`. No gradients flow through the detector — ever. The `@torch.no_grad()`
wiring in `detect()` is load-bearing, not incidental. This forces MAPPO (RL), not
direct-gradient. Direct-gradient backprops through the detector and is white-box by
definition — it is not used, even as a training shortcut.

If convergence is hard, the correct responses are curriculum/reward-shaping, NOT
switching to gradient access. The black-box constraint holds during training as well
as evaluation.

### Observation model: causal delay

The jammer observes `tx[t-1]`, not `tx[t]`. At t=0, zeros. This is a 3-line change
in the rollout loop; OBS_DIM, agent architecture, buffer, and GAE all stay the same.

**Rationale:** sim06's optimum was the trivial `jam ≈ −2·tx` because the jammer saw
the exact current symbol. With i.i.d. QPSK, `tx[t-1]` is uninformative about `tx[t]`,
so the cancellation shortcut is mathematically unreachable. The jammer must learn a
blind waveform distribution — a genuinely non-trivial learning problem.

**What the jammer actually is:** a BLIND jammer learning a fixed stealthy waveform
distribution, not a reactive function of the current signal. The NSF is therefore
essentially unconditional (conditioned on an uninformative observation for data symbols).
It is learning the shape of a distribution to sample from.

**EXCEPTION — pilots:** OFDM symbols 2 and 11 carry deterministic pilot values. When
the jammer observes `tx[t-1]` and that happens to be a pilot (at t=3 or t=12), it can
recognize the known pattern. Any concentration of energy on pilot-adjacent symbols is
**protocol-aware jamming discovered through learning** — a key expected result, not an
artifact. Tracked via `pilot_power_ratio` metric and per-symbol power bar chart.

### Generative model role

The NSF learns a largely unconditional stealthy waveform distribution. The observation
is uninformative for data symbols, so the flow is NOT learning a mapping obs→jam — it is
learning the shape of a distribution to sample from. This is why a normalizing flow fits
the blind setting: it can represent complex, non-Gaussian waveform distributions with
exact `log_prob` for PPO's importance ratio. A GAN-discriminator-as-detector framing
does NOT apply here because the detector is black-box (no discriminator gradients).

### Key parameters

| Parameter | Value | Rationale |
|---|---|---|
| Observation | `tx[t-1]` (causal) | De-trivializes; blind for data, pilot-aware |
| TOTAL_FRAMES | 100,000 | Diagnostic first; extend via checkpoint-resume if learning |
| ENTROPY_COEFF | 0.005 | Moderate: some exploration without power blowup |
| BETA_DETECT | 0.3 (log-shaped) | Amplifies gradient near P(jam)≈1 |
| WARMUP_ITERS | 200 | β ramps from 0; learn power control first |
| GAMMA_POWER | 0.05 | Moderate power penalty |
| N_JAMMERS | 2 (fixed) | Permutation-invariant encoder worthless at N=2 |
| Detector gradients | None (black-box) | `@torch.no_grad()` in `detect()` |

**Checkpoint-and-resume:** saves full state (agents, critic, optimizers, logs,
iteration) so training can span multiple 8h SLURM jobs. Resume with
`--resume ../artifacts/sim07/jammer/run001_ckpt.pt`.

**Expected convergence:** uncertain. First run (100k frames, ~3h) is a "does it learn
at all" diagnostic, not a final result. Extend via checkpoint-resume if learning signal
appears. If P(jam) stays flat at 0.999 (same broadband-noise wall as sim06), the causal
delay alone hasn't helped and curriculum/reward-shaping is the next lever.

**Key convergence risk:** the NSF's initial output is still broadband noise → P(jam)≈0.999
→ no detection gradient. The causal delay changes the PROBLEM (blind vs omniscient) but
not the INITIALIZATION. sim06's probe showed the CNN catches even power=0.01 broadband
noise. If the jammer can't accidentally produce sparse/structured output early in training,
it will face the same constant-P(jam) wall. The β warmup (200 iters) is designed to let
the agent learn power control before detection kicks in — if this works, the agent should
settle at moderate power and then adapt to the detection signal.

### sim07 run history (2026-06-30)

**run001 (job 101622, baseline causal/blind, no mitigations, 150 iters before kill):**
Confirmed the predicted convergence risk exactly. `P(jam)` pinned at 0.999 for the entire
run — zero variance. `ENTROPY_COEFF=0.005` actively made things worse: policy entropy shot
to its ceiling (~294) within 10 iterations and total jam power diverged 4→33 (entropy bonus
rewards high-variance broadband output, which is exactly what the detector catches hardest).
Critic loss climbed monotonically (0→58) — the critic chasing a moving target as β warmup
ramped, never converging. Reward fell monotonically. No learning signal anywhere. Killed.

**run002 (job 101632, + hard per-agent power cap at 1.0, entropy coeff → 0):**
Mechanically fixed the power blowup (total power flat at 2.00 instead of 33) but did NOT
fix detection — `P(jam)` still pinned at 0.999, all metrics (BER=0.243, entropy=178.8,
power=2.00) bit-for-bit flat across 60 iterations. Root cause: capping power doesn't help
because the detector catches **broadband noise at any power level** (sim06 probe: even
power=0.01/SC broadband → P(jam)=1.0). The NSF's raw output is ~N(0,1) across all 128
dims regardless of overall scale — i.e. inherently broadband — so power scaling alone
can't put the agent into a regime where stealth is even possible. Killed.

**run003 (job 101650, + top-K=8 subcarrier masking on top of run002's fixes):**
First real movement: `P(jam)` dropped from pinned-0.999 to ~0.985–0.996 (noisy but
distinctly lower). Still **not learning** — entropy varied by <0.05% across 190 iterations
(178.75–178.83, pure sampling noise), BER/power similarly flat. Critic loss kept climbing
(0→40+). Diagnosis: K=8 active subcarriers is still too many for the detector to treat as
stealthy — confirmed by direct spectrogram probe (below), so there was no usable gradient
to learn from even with masking in place. Killed.

**Spectrogram probe (2026-06-30) — root-caused the sparsity threshold directly:**
Generated actual spectrograms through the trained CNN detector at controlled subcarrier
counts (`artifacts/sim07/diagnostics/spectrogram_probe_comparison.png` and `_v1.png`).
Result is a **sharp, power-independent cliff between 3 and 4 simultaneously active
subcarriers**:

| Active SCs | Power=1.0 | Power=8.0 |
|---|---|---|
| 1 | P(jam)=0.0002 | P(jam)=0.0001 |
| 2 | P(jam)=0.0002 | P(jam)=0.0017 |
| 3 | P(jam)=0.0013 | P(jam)=0.0119 |
| **4** | **P(jam)=0.9955** | **P(jam)=0.9955** |
| 8 | P(jam)=0.9973 | — |
| 64 (broadband) | P(jam)=0.9999 (even at power=0.01/SC) | — |

The CNN appears to key off the *number* of anomalous spectrogram rows present, not their
magnitude (P(jam) is nearly identical at power=1 and power=8 for a given SC count — only
the SC count moves the needle, until 4+ SCs are active, at which point it's saturated
regardless of power). Visually: a single bright frequency line (1 SC) is indistinguishable
from normal channel/pilot variation; 4+ simultaneous lines are not. Plausible explanation:
the detector's training set only had "1 active tone" (single-tone jammer) and "all tones"
(barrage) as sparse/dense extremes, so it never learned a graceful boundary in between —
it generalizes conservatively (flags as jammed) anywhere in the unseen middle ground.

**run004 (job 101657, top-K reduced 8→1, killed ~2026-07-02):** K=1 masking reduced P(jam)
to ~0.35–0.44 (as hoped — well below the 0.999 wall) but NOT near-zero as the static probe
predicted (expected P(jam)≈0.0002 for 1-SC jamming). Root cause identified via spectrogram
probe: **frequency-hopping**. The NSF resamples a fresh jam waveform at every one of 14 OFDM
symbols per frame. Each symbol potentially targets a *different* top-1 subcarrier. The
frame-level spectrogram aggregates all 14 activations → looks multi-tone/broadband → detected
at P(jam)≈0.46. Confirmed by a controlled probe: `hopping_2sc(2agents)` at power=1 gives
P(jam)=0.4633, matching the live run exactly, vs P(jam)≈0.0002 for a temporally-fixed single
SC. The discrepancy was not a learning failure — it was a structural mismatch between the
per-symbol rollout (resamples every symbol) and the frame-level spectrogram (aggregates all 14
symbols). Killed.

**run005 (job 101817, + temporally-stable held-frame action, 2026-07-02):** Fixes the
frequency-hopping problem by restructuring the rollout loop:
1. Sample jam waveform ONCE per agent per frame using `obs0 = zeros` (canonical blind context)
2. Hold this fixed across ALL data-symbol timesteps (t not in {3, 12})
3. Resample fresh ONLY at pilot-reactive timesteps t=3 and t=12 (where `tx[t-1]` is a known
   pilot → pilot-aware jamming discovered through learning; `PILOT_REACT_TIMESTEPS = {3, 12}`)

The held-frame action means the frame-level spectrogram sees a single stable frequency line
(or two, if both agents pick different SCs) across all 14 symbols — matching the static probe
condition where P(jam)≈0.0002. This is the first run where the agent should actually observe
P(jam) close to zero when it jams sparsely, providing a usable PPO gradient.

Raw jam_flat from the NSF (pre-masking) is still stored in the replay buffer for the PPO
importance ratio — the `apply_sparsity_and_power_cap()` transform is applied post-sample and
the log_prob from the NSF over raw outputs is used for PPO, preserving importance ratio
correctness.

**Inductive bias caveat (own concern, raised and discussed 2026-06-30):** top-K masking is
a real architectural prior — it presupposes "the solution is sparse" rather than letting
the agent discover this through gradient descent. Argued (and still believe) this is
justified as a *feasibility check*: an NSF initialized to ~N(0,1) across 128 dims has no
natural pathway to produce sparse samples (concentration-of-measure in high dimensions
means no batch element looks meaningfully different from any other), so the policy
gradient is provably flat in the unmasked regime — this isn't a "needs more steps" problem,
it's structurally the same wall as sim02/sim03c (Gaussian/GMM policies structurally
incapable of producing non-Gaussian output). If K=1 masking produces a working policy, a
natural follow-up ablation is removing the mask and confirming it fails unconstrained —
turning the inductive-bias compromise into a documented finding ("unconstrained continuous
RL cannot discover sparse evasive strategies from broadband initialization without a
structural prior") rather than a quietly-shipped shortcut.

**Strategic fallback (discussed, not yet decided):** if even K=1 doesn't produce meaningful
BER, or if full CNN evasion turns out to be unreachable by black-box RL regardless of
masking, the recommended pivot is away from "did we fully evade the strongest detector"
as a binary claim, toward reporting jammer effectiveness/evasion across the **full detector
roadmap** (power threshold, kurtosis, GLRT, pilot variance, CNN) — i.e. showing the
cooperative jammer defeats simple statistical detectors outright and meaningfully reduces
(without necessarily eliminating) CNN detection. This is more honest, lower-risk, and
consistent with the negative/boundary-result narrative already established by sim02/03c/06.

**Outputs:** `artifacts/sim07/jammer/` — training curves, IQ scatter, per-OFDM-symbol
power bar chart (pilot vs data symbols). `artifacts/sim07/diagnostics/` — spectrogram
probe comparison images.

**How to run:**
```bash
cd "Tabula Rasa/simulation07"
sbatch submit.sh

# To resume:
# Edit submit.sh to uncomment RESUME= line with checkpoint path
sbatch submit.sh
```

---

## Staging / roadmap

**Principle: one axis per step.** If results misbehave, the cause is unambiguous.

```
sim06   plumbing milestone — lossless + omniscient
        detector: 99.79% accuracy ✓
        MAPPO: failed (broadband noise always detected; scalar reward
          cannot teach input-correlated waveforms even in 2D — sim06b)
        probe: CNN fooled by 1-SC jamming at moderate power ✓
        NO scientific claims from sim06 — the lossless omniscient setup
          has a trivial analytical optimum (jam=-2*tx)

sim07   observation fixed (current step)
        causal tx[t-1] → blind jammer, de-trivializes the problem
        black-box threat model locked in (score-based, no detector gradients)
        same lossless channel — isolates observation change only

sim08   realistic channel + noise + SINR (deferred)
        rx = h_tx·tx + Σ h_jam·jam + noise
        per-link path loss, phase, fading via Sionna channel models
        spatial cooperation: who jams whom, accounting for channel gains
```

### Explicitly deferred (with reasons)

- **Realistic channel/SINR (sim08):** deliberately separated from the observation
  change so each axis is testable in isolation. Adding both at once would make
  failure diagnosis ambiguous.
- **Permutation-invariant encoder + N≥4 scaling:** only earns its keep at N≈6–8
  (see analysis). At N=2, fixed-order concatenation MLP is strictly simpler with
  no measurable downside. Separate reduced-setting experiment if pursued.
- **Co-adaptive/learning defender:** currently fixed-policy (frozen CNN) by design.
  An adaptive defender creates a non-stationary training environment that compounds
  the convergence difficulty. Deferred until the jammer reliably converges against
  the fixed detector.

### Known limitation (keep visible)

sim06's lossless channel and sim07's lossless channel are NOT realistic. No results
from a lossless channel make scientific claims in the paper. The channel is a
controlled simplification for isolating observation-model and training-algorithm
effects. Realistic channel (sim08) is where the paper's experimental claims begin.

---

## Detector roadmap

| Detector | Used for | Source |
|---|---|---|
| Power threshold | sim00, sim01 (done) | scratch — 2 lines |
| Kurtosis test | sim02–04 training reward (done) | `scipy.stats.kurtosis` / PyTorch |
| GLRT | evaluation only | `scipy.stats` + ~20 lines custom |
| Pilot variance | sim07 evaluation | scratch ~10 lines |
| CNN on spectrogram (flat QPSK) | sim05 (failed — needs OFDM) | `torchvision` EfficientNet-B0 |
| CNN on spectrogram (OFDM) | sim06 training reward (done: 99.79%) | EfficientNet-B0, Li et al. 2022 |
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

### On CNN-based jamming detection (sim05 detector justification)

CNNs on raw IQ samples / spectrograms are the established SOTA for jamming detection. The
key papers that motivate using a CNN detector in sim05:

**Foundational (DL on physical-layer signals):**
- O'Shea & Hoydis, "An Introduction to Deep Learning for the Physical Layer" (IEEE TCCN
  2017, ~2500 citations). Seminal paper on CNNs/autoencoders applied to raw IQ data.
  Justifies using learned features over expert-crafted ones for any signal-level task.
- O'Shea, Corgan, Clancy, "Convolutional Radio Modulation Recognition Networks" (EANN
  2016, ~1100 citations). First CNN directly on raw IQ for modulation classification.

**Jamming-specific:**
- Erpek, Sagduyu, Shi, "Deep Learning for Launching and Mitigating Wireless Jamming
  Attacks" (IEEE TCCN 2019, ~250 citations). CNN classifier detects jamming; frames it as
  adversarial ML. **Most directly relevant** — our sim05 is the jammer side of this arms race.
- Lichtman, Poston, Reed, "Jamming Signals Classification Using CNN" (IEEE SPAWC 2018).
  CNN classifies jammer types from 2D IQ histograms, 91% accuracy in NLOS.
- Li et al., "Jamming Detection in OFDM-Based UAVs via Spectrogram-Tailored ML" (IEEE
  Access 2022). CNN on spectrograms, 99.8% accuracy, 0.03% false alarm — UAV context
  matches our future scenario.
- TU Darmstadt, "Detecting 5G Signal Jammers Using Spectrograms" (IEEE 2024). Generalizes
  CNN detection to 5G; "watchdog" design with both supervised and unsupervised variants.

**Our novelty vs these papers:** they all build *detectors*. We build *jammers that learn to
evade* these detectors. The CNN detector is the adversary our MARL agents train against — a
frozen, pretrained "opponent" that represents the best known detection approach. No existing
paper trains a cooperative jammer against a learned CNN detector.

### On MAPPO / MASAC (sim05 RL algorithm justification)

**Foundational RL:**
- Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017). PPO foundational
  paper. Clipped surrogate objective, on-policy, stable training. MAPPO builds on this.
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic
  Actor" (ICML 2018). SAC foundational — entropy-regularized objective prevents premature
  convergence in continuous action spaces. Off-policy = sample efficient.
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (arXiv 2018). SAC v2
  with automatic entropy temperature tuning — what modern implementations use.

**Multi-agent:**
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS
  2022). **MAPPO foundational paper.** Shows that simple PPO with parameter sharing +
  proper normalization + centralized value function matches or beats QMIX, MAVEN, MADDPG
  across cooperative benchmarks. Directly justifies MAPPO as first-line choice.
- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
  (NeurIPS 2017, MADDPG). Introduced the **CTDE paradigm**: centralized critic sees all
  agents' observations during training, actors execute with local observations only.
- Schroeder de Witt et al., "Is Independent Learning All You Need in the StarCraft
  Multi-Agent Challenge?" (arXiv 2020). Demonstrates that independent learners with proper
  tuning rival complex CTDE methods — supports MAPPO-style simplicity.

**MASAC note:** there is no canonical "MASAC" paper. Multi-agent SAC is implemented by
applying MADDPG's CTDE pattern (centralized critic) with SAC as the base algorithm. Cite
SAC + MADDPG and describe the combination.

**Decision (2026-06-23): start with MAPPO, then compare MASAC.**
- MAPPO is simpler (on-policy, no replay buffer), well-validated for cooperative tasks
  (Yu et al.), and directly compatible with NSF's `log_prob`.
- MASAC is more sample-efficient (off-policy, replay buffer) — important when each step is
  expensive. Test as a second algorithm once MAPPO baseline works.
- Both use CTDE: centralized critic sees both agents' observations + actions during training;
  each actor only sees its own observation at execution time.

### On NSF as RL policy distribution (novelty justification)

Using a normalizing flow instead of the standard diagonal Gaussian as a PPO/SAC policy is
a key component of our approach. The literature basis:

- Durkan, Bekasov, Murray, Papamakarios, "Neural Spline Flows" (NeurIPS 2019). The NSF
  architecture we use — rational-quadratic spline coupling transforms for density estimation.
- Ward, Smofsky, Bhatt, "Normalizing Flows for Reinforcement Learning" (ICML Workshop 2019).
  **Directly proposes flow-based policies in PPO.** Shows flow policies capture multimodal
  action distributions and improve performance on continuous control benchmarks.
- Mazoure et al., "Soft Actor-Critic with Normalizing Flows Policies" (2020). Integrates
  flows into SAC's max-entropy framework — relevant if we use MASAC.

**Our novelty:** Ward et al. showed flow policies help in standard single-agent RL on
MuJoCo benchmarks. **Nobody has used them for cooperative MARL, and nobody has applied them
to wireless jamming.** The combination of NSF policy + MAPPO/MASAC + cooperative waveform
generation is novel. The flow is essential because a diagonal Gaussian policy is structurally
incapable of producing non-Gaussian signal statistics (proven in sim02/sim03c) — the jammer
must shape its output distribution to evade statistical detection, which requires an
expressive generative model.

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
- **Use Sionna wherever possible.** OFDM chain, source, mapper, demapper all via Sionna on GPU
  (`sn.config.device`). Only hand-write what Sionna doesn't cover (CNN detector, NSF jammer).
  sim04b validated Sionna on GPU is viable and performant.
- **Episode = 1 OFDM frame (14 symbols) in sim06+.** Frame-level reward from CNN detector
  broadcast to all timesteps. Per-symbol credit assignment deferred to future work.

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
