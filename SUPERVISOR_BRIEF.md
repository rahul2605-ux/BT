# Bachelor Thesis — Progress Brief
**Student:** Rahul Rahman  
**Supervisor:** Antonio Di Maio  
**Date:** 2026-04-30  
**Topic:** Adversarial Cooperative PHY Jamming with MARL

---

## 1. What Has Been Built

A complete simulation and training framework for cooperative OFDM jamming.

### Software Stack

| Component | What it does | Status |
|---|---|---|
| `WirelessEnv` | Gymnasium env — N jammers vs M defenders over OFDM | ✓ complete |
| `MockChannel` | Fast NumPy channel (log-distance PL, random phase) | ✓ complete |
| `SionnaChannel` | CDL-D fading via Sionna 2.0 / PyTorch | ✓ complete |
| `TimedBarrageJamming` | Stage 1a fixed baseline — full-power all subcarriers | ✓ complete |
| `train_stage1.py` | Single-jammer SB3 PPO with auto-generated result plots | ✓ complete |
| `CTDEMlpPolicy` | Parameter-sharing actor + centralized critic (SB3-compatible) | ✓ complete |
| `train_ctde.py` | N-jammer CTDE team training | ✓ complete |
| Unit tests | 37 tests, all passing | ✓ complete |

### Architecture Summary

```
N jammers                        M defenders (passive)
Each jammer:                     Fixed TX power
  obs_i = [SINR(76) | pos_i(3) | step_frac]
  act_i = [power_sub0, ..., power_sub75]  ← per-subcarrier dBm

channel.get_coefficients(positions)
  → H_tx  (M, N_leg, 76) complex   ← path loss × CDL-D fading
  → H_jam (M, N_jam, 76) complex

SINR[rx, sub] = |H_tx · √P_tx|² / (Σ_jam |H_jam| · P_jam + noise)

reward = −mean(SINR)   ← jammer minimises defender link quality
```

### CTDE Policy (Stage 1 baseline)

```
Global obs (N_jam × 79 + 1)
         │
  ┌──────┴──────────────────────────┐
  │  Actor (shared MLP, Stage 1)    │   → per-jammer action (N_sub,) each
  │  local obs_i (80,) → MLP → feat │     same weights applied to each jammer
  │                                 │
  │  Critic (centralized MLP)       │   → V(global state)
  │  full global obs → MLP          │
  └─────────────────────────────────┘
```

---

## 2. Preliminary Results (MockChannel, 100k steps, single jammer)

*These are early training curves — not converged results. Purpose is to confirm the pipeline works end-to-end.*

### Barrage Baseline (fixed strategy, 1 jammer)

Multi-jammer sweep showing how interference scales with jammer count:

| N jammers | Mean SINR after barrage (dB) |
|---|---|
| 1 | 44.4 |
| 2 | 43.2 |
| 3 | 40.2 |
| 4 | 39.6 |
| 5 | 39.4 |

Note the diminishing returns — barrage spreads power uniformly across all subcarriers, so adding more jammers has less marginal impact after a point. A selective jammer could do better with the same total power budget.

Plots in `plots/multi_agent/`:
- `exp1_sweep_jammers.png` — SINR vs jammer count
- `exp3_heatmap.png` — per-subcarrier SINR over an episode

### PPO Training (single jammer, 100k steps)
Plots in `plots/stage1/` (auto-generated):
- `training_curves.png` — episode reward and mean SINR over training
- `policy_comparison.png` — trained PPO vs barrage vs random (box plots + learned power allocation)

---

## 3. Channel Model — Key Decision Needed

### Current implementation
Two-component hybrid:
1. **Large-scale path loss:** log-distance `PL(d) = 40 + 10·3.5·log10(d)` dB
2. **Small-scale fading:** CDL-D from Sionna (LOS-dominant, 30 ns delay spread)

The combination gives frequency-selective SINR across the 76 OFDM subcarriers — different subcarriers see different channel gains, which is important for the thesis argument (selective jammers outperform barrage).

### Known limitation
The path loss and CDL model are not fully consistent — CDL-D assumes specific geometry (angle-of-arrival, angle-of-departure) that is not coupled to the agent positions in the current implementation. Positions affect only path loss; CDL fading is generated independently of geometry.

**Q for ADM:** Is this an acceptable simplification for Stage 1 baselines, or should we switch to `sionna.phy.channel.tr38901.UMa` which handles both path loss and geometry-aware fading together?

---

## 4. Open Design Questions

### 4.1 OFDM Parameters
Currently using:
```
n_subcarriers = 76       ← FFT size
subcarrier_spacing = 15 kHz   ← LTE/NR default
carrier_frequency = 3.5 GHz
```
**Q:** Are these the right parameters for the UAV scenario? Should we use a different numerology?

### 4.2 Jammer Observation
Each jammer currently observes the **defender's SINR** — which is physically unrealistic (jammers don't know the defender's receive quality directly).

More realistic alternatives:
- Sensed received power spectrum at the jammer's antenna
- Estimated channel quality from a pilot-like preamble sniffing

**Q:** What should the jammer observe? Defender SINR (simplification) is fine for Stage 1, but matters for Stage 3 when we claim realism.

### 4.3 Generative Model Architecture (Stage 3)
The thesis needs the jammer to output a **structured waveform** rather than a power-allocation vector. Current placeholder: `waveform_type = "power"` (direct per-subcarrier power). The latent representation (`latent_dim = 16`) is scaffolded but not yet connected to a generative model.

**Q:** What architecture for the generative model?
- **Direct latent parameterization** — `z ∈ R^16 → Linear → complex OFDM symbol` — simplest, most stable to train
- **VAE** — adds latent regularization, may help generalization
- **GAN** — adversarial training, more expressive but harder to train

Our preference is direct latent (simpler, fewer hyperparameters) unless there's a theoretical reason to prefer another.

### 4.4 TX/RX Positions
Currently, legitimate TX and RX nodes share the same position array — they are the same agent. The system model has:
```
legitimate.count = 1   # one TX/RX node
```
**Q:** Should TX and RX be separate? In the UAV scenario, are both roles at the same platform, or is one a ground station and one a UAV?

### 4.5 Mobility
`mobile = True` is available in `AgentSpec` but not yet implemented in `step()`. Agents are static within each episode, resampled at reset.

**Q:** Should mobility be added before Stage 2 experiments, or is static positioning sufficient to demonstrate the permutation-sensitivity weakness?

---

## 5. Planned Stage 2 Experiments

Once we have converged Stage 1 baselines, Stage 2 runs two experiments:

**Experiment A — Permutation sensitivity:**
1. Train CTDE with N_jam=3, fixed jammer ordering in obs
2. At test time, permute jammer indices (jammer 0 → slot 1, jammer 1 → slot 0, etc.)
3. Measure SINR degradation vs unshuffled ordering
4. Expected result: MLP performance degrades for identical physical scenario

**Experiment B — Team-size scaling:**
1. Train CTDE independently for N_jam = 1, 2, 3, 4, 8
2. Plot mean SINR vs N_jam (post-training)
3. Expected result: MLP CTDE improvement saturates or degrades at larger N while barrage continues improving

These two results together justify replacing the MLP with a permutation-invariant encoder.

---

## 6. Next Steps (proposed)

| Priority | Task |
|---|---|
| Immediate | Run converged Stage 1 baselines on Colab (500k–1M steps, SionnaChannel) |
| Immediate | Confirm OFDM parameters + TX/RX model with ADM |
| After confirmation | Stage 2 permutation + scaling experiments |
| Stage 3 | Implement Set Transformer team encoder |
| Stage 3 | Connect to generative waveform model (architecture TBD) |
