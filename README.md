# BT — Adversarial Cooperative PHY Jamming with MARL

**Program:** BSc Informatik, FS26 (Spring 2026)
**Author:** Rahul Rahman (`rrahman` / `rulecito`)
**Supervisor:** Antonio Di Maio (ADM)
**Last updated:** 2026-04-30

---

## Thesis Goal

Train a **cooperative multi-agent jammer team** that learns to jam UAV OFDM communications using **learned generative waveforms**, simulated in NVIDIA Sionna.

Research question: does a **permutation-invariant team encoder** (Set Transformer or GNN) produce better cooperative jamming than a plain MLP-based CTDE policy — and why?

---

## System Model

```
Jammers (Nj agents)                  Defenders / UAVs (Nl agents)
─────────────────────                ────────────────────────────
Mobile cooperative team              OFDM communicators
Each jammer:                         Adapt position + freq + power
  latent vector (dim=16)             (NOT in training scope yet)
  → generative model
  → structured jam waveform

         ↕  CDL-D fading channel (NVIDIA Sionna 2.x)
             complex channel coefficients H ∈ C^(N_rx × N_sub)
             positions update each step → new realization each step
```

**Primary metric:** SINR per subcarrier, spectral efficiency
**Primary focus:** Jammer (attacker) side — defender adaptation is out of scope

---

## Research Methodology (3 Stages)

### Stage 1 — Baselines ✓
1. **Barrage jammer** — uniform noise across all subcarriers (`TimedBarrageJamming`)
2. **PPO single-jammer** — MLP policy, SB3 PPO (`train_stage1.py`)
3. **PPO CTDE team** — parameter-sharing actor + centralized critic (`train_ctde.py`)

### Stage 2 — Weakness Analysis ← next
Demonstrate MLP CTDE limitations:
- Permutation sensitivity: shuffle jammer ordering → policy performance drops
- No structural inductive bias for team coordination

### Stage 3 — Novel Method
Replace MLP team encoder with a **permutation-invariant encoder** (Set Transformer or GNN).

**Key open question:** Generative model architecture not yet fixed — GAN, VAE, or direct latent. Confirm with ADM.

### CTDE
**Centralized Training, Decentralized Execution.** Implementation choice, not a contribution.

---

## Development Workflow

| Where | Channel | GPU | When |
|---|---|---|---|
| MacBook (local) | `MockChannel` | None — NumPy only | Editing, unit tests |
| Google Colab | `SionnaChannel` CDL-D | Free T4 | All training (Stage 1+) |
| ETH Euler HPC | `SionnaChannel` CDL-D | A100 | Large-scale / long runs |

**Running locally:**
```bash
cd BT/
PYTHONPATH=. conda run -n sionna-thesis python <script>
PYTHONPATH=. conda run -n sionna-thesis python -m pytest tests/ -v
```

**Running in Colab:**
```python
!git clone <repo_url>
import sys; sys.path.insert(0, '/content/BT')

# Single-jammer PPO (Stage 1b)
!python simulations/train_stage1.py --total-steps 500000 --n-envs 4 --channel-mode sionna

# CTDE team (Stage 1c) — 3 jammers
!python simulations/train_ctde.py --n-jammers 3 --total-steps 1000000 --channel-mode sionna --n-envs 4

# Multi-agent simulation (fixed strategies)
!python simulations/multi_agent_sim.py --channel sionna --out /content/plots
```

---

## Repository Layout

```
BT/
├── core/
│   ├── config.py              # All dataclass configs + enums  (READ SECOND)
│   ├── strategy.py            # BaseStrategy ABC + TimedBarrageJamming, RandomStrategy
│   └── scatter.py             # Agent placement: Random, Symmetric, Custom + factory
│
├── channel/
│   ├── base.py                # BaseChannel ABC + MockChannel — complex H interface
│   └── sionna_channel.py      # SionnaChannel — CDL-D via Sionna 2.x / PyTorch
│
├── envs/
│   └── wireless_env.py        # gymnasium.Env — N jammers, per-jammer actions  (READ THIRD)
│
├── policies/
│   └── ctde_mlp.py            # CTDEMlpPolicy — parameter-sharing actor + centralized critic
│
├── wrappers/
│   ├── sinr_normalize.py      # ObservationWrapper: SINR dB → [0, 1]
│   ├── rescale_action.py      # ActionWrapper: policy [-1,1] → [0, max_power_dbm]
│   └── fixed_opponent.py      # Future: freeze one team, single-agent interface
│
├── simulations/
│   ├── simple_barrage.py      # Stage 1a: one episode, barrage, SINR plots
│   ├── train_stage1.py        # Stage 1b: single-jammer SB3 PPO  (READ FIRST)
│   ├── train_ctde.py          # Stage 1c: CTDE team PPO, N jammers
│   ├── plot_stage1.py         # Standalone plotter (called automatically after training)
│   └── multi_agent_sim.py     # Fixed-strategy multi-agent simulation
│
├── tests/
│   ├── test_scatter.py        # 11 tests
│   ├── test_channel.py        # 5 tests
│   ├── test_env.py            # 9 tests
│   └── test_wrappers.py       # 12 tests  →  37 total, all passing
│
├── models/                    # Created at runtime
│   ├── stage1/                # single-jammer PPO
│   └── ctde/                  # CTDE team PPO
│
├── logs/                      # metrics.npz per run
├── plots/                     # auto-generated after training
└── CONTEXT.md
```

---

## How the Files Connect

```
train_ctde.py  (or train_stage1.py)  ←── START HERE
│
├── core/config.py             every config knob — EnvironmentConfig, AgentSpec, OFDMConfig
│
├── envs/wireless_env.py       the simulation loop
│   │  reset() → scatter agents → initial obs
│   │  step(action) → get_coefficients() → sinr_from_power() → reward → obs
│   │
│   ├── core/scatter.py        place(count, bounds) → positions (N,3)
│   │
│   ├── channel/base.py        get_coefficients() → H_tx, H_jam complex
│   │   MockChannel            random-phase path loss  [NumPy, no GPU]
│   │
│   └── channel/sionna_channel.py
│       SionnaChannel          CDL-D path loss × complex CDL H  [PyTorch, Colab/Euler]
│
├── wrappers/                  obs dB→[0,1]   action [-1,1]→[0,30dBm]
│
├── policies/ctde_mlp.py       ParameterSharingExtractor + CTDEMlpPolicy
│   actor:  shared MLP applied to each jammer's local obs independently
│   critic: separate MLP over full global state (centralized)
│
└── stable_baselines3.PPO      trains the team, reward = −mean(SINR)
```

**Data through one step:**
```
action = (N_jam × N_sub,) flat  →  reshape(N_jam, N_sub)  — per-jammer powers

get_coefficients(positions) → H_tx (N_rx, N_tx, N_sub) complex
                             → H_jam (N_rx, N_jam, N_sub) complex

sinr_from_power(H_tx, H_jam, tx_pwr, jam_pwr) → SINR (N_rx, N_sub) dB
reward = −mean(SINR)
obs = [SINR(76) × N_jam | pos_i(3) × N_jam | step_frac(1)]
```

---

## Module Reference

### `channel/base.py` — Channel Interface

**Abstract method:**
```python
get_coefficients(tx_pos, rx_pos, jam_pos, n_sub)
    → H_tx  (N_rx, N_tx,  N_sub) complex64
    → H_jam (N_rx, N_jam, N_sub) complex64
```

**Static SINR helpers (call with pre-computed H):**
```python
BaseChannel.sinr_from_power(H_tx, H_jam, tx_pwr_dbm, jam_pwr_dbm, noise_dbm)
    → SINR (N_rx, N_sub) dB    # incoherent — Stage 1 power actions

BaseChannel.sinr_from_signals(H_tx, H_jam, tx_pwr_dbm, jam_signals, noise_dbm)
    → SINR (N_rx, N_sub) dB    # coherent   — Stage 3 complex actions
    # interference = |Σ_jam H_jam · j_jam|²  (N² gain vs incoherent N)
```

**Why static?** The env calls `get_coefficients()` once per step, stores H, builds obs from H, then calls `sinr_from_signals(H, ...)` with the policy action — same H for both. If H were regenerated inside compute_sinr, the jammer would coordinate against a different channel than it observed.

**Convenience wrappers** (call `get_coefficients()` internally):
```python
channel.compute_sinr(...)           # incoherent, backward-compat
channel.compute_sinr_complex(...)   # coherent, for standalone use
```

**MockChannel:** random phase per (link, subcarrier). Cancels in `sinr_from_power` (|e^{jφ}|²=1). Phases are redrawn every call — cannot support learned phase coordination. Use SionnaChannel for coherent jamming experiments.

**SionnaChannel** (`sionna.phy.channel.tr38901.CDL`, PyTorch, CDL-D model):
- Phase is now preserved — `_cdl_coefficients()` returns complex H instead of |H|² dB
- `device=None` always — Sionna has its own device registry, never pass `"cuda"`

---

### `envs/wireless_env.py` — WirelessEnv

**Observation** `(N_jam × (N_sub + 3) + 1,)`:
```
[SINR(N_sub)] × N_jam    shared defender SINR repeated per jammer
[norm_pos(3)] × N_jam    each jammer's own normalised position
[step_frac(1)]
Reshape to (N_jam, N_sub+3) for Set Transformer token input.
N_jam=1 → (80,) — backward compatible.
```

**Action** `(N_jam × N_sub,)`: per-jammer per-subcarrier power. Reshape to `(N_jam, N_sub)` in step().

**Per step:**
1. `get_coefficients()` → H_tx, H_jam stored as `self._H_tx`, `self._H_jam`
2. `sinr_from_power(H, ...)` → SINR, reward
3. H available for future `sinr_from_signals()` call (Phase 1.4 complex actions)

**Outstanding TODOs:**
1. TX/RX positions are same array — needs separate arrays (confirm with ADM)
2. `mobile=True` scaffold exists but is not wired into `step()`

---

### `policies/ctde_mlp.py` — CTDEMlpPolicy

```python
from policies.ctde_mlp import CTDEMlpPolicy

model = PPO(
    policy=CTDEMlpPolicy,
    env=env,
    policy_kwargs=dict(n_jammers=N_jam, n_subcarriers=76, hidden_size=256),
)
```

**`ParameterSharingExtractor`:**
- Actor: same 2-layer MLP applied to each jammer's local obs `(80,)` independently
  → `(B × N_jam, 80) → (B × N_jam, hidden) → reshape → (B, N_jam × hidden)`
- Critic: separate 2-layer MLP over full global obs `(N_jam × 79 + 1,)`
  → `(B, N_jam×79+1) → (B, hidden)`

SB3's `action_net = Linear(N_jam × hidden, N_jam × N_sub)` maps concatenated features to joint action.

**The Stage 2 weakness:**  
`action_net` maps the CONCATENATION `(hidden_0 | hidden_1 | ... | hidden_{N-1})` linearly to actions. Swapping jammer indices in the obs shuffles which feature block feeds which output slice — performance degrades for an identical physical scenario. Set Transformer (Stage 3) aggregates features permutation-invariantly before the output projection.

---

### `simulations/train_ctde.py` — CTDE Training

```bash
# 3 jammers, MockChannel
PYTHONPATH=. conda run -n sionna-thesis python simulations/train_ctde.py --n-jammers 3

# Colab, SionnaChannel
python simulations/train_ctde.py --n-jammers 3 --total-steps 1000000 --channel-mode sionna --n-envs 4
```

Saves to `models/ctde/`, `logs/ctde/`. Auto-plots to `plots/ctde_n{N}/`.

---

### `tests/` — 37 Tests, All Passing

```bash
PYTHONPATH=. conda run -n sionna-thesis python -m pytest tests/ -v
```

---

## Current Status & Roadmap

### Done ✓
- [x] Phase 0 — env, wrappers, tests (37 passing)
- [x] Phase 1.1 — `train_stage1.py` single-jammer PPO + auto-plots
- [x] Phase 1.1b — `channel/sionna_channel.py` CDL-D, Colab-ready
- [x] Phase 1.1c — `multi_agent_sim.py` fixed-strategy multi-agent simulation
- [x] **Phase 1.2 — Channel model redesign: complex H, `sinr_from_power`, `sinr_from_signals`**
- [x] **Phase 1.3 — Per-jammer actions: `(N_jam × N_sub,)` action space, backward-compat**
- [x] **Phase 1.4 — CTDE policy: `CTDEMlpPolicy` + `train_ctde.py`**

### Next
```
Phase 2 — Weakness Analysis
  2.1  Permutation experiment: train CTDE, shuffle jammer ordering at test time
       → measure SINR degradation vs unshuffled
  2.2  Team-size ablation: train with N_jam=1,2,3,4,8
       → show MLP CTDE performance degrades as N grows
  2.3  Document findings — motivation for Stage 3

Phase 3 — Novel Method
  3.1  Confirm generative model architecture with ADM (direct latent is simplest)
  3.2  Implement Set Transformer or GNN team encoder in policies/
  3.3  Wire: local obs → encoder → z_team → generative model → complex waveform
  3.4  Train with complex actions (sinr_from_signals already in place)
  3.5  Compare against Stage 1 baselines

Phase 4 — Write-up
```

---

## Known Bugs

| File | Line | Issue |
|---|---|---|
| [core/config.py](core/config.py#L77) | 77 | `EnvironmentConfig.__post_init__` at module scope — validation never runs |
| [envs/wireless_env.py](envs/wireless_env.py) | ~step | TX/RX positions same array — needs separate arrays, confirm with ADM |

---

## Supervisor Guidance

- CTDE is not a contribution — it is an implementation choice
- MLP failure is a *limitation* that structured encoders improve, not catastrophic failure
- Literature review: ~1–1.5 columns, closely related work only
- Confirm `n_subcarriers=76` and pilot structure with ADM before locking obs shapes

---

## Dependencies

| Package | Purpose | Where |
|---|---|---|
| `numpy` | All numerical ops | Everywhere |
| `gymnasium 1.2.3` | RL env interface | Local + Colab |
| `stable-baselines3 2.8` | PPO training | Local + Colab |
| `torch 2.11` | Neural nets + Sionna backend | Colab + Euler |
| `sionna 2.0.1` | CDL-D channel (PyTorch-based, **not TF**) | Colab + Euler |
| `matplotlib 3.10.8` | Plots | Local + Colab |
| `pytest` | Unit tests | Local |

**Conda env:** `sionna-thesis` (Python 3.11)

**Sionna 2.0:** always pass `device=None` to CDL — never `"cuda"`. Module path: `sionna.phy.channel.tr38901.CDL`.
