# BT — Adversarial Cooperative PHY Jamming with MARL

**Program:** BSc Informatik, FS26 (Spring 2026)
**Author:** Rahul Rahman (`rrahman` / `rulecito`)
**Supervisor:** Antonio Di Maio (ADM)
**Last updated:** 2026-04-27

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
             non-stationary: positions update each step
             → Sionna recomputes channel coefficients,
               path loss, Doppler each timestep
```

**Primary metric:** SINR per subcarrier, spectral efficiency
**Primary focus:** Jammer (attacker) side — defender adaptation is out of scope

---

## Research Methodology (3 Stages)

### Stage 1 — Baselines
1. **Barrage jammer** — uniform noise across all subcarriers (`TimedBarrageJamming`) ✓
2. **PPO single-jammer** — MLP policy trained with SB3 PPO, MockChannel or SionnaChannel ✓
3. **PPO CTDE team** — centralized critic during training, decentralized execution at test time *(next)*

### Stage 2 — Weakness Analysis
Empirically identify limitations of MLP-based CTDE:
- Permutation sensitivity to teammate ordering
- No structural inductive bias for team coordination

### Stage 3 — Novel Method
Replace MLP team encoder with a **permutation-invariant encoder** (Set Transformer or GNN).
This encoder conditions the generative signal model.
Goal: outperform Stage 1 baselines in at least one well-defined scenario.

**Key open question:** Generative model architecture not yet fixed — GAN, VAE, or direct latent parameterization? Confirm with ADM.

### CTDE
**Centralized Training, Decentralized Execution:** centralized critic sees global state during training; each jammer acts on local obs only at test time. This is an implementation choice, not a contribution (per supervisor).

---

## Development Workflow

| Where | Channel | GPU | When |
|---|---|---|---|
| MacBook (local) | `MockChannel` | None — pure NumPy | Editing, unit tests |
| Google Colab | `SionnaChannel` CDL-D | Free T4 | All training (Stage 1+) |
| ETH Euler HPC | `SionnaChannel` CDL-D | A100 | Large-scale / very long runs |

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

# Single-jammer PPO training with Sionna
!python simulations/train_stage1.py --total-steps 500000 --n-envs 4 --channel-mode sionna

# Multi-agent simulation (fixed strategies)
!python simulations/multi_agent_sim.py --channel sionna --out /content/plots

# Display results inline
from IPython.display import Image, display
for f in ['exp1_sweep_jammers.png', 'exp2_sweep_defenders.png', 'exp3_heatmap.png']:
    display(Image(f'/content/plots/{f}'))
```

No `pyproject.toml` needed — plain `sys.path.insert` is sufficient.

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
│   ├── base.py                # BaseChannel ABC + MockChannel (NumPy, no GPU)
│   └── sionna_channel.py      # SionnaChannel — CDL-D via Sionna 2.x / PyTorch
│
├── envs/
│   └── wireless_env.py        # gymnasium.Env — the simulation core  (READ THIRD)
│
├── wrappers/
│   ├── sinr_normalize.py      # ObservationWrapper: SINR dB → [0, 1]
│   ├── rescale_action.py      # ActionWrapper: policy [-1,1] → env [0, max_power_dbm]
│   └── fixed_opponent.py      # Future: freeze one team for single-agent RL training
│
├── simulations/
│   ├── simple_barrage.py      # Stage 1a: one episode, barrage jammer, SINR plots
│   ├── train_stage1.py        # Stage 1b: SB3 PPO training, auto-generates plots  (READ FIRST)
│   ├── plot_stage1.py         # Standalone plotter: training curves + policy comparison
│   └── multi_agent_sim.py     # Multi-jammer/defender simulation (fixed strategies)
│
├── tests/
│   ├── test_scatter.py        # 11 tests
│   ├── test_channel.py        # 5 tests
│   ├── test_env.py            # 9 tests
│   └── test_wrappers.py       # 12 tests  →  37 total, all passing
│
├── models/                    # Created at runtime by train_stage1.py
│   └── stage1/
│       ├── best/best_model.zip
│       ├── checkpoints/
│       └── ppo_jammer_final.zip
│
├── logs/                      # Created at runtime
│   └── stage1/metrics.npz     # episode rewards + SINR per step
│
├── plots/                     # Created at runtime
│   ├── stage1/                # training_curves, policy_comparison, episode_heatmap
│   └── multi_agent/           # exp1_sweep_jammers, exp2_sweep_defenders, exp3_heatmap
│
└── CONTEXT.md                 # Raw project brief from earlier Claude.ai sessions
```

---

## How the Files Connect

```
train_stage1.py  ←─── START HERE (or multi_agent_sim.py)
│
├── core/config.py            ← READ SECOND — every config knob lives here
│
├── envs/wireless_env.py      ← READ THIRD — the heart of the simulation
│   │   reset() → place agents → initial obs
│   │   step(action) → compute SINR → reward → obs
│   │
│   ├── core/scatter.py       place(count, bounds) → positions (N,3)
│   │
│   ├── channel/base.py       compute_sinr(positions, powers)
│   │     MockChannel         → SINR (N_rx, N_sub) dB   [NumPy, local]
│   │
│   └── channel/sionna_channel.py
│         SionnaChannel       → SINR with CDL-D fading   [PyTorch, Colab/Euler]
│
├── wrappers/sinr_normalize.py   obs[:76] raw dB  → [0, 1]
├── wrappers/rescale_action.py   policy [-1,1]    → [0, 30 dBm]
│
├── stable_baselines3.PPO     learns: obs → action  to maximise reward
│     reward = −mean(SINR)    jammer wants SINR low at defender
│
└── plot_stage1.py            auto-called at end of training
      reads metrics.npz + saved model
      writes: training_curves.png, policy_comparison.png, episode_heatmap.png
```

**Data flowing through one step:**
```
scatter.place()  ──► positions (x,y,z)
                             │
step(action)                 │
  jam_power = action [76 dBm values]
  channel.compute_sinr(positions, powers) ──► SINR (N_rx, 76) dB
  reward = −mean(SINR)
  obs = [SINR_per_sub(76) | norm_pos(3) | step_frac(1)]  shape=(80,)
```

---

## Module Reference

### `core/config.py`

**`OFDMConfig`:**
```python
n_subcarriers: int = 76           # ← confirm with ADM before locking obs shapes
subcarrier_spacing_hz: float = 15e3
carrier_frequency_hz: float = 3.5e9
modulation: str = "QPSK"
```

**`AgentSpec`:**
```python
role: AgentRole                   # LEGITIMATE | JAMMER
count: int
policy: Optional[BaseStrategy]    # None = RL trains this; set = evaluation/fixed
ofdm: OFDMConfig
scatter_mode: ScatterMode         # RANDOM | SYMMETRIC | CUSTOM
max_power_dbm: float = 30.0       # 30 dBm = 1 Watt
mobile: bool = False              # mobility scaffold exists but NOT wired into step()
waveform_type: str = "power"      # Stage 1; "latent" planned for Stage 3
latent_dim: int = 16              # size of latent vector for generative model
```

**`EnvironmentConfig`:**
```python
space_bounds: tuple = (100.0, 100.0, 50.0)  # metres
channel_mode: str = "mock"        # "mock" | "sionna"
max_steps: int = 200
seed: Optional[int] = 42
```

> **Known bug:** `EnvironmentConfig.__post_init__` is at module scope (line 77) — not inside the class. Validation never runs. Fix before production.

---

### `core/strategy.py`

```python
BaseStrategy.act(obs: np.ndarray) -> np.ndarray  # local obs → action vector
BaseStrategy.reset() -> None                      # called at episode start
```

| Class | Stage | Notes |
|---|---|---|
| `RandomStrategy(action_space)` | dev/test | uniform sample |
| `TimedBarrageJamming(n_sub, silent_steps, max_power_dbm)` | Stage 1a | silent N steps then full power |
| *(MLP CTDE policy)* | Stage 1b | **not yet implemented** |
| *(Set Transformer / GNN)* | Stage 3 | placeholder at bottom of file |

---

### `channel/base.py` — MockChannel

```python
compute_sinr(tx_pos, rx_pos, jam_pos, tx_power_dbm, jam_power_dbm,
             noise_power_dbm, n_subcarriers) -> np.ndarray  # (N_rx, N_sub) dB
```

Log-distance path loss: `PL(d) = 40 + 10·3.5·log10(d)` dB. No fading. All subcarriers see identical SINR. Fast, no GPU. Use for local dev and unit tests only.

---

### `channel/sionna_channel.py` — SionnaChannel

**Sionna 2.0 is PyTorch-based** (not TensorFlow like 0.x). Module path: `sionna.phy.channel.tr38901.CDL`.

Two-component model:
1. **Large-scale path loss** — same log-distance formula as MockChannel
2. **Small-scale fading** — CDL-D (LOS-dominant, 13 paths, 30 ns delay spread): frequency-selective SINR across subcarriers

**Critical Sionna 2.0 gotcha:** Sionna uses its own device registry (`sionna.phy.config.available_devices`). Never pass `"cuda"` — it will raise `ValueError: Invalid device: cuda`. Always pass `device=None` to the CDL constructor and read back `sionna.phy.config.device` to know the chosen device.

```python
# Wrong — crashes on Colab
CDL(..., device="cuda")

# Correct — Sionna auto-selects from its own registry
CDL(..., device=None)
torch_device = "cuda" if sionna_config.device != "cpu" else "cpu"
```

Swap channel at config level — everything else stays identical:
```python
EnvironmentConfig(channel_mode="mock")    # local
EnvironmentConfig(channel_mode="sionna")  # Colab/Euler
```

---

### `envs/wireless_env.py` — WirelessEnv

Single-agent flat `gym.Env`. RL agent controls **jammer side**. Defenders are passive.

**Observation** `(80,) float32`:
```
obs[:76]   SINR per subcarrier (dB) at first defender RX
obs[76:79] defender position normalised to [0, 1]
obs[79]    step fraction (current_step / max_steps)
```

**Action** `(76,) float32`: per-subcarrier power in dBm, range `[0, max_power_dbm]`

**Reward:** `−mean(SINR_dB)` — jammer minimises defender link quality

**Modes:**
- `jammers.policy = None` → training mode (RL supplies action)
- `jammers.policy = SomeStrategy()` → evaluation mode (policy acts, action arg ignored)

**Outstanding TODOs:**
1. TX/RX positions are the same array (`rx_positions=self.leg_positions`) — needs separate TX/RX once confirmed with ADM
2. All jammers share one action via `np.tile` — Phase 1.3 will give each jammer its own action
3. `mobile=True` in AgentSpec is not wired into `step()`

---

### `wrappers/`

Stack in this order before handing to SB3:
```python
env = WirelessEnv(config)
env = SINRNormalizeObservation(env, n_subcarriers=76)  # obs[:76]: dB → [0,1]
env = PowerRescaleAction(env)                          # action: [-1,1] → [0,30 dBm]
```

`FixedOpponentWrapper` — for future multi-agent dict env. Freezes one team at a fixed strategy, exposes only the learner's obs/action to SB3. Not usable against the current flat env.

---

### `simulations/train_stage1.py` — PPO Training

```bash
# local (MockChannel)
PYTHONPATH=. conda run -n sionna-thesis python simulations/train_stage1.py \
    --total-steps 500000 --n-envs 4

# Colab (SionnaChannel)
python simulations/train_stage1.py --total-steps 500000 --n-envs 4 --channel-mode sionna

# evaluate a saved model only (skip training)
python simulations/train_stage1.py --eval-only models/stage1/best/best_model
```

Saves to:
- `models/stage1/best/best_model.zip` — best checkpoint (EvalCallback)
- `models/stage1/checkpoints/` — every 50k steps
- `models/stage1/ppo_jammer_final.zip` — end of training
- `logs/stage1/metrics.npz` — episode rewards + SINR per step

Auto-calls `plot_stage1.py` at the end.

SB3 PPO hyperparameters: `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01`.

---

### `simulations/plot_stage1.py` — Result Plots

Auto-called by `train_stage1.py`. Can also run standalone:
```bash
python simulations/plot_stage1.py \
    --model models/stage1/best/best_model \
    --log   logs/stage1/metrics.npz \
    --out   plots/stage1 \
    --channel mock   # or sionna
```

Generates:
- `training_curves.png` — reward + defender SINR over timesteps (smoothed)
- `policy_comparison.png` — SINR box plots: PPO vs barrage vs random; learned power per subcarrier
- `episode_heatmap.png` — per-subcarrier SINR over one episode: PPO vs barrage

---

### `simulations/multi_agent_sim.py` — Multi-Agent Simulation

Fixed-strategy simulation. No RL training. N jammers + M defenders, all using `TimedBarrageJamming`.

```bash
# local
PYTHONPATH=. conda run -n sionna-thesis python simulations/multi_agent_sim.py

# Colab with Sionna
python simulations/multi_agent_sim.py --channel sionna --out /content/plots
```

Three experiments:
- **Exp 1:** sweep N_jammers (1–5), N_def=1 → SINR vs jammer count (shows diminishing returns of barrage)
- **Exp 2:** sweep N_defenders (1–3), N_jam=3 → per-defender SINR traces
- **Exp 3:** N_jam=3, N_def=2, per-subcarrier SINR heatmap over episode

> Note: all jammers share the same action in the current env (`np.tile`). Phase 1.3 will give each jammer its own learned action.

---

### `tests/` — 37 Unit Tests, All Passing

```bash
PYTHONPATH=. conda run -n sionna-thesis python -m pytest tests/ -v
```

| File | Tests | Covers |
|---|---|---|
| `test_scatter.py` | 11 | shapes, bounds, left/right halves, altitude, custom validation, factory |
| `test_channel.py` | 5 | output shape, finite values, no-jammer baseline, jammer proximity, multi-RX |
| `test_env.py` | 9 | obs/action shapes, reward = −mean(SINR), truncation, info keys, seeded reproducibility |
| `test_wrappers.py` | 12 | SINR normalised to [0,1], action maps ±1 to boundaries, roundtrip invertibility |

---

## Current Status & Roadmap

### Done ✓
- [x] **Phase 0.1** — `envs/wireless_env.py`
- [x] **Phase 0.2** — `simulations/simple_barrage.py` + barrage plot
- [x] **Phase 0.3** — `wrappers/` (SINRNormalize, PowerRescale, FixedOpponent)
- [x] **Phase 0.4** — `tests/` (37 passing)
- [x] **Phase 1.1** — `simulations/train_stage1.py` (SB3 PPO, metrics callback, auto-plots)
- [x] **Phase 1.1b** — `simulations/plot_stage1.py` (training curves, comparison, heatmap)
- [x] **Phase 1.1c** — `channel/sionna_channel.py` (CDL-D, Sionna 2.0, Colab-ready)
- [x] **Phase 1.1d** — `simulations/multi_agent_sim.py` (N jammers × M defenders, fixed strategies)

### Next
```
Phase 1 — Stage 1 Baselines (continued)
  1.3  Extend env: per-jammer actions (N_jam × N_sub action space)
       → each jammer gets its own action vector, not np.tile of one shared action
  1.4  CTDE centralized critic: PPO with shared MLP policy + value fn over global state
  1.5  Long runs on Euler if Colab session limits hit

Phase 2 — Weakness Analysis
  2.1  Shuffle teammate ordering mid-training → measure SINR degradation
  2.2  Ablation: vary Nj (1, 2, 4, 8) → show MLP CTDE breaks at scale
  2.3  Document as motivation for Stage 3

Phase 3 — Novel Method
  3.1  Confirm generative model architecture with ADM (GAN / VAE / direct latent)
  3.2  Implement Set Transformer or GNN team encoder
  3.3  Wire: local obs → encoder → latent → generative model → waveform
  3.4  Train + compare against Stage 1 baselines
  3.5  Ablations: encoder type, latent dim, team size

Phase 4 — Write-up
  4.1  Figures: SINR curves, spectral heatmaps, team-size scaling
  4.2  Results tables, ablation analysis
  4.3  Paper draft
```

---

## Known Bugs

| File | Line | Issue |
|---|---|---|
| [core/config.py](core/config.py#L77) | 77 | `EnvironmentConfig.__post_init__` at module scope — validation never runs |
| [envs/wireless_env.py](envs/wireless_env.py#L203) | ~203 | `rx_positions=self.leg_positions` — TX and RX use same array; needs separate positions once confirmed with ADM |

---

## Supervisor Guidance

- Do not frame CTDE as a contribution — it is an implementation choice
- Do not claim MLP failure is catastrophic — frame as a *limitation* that structured encoders improve
- Keep literature review to ~1–1.5 columns, closely related work only
- Confirm `n_subcarriers=76` and pilot structure with ADM before finalising obs/action shapes

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

**Sionna 2.0 note:** completely rewrote from TensorFlow to PyTorch between 0.x and 2.0. Module paths changed: `sionna.channel.tr38901` → `sionna.phy.channel.tr38901`. Device strings: do not use PyTorch's `"cuda"` — pass `device=None` and let Sionna auto-select.
