# BT — Adversarial Cooperative PHY Jamming with MARL

**Program:** BSc Informatik, FS26 (Spring 2026)
**Author:** Rahul Rahman (`rrahman` / `rulecito`)
**Supervisor:** Antonio Di Maio (ADM)
**Last updated:** 2026-04-27

---

## Thesis Goal

Train a **cooperative multi-agent jammer team** that learns to jam UAV OFDM communications using **learned generative waveforms**, simulated in NVIDIA Sionna.

The research question: does a **permutation-invariant team encoder** (Set Transformer or GNN) produce better cooperative jamming than a plain MLP-based CTDE policy — and why?

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

         ↕  CDL-D fading channel (NVIDIA Sionna)
             non-stationary: positions update each step
             Sionna recomputes channel coefficients,
             path loss, and Doppler each timestep
```

**Primary metric:** SINR per subcarrier, spectral efficiency
**Primary focus:** Jammer (attacker) side — defender adaptation is out of scope

---

## Research Methodology (3 Stages)

### Stage 1 — Baselines
1. **Barrage jammer** — uniform noise across all subcarriers (`TimedBarrageJamming`)
2. **PPO CTDE team** — centralized critic during training, decentralized execution at test time; waveform = structured noise via MLP policy

### Stage 2 — Weakness Analysis
Empirically and theoretically identify limitations of MLP-based CTDE:
- Permutation sensitivity to teammate ordering
- No structural inductive bias for team coordination

### Stage 3 — Novel Method
Replace MLP team encoder with a **permutation-invariant encoder**:
- Option A: **Set Transformer**
- Option B: **Graph Neural Network**

This encoder conditions the generative signal model.
Goal: outperform baselines in at least one well-defined scenario.

**Key open question:** Generative model architecture not yet fixed — GAN, VAE, or direct latent parameterization?

### What CTDE means
**Centralized Training, Decentralized Execution:**
- During training: centralized critic sees the global state (all agents' obs + actions) → lower variance
- During execution: each jammer acts using only its local observation
- This is an implementation choice, not a contribution (per supervisor)

---

## Development Workflow

| Where | Channel | GPU | When |
|---|---|---|---|
| MacBook (local) | `MockChannel` | None (NumPy only) | Editing, testing |
| Google Colab | `MockChannel` → `SionnaChannel` | Free T4 | Training runs |
| ETH Euler HPC | `SionnaChannel` | A100 | Production scale |

**Running locally:**
```bash
# Always run from repo root with PYTHONPATH set
cd BT/
PYTHONPATH=. conda run -n sionna-thesis python <script>

# Tests
PYTHONPATH=. conda run -n sionna-thesis python -m pytest tests/ -v

# Barrage simulation
PYTHONPATH=. conda run -n sionna-thesis python simulations/simple_barrage.py
```

**Running in Colab:**
```python
!git clone <repo_url>
import sys
sys.path.insert(0, '/content/BT')
# then import normally: from core.config import ...
```

No `pyproject.toml` or packaging needed — plain `sys.path` is sufficient.

---

## Repository Layout

```
BT/
├── core/
│   ├── config.py            # All dataclass configs + enums
│   ├── strategy.py          # BaseStrategy ABC + built-in strategies
│   └── scatter.py           # Agent placement strategies + factory
├── channel/
│   └── base.py              # BaseChannel ABC + MockChannel
├── envs/
│   └── wireless_env.py      # Gymnasium env (single-agent flat interface)
├── simulations/
│   ├── simple_barrage.py    # Stage 1 smoke test — barrage vs static defender
│   └── barrage_baseline.png # Output plot from simple_barrage.py
├── wrappers/
│   ├── sinr_normalize.py    # ObservationWrapper: SINR dB → [0, 1]
│   ├── rescale_action.py    # ActionWrapper: [-1, 1] → [0, max_power_dbm]
│   └── fixed_opponent.py    # Wrapper: freeze one team, expose single-agent interface
├── tests/
│   ├── test_scatter.py      # 11 tests
│   ├── test_channel.py      # 5 tests
│   ├── test_env.py          # 9 tests
│   └── test_wrappers.py     # 12 tests  →  37 total, all passing
├── CONTEXT.md               # Raw project brief from Claude.ai chat
└── README.md
```

---

## Module Reference

### `core/config.py` — Configs & Enums

**`AgentRole`** enum: `LEGITIMATE` | `JAMMER`

**`ScatterMode`** enum: `RANDOM` | `SYMMETRIC` | `CUSTOM`

**`OFDMConfig`** — waveform parameters:
```python
n_subcarriers: int = 76           # FFT size (resource grid width)
subcarrier_spacing_hz: float = 15e3   # 15 kHz — LTE/NR default
carrier_frequency_hz: float = 3.5e9  # 3.5 GHz — common 5G band
n_pilot_symbols: int = 2
modulation: str = "QPSK"
```
> **TODO:** Confirm n_subcarriers=76 and pilot structure with ADM before finalising obs/action shapes.

**`AgentSpec`** — per-team config:
```python
role: AgentRole
count: int
policy: Optional[BaseStrategy] = None   # None = RL trains this side
ofdm: OFDMConfig
scatter_mode: ScatterMode = RANDOM
positions: Optional[np.ndarray] = None  # required if CUSTOM
max_power_dbm: float = 30.0             # 30 dBm = 1 Watt
mobile: bool = False                    # mobility not yet wired into step()
waveform_type: str = "power"            # Stage 1 baseline; "latent" for Stage 3
latent_dim: int = 16                    # size of latent vector for generative model
```

**`EnvironmentConfig`**:
```python
space_bounds: tuple = (100.0, 100.0, 50.0)  # 3D arena in metres
legitimate: AgentSpec
jammers: AgentSpec
dt_s: float = 0.1
max_steps: int = 200
channel_mode: str = "mock"   # "mock" | "sionna"
seed: Optional[int] = 42
```

> **Known bug:** `EnvironmentConfig.__post_init__` is defined at module scope (line 77), not inside the class — validation never runs. Fix before going to production.

---

### `core/strategy.py` — Agent Policies

**`BaseStrategy`** ABC:
```python
act(observation: np.ndarray) -> np.ndarray   # local obs → action
reset() -> None                               # called at episode start
```

| Class | Stage | Description |
|---|---|---|
| `RandomStrategy` | dev/test | Uniform sample from action space |
| `TimedBarrageJamming` | Stage 1 | Silent for N steps, then full-power all subcarriers |
| *(MLP CTDE policy)* | Stage 1 | **Not yet implemented** |
| *(Set Transformer / GNN policy)* | Stage 3 | "THE strategy" — placeholder at file bottom |

---

### `core/scatter.py` — Agent Placement

`make_scatter(mode, **kwargs) -> BaseScatter` factory:

| Mode | Class | Behaviour |
|---|---|---|
| `"random"` | `RandomScatter` | Uniform 3D random within arena bounds |
| `"symmetric"` | `SymmetricScatter(side, altitude_fraction)` | Regular grid on left (`x < x_max/2`) or right half |
| `"custom"` | `CustomScatter(positions)` | Caller-supplied `(N, 3)` float32 array |

All `place(count, space_bounds, rng)` calls return `(count, 3) float32`.

---

### `channel/base.py` — Channel Model

**`BaseChannel`** ABC requires one method:
```python
compute_sinr(tx_positions, rx_positions, jam_positions,
             tx_power_dbm, jam_power_dbm, noise_power_dbm,
             n_subcarriers) -> np.ndarray  # (N_rx, N_subcarriers) in dB
```

**`MockChannel`** — local dev only, no GPU:
- Log-distance path loss: `PL(d) = PL0 + 10·α·log10(d/d0)` in dB
- Default: `α=3.5` (urban), `PL0=40 dB`, `d0=1 m`
- Aggregates TX signal, per-subcarrier jammer interference, and thermal noise
- Handles zero jammers (empty arrays) correctly

**`SionnaChannel`** — Euler only (not yet implemented):
- CDL-D fading profile (LOS-dominant, suited for UAV scenarios)
- Use `sionna.channel.tr38901.CDL` with profile `"D"`
- Positions updated every step → Sionna recomputes coefficients + Doppler
- Guarded import: only imported when `channel_mode="sionna"`

---

### `envs/wireless_env.py` — Gymnasium Environment

Single-agent flat `gym.Env`. The RL agent controls the **jammer side**. Defenders are passive (fixed TX power, no learned policy yet).

**Observation** `(80,) float32`:
```
obs[:76]   — SINR per subcarrier (dB) at first legitimate RX node
obs[76:79] — legitimate node position, normalised to [0, 1]
obs[79]    — step fraction (current_step / max_steps)
```

**Action** `(76,) float32`:
```
Per-subcarrier transmit power in dBm, range [0, max_power_dbm]
All jammers share the same action for now (true MARL dict interface is Phase 1.3)
```

**Reward:**
```python
reward = -mean(SINR_dB)   # jammer wants to minimise defender link quality
```

**Modes:**
- `jammers.policy = None` → training mode, RL framework supplies action
- `jammers.policy = SomeStrategy()` → evaluation mode, policy decides (action arg ignored)

**Key TODOs in this file:**
1. TX and RX positions are the same array (line with `rx_positions=self.leg_positions`) — needs separate TX/RX once system model is finalised with ADM
2. All jammers share one action — needs per-agent dict interface for true MARL (Phase 1.3)
3. Mobility: `mobile=True` in AgentSpec is not yet wired into `step()`
4. Wire in `SionnaChannel` when `channel_mode="sionna"`

---

### `simulations/simple_barrage.py` — Stage 1 Smoke Test

Runs one 150-step episode: jammer silent for 30 steps, then full barrage.
Produces `simulations/barrage_baseline.png` with:
1. Mean SINR over time — sharp drop visible at step 30
2. Per-subcarrier SINR heatmap — all subcarriers hit uniformly (expected for barrage)

```bash
PYTHONPATH=. conda run -n sionna-thesis python simulations/simple_barrage.py
```

---

### `wrappers/` — Gymnasium Wrappers

Stack order matters — apply in this order:
```python
env = WirelessEnv(config)
env = SINRNormalizeObservation(env, n_subcarriers=76)
env = PowerRescaleAction(env)
# env is now SB3-ready
```

#### `SINRNormalizeObservation(env, n_subcarriers, sinr_min_db=-20, sinr_max_db=40)`
Maps `obs[:n_sub]` from raw dB to `[0, 1]` using a fixed clip range.
Passes `obs[n_sub:]` (position + step fraction) through unchanged.
Why fixed range: SINR distribution shifts as policy improves — running stats are unstable early in training.

#### `PowerRescaleAction(env)`
Maps policy output `[-1, 1]` to env action space `[0, max_power_dbm]`.
```
env_action = low + (policy_action + 1) / 2 * (high - low)
```
Has `reverse_action()` for logging actual power values during training.

#### `FixedOpponentWrapper(env, opponent, learner_key="jammer")`
**For future multi-agent dict env only — not usable against current flat `WirelessEnv`.**
Freezes one team at a fixed `BaseStrategy`, exposes only the learner's obs/action to SB3.
Enables curriculum training:
- Phase 1: train jammer vs `TimedBarrageJamming` (fixed defender)
- Phase 3: swap in trained defender, co-train both sides

---

### `tests/` — Unit Tests (37 total, all passing)

| File | Tests | What's covered |
|---|---|---|
| `test_scatter.py` | 11 | Shape, bounds, left/right halves, altitude, custom validation, factory |
| `test_channel.py` | 5 | Output shape, finite values, no-jammer baseline, jammer distance effect, multi-RX |
| `test_env.py` | 9 | Obs/action shapes & dtype, reward = −mean(SINR), truncation, info keys, seeded reproducibility |
| `test_wrappers.py` | 12 | SINR clipped to [0,1], shape preserved, clip floor/ceiling, action space [−1,1], correct mapping, roundtrip |

```bash
PYTHONPATH=. conda run -n sionna-thesis python -m pytest tests/ -v
```

---

## Current Status & Roadmap

### Done ✓
- [x] Phase 0.1 — `envs/wireless_env.py` (env, reset, step, reward, render)
- [x] Phase 0.2 — `simulations/simple_barrage.py` (Stage 1 smoke test + plots)
- [x] Phase 0.3 — `wrappers/` (SINRNormalize, PowerRescale, FixedOpponent)
- [x] Phase 0.4 — `tests/` (37 tests, all passing)

### Up Next
```
Phase 1 — Stage 1 Baselines
  1.1  PPO training script — single jammer, MLP policy, MockChannel, SB3
  1.2  Logging — wandb or tensorboard: reward, mean SINR, episode length
  1.3  True multi-agent: per-jammer obs/action dict interface (Nj > 1)
  1.4  CTDE: add centralized critic seeing full global state
  1.5  Validate on Euler with SionnaChannel + CDL-D

Phase 2 — Weakness Analysis
  2.1  Shuffle teammate ordering → measure SINR degradation
  2.2  Ablation: vary Nj (1, 2, 4, 8) → show MLP CTDE scaling breaks
  2.3  Document findings as motivation for Stage 3

Phase 3 — Novel Method
  3.1  Decide generative model architecture with ADM (GAN / VAE / direct latent)
  3.2  Implement Set Transformer or GNN team encoder
  3.3  Wire encoder into CTDE: local obs → encoder → latent → generative model → waveform
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
| [core/config.py](core/config.py#L77) | 77 | `EnvironmentConfig.__post_init__` defined at module scope — validation never runs |
| [envs/wireless_env.py](envs/wireless_env.py#L205) | 205 | `rx_positions=self.leg_positions` — TX and RX use the same position array; needs separate TX/RX positions once system model is confirmed with ADM |

---

## Supervisor Guidance

- Do not frame CTDE as a fundamental contribution — it is an implementation choice
- Do not claim MLP failure is catastrophic — frame it as a *limitation* that structured encoders improve upon
- Keep literature review to ~1–1.5 columns, focused on closely related work only
- Confirm OFDM config values (`n_subcarriers=76`, pilot structure) before finalising observation shapes

---

## Dependencies

| Package | Used for | Where |
|---|---|---|
| `numpy` | All numerical ops | Everywhere |
| `gymnasium` | RL env interface | Local + Colab |
| `stable-baselines3` | PPO training | Phase 1.1+ |
| `matplotlib` | Simulation plots | Local + Colab |
| `pytest` | Unit tests | Local |
| `sionna` | CDL-D channel simulation | Euler only |
| `torch` | Neural network backend | Colab + Euler |

Conda env: **`sionna-thesis`** (Python 3.11, gymnasium 1.2.3, matplotlib 3.10.8)
