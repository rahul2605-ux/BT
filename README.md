# BT — Bachelor's Thesis: RL-Based Anti-Jamming in OFDM Wireless Networks

**Program:** BSc Informatik, FS26 (Spring 2026)
**Author:** Rahul Rahman (`rrahman` / `rulecito`)
**Last updated:** 2026-04-27

---

## What This Project Is

A reinforcement learning environment and training pipeline for studying **adversarial jamming and anti-jamming strategies** in OFDM wireless networks (think LTE/5G-style radio). The end goal is to train an RL policy ("THE strategy") that can out-maneuver a jammer by intelligently allocating power across subcarriers.

The simulation runs locally using a fast mock channel model and is intended to scale to ETH Zurich's **Euler HPC cluster** using NVIDIA Sionna (GPU ray-tracing channel simulator) for physically accurate propagation.

---

## Repository Layout

```
BT/
├── core/
│   ├── config.py        # All dataclass configs (OFDMConfig, AgentSpec, EnvironmentConfig)
│   ├── strategy.py      # BaseStrategy ABC + built-in strategies (Random, TimedBarrage, ...)
│   └── scatter.py       # Agent placement: Random, Symmetric, Custom; factory make_scatter()
├── channel/
│   └── base.py          # BaseChannel ABC + MockChannel (log-distance path loss)
├── envs/
│   └── wireless_env.py  # [EMPTY] — Gymnasium environment to be implemented
├── simulations/
│   └── simple_barrage.py# [EMPTY] — script: run a barrage jamming episode end-to-end
├── wrappers/            # [EMPTY] — Gymnasium wrappers (e.g. observation normalization)
├── tests/               # [EMPTY] — unit tests
└── README.md
```

---

## Architecture

### Agents

There are two **roles** (`core/config.py:AgentRole`):
- `LEGITIMATE` — the communicating node trying to maintain high SINR
- `JAMMER` — adversarial node trying to destroy the legitimate link

Each role is configured via `AgentSpec`, which bundles:
- `count` — number of agents of this type
- `policy` — a `BaseStrategy` (optional; `None` = env will assign a default or random policy)
- `ofdm` — `OFDMConfig` (subcarriers, spacing, carrier freq, pilot symbols, modulation)
- `scatter_mode` — how agents are placed spatially (random / symmetric / custom)
- `max_power_dbm` — power budget (30 dBm = 1 W by default)
- `mobile` — whether agents move between steps
- `waveform_type` — `"power"` (current) — describes what the action vector represents
- `latent_dim` — 16 — for latent-space policy architecture

### Environment Config (`EnvironmentConfig`)

```python
space_bounds: (100m, 100m, 50m)   # 3D simulation arena
channel_mode: "mock" | "sionna"   # mock for local, sionna for Euler
dt_s: 0.1                          # time step (seconds)
max_steps: 200                     # episode length
seed: 42
```

### Channel Model (`channel/base.py`)

`BaseChannel` requires a single method: `compute_sinr(...)` → `(N_rx, N_subcarriers)` SINR in dB.

**MockChannel** (local dev & unit tests):
- Log-distance path loss: `PL(d) = PL0 + 10 * alpha * log10(d/d0)` [dB]
- Default params: `alpha=3.5` (urban), `PL0=40 dB`, `d0=1 m`
- Aggregates TX signal power, per-subcarrier jammer interference, and thermal noise
- **Not physically accurate** — purely for fast iteration without GPU dependency

**SionnaChannel** (planned, Euler cluster):
- Replace MockChannel with NVIDIA Sionna's ray-tracing channel simulation
- Requires GPU; not yet implemented
- `channel_mode="sionna"` in EnvironmentConfig is the flag to switch

### Strategies (`core/strategy.py`)

`BaseStrategy` ABC has:
- `act(observation: np.ndarray) -> np.ndarray` — maps local obs to action
- `reset() -> None` — called at episode start (override for stateful policies)

Built-in strategies so far:
| Class | Description |
|---|---|
| `RandomStrategy` | Samples uniformly from action space (baseline) |
| `TimedBarrageJamming` | Silent for N steps, then full-power jam across all subcarriers |

Placeholders in the file:
- "More complex strategies" section — e.g. reactive jamming, sweep jamming
- "THE strategy" section — the learned RL anti-jamming (or anti-anti-jamming) policy

### Scatter (Agent Placement) (`core/scatter.py`)

`make_scatter(mode, **kwargs)` factory:
| Mode | Class | Use case |
|---|---|---|
| `"random"` | `RandomScatter` | Uniform random 3D placement |
| `"symmetric"` | `SymmetricScatter` | Grid on left/right half of arena (side="left"\|"right") |
| `"custom"` | `CustomScatter` | Fixed positions supplied by caller as `(N, 3)` ndarray |

---

## Key Concepts / Domain Knowledge

- **OFDM**: Orthogonal Frequency-Division Multiplexing — the waveform used in LTE/5G. The spectrum is divided into `n_subcarriers` orthogonal channels. Each subcarrier can be jammed independently.
- **SINR**: Signal-to-Interference-plus-Noise Ratio — the key metric. Higher SINR → better link quality → higher throughput.
- **Barrage jamming**: Jammer broadcasts at full power across all subcarriers — the simplest and most aggressive attack.
- **Power allocation**: The action space for the legitimate node (or jammer) is a vector of per-subcarrier power levels.
- **dBm**: Power in decibels relative to 1 milliwatt. 30 dBm = 1 Watt. SINR is also expressed in dB.

---

## What Exists vs What Needs to Be Built

### Done
- [x] `EnvironmentConfig` / `AgentSpec` / `OFDMConfig` dataclasses with validation
- [x] `AgentRole` and `ScatterMode` enums
- [x] `BaseStrategy` + `RandomStrategy` + `TimedBarrageJamming`
- [x] `BaseScatter` + `RandomScatter` + `SymmetricScatter` + `CustomScatter` + factory
- [x] `BaseChannel` + `MockChannel` with pairwise path-loss and per-subcarrier SINR

### Not Yet Built (priority order)
- [ ] **`envs/wireless_env.py`** — the main `gymnasium.Env` subclass
  - `__init__`: wire up config → scatter agents → instantiate channel
  - `reset()`: re-scatter (if not fixed), return initial obs
  - `step(action)`: apply action, compute new SINR, build obs, compute reward, check termination
  - Observation space: SINR per subcarrier + agent positions (TBD with advisor)
  - Action space: per-subcarrier power vector (continuous, Box)
  - Reward: e.g. mean SINR, or sum of log(1 + SINR)
- [ ] **`simulations/simple_barrage.py`** — end-to-end episode with `TimedBarrageJamming` vs a static legitimate node; plot SINR over time
- [ ] **`wrappers/`** — at minimum: observation normalizer, reward shaper
- [ ] **`tests/`** — unit tests for scatter, channel, config validation
- [ ] **`channel/sionna.py`** — `SionnaChannel` for Euler; wraps `sionna.channel.tr38901` or similar
- [ ] **RL training script** — PPO or SAC (likely via Stable-Baselines3 or CleanRL) for "THE strategy"

---

## Design Decisions & Open Questions

- `OFDMConfig` has a `TODO: confirm protocol semantics with ADM` — the pilot symbol / subcarrier count must be confirmed with the thesis advisor before locking down observation shapes.
- `waveform_type="power"` is the only implemented type. A latent waveform representation (using `latent_dim=16`) is anticipated but not yet defined.
- `mobile=False` — mobility is scaffolded but positions don't update between steps yet; this needs to be wired into `wireless_env.step()`.
- The `EnvironmentConfig.__post_init__` is defined outside the class body (bug at `config.py:77`) — it is a standalone function that never runs. Needs to be indented inside the class.

---

## Bug Tracker

| File | Line | Issue |
|---|---|---|
| [core/config.py](core/config.py#L77) | 77 | `__post_init__` for `EnvironmentConfig` is defined at module scope, not inside the class — validation never runs |

---

## Dependencies

- **Python 3.10+**
- `numpy` — all numerical ops
- `gymnasium` — RL environment interface (not yet imported but planned for `envs/`)
- `sionna` — NVIDIA channel simulator (Euler only, GPU required)
- Likely: `stable-baselines3` or `cleanrl` for RL training
- Dev environment: **conda** (VSCode configured with conda env manager)

---

## Notes for Future Self

1. Start with the `wireless_env.py` — nothing else can run without it.
2. The `simple_barrage.py` simulation is a good smoke test once the env exists.
3. Run MockChannel locally; only switch to SionnaChannel when submitting jobs to Euler.
4. The `latent_dim` field on `AgentSpec` suggests the eventual policy will encode the waveform into a latent space — keep this in mind when designing the observation/action space.
5. Fix the `EnvironmentConfig.__post_init__` indentation bug before the env goes live.
6. Confirm OFDM config values (especially `n_subcarriers=76` and pilot structure) with advisor before finalizing observation shapes.
