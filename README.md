# BT — Bachelor's Thesis: Adversarial Cooperative PHY Jamming with MARL

**Program:** BSc Informatik, FS26 (Spring 2026)
**Author:** Rahul Rahman (`rrahman` / `rulecito`)
**Supervisor:** Antonio Di Maio (ADM)
**Last updated:** 2026-04-27

---

## Thesis Goal

Train a **cooperative multi-agent jammer team** that learns to jam UAV OFDM communications using **learned generative waveforms**, simulated in NVIDIA Sionna.

The research question is whether a **permutation-invariant team encoder** (Set Transformer or GNN) produces better cooperative jamming than a plain MLP-based CTDE policy — and why.

---

## System Model

```
Jammers (Nj agents)          Defenders / UAVs (Nl agents)
─────────────────────        ────────────────────────────
Mobile cooperative team      OFDM communicators
Each jammer:                 Adapt position + frequency + power
  latent vector (dim=16)
  → generative model
  → structured jam waveform

         ↕  CDL-D fading channel (NVIDIA Sionna)
             non-stationary: positions update each step
             → Sionna recomputes channel coefficients,
               path loss, Doppler each timestep
```

**Primary metric:** SINR per subcarrier, spectral efficiency
**Primary focus:** Jammer (attacker) side
**Out of scope for now:** defender adaptation, mobility optimization for defenders

---

## Research Methodology (3 Stages)

### Stage 1 — Baselines
1. **Barrage jammer** — uniform noise across all subcarriers (`waveform_type="power"`, `TimedBarrageJamming`)
2. **PPO CTDE team** — centralized critic during training, decentralized execution; waveform = structured noise via MLP policy

### Stage 2 — Weakness Analysis
Empirically and theoretically identify limitations of MLP-based CTDE:
- Permutation sensitivity to teammate ordering
- No structural inductive bias for team coordination

### Stage 3 — Novel Method
Replace the MLP team encoder with a **permutation-invariant encoder**:
- Option A: **Set Transformer**
- Option B: **Graph Neural Network**

This encoder conditions the generative signal model. Goal: outperform baselines in at least one well-defined scenario.

**Key open question:** Generative model architecture not yet fixed — GAN, VAE, or direct latent parameterization?

---

## CTDE Explained

**Centralized Training, Decentralized Execution:**
- During training: a centralized critic sees the global state (all agents' observations + actions) to reduce variance
- During execution: each jammer agent acts using only its local observation
- This is standard MARL — supervisor says it is an *implementation choice*, not a contribution in itself

---

## Repository Layout

```
BT/
├── core/
│   ├── config.py        # All dataclass configs (OFDMConfig, AgentSpec, EnvironmentConfig)
│   ├── strategy.py      # BaseStrategy ABC + built-in strategies (Random, TimedBarrage, ...)
│   └── scatter.py       # Agent placement: Random, Symmetric, Custom; factory make_scatter()
├── channel/
│   └── base.py          # BaseChannel ABC + MockChannel (log-distance path loss, local dev only)
├── envs/
│   └── wireless_env.py  # [EMPTY] — Gymnasium/PettingZoo environment to be implemented
├── simulations/
│   └── simple_barrage.py# [EMPTY] — Stage 1 baseline: barrage jammer smoke test
├── wrappers/            # [EMPTY] — Gymnasium wrappers (obs norm, action rescale, single-agent shim)
├── tests/               # [EMPTY] — unit tests
├── CONTEXT.md           # Raw project brief from Claude.ai chat logs
└── README.md
```

---

## Code Architecture

### Configs (`core/config.py`)

| Class | Purpose |
|---|---|
| `OFDMConfig` | Waveform params: 76 subcarriers, 15 kHz spacing, 3.5 GHz carrier, QPSK. **Confirm with ADM before locking obs shapes.** |
| `AgentSpec` | Per-team config: role, count, policy, scatter mode, max power (30 dBm), `latent_dim=16`, `waveform_type` |
| `EnvironmentConfig` | Arena (100×100×50 m), channel mode, dt=0.1 s, 200 steps/episode |

`latent_dim=16` on `AgentSpec` is the size of the latent vector each jammer outputs, which feeds into the generative waveform model.

`waveform_type="power"` is the Stage 1 baseline (direct per-subcarrier power). Stage 3 will use a learned latent type.

### Strategies (`core/strategy.py`)

`BaseStrategy.act(obs) -> action` maps local observation to action vector.

| Class | Stage | Notes |
|---|---|---|
| `RandomStrategy` | dev/test | Uniform sample from action space |
| `TimedBarrageJamming` | Stage 1 | Silent N steps, then full-power all subcarriers |
| *(MLP CTDE policy)* | Stage 1 | To be added — PPO with centralized critic |
| *(Set Transformer / GNN policy)* | Stage 3 | "THE strategy" placeholder at bottom of file |

### Scatter — Agent Placement (`core/scatter.py`)

`make_scatter(mode, **kwargs)` factory:

| Mode | Description |
|---|---|
| `"random"` | Uniform 3D random |
| `"symmetric"` | Grid on left or right half of arena (`side="left"/"right"`) |
| `"custom"` | Fixed positions supplied as `(N, 3)` array |

### Channel (`channel/base.py`)

`BaseChannel.compute_sinr(...) -> (N_rx, N_subcarriers)` in dB.

| Class | When | Notes |
|---|---|---|
| `MockChannel` | Local dev / unit tests | Log-distance path loss, no GPU needed, not physically accurate |
| `SionnaChannel` | Euler HPC cluster | CDL-D fading profile, GPU required, to be implemented |

CDL-D is a clustered delay line fading model (LOS-dominant, used for UAV-like scenarios). Sionna recomputes channel coefficients every timestep as positions update.

---

## What Exists vs What Needs to Be Built

### Done
- [x] Config dataclasses (`OFDMConfig`, `AgentSpec`, `EnvironmentConfig`) with enums
- [x] `BaseStrategy` + `RandomStrategy` + `TimedBarrageJamming`
- [x] `BaseScatter` + all three scatter modes + factory
- [x] `BaseChannel` + `MockChannel`

### Not Yet Built (rough priority order)
- [ ] **`envs/wireless_env.py`** — the Gymnasium (or PettingZoo) multi-agent env
  - Decide: single `gym.Env` with dict actions, or full PettingZoo AEC/parallel API
  - `reset()`: scatter agents, init Sionna scene, return initial obs
  - `step(action_dict)`: apply waveforms, compute SINR, compute reward, update positions
  - Observation per agent: local SINR + possibly teammates' positions (for centralized critic)
  - Action per jammer: latent vector (Stage 3) or per-subcarrier power (Stage 1)
  - Reward: e.g. negative mean SINR at defender (jammer wants to *minimize* defender SINR)
- [ ] **`simulations/simple_barrage.py`** — Stage 1 smoke test, plot SINR over time
- [ ] **`channel/sionna.py`** — `SionnaChannel` wrapping `sionna.channel.cir_to_ofdm_channel` with CDL-D
- [ ] **Wrappers** — obs normalizer, action rescaler, single-agent shim for solo training
- [ ] **MLP CTDE training script** — PPO, Stage 1 sophisticated baseline
- [ ] **Permutation-invariant encoder** — Set Transformer or GNN, Stage 3
- [ ] **Generative waveform model** — architecture TBD (GAN / VAE / direct latent)
- [ ] **`tests/`** — unit tests for scatter, channel, config, env

---

## Known Bugs

| File | Line | Issue |
|---|---|---|
| [core/config.py](core/config.py#L77) | 77 | `EnvironmentConfig.__post_init__` defined at module scope, not inside class — validation never runs |

---

## Supervisor Guidance (What NOT to Do)

- Do not frame CTDE as a fundamental contribution — it is an implementation choice
- Do not claim MLP failure is catastrophic — frame it as a *limitation* that structured encoders improve upon
- Keep literature review to ~1–1.5 columns, focused on closely related work only

---

## Dependencies

- `numpy` — numerical ops
- `gymnasium` — single-agent RL env interface (local dev)
- `pettingzoo` — possibly, for multi-agent env interface
- `sionna` — NVIDIA channel simulator, CDL-D (Euler cluster, GPU required)
- `torch` — neural network backend for policies
- Likely: `stable-baselines3` or `cleanrl` for PPO
- Dev environment: **conda** on macOS (VSCode + conda env manager)

---

## Notes for Future Sessions

1. **Start with `envs/wireless_env.py`** — nothing else runs without it. Decide gym vs PettingZoo interface first.
2. The reward is from the **jammer's perspective**: maximize interference → minimize defender SINR. This is opposite to what the MockChannel's SINR output represents (it's currently written from the defender's perspective).
3. The `latent_dim=16` on `AgentSpec` is the latent vector for the generative waveform — Stage 1 ignores it (uses direct power), Stage 3 feeds it through an encoder → generative model.
4. CDL-D is LOS-dominant (good for UAV links). When implementing `SionnaChannel`, use `sionna.channel.tr38901.CDL` with profile `"D"`.
5. Fix the `EnvironmentConfig.__post_init__` indentation bug before the env goes live.
6. Confirm `n_subcarriers=76` and pilot structure with ADM before finalizing observation/action shapes.
7. The permutation-invariant encoder operates over the jammer *team* — it needs to aggregate teammates' latent states in a way that doesn't depend on ordering.
