Thesis Project Brief: Adversarial Cooperative PHY Jamming with MARL
Context
Bachelor thesis supervised by Antonio Di Maio. Topic: cooperative multi-agent jamming of UAV OFDM networks using learned generative waveforms, simulated in NVIDIA Sionna.

System Model

Jammers (Nj agents): cooperative team, mobile, each outputs a latent vector → learned generative model → structured jamming waveform
Defenders (Nl UAV nodes): communicate over OFDM, adapt position + frequency + power
Channel: CDL-D fading profile in NVIDIA Sionna, non-stationary by construction (positions update each timestep → Sionna recomputes channel coefficients, path loss, Doppler)
Metric: SINR per subcarrier, spectral efficiency


Methodology (3 stages, focus is attacker side first)
Stage 1 — Baseline Experiments

Simple baseline: barrage jammer (uniform noise)
Sophisticated baseline: PPO team with CTDE (centralized critic during training, decentralized execution), waveform = structured noise via MLP policy

Stage 2 — Weakness Analysis

Empirically and theoretically identify limitations of MLP-based CTDE policy (e.g. permutation sensitivity to teammate ordering, no structural inductive bias for team coordination)

Stage 3 — Novel Method

Replace MLP team encoder with a permutation-invariant encoder: Set Transformer or Graph Neural Network
This conditions the generative signal model
Goal: outperform baselines in at least one well-defined scenario


Key Design Decisions / Open Questions

Generative model architecture for waveform not yet fixed (GAN, VAE, or direct latent parameterization?)
Sionna integration: positions fed back each timestep to recompute channel
Defender side is a future extension, not in scope for now
No mobility optimization for defenders in scope


What NOT to do (supervisor feedback)

Do not assume CTDE is a fundamental contribution — it is an implementation choice
Do not assume MLP failure is catastrophic; frame it as a limitation that structured encoders can improve
Keep literature review to ~1–1.5 columns focused on closely related work only