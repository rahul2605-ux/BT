# Paper progress — Related Works section

Working on the `\section{Related Work}` for `main.tex` (replacing the sprawling
draft currently in there — see "What to do with the existing content" below).
Target: **1–1.5 IEEE columns + one comparison table**, per supervisor feedback.

## Context: what the paper's contribution actually is

Three axes, none of which any single prior paper combines (see `Tabula Rasa/README.md`
for the full experimental backstory):
1. **IQ-level waveform synthesis** — not channel selection, not power allocation
2. **Cooperative MARL** — multiple jammer agents coordinate via CTDE
3. **Learned stealth** — evading a frozen CNN detector via a black-box threat model
   (score-only, no detector gradients)

Architecture: MAPPO + Neural Spline Flow (NSF) policy. This is a combination of
existing methods (see discussion below on "architectural contribution" framing) —
the paper's value is in (a) the problem formulation, (b) a systematic ablation
trail showing *why* simpler policies fail (Gaussian, GMM, GAN all ruled out with
evidence from sim02/03c), and (c) what the agents discover when they do learn.

## Draft

`related_works_draft.tex` in this directory is a reference draft — structure and
flow, not meant to be copied verbatim. Three `\paragraph{}` blocks + a gap
statement + a `table*` comparison table. ~15 references as currently drafted.

## Reference triage

### Tier 1 — already in `refs.bib`, keep
- `jamming_survey_2024` — Pirayesh & Zeng survey (opening cite)
- `article` — Zhang & Wu, HDRL Stackelberg cooperative jammers
- `valianti2024cooperative` — cooperative Q-learning jamming
- `abolhassani2025coordinated` — QMIX anti-jamming defense
- `NIPS2017_68a97503` — MADDPG / CTDE (Lowe et al.)
- `wen2025generative` — GAN-aided covert comms for cooperative jammers

### Tier 1 — NOT yet in `refs.bib`, need to add
- **Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games" (NeurIPS 2022)** — MAPPO foundational paper, you use MAPPO.
  https://arxiv.org/abs/2103.01955 → bib key `yu2022mappo`
- **Li et al., "Jamming Detection and Classification in OFDM-Based UAVs via
  Spectrogram-Tailored ML" (IEEE Access 2022)** — the detector you replicate
  (99.79% accuracy match). https://www.researchgate.net/publication/358452834
  → bib key `li2022jamming_ofdm`
- **Erpek, Sagduyu, Shi, "Deep Learning for Launching and Mitigating Wireless
  Jamming Attacks" (IEEE TCCN 2019)** — frames jamming as adversarial ML.
  https://arxiv.org/abs/1807.02567 → bib key `erpek2019deep`
- **Shi, Davaslioglu, Sagduyu, "Generative Adversarial Network in the Air"
  (IEEE TCCN 2021)** — GAN spoofing, closest generative-model prior art.
  https://arxiv.org/abs/2007.08363 → bib key `sagduyu2021gan`
- **Ward, Smofsky, Bhatt, "Normalizing Flows for Reinforcement Learning"
  (ICML Workshop 2019)** — flow policies in PPO, your architectural foundation.
  No clean arXiv link found — search ICML 2019 workshop proceedings.
  Alternative/backup: Mazoure et al., "Improving Exploration in SAC with NF
  Policies" (https://arxiv.org/abs/1906.02771) if Ward is hard to track down.
  → bib key `ward2019nf_rl`
- **Durkan et al., "Neural Spline Flows" (NeurIPS 2019)** — the NSF
  architecture itself. https://arxiv.org/abs/1906.04032 → bib key `durkan2019nsf`

### Decided: DROP
- **PyJama (Ulbricht/Marti et al., SPAWC 2024)** — discussed and dropped.
  Reasoning: shares almost none of the three novelty axes (no waveform
  synthesis, no RL, no multi-agent, no stealth objective) — its only
  connection is "also uses Sionna," which is tooling, not a scientific
  contribution. Would just be a row of "No" across every table column. If
  mentioned at all, belongs in the System Model / Methodology section when
  introducing Sionna as the simulation backend, not in Related Work.

### Open — still deciding
- **Hameed, Gyorgy, Gunduz, "The Best Defense Is a Good Offense: Adversarial
  Attacks to Avoid Modulation Detection" (IEEE TIFS 2021)**
  https://arxiv.org/abs/1902.10674 → bib key `hameed2021offense`
  Transmitter-side (not jamming) — perturbs own symbols to evade a modulation
  classifier while staying decodable. Closest conceptual match to "shape
  signal statistics to evade a learned detector," but it's covert comms, not
  disruption. Worth it or too tangential?
- **Ziemann & Metzler, "Adaptive LPD Radar Waveform Design with Generative
  Deep Learning" (2024)** https://arxiv.org/abs/2403.12254 → bib key
  `ziemann2024lpd`
  Radar domain, not comms. Strongest existing validation of the
  "generative model for dual-objective (effective + undetectable) waveform"
  paradigm — GAN generator vs. critic, same dual objective as your BER+stealth
  reward. Cross-domain cite — strengthens the framing or distracts?

### Tier 3 — considered, probably cut for space
- Sagduyu et al., "Multi-Agent Adversarial Attacks for Multi-Channel
  Communications" (AAMAS 2022) — multi-agent but discrete channel selection,
  not waveform synthesis. https://arxiv.org/abs/2201.09149
- Kim, Sagduyu et al., "Channel-Aware Adversarial Attacks..." (IEEE TWC 2022)
  — relevant once sim08 (realistic channel) lands, not before.
  https://arxiv.org/abs/2005.05321
- Flowers et al., "Real-time Over-the-air Adversarial Perturbations..." (2022)
  — validates practical feasibility over SDR, single-agent perturbation-based.
  https://arxiv.org/abs/2202.11197

## On "is this just an architectural combination?"

Came up in conversation — own concern, not supervisor feedback. Worth keeping
in mind while drafting the intro/framing of Related Work and the paper overall:

The honest answer is yes, MAPPO + NSF is a combination of existing methods, and
that's *fine* as a contribution **as long as the paper leads with the problem
and findings, not the architecture**. What justifies the combination:
1. A systematic ablation trail (sim02 Gaussian → sim03c GMM → GAN ruled out →
   NSF) showing *why* each simpler alternative structurally fails, not just
   "we picked NSF because it's good."
2. The problem formulation itself (cooperative blind RL jammer vs. black-box
   neural detector on OFDM) is novel — nobody has posed this exact problem.
3. What matters most: whatever emergent behavior comes out of sim07/sim08
   (e.g. pilot-aware jamming, sparse spectral structure). That's the actual
   paper-worthy result; the architecture is just the tool that got there.

Comparable precedent: PyJama is "Sionna + SGD," Sagduyu's GAN spoofing paper is
"GAN + wireless channel" — both published at solid venues. The bar is whether
the combination produces insight the parts alone couldn't.

## Next steps for the RW session

1. Resolve Hameed/Ziemann inclusion (see "Open" above)
2. Finalize reference list, add missing bib entries to `refs.bib`
3. Tighten `related_works_draft.tex` content into final prose (own voice, not
   copied from the draft)
4. Replace the `New Related Works` / `Related Works` / `Literature Review`
   sections in `main.tex` with the final version
5. Decide what to do with the existing sprawling content in `main.tex`
   (lines ~71–425 as of 2026-06-30) — likely comment out or move to a thesis-only
   appendix rather than delete, since some of it (CTDE explanation, mobility
   evasion discussion) may be useful for the full thesis even if cut from the
   paper

## What to do with the existing content in main.tex

`main.tex` currently has three separate, overlapping draft sections:
`New Related Works` (table stub), `Related Works` (long, many `\adm`/`\rar`
inline comments from supervisor + self), and `Literature Review` (more draft,
partially duplicate). These were exploratory writing, not structured. The plan
is to replace all three with one clean `\section{Related Work}` per the draft
here. Supervisor comments (`\adm{}`) embedded in the old text flagged: cut
LLM/WirelessAgent content (tangential), cut R-SFLLM (application-specific, out
of scope), keep CTDE/mobility content for thesis but not the paper, MDPI
journals (`electronics14163307`) are weak venues to avoid citing.
