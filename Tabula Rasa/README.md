# Tabula Rasa — Jamming Simulations

Fresh restart of baseline experiments. Goal: build intuition step by step before reintroducing RL agents, fading channels, and multi-antenna setups. Each simulation adds exactly one layer of complexity.

**Hard rule:** only library code (Sionna, SB3, gymnasium, scipy). No reuse from the old project.

**Stack:** Python, Sionna 2.x (`sionna.phy`, PyTorch backend), stable-baselines3, gymnasium, numpy, matplotlib.

---

## Current status (updated 2026-07-08)

> **READ THIS FIRST.** Single entry point — a major pivot plus the full 2026-07-02→08
> result arc. Read it top to bottom; it supersedes the older sim01–sim07 sections (kept
> for history). **The arc has a mid-course CORRECTION (the "Recheck" below): early Phase-0
> "stealth" numbers are superseded — read through to the sim08 milestone-2 FULL-SUITE result
> for the current bottom line.** Detailed writeups in the Phase 0 / Phase 0.5 / Simulation 08
> (milestones 1 and 2) sections further down.

### TL;DR — where we are

The original goal (a cooperative MARL jammer that fools the CNN, pure black-box) hit a
**confirmed dead end** at sim07. We pivoted to characterizing the detector and the
effectiveness–detectability frontier. The arc (every experiment ≤1 min on GPU):
- **Phase 0:** the SOTA spectrogram CNN is an *out-of-band-emission detector* — near-blind to
  spectrally-compliant in-band interference.
- **Phase 0.5:** retraining it to catch in-band jamming costs false-alarm rate/accuracy.
- **Recheck (CORRECTION):** the CNN-only "stealthy BER 0.42" was misleading — a trivial ENERGY
  detector catches any effective in-band jammer on the *lossless* channel. The out-of-band CNN
  finding is real (survives a spectrogram bug-fix); the stealth claim is not, on the lossless channel.
- **sim08 m1 (the payoff):** on a *realistic* fading+noise channel the stealth region REAPPEARS — a
  low-power jammer hides under the noise floor from the energy detector while causing **BER ≈ 0.20**.
  This is the paper's positive result, and it exists only on realistic channels.
- **sim08 m2 (the full suite):** a *channel-valid* CNN (retrained on faded signals, acc 94.3%) does
  cover the energy detector's blind spot — the CNN+energy **suite ≡ the CNN alone** (the energy
  detector catches *nothing* the CNN misses on the faded channel), pulling stealthy BER from ~0.22
  (energy-only) down to **0.065–0.11**. But the region does **not** close: a residual sparse
  ~16-subcarrier jammer stays stealthy-AND-effective (BER 0.065 at 30 dB, ~160× the 0.0004 floor),
  and the CNN pays a false-alarm cost (~20% FAR at 5 dB). The stealthy-effective region survives the
  full suite on realistic channels.

**Bottom line:** the paper's core is *the realistic channel vs the full detector suite*, and it now
holds against the complete suite: even a channel-valid CNN + energy detector leaves a residual
stealthy-effective region (BER ~0.065–0.11 at P(det)≤0.5). **Next: a *learned* channel-aware /
cooperative jammer (milestone 3) to WIDEN that residual region against the full suite** — the
matched-P_energy comparison is the metric that proves it.

### The pivot — why sim07 (blind MAPPO) was abandoned

sim07 (blind causal MAPPO, black-box) was confirmed a dead end through **run005 (job 101817)**
— all five runs failed identically: BER≈0.013, P(jam)≈0.3 (pure sampling noise), policy
entropy bit-for-bit flat (~178.4 = the policy never moved), reward falling as the β warmup
ramps a constant detection penalty. Two **structural** causes (not tuning):
1. Reward variance across the batch ≈ 0 → normalized advantages are pure noise → PPO has no
   gradient. Same wall as sim06/06b: a scalar frame-level reward carries no per-dimension info.
2. The K=1 sparsity that sim07 needed (to get under the detector's apparent "cliff") caps BER
   at ~2/52 ≈ 0.04 — too low to matter. The RL-over-raw-IQ negative result (sim06/06b/07)
   still stands as a publishable ablation, but it is no longer the plan.

### Today's findings (2026-07-02) — the new spine

**Phase 0 — effectiveness–detectability frontier, no RL (job 101860).**
Pure-inference sweep (`frontier/frontier_sweep.py`) over (# active subcarriers × power ×
structure) through the frozen sim06 detector, recording BOTH P(det) and BER. Result:
**the "99.79% SOTA detector" is essentially an OUT-OF-BAND-EMISSION detector, not a jamming
detector.** Decisive control (same power, same BER, differ only in whether the 12 guard/DC
null bins are hit): in-band broadband n=52 pwr=8 → P(det)=**0.011**, BER=0.36; the same
signal spread over all 64 bins (out-of-band) → P(det)=**0.999**; out-of-band at pwr=0.03
causing BER=0 → P(det)=**0.9999** (flags a harmless signal). A blind, in-band, temporally-held
jammer reaches **BER 0.25 @ P(det)≤0.05, 0.33 @ ≤0.10, 0.42 @ ≤0.50**; omniscient jam=−2·tx
reaches BER 1.0 @ P(det)=0.0016. This **reconciles** the old sim06 "broadband→0.999" wall (that
probe injected into guard/DC bins) and sim07's "4-subcarrier cliff" (its top-K over the full
64-bin grid was selecting guard/DC bins). The earlier "stealth–effectiveness tension is
fundamental / BER capped at 0.04" conclusion was WRONG — it was an out-of-band-leakage artifact.
Outputs: `artifacts/frontier/{frontier,cliff,ber_vs_nactive,coherence}.png`, `results.json`.

**Phase 0.5 — retrain the detector with in-band jammers (job 101866).**
`frontier/retrain_detector_inband.py` trains a fresh EfficientNet on clean + 4 classical +
in-band jammers, then re-sweeps. Outcome = "blind spot closes, but only at a real cost →
genuine tradeoff": in-band broadband P(det) 0.011→**0.995**, BUT overall accuracy
99.79%→**90.5%**, FAR 0%→**3.8%** (up to ~50% on near-zero-power in-band perturbations),
in-band detection only reaches **68%** (a third still evade), and a residual **sparse** blind
spot remains (n=3 evades at P(det)≤0.5, BER 0.02). Stealthy BER collapses 0.42→0.02; even the
omniscient jam=−2·tx (statistically clean rx=−tx) is now flagged at ≈0.15. **A spectrogram-CNN
cannot catch spectrally-compliant in-band interference without sacrificing false-alarm rate and
clean accuracy — an intrinsic tradeoff, not just a data gap.** Caveat: this is a quick 50-epoch
naive retrain; in-band samples were labeled "jammed" even when BER≈0, which inflates FAR — a
careful version (BER-thresholded labels, threshold calibration) would sharpen the exact numbers.
Detector saved: `artifacts/frontier/detector/run001_best.pt`. Re-swept frontier:
`artifacts/frontier_inband/`.

**sim08 milestone 1 — realistic channel + channel-aware frontier (job 101870).**
`simulation08/channel.py` (`MultiLinkChannel`: per-link frequency-selective TDL fading + AWGN
at target Eb/N0 + perfect-CSI ZF equalization) and `simulation08/frontier_channel.py`. Results:
(1) channel is physically correct (clean BER floor 0.088 @ 5 dB → 0.0003 @ 30 dB); (2) **a
sparse in-band jammer imposes an SNR-INDEPENDENT BER floor** (~0.05–0.07 @ n=8 across all SNR;
broadband ~0.34) — the jammer wins in the high-SNR regime because the impairment is
interference, not noise; (3) **channel-aware beats blind subcarrier selection** by up to **+70%
BER at equal power** (largest when jammer power is constrained, i.e. the stealthy regime; the
genie is not optimal so this is a lower bound) → direct motivation for a *learned* channel-aware
jammer. Caveat: the sim06 detector is lossless-trained and **invalid on the faded channel**
(clean FAR≈13%, no clean/jammed separation), so its P(det) here is indicative only → fixed in
milestone 2. Outputs: `artifacts/sim08/frontier/{ber_vs_snr,channelaware_vs_blind}.png`.

### Recheck (2026-07-03, job 102115) — IMPORTANT CORRECTION to Phase 0

Prompted by scrutiny of the Li et al. replication (their repo confirms EfficientNet-B0 is one
of their 4 CNNs, but they use REAL over-the-air SDR data + train from scratch; spectrogram
generation is undocumented). Two faithfulness fixes:
1. The detector spectrogram was **real-part-only** (a bug) → corrected to the standard **complex
   two-sided STFT** (`detector.py`); detector retrained → `artifacts/sim06/detector/run003_best.pt`
   (val acc 99.8%, FAR 0%).
2. Added an **energy detector** (mean received power vs a clean-calibrated 1% FAR threshold) so
   the frontier is evaluated against the **detector SUITE**, not the CNN alone. Code:
   `frontier/recheck_suite.py`, `submit_recheck.sh`; outputs `artifacts/frontier_recheck/`.

**(A) The out-of-band finding SURVIVES the complex-STFT fix — it was real.** Same power/BER
(0.362): in-band P(CNN)=**0.086** vs out-of-band P(CNN)=**1.000** (out-of-band = 1.000 at every
power). The spectrogram CNN genuinely is a band-edge/out-of-band detector.

**(B) The energy detector DEMOLISHES the "stealthy BER 0.42" claim.** Against the suite
(CNN OR energy), max stealthy BER: P(det)≤0.05 → **none**; ≤0.10 → **0.000**; ≤0.50 → **0.005**.
In-band jamming that flips bits necessarily raises power → a trivial power meter catches it. The
original "in-band jammer reaches BER 0.42 at P<0.05" was an artifact of evaluating against the
CNN **alone**. **The "SOTA CNN is blind → jammer wins" story is dead as stated.**

**Crucial caveat (keeps sim08 alive):** this is the **noiseless lossless** channel — clean power
is a razor-sharp constant, so the energy threshold catches any added power. On a **realistic
channel** (finite SNR + fading) the threshold is looser and a low-power jammer can hide **under the
noise floor** — confirmed next.

### sim08 + energy detector (2026-07-03, job 102305) — STEALTH REVIVED on the realistic channel

Answers the recheck's open question. Folded the energy detector (calibrated per-SNR on FADED clean
frames, 1% FAR) into `frontier_channel.py`. Result: on the fading+noise channel a jammer stays
stealthy (P_energy≤0.5) while causing **BER ≈ 0.20–0.24 across ALL SNRs** (5–30 dB; clean floor
0.093→0.0003) — vs the lossless channel where the energy detector caught everything (max stealthy
BER 0.005). At 30 dB this drives a near-error-free link to BER 0.20 while hiding under the energy
detector. **The noise floor + fading provide cover the lossless channel could not → the
stealthy-AND-effective region is real, but only on realistic channels.** Caveats: (i) channel-aware
vs blind washed out in this max-over-configs metric (both ~0.2) — a matched-P_energy comparison is
needed to prove the channel-aware/learned-jammer advantage; (ii) this is energy detector + the
lossless-trained CNN (invalid on faded), so the FULL-suite claim (evades energy AND a channel-valid
CNN) needs milestone 2. Plot: `artifacts/sim08/frontier/stealth_vs_energy.png`.

### sim08 milestone 2 (2026-07-08, jobs 102316 + 102319) — the FULL SUITE on the realistic channel

Completes the suite: a **channel-valid CNN** + the energy detector, evaluated per-sample against the
frontier. Two steps.

**(1) Channel-valid detector (`retrain_detector_channel.py`, job 102316).** Retrained EfficientNet-B0
on the complex-STFT spectrogram of signals passed through `MultiLinkChannel` (TDL fading + AWGN),
Eb/N0 drawn uniformly in [5,30] dB per mini-batch. Classes: faded clean (0), faded + classical
time-domain jammer (1), faded + in-band freq-domain jammer through its own `h_jam` (sparse blind /
channel-aware / broadband, 1). Result: **held-out faded-mix acc 94.3%, DR 91.2%, FAR 2.3%, F1 0.94**
— a genuine clean/jammed boundary on the faded channel (the lossless-trained detector had ~13% FAR
and no separation here). Per-SNR on fresh faded samples: clean FAR 10%@5 dB → 1%@30 dB; in-band DR
88%@5 dB → 76%@30 dB (i.e. it still misses ~24% of in-band jammers at high SNR — a residual blind
spot). Saved: `artifacts/sim08/detector/run001_best.pt`.

**(2) Full-suite frontier (`frontier_channel.py` + per-sample `p_suite`, job 102319).** Folded the
CNN and energy detector into a per-sample SUITE (CNN hit OR energy hit) and swept the frontier.
Headline — max BER a jammer can cause while staying stealthy (P(det)≤0.5), realistic channel:

| Eb/N0 | clean floor | energy-only | CNN-only | **SUITE** |
|---|---|---|---|---|
| 5 dB  | 0.0795 | 0.244 | 0.114 | **0.114** |
| 10 dB | 0.0354 | 0.232 | 0.097 | **0.097** |
| 15 dB | 0.0091 | 0.223 | 0.073 | **0.073** |
| 20 dB | 0.0035 | 0.214 | 0.074 | **0.074** |
| 30 dB | 0.0004 | 0.224 | 0.065 | **0.065** |

Two findings:
- **The suite ≡ the CNN on the realistic channel.** SUITE = CNN-only at *every* SNR; across the whole
  sweep the energy detector catches **0** configs the CNN misses. Any jammer that evades the CNN
  (sparse, thin) also evades the power meter, so the OR adds nothing. The channel-valid CNN fully
  subsumes the energy detector here — the mirror image of the *lossless* recheck, where the energy
  detector subsumed the CNN. (So the expensive CNN earns its keep specifically on realistic channels.)
- **The stealthy-effective region shrinks but survives.** The CNN cuts stealthy BER from ~0.22
  (energy-only) to 0.065–0.11, but never to the floor: a residual **sparse ~16-subcarrier, power≈1**
  jammer stays under both detectors (p_cnn≈0.40–0.47, p_energy≈0.02) while causing BER 0.065–0.11.
  Most striking at high SNR: at 30 dB it drives a near-error-free link (floor 0.0004) to BER 0.065 —
  a ~160× degradation while stealthy, because the impairment is interference, not noise. Cost: the
  CNN's clean FAR is high at low SNR (~20% @ 5 dB), modest (1–3%) above.

**Bottom line:** on the realistic channel the full CNN+energy suite is much stronger than either
detector on the lossless channel, yet a stealthy-AND-effective region **still survives** it. That
residual region — and whether a *learned* channel-aware/cooperative jammer can widen it — is exactly
the milestone-3 question. Plot: `artifacts/sim08/frontier/stealth_suite_vs_snr.png` (CNN-only curve
sits exactly under the suite curve). Caveat: the in-band training labels are not BER-thresholded
(inflates FAR, as in Phase 0.5); a calibrated-threshold version would sharpen the exact FAR numbers.

### Revised paper framing (post-recheck)

From *"a cooperative MARL jammer that fools a CNN"* → to a two-sided, honest contribution:
1. **Detector characterization / complementarity:** a SOTA CNN OFDM jamming detector detects
   out-of-band emissions, not jamming — near-blind to in-band interference. But on its own that's
   not a stealth win: a trivial energy detector covers the in-band-power blind spot on idealized
   channels, so the expensive CNN mainly earns its keep against out-of-band/structured jammers.
   (Consistent with why Li et al. fuse feature+spectrogram models.) Closing the CNN's in-band blind
   spot by retraining also costs FAR/accuracy (Phase 0.5).
2. **The paper's core lives on the realistic channel vs the FULL SUITE — now demonstrated
   (milestone 2, jobs 102316+102319).** The full CNN+energy suite (with a *channel-valid* CNN)
   leaves a residual stealthy-effective region: BER 0.065–0.11 at P(det)≤0.5, ~160× the clean floor
   at 30 dB. On the faded channel the suite ≡ the channel-valid CNN (energy detector redundant),
   the mirror image of the lossless recheck (where energy subsumed the CNN) — so the expensive CNN
   earns its keep specifically on realistic channels. Remaining: show a channel-aware/cooperative
   *learned* jammer *widens* that residual region (matched-P(suite) comparison → the learned-jammer
   contribution). Target claim: *cooperative learned > single-agent > blind > classical*. The
   RL-over-raw-IQ negative result (sim06/06b/07) is an ablation, not the headline.

### Next steps (in order)

0. **DONE — sim08 milestone 2 (channel-valid detector + full suite).** `retrain_detector_channel.py`
   (job 102316) + full-suite `frontier_channel.py` (job 102319). Result above: the suite ≡ the CNN on
   the realistic channel, and a residual stealthy-effective region (BER 0.065–0.11 @ P(det)≤0.5)
   survives it. This is the paper's core detector figure. Detector: `artifacts/sim08/detector/run001_best.pt`.
1. **Matched-P(suite) comparison — proves the learned-jammer contribution.** In the residual stealthy
   region, compare channel-aware vs blind BER at *matched* SUITE-detection probability (not the
   max-over-configs metric, which washed the advantage out). The genie channel-aware ≈ blind in the
   current metric; the matched-detectability curve is what should separate them and justify "a
   channel-aware jammer widens the stealthy region." Extend the m2 frontier data (already has p_suite).
2. **sim08 milestone 3 — learned / cooperative channel-aware jammer (widen the residual region).**
   The residual sparse ~16-SC region is the target a learned jammer should push into against the full
   suite. Multiple agents + per-link channel diversity make "who jams which subcarrier at what power"
   a genuine coordination problem. Strongly favor the **surrogate-gradient transfer-attack** threat
   model (train a differentiable surrogate detector, backprop the jammer through it à la sim03b/sim04,
   evaluate transfer to the frozen CNN+energy suite) — it yields a training gradient, unlike the
   abandoned pure-black-box PPO. Frame the action as low-dim power/subcarrier allocation, not raw-IQ
   waveform synthesis (the regime where MARL failed in sim06/07).
3. **Refinements:** BER-thresholded in-band labels + threshold calibration for the m2 detector
   (current labels inflate FAR, esp. the ~20% @ 5 dB); regenerate the channel-aware-vs-blind plot at
   power=1 (gain ~70% vs ~24% at power=4); extend the suite with more classical detectors
   (kurtosis/GLRT/pilot-variance); optionally add TX/jammer **distance → path-loss gain** geometry
   (`MultiLinkChannel` already exposes `tx_gain_db`/`jammer_gains_db`) so position heterogeneity makes
   the m3 cooperation problem non-trivial.

### System as implemented now

1 TX → 1 RX, 64-SC OFDM (QPSK, 802.11a-like, Sionna). Two channels: the **lossless** grid
(`simulation06/ofdm.py`) and the **realistic** frequency-selective TDL fading + AWGN channel
(`simulation08/channel.py`). **NOTE: `detector.py`'s spectrogram is now the CORRECTED complex
two-sided STFT (was real-part-only) — detectors must be trained on this representation.** Detectors
on disk: `simulation06/artifacts/.../run002_best.pt` (original real-part STFT, superseded),
**`artifacts/sim06/detector/run003_best.pt` (complex-STFT, LOSSLESS-trained — the corrected lossless
CNN)**, the Phase 0.5 in-band-augmented `artifacts/frontier/detector/run001_best.pt` (real-part era),
and **`artifacts/sim08/detector/run001_best.pt` (complex-STFT, FADED-channel-trained — the
channel-valid CNN, milestone 2; the one to use on the realistic channel)**. The **energy detector**
(mean received power vs a clean-calibrated threshold = Li et al.'s power feature) is implemented in
`frontier/recheck_suite.py` and (per-SNR calibrated, folded into a per-sample CNN∨energy suite) in
`simulation08/frontier_channel.py`. Jammer families: sparse (blind / channel-aware), broadband
(in-band / out-of-band), omniscient (jam=−2·tx), plus the 4 classical jammers in
`simulation06/train_detector.py`.

### Key files

- `frontier/frontier_sweep.py` + `submit.sh` — Phase 0 frontier (lossless). Has `build_jam` (jammer
  families) + `detect_chunked`, reused everywhere.
- `frontier/retrain_detector_inband.py` + `submit_phase05.sh` — Phase 0.5 detector retrain + re-sweep.
- `frontier/recheck_suite.py` + `submit_recheck.sh` — RECHECK: retrain on complex STFT + energy-detector
  suite (job 102115). Energy detector = `frame_power()` vs clean-calibrated threshold.
- `frontier/spectrogram_figure.py` + `submit_fig.sh` — in-band vs out-of-band illustrative figure
  (`artifacts/frontier/inband_vs_outofband.png`).
- `simulation08/channel.py`, `simulation08/frontier_channel.py` (per-SNR energy detector + per-sample
  CNN∨energy SUITE), `simulation08/submit.sh` (retrain m2 detector + suite frontier),
  `simulation08/submit_frontier.sh` (suite frontier only, reuses saved detector) — sim08 realistic channel.
- `simulation08/retrain_detector_channel.py` — milestone 2: channel-valid CNN retrain (faded
  clean+classical+in-band, Eb/N0 5–30 dB) → `artifacts/sim08/detector/run001_best.pt`.
- `simulation06/{ofdm,detector,jammer,train_detector}.py` — OFDM chain, detector (complex STFT now),
  classical jammers. `train_detector.py` retrains the CNN via the (now-corrected) spectrogram.
- Artifacts: `artifacts/frontier/` (Phase 0 + Phase 0.5 detector), `artifacts/frontier_inband/`,
  `artifacts/frontier_recheck/` (suite), `artifacts/sim08/frontier/` (incl. `stealth_vs_energy.png`,
  `stealth_suite_vs_snr.png` = m2 full-suite figure), `artifacts/sim08/detector/` (run001 =
  channel-valid CNN), `artifacts/sim06/detector/` (run003 = corrected lossless CNN). Run log: `artifacts/RUNS.md`.

### Historical ladder (sim00–07, condensed — full sections below)

sim00–04 progressive complexity, lossless → 2-jammer NSF direct-gradient (found QPSK-like
structure, BER 0.33+, kurtosis evasion). sim05 CNN needs OFDM (flat QPSK failed, 78.9%). sim06
CNN-on-OFDM = 99.79% ✓; MAPPO jammer failed (broadband always detected, no reward gradient).
sim06b confirmed scalar reward can't teach input-correlated waveforms even in 2D. sim07 blind
causal MAPPO = dead end (see the pivot above). **Note:** the sim06/07 "broadband always
detected" and "4-subcarrier cliff" claims in those sections are now explained by Phase 0 as
out-of-band-leakage artifacts — read Phase 0 for the correction.

### Paper status

Related-works section in progress in a separate session — see `paper/README.md` for reference
triage, structure decisions, and open questions (Hameed/Ziemann inclusion, PyJama dropped).
System model and methodology sections are stubs. No lossless-channel results go in the paper;
Phase 0/0.5 are mechanism studies, and sim08 (realistic channel) is where the paper's channel
claims begin.

### Cluster/compute notes

Jobs land on `studgpu-node01`, a 5060ti node — the fastest standard GPU here (others: 1080ti
×24, 2080ti ×4, gb10 [DGX Spark-style] ×6 nodes/1 GPU each). No GPU-hour budget on the
`projects` account, only a concurrency cap of **1 GPU job at a time** (`MaxJobsPU=1`). The
`projects_4gpus` account is provisioned but unfunded (`GrpTRESMins=gres/gpu=0`); even if funded
its `MaxJobsPU=1` allows one ≤4-GPU job, NOT parallel single-GPU sweeps. Env:
`/work/scratch/rrahman/bt_env` (Sionna 2.0.1, PyTorch, torchvision). Submit with `sbatch
submit.sh` from each sim dir; do NOT run compute on the login node. Today's jobs all finished in
<1 min once scheduled.

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

## Frontier analysis (Phase 0) — effectiveness vs detectability (no RL)

> **⚠ CORRECTED — read the "Recheck (2026-07-03)" note in Current status first.** The "in-band
> jammer reaches BER 0.42 at P(det)<0.05" numbers below are **CNN-only** and misleading: an energy
> detector catches that jamming (it raises power). The out-of-band mechanism is real (survives the
> complex-STFT fix); the "stealth" claim is not, on the lossless channel.

**Files:** `frontier/frontier_sweep.py`, `frontier/submit.sh`

**What it is:** a pure-inference sweep (no training) that characterizes the frozen
sim06 CNN detector against controlled interference. For a grid of (number of active
subcarriers × per-SC power × structure) it injects a waveform into clean OFDM frames
and records BOTH P(detect) and BER, producing the achievable BER-vs-P(detect) frontier.
Reuses `simulation06/{ofdm,detector}.py` unchanged. Runs in ~40s on GPU (B=128).

**Motivation:** sim07 (blind MAPPO) was confirmed a dead end through run005 — same flat
failure as sim06/06b (BER≈0.013, P(jam)≈0.3 noise, policy entropy bit-for-bit flat).
Before spending more compute on RL, bound analytically what any jammer could achieve at
each stealth level, and find out whether learning can beat a random sparse strategy.

**Headline result (run job 101860, B=128, CONFIRMED) — the detector is an
OUT-OF-BAND-EMISSION detector, not a jamming detector.** The single decisive comparison,
same power and same BER, differing only in whether the interference touches the 12
guard/DC null bins:

| Strategy | n_active | power | P(detect) | BER |
|---|---|---|---|---|
| broadband **in-band** (52 eff SCs) | 52 | 8 | **0.011** | 0.360 |
| broadband **out-of-band** (all 64 bins) | 52 | 8 | **0.999** | 0.363 |
| broadband **out-of-band** | 52 | 0.03 | **0.9999** | 0.000 |

The detector fires at ~1.0 on *any* out-of-band energy (even power 0.03 causing zero BER)
and is near-blind (P(det)≈0.01) to in-band interference causing BER=0.36. This reconciles
the sim06 "broadband → P(jam)=0.999 even at power 0.01" wall: that probe added noise across
all 64 FFT bins **including guard/DC nulls** (an out-of-band spectral-mask violation). It
also likely explains sim07's "4-subcarrier cliff" — the jammer's top-K selection over the
full 64-bin grid was selecting guard/DC bins as it activated more subcarriers.

**Achievable stealthy operating points (blind, in-band, temporally-held):**

| Detection ceiling | Max BER | Config |
|---|---|---|
| P(det) ≤ 0.05 | **0.252** | all 52 SCs, power 1 |
| P(det) ≤ 0.10 | **0.335** | all 52 SCs, power 2 |
| P(det) ≤ 0.50 | **0.419** | all 52 SCs, power 8 |
| omniscient (jam=−2·tx) | **1.000** | all 52 SCs, P(det)=0.0016 |

So a large stealthy-AND-effective region exists — the earlier "BER capped at ~0.04 by the
sparsity needed for stealth" conclusion was an artifact of out-of-band leakage, not a
fundamental tradeoff. Temporal coherence (held vs hopping) turned out to be a minor axis;
**in-band vs out-of-band is the dominant one.**

**Outputs:** `artifacts/frontier/{frontier,cliff,ber_vs_nactive,coherence}.png`,
`artifacts/frontier/results.json`.

**How to run:**
```bash
cd "Tabula Rasa/frontier"
sbatch submit.sh          # full GPU sweep (~40s + queue)
# or: python frontier_sweep.py --smoke --device cpu   # quick sanity
```

**Implications for the plan:**
- The paper now has a positive, defensible finding: a 99.79%-accurate SOTA CNN jamming
  detector is essentially a spectral-mask detector, near-blind to spectrally-compliant
  in-band interference at any power.
- Next (Phase 0.5): retrain/augment the detector with in-band jammer samples (held/hopping/
  matched) — does the blind spot close and force a genuine stealth/effectiveness tradeoff?
- Then a direct-gradient or surrogate-transfer jammer to reach the frontier-optimal in-band
  waveform, and **sim08 (realistic channel)** where cooperative channel-aware allocation is
  where learning genuinely helps and where paper claims begin.

### Phase 0.5 — retrain the detector with in-band jammers

**Files:** `frontier/retrain_detector_inband.py`, `frontier/submit_phase05.sh`

**What it is:** trains a fresh EfficientNet on clean + the 4 classical jammers + **in-band
jammers** (sparse/broadband/held/hopping/matched, injected in the frequency domain on the
52 effective SCs — the spectrally-compliant family the original detector never saw), then
re-runs the frontier sweep against it. Tests whether the out-of-band blind spot can be
closed. Detector → `artifacts/frontier/detector/run001_best.pt`; re-swept frontier →
`artifacts/frontier_inband/`.

**Result (job 101866) — the blind spot closes, but only at a real cost → genuine tradeoff:**

| | Original detector | Retrained (+ in-band) |
|---|---|---|
| Overall accuracy | 99.79% | **90.5%** |
| FAR | 0.00% | **3.81%** |
| in-band broadband n=52 pwr=8 → P(det) | 0.011 | **0.995** |
| in-band DR (fresh in-band jammers) | ~0% | **68%** (a third still evade) |
| max stealthy BER at P(det) ≤ 0.05 | 0.252 | **none stays this stealthy** |
| max stealthy BER at P(det) ≤ 0.50 | 0.419 | **0.020** (sparse, n=3) |

Teaching the CNN to catch in-band interference works for high-occupancy jammers but drops
clean-signal accuracy 99.8→90.5%, raises FAR 0→3.8% (up to ~50% on near-zero-power in-band
perturbations), still misses ~32% of in-band jammers, and leaves a residual **sparse** blind
spot (n=3 evades at P(det)≤0.5). The re-swept frontier collapses: stealthy BER drops from
0.42 to 0.02, and every high-BER point is pushed to P(det)≈1.0. Even the omniscient
`jam=−2·tx` (statistically clean `rx=−tx`) is now flagged at P(det)≈0.15. **Conclusion: a
spectrogram-CNN detector cannot catch spectrally-compliant in-band interference without
sacrificing false-alarm rate and clean accuracy — an intrinsic effectiveness–detectability
tradeoff, not just a training-data gap.**

**Caveats:** this retrain is a quick 50-epoch naive augmentation, and in-band samples were
labeled "jammed" even when they cause BER≈0 (which inflates FAR). A careful detector-design
study (BER-thresholded labels, threshold calibration, more data) would sharpen the exact
tradeoff — but the qualitative result (closing the blind spot costs FAR) is robust.

---

## Simulation 08 — realistic channel (milestone 1: channel-aware frontier)

**Files:** `simulation08/channel.py`, `simulation08/frontier_channel.py`, `simulation08/submit.sh`

**What it is:** the realistic-channel phase where paper claims begin. Adds finite SNR and
per-link frequency-selective fading to the frontier. `channel.py` (`MultiLinkChannel`) uses
Sionna `tr38901.TDL` (model C, 100 ns delay spread, 5.2 GHz) via `GenerateOFDMChannel` to give
an independent per-subcarrier frequency response for each link (TX→RX and each jammer→RX),
plus per-link average path gain (geometry) and AWGN at a target Eb/N0. The jammer transmits
through its OWN channel `h_jam` (a blind jammer doesn't know it); the RX does perfect-CSI ZF
equalization of the TX link. After equalization the effective interference on subcarrier n is
`(h_jam[n]/h_tx[n])·jam[n]` — so hitting subcarriers where the jammer is strong relative to
the TX matters, which is exactly the structure a learned, channel-aware jammer can exploit.

**Milestone 1 result (job 101870, B=128, 54s):**
1. **Channel is physically correct** — clean BER floor waterfalls 0.088 (5 dB) → 0.0003 (30 dB)
   for QPSK over fading with ZF equalization.
2. **A sparse in-band jammer imposes an SNR-independent BER floor.** Under an 8-subcarrier
   jammer BER stays pinned at ~0.05–0.07 across 5–30 dB, while the clean link would be
   essentially error-free at high SNR. Broadband in-band pins BER ~0.34 at all SNR. This is the
   clean "the jammer wins in the high-SNR regime" result — the impairment is interference, not
   noise, so more transmit power can't fix it.
3. **Channel-aware beats blind subcarrier selection.** A genie that picks the top-n subcarriers
   by `|h_jam/h_tx|` gets up to **~70% more BER than blind at equal power** (blind n=8=0.038 vs
   channel-aware n=8=0.064 @ 20 dB, power=1), and the gain grows with SNR and is largest when
   jammer power is constrained (the stealthy regime). Since the genie isn't optimal, this is a
   *lower bound* on the channel-aware benefit → direct motivation for a learned jammer.

**Energy detector on the realistic channel (job 102305) — REVIVES the stealth premise.** After
the Phase 0 recheck showed a power-threshold energy detector demolishes stealth on the *lossless*
channel (it catches any added power), we folded the same energy detector (calibrated per-SNR on
FADED clean frames to 1% FAR) into `frontier_channel.py`. On the **fading + noise** channel a
jammer stays stealthy (P_energy ≤ 0.5) while causing **BER ≈ 0.20–0.24 across all SNRs** (clean
floor 0.093 at 5 dB → 0.0003 at 30 dB). At 30 dB that's a near-error-free link driven to BER 0.20
*while hiding under the energy detector* — impossible on the lossless channel (max stealthy BER
0.005 there). The noise floor + fading make clean power fluctuate, loosening the threshold and
giving a low-power jammer room to hide. **So the stealthy-and-effective region is real, but only
on realistic channels** — which is where the paper's claims live anyway. Caveat: the channel-aware
vs blind advantage washes out in this max-over-configs metric (both ~0.2; needs a matched-P_energy
comparison); and this is energy detector + lossless-trained CNN, so the *full-suite* claim needs
the channel-valid CNN (milestone 2). Plot: `artifacts/sim08/frontier/stealth_vs_energy.png`.

**Detector caveat (→ milestone 2):** the CNN here is the sim06 detector, trained on a lossless
channel and **invalid on the faded channel** (clean false-alarm ≈13%, no clean/jammed separation),
so its P(det) numbers are indicative only. A channel-valid detector, retrained on faded clean +
classical + in-band signals, is milestone 2 — after which the frontier gets real *full-suite*
detectability numbers (CNN + energy).

> **UPDATE (2026-07-08): milestone 2 is DONE** (`retrain_detector_channel.py`, jobs 102316+102319).
> Channel-valid CNN acc 94.3%; the full CNN+energy suite ≡ the CNN on the faded channel; a residual
> stealthy-effective region (BER 0.065–0.11 @ P(det)≤0.5) survives it. See the **"sim08 milestone 2"
> subsection in Current status** (top of file) for the full writeup — it supersedes this caveat.

**Outputs:** `artifacts/sim08/frontier/{ber_vs_snr,channelaware_vs_blind,stealth_vs_energy,stealth_suite_vs_snr}.png`,
`results.json`; detector `artifacts/sim08/detector/run001_best.pt`.

**How to run:**
```bash
cd "Tabula Rasa/simulation08"
sbatch submit.sh
# or: python frontier_channel.py --smoke --device cpu   # quick sanity
```

**sim08 roadmap:** (1 ✓) channel + channel-aware frontier. (2) retrain a channel-valid
detector on faded signals; re-sweep for real detectability. (3) cooperative MARL jammer —
multiple agents with per-link channel diversity make "who jams which subcarrier at what power"
a genuine coordination problem; target claim: cooperative learned > single-agent > blind >
classical, all against the same channel-valid detector.

---

## Staging / roadmap

**Principle: one axis per step.** If results misbehave, the cause is unambiguous.

```
sim06   plumbing milestone — lossless + omniscient — DONE
        detector: 99.79% accuracy ✓ ; MAPPO jammer failed (scalar reward)
sim07   blind causal MAPPO — DEAD END (see Current status → the pivot)
Phase 0 frontier (no RL) — DONE ✓ detector = out-of-band detector;
        in-band jammer reaches BER 0.42 @ P(det)<0.05
Phase 0.5 retrain detector w/ in-band — DONE ✓ blind spot closes but
        costs FAR/accuracy → intrinsic effectiveness–detectability tradeoff
sim08   realistic channel — IN PROGRESS
  m1 ✓  freq-selective TDL fading + channel-aware frontier
        (sparse jammer = SNR-independent BER floor; channel-aware > blind)
  m2 →  retrain a CHANNEL-VALID detector on faded signals; re-sweep
  m3 →  cooperative MARL jammer (per-link channel diversity = real
        coordination); target: cooperative learned > single > blind > classical
```

### Explicitly deferred (with reasons)

- **Realistic channel/SINR (sim08):** ~~deferred~~ — milestone 1 DONE (see Current
  status and the "Simulation 08" section). Channel model is
  `simulation08/channel.py`. Milestones 2 (channel-valid detector) and 3
  (cooperative MARL) are next.
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
