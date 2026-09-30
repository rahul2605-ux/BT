# Tabula Rasa — learned jamming under detection constraints

**Bachelor's thesis (ETH D-INFK), supervisor A. Di Maio.** Target: **ICC, deadline 2026-10-02.**
Last consolidated: 2026-09-28.

> **This file is the single entry point.** It is organised as:
> **[1. The whole picture](#part-1--the-whole-picture)** ·
> **[2. Goal & approach](#part-2--goal--approach)** ·
> **[3. Current state](#part-3--current-state)** ·
> **[4. Open questions & ideas](#part-4--open-questions--ideas)** ·
> then **[Appendix A: experiment history](#appendix-a--experiment-history)** (what each simulation
> falsified), **[Appendix B: supervisor record](#appendix-b--supervisor-record)**,
> **[Appendix C: engineering notes](#appendix-c--engineering-notes)**.
> Cluster operations live in **[`cluster/README.md`](cluster/README.md)** and are not repeated here.

---

# PART 1 — THE WHOLE PICTURE

## 1.1 What this is, in one page

> **STATE 2026-09-28 — the current state is in [§3.1](#31-status-line); the blocks below are older and
> kept as history.** In short: the single-jammer study is re-done on the run003 method (§3.3f/§3.3g) and
> stress-tested under shadowing (S1, §3.3k). Its candidate story went to the supervisor in the first update
> since 2026-09-12 (§B.1), with the proposal to leave MARL out of the ICC paper (§3.3j) and run one arms-race
> round (D6) instead. ICC deadline Fri 2026-10-02.
>
> **STATE 2026-09-17 — direction reframed (lit review + design), still pending the same sign-off (§4.1 #0b).**
> A literature review on jamming-waveform synthesis (Amuru–Buehrer TIFS 2015 and the generative/learned
> jammers: Zhou CGAN, DDPM WCNC'25, adversarial-jamming, jamming-bandits) settled two things: the
> BER-optimal jammer against known QPSK is **closed-form (Amuru)**, and the generative papers only
> *converge to* known optima — so a generator is an **instrument, not the finding**. The reframed,
> buildable contribution the user chose: a **characterisation study** — map a generative jammer's
> detectability against a **detector suite** on the matched BER–P(det) plane, with the **NP-optimal test
> as the ceiling detector**, so the claim is the *deployed-detector-vs-optimal-warden gap* (§2.1
> surviving item 1), not "novel stealthy jamming". Stealth returns here **as a measured axis**, so this
> is a *second option* for the supervisor alongside the 2026-09-16 drop-stealth proposal below — the
> same #0b email now presents both. Staged plan and design in §2.10 / §3.4. Desk work only — no code, no
> compute; `verify.py` still valid. **Nothing folded in as settled until he replies.**
>
> **STATE 2026-09-16 — stealth proposed for FULL retirement, awaiting supervisor sign-off (§4.1 #0b).**
> A session of experiments + literature search concluded stealth is not worth building on *at all*,
> extending the 2026-09-12 headline retirement (§2.1) to the **CGAN step-2 stealth conditioning** too:
> any undetectable-and-effective attack collapses to a trivial/genie solution, so even a method that
> "beats" simpler attacks under a detection constraint wins only a weak region. Refined rationale in
> §2.1 ("Sharpened 2026-09-16"). **Proposed replacement:** keep the reproduced CGAN jamming *waveform*
> and strengthen it with **MARL multi-jammer coordination as an *effectiveness* result** — a co-located
> detector means coordination buys coherent combining, not covertness — with detectability kept as an
> evaluation axis and adaptation cost as a possible secondary claim. **A short email stating this was
> drafted 2026-09-16; nothing is folded in as settled until he replies** (§2.10, §4.1 #0b).
>
> **STATE 2026-09-14 — exploratory track opened: [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).**
> The findings up to 2026-09-12 were judged too weak to carry the thesis. A new direction is being
> tried: **reproduce the CGAN jamming-waveform generator of Zhou et al. (ISSET 2025) on QPSK, then
> condition it on stealth** and evaluate against three detectors (power threshold · a statistical
> test, still to be chosen · the Zhang & Krunz 2023 CWT-CNN). The coordination direction described in
> the rest of this page is **paused, not superseded**. Live code for the new track is in `cgan/`.
> **Step 1 is closed as a partial reproduction (user decision 2026-09-16, §2.10; results §3.3b).** The
> GAN beats noise by 4.4–6.7 dB at BER 1e-3 (Zhou: 2–6 dB), but the first-reported gap match had scored the GAN
> synchronised and "optimal" asynchronous; under one consistent model the GAN *beats* "optimal". run001
> under async is step 2's reference generator. Next: the step-2 statistical detector (§4.2 Q8).
>
> **STATE 2026-09-16 — sim08 ablations done: [§3.3c](#33c-sim08-ablations--noise-jammer-power-number-of-jammers-2026-09-16).**
> Agreed with the supervisor: noise, jammer power and #jammers on the frozen sim08 detector suite, from
> read-only code in `sim08_ablation/`.

A **cooperative multi-agent generative jamming attacker** is built and evaluated against a link and
its detector. The question is what a *coordinated, learned* attacker achieves that a single or
uncoordinated one cannot — with detectability retained as an **evaluation axis** (how do two jammers
compare at equal exposure) rather than as the objective being optimised.

The project began as "a cooperative multi-agent RL jammer that fools a CNN detector". That bundled
four bets: (a) multi-agent cooperation, (b) reinforcement learning over raw IQ, (c) black-box access
to the detector, (d) **stealth as the attacker's objective.** A ladder of thirteen simulations
(sim00 → sim08) **falsified (b)+(c) as a method** — RL-over-raw-IQ with black-box access is
structurally untrainable here, not merely badly tuned. A literature check on 2026-09-12 then
**retired (d) as a headline** (§2.1). **(a) is what remains, and as of 2026-09-12 it is the
direction.**

After a supervisor mandate to **simplify as hard as possible** (2026-08-21), the working model is no
longer the 64-subcarrier OFDM stack but **M0**: one QPSK symbol, one channel, AWGN with swept σ.
The retreat buys something the big stack could never have: at this size the **Neyman–Pearson optimal
detector is computable in closed form**, so detectability claims read *"no detector can do better
than X"* instead of *"our CNN failed to catch it"*. That machinery is built, verified and keeps its
value under the new direction — it is what makes "at equal exposure" a measurable statement rather
than a hand-wave.

**Hard rule:** only library code (Sionna, SB3, gymnasium, scipy, PyTorch). No reuse from the old
project.

**Stack:** Python 3.11, PyTorch 2.9.1+cu128, Sionna 2.x (`sionna.phy`, PyTorch backend),
stable-baselines3, gymnasium, zuko, numpy, matplotlib.

## 1.2 The arc, in one table

Each row is a step of the ladder and **what it killed**. Full writeups in
[Appendix A](#appendix-a--experiment-history).

| Step | What it tested | Verdict |
|---|---|---|
| sim00 | observational baseline, lossless | measurement chain is trustworthy |
| sim01 | PPO jammer vs power threshold | works, but power tuning is the only available strategy |
| sim02 | + kurtosis detector | **a diagonal-Gaussian policy structurally cannot produce non-Gaussian output** |
| sim03 | NSF (normalizing flow) policy + PPO | flow is expressive enough; PPO still does not move it |
| sim03b | NSF + **direct gradient**, no RL | **the one method on the ladder that worked** — kurt −1.30, BER 0.17 |
| sim03c | GMM policy + PPO | closed, negative: GMM permutation symmetry absorbs the gradient |
| sim04 | 2 cooperative jammers, direct gradient | coordinated solution exists and is gradient-reachable (BER 0.35 @ power 0.9) |
| sim04b | Sionna on GPU | validation only |
| sim05 | CNN spectrogram detector on flat QPSK | fails (78.9%) — **spectrograms need OFDM** |
| sim06 | OFDM + CNN detector (99.79%) + MAPPO jammer | detector replicates Li et al.; **MAPPO jammer fails — no reward gradient** |
| sim06b | same in 2D | **not a dimensionality problem** — a scalar reward cannot teach input-correlated output |
| sim07 | blind causal MAPPO, black-box | **dead end, confirmed.** The pivot happens here |
| Phase 0 | frontier sweep, no RL | the "SOTA detector" is an **out-of-band-emission detector** |
| Phase 0.5 | retrain it on in-band jammers | blind spot closes but costs accuracy 99.8→90.5%, FAR 0→3.8% |
| Recheck | complex-STFT fix + energy detector | out-of-band finding survives; **energy detector kills stealth on the lossless channel** |
| sim08 m1 | realistic TDL fading + AWGN | **sparse jammer imposes an SNR-independent BER floor**; stealth region reappears |
| sim08 m2 | channel-valid CNN + energy suite | **suite ≡ CNN** on the faded channel; residual stealthy region survives |
| sim08 dense | matched-detectability re-sweep | **refutes "channel-aware > blind"** — the gain was a matched-*config* artifact |
| **M0 / E1** | minimal model + NP-optimal detector | **no realizable attack achieves stealthy jamming** (§3.3) |
| **Lit check** | is E1's headline novel? (2026-09-12, desk work) | **no** — the impossibility half is known three times over (§2.1). **Kills stealth as the objective**; the learned-vs-NP gap survives |
| CGAN step 1 | reproduce Zhou 2025's CGAN jammer on a QPSK waveform link | **partial**: beats noise, but not Zhou's ordering once sync is consistent (§3.3b) |
| D0 | classical attacks × 4 detectors on a 3-D waveform link, SNR 30 dB | **the M0/E1 impossibility reproduced at waveform level**; uncoordinated jammers buy nothing (§3.3d) |
| D2a | learned shaped noise (48 params, CMA-ES) | **shaping buys effectiveness, not stealth** (§3.3e) |
| D1 / D2 | plain GAN vs white-box detector-aware GAN, 4 detectors, SNR 30 dB | **at matched BER, −19.8 pp P(det) vs the CNN at 12.8 dB less power**; nothing reaches the α budget (§3.3f) |
| **E2** | the noise ablation of D1/D2 over SNR 0–40 dB | **30 dB was near the worst place to measure it: the gain peaks at −85 pp at 15 dB**; detectors sharpen with SNR, the damage threshold does not (§3.3g) |
| E3 | K = 1/2/4 jammers under fading, aligned vs random timing (transfer) | **fading helps a single jammer, so a team has nothing to recover; the only team lever is timing** (§3.3h) |
| D4a | the delay-decay curve: follower timing error σ (transfer) | **coordination is worth ~1 symbol of timing accuracy**; the floor is reached by σ ≈ 8–16 (§3.3i) |
| D4b | a learned per-drone (power, transmit advance) policy, direct gradient | **learns the power lever (full power), not delay compensation**; loses to δ = τ at σ = 0, matches it for σ ≥ 1. Paused; E3/D4 all froze the under-trained generator (§3.3j) |
| GAN rework (run002–run005) | 4000 steps, random init, grey-box surrogate, on/off jamming, learned power | **the CNN-targeted generator is flagged on ~14 % of frames vs 100 % for its control at matched BER; energy detection holds every jammer to ≈ 1 broken frame per extra alarm; a trivial on/off jammer wins on average BER** (§3.3f/§3.3g) |
| S1 | per-frame shadowing on the victim's link, σ = 0–3 dB | **shadowing blinds a naive energy detector (lost by 0.1 dB, blind by 1 dB), not a gain-aware one; the CNN stays the weak link at every σ** (§3.3k) |

## 1.3 Where everything lives

```
BT/
├── Tabula Rasa/            <- this repo's working tree
│   ├── README.md           <- you are here; the only planning document
│   ├── CLAUDE.md           <- operating rules for Claude Code (commands, contracts, gotchas)
│   ├── m0/                 <- live code: the minimal model (coordination track, paused)
│   ├── cgan/               <- live code: the CGAN exploratory track (§2.10), being built
│   ├── final/              <- live code: the paper's final experiment (§3.4, §3.3q), copies of cgan/ code
│   ├── sim08_ablation/     <- live code: new sweeps over the frozen sim08 stack, imported read-only (§3.3c)
│   ├── source_papers/      <- PDFs the CGAN track reproduces/uses (IEEE-licensed; git-ignored)
│   ├── cluster/README.md   <- cluster ops; read before submitting anything
│   ├── artifacts/          <- all outputs, one dir per simulation; only m0, cgan, sim08, sim08_ablation
│   │                          are on the cluster — see the note below the tree
│   ├── graphify-out/       <- queryable knowledge graph of this repo (C.5); regenerable
│   ├── .claude/skills/     <- project-local Claude Code skills; so far only `no-ai-slop` (C.6)
│   ├── paper_drafts/       <- LaTeX sections drafted here, pasted into Overleaf by hand
│   ├── proposal/           <- registration proposal; `*_reviewed_2026-09-12.tex` = his annotated copy
│   ├── frontier/ simulation00..08/   <- FROZEN. Appendix material. Do not extend.
│   └── live.main.tex, first_results.py   <- scratch
└── paper/                  <- git subtree of the Overleaf document (see 1.4)
    ├── main.tex, refs.bib          <- Overleaf's; READ ONLY, never edit here
    ├── Literature_Review.md        <- Related Work prose, current framing
    ├── Sources_And_Evaluation.md   <- the reference database + refs.bib surgery plan
    └── Research_Landscape_2026.md  <- literature currency check, Aug 2026
```

> **Frozen-stack artifacts are off the cluster (2026-09-24, home quota).** `artifacts/{sim01, sim02,
> sim03, sim03b, sim03c, sim04, sim04b, sim05, sim06, sim06b, sim07, frontier, frontier_inband,
> frontier_recheck}` (187 files, 428 MB) were moved to **`BT/archive/`**: outside the repo tree,
> excluded from git by the local `.git/info/exclude`, with a `MANIFEST.txt`. The user copies it off
> and then deletes it from the cluster. That is not urgent: after the VS Code cleanup home is 4.9 G (§3.1).
> It also holds the pre-archive knowledge graph (C.5). Every artifact in it is also in
> git at `0fe1d6d`: `git checkout 0fe1d6d -- "Tabula Rasa/artifacts/sim04"` restores a directory.
> Appendix A and A.9 still cite these paths; they mean *in git*, not *on disk*. `sim08/` stays because
> `sim08_ablation/` loads its detector and `frontier_dense/results.json`.

> **If you meet a reference to a file that no longer exists**, it was folded into this README on
> 2026-09-10, when six overlapping documents were consolidated to remove ~2,400 lines of duplication:
> `SUPERVISOR_TODO.md` → [Appendix B](#appendix-b--supervisor-record) (its §§1–14 map onto B.2/B.3) ·
> `artifacts/RUNS.md` → [A.0](#a0-run-index) · `simulation03/README.md`, `simulation03c/README.md` →
> [A.2](#a2-sim0203c--what-a-policy-distribution-can-and-cannot-represent) ·
> `simulation03/USECLUSTER.md` → [C.4](#c4-cluster-quick-reference) · `paper/README.md` → §4.2 Q6 and
> `paper/Literature_Review.md`. All are recoverable from git history. Drafts in `paper_drafts/` and
> the Overleaf document may still cite the old section numbers.

**Live code (M0):**

| file | what |
|---|---|
| `m0/link.py` | constellation, channel, BER/SER, analytic references |
| `m0/attacks.py` | the 8-tier attacker ladder + hard power projection |
| `m0/detectors.py` | energy (1- and 2-sided), learned CNN on the IQ histogram, NP-optimal LRT |
| `m0/verify.py` | ~50 checks against analytic predictions — **run this first**, exit 0 = all pass |
| `m0/train_detector.py` + `submit_train.sh` | one CNN per σ (job 2243867, 89 s) |
| `m0/frontier_m0.py` + `submit_frontier.sh` | the E1 sweep, 8-task array (job 2243879, 8 s/task) |
| `m0/figures.py`, `m0/figure_geometry.py` | all figures → `artifacts/m0/*.png` |

**Live code (CGAN track, `cgan/`):** being built. The planned modules and their order are in
[§3.4](#34-the-plan-20-days--re-cut-2026-09-12-for-the-pivot) (items C0–C6); this table gets one row
per module as each lands. **Everything except `digitise_fig6.py` and `gan_figures.py` needs Sionna, so it runs via sbatch** (those two read JSONs/pixels only and run on the login node).

| file | what |
|---|---|
| `cgan/digitise_fig6.py` → `paper_fig6.json` | C0: Zhou Fig. 6 pixel-digitised, axes calibrated on its own gridlines (the only login-node-safe script) |
| `cgan/link.py` | Sionna QPSK waveform link (upsample → pulse → AWGN + jammer → matched filter → sign decision), closed-form Gaussian-jammer BER, `measure_ber` with an error stopping rule, paper-curve helpers |
| `cgan/jammers.py` | `noise` (full / inband), `optimal` (locked / random_phase / async), exact JSR scaling |
| `cgan/models.py`, `cgan/losses.py` | C3: generator + discriminator (score + auxiliary-classifier heads), `PeakScaler`; the seven loss terms |
| `cgan/verify.py` + `submit_verify.sh` | the test suite — **run first**; exit 0 = all pass (121 link checks + 43 baseline checks, `--baselines-only` to run just the latter; job 2266130) |
| `cgan/calibrate.py` + `submit_calibrate.sh` | C2: fit the unstated link parameters to Fig. 6 → `artifacts/cgan/calibration.json`, `c2_calibration.png` |
| `cgan/train_cgan.py` + `submit_train.sh` | C5: train the CGAN → `artifacts/cgan/runNNN_G.pt`, `_losses.{json,png}` |
| `cgan/evaluate.py` + `submit_eval.sh` | C6: BER vs JSR (noise/optimal/gan) overlaid on Fig. 6 → `artifacts/cgan/runNNN_ber_vs_jsr.{json,png}` |
| **`cgan/scene.py`** | baselines (§3.3d): 3-D drops, Sionna RT LOS path gains (Mitsuba **LLVM** variant — no OptiX on the cluster), `open_loop_uplink_power_control` to 30 dB SNR, received JSR; `load_test_drops()` caches `artifacts/cgan/baselines/test_drops.json` |
| **`cgan/channel.py`** | per-jammer async delay/phase via `cir_to_time_channel` + `ApplyTimeChannel`; `torch.complex64` only (half-precision `complex32` is wrong) |
| **`cgan/attacks.py`** | `noise` · `pulsed_qpsk(p)` · `omniscient(η)` · Li et al.'s 4 jammer types; `frames()` returns one received batch; hard power projection |
| **`cgan/detectors.py`** | power (1/2-sided), kurtosis, exact noise LRT, fixed-scale spectrogram + EfficientNet-B0; `calibrate`/`p_detect` ported from `m0/detectors.py` |
| **`cgan/train_spectrogram_cnn.py`** + submit | retrain Li et al. CNN + calibrate every detector → `artifacts/cgan/baselines/{detector_spec.pt, thresholds.json, clean_lrt_parts.pt, spec_cnn_training.json}` |
| **`cgan/baselines.py`** + `submit_baselines.sh` | the sweep: `--array=1` (K=1 universal curve + test drops), `--array=2-4` (K>1 over 50 drops), `--smoke`; → `artifacts/cgan/baselines/run001/sweep_K{1..4}.json` |
| **`cgan/baselines_figures.py`** + submit | 9 figures + `summary.json` from the sweeps |
| **`cgan/attacks.py`** (D1/D2: `gan_tx`, `gan` spec, `ber_logprob`/`log_expected_ber`) · **`detectors.py`** (`soft_pdet`) · **`models.py`** (`load_generator`) · **`verify.py`** §14 | the GAN eval path + the differentiable log-BER and soft-P(det) terms + the differentiability checkpoint |
| **`cgan/eval_gan.py`** + `submit_eval_gan.sh` / `submit_eval_gan_array.sh` | D1: place any generator (`--task N` or `--gen`) on the D0 BER–P(det) plane, with confirmation → `artifacts/cgan/gan/run001/<tag>.json` |
| **`cgan/train_gan.py`** + `submit_train_gan.sh` | D2: white-box detector-aware training, one generator per (target detector, β) → `artifacts/cgan/gan/run001/task{T}_G.pt` + `task{T}.json` |
| **`cgan/gan_figures.py`** | D1/D2 figures (login node; reads eval JSONs) → `artifacts/cgan/gan/run001/fig_*.png` |
| **`cgan/calibrate_snr.py`** | E2 (§3.3g): re-calibrate every detector at any SNR — re-fit the CNN's colour scale + all thresholds on clean frames at that level, **CNN weights frozen at 30 dB**; caches `artifacts/cgan/baselines/snr/<tag>/{thresholds.json, clean_lrt_parts.pt}`. `defender_at(L, snr)` returns the deployed defender unchanged at 30 dB |
| **`cgan/snr_ablation.py`** + `submit_snr_ablation.sh` | E2 driver: one SNR level per array task (`--array=0-9`, 0–40 dB in 5 dB steps + a noiseless anchor), classical envelope + all 21 generators, reusing `baselines.measure/confirm` and `eval_gan.sweep` → `artifacts/cgan/snr_ablation/run001/snr_<tag>.json`. **Task 6 = 30 dB is the regression check: it must reproduce §3.3f** |
| **`cgan/regress_snr30.py`** | E2 regression gate (login node): the 30 dB level vs the deployed §3.3d/§3.3f artifacts — **run before spending the array**. Uses a different noise model per quantity, which is load-bearing: BER must be compared in the ERROR COUNT capped at the frame count (bursty jammers concentrate errors in few frames, `verify.mc_tol`'s `n_eff`), never as a fixed log10 band, or the measurement floor is flagged as a regression |
| **`cgan/snr_examples.py`** + `submit_snr_examples.sh` | E2 example frames (Sionna, ~10 s): the spectrogram images the CNN receives at 0/15/30 dB and noiseless, clean vs D1 vs D2 at matched BER, with P(det) on fresh frames → `artifacts/cgan/snr_ablation/run001/examples.npz` (drawn by `snr_figures.py`) |
| **`cgan/snr_figures.py`** + `submit_snr_figures.sh` | E2 figures + summary table (login node; reads the ablation JSONs) → `artifacts/cgan/snr_ablation/run001/fig_*.png` |
| **`cgan/email_figures.py`** | jargon-free supervisor figures (login node; reuses `snr_figures`/`outlier_figures` loaders); `CGAN_RUN` picks the email. run001 (2026-09-24): the 15 dB CNN trade-off curve, and P(det) at matched BER + power vs SNR → `artifacts/cgan/snr_ablation/run001/email/`. **run003 (2026-09-28): CNN P(det) at matched BER vs the β = 0 control over SNR (no power panel, §2.7); broken frames per extra alarm, CNN vs one-sided power** → `artifacts/cgan/snr_ablation/run003/email/` |
| **`cgan/arms_eval.py`** + `submit_arms_eval.sh`, **`arms_figures.py`**, `submit_arms_cnn.sh` | D6 round 1 (§3.3o): `submit_arms_cnn.sh` retrains the CNN per defender (`train_spectrogram_cnn.py --jsr-range / --extra-gens / --snr-db`) → `artifacts/cgan/baselines/arms/snr<SNR>_<arm>/`; `arms_eval.py` scores each (SNR, defender) on paired frames — one card type, pinned `titan_rtx` — plus Li et al.'s own classes → `…/arms/eval/round1/snr<SNR>_<arm>.json`; `arms_figures.py [--figs]` (login node) prints the cost and gain tables → `fig_gain.png`, `fig_cost.png` |
| **`cgan/team_fading.py`**, **`team_policy.py`** + submits, **`team_figures.py`** | the multi-jammer track, **PAUSED 2026-09-26**. `team_fading`: E3 (K jammers under fading, §3.3h) and D4a (`--sigmas`, the delay-decay curve, §3.3i), by transfer. `team_policy`: D4b, the learned per-drone (u_k, δ_k) policy — `--mode train/eval/diag`, differentiable `frac_delay`, four evaluation arms (§3.3j). `team_figures` (login node): `--timing`, `--policy` tables. All three freeze `run001/task14_G.pt` |

**Live code (final experiment, `final/`, built 2026-09-30):** copies of `cgan/` code, never imports
(each copy's first line names its source at 68c97a4). Everything that imports Sionna runs via sbatch,
pinned `titan_rtx`; `figures.py` runs on the login node. Results: §3.3q.

| file | what |
|---|---|
| `final/env.py` | the six environments (σ_N, Rician K), SNR 15 dB = Es/N0 24 dB, N_SYM 128, α 0.05, artifact + net_scratch paths; `FINAL_SMOKE=1` redirects everything to `artifacts/final/_smoke/` |
| `final/link.py`, `channel.py` | the Sionna QPSK link (`cgan/link.py` pruned; `fading_gain` = per-frame Rician amplitude, unit mean power; `scale_to_jsr` moved in) and the async jammer channel (`cgan/channel.py`, two imports changed) |
| `final/attacks.py` | noise, pulsed (Amuru p ∈ {1…0.01}), omniscient genie, Li's tone / pulse comb, `gan_tx`, a simulated `onoff` (verify only), `frames` (+ fading, σ_N, per-frame DC share), the damage term with a per-frame noise variance |
| `final/detectors.py`, `defender.py` | `energy` (mean \|z_k\|² after the MF, one- and two-sided), `energy_csi`, kurtosis, the spectrogram CNN; full-band `power` for the regression gate only. `Defender`, `measure`, `calibrate_env` (the CFAR thresholds on 20k clean frames of the environment) |
| `final/models.py` | `Generator`, `load_generator` |
| `final/train_cnn.py` + `submit_train_cnn.sh` | chain step 1: Li's CNN retrained on the environment + every CFAR threshold → `artifacts/final/<env>/cnn/` |
| `final/bands.py` + `submit_bands.sh` | chain step 2: the JSR training band per target, S4's rule made automatic → `<env>/bands.json` (30 dB reference cached in `artifacts/final/bands_ref30.json`) |
| `final/train_gan.py` + `submit_train_gan.sh` | chain step 3: 8 generators per environment (control ×3, `cnn_b10` ×3, `energy_b10`, `kurtosis_b10`), run003's recipe → `<env>/gan/` (checkpoints on net_scratch, symlinked) |
| `final/evaluate.py` + `submit_evaluate.sh` | chain step 4: 22 attackers × JSR −50…+15 dB × 512 frames × every detector → `<env>/eval.json`. The random push (2026-09-30) is appended last (`added()`), so no earlier attacker's position seed moved; `--only random_push_e0.1` adds it to an existing `eval.json` |
| `final/submit_env.sh <env> [--smoke]` | submits steps 1–4 as one `--dependency=afterok` chain (login node, sbatch only) |
| `final/figures.py` | login node: `--env E` / `--all` → `fig1_vs_jsr` (excess BER, excess PER and P_det stacked on one JSR axis), `fig2_damage_vs_pdet`, `table.md`, `summary.json`; `--compare` → `artifacts/final/compare/` (P_det at matched damage across envs, the CNN-conditioned cGAN's gains over Amuru's best p and the vanilla cGAN, the DC share). The figures draw six attackers per detector column (`DRAWN`: genie flip, random push, barrage, Amuru at its best p, vanilla cGAN = control, the cGAN conditioned on that column's detector) over energy / kurtosis / CNN; tables hold every attacker and detector. The pre-2026-09-30 PNGs `fig1_damage_vs_jsr`, `fig2_pdet_vs_jsr`, `fig3_damage_vs_pdet` are left on disk, superseded |
| `final/verify.py` + `regress.py` + `submit_verify.sh` | the test suite (closed forms, energy, Rician, σ_N, CFAR in all six envs, bands at base, baselines, one training step per target) and the regression gate vs S4 → `artifacts/final/regress/s4_gate.json`. **Run first**; job 2277992 (97/97), rerun with the random push job 2278412 (100/100), both exit 0 |

**Live code (sim08 ablations, `sim08_ablation/`):** imports `simulation08/` and `simulation06/`
read-only; every entry point is an sbatch script (Sionna).

| file | what |
|---|---|
| `sim08_ablation/verify.py` + `submit_verify.sh` | 37 checks incl. re-measuring frozen job 102390's points — **run first** (job 2261123) |
| `sim08_ablation/ablation.py` + `submit_sweep.sh` | one noise level per array task: N_J × power × n_active sweep, then a confirmation pass on fresh frames → `artifacts/sim08_ablation/run001/sweep_*.json` |
| `sim08_ablation/figures.py` + `submit_figures.sh` | CPU post-processing → 5 figures + `summary.json` (`--smoke` runs on a partial sweep) |

**Frozen code** (sim06/07/08 + frontier) is inventoried in [A.9](#a9-frozen-code-inventory).

## 1.4 The paper — Overleaf is authoritative, this repo only reads

> **RULE: never write to the paper from this repo. Pull only.** The authoritative document is in
> **Overleaf**, which syncs to `github.com/rahul2605-ux/BT-Paper`, fetched here as the `overleaf`
> remote and merged into `paper/` as a git **subtree**. Edits happen in Overleaf, by the user.

The current Overleaf document **is the thesis**. Di Maio will open a separate one for the ICC paper
— his note at `overleaf/main` L463: *"VDN; QMIX are interesting info but not for paper, rather for
your thesis. To keep nice separation, I will create a new overleaf for paper and we keep this for
thesis."* Reinforced at L441 (*"do not throw away anything that I suggest not including in the
paper"*). This was read as meaning ICC's ~6-page limit does not constrain the appendix — **superseded
2026-09-15: the user writes the Experiment History to the 6-page conference budget** (§3.5); long
forms stay in [Appendix A](#appendix-a--experiment-history) for the thesis.

```bash
cd /home/rrahman/BT
git fetch overleaf                                # needs GitHub credentials
git show overleaf/main:main.tex                   # read straight out of the ref
git log -1 --format='%h %ad %s' overleaf/main     # what state am I looking at?
git subtree pull --prefix=paper overleaf main --squash   # only to version a snapshot
```

**Status as of 2026-09-10:** the subtree was pulled today, so `paper/main.tex` is **byte-identical to
`overleaf/main`** at commit `d662bdc` ("a4 draft done", Sep 10 13:50). *(This supersedes the earlier
rule that `paper/main.tex` was a stale 2026-08-17 snapshot never to be read — it was true until
today's pull.)* Re-verify with the `git show ... | diff` one-liner before trusting it, since Overleaf
moves independently and `paper/` only updates on an explicit pull. **The newest source on disk is the
user's hand-saved copy `paper_drafts/overleaf.tex` (2026-09-27 22:16)** — far newer than `overleaf/main`
here (2026-09-10), since `git fetch overleaf` needs credentials.

`BT-Paper` holds only `main.tex` + `refs.bib`. The `.md` research files in `paper/` are ours and are
not synced. **Auth:** `BT-Paper` is private, so `git fetch overleaf` prompts for credentials and
fails non-interactively. Never embed a token in the remote URL or paste one into a chat session.

## 1.5 Compute

Migrated 2026-09-08 from the INFK student cluster to **ITET/TIK**. **Read
[`cluster/README.md`](cluster/README.md) before submitting anything** — there are two CUDA traps that
silently break jobs. The five facts that change how experiments are designed:

- **The 1-GPU-job concurrency cap is gone.** Parallel sweeps and array jobs are now possible; the
  constraint that shaped every earlier experiment no longer applies.
- Walltime 1 h → **2 days**; 12 usable GPU nodes instead of 1; no preemption.
- Submit host `tik42x.ee.ethz.ch`, account **`disco-med`**. **Do not set `--partition`** (a lua
  plugin overrides it). `--gres=gpu:1`, never `--gpus=1`, plus the mandatory Pascal-exclusion
  `--constraint`.
- Env: `/itet-stor/rrahman/net_scratch/bt_env`. The home quota (soft limit **6.8 G**, 5-day grace) is
  invisible until you hit it; keep bulky things on net_scratch. It was hit 2026-09-23, mostly by
  stale VS Code server builds (`cluster/README.md`, Storage).
- Workflow unchanged: `cd` into a sim dir and `sbatch submit.sh`. Never compute on the login node.

All 19 submit scripts were migrated and verified end-to-end (job 2243247 reproduced the recorded m2
numbers in 2.9 s). **M0 runs on CPU in seconds**, so compute is not currently a constraint at all.

**Going back to INFK has been considered and rejected three times (2026-09-16, -09-20, -09-21); do not
re-litigate it without new evidence.** ITET wins on every axis in `cluster/README.md`'s comparison table
(unlimited concurrent jobs vs INFK's `MaxJobsPU=1`, 2-day walltime vs 1 h, 12 usable GPU nodes vs INFK's
single 5060 Ti), and the 2026-09-21 measurement settles it: over **107 jobs/array-tasks since 2026-09-14
the median queue wait is 0.1 min**, mean 4.5 min, worst-ever 43.6 min; 87/107 started within 5 minutes
(`sacct -S <date> -u rrahman --format=JobID,Submit,Start`). A migration would also cost a full venv
rebuild — INFK storage is not mounted on `tik42x` (`/work/scratch` does not exist there), so torch cu128
+ Sionna 2.0.1 + the `llvm_ad_mono_polarized` mitsuba variant + torchvision would all have to be
reinstalled and every artifact copied — against an **unverified Blackwell/sm_120** target (§3.3f).

**The methodological reason outranks all of that:** D2's rows are only meaningful next to the D0/D2a
rows they are compared with, and those were measured on ITET. Moving part of one study to a different
GPU architecture introduces an uncontrolled variable into exactly the comparison the study exists to
make (matched detectability, §2.8).

**2026-09-21 was a genuine outlier, not a trend:** one user held 17 of 23 running `disco.med` jobs with
2-day arrays, pushing our chain's estimated start ~9 h out. Note SLURM's estimate is a worst-case
ceiling computed from other jobs' *time limits*, not a prediction. The right response is
`--dependency=afterok:` chaining so the pipeline runs unattended, not a change of cluster.

---

# PART 2 — GOAL & APPROACH

## 2.1 The question, and what has been falsified

**STATE: the question changed on 2026-09-12.** Read this section before anything else in Part 2 —
several later sections still carry the old framing where it is still useful, and say so.

**The question, as of 2026-09-12:** **what does *coordination* buy an attacker that a single
jammer cannot get, and can a generative policy find it?** Detectability is how attackers are
compared (at equal exposure), not what they maximise.

**The question it replaces:** *can a learned jammer evade a SOTA learned detector while staying
effective?* That was the anchor from the proposal through E1. It is retired as a headline for the
reason below.

### Why stealth was retired as the objective

E1 (§3.3) measured that no realizable attack in M0 is both effective and undetectable. A literature
check on 2026-09-12 asked whether that finding was novel, and found that **the impossibility half is
established in at least three independent literatures**:

| Prior result | Where | What it already says |
|---|---|---|
| **Square-root law** — only O(√n) bits are transmissible covertly in n channel uses; positive *rate* ⇒ detection probability → 1 against an optimal warden | Bash, Goeckel & Towsley, **IEEE ISIT 2012**; extended IEEE JSAC 31(9), 2013 | A jammer is a transmitter. "You cannot inject meaningful per-symbol energy and stay under an optimal detector" is the SRL restated for a hostile one. E1 is its finite-blocklength QPSK instance. |
| **Symmetrizability** — if the jammer can make the channel symmetric, deterministic capacity is exactly zero | Csiszár & Narayan, 1988 (arbitrarily-varying channels) | The general form of E1's constellation-symmetry result. |
| **Disguised jamming** — a jammer that replicates the legitimate codebook is indistinguishable at the receiver; defence is to break the symmetry with shared secret randomness (SP-OFDM) | Tongtong Li's group — CDMA: **IEEE TIFS 2016**; OFDM: **IEEE TIFS 2020** | Our `counter_flip`/`permute` result, mechanism *and* genie requirement included, in the wireless setting. |

Two further collisions, recorded so they are not rediscovered:
- **G1's formulation is not ours.** `max damage s.t. D(p₁‖p₀) ≤ δ` against an NP/χ² detector, with
  Chernoff–Stein converting the divergence into a detection bound, is the standard **stealthy
  false-data-injection** program in the cyber-physical-systems literature, roughly a decade old.
  G1 instantiates a known program; it does not introduce one.
- **The effectiveness half of the ladder is published.** Amuru & Buehrer, *"Optimal jamming
  strategies in digital communications — impact of modulation"*, **IEEE GLOBECOM 2014**, extended in
  **IEEE TIFS 10(10), 2015** — the optimal jamming waveform against digital AM-PM constellations,
  without the detectability axis. That is our closed-form boundary attack.

**The inference that follows, and it is the reason for the pivot.** The near-absence of stealthy
jamming from the jamming literature is not an oversight left lying around — it is a **rational
response to a known limit.** The SRL says covertness costs effectiveness catastrophically, so an
attacker who does not need to hide simply does not pay that cost, and gets far better results. A gap
that exists *because the thing is known not to be worth doing* cannot carry an Introduction.

**What survived the check** (searched for, not found — so still claimable, with the caveat below):
1. **The learned-vs-NP gap as a measured quantity.** The covert-comms and CPS literatures assume an
   optimal or fixed analytic adversary. "How far short of optimal does a deployed CNN detector fall,
   and how does that gap move with attack sparsity?" appears unmeasured. §3.3's 0.73 → 0.16
   duty-dependence is the number.
2. **The one-sided energy detector's structural blind spot to power-*reducing* attacks**, shown
   against the optimal test as reference (P(det) = 0.0000 while causing BER 0.25).
3. **A BER–P(det) frontier across attacker-knowledge tiers on one model, at matched detectability.**
4. **The matched-detectability methodological correction** (§2.7) — retracting our own +70% because
   the comparison held the wrong variable fixed.

> **Caveat on the negative half of that list.** These were web searches on 2026-09-12, not a
> systematic IEEE Xplore sweep. Treat "not found" as strong evidence, not proof of absence. **Before
> any of items 1–4 is claimed as novel in print, read Li et al. (TIFS 2016, 2020) and Bash et al.
> (JSAC 2013) in full** — the overlap with item 3 in particular has not been checked at the level of
> individual claims.

### Sharpened 2026-09-16 — stealth retired *entirely*, not just as headline (pending sign-off)

A session of experiments + literature search extended the retirement to the CGAN step-2 stealth
conditioning (§2.10). Three arguments, none model-specific, so they hold regardless of what a
generator or MARL policy learns:
- **Channel knowledge ≠ symbol knowledge.** Even a jammer with perfect CSI/geometry (monitored over
  days, incl. other jammers' positions for MARL) still cannot know the *current* scrambled symbol
  before it must transmit, so its signal is symbol-independent and the received law changes ⇒ the
  square-root bound survives. CSI buys **efficiency and coordination, not covertness.**
- **The realistic warden is co-located with the victim.** Jamming detection happens *at the receiver*
  (Zhang & Krunz; Spuhler et al., TWC 2014); the spatially-separate "Willie" is a covert-*comms*
  abstraction (Bash et al.), not the jamming-detection model. So multi-jammer coherent combining is an
  **effectiveness** gain (more BER per total transmit power), **not** a stealth gain — the stealth
  version would need a separate warden, which is not standard. *(User confirmed co-located as the only
  realistic model, 2026-09-16.)*
- **FEC + interleaving largely nulls covert jamming.** Covert ⇒ raw excess BER = O(√n)/n → 0, below any
  fixed code's correcting radius; the only escape (bursting past per-codeword capacity) is exactly the
  most detectable structure and is de-concentrated by interleaving (classic partial-band-jamming
  result). So the covert-*optimal* jammer is non-covert — which makes FEC-out (§2.9) a **load-bearing**
  assumption for any raw-BER stealth claim.

**Consequence:** the effective attacks are the loud ones; optimising for stealth just finds weak/trivial
solutions. This is *proposed*, not settled — folded in only after the supervisor replies (§4.1 #0b).
The user's note: this "should have been trivial" but only became clear through the experiments, so it
stays out of the paper as a discussion and only scopes the threat model.

### Corroboration from the supervisor, same day, independently

His proposal feedback arrived **2026-09-12**, the same day as the literature check and with no
knowledge of it ([B.1](#b1-correspondence-log), [B.2](#b2-his-verbatim-points-and-what-each-changed)).
It points the same way, and sharpens the destination:

- He flags the contribution as reading like *"the application of some neural architecture"* — §4.1
  item 5, in his words — and names the escape: **novelty in "how defenders and attackers interact,
  coordinate, and syncronize within themselves"**, not in the architecture.
- He supplies the mechanism that makes coordination hard and therefore interesting: **inter-jammer
  communication delay, hence desynchronisation**, which *"makes it hard for jammers to react to
  honest symbol"* — and, symmetrically, hard for the defender.

**Consequence, and it is a real sharpening rather than a restatement.** RQ1 as first written after
the pivot asked whether coordination beats independence. That is a thin question: with a shared
clock and zero delay, coordination trivially wins, and the answer is a number nobody will argue
about. **The question his two comments define is the good one — how the coordination gain decays as
the inter-jammer link degrades** — because it has a floor (independent jammers), a ceiling (perfect
coordination, which sim04 already reached), and a realistic middle nobody has measured. It is also
the axis on which he has *twice* asked for the attacker to be handicapped (§B.2, desync).

### What the pivot does and does not touch

- **Nothing about M0 is discarded.** `link.py`, `attacks.py`, `detectors.py`, `verify.py` and E1 all
  stand; the numbers are correct and reproducible. What changes is the *sentence they support*.
- **`P_NP` keeps its job**, in a smaller role: it is the instrument that makes "at equal exposure"
  well-defined when comparing 1 jammer against N (§2.6).
- **The falsified methods stay falsified.** The pivot is *back to (a) cooperation*, **not** back to
  PPO-over-raw-IQ. Actions remain low-dimensional perturbation parameters; the training method
  remains direct/surrogate gradient (sim03b/sim04), which is the one thing on the ladder that worked.
  See §3.6.

**Scope caveat that still stands:** any detector-facing result here is **single-round** against a
**frozen** detector — an evasion result, not an adaptive arms race.

## 2.2 The motivating argument — the UAV detection gap ⚠ SUPERSEDED as the Introduction

> **STATE (2026-09-12): this argument is NO LONGER what the Introduction argues.** Its load-bearing
> step is **point 5**, and the literature check (§2.1) falsified it: the attack side has not
> optimised non-detectability because the **square-root law says it is not worth optimising**, not
> because nobody thought of it. An Introduction built on point 5 walks into a citation the reviewer
> already knows.
>
> **Kept, in full, for two reasons.** (i) Points 1–4 are still true and still useful — they are a
> correct description of how reactive defenses are structured, and are reusable as *context* in the
> new Introduction, just not as the gap. (ii) The proposal handed to Di Maio on 2026-09-01 argues
> exactly this, so when his feedback arrives it will be feedback on *this* text; it has to be
> readable. **Do not delete it, and do not build on point 5.**
>
> **A second, independent reason it fails, from his 2026-09-12 feedback** (§B.2): the "defenders are
> merely reactive" criticism in points 2–4 cuts both ways, and he said so — *"doesn't the proposed
> method also use some form of prediction of activity? i.e., jammer predicts distribution of honest
> codewords, and defender predicts jammer presence?"* Our attacker predicts too. So even setting the
> square-root law aside, the rhetorical structure of points 2–5 was unsound. **Two independent
> failures, found the same day, by different routes.**
>
> The replacement motivation is §2.3's RQ1: coordination has no closed form, and that is a gap of a
> different kind — an unsolved optimisation, not an unnoticed opportunity. **His own framing
> question — does this motivate a *proactive* or an *adaptive* defender? — is the right way to close
> the Introduction**, and it is answerable rather than rhetorical (§B.2).

Settled 2026-09-01 while drafting the registration proposal. This is what the Introduction argued
until 2026-09-12.

1. UAV / mobile ad-hoc anti-jamming has converged on learned, often multi-agent countermeasures —
   relay repositioning, coordinated spectrum access, trajectory + power adaptation.
2. Every one of them is **reactive**: gated on a detector output, a jammer classification, or an
   observed link collapse serving the same role.
3. That trigger is trusted because detection is reported as solved (>99% spectrogram CNNs, in exactly
   the OFDM/UAV setting these defenses assume).
4. **Therefore the stack has a single point of failure**: a countermeasure triggered by detection
   cannot be triggered by an attack it never detects. The defense does not degrade gracefully — it
   never activates, and the degradation is experienced as ordinary channel impairment.
5. **The gap:** the attack side has not optimized non-detectability as a first-class objective (low
   power / low duty cycle is pursued as *efficiency*, with stealth a by-product), and the defense
   side has assumed it would not have to.
6. **Hence the method.** *Multi-agent* because keeping each jammer below threshold while their
   contributions add at the victim has no closed form. *Generative* because the signature must be
   **shaped**, not a channel merely chosen.

> **Honesty constraint on claim 5 — this is where it broke.** The 2026-09-01 version of this note
> already flagged that "nobody studies stealthy jamming" is false and easy to attack, and proposed
> defending the **conjunction** — explicit stealth objective **and** a learned detector **and** a
> trade-off curve. The 2026-09-12 check (§2.1) shows that defence is not enough: the conjunction is
> narrower than the prior work, but the *reason* the conjunction is unoccupied is the square-root
> law, and pointing at an unoccupied cell does not answer "why is it empty?". **The nearest
> neighbours are no longer `wen2025generative` and `valianti2024cooperative` but `bash2013limits`,
> Csiszár–Narayan and the disguised-jamming line — and those are not neighbours, they are the result.**

The characterization work (Phase 0/0.5, m1, m2, matched-detectability) was an *instrument* of this
argument. Under the new direction it is appendix material only (§3.5) — it characterises a detector
for a question no longer being asked.

## 2.3 Research questions

**STATE: revised 2026-09-12.** The registered wording is kept below each revision, because
registration (§4.1) was submitted against it and the proposal Di Maio is reviewing uses it.

- **RQ1 — the coordination gain, and what it costs to actually have it.** What does a *coordinated*
  team of N_J generative jammers achieve that N_J independent ones, and a single jammer at the same
  **total** power, do not — **and how does that gain decay as the inter-jammer link degrades?**
  Measured **relative to baselines** (barrage / closed-form minimum-energy / single-agent learned /
  omniscient cancellation as ceiling) and reported **at matched detectability** as well as matched
  power.
  **The second clause is the contribution, and it is his** (§B.2, 2026-09-12): coordination with a
  perfect shared clock is not a research question, because sim04 already reached that ceiling and
  nobody disputes it. Coordination over a link with **delay, and the desynchronisation it causes**,
  has a floor, a ceiling and an unmeasured middle. It is also his answer to "the contribution looks
  like applying a neural architecture" — the novelty lives in *how the agents coordinate and
  synchronise*, not in the network.
  **Deliverable shape: a decay curve** — coordination gain against inter-jammer delay / phase error,
  with the independent-jammer floor and the perfect-coordination ceiling drawn in. Same envelope
  convention as every other figure (§2.7).
  *Registered wording:* "can a cooperative multi-agent generative policy degrade the link while
  staying below a SOTA learned detector's threshold, and what trade-off does it achieve relative to
  baselines?" — the baseline envelope is unchanged; what changed is that the *objective* is the
  coordination gain and detectability is the axis it is reported against (§2.1).
  **Why this is a real question and stealth was not:** the single-link minimum-energy attack has a
  closed form (Amuru & Buehrer, §2.1) and E1 confirms it is at the limit. **N_J jammers splitting
  power and phase so their perturbations add at the victim has no closed form** — it is a genuine
  joint optimisation, which is exactly what a learner is for (§4.2 Q2 items 3–4).
- **RQ2 — adaptation cost**, round-based and offline: frozen detector → attacker optimized
  against it → detector retrained → attacker re-optimized. What does re-closing the gap cost the
  defender, and how much does the attacker recover?
  **Status after the pivot: DEMOTED but not dropped.** It is the strongest surviving detector-facing
  claim (§2.1 item 1) and it is Di Maio's mandated headline (§2.9), so it cannot simply be cut —
  but it is no longer what RQ1 serves, and G7 sits behind G5/G6 in the plan (§3.4). **This tension
  is real and he has to resolve it** (§4.1 #0).
  **What counts as "expensive" — still to be pinned down.** Candidates, all measurable with existing
  tooling: Δaccuracy · ΔFAR · training samples required · GPU-hours · how much performance the other
  side recovers · and, uniquely available in M0, **distance from the NP-optimal detector**, i.e. how
  much adaptation budget is even left. The deliverable is a **cost curve per round**, not a win/loss.

**A third RQ was drafted and DROPPED: countermeasure-level evaluation** ("how much of the reactive
stack still functions"). It needs a full detect→react pipeline *plus* a proactive counterpart *plus*
a network-throughput model — a second thesis, and the comparison would mostly measure our own two
countermeasure implementations rather than the attack. The intro argument survives as **motivation**;
the consequence for reactive defenses is stated as an argument, not a measured result. **Methodology
must carry a scope sentence saying so.**

**Working title** (option A of five in `proposal/proposal.tex`'s header comment): *Cooperative
Multi-Agent Generative Jamming of UAV Networks under Detection Constraints*. Chosen because it names
method, target and constraint without committing to a result. **After the 2026-09-12 pivot this
title has aged well** — "cooperative multi-agent generative jamming" is now literally the subject,
and "under detection constraints" reads correctly as the evaluation axis. Keep it. *(The sharper
"Breaking the Detection Assumption: …" alternative is now actively wrong — it promises the stealth
finding that §2.1 retired. Do not revive it.)*

## 2.4 The model — M0 and M1

Per the 2026-08-21 mandate. **M0** = the simplified base (built). **M1** = M0 + spatial, the *only*
sanctioned extension. Naming is ours; he did not name them.

> **M1 gained a required component on 2026-09-12** (§B.2): the **inter-jammer link is explicit, and
> it has delay.** It is not a modelling nicety — under §2.3's revised RQ1 the delay is the swept
> variable, so M1 without it cannot answer the research question. Minimum viable form: a per-jammer
> timing/phase offset, plus a one-parameter staleness on whatever each jammer knows about its
> partners. His single-channel confirmation in the same message means **nothing else** gets added.
>
> He also volunteered that a **journal extension (TWC) is the path for whatever the 6-page ICC limit
> forces out** — so multi-carrier, fading and mobility are *deferred*, not cut, and the layer
> ordering in `proposal/proposal.tex`'s "incremental system model" stays as written.

| | **M0 — built** | **M1 — the one sanctioned extension** | *(sim06–08, frozen)* |
|---|---|---|---|
| Subcarriers | **1** — no OFDM grid, no IDFT, no guard/DC/pilot bins | 1 | 64-SC OFDM |
| Channel | **one** fixed realization, `h = 1` | per-jammer link gain/phase, superposing at RX | TDL freq-selective, per-link, per-frame |
| Propagation | **none** — no delay, no path loss (literal) | none; geometry enters only as per-link gain/phase | — |
| Noise | **AWGN, σ swept** | single global σ, same sweep | per-SNR Eb/N0 5–30 dB |
| Jammers | 1 | **N_J ≥ 2, coordinated** — the point of M1 | 1 (frontier) / MAPPO team (dead) |
| Inter-jammer link | — | **explicit, with delay** — the swept variable of RQ1 (added 2026-09-12) | — |
| Legit users | 1 TX → 1 RX | 1, then sweep | 1 |
| Attacker action | per-symbol complex perturbation, hard power budget | + the split between jammers | full per-subcarrier IQ (falsified) |
| Detector | energy meter + learned CNN on the IQ histogram + **the NP-optimal test** | same | spectrogram CNN + energy |
| Metrics | **BER, SER**, P(detect) at fixed FAR | + coordination gain over N_J independent jammers | BER, P(suite) |

**Why single-subcarrier is the right cut, specifically.** It deletes exactly the things that burned
the project: the guard/DC/out-of-band bins that produced *and then invalidated* the Phase-0 headline;
the equalization/CSI bookkeeping that made m1's "channel-aware" criterion ambiguous
(`|h_jam/h_tx|` vs small `|h_tx|`); and the spectrogram representation, which took a real-vs-complex
STFT bug to get right. What remains is a 2-D constellation — **the picture the supervisor reasons in
every time** ("add vector in random direction in I/Q plot", "points located around the symbol
classification boundary").

**Consequence to state explicitly: with σ = 0 there is no stealth problem.** The clean received
constellation is four exact points, so *any* perturbation is detected with probability 1. **Stealth
exists only because of noise.** That is why the noise sweep is the primary ablation, and it is the
M0-level explanation of the sim08-m1 result (the stealth region appeared only once the channel had a
noise floor to hide under). Getting that mechanism into a model simple enough to *derive* is a
genuine contribution, not a retreat.

## 2.5 The attacker ladder and the baseline envelope

`m0/attacks.py` implements eight tiers: `none, barrage, gaussian, boundary_blind, boundary_genie,
permute, counter_null, counter_flip`, all with hard power projection.

Every results figure carries the same envelope, so the reader always sees where a curve sits between
floor and ceiling:

| Baseline | Attacker knowledge | Role |
|---|---|---|
| **No attacker** | — | clean BER/SER floor **and** the detector's FAR — the lower reference on *both* axes |
| **Random-direction I/Q vector** (barrage) | none | naive floor; what the boundary attack must beat |
| **Boundary / min-energy** | symbol + channel | closed form, **no training** — the real bar for any learner |
| **Counter signal** (`−H₀X`, `−2H₀X`) | symbol + channel + perfect sync | **"impossible to beat" ceiling** — he asked for it by name |
| **Learned / coordinated** | per the assumption tier | the proposed method |

> **Do not forget the visual he will look for:** re-plot the **IQ scatter under attack** and check
> that the perturbed points now cluster *at the decision boundary* rather than scattering
> isotropically. That picture is what his whole minimum-energy observation was about, and it is the
> fastest way for him to see that it was implemented.

**Closed-form frontiers (σ = 0).** A QPSK symbol sits at distance `1/√2` from its nearest decision
boundary, so a push of `ρ > 1/√2` (energy 0.5) flips exactly one bit. Spending budget `P` on a
fraction `duty` of symbols gives `ρ = √(P/duty)`, which crosses only while `duty < 2P`:
- **genie** (knows `s`, pushes toward the boundary): `BER*(P) = min(P, 0.5)`
- **blind** (pushes along a random axis direction, independent of `s`): `BER*(P) = min(P/2, 0.25)`
- The ratio is **exactly 2.00** at every power — that is the entire value of knowing the symbol, and
  it is a **sensing** capability, not a learnable one. For an iid scrambled payload a causal jammer
  can never have it, so no amount of learning closes that factor of two.

**The counter signal must be both motivated away and run.** It needs per-symbol knowledge of X,
exact amplitude and phase of both channel responses, and sample-level synchronization — the jammer
must already be a perfect receiver *and* be phase-locked to the victim. It also degrades
**ungracefully**: cancellation is an exact-inverse operation, so residual CFO/timing/phase error
destroys it. The boundary attack needs the same symbol knowledge but only has to land the received
point **in the wrong half-plane**, so it tolerates phase error far better. That contrast is a
*testable claim* and it is what the desync axis is for — hence the ordering: **boundary attack
first, desync second.**

## 2.6 The defender, and why the NP-optimal test is the whole point of M0

Three detectors, all in `m0/detectors.py`, all calibrated to a target false-alarm rate α:

1. **Energy meter** — one-sided and two-sided variants. The classical one-sided test exists because
   it was designed against barrage jammers that ADD power; minimum-energy attacks push symbols
   *toward* the origin, so received power FALLS and the one-sided test is blind to them. Both are
   implemented; **report both.**
2. **Learned CNN** on the received IQ 2-D histogram — the M0-scale stand-in for the spectrogram CNN.
3. **The Neyman–Pearson likelihood-ratio test** — available in closed form once σ and a perturbation
   model are stated.

**Why (3) justifies the entire retreat to M0.** Using an LRT as the optimality benchmark that
practical detectors are measured against is textbook spectrum-sensing methodology (Kay, *Detection
Theory*, 1998; Axell–Leus–Larsson–Poor, IEEE SPM 2012, which sets up exactly our LRT / energy / GLRT
tiering). What is new here is the **direction**: that literature uses the bound to certify a
*detector*; we use it to certify an *attack* — "this evades the optimal test" ⇒ **no detector catches
it**. Say this explicitly in the Defender Model. It upgrades every claim of the form *"the CNN is
blind to X"* into *"**no** detector can do better than Y"*, and it gives the adaptation-cost headline
the reference point it otherwise lacks: how far a retrained detector still is from optimal **is** the
remaining adaptation budget.

Sanity ordering that must hold everywhere: `P_NP ≥ P_learned ≥ P_energy`. An "optimal" detector
losing to a CNN would mean the maths was wrong. `m0/verify.py` checks it.

## 2.7 Metrics and figure conventions

- **BER and SER** (he asked for SER by name), P(detect) at a fixed FAR. In `cgan/` SER is stored per
  sweep point and is a fixed transform of BER except for jammers that hit I and Q together (§3.3g
  finding 8) — report it as a column. PER is stored at every sweep point since S1 (§3.3k); the
  threshold-free warden error ξ (below) needs stored statistic quantiles; SINR/JNR axes are parked (§4.3).
- **The FAR is the detector's operating point, not a filter on the results** (user, 2026-09-24:
  *"we just want to study the trade-off of how much damage per Pdet, so it makes sense to show
  everything, not just < FAR"*). Every P(det) is read at one threshold, calibrated to the detector's
  own clean false-alarm rate α so detectors are comparable, and α is drawn as the dotted floor (P(det)
  below it is noise). The object of study is the **whole damage-vs-P(det) curve**; a "stealthy BER at
  α" number is at most its left edge, never a result of its own. *This bullet used to read "the stealth
  budget is the detector's own clean FAR". That was written while stealth was the attacker's objective,
  after BER at `P(det) ≤ 0.5` had been passed off as stealthy — a jammer caught in half of all frames
  is caught within a few frames. The argument still holds for any claim of covert operation, which the
  project stopped making on 2026-09-16 (§2.1).*
- **Compare at matched detectability, not matched configuration. Non-negotiable.** This is the one
  trap the project has already fallen into: sim08-m1's "+70% channel-aware" evaporated entirely once
  BER was compared at matched P(detect) instead of matched jammer *config*. Report both ways; if a
  claim survives only the matched-power view, it is the m1 mistake repeated.
- **Figure format he asked for specifically:** one panel, **two y-axes** — BER (and SER) left,
  P(detect) right, against the swept parameter (σ, or power budget) — with the **no-attacker** and
  **omniscient-attacker** references drawn in. Keep the parametric **BER-vs-P(det) frontier** plot as
  the companion: the dual-axis view is what he wants to read, the frontier view is what supports
  matched-detectability comparisons. **Produce both, for the same runs.**
- **The frontier is drawn over every budget, not read at α alone** (user, 2026-09-19). For each
  attack, plot the most BER it reaches while flagged in at most x of frames, x = 0..1, with the FAR
  as a dotted line and the raw sweep points faint underneath. Quote matched BER at several budgets
  (0.05 / 0.1 / 0.25 / 0.5), not only the stealthy value at α. Attacks that are all at BER 0 under
  α are still ordered by where they start to hurt. Implemented in `cgan/shaped_figures.py`
  `fig_frontier` (§3.3e) and `cgan/gan_figures.py` `fig_frontier` (§3.3f). **The FAR stays fixed while
  x sweeps**: a figure whose budget axis also moves the threshold — E2's `fig_frontier_by_snr`
  (§3.3g) — is the ≤ FAR filter in disguise and is to be replaced (§4.3).
- **The stealthy region is one end of the curve, not the evaluation** (user, 2026-09-28, restated).
  Draw the frontier on a **linear** BER axis 0–1 as well as log (the genie reaches BER 1.0 by inverting
  every symbol, at the FAR; 0.5 = random guessing is drawn as a reference), and report **damage at
  P(det) ≤ 0.5** (BER and PER) as a standing read-out next to P(det) at matched damage.
  `cgan/shadow_figures.py --linear σ` → `artifacts/cgan/snr_ablation/shadow_run003/
  fig_frontier_linear_<σ>.png` + the P(det) ≤ 0.5 table (§3.3f, "linear view").
- **Damage for detection is the main metric; power is secondary** (user, 2026-09-27). Energy was
  motivation, not the objective: the jammer chooses its power (the envelope over the JSR sweep, or a
  learned power), and figures read damage against P(det) with power as a hidden parameter. Report BOTH
  bit damage (average BER) and frame damage (PER): they rank jammers differently (§3.3g E2b). **Any
  stealth claim must beat the on/off baseline** — a loud jammer used on a fraction of frames — which
  wins on average BER. Against the one-sided energy detector the alarms fall mostly on the frames where
  the jammer FAILED: a push that flips a bit opposes the symbol and lowers the frame's energy, one that
  does not adds energy (§3.3k). So the damaged frames themselves largely go unflagged (the S3 question,
  §3.4).
- **Only literature-grounded measures (user, 2026-09-29).** What the jamming and covert-communication
  literature reports, and nothing invented: (1) the **damage-detection trade-off** — PER / BER vs
  per-frame P(det) at α, one point per JSR (stealthy-jamming papers report damage and detection this way);
  (2) **P(det) at matched damage** — at excess PER 0.1 / 0.5 and excess BER 3e-4 — the main scalar;
  (3) the warden's **detection error ξ = min_t (P_FA + P_MD)**, threshold-free (covert communication,
  Bash et al. JSAC 2013), wherever a sweep stored statistic quantiles; (4) **detection delay** — frames a
  sequential test (Wald SPRT, 1 %/1 %) on the per-frame alarms needs, times PER = frames broken before
  it decides. (4) is our combination of standard parts and only re-expresses (2); say so where it is
  used. Code: `cgan/probe_readout.py` (`xi_auc`, `sprt_frames`, `matched_per`), `cgan/probe_report.py`.
  **Withdrawn: "damage per extra alarm"** (damage / (P(det) − FAR), best over JSR; E2b's own unit, in no
  paper). Its best value is usually the saturated corner 1/(1 − FAR) = 1.05, which hid S4's effect
  (§3.3m), and it is a maximum over noisy ratios (S2's first seed: 14.4, not replicated, §3.3l). E2,
  E2b, §3.3f and the older numbers in §3.1 still quote it; re-measuring them is on the list (§4.3).
  **The final experiment (§3.4) reports only (1) and (2);** ξ and the delay are dropped there (user,
  2026-09-29).
- **P_det — the definition (settled 2026-09-29; answers the supervisor's "P_det not specified").**
  - **Statistic and threshold.** Per frame (N = 128 symbols), each detector computes a statistic T,
    larger = more suspicious. Its threshold τ_α is the (1 − α)-quantile of T over 20 000 clean frames.
    α = 0.05, the same for every detector.
  - **P_det.** P_det(JSR) = Pr[T(r) > τ_α | jammer present at received JSR], estimated on 512 frames
    per JSR point. Both probabilities run over everything random in the environment: bits, noise, the
    jammer's latent input, delay and phase, and the channel draws.
  - **CFAR rule.** The clean frames that set τ_α come from the *same environment* the detector is
    evaluated in, including every variation the defender cannot know (noise level, fading). The
    false-alarm rate therefore stays α, and the threshold rises instead. That loss is the realistic
    cost of not knowing. A detector calibrated on an idealised environment (S5's "naive") is never
    reported as a P_det; at most its realised FAR is quoted.
  - **Silent frames.** Frames on which an on/off jammer is silent count, for P_det and for damage alike.
  - **Per frame.** P_det is not "the jammer is ever caught", which grows with the number of frames.
  - **Damage** (excess BER / SER / PER over the same environment's clean frames) is measured on the
    same frames.
  - **P_det at matched damage:** find the JSR where the jammer reaches the damage level (damage vs JSR),
    and read P_det there (P_det vs JSR).
  - **Paper form:** P_det = Pr[T > τ | H1(JSR)], with τ such that Pr[T > τ | H0] = α. The soft P_det used
    in training (`detectors.soft_pdet`) is a Methodology surrogate, not the reported quantity.
- **SNR in the paper is Es/N0** (2026-09-29). `cgan/`'s SNR is power per sample over the 8× simulated
  band, so **Es/N0 = SNR + 9.03 dB**: our 15 dB is Es/N0 24 dB and our 30 dB is 39 dB. README numbers
  keep the code's SNR unless marked. Any literature SNR has to be converted before it is compared.
- **The final experiment's figure set is frozen** (user, 2026-09-29: fix it once instead of
  re-inventing measures). Per environment and detector:
  - (1) damage vs JSR: BER with SER dashed, and PER on a second panel;
  - (2) P_det vs JSR, α dotted;
  - (3) damage vs P_det, the same points with JSR hidden. This is where equal damage is compared;
    comparing at equal JSR is the matched-config trap above;
  - (4) one table: P_det at excess PER 0.1 and excess BER 3e-4.

  The supervisor's dual-axis request is (1) + (2).

## 2.8 Settled method decisions

Do not re-litigate these; they are decided. Reasoning kept because it gets revisited when his
corrections come back.

> **2026-09-14:** the exploratory CGAN track reopens the "NOT a GAN" and "never raw IQ" rows **outside
> M0 only**, with the reason for each — see [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).
> Inside M0 both rows stand.

| Decision | Why |
|---|---|
| **Reward = `BER − β·detections`. Nothing else.** | His "most agnostic reward". Every proxy term (idle penalty, power penalty, kurtosis penalty) from sim01–04 is **deleted**. **Re-confirmed by the user 2026-09-28:** detection stays a *penalty* in training and P(det) stays primarily an evaluation metric; a fixed-detectability constraint (set the power each step so soft P(det) equals a budget, maximise damage) was proposed and **declined** — "the jammer does as much damage as it can and is punished for detections". |
| **Power budget is a hard environment/action-space constraint**, not a reward term | His instruction, and sim04-run007 is the concrete proof: `GAMMA = 0.02` was negligible against BER gains, so nothing constrained power and it climbed monotonically to 4.0. |
| **Conditional generator + direct gradient. NOT a GAN.** | A GAN discriminator is a *density-ratio estimator* — it exists for the **likelihood-free** case. In M0 the ratio is **closed form**, so an adversarially trained discriminator would spend its budget approximating a function we can already write down. Use a reparameterised `G_θ(z; c) → d` + hard power projection, trained by direct gradient on the exact objective. That is sim03b's method — the one thing on the ladder that worked — and it drops GAN instability, mode collapse and discriminator scheduling from the risk list. `zhou2025cgan` stays in Related Work as the nearest neighbour, not as the method. |
| **The optimality gate is a *divergence*-constrained convex program** | `max BER s.t. E|d|² ≤ P, P_det^NP ≤ β` is **not convex** — the optimal test depends on π, so the constraint moves as the variable moves. Replace the detection constraint with `D(p₁‖p₀) ≤ δ` (or TV): BER is linear in π, power is linear in π, and the divergence is convex in `p₁` which is linear in π ⇒ a genuine convex program on a discretised `d`-grid. **Pinsker's inequality converts δ into a bound on *every* detector's error probability**, so the answer is detector-free and therefore a true ceiling. This is exactly the covert-communication formulation (`bash2013limits`), already cited for the stealth-budget convention — method and citation line up. **⚠ 2026-09-12: this program is prior art, not ours** — it is the standard stealthy-FDI formulation in the cyber-physical-systems literature, Chernoff–Stein included (§2.1). Still worth running as G1, but write it up as *instantiating* a known program. |
| **CTDE: the jammer is deaf to its own reward at execution** | BER is a **training-time** construct, available to the centralized critic only. A deployed jammer cannot measure the victim's BER. The executed policy observes its own waveform, its own channel estimate, and *at best* a 1-bit delayed noisy **ACK/NACK**. Not BER, not P(detect). This is the attacker-side mirror of his defender-side "no ground-truth labels at execution time", and it must be labelled as such in the threat model. |
| **PettingZoo for the multi-agent env API; BenchMARL only if an off-the-shelf MARL algorithm is genuinely needed; SB3 for single-agent baselines over the same env. Never RLlib.** | His words: *"RLlib is famous for being too complex for what we need, so I would avoid it."* Confirmed absent from the repo. Surrogate gradients stay the **primary** method; MARL is the comparison, not the default. |
| **Actions are low-dimensional perturbation *parameters*, never raw IQ** | This is what killed sim06/07. |
| **The detector is frozen; the arms race is round-based and offline** | His: *"this can only happen at training time: there are no ground-truth labels at execution time"*. |
| **Report the measured gain; do not inflate a bounded result into an impossibility claim** (user, 2026-09-23) | A small measured improvement *is* a finding. At the α budget every jammer reads 0, so that cut cannot distinguish anything and must not be the headline; the reportable claim is that **at matched BER, P(det) is smaller** (§3.3f: −19.8 pp vs the CNN). The α-budget zeros bound the gain, they are not the claim. This is the retired stealth headline's error (§2.1) run in reverse — over-claiming a negative is as wrong as over-claiming a positive. Pairs with the matched-detectability rule in §2.7 and with [[show-full-frontier]]. |

**Two of his items are closed by the row above, and should be written up as closed rather than left
hanging.** (i) *"Look at the GAN literature and see whether it transfers"* — it was looked at, and the
answer is a principled **no** for M0, with the reason (closed-form density ratio) and the citation
trail (`mohamed2016implicit` derives the GAN objective from hypothesis testing; Goodfellow's optimal
`D* = p_data/(p_data+p_g)` says the same thing). One sentence in Related Work, not silence.
(ii) *"Formulate it explicitly as a zero-sum game between detectors and jammers"* — still **owed** in
the write-up. The divergence formulation is the clean way to do it: the attacker maximizes BER subject
to `D(p₁‖p₀) ≤ δ`, and Pinsker converts δ into a bound on the defender's best achievable error, so the
two objectives are explicitly opposed with a stated value function.

**Nuance, do not skip it:** `D_NP` depends on the attack law, so if the generator moves, `p₁` moves
and the optimal test moves with it. It **is** still a minimax problem. The difference from a GAN is
that the inner best response is **analytic** — recompute the LRT rather than learn it — which is both
stronger and stable. Under the divergence formulation the inner problem disappears entirely.
**Implementation trap:** a hard 2-D IQ histogram is **not differentiable**, so if the learned
detector is in the loop it needs soft binning or a KDE. `D_NP` is differentiable as written.

## 2.9 Supervisor mandates — settled, not up for discussion

Full record and open items in [Appendix B](#appendix-b--supervisor-record).

- **"Simplify, as much as possible."** His reasoning is blunt: *the simple simulations already do not
  work, so the complicated ones certainly will not.* sim06–08 are **frozen**. Every layer the ladder
  added is now a liability for *understanding*, not an asset.
- **Adaptation cost is the headline claim**, not "the jammer evades the CNN".
- **Hard quota: only the 2–3 strongest experiments go in the main paper**, everything else to the
  appendix. Candidates: **E1** the M0 trade-off frontier with the full baseline envelope; **E2** the
  noise-level ablation; **E3** the spatial/multi-jammer coordination result.
- **The omniscient jammer is the "impossible to beat" reference and must appear in every results
  figure**, not just in prose.
- **Intro must scope out bit-error recovery** — FEC/ARQ/retransmission is out of scope, assumed
  handled by a higher layer — **and then motivate why raw BER/SER is still the right target**: it is
  the input any recovery layer receives, and pushing it past the code's correcting capability is what
  becomes outage. **⚠ 2026-09-16: this scoping is load-bearing for any stealth/detectability claim** —
  with FEC+interleaving *in* scope, covert jamming (raw BER O(1/√n)) sits below the code's correcting
  radius and is nulled (§2.1 "Sharpened"). State it as a deliberate assumption, not a convenience.
- **"Put as much info as possible in Overleaf."** The document is the working record, not a write-up
  phase at the end. Assumption table, baseline table and ablation list go in *now*, as stubs if
  necessary.
- **The noise-level sweep is the primary ablation**, on a **log grid** ("change exponentially"), and
  he has predicted its direction (detection falls as noise rises).
- **⚠ NEW 2026-09-12 — the novelty must live in coordination and synchronisation, not in the neural
  architecture.** His words: the contribution *"seems the application of some neural architecture"*,
  and the fix is *"a novelty in the neural architecture or in how defenders and attackers interact,
  coordinate, and syncronize within themselves"*. We take the second branch. **Operationally this
  forbids a paper whose method section is a network diagram** — the method section has to be about
  the coordination protocol and what delay does to it (§2.3 RQ1).
- **⚠ NEW 2026-09-12 — model the inter-jammer communication delay and the desynchronisation it
  causes.** Required in M1 (§2.4), swept in RQ1. He notes it degrades the *defender* too, which is
  worth keeping: it is a symmetric realism constraint, not just an attacker handicap.
- **⚠ NEW 2026-09-12 — the Introduction must show the single-agent → multi-agent transition
  explicitly**, and he considers the single-agent case *"already a problem in itself"* worth
  understanding. Both halves matter: the first is a writing instruction, the second is a sanctioned
  fallback if the multi-agent work does not land in time.

**Coverage — which mandate is discharged by what.** Added 2026-09-10 because the mandates, the plan
(§3.4) and the written deliverables (§B.3) were three separate lists with no way to check that every
mandate has an owner. Verify this table before claiming a mandate is met. **Note the split:** the two
reward mandates are recorded in **§2.8** (method decisions), not in the list above — they are his
instructions but were filed as decisions. They are listed first here so this table covers all of them;
§2.8 remains where the reasoning lives, and is not restated.

| Mandate | Discharged by | Status |
|---|---|---|
| **Reward = `BER − β·detections`, nothing else** (recorded §2.8, verbatim §B.2) | M0 carries no proxy terms; the historical objectives it replaces are tabulated in [A.0](#a0-run-index) | **code clean, not yet exercised** — M0 has no trained attacker, so this first *binds* at G5 |
| **Power is a hard environment constraint, never a reward term** (recorded §2.8) | `m0/attacks.py:62` `project_power`, applied at `attacks.py:220` — a projection onto the budget, not a normalisation | **done** |
| Simplify, as much as possible | M0 replaces sim06–08; sim00–08 + `frontier/` frozen (§3.6) | **done** |
| Adaptation cost is the headline | G7 (R0/R1/R2); the NP−learned gap E1 already measures | **not started, and now in tension with RQ1** — the 2026-09-12 pivot makes the coordination gain the headline and demotes adaptation cost to the strongest *detector-facing* claim. He has to choose; §4.1 #0 |
| Only 2–3 experiments in the main paper | triage table (§B.3); candidates E1/E2/E3 | **blocked on him** (§4.1 #4), and the candidate list changed on 2026-09-12: **E3 (coordination) is now the lead**, E1 demoted to a calibration/appendix result |
| Omniscient jammer in **every** results figure | `m0/figures.py` | **PARTIAL — 2 of 4.** `fig_tradeoff` (L92) and `fig_frontier` (L135) carry `counter_flip`; **`fig_detectors` (L162) and `fig_stealth_vs_sigma` (L202) do not** — their attack lists omit `counter_null`/`counter_flip` |
| Intro scopes out FEC/ARQ, then motivates raw BER/SER | Intro rewrite (§B.3) | **not started** |
| Put as much info as possible in Overleaf | assumption/baseline/ablation stubs (§B.3) | **not started** |
| Noise sweep = primary ablation, log grid | G2 (M0) · §3.3c (sim08) · **E2 (§3.3g)** | **done on the live CGAN models 2026-09-23** (SNR 0–40 dB + noiseless anchor, all 21 generators, §3.3g) and on the sim08 stack 2026-09-16 (§3.3c). M0/E1's σ grid is still not log-spaced throughout (§3.2) — not selected for E2 |
| Untrainability written up as a *result* | Experiment History summary (§3.5) | **drafted, not pasted** — one paragraph in `sec_exphist.tex`; the long write-up was cut 2026-09-15 for the page limit |
| Novelty in coordination/synchronisation, not architecture (2026-09-12; user re-raised it 2026-09-26) | §2.3 RQ1's delay-decay curve; G6 / D4 (§3.4) | **PARTIAL 2026-09-26.** *Interaction:* the Methodology draft is reframed to lead with training through the deployed detector, and the architecture is one cited sentence (`paper_drafts/sec_methodology.tex`). *Attacker-side sync:* the delay-decay curve is **measured by transfer, not learned** (D4a, §3.3i: ~1 symbol of timing accuracy, bounded by containment §3.3h). The learned policy D4b has a first cut and is **paused** until the GAN rework settles (§3.3j). It learns the power lever but not delay compensation. Sync enters the paper only if time allows (user); until then it is parked as Future Work. *Defender-side coordination:* none; detectors decide alone by design |
| Inter-jammer delay + desync modelled and swept (2026-09-12) | M1 spec (§2.4); G6; D4a | **PARTIAL 2026-09-26** — swept on the CGAN link as leader–follower timing error σ ∈ [0, 16] symbols (D4a, §3.3i). The delay's effect on the *defender*, which he also flagged, is not modelled. M1 does not exist |
| Intro shows single→multi transition (2026-09-12) | Intro rewrite (§B.3) | **not started** |

The paper-side home for these mandates was a closing "What the History Determines" subsection; it was
**dropped 2026-09-15** with the condensed Experiment History (§3.5). State each mandate as a design
constraint where the paper uses it, never as an instruction.

## 2.10 Exploratory track: CGAN jamming waveforms under detection

**STATE 2026-09-16: opened 2026-09-14 by the user. Step 1 closed as a partial reproduction (user
decision, below; results §3.3b), with run001 under async as step 2's reference generator. Step 2 not
started — it waits on the statistical-detector choice (§4.2 Q8).**

**⚠ STATE 2026-09-16 — step 2 as written (stealth conditioning) is PROPOSED FOR REMOVAL, awaiting
supervisor reply (§4.1 #0b).** After the stealth review (§2.1 "Sharpened 2026-09-16") the user proposes
dropping stealth conditioning entirely and instead strengthening the reproduced GAN *waveform* with
**MARL multi-jammer coordination as an effectiveness result**: decide a jamming signal, split it across
jammers at different powers under a shared **energy budget** so it combines at the victim, then extend
to a **zero-sum game** of jammers vs legitimate nodes and compare against the anti-jamming baselines.
Detectability stays an evaluation axis (co-located detector); adaptation cost stays a possible secondary
claim. Consequence if he agrees: the §4.2 Q8 statistical-detector choice and the Q10 Zhang & Krunz
adaptation are moot. **Nothing here is folded in as settled — do not delete the step-2 plan below until
he replies.**

**⚠ STATE 2026-09-17 — reframed as a characterisation study; the direction email now offers the
supervisor two options (§4.1 #0b).** A literature review on jamming-waveform synthesis (this session,
desk work — no code, no compute; `verify.py` still valid as last run) reached three conclusions:
- **Amuru & Buehrer (TIFS 2015) is the closed-form ceiling.** The BER-maximising jammer against known
  BPSK/QPSK/QAM over AWGN is a two-level (pulsed-QPSK) distribution — our `boundary`/`counter` attacks
  and Zhou's "optimal" are instances of it. It *also* solves the coordinated multi-jammer optimum
  (Theorem 4): **3 dB/doubling, but only under perfect synchronisation; uncoordinated ⇒ single-jammer**
  — the analytic twin of the sim08 §3.3c #4 null and of the desync axis. Add it as the reference.
- **The generative papers only converge to *known* optima** (often to the *matched* jammer, which
  Amuru proves is suboptimal — cf. §3.3b, where run001 reached neither). So a generator is an
  **instrument, not the finding**; "a GAN found the near-optimum" is a convergence demo, and it hits a
  certification paradox — you can only call it "near-optimum" where the optimum is computable, i.e.
  where you did not need the GAN.
- **The one arguably-novel angle is the deployed-detector-vs-NP-optimal-warden gap** (§2.1 surviving
  item 1): a frozen CNN/CWT detector has a far looser false-alarm budget than the NP test, so the
  covert headroom against a *real* detector is measurable, with the NP test as the ruler.

**Chosen build (option B, staged — the user's decision this session):** report a generative jammer's
detectability across a **detector suite** on the matched BER–P(det) plane, **with the NP-optimal test
on the same axes as the ceiling**, framed as *characterisation of the detector gap*, effect allowed to
be negligible — the point is to demonstrate the system, not to claim a strong attack. Plan in §3.4
(D-series). Design constraints fixed this session, so the matrix is not another matched-config/sync
artifact:
1. **NP-optimal test is a detector row, non-negotiable** — else it is "our detectors missed it" again
   (§2.6). At waveform level the NP test is exact only for the analytic jammers (barrage, pulsed-QPSK);
   the GAN's received density needs a KDE — or run the exact-NP panel in M0.
2. **Two panels:** an **M0** panel (energy · kurtosis · CNN-on-IQ-histogram · **exact NP**) for rigour,
   and a **waveform** panel (energy · kurtosis · spectrogram-CNN · CWT · NP-vs-analytic) for realism —
   the spectrogram/CWT detectors cannot run per-symbol, which forces the waveform layer and costs the
   clean NP ceiling.
3. **"Stealthy GAN" requires a detector-aware training term** — a GAN trained on Zhou's imitation
   losses is not stealthy by design; without step-2 conditioning the honest title is "detectability of
   a reproduced GAN jammer".
4. **A multi-jammer/MARL row must be *coherent combining*, not N independent GANs** — the latter buys
   nothing at matched detectability (Amuru Thm 4, sim08 §3.3c #4). MARL is a **gated tail** ("only if
   the single-jammer study bites"), and there the generator is incidental — the per-agent signal can be
   Amuru's; the coordination policy is the object with no closed form.

**Learned control tier (D2a) — decided 2026-09-18 (user), run 2026-09-19; results
[§3.3e](#33e-learned-shaped-noise-control-d2a--shaping-buys-effectiveness-not-stealth-2026-09-19).** A stealth-trained GAN (D2) that beats *fixed* jammers mixes up two
effects: having a stealth objective at all, and generating raw IQ. The fair control is a structured
jammer family trained with the **same** objective against the **same** detectors, so GAN vs control
isolates the hypothesis class. Decided:
- **Add, do not replace.** The fixed rows stay as the envelope (§2.5: floor, Amuru family, genie, NP).
- **Shaped Gaussian noise only** (`attacks.shaped`: 32 log-PSD gains + a 16-slot periodic envelope,
  48 parameters — low-dimensional parameters, within §2.8). A 1–2-parameter family (noise, pulsed(p))
  would only rediscover the grid, which already optimises it exactly.
- **Black-box, score-based CMA-ES**: the attacker sees the detector's per-frame scores, never its
  gradients. Not RL, because a static waveform choice is not a sequential decision problem, and A.5
  records scalar-reward policy gradients stalling against a CNN. RL stays reserved for D4.
- **Reward exactly §2.8's**; power by the existing per-frame **equality** projection. A ≤ budget
  would let a learner jam ~0.7 % of frames at full power, which hides inside the 2σ confirmation
  tolerance.
- **Caveat for D2:** the GAN will be trained white-box (gradients through D); this control is
  black-box. With 48 parameters CMA-ES should reach the family's optimum either way (the seed check
  agrees, §3.3e), so the control measures the family's best, not the optimiser's.

This reframing **supersedes neither** the step-2 plan below nor option A: it *absorbs* option A's MARL
as the gated tail and returns stealth as a measured axis. **Nothing folded in as settled until he
replies (§4.1 #0b).**

Everything else in Part 2 describes the coordination direction, which is **paused, not superseded**:
nothing in §2.1–2.9 was re-validated when this track opened. Where this track contradicts a settled
decision, the contradiction is spelled out below rather than silently overridden.

**Why.** The findings up to 2026-09-12 — E1's impossibility result (prior art, §2.1) and a
coordination question with no code behind it yet (§3.4 G6) — were judged too weak to build on. This
track tests whether a different idea is stronger before any further commitment.

**Goal, in two gated steps.**
1. **Reproduce Zhou, Tan & Xu 2025** — *"Communication Jamming Waveform Generation Technology Based on
   Conditional Generative Adversarial Networks"*, ISSET 2025, pp. 373–377
   (`source_papers/L.Zhou 2025.pdf`; Overleaf bibkey `11184988`, drafts use `zhou2025cgan`). A CGAN
   (generator conditioned on a modulation label; discriminator with scoring, feature and
   auxiliary-classifier heads) generates jamming waveforms whose BER-vs-JSR curve, at SNR 30 dB, sits
   close to an "optimal" matched-modulation jammer and needs 2–6 dB less JSR than Gaussian noise to
   reach BER 1e-3. **Scope: QPSK only** (user decision 2026-09-14). The conditioning path (label
   embedding, auxiliary classifier) is kept with `n_classes = 1`, because step 2 reuses it.
2. **Condition the generator on stealth** — only after step 1 is signed off. Evaluate against three
   detectors: **(i) power threshold** (energy detector); **(ii) one statistical detector — which one is
   still to be discussed** (§4.2 Q8); **(iii) Zhang & Krunz 2023** as the SOTA learned baseline —
   *"Detection and Classification of Smart Jamming in Wi-Fi Networks Using Machine Learning"*, MILCOM
   2023, pp. 919–924 (`source_papers/Zhang & Kunz 2023.pdf` — the filename misspells Krunz; bibkey
   `zhang2023detection`): Morlet CWT scalogram → 4-conv-layer DCNN₁ (32,292 parameters).

**Success bar for step 1 — decided 2026-09-14.** A number-for-number match is not achievable: the
paper leaves most link and training parameters unstated (§4.2 Q7), and its Figs. 4–6 plot BERs down to
~1e-8 while its stated test (10⁴ symbols × 100 repetitions ≈ 2·10⁶ bits) cannot resolve anything
below ~5·10⁻⁷. So:
- the unstated link parameters are **calibrated** by fitting the Gaussian-noise curve — the only one
  with a closed form — to the digitised Fig. 6 (QPSK);
- **REVISED 2026-09-14 at the C2 checkpoint (user decision): report only, no pass/fail.** Step 1
  reports, at BER = 1e-3, **JSR_GAN − JSR_opt** and **JSR_noise − JSR_GAN** on the calibrated link,
  next to the paper's own values from the digitised Fig. 6 (**1.31 dB** and **4.44 dB**), with our
  curves overlaid on the paper's. Whether that counts as a reproduction is judged from those numbers,
  not from a threshold.
- *Superseded:* the original bar was "GAN − Optimal ≤ 1 dB and Noise − GAN ∈ [2, 6] dB". It was
  dropped because C0 showed the paper's own figure fails its first half (1.31 dB).

**Step 1 closed — decided 2026-09-16 (user): a partial reproduction.** Judged from the report-only
numbers (§3.3b):
- **Reproduced:** the paper's headline — the GAN beats white noise at BER 1e-3 by 6.66 dB (async) /
  4.38 dB (locked), against its "2–6 dB".
- **Not reproduced, recorded as a finding:** the ordering Optimal ≳ GAN (ours −1.30 dB async, paper
  +1.31) and the low-JSR curve shapes (a link property, §4.2 Q7(v)).
- **Step 2's reference generator is `artifacts/cgan/run001_G.pt`, evaluated async**
  (`run001_async_ber_vs_jsr.json`). run002 is not used.
- **Trap:** `train_cgan.py`'s defaults are run002's recipe, and run001's loss definitions (MSE feature
  matching, log1p STFT, headroom 1.0) are no longer in the code. Retraining for step 2 must first settle
  the recipe (§4.2 Q11); the existing checkpoint can be used as is.

**The link — decided 2026-09-14 at the C2 checkpoint, as stated assumptions (`cgan/link.py` `LINK`).**
- **sps 8.** A common simulation choice, well above the Nyquist minimum (sps > 1 + β).
- **RRC pulse, β = 0.35, span 32.** The classic default roll-off.
- **White-noise ("full"-band) jammer.** A classical barrage jammer.
- **Asynchronous, random-phase structured jammers — Optimal *and* GAN.** A jammer is not synchronised
  to its victim unless that is assumed. **Extended to the GAN on 2026-09-15 (user decision)**: G is
  trained on segments cropped on the victim's symbol grid at carrier phase 0, so tiling its output
  unchanged had scored it as a *locked* jammer while Optimal was async (`jammers.desync` now applies one
  model to both; locked-for-both is reported as a sensitivity row, §3.3b). The paper states no sync
  at all. "No offset" is not the neutral reading — it assumes the victim's clock and carrier phase are
  known at the receiver — and on our link it makes Optimal no better than noise at BER 1e-3 (locked
  Noise − Optimal 0.10 dB, random-phase 2.06 dB, async 5.33 dB; paper 5.75; C2 `calibration.json`).

Each choice has a reason independent of Zhou's figure. The C2 calibration is kept as a record of where
they land against it:
- **Noise** crosses 1e-3 at −0.67 dB (paper −1.32).
- **Optimal** crosses at −6.00 dB (paper −7.06).
- **Noise − Optimal** is 5.33 dB (paper 5.75).

Fitting was considered and rejected. The paper's Noise curve fits no single sps, and its far-left
Optimal points come from a figure with other signs of not being plain simulation output (§4.2 Q7).
Tuning β to 0.25 would have gained 0.35 dB by fitting that figure.

**Why sps > 1 is not a detail** — it came up at the checkpoint and holds for every later stage. M0
(one number per symbol) and sim06–08 (OFDM, jammer per resource element) both put the jammer **on the
victim's own symbol grid**, an unstated perfect-synchronisation assumption (§4.4). Two things follow:
- Zhou's noise-vs-optimal gap comes from **bandwidth** (the matched filter averages out wideband noise
  by a factor of sps) and **timing** (only a time-offset jammer causes errors below −3 dB JSR, §3.4
  C2). Neither exists on a symbol grid. **At sps = 1 Zhou's claim reverses** (checked 2026-09-15,
  closed form + C2's synchronous curves, which do not depend on sps): white noise crosses BER 1e-3 at
  −9.84 dB, Optimal at −0.77 dB (locked) or −2.73 dB (random phase), so *noise* is the better jammer
  by 7–9 dB, and a 1024-sample segment is i.i.d. symbols with a flat spectrum, leaving the STFT loss
  nothing to match. This is why `cgan/` alone runs at sps 8; M0 and the rest of the thesis stay at
  one sample per symbol, and spatial effects stay out of scope for steps 1–2.
- Once **spatial positions** enter, distances become continuous propagation delays and carrier
  phases. A delay is representable only on an oversampled waveform (sps > 1), where Sionna's
  `cir_to_time_channel` / `ApplyTimeChannel` apply. With sps = 1 a delay can only be a whole number of
  symbols, so neither geometry nor the inter-jammer delay of M1 (§2.4) could be expressed.

A physical symbol rate and carrier frequency are needed only once real distances enter; step 1 does not
need them.

**Decision 0 — libraries, decided 2026-09-14.**

| Component | Chosen | Why | Rejected |
|---|---|---|---|
| **GAN** (G, D, losses, loop) | **plain PyTorch** | Zhou's model is a custom composite — four generator loss terms (STFT, adversarial, feature matching, I/Q distance) and three discriminator terms (adversarial, gradient penalty, auxiliary classifier) on 1-D I/Q, not images — so any GAN framework would be overridden exactly where the paper is specific. Step 2 changes the objective, which is easiest when every term is visible in one file. Already in the venv. | **StudioGAN** (image-only, YAML-driven, last updated Aug 2024, torch 1.13 image); **TorchGAN** (CI covers torch 1.8/1.9 only); **Lightning** (not GAN-specific, adds a dependency for a single-GPU job); **TF-GAN** (splits the stack) |
| **Link** (mapping, pulse shaping, matched filter, AWGN, BER) | **Sionna 2.0.1** — `sionna.phy.mapping`, `sionna.phy.signal` (`RootRaisedCosineFilter`, `CustomFilter`, `Upsampling`, `Downsampling`), `sionna.phy.channel.AWGN`, `compute_ber` | Tested filters and mappers; listed in §1.1's library rule. | plain torch port of `m0/link.py` |

**Consequence of the Sionna choice:** `import sionna` fails on the login node (§C.2), so **everything
in `cgan/` runs through sbatch, `verify.py` included** — unlike M0's CPU exception. The GAN modules
stay pure torch and do not import Sionna.

**What this track reopens, and why each is legitimate here.**
- **§2.8 "Conditional generator + direct gradient. NOT a GAN."** The argument was that a
  discriminator estimates a density ratio, which **in M0 is closed form**. At waveform level, against a
  learned CWT-CNN detector, it is not closed form, so the GAN is back in the likelihood-free case it
  exists for. The M0 argument stands for M0.
- **§2.8 "Actions are low-dimensional perturbation parameters, never raw IQ".** What sim06/06b/07
  falsified (§A.5) was **policy-gradient RL over raw IQ, driven by a scalar black-box reward**. A
  generator trained by **direct gradient through a differentiable discriminator** is a different
  method: the gradient reaches every output sample. Do not read this track as reviving the falsified
  method, and do not use it as licence to try PPO over IQ.
- **§3.6 "Stealth as the attacker's objective — retired".** Step 2 reintroduces stealth, as a
  **conditioning variable** on an exploratory track. The square-root-law argument of §2.1 has **not**
  gone away: a covert jammer against an optimal warden pays for covertness in effectiveness. Step 2's
  results must be read against that bound, not presented as if it did not exist.

**Still binding on this track** (§2.7–2.8): compare at **matched detectability**, not matched
configuration; the stealth budget is the detector's own **clean false-alarm rate**; **power is a hard
constraint** — here, the jammer waveform is scaled to the target JSR by projection, never penalised in
a loss.

**Decision 2026-09-27 — the method (user: "we want best results").** The detector-aware generator is
trained **from random weights, 4000 steps** (run003), not warm-started from the imitation-trained Zhou
generator (400 steps). Zhou is cited for the generator's layout only, described as "a generic
convolutional network with a dense output layer … nothing in the method depends on it" — Zhou cites no
source for it, and 8.39 M of its 8.6 M parameters are the final dense layer. The imitation start and the
400-step budget are Experiment History (`paper_drafts/sec_exphist.tex`, "Imitation as initialization"),
promotable to the main text if space allows. The frozen deployed detector is framed as the generator's
only discriminator (the co-adaptive version = D6). **Power:** fixed-power runs keep the equality
projection; `--power learned` makes it a capped choice (≤ per frame, §3.3f run005).

**Risks recorded at opening.**
- The supervisor mandate of 2026-09-12 (§2.9): novelty must not live *"in the neural architecture"*. A
  reproduced CGAN is architecture by construction; step 2's contribution has to be the stealth
  conditioning and what it reveals, not the network.
- Zhou's "optimal" jammer is a matched-modulation waveform, **not** the BER-maximising jammer under a
  power constraint (Amuru & Buehrer, §2.1) — see §4.2 Q9.
- Zhang & Krunz is an OFDM Wi-Fi *classifier* of preamble/pilot/interleaving attacks; on single-carrier
  QPSK it can only be used as an adaptation — see §4.2 Q10.

---

# PART 3 — CURRENT STATE

## 3.1 Status line

**STATE 2026-09-30 (r2c, morning): the final experiment is built and run (session B). Results: §3.3q.
Wed 30.9 is the read-out with the user; Thu–Fri writing; the supervisor meets the user Thu 1.10 17:00.**
- **What exists.** `final/` (§1.3): copies of `cgan/` code, one `--dependency` chain per environment,
  verify 100/100 + the regression gate vs S4 (job 2278412; 97/97 before the random push, job 2277992). All
  six environments evaluated (22 attackers × JSR −50…+15 dB × 512 frames, Es/N0 24 dB), figures +
  cross-environment comparison in `artifacts/final/`. Figures redrawn 2026-09-30 on the user's request:
  damage and P_det stacked on one JSR axis, six attackers per detector (§1.3 `final/figures.py`). Nothing
  committed.
- **Report page (private), written for the user, who will ask questions about it in later sessions:**
  <https://claude.ai/artifact/MDtg7r4sLU8rJ8ZwopBEGd>. Its HTML source is in git-untracked
  `artifacts/final/report/index.html`; it loads its figures from `artifacts/final/<env>/fig*.png` +
  `compare/fig_*.png` (published as `figs/…`) and builds its per-environment tables in the browser from
  `artifacts/final/<env>/summary.json` (published as `data/<env>_summary.json`). To change it, read it
  back with the Artifact tool (`action: read`, that URL) and republish to the same `url`.
- **The outcome, bluntly (§3.3q):** against the CNN at matched damage the CNN-targeted GAN has **no
  meaningful gain over the classical envelope** (within ±4 pp; −4.6 ± 0.9 pp only at `noise_1db`, where
  the control reaches the same floor), **the detector term buys nothing** (the β = 0 control hides as well
  or better), and the only large gain is over Zhou's CGAN (−86 to −93 pp, a weak baseline). What holds is
  the detector side: **noise-level uncertainty blinds the CNN, not energy after the matched filter; fading
  blinds the naive energy detector, not the CNN; a gain-aware energy detector is immune to fading.** At
  excess BER 3e-4 the on/off jammer sits at the FAR on every detector.
- **Two §3.4 premises were wrong and are corrected there:** energy vs full-band power white-noise
  transitions are 10 dB apart (not ≈ 1), and energy's noise share is ≈ 1/10 of power's (not 1.35/8).
- **Session A** (literature, in parallel) wrote `paper_drafts/sec_setup_operating_point.tex` and
  `refs_new.bib` entries and edits §4.2 Q13; its report is its own.
- **NEXT (user, today):** read both reports; choose the system model (recommendation: `both`, by
  literature realism — no environment favours the GAN by more than 5 pp); decide what the paper claims
  given the GAN shows no gain over the literature baselines (the detector-side finding is the robust one).
- Earlier facts that still hold: P_det is defined in §2.7; D6 is parked (§3.4); `cgan/`'s SNR is power
  per sample over the 8× band, so every "15 dB" in this README is Es/N0 24 dB.

The block below is the state before the final experiment.

**STATE 2026-09-29 (r2c). The whole background batch except S3 is done and merged into `main`'s working
tree (uncommitted): Track 1's D6 round 1 (§3.3o, committed 79375cb) and Track 2's S2 / S4 / S5 (§3.3l–n,
merged 2026-09-29; merged suite verify job 2277747, all 23 sections, exit 0). S3 (per-symbol detector) is
specified, not built. Two things changed the picture: (i) the user rejected "damage per extra alarm" —
results are now stated in literature-grounded measures (§2.7), which overturned S4's first write-up;
(ii) S5 showed energy detection loses at 15 dB when the receiver does not know its noise level. Report
page for S1/S2/S4/S5 with figures: <https://claude.ai/artifact/LuFXpEu5cW4SwcPPfwqVu7> (private).
The update email of 2026-09-27/28 (§B.1) is still unanswered.** The user's own verdict on the results
stands: *"it's not really an interesting finding"*; the goal is a paper that holds without a clear
positive result. **No Results/Setup/Intro/Conclusion text is drafted yet.** MARL (D4) is out of the
paper as proposed (§3.3j) — thesis/TWC material.

**What the measurements now say — at matched damage, quoted with SNR (measures: §2.7):**
1. **Energy detection holds only when the receiver knows its link gain AND its noise level.** Numbers
   for the best generator (run003 CNN β 10) at excess PER 0.1 — fraction of frames the one-sided energy
   detector flags / frames broken before a 1 %/1 % sequential test on its alarms decides:
   - known gain and noise: **0.63 / 0.3 at 30 dB, 0.26 / 1.8 at 15 dB** (§3.3m; a generator trained at
     15 dB does not do better, S4);
   - unknown gain (S1, 30 dB, naive detector): 0.11 / 16 under 0.1 dB of shadowing, 0.06 / ≈ 700 under
     1 dB; a gain-aware detector stays at 0.51–0.65 at every σ (§3.3k);
   - unknown noise (S5, 15 dB, ±1 / ±2 dB): honest CFAR 0.14 / 7 and 0.09 / 57; naive detector 0.28 / 7
     and 0.32 / 21 with its false-alarm rate at 0.14 / 0.23 (§3.3n). At 30 dB noise uncertainty does
     nothing (the noise is 0.1 % of the received power).
   The mechanism is the SNR wall (Tandra & Sahai): the threshold sits just above the expected clean frame
   power, and any unknown spread in that power either raises false alarms or widens the threshold. **The
   old wording ("≈ 1 broken frame per extra alarm, a floor no jammer beats") is withdrawn**: its unit
   read the saturated corner 1/(1 − FAR) (§2.7).
2. **The learned CNN detector is the weak link.** The CNN-targeted generator (run003, random init) is
   flagged by the CNN on ≤ 0.11 of frames at matched BER 3e-4 over 0–30 dB SNR (its β = 0 control: 1.00
   at 30 dB); at 15 dB, 7–8 % at PER 0.1 even by a CNN retrained at 15 dB (§3.3m); with free power
   (run005, CNN β = 100) it breaks 35 % of frames at −4 dB while the CNN flags it 0.000 at 30 dB; a
   self-trained surrogate CNN transfers with no gap (run004, grey-box) (§3.3f).
   **⚠ Conditional on SNR (D6, 2026-09-28, §3.3o).** At 30 dB one retrain on Li's four types down to
   −35 dB, never shown a generator, lifts the CNN on held-out generators from 0.16–0.18 to 0.58–0.74 at
   matched BER (energy on the same frames 0.60–0.81); the round-0 attackers stay below energy (0.44–0.48
   vs 0.57–0.65 even for a CNN retrained on them). At 15 dB no retrain moves the CNN (≤ 0.09 vs energy
   0.22–0.43), and retraining costs it 4–5 pp on Li's own classes.
3. **On bit damage, a trivial on/off jammer wins** (loud matched QPSK on a fraction of frames): 3–18×
   more bits per extra alarm than any shaped waveform (§3.3g E2b). The winner depends on the damage
   metric — average BER rewards concentrating errors, frame errors reward spreading them. *(Measured in
   the withdrawn unit; to re-measure at matched damage, §4.3.)*
4. **Why sim04 looked far better:** it was a genie (knew the current symbol, lossless, synchronous) and
   was judged by kurtosis alone at a hand-set threshold — information and measurement, not method
   (§A.3, `artifacts/cgan/iq/fig_iq_sim04_vs_gan.png`).
5. Inside the strict α budget, confirmed stealthy BER is still 0 for every non-genie jammer.

**Story proposed to the supervisor (2026-09-28, awaiting his reply):** *learned jamming detectors are
the weak link; energy detection holds every jammer to ≈ 1 broken frame per alarm* — a measurement study
with the generator as the instrument. **⚠ Its second half is false as stated (2026-09-29).** The email
says "a plain energy detector isn't fooled"; the data say energy detection holds only with a known link
gain (S1) and a known noise level (S5, at 15 dB), and the "≈ 1 frame per alarm" unit is withdrawn
(§2.7). The Wednesday draft must state finding 1 with its conditions, and the supervisor must be told
the claim changed. The first half (the CNN is the weak link) survives every probe, with D6's SNR
condition (finding 2). Which story the paper tells is open — the user's call after the next discussion.

**NEXT: two parallel fresh sessions overnight 29/30.9, both specified in §3.4 "Final experiment".**
- **Session A (literature and paper text):** verify Es/N0 24 dB, σ_N 0.5 / 1 dB, K 12 / 28 dB and
  α 0.05 against the originals; write the Setup paragraph; publish a report.
- **Session B (build and run):** `final/`, then verify, then the regression gate against S4, then the
  six environment chains, figures and the cross-environment comparison; publish a report.

A does not block B, because every value is already decided. S3 (the per-symbol detector) stays
specified and unbuilt, outside the final experiment.

**The week, re-planned by the user on 2026-09-29:** Wed 30.9 all experiments · Thu 1.10 writing, and the
meeting at 17:00 · Fri 2.10 writing and submit. The email of 2026-09-27/28 had promised a full draft to
him on Wednesday evening. Whether that still goes out is the user's call.
The supervisor's reply came on 2026-09-29 (§B.1): a single-jammer paper is acceptable if it has
outperformance quantification.

**Where the pieces are.** Method: §2.10 "Decision 2026-09-27". Metric conventions: §2.7. Runs:
§3.3f (run003 paper method · run004 grey-box · run005 learned power), §3.3g (E2 on run003 · E2b
on/off + outlier alarm + PER). Figures to look at first: `artifacts/cgan/snr_ablation/run003/
fig_headline_vs_control.png`, `artifacts/cgan/outlier_alarm/run003_per/fig_damage_per_extra_alarm.png`,
`…/fig_frontier_ber_vs_per_spec_cnn.png`, `artifacts/cgan/iq/fig_iq_sim04_vs_gan.png`. The two figures
attached to the 2026-09-28 email: `artifacts/cgan/snr_ablation/run003/email/` (`cgan/email_figures.py`).

> **Cluster home is under quota; delete this block once `~/.over_quota` is gone** (still present 2026-09-27). All new generator checkpoints (run002–run005, the ablations) live on net_scratch `cgan_gan/`, symlinked from `artifacts/cgan/gan/` — keep it that way. The stale VS Code
> builds are deleted, and `du -sh ~` measured **4.9 G** on 2026-09-26 and **5.2 G** on 2026-09-27 against the **6.8 G soft limit**
> (`~/.vscode-server` 2.2 G, `~/BT` 2.5 G). The empty `~/.over_quota` flag from the 2026-09-23 ISG
> warning is still there; whether ISG clears it on its own is not known. If it survives the end of the
> grace period (~2026-09-28) with home still under 6.8 G, ask ISG. Procedure: `cluster/README.md` (Storage).

**0d · D1 plain GAN + D2 detector-aware GAN, analytic detectors — DONE 2026-09-20 ([§3.3f](#33f-cgan-under-detection--d1-plain-gan-and-d2-detector-aware-gan-all-four-detectors-2026-09-20-cnn-2026-09-23)).**
The actual step-2, built white-box (§3.3f). Plain GAN (D1) and 13 trained generators (D2: power-1s/2s,
kurtosis × β{1,10,100,1000}) placed on the BER–P(det) plane. **Result: stealthy BER ≈ 0 against every
detector — an 8.6M-parameter white-box generator does no better than D2a's 48-parameter control.** The CNN
was not targeted then; it was closed 2026-09-23 with the same answer (§3.3f). Report:
<https://claude.ai/artifact/PBXBr7EQQ3huK7rocvSMxX>.

**0f · E2, the noise ablation on the live models — DONE 2026-09-23 ([§3.3g](#33g-e2--the-noise-ablation-on-the-live-models-30-db-is-near-the-worst-place-to-measure-the-gain-2026-09-23)).**
The supervisor's *mandated primary ablation* (§B.2), which until now existed only on the frozen sim08
stack (§3.3c). 10 levels (SNR 0:5:40 dB + a noiseless anchor) × 7 classical attacks × all 21
generators, every detector re-calibrated per level, CNN weights frozen at 30 dB. **The 30 dB level
reproduces §3.3f** (regression gate: 0/264 P(det) points outside 4σ in all 28 rows; BER deviations
unbiased scatter). **Headline (2026-09-23, 400 steps, vs D1 — superseded by E2 on run002, §3.3g):
the matched-BER gain peaks at −85.1 pp at 15 dB SNR, 5× the 30 dB value, and vanishes by 40 dB.** Mechanism: the *stealth edge* falls ~0.7 dB per dB of SNR — the
detector sharpens as the link cleans — while the BER onset is flat, so the gap grows with SNR. This
**refutes the mechanism proposed when the ablation was specified** (the edge was expected to be flat
in JSR). Its `fig_frontier_by_snr.png` moves the FAR with the budget — the ≤ FAR view the user
rejected on 2026-09-24 (§2.7) — and is to be replaced by fixed-α trade-off curves (parked, §4.3). **Tier 1 is evaluation only — it measures transfer, not achievability; Tier 2
(retrain per level) is gated on this.** Report page, 8 figures:
<https://claude.ai/artifact/PonQqobu2xpk52oWYfyoAB> (private, not yet shared).

**0b · Learned control tier D2a — DONE 2026-09-19 ([§3.3e](#33e-learned-shaped-noise-control-d2a--shaping-buys-effectiveness-not-stealth-2026-09-19)).**
User decision 2026-09-18: add a *learned* structured jammer as the fair control for the stealth GAN
(§2.10). Shaped Gaussian noise (48 parameters), CMA-ES, black-box and score-based, one jammer per
detector, 99 runs. **Result: shaping buys effectiveness, not stealth.**
- **Effectiveness:** shaping crosses BER 1e-3 **15.9 dB** earlier than white noise.
- **Stealth:** the CNN is evaded completely up to −20 dB, but only by moving out of the victim's
  band, where the jammer does nothing. **Stealthy BER stays 0 against every detector.**

**0 · Classical baselines on the 3-D waveform link — DONE ([§3.3d](#33d-classical-baselines-on-the-3-d-waveform-link--the-pre-gan-envelope-2026-09-17)).**
The user directed the GAN1/2/3 build and, as its preparation, a full classical baseline set: every
attack (barrage · pulsed-QPSK · genie) × four separate detectors (power 1/2-sided · kurtosis ·
spectrogram-CNN · exact noise NP test) × K = 1–4 jammers, on a new 3-D geometry (Sionna RT, LOS,
TX power-controlled to 30 dB SNR; **no synchronisation modelled** — decided this session, checked in
`verify.py`). New flat modules in `cgan/` (`scene`, `channel`, `attacks`, `detectors`, `baselines`,
`train_spectrogram_cnn`, `baselines_figures`); step-1 files untouched. **Result: the M0/E1
impossibility reproduced at waveform level** — every *realisable* attack has confirmed stealthy BER 0 at
K = 1–4; only the genie flip (BER 1, invisible to all) and push (invisible to 1-sided power + CNN, caught
by 2-sided power + kurtosis) reach the stealth region; uncoordinated multi-jammer buys nothing; the NP
test flags noise ~35 dB below where it does damage; Li et al. CNN reproduced (99.94 % at +10 dB). Report
page (9 figures): <https://claude.ai/artifact/YEEnZ7rkkNmcWuLJJe3yoy> (private until shared). **Q8 is
resolved (kurtosis).** Not gated on the supervisor — it is the shared spine both #0b options need.

**1 · sim08 ablations — DONE ([§3.3c](#33c-sim08-ablations--noise-jammer-power-number-of-jammers-2026-09-16)).**
Agreed with the supervisor (user, 2026-09-16) and run from `sim08_ablation/`, which imports the frozen
sim08 stack read-only. Headlines: within the suite's own false-alarm rate the strongest jammer holds
**BER 0.005–0.016 at every Eb/N0 ≥ 15 dB** while the clean floor falls away (×1.4 at 15 dB → ×109 at
40 dB); detection is set by the jammer's **total power** (not its spread); **uncoordinated
multi-jammer buys nothing at matched detectability** — the m1 trap again, and the baseline RQ1's
coordination must beat; *suite ≡ CNN* holds over 0–40 dB. A shareable report page with all five
figures exists: <https://claude.ai/artifact/V3cFxTXKidDbjdqvf3yiTR> (private until shared).

**2 · CGAN step 1 — CLOSED 2026-09-16 as a partial reproduction ([§3.3b](#33b-cgan-step-1-result--ordering-not-reproduced-once-synchronisation-is-consistent-2026-09-15), decision in [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection)).**
The 2026-09-14 "gaps match Zhou" claim scored the GAN as a synchronised jammer and Zhou's "optimal" as
asynchronous. Under one consistent sync model the GAN *beats* "optimal" (GAN − Optimal −1.30 dB async,
paper +1.31) while still beating noise (6.66 dB), and run002 — the recipe backed by Zhou's own
references — is a worse imitation. The user accepted run001 under async as step 2's reference generator
(option (a)); the ablation that would explain run002 (option (b)) is parked as §4.2 Q11.

**3 · Direction — TWO options for the supervisor, AWAITING REPLY ([§4.1 #0b](#41-blocking--needs-supervisor-input)).**
Both keep the reproduced GAN *waveform* and report on the matched BER–P(det) plane; the same #0b email
now carries both. **(A) 2026-09-16 — drop stealth, add MARL coordination as an *effectiveness* result**
(energy-budget power split → coherent combining at a co-located detector), then a zero-sum
jammer-vs-legitimate game vs the anti-jamming baselines. **(B) 2026-09-17 — keep stealth as a
*characterisation* result**: a generative jammer's detectability across a detector suite vs the
NP-optimal ceiling — the *deployed-detector-vs-optimal-warden gap* (§2.1 surviving item 1), effect
allowed to be negligible. The user's chosen build (§3.4 D-series) stages toward (B) with A's MARL as a
gated tail ("only if it bites"). **Not folded into §2.10/§4.2 as settled until he replies (user
instruction).**

**0e · D2 vs the spectrogram CNN — DONE 2026-09-23 ([§3.3f](#33f-cgan-under-detection--d1-plain-gan-and-d2-detector-aware-gan-all-four-detectors-2026-09-20-cnn-2026-09-23)).**
The last untested detector. **No surrogate CNN was trained** (this corrects the 2026-09-20 plan): the
only non-differentiable step in the eval path is the 256-entry viridis LUT, a staircase with zero
gradient, so `detectors.spectrogram_image(..., grad=True)` keeps its exact forward value and substitutes
the colormap's local slope in the backward pass (a straight-through estimator). The attacker therefore
differentiates the **deployed** weights — white-box like every other D2 target, no transfer gap to price,
no retrain. **Result, in the framing the user chose 2026-09-23: at matched BER the CNN-targeted generator is
measurably less detectable** (P(det)_CNN 1.000 → 0.802 vs the plain GAN, −19.8 pp, ~11σ) **at 12.8 dB
less power** — a small but real finding. Its limits: the gain vanishes above BER ~1e-3, a *classical*
Amuru pulsed jammer still evades the CNN better (0.421 — at 30 dB only; below ~22 dB SNR the learned
generator is the stealthier one, §3.3g 1b), β has an interior optimum at 1–10, and nothing reaches the α
budget. Numbers and the full table: §3.3f.

**Two traps were found by inspection before burning a cluster cycle, and both would have produced a
convincing-looking null.** They are the reusable lesson of this round:
1. **The JSR band must straddle the *target* detector's transition.** `(-32, 0)` was set for power
   (flags from ~−27 dB); the CNN's transition is 12–16 dB lower, so every sampled JSR would have been
   saturated-detected with `sigmoid'((stat−thr)/scale)` ~1e-11 and **no β could have recovered a
   gradient**. `train_gan.JSR_BANDS` now gives `spec_cnn` (−48, −16). The training logs confirm the fix:
   `P_det_soft` moves 0.99 → 0.17–0.31 instead of sitting pinned at 1.
2. **β must be scaled to the detector, not inherited.** {1…1000} suited the analytic detectors; the CNN
   sweep runs to 1e6 (tasks 17–19). *Secondary to the band* — the original evidence for this (a 0.041
   detector-term gradient) was measured inside the saturated region, so it mostly measured trap 1.

**Known and measured: the deployed CNN statistic runs under `autocast(fp16)` on CUDA, D2 training in
fp32** (fp16 gradients underflow without a GradScaler). The straight-through LUT itself is *exact* —
identical images on GPU, identical statistic on CPU where neither path autocasts. The fp16/fp32
difference alone reaches 0.375 × clean std at its worst over 1024 clean frames, but moves only 0.6 % of
clean decisions and no jammed P(det). `verify.py` §14d measures the two effects **separately** and
asserts only decision-level equivalence; it prints the fp16 bias each run, so revisit only if it grows.

**Two parallel next actions (user, 2026-09-26).**
1. **Finish the paper draft — the user is doing this, due Sunday 2026-09-27.** The write-up of the
   single-jammer D-series and E2: `paper_drafts/*.tex` to paste into Overleaf (§1.4), figures in
   `artifacts/cgan/`. So the supervisor can work on it Monday 2026-09-28, leaving Tue–Fri for his
   corrections before ICC on Fri 2026-10-02. **The email to him goes out Sunday night.** System Model
   and Methodology drafts (`paper_drafts/sec_system_model_waveform.tex`, `sec_methodology.tex`, both
   untracked) are being edited by the user this session; §3.3h/E3 and MARL do **not** go in the paper
   until MARL has results. **No E3/team_fading numbers in the paper yet either** — the pre-check is
   internal, its role is to steer D4.
2. **Build the MARL coordination experiment (D4) in the background — the extend-vs-MARL fork resolved to
   MARL (user, 2026-09-26).** The E2 follow-ons (§4.3) are shelved. **The design is settled (§4.2 Q12,
   2026-09-26): the claim is the delay-decay curve, each drone observes its own geometry, and the
   actions are a power fraction and a transmit advance.** D4a is done (§3.3i): the team stops matching
   one jammer at σ ≈ 0.5–1 symbol, and uncompensated geometry sits right there (1.1 symbols rms), so
   both actions stay. **D4b first cut done, then PAUSED (user, 2026-09-26): the user is reworking the GAN
   itself, which could change everything MARL measures, so MARL waits for it (§3.3j).** The resume steps
   are at the end of §3.3j. The teammate-information arm (§4.2 Q12) is not built.

> **How to state it** is now a settled method decision — see the last row of
> [§2.8](#28-settled-method-decisions). In short: report the matched-BER gain; the α-budget zeros bound
> it and are not the claim.

`gan_figures.pick_cnn_tag()` chooses which CNN-targeted β the figures draw, by a stated rule: **lowest
P(det)_CNN at matched BER 1e-3**, the operating point the headline is quoted at. It draws **β = 10**.
The earlier rule ranked by confirmed stealthy BER at the α budget, where every row ties at 0, so its
tie-break silently drew β = 1e5 — the *least* interesting generator; corrected 2026-09-23 and the
figures regenerated.

**Report pages — do not send:** the §3.3f page (<https://claude.ai/artifact/2fRAFrMoYAfCZWPe1KvTfV>,
2026-09-23) and the E2 page (<https://claude.ai/artifact/PonQqobu2xpk52oWYfyoAB>) are built on the run001
generators, superseded by run003 (§3.3f). None was ever shared; the 2026-09-28 email attached two new
figures instead.

**Supervisor contact.** The sim08 ablation axes were discussed with him before 2026-09-16; after that he
heard nothing until the **2026-09-28 update email** (§B.1). The #0 and #0b emails and an update drafted
2026-09-24 were never sent. **Why MARL is out of the paper (the reasoning behind the email's proposal):**
with the detector at the victim's receiver the detector sees exactly the jammer sum the victim decodes, so
a coordinated team can save *transmit* power (coherent combining, Amuru Thm 4) but cannot beat one jammer
of equal *received* power on the BER–P(det) plane. Coordination only moves the plane when the detector sees
a different mixture (separated detector, several receivers, mobility), which needs the parked geometry
layer plus a learned decentralised policy — about a week of code. So: ICC = the single-jammer study, and
MARL follows for the thesis/TWC. The email offers one small MARL experiment if he insists, flagged as a
likely negative result.

**Paper drafts (2026-09-15):** the Experiment History was cut from a 10-part appendix (~4,300 words) to
a ~1-column summary, `paper_drafts/sec_exphist.tex`, for the 6-page limit; parts 9–10 dropped. Not yet
pasted into Overleaf; the user was still editing it as of 2026-09-15 (§3.5). **It predates §3.3c** and
does not mention the ablations.

**System Model draft (2026-09-24):** the "document all findings" write-up began this session with the
System Model. `paper_drafts/sec_system_model_waveform.tex` (new) describes the BUILT waveform system,
checked line by line against `cgan/`: single-carrier QPSK, received-JSR + async jammers, information
tiers T0/T1/T2 (new labels, defined there), **only the generative jammer's signal model** (an abstract
generator G, the segment-stream waveform `eq:gan`, tier T1; Zhou's architecture and both training stages
are in the Methodology draft below, per its split rule), four *separate*
detectors + the NP reference (exact for barrage only, no detector ordering claimed), black-box vs
white-box threat model, and the objective with its frontier. **Three user decisions shape it:** (i) the
section holds **only the system** — metrics, reporting conventions, training method, values, scope and
future work are parked verbatim at the end of the file, sorted by target section (Methodology /
Experiment Setup / Introduction / Conclusion); (ii) **the existing baseline attacks** (barrage, pulsed,
shaped control, genie + ladder table) **go to Experiment Setup**, while the detectors stay because they
define how detection is measured; (iii) it is **geometry-free** — no 3-D placement, path
loss or power control, and an explicit no-fading, perfectly synchronised receiver (conservative for the
attacker on BER, for the defender on P(det)). The geometry layer the experiments ran on is parked as a
ready-to-add subsection. It is additive — `verify.py` shows a single jammer depends on received JSR alone
— so a later MARL/3-D/mobility extension adds it back without touching the rest. With fixed positions and
a co-located detector geometry is only a gain vector; it becomes irreducible only with a detector away
from the victim, mobile jammers, or a non-empty scene. ~1,000 words of typeset text (comments stripped,
measured 2026-09-26); the header holds the measured constants and the cite keys still owed. The typeset
text has had one style pass with the project's `no-ai-slop` skill (C.6), and the parked blocks have not;
the parked Setup text still reads "α is each detector's operating point, *not a filter on the results*".
That pass made two wording changes worth checking before pasting: the learned detector is now "the
single-carrier counterpart of the spectrogram classifier of Li et al." (the "state of the art" claim went,
because one citation cannot carry it), and the two-sided energy test is now said outright to be "not blind"
to power-reducing attacks, which the text had only implied (§3.3d: two-sided power catches the genie push). **This is the System Model
matching the D-series direction; the older `paper_drafts/sec_system_model.tex` is the M0 single-symbol,
h₀-equalised draft (coordination framing) and does not describe the built waveform system.** Neither is
in Overleaf, and `main.tex` §System Model still carries the OFDM/TDL/N_J setting no experiment supports.
**The user picked the waveform model on 2026-09-24** ("the new system model"); the M0 draft is thesis
material.

**Methodology draft (2026-09-24):** `paper_drafts/sec_methodology.tex` (new, ~680 words, ≈1.1 column).
**The split rule (user, 2026-09-24):** the System Model says what the jammer can transmit, under which
constraints and knowledge (an abstract generator G, `eq:gan`, tier T1); the Methodology says how its
weights are chosen, in two stages. (1) *Generator and Plain Training*: Zhou's architecture, the
discriminator, and the four imitation losses, with no BER or detector term. (2) *Detector-Aware
Fine-Tuning*: the log E[BER] + logistic soft-P(det) surrogates (`eq:loss`, `eq:softpdet`), the power
projection inside the forward pass, the per-target JSR band, the straight-through colour LUT, and what
the D1/D2/D2a comparisons isolate. The shaped control's CMA-ES recipe and the evaluation protocol are
parked at the end of the file for Experiment Setup. **The user judged both drafts not yet correct
(2026-09-24); they are due in Overleaf by Sunday 2026-09-27 with the rest of the paper.** Both drafts cite `Sec.~\ref{sec:setup}`, so Overleaf's Experiment Setup
needs `\label{sec:setup}`. This one also needs two new bib entries
(`bengio2013estimating`, `hansen2016cma`, now in `refs_new.bib`). Not yet in Overleaf, where
§Methodology is still `\rar{TODO}`.

---

**STATE 2026-09-12 (coordination plan — PAUSED 2026-09-14, kept as written): the project changed direction, and the supervisor's proposal feedback arrived
the same day pointing the same way. Read [§2.1](#21-the-question-and-what-has-been-falsified) first.**
Stealth is retired as the attacker's objective — the impossibility result E1 measured is prior art
three times over — and the direction is now **cooperative multi-agent generative jamming, with the
coordination gain measured as a function of inter-jammer delay**, and detectability kept as the
comparison axis. M0's code and E1's numbers all stand; what changed is the claim they support.

**The two events are independent and mutually reinforcing.** The literature check (ours) killed the
old headline. His feedback (§B.2) — *"the main contrib now seems the application of some neural
architecture … a novelty in … how defenders and attackers interact, coordinate, and syncronize"*
plus the inter-jammer-delay note — supplies the replacement, and it is a sharper question than the
one we would have written alone. **He does not yet know about the literature collision.**

**20 days to ICC (2026-10-02).** This is the tightest the schedule has been, because the pivot moves
the lead experiment from E1 (done) to E3/coordination (not started, needs the M1 multi-jammer
extension that does not exist yet). **The plan in §3.4 has been re-cut accordingly and is now
front-loaded on G6.**

**Blocking** — see [§4.1](#41-blocking--needs-supervisor-input): thesis **registration** (open since
2026-09-01, still the longest-running item) · and **top of the list: sign-off on the pivot** (§4.1
#0). *Proposal feedback (#3) and the inter-jammer coordination assumption (#8) both closed on
2026-09-12.* Do not spend the remaining 20 days building toward a headline he has not agreed to —
though note his feedback already endorses the *direction*, so #0 is narrower than it was this
morning: it is the literature collision and the RQ1/RQ2 tension, not the change of subject.

**The single next action is [§4.1 #0](#41-blocking--needs-supervisor-input) — email Di Maio.** It is
half a day of writing at most, it unblocks everything, and every compute item below reads differently
depending on his answer. Draft it around three points: (i) the literature collision, stated plainly
with the three citations; (ii) the proposed new RQ1; (iii) the RQ1-vs-RQ2 tension he has to break
(§2.3). **Do not start G6 before that reply** — it is 4–5 days of new code committed to one branch of
the fork.

> **⚠ The table and next-action just above belong to the PAUSED coordination plan (2026-09-12), NOT to
> now.** The current state and single next action are at the top of §3.1. The list below is preserved
> so the coordination plan can be resumed intact if the CGAN track is dropped — it is not a to-do list.

**Next session of the coordination plan, in this order (paused).**

| # | Do | Where | Note |
|---|---|---|---|
| 0 | **Email Di Maio: the pivot** | §4.1 #0 | *Do this first, before any other work.* Everything below is contingent on it. Open by adopting his coordination/synchronisation steer, then disclose the literature collision, then ask him to break the RQ1/RQ2 tension. |
| 1 | **Read the three prior-art papers in full** | §2.1 table | Bash JSAC 2013 · Li TIFS 2016 + 2020 · Amuru TIFS 2015. Desk work, no compute. Needed before *any* novelty claim goes in print, and needed to write the email in #0 credibly. |
| 1b | **Apply his prose fixes to `proposal/proposal.tex`** | §B.3 | Small and unblocked: broadband-jamming citation, the two surviving grammar items, the single→multi transition paragraph, the delay/desync passage. **Do not** rewrite the proposal's RQs before #0 returns. |
| 2 | **Replace Overleaf's Experiment History with `paper_drafts/sec_exphist.tex`** | §3.5 | Condensed 2026-09-15 to ~1 column; it also removes the wrong two-agent claim ("roughly double") still live in Overleaf. |
| 5 | Add the omniscient reference to `fig_detectors` and `fig_stealth_vs_sigma` | §2.9 coverage, `m0/figures.py:162,202` | ~15 min, CPU-only. Still correct under the pivot — the omniscient ceiling is a baseline, not a stealth claim. |
| 6 | G6 planning only (**not code**) until #0 returns | §3.4 | Spec the M1 multi-jammer extension on paper so it can start the hour his reply lands. |

**Do not** re-derive the sim04 numbers from the README alone — §A.3's figures came from the SLURM
logs on 2026-09-10 and the prose that preceded them was wrong. `simulation04/runs/slurm_99211.out`
(run001) and `slurm_100041.out` (run007) are the sources. **sim04 matters more after the pivot than
before it:** it is the only existing evidence that a coordinated solution is gradient-reachable, so
it is now RQ1's precursor rather than a historical footnote.

> **Where the last working session stopped (2026-09-12).** No experiments were run and nothing in
> `m0/` was modified, so `verify.py` is still valid as last run. Two things happened, both desk work:
> a **literature check on E1's novelty** (§2.1) came back negative and triggered the pivot, and
> **Di Maio's proposal feedback arrived** and was transcribed into [B.2](#b2-his-verbatim-points-and-what-each-changed)
> with its consequences propagated. Nothing in `paper_drafts/` was edited that day (the
> experiment-history drafts were later condensed, 2026-09-15, §3.5). His annotated proposal was saved to
> **`proposal/proposal_reviewed_2026-09-12.tex`** (new file, untracked); `proposal/proposal.tex` is
> the older pre-feedback draft and both are kept (§B.3).

## 3.2 What exists and is verified

**M0** (`m0/`, artifacts in `artifacts/m0/`). One QPSK symbol at a time, one channel (`h = 1`), AWGN
with swept σ. No OFDM, no fading, no pilots. `y = s + d + w`, decisions by per-axis sign test. That
is the entire model. Everything runs in seconds.

**Verified, not assumed** (`m0/verify.py`, ~50 checks, exit 0 = all pass): unjammed BER matches
`Q(1/(σ√2))` to Monte-Carlo tolerance; SER matches `1−(1−BER)²`; every attack matches its geometric
prediction; all detectors calibrate to the target false-alarm rate; the ordering `P_NP ≥ P_L ≥ P_E`
holds everywhere.

**E1** — the sweep, an 8-task SLURM array, one σ per task (job 2243879, 8 s/task; detectors trained
by job 2243867, 89 s). *(`m0/runs/` also holds **job 2243871**, a superseded first pass with 11 powers
topping out at 1.0 → 135 rows/σ. The on-disk JSONs are 2243879's — 172 rows, powers extended to 2.0.
Ignore 2243871; it is kept only as the SLURM log.)* Grid: **σ ∈ {0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}** (Eb/N0 28.0 → 0.0
dB), **13 power values** 0.001–2.0, duty-cycle variants, α = 0.05, 4000 frames × 256 symbols.
Outputs `artifacts/m0/frontier/results_sigma*.json` + `e1_{frontier,tradeoff,detectors,stealth_vs_sigma}.png`.

> **Gap vs the stated design: the σ grid is not log-spaced throughout.** It is log-spaced at the low
> end (0.02, 0.05, 0.1) and roughly linear above (0.15, 0.2, 0.3, 0.4, 0.5), where he asked for
> exponential spacing throughout — a `--sigmas` string and a detector retrain, minutes on CPU; not
> selected for E2. **σ = 0 is not simulated by design, and that is correct:** `m0/frontier_m0.py`
> documents that the Gaussian densities degenerate there (any perturbation is detected with
> probability one) and states the anchor analytically, BER*(P) = min(P, 0.5). So σ = 0 is a sentence
> in the write-up, not a sweep point. *(Corrected 2026-09-24: this note used to call σ = 0 a missing
> sweep point "cheap to fix".)* **Two things are lost by omitting σ = 0:** the cleanest
> possible demonstration that *stealth is a noise phenomenon* (at σ = 0 the "less detected" half of
> the claim cannot hold for **anyone**, so the comparison collapses to BER/SER at matched power —
> say that explicitly), and a **free calibration point for RQ2**, since any gap between the learned
> detector and the NP-optimal one at σ = 0 is *pure detector suboptimality*.
>
> **Range sanity, for the record** (unit-energy QPSK, σ per real dimension, N₀ = 2σ²): σ = 0.5 →
> Eb/N₀ ≈ 0 dB, σ = 0.1 → ≈14 dB, σ → 0 → ∞. So [0, 0.5] spans the useful range and **overlaps
> sim08's 5–30 dB from below**, which keeps the appendix results comparable to the new ones. The
> Eb/N0 column in §3.3 confirms the implemented grid does this.

## 3.3 E1 result — correct, reproducible, and no longer the headline

> **STATE (2026-09-12): every number in this section stands. Its *status* changed.** The literature
> check (§2.1) found the impossibility finding to be prior art — square-root law, symmetrizability,
> disguised jamming — so E1 is **no longer the paper's lead result**. Its surviving roles, in order
> of strength: (i) the **adaptation-cost reference point** at the end of this section, which is the
> one measurement the check did *not* find prior art for; (ii) the **energy-detector blind spot**,
> likewise; (iii) a **calibration result** — evidence that the M0 instrument reproduces known theory,
> which is what licenses using it on the coordination question. Write it up as (i)+(ii) with (iii)
> as the framing, and **cite the prior art rather than competing with it**.

Max **excess** BER over the clean floor, while staying under the detector's own false-alarm rate
(α = 0.05), against the **NP-optimal** detector:

| σ | Eb/N0 | clean floor | barrage | gaussian | boundary_blind | boundary_genie† | permute† | counter_flip† |
|---|---|---|---|---|---|---|---|---|
| 0.02–0.10 | 28–14 dB | 0.0000 | — | — | — | 0.500 | 0.500 | 1.000 |
| 0.15 | 10.5 dB | 0.0000 | — | — | — | — | 0.500 | 1.000 |
| 0.20 | 8.0 dB | 0.0002 | — | — | — | 0.051 | 0.500 | 1.000 |
| 0.30 | 4.4 dB | 0.0093 | — | — | **0.0002** | 0.491 | 0.491 | 0.982 |
| 0.40 | 1.9 dB | 0.0386 | — | — | — | 0.462 | 0.462 | — |
| 0.50 | 0.0 dB | 0.0786 | — | **0.0007** | **0.0003** | 0.211 | 0.421 | 0.843 |

† genie-only: requires per-symbol knowledge of the transmitted symbol.

**No realizable (blind) attack achieves meaningful stealthy jamming in M0** — the best is an excess
BER of 3×10⁻⁴. Only genie attacks work. This is the M0-level, *derivable* version of the sim08
"honesty correction", which measured the same thing empirically on the OFDM stack. **It is also, per
§2.1, the finite-blocklength QPSK instance of the square-root law** — which is why it reads as
confirmation rather than discovery.

> **Sharpened by re-analysis of the raw results, 2026-09-10 — the negative result is stronger than
> the table suggests.** Every stealthy `boundary_genie` point in the entire sweep, at every σ, sits
> at **exactly ρ = √(P/duty) = √2**. There are no exceptions: at any other ρ, `p_np` is exactly
> 1.000. But ρ = √2 along an axis is precisely the **constellation-permuting** point — the push maps
> the QPSK alphabet onto itself. So `boundary_genie`'s apparent success is *not* an independent
> finding about the boundary attack; it is the same symmetry degeneracy as `permute` and
> `counter_flip`, and the README's own caveat applies to it ("knife-edged … a *bound*, not an attack
> anyone can mount"). **The honest reading of E1: every stealthy-and-effective point is a
> constellation-symmetry artifact requiring the genie. The only non-symmetry stealthy points in the
> whole sweep are `boundary_blind` at 2–3×10⁻⁴ and `gaussian` at 7×10⁻⁴ — i.e. nothing.** The E1
> figures and any prose quoting the genie row should carry this. *(Data caveat: the `power` column is
> not meaningful for the `counter_*` attacks — `counter_flip` reports `power = 0.0` while its actual
> energy is 4.)*

**Two detector blind spots — real, but modest, and both genie-only in practice.**
1. *The classical energy detector is one-sided*, because it was designed against barrage jammers that
   ADD power. Minimum-energy attacks push symbols *toward* the origin, so received power FALLS
   (**1.020 → 0.525 at σ = 0.1** — exactly the analytic prediction: symbol power 1 → 0.5 when pushed
   onto the boundary, plus 2σ² = 0.02), and the one-sided test therefore fires **less often than on
   clean frames**. The sweep shows this about as starkly as it can be shown: at σ = 0.1, duty = 1,
   `boundary_genie` gives **one-sided energy exactly 0.0000 at every ρ from 0.03 to 1.23** — against
   its own realised FAR of 0.0428 — including at ρ = 0.707 where it is causing **BER 0.25**. Over
   that same range the **two-sided** test and the NP test both sit at **1.0000**. So the two-sided
   variant closes this blind spot completely, at the cost of being weaker against power-adding
   attacks. Both are implemented; **report both.** *(The one exception is ρ = √2, where all three
   collapse to ≈FAR — that is the permutation degeneracy of blind spot 2, not an energy-detector
   property.)*
2. *Constellation-permuting attacks are exactly undetectable.* Because the payload is iid uniform,
   any perturbation mapping the QPSK alphabet onto itself leaves the received law invariant, so
   `p₁ ≡ p₀` and no test has any power (`p_np` is exactly 0.0000, not merely below α). Members:
   `ρ = √2` along an axis (energy 2, BER 0.5) and `d = −2s` (energy 4, BER 1.0). Frontier
   `BER*(P) = min(P/4, 0.5)` — invisibility costs exactly 4× the power. **Do not oversell this — and
   as of 2026-09-12 the ceiling on how much it can be sold is known.** It is **symmetrizability**
   (Csiszár–Narayan, 1988) and **disguised jamming** (Li et al., IEEE TIFS 2016/2020) in QPSK form:
   both the mechanism and the genie requirement are published, and TIFS 2020 additionally gives the
   defence (break the symmetry with shared secret randomness). Present it as *"the minimal model
   reproduces the known symmetrizability result exactly, including the 4× power cost"* — a
   calibration claim, correctly cited. The earlier caveats all still apply: knife-edged, needs the
   genie, and the `d = −2s` case was already known to us ("statistically clean rx = −tx").

**The adaptation-cost reference point now exists — and after 2026-09-12 this is the most valuable
paragraph in the section**, because it is the one measurement §2.1's check found no prior art for.
At σ = 0.2, α = 0.05, `boundary_blind` at
P = 0.01, **duty = 0.0196**: one-sided energy **0.123**, two-sided energy 0.090, learned CNN
**0.116**, **NP-optimal 0.847**. The learned detector catches ~12% of what an optimal one catches
~85% of — **that gap is the remaining adaptation budget, as a number.** *(Re-read off
`results_sigma0.2.json` on 2026-09-10; earlier notes quoted 0.129 / 0.112 / 0.839, which was
transcription drift. The JSON row keys are `p_e1`, `p_e2`, `p_l`, `p_np` — **not**
`p_energy`/`p_learned`, which silently return `None` and make the detectors look unevaluated.)*

> **Quote that number with its duty cycle, or not at all.** The gap is strongly duty-dependent and
> `duty = 0.0196` is the single most favourable row in the sweep. At the same (σ, P) for
> duty 0.1 / 0.25 / 0.5 / 1.0 the learned detector gets 0.309 / 0.396 / 0.429 / 0.431 against NP
> 0.630 / 0.607 / 0.595 / 0.586 — i.e. **the adaptation budget shrinks from 0.73 to 0.16** as the
> attack spreads out. The honest statement is *"the learned detector's shortfall against the optimal
> test is largest exactly where the attack is sparsest"*, not a single headline number. Reporting the
> best row alone would repeat the m1 mistake (§2.7).

## 3.3b CGAN step-1 result — ordering not reproduced once synchronisation is consistent (2026-09-15)

**Superseded claim — do not quote.** On 2026-09-14 this section read "Zhou 2025 reproduced on QPSK",
with Noise − GAN 4.34 dB and GAN − Optimal +1.01 dB against the paper's 4.44 / +1.31. That run001
evaluation tiled G's output from frame index 0 — on the victim's symbol grid at carrier phase 0 — so it
scored the GAN as a **locked** jammer while "optimal" was **async**. The match came from that mismatch.

On the calibrated link (§2.10: sps 8, RRC 0.35, white noise), SNR 30 dB, JSR −10:2:10 dB, ≥ 100 bit
errors or 10⁹ bits per point. JSR (dB) at BER 1e-3, lower = stronger jammer; GAN − Optimal > 0 means
Optimal is stronger, as in Zhou's figure:

| run / sync model | Optimal | GAN | Noise | Noise − GAN | GAN − Optimal | artifact (eval job) |
|---|---|---|---|---|---|---|
| **Zhou Fig. 6** | −7.06 | −5.76 | −1.32 | 4.44 | **+1.31** | `cgan/paper_fig6.json` |
| run001 as first reported — mismatched | −6.05 | −5.03 | −0.69 | 4.34 | +1.01 | `run001_ber_vs_jsr.json` (2259409) |
| **run001, both async** (the decided model) | −6.05 | **−7.35** | −0.69 | 6.66 | **−1.30** | `run001_async_ber_vs_jsr.json` (2260623) |
| run001, both locked | −0.61 | −5.07 | −0.69 | 4.38 | −4.47 | same file, sensitivity row |
| run002, both async | −6.05 | < −10 | −0.69 | > 9.3 | < −3.95 | `run002_ber_vs_jsr.json` (2260806) |
| run002, both locked | −0.61 | < −10 | −0.69 | > 9.3 | < −9.4 | same file, sensitivity row |

Optimal crossings are lower bounds (zero errors at the left bracket; `jsr_at_1e3_is_lower_bound`).

**1. With one sync model, our GAN beats Zhou's "optimal" jammer** — the paper's ordering Optimal ≳ GAN
is not reproduced. What survives is the paper's headline claim: the GAN beats white noise at BER 1e-3,
by 4.4 dB (locked) or 6.7 dB (async), against its "2–6 dB".

**2. Why run001 beats "optimal" (reading).** G learned only a rough QPSK: constellation EVM 0.39 at the
symbol instants and 0.65 half a symbol off (`run001_async_iq.png`). Its irregular excursions reach the
decision boundary sooner than a clean, bounded QPSK jammer can, which also gives the async GAN curve a
smooth low-JSR tail more like Zhou's than our Optimal's cliff.

**3. run002 — the recipe from Zhou's own references (§4.2 Q7) — is a worse imitation.** WGAN-GP with
n_critic 5 and β (0.5, 0.9) from [7]; L1 feature matching (weight 2), log-clamped STFT loss (weight
45) and headroom 0.95 from [5]. Train job 2260624 (80 min). Eval 2260778 lost its GPU on artongpu01;
the rerun is 2260806. EVM is 0.70. BER is already 3.6e-3 at −10 dB, so it acts as a peaky noise-like
jammer, and it plateaus at 0.31 at +10 dB (Zhou's GAN: 0.54). Its final I/Q-distribution loss is 0.57
against run001's 0.09, while the ×45 STFT term fell 5.95 → 1.04. Reading: the spectrogram weight
pulled G away from QPSK amplitudes, so Fre-GAN's weights do not transfer to this task.

**4. The discriminator wins in both runs.** run001 ends with D adversarial loss 0.011 and G adversarial
loss 5.80, i.e. D(G(z)) ≈ 0.3 %. G's QPSK-likeness came from the STFT, I/Q and feature-matching terms,
not from fooling D. This corrects the 2026-09-14 "stable, all four terms healthy". run002's WGAN critic
gap plateaus at ≈ 57 from iteration ~3,000 to 10,000: training stalled rather than converged.

**5. The curve shapes are a link property, not a GAN problem.** Our Noise and Optimal crossings sit
0.6–1.0 dB right of the paper's, which is the C2 calibration offset (§2.10). Zhou's smooth low-JSR tails
versus our cliffs are explained by the implied-gain diagnostic in §4.2 Q7(v).

**Closed 2026-09-16 (user): option (a)** — run001 under async is a partial reproduction and step 2's
reference generator; the ordering failure above is the finding (decision in §2.10). Option (b), the
ablation that would say *why* run002 failed, was not run and is parked as §4.2 Q11.

**Caveat carried into step 2.** BER is in no loss term (§4.2 Q7): the GAN is effective purely by
imitating the QPSK waveform. So "conditioning on stealth" (step 2) gets no help from the reproduced
objective and is a genuine addition, not a tweak.

## 3.3c sim08 ablations — noise, jammer power, number of jammers (2026-09-16)

Agreed with the supervisor: go back to the sim08 realistic channel and ask which jammer setting is
strongest at matched detectability, and where the frozen CNN is evaded. Two of his named ablation axes
(§B.2): **noise level** and **scenario size (#jammers)**, both on log grids, with **jammer power** as
the third. `sim08_ablation/` is new code that **imports `simulation08/` and `simulation06/`
read-only** — nothing frozen was edited ([A.9](#a9-frozen-code-inventory)).

**Setup.** 18 noise levels (Eb/N0 0:2.5:40 dB plus a **noiseless anchor**) × 13 powers per active
subcarrier (0.01–10, four per decade) × 7 n_active (1…52) × N_J ∈ {1, 2, 4} **independent,
uncoordinated, blind** jammers at **equal total power** (each sends `power/N_J` on its own random
subcarriers through its own TDL-C link) = 4,914 configs at B = 512, plus a 4,096-frame clean reference
per level. Detectors are frozen and unchanged: the channel-valid CNN
(`artifacts/sim08/detector/run001_best.pt`, threshold 0.5) and the 1 %-FAR energy detector, combined
per frame as the suite.

- **P(detect)** is a per-frame flag rate over independent frames: each frame has fresh bits, fresh fading
  on both links and a fresh jammer draw. It is the hit rate with the jammer on, and the false-alarm
  rate (FAR, the stealth budget) with it off. It is not a time-to-detection.
- **The jammer is one family only: sim08's `sparse_blind`.** It picks `n_active` in-band subcarriers
  uniformly at random per frame and puts amplitude `√power` on each, with a random phase that is held
  across all 14 OFDM symbols. So it is unmodulated tones through its own TDL-C link.
- `broadband_inband` is simply `n_active = 52` of this family, so it is in the grid.
- `sparse_channelaware` was left out: it bought ≈ 0 at matched detectability in A.7.
- Time-domain structure is **not covered**: no hopping, no pulsing or duty cycle, no modulated
  waveform. See §4.3. Every headline number is a **confirmation pass on 4,096 fresh frames**: picking
the max-BER config under a noisy P(detect) favours lucky draws, so the frontier pick is re-measured,
and picks whose confirmed P(detect) lands > 2σ above the budget are flagged. Jobs: 2261123 verify
(37 checks, incl. reproducing job 102390 at N_J = 1) · 2261126 / 2261146 / 2261173 sweep · 2261174
figures. Figures and `summary.json` in `artifacts/sim08_ablation/run001/`; shareable report page with
all five figures: <https://claude.ai/artifact/V3cFxTXKidDbjdqvf3yiTR> (version 2, numbers re-verified
against `summary.json` on 2026-09-16). Its HTML source is not in the repo: to update it, read the
artifact by URL, edit, and republish to the same URL; figures are served as `fig/*.png` from
`artifacts/sim08_ablation/run001/`.

**1. Noise changes the detector's false-alarm rate, not its hit rate.** For a fixed jammer (n = 16,
power 1, JSR ≈ −5 dB) suite P(detect) is **flat at 0.42–0.60 across the whole range**, while the CNN's
clean FAR collapses: **0.42 (0 dB) → 0.031 (10 dB) → 0.010 (20 dB) → 0.013–0.025 (25–40 dB and
noiseless)**. Discrimination (hit rate − FAR) therefore *grows* with SNR, 0.17 → 0.48 (suite, 0 → 40 dB). His predicted direction —
*"increase noise … less detection"* — holds **as a statement about the false-alarm rate and
discrimination, not about the jammer's detection probability**, which noise barely moves.

**2. At matched detectability the strongest stealthy jammer is SNR-independent in absolute terms.**
Within the suite's own clean FAR, the best confirmed jammer causes **BER 0.005–0.016 at every Eb/N0
≥ 15 dB**, including the noiseless anchor — essentially constant — while the clean floor falls from
1.1e-2 to 4.5e-5. Its advantage over the floor therefore grows ×1.4 (15 dB) → ×2.4 (20) → ×24.5 (30)
→ ×109 (40 dB) (best over N_J, preferring picks that passed confirmation; `summary.json`). Below
~12.5 dB the best stealthy jammer is within ×1.2 of the floor, i.e. it does nothing. This **refines**
[A.7](#a7-sim08--the-realistic-channel-and-the-honest-metric): restricted to one jammer as A.7 was, the
absolute numbers reproduce (0.013 at 20 dB — a pick that failed confirmation — and 0.0054 at 30 dB, vs
A.7's 0.011 / 0.004), but "the stealthy region shrinks at high SNR" was the wrong reading — it holds
its BER while the floor drops away beneath it.

**3. Detection is set by the jammer's TOTAL power, not by how it is spread** (within this jammer
family). At every Eb/N0 ≥ 7.5 dB the largest total power that stays indistinguishable from clean (2σ,
one jammer) is **0.7–1.8, received JSR −18.5 to −14.5 dB**, across a 1000× power grid and a 52×
spread in n_active — the diagonal frontier in `stealth_map.png`. At 0–5 dB it rises to 2.3–3.6,
because there the budget itself is 14–43 % false alarms. So the attacker's only real lever against
this detector is total power; the shape of the occupancy is free. The strongest settings are
correspondingly varied — 1 to 52 subcarriers per jammer, one to four jammers, total power 0.5–1.6 —
and equally good.

**4. More jammers at equal total power buy nothing at matched detectability — the m1 trap again.**
Mean over Eb/N0 ≥ 15 dB, one jammer vs two vs four at the same total power (n = 16 each):
BER **0.066 → 0.084 → 0.096** (+45 %), but suite detection **0.475 → 0.513 → 0.544**. At matched
detectability the gain disappears: median confirmed stealthy BER **0.0078 / 0.0092 / 0.0088**, fully
overlapping. This is the same error shape as the retracted "+70 % channel-aware" (§2.7): splitting
power is *louder*, and the extra BER is bought, not free. **For RQ1 this is the baseline coordination
has to beat** — uncoordinated multi-jammer is worth zero here.

**5. "Suite ≡ CNN" now holds across the whole noise range.** Only **26 of 4,914** configs are caught by
the energy detector when the CNN misses them (0.5 %). A.7 established this for 5–30 dB; it extends to
0–40 dB and the noiseless anchor. The expensive CNN is still the whole detector.

**6. The frozen CNN degrades gently outside its training range** (Eb/N0 5–30 dB, power 0.3–8): clean
FAR stays 1.3–2.0 % at 32.5–40 dB and noiseless, and it still detects jammers down to 0.01–0.03 per
subcarrier, 10–30× below its training minimum. No blind spot opens up out of distribution.

**Caveats.** The CNN runs at threshold 0.5 (not FAR-calibrated, unlike the energy detector) — as in all
sim08 work; the budget is each detector's *own* clean FAR, so the comparison is still matched. **35 %
of frontier picks at the FAR budget failed confirmation** (38 / 108; 18 % over all budgets) and are
drawn hollow — selecting under a noisy P(detect) really does favour lucky draws, so never quote an
unconfirmed sweep max. Our clean CNN FAR measured on 4,096 frames is
about **twice** job 102390's B = 512 estimates at 10–15 dB (0.031 vs 0.014, 0.014 vs 0.006); the dense
sweep's estimates were noisy at that batch size — quote these. Unchanged from sim08 and still binding:
held random-phase jammers only, perfect-CSI ZF, the jammer sits on the victim's symbol grid (§4.4), the
detector is frozen (no retraining round), and the multi-jammer arm is **uncoordinated** — it is not
G6/E3.

## 3.3d Classical baselines on the 3-D waveform link — the pre-GAN envelope (2026-09-17)

**User-directed build (this session), preparation for the GAN1/2/3 track (§3.4).** New flat modules in
`cgan/` (`scene.py`, `channel.py`, `attacks.py`, `detectors.py`, `baselines.py`, `train_spectrogram_cnn.py`,
`baselines_figures.py`) extend the step-1 link into a **3-D geometry** and measure every classical attack
against **four separate detectors** for K = 1–4 jammers, stored as the baseline set the GANs are judged
against. Step-1 files untouched (run001's evaluation still reproduces). **Not gated on the supervisor** —
it is the shared spine both option A and B need (§4.1 #0b); nothing in §2.10/§4.2's *direction* status is
folded in as settled.

**Model (settled with the user 2026-09-17).** LOS free-space channel from **Sionna RT** (empty scene,
`max_depth=0`; Mitsuba **LLVM** variant — the cluster has no OptiX, job 2266119); nodes uniform in a
1 km × 1 km × 150 m box, ≥ 50 m apart, nested drops so K = 1–4 share placements; QPSK sps 8 RRC 0.35,
f_c 2.4 GHz, 1 MBd, NF 7 dB; **TX power-controlled to SNR 30 dB at R** (`sionna.sys`), so the clean
received law is identical in every drop → one detector calibration serves all geometries. **No
synchronisation is modelled** (decision this session): a jammer's timing is computable from positions but
its carrier phase is not (λ/2π ≈ 2 cm), and async is not weaker where it matters (§3.3b), so every
realisable jammer is asynchronous and only the genie ceiling is synced. Two simplifications are **checked
in `verify.py`**, not assumed: geometric delay/phase on-vs-off gives the same BER and P(det); a single
jammer depends on received JSR alone.

**Detectors (four, separate, never OR-combined; each at its own clean FAR α):** one- and two-sided
**power**, **kurtosis**, the **Li et al. spectrogram CNN** (EfficientNet-B0, retrained), and the exact
**Neyman–Pearson test for a white-Gaussian jammer** (the optimality ceiling, noise rows only). Q8 is
thereby **resolved: kurtosis** is the statistical detector (§4.2 Q8).

**Attacks:** `noise` (barrage) · `pulsed_qpsk(p)` p ∈ {1, .5, .25, .1} (p = 1 is Zhou's "optimal"; p < 1
is Amuru's pulsed family) · `omniscient(η)` η ∈ {.1, 1} (the genie ceiling; η = 1 is the symbol flip).

**Result — the M0/E1 impossibility, reproduced at waveform level in 3-D against a SOTA detector:**
- **No realisable attack is both effective and stealthy.** Confirmed stealthy BER (max BER with
  P(det) ≤ α = 0.05, re-measured on fresh frames) is **0** for noise and every pulsed variant, at
  K = 1–4. They cross BER 1e-3 only at JSR 0 / −6 / −13 dB, tens of dB after every detector already reads
  P(det) = 1.
- **Only the genie reaches the stealth region.** The **omniscient flip** causes BER 1.0 while every
  detector sits at its 0.05 floor (received law ≡ clean). The **omniscient push** causes BER up to 1.0
  invisible to one-sided power (P(det) = 0.000, the classical blind spot to power-*reducing* attacks) and
  to the CNN, but **two-sided power and kurtosis catch it**.
- **Uncoordinated multi-jammer buys nothing at matched detectability** — stealthy BER stays 0 for
  K = 1/2/3/4 at equal total power; the BER-vs-budget curves overlap (the m1 trap again, §3.3c #4). This
  is the baseline any coordination result must beat.
- **The optimal warden has huge headroom over the deployed CNN**: the NP test flags noise at −35 dB,
  ~35 dB below where noise does any damage — the measurable deployed-detector-vs-optimal gap (§2.1 item 1).
- **Li et al. reproduced**: retrained EfficientNet-B0 reaches **99.94 % at JSR +10 dB** (their regime,
  AUC 0.993); 96 % over a JSR mix down to −20 dB where jamming is below noise. **A finding for A.4:** the
  frozen sim05/06 code stretched each spectrogram to its own 2–98 % contrast, which erases any jammer that
  lifts the whole band evenly — a **fixed** dB scale (Li et al.'s waterfall convention) is required; this
  is a cleaner explanation of sim05's "barrage undetected" than "spectrograms need OFDM".

Jobs: 2266131 (CNN, 100 epochs) · 2266138 (K = 1) · 2266140 (K = 2–4 array) · 2266150 (figures).
`verify.py` +43 baseline checks on top of the 121 link checks, all pass (job 2266130). Outputs in
`artifacts/cgan/baselines/run001/` (9 figures + `summary.json` + `sweep_K{1..4}.json`);
detector + thresholds in `artifacts/cgan/baselines/`. **Shareable report page (all 9 figures):**
<https://claude.ai/artifact/YEEnZ7rkkNmcWuLJJe3yoy> (private until shared).

**Next: GAN1** — the plain reproduced GAN (run001_G) placed on this same BER–P(det) plane; it will land
on the floor with noise/pulsed, which is the honest "detectability of a reproduced GAN jammer" result.

## 3.3e Learned shaped-noise control (D2a) — shaping buys effectiveness, not stealth (2026-09-19)

**The user's decision (2026-09-18, reasoning in §2.10):** the control for D2's stealth GAN. The same
objective, the same detectors, the same link and scene as §3.3d (K = 1), with a structured family
instead of raw IQ.

**Method.**
- **Family:** `attacks.shaped(θ)`, θ ∈ ℝ⁴⁸:
  - white complex Gaussian noise, FFT-shaped by 32 log-PSD gains over the simulated band;
  - times a periodic 16-symbol envelope of log-power gains, in the jammer's own clock with a random
    offset per frame (the jammer does not know where the victim's frame starts);
  - asynchronous through `channel.receive`, scaled to the JSR per frame with equality;
  - θ = 0 is exactly the barrage jammer (`verify.py` §13).
- **Optimiser:** CMA-ES (`pycma`) from θ = 0 in `train_shaped.py`. σ₀ = 1 in log-power units,
  150 generations, 256 frames per evaluation, common random numbers per generation.
- **Ranking:** exactly §2.8's reward, **E[BER] − β·P̂(det)**:
  - P̂(det) is the target detector's hard flag rate at its α = 0.05 threshold;
  - E[BER] is *exact* over the AWGN: Q(margin/σ) at the noiseless matched-filter sample
    (`attacks.expected_ber`, checked against Monte-Carlo for white and in-band noise).
- **Tie-break.** With hard P̂(det) alone, the first smoke run (job 2267047) was exactly flat at
  −20 dB and CMA-ES stopped at generation 1. The CNN flagged every candidate in every frame, and at
  SNR 30 dB a fixed jammer leaves Q(~76) per bit, which underflows — the A.5 wall again. Ties are
  therefore broken by the detector's **score** (soft miss rate, weight 1e-6, below the reward's
  resolution). β = 0 is ranked by log E[BER], the same order without underflow.
- **Grid:** targets {power 1-sided, power 2-sided, kurtosis, CNN} × β ∈ {0.01, 1}, plus β = 0;
  JSR −40:4:0 dB. The plan said β ∈ {1, 10}, but BER stays below ~0.05 on this grid, so both would
  select the same "stealth first" point; 0.01 is the trading value.
- **Confirmation:** every returned θ is re-measured on 4096 fresh frames against all five detectors,
  with θ = 0 on the same frames.

**Results** (`artifacts/cgan/learned/run001/`, `summary.json`). Predictions P1–P3 were written down
before the runs:
- **Checks pass.** The learned jammer is never worse than white noise (0 violations in 105 runs: 99 + the 6 seed-check runs). A
  second seed at 3 JSRs × {kurtosis, CNN} agrees in all 6 pairs (largest |ΔP(det)| 0.009).
- **P3 holds: stealthy BER is 0 for every trained jammer against every detector.** Against the two
  power detectors no grid point is stealthy at all: at −40 dB they already read 0.06–0.11, above
  α + 2σ = 0.057, and BER there is 0.
- **The CNN is evadable, but only where evasion is harmless.**
  - The CNN-trained jammer holds the CNN at **0.03–0.09 from −40 to −20 dB**; white noise reads 0.87
    at −28 dB and 1.00 from −24 dB. At −20 dB one-sided power catches it at **1.00**.
  - It evades by going **out of band**: 0–1 % of its power is in-band at −20 and −16 dB (β = 0.01).
  - The matched filter rejects out-of-band noise, so BER stays **0** everywhere the CNN is fooled.
    The first errors (−12 dB, BER 7.9e-3) come with the CNN back at 0.97.
  - Evading the CNN and hurting the victim live in disjoint parts of the spectrum.
- **P1 fails as written: one-sided power is not blind to waveform shape.**
  - Near the false-alarm floor, the power-trained jammer lowers power's P(det) by moving out of band
    (0 % in-band at −32 dB): 0.146 vs 0.208 (white) at −32 dB, 0.084 vs 0.105 at −36 dB. There is no
    overlap with the victim, so frame power varies less.
  - Far above the floor it goes in-band and lowers P(det) by up to 0.19 (−20 dB: 0.79 vs 0.99). A
    larger cross term widens the spread of frame power. run001 already showed this (pulsed p = 1:
    0.77 vs noise 0.99 at −20 dB).
  - Neither effect buys BER at P(det) ≤ α. The 2026-09-18 chat claim that power is
    "waveform-blind" was wrong in that literal form.
- **Kurtosis cannot be dodged where the jammer is effective.** Every shaped jammer is invisible to
  kurtosis up to −20 dB. The first errors (−16 dB) come with kurtosis at 0.2–0.5. Gaussian noise
  Gaussianises the received amplitude whatever its spectrum, so this family has no way around it
  (a constant-envelope family was offered and not chosen, §4.4).
- **P2 exceeded: shaping buys 15.9 dB of effectiveness, not the predicted ≤ 9 dB.**
  - At β = 0 the jammer crosses BER 1e-3 at **−16.4 dB**, against −0.5 dB for white noise.
  - The gain is 99–100 % in-band power (up to 9 dB, the sps-8 processing gain) **plus**
    concentration in time: effective duty 0.06, i.e. all power in one of the 16 slots.
  - It also crosses ~3.4 dB before fixed pulsed p = 0.1 (−13 dB). But p = 1/16 is not in the fixed
    grid, so this is not a claim that learning beats Amuru's family.
- **At looser budgets (user request 2026-09-19: compare at every budget, not only at α).**
  `fig1_frontier.png` plots, per detector, the most BER each attack reaches while flagged in at most
  x of frames, for every x from 0 to 1, with the FAR dotted. `summary.json` `matched_ber` gives
  budgets 0.05 / 0.1 / 0.25 / 0.5.
  - **Power (either side): no attack causes an error below P(det) ≈ 0.83 (1-sided) / 0.89
    (2-sided).** At 30 dB SNR errors need the jammer within ~15 dB of the signal, and power flags
    half the frames from −27 dB.
  - **CNN:** fixed **pulsed p = 0.1 is the best attack at every budget** (BER from P(det) 0.38;
    2.6e-3 at 0.5). Gaussian shaped noise does no damage below 0.6–0.97. A structured in-band
    jammer is less visible to the CNN than any Gaussian shape.
  - **Kurtosis:** matched QPSK (p = 1) reaches BER 0.4 at P(det) ≈ 0.1, because a strong
    same-modulation jammer barely changes the amplitude distribution. Shaped β = 0 beats pulsed
    p = 0.1 below P(det) 0.9 (1.5e-3 at 0.5).
- **Consequence for D2:** a stealth-trained generator facing the CNN should be expected to find the
  same out-of-band evasion. "Stealthy GAN" means BER > 0 at P(det)_CNN ≤ α, which this control never
  reached.

**Jobs:**
- verify 2267049: 178/178 pass, 11 of them new. 2267046 ran out of memory on an 11 GB 2080 Ti, so
  `submit_verify.sh` now requires a 24 GB card.
- smoke 2267047 (flat, stopped at generation 1) → 2267050 (after the tie-break).
- array 2267051 (10 tasks): power and kurtosis 3–5 min each, CNN 58 and 65 min, seed check 19 min,
  about 2 GB RAM each.
- figures 2267090 (final; 2267087 and 2267089 were superseded renders).

**Outputs:**
- `fig1_frontier.png`: most BER at P(det) ≤ x, for x = 0..1, per detector;
- `fig2_transfer.png`: trained-against × evaluated-by;
- `fig3_shapes.png`: learned PSD and envelope, with in-band share and duty;
- `fig4_convergence.png`;
- `fig5_ber_vs_jsr.png`;
- `task{0..9}.json` + `summary.json`.

## 3.3f CGAN under detection — D1 plain GAN and D2 detector-aware GAN, all four detectors (2026-09-20, CNN 2026-09-23)

**User-directed build (this session), the actual step-2 (§3.4 D1/D2).** New/extended flat modules in
`cgan/`: `attacks.gan_tx` + a `gan` attack spec (the eval path on the 3-D scene link),
`attacks.ber_logprob`/`log_expected_ber` (differentiable log-domain BER), `detectors.soft_pdet`,
`models.load_generator`, `eval_gan.py` (places any generator on the D0 plane), `train_gan.py` (the D2
trainer), `gan_figures.py`. `verify.py` gains section 14. Step-1 and D0/D2a files untouched.

**Method (decided this session).**
- **Generator:** the reproduced Zhou CGAN generator (`models.Generator`), warm-started from `run001_G`
  (async). D2 does **not** re-run Zhou's adversarial losses — it fine-tunes by **direct gradient**
  (this reframes §4.2 Q11: the "which GAN recipe" question is moot for D2).
- **Objective (§2.8):** `−(log E[BER] − β·P_det_soft)`, power imposed by the hard equality projection
  (never a loss term). Effectiveness is `attacks.log_expected_ber` (log domain — E[BER] underflows at
  SNR 30 dB, the A.5/D2a wall); stealth is `detectors.soft_pdet` at the detector's calibrated α
  threshold. Because log E[BER] runs ~1e3 in the stealth region, **β sweeps geometrically
  {1, 10, 100, 1000}** (not D2a's {0.01, 1}); the comparison to D2a is made on the **measured**
  BER–P(det) plane, so it is objective-agnostic.
- **Threat model (decided this session).** Frozen, pre-trained detectors; generator frozen at
  deployment (no test-time query). **White-box** (train against the deployed detector) = attacker upper
  bound. Only the CNN has secret weights; power/kurtosis are analytic, so white-box ≡ black-box for
  them. A **transfer/grey-box** row (surrogate CNN, independent seed) would price the weight assumption;
  the black-box floor (D2a) and the NP ceiling bracket it. Transfer is a gated extension.
- **One generator per (target detector, β)**, JSR sampled over the active band [−32, 0] dB per step,
  Adam 2e-4, 400 steps. Placement + confirmation via `eval_gan.py` (reuses `baselines.measure`/`confirm`).

**Verified (job 2267202, `--baselines-only`, all pass incl. section 14):** the perfect-generator gan
path ≡ pulsed(1) async (14a); realised JSR exact per frame (14b); **the full D2 loss produces a finite,
nonzero gradient on all of G's parameters** — autograd flows through Sionna's `ApplyTimeChannel`, the
matched filter and the detector (14c, ‖grad‖ ≈ 2.7e4). This de-risked the white-box approach before
training.

**Result — the effectiveness/detectability wall holds for an 8.6M-parameter white-box generator too.**
- **D1 plain GAN (`run001_G`):** confirmed stealthy BER (max BER at P(det) ≤ α = 0.05, re-measured on
  4096 fresh frames) is **0 against every detector** — on the floor with noise/pulsed. Max BER when
  loud 0.44. The honest "detectability of a reproduced GAN jammer" baseline.
- **D2 (13 generators: eff + power-1s/2s + kurtosis × β{1,10,100,1000}):** confirmed stealthy BER
  **≈ 0 against every detector** (a few ~1e-5–8e-5 vs kurtosis, at the measurement floor). Every
  generator reaches max BER ~0.40 when loud (effectiveness-only crosses BER 1e-3 at ~−18 dB, on par
  with D2a's shaped control), so the zero is a **real tradeoff, not a broken optimiser**. **An
  8.6M-parameter neural generator with white-box gradients does no better than D2a's 48-parameter shaped
  control** — the wall is not a hypothesis-class limit. Consistent with the square-root-law (§2.1).
- **Cross-detector (kurtosis is the soft spot):** matched-QPSK reaches BER ≈ 0.4 at P(det)_kurtosis ≈
  0.1 (just outside the strict budget) — a strong same-modulation jammer barely changes the amplitude
  distribution (§3.3e). Against the CNN the Amuru pulsed jammer is the stealthiest classical attack.

**The CNN closes the set (2026-09-23): same answer, and now the gap is measured.** Seven more
generators (tasks 13–19, spec_cnn × β{1, 10, 100, 1000, 1e4, 1e5, 1e6}), trained **white-box against the
deployed CNN** through the straight-through colour LUT (§3.1) — no surrogate, so the attacker sees the
real weights.
- **Confirmed stealthy BER is exactly 0 for all seven**, at α = 0.05 on 4096 fresh frames, while max BER
  when loud is **0.377–0.433**. The generators work; they cannot be stealthy and effective at once.
- **The optimiser was genuinely given the boundary.** `P_det_soft` falls from ~0.99 to 0.17–0.31 during
  training (e.g. β = 1e5: 0.889 → 0.305 at comparable JSR), so this is a real tradeoff and not a dead
  gradient — the failure mode the per-target JSR band (§3.1) was introduced to avoid.
- **At matched BER the gain is real and is THE headline (user decision 2026-09-23).** Reporting only at
  the α budget is the wrong cut: there everything is 0 and no jammer can be distinguished from any
  other. Driven to the same BER = 3e-4. **⚠ CORRECTED 2026-09-26 — attribution.** This table was
  first read as "D1 vs D2 = detector-aware training alone". It is not: D1 → D2 also adds the **BER
  term**. The control for the detector term is the **β = 0 generator** (`eff`, same warm start,
  recipe and eval path, no detection term). The middle column is new:

  | detector | D1 plain GAN | β = 0 (BER only) | D2 vs CNN, β = 10 | D1 → β = 0 | β = 0 → D2 |
  |---|---|---|---|---|---|
  | spectrogram CNN | 1.000 | 0.977 | **0.802** | −2.3 pp | **−17.5 pp** |
  | power (1-sided) | 1.000 | 0.886 | **0.688** | −11.4 pp | −19.8 pp |
  | kurtosis | 0.987 | 0.086 | 0.104 | **−90.1 pp** | +1.8 pp |

  - **Only the CNN term moves its own detector.** The power- and kurtosis-TARGETED generators are no
    better than β = 0 on their own detector (one-sided power 0.861–0.910 vs 0.886; kurtosis 0.082–0.099
    vs 0.086). The old kurtosis cell, 0.060 "(β = 1)", was the **CNN-targeted** β = 1 generator's
    kurtosis score — the minimum across generators, not a kurtosis-targeted result. **Do not quote
    "−92.7 pp against kurtosis" as a detector-awareness gain:** −90 pp of it is BER training.
  - Power: β = 0 already reaches BER 3e-4 at −19.0 dB, so **~10.6 of the 12.8 dB saving is BER
    training**; the CNN term adds ~2.2 dB.
  - At 400 steps these generators are **under-trained** (next bullet), so all of the above is
    being re-measured at 4000 steps (run002, 2026-09-26).

  D2 reaches that BER at **−21.2 dB instead of −8.4 dB — 12.8 dB less power** (split as above). Binomial SE is
  ±1.8 pp at 512 frames, so the CNN result is ~11σ; **differences below ~4 pp are not meaningful.**
  Matched-BER points are interpolated on the 1 dB JSR grid, linearly in log BER.
- **Imitation start and training budget — ablation 2026-09-26 (user: "we avoid inductive bias but
  train the GAN to imitate the waveform?").** No D2 generator had ever been trained without the
  `run001_G` warm start, and 400 steps was never checked for convergence. `train_gan.py --init random`
  (Zhou's architecture, fresh weights) and `--steps 4000`, tasks 0 / 9 / 14, same seeds, same eval
  path. Matched BER 3e-4, SNR 30 dB, one seed per arm (4-seed replicates below the list):

  | generator | CNN | power 1s | kurtosis | JSR [dB] | max BER |
  |---|---|---|---|---|---|
  | β = 0, warm, 400 *(reported)* | 0.977 | 0.886 | 0.086 | −19.0 | 0.40 |
  | β = 0, warm, 4000 | **0.601** | 0.806 | 0.071 | −22.1 | 0.33 |
  | β = 0, cold, 400 | 1.000 | 0.995 | 0.252 | −16.4 | 0.27 |
  | β = 0, cold, 4000 | 0.988 | 0.767 | 0.104 | −20.9 | 0.19 |
  | CNN β = 10, warm, 400 *(headline)* | 0.802 | 0.688 | 0.104 | −21.2 | 0.38 |
  | CNN β = 10, warm, 4000 | 0.553 | 0.638 | 0.114 | −22.2 | 0.33 |
  | CNN β = 10, cold, 400 | 1.000 | 0.951 | 0.246 | −17.6 | 0.24 |
  | CNN β = 10, **cold, 4000** | **0.105** | 0.600 | 0.080 | **−23.0** | **0.024** |
  | kurtosis β = 1, warm 400 / warm 4000 / cold 400 / cold 4000 | 0.984 / 0.674 / 1.000 / 1.000 | 0.910 / 0.823 / 0.997 / 0.983 | 0.092 / 0.073 / 0.197 / 0.183 | −18.5 / −22.0 / −16.8 / −17.8 | 0.41 / 0.35 / 0.27 / 0.29 |

  1. **400 steps under-trains.** With 4000 steps the β = 0 generator — BER only — beats the 400-step
     CNN-targeted headline generator *against the CNN* (0.601 vs 0.802). The headline was measured
     before convergence; re-run as run002 (warm, 4000 steps, all 20 tasks) + E2 on it.
  2. **The imitation start is an inductive bias toward the victim's modulation, and it trades
     stealth for damage capacity.** At 400 steps cold is worse everywhere (a head start, nothing
     more). At 4000 steps the cold CNN-targeted generator finds a jammer the warm one never does:
     CNN 0.105 at BER 3.7e-4 (warm 0.553), 0.135 at BER 1.3e-3 — but its BER saturates near 2 %
     even at +10 dB (warm 20–30 %). The warm start keeps G QPSK-like, which is what makes it both
     damaging when loud and visible to the CNN. One-sided power still flags it (0.600): power at
     matched BER is set by JSR, which no shaping hides.
  3. **The cold kurtosis-targeted run stalled** (kurtosis 0.18 at 4000 steps vs 0.07 warm).
  4. **α budget unchanged:** confirmed stealthy BER is still 0 for every arm. Cold CNN 4000 has its
     CNN stealth edge near −25 dB and BER onset at −23 dB — the gap narrows from ~30 dB to ~2 dB
     against the CNN, but one-sided power closes it.
  5. **Seeds (r0–r3, 4000 steps, tasks 0/14; `warm002_4k*`, `cold002_4k*`) confirm it.** Mean ± std:

     | | CNN | power 1s | kurtosis | JSR [dB] | max BER |
     |---|---|---|---|---|---|
     | β = 0, warm | 0.633 ± 0.084 | 0.830 ± 0.038 | 0.077 ± 0.008 | −21.8 ± 0.7 | 0.33 |
     | β = 0, cold | 0.995 ± 0.005 | 0.837 ± 0.079 | 0.111 ± 0.024 | −20.3 ± 0.4 | 0.20 |
     | CNN β = 10, warm | 0.480 ± 0.074 | 0.675 ± 0.082 | 0.110 ± 0.017 | −22.4 ± 0.2 | 0.30 |
     | CNN β = 10, cold | **0.146 ± 0.024** | 0.651 ± 0.085 | 0.078 ± 0.015 | −22.8 ± 0.3 | **0.030** |

     The warm and cold CNN arms do not overlap (warm ≥ 0.359, cold ≤ 0.168). The detector term is
     −85 pp from a cold start but only ~−15 pp (≈ 2.7 SE) from the imitation start. **Why the cold
     jammer saturates:** it is pulsed — 97 % of its power in 5 % of symbols (warm 82 %, D1 13 %), so
     BER ≤ ~duty/2 — which is Amuru's low-power optimum and exactly what the CNN band (−48, −16) dB
     rewards; it never sees high JSR. **It is not the weaker jammer:** BER at matched CNN P(det) 0.25 /
     0.5 / 0.9 is 1.5e-3 / 2.6e-3 / 3.3e-3 cold vs 5.5e-6 / 4.9e-4 / 2.2e-3 warm (also ≥ warm against
     power and kurtosis). Warm only wins at JSR ≥ 0 dB, where every detector flags 100 %. Single-run
     variance is itself ~0.07 in CNN P(det): run002's task 14 re-run of the warm β = 10 recipe reads
     0.611 against `warm002_4k`'s 0.553 (GPU non-determinism over 4000 steps), so **any one generator's
     number carries ~±0.08, and picking the best β out of seven is selection on that noise.**
- **run002 — the headline re-measured at 4000 steps (2026-09-26, jobs 2273985 train / 2273986 eval /
  2273987 D1).** All 20 tasks, warm start, otherwise the run001 recipe. Matched BER 3e-4, SNR 30 dB
  (`artifacts/cgan/gan/run002/`; checkpoints symlinked to net_scratch):

  | generator | CNN | power 1s | power 2s | kurtosis | JSR [dB] |
  |---|---|---|---|---|---|
  | D1 plain | 1.000 | 1.000 | 1.000 | 0.984 | −8.2 |
  | **β = 0 (BER only)** | **0.579** | 0.834 | 0.868 | 0.078 | **−22.1** |
  | CNN β = 1 | **0.413** | 0.791 | 0.855 | 0.076 | −22.7 |
  | CNN β = 10 | 0.611 | 0.663 | 0.813 | 0.101 | −22.3 |
  | CNN β = 100 / 1000 / ≥ 1e4 | 0.994 / 1.000 / 1.000 | | | | −20.4 / −17.4 / ≤ −14.9 |
  | power 1s β = 1–100 | 0.50–0.65 | 0.80–0.83 | | 0.08 | −22.0 to −22.2 |
  | power 2s β = 1–100 | 0.61–0.62 | 0.82–0.83 | 0.85–0.89 | 0.08 | −22.0 to −22.1 |
  | kurtosis β = 1–100 | 0.60–0.68 | 0.81–0.82 | | 0.070–0.078 | −22.0 to −22.1 |

  - **D1 → β = 0 is the effect:** −42 pp against the CNN, −17 pp against power, −91 pp against
    kurtosis, 13.9 dB less power. **β = 0 → best CNN β is the detector term:** −17 pp (β = 1),
    +0.6 dB; β = 10 is within single-run noise of β = 0 here (seeds above: −15 pp on average).
  - Power/kurtosis targets: nothing beyond β = 0 on their own detector, as at 400 steps. β = 1000
    breaks every target (the kurtosis β = 1000 run needs −11.8 dB).
  - Limit (iii) below still holds: β has an interior optimum, now at β = 1 for the CNN.
  - Regression: E2's 30 dB level reproduces this eval — 5544 generator P(det) pairs, 12 with
    |z| > 3 (15 expected by chance), none > 4, mean z −0.016. `regress_snr30.py` prints FAIL on 3 CNN
    points (0.090–0.104 vs its 0.088 tolerance) because that tolerance is 4σ of ONE 512-frame
    estimate, not of a difference of two (√2 larger, 0.125) — a latent gate bug, §4.3 housekeeping.
    The array ran alongside the gate, not after it (deadline).

  Jobs: 2273958 verify · 2273959/2273960 cold 400 · 2273967/2273969 cold 4000 · 2273968/2273970 warm
  4000 · seeds r1–r3 2273989–2274000 · run002 2273985 (train) / 2273986 (eval) / 2273987 (D1) ·
  E2 on run002 2273988. Ablation checkpoints live on net_scratch (`/itet-stor/rrahman/net_scratch/
  cgan_gan/`), symlinked from `artifacts/cgan/gan/` — home quota (§C.4). `run001/` untouched;
  `train_gan.py` refuses to write any non-original recipe into it.
- **run003 — the paper method: random init, 4000 steps (2026-09-27, jobs 2274753–2274759,
  2274789–2274792, 2274822/2274823).** All 20 tasks, `--init random`. Matched BER 3e-4, SNR 30 dB:

  | generator | CNN | power 1s | power 2s | kurtosis | JSR [dB] | BER @ +10 dB |
  |---|---|---|---|---|---|---|
  | β = 0 (control) | 0.999 | 0.947 | 0.945 | 0.081 | −20.4 | 0.090 |
  | **CNN β = 1** | **0.078** | **0.574** | 0.787 | 0.090 | **−23.0** | 0.018 |
  | **CNN β = 10** *(headline, rule)* | **0.113** | 0.615 | 0.822 | 0.113 | **−23.0** | 0.018 |
  | CNN β = 100 / ≥ 1000 | 0.697 / 1.000 | 0.635 / 1.000 | | | −22.0 / ≥ −6.0 | |
  | power 1s β = 1–100 | ≥ 0.995 | 0.82–0.93 | | 0.11–0.15 | −18.8 to −20.3 | |
  | power 2s β = 1 | 0.959 | 0.675 | 0.806 | 0.126 | −21.5 | 0.056 |
  | kurtosis β = 10 | 1.000 | 0.979 | 0.978 | 0.068 | −19.2 | 0.071 |

  - **The CNN-targeted generator is the best generator on every detector at once** (vs run002's best,
    CNN β = 1 from the imitation start: CNN 0.413, power 1s 0.791, power 2s 0.855, JSR −22.7).
  - **Detection term vs its control: −89 pp against the CNN, 2.6 dB less power.** The random-init
    control is itself CNN-visible (0.999) — from random weights the damage term alone finds a
    CNN-visible waveform; the detection term is what steers it away.
  - Power/kurtosis targets: no gain from random init (worse than run002 on power, equal on kurtosis).
    Only the learned detector is a target worth training against. **Why power in particular cannot be
    trained against here:** with fixed power every training frame is forced to a JSR from the band, so the
    generator controls only the SHAPE, which a power detector does not see; the power is chosen afterwards
    by the evaluation sweep. Letting the generator learn its power is run005 / the parked "king GAN" (§4.3).
  - β = 10 has five runs in all (0.105, 0.160, 0.168, 0.152 from the seed ablation; 0.113 here):
    **0.14 ± 0.03** — quote that, not one run.
- **run004 — grey-box surrogate (2026-09-27, jobs 2274864 verify · 2274865 CNN · 2274866 train ·
  2274867/2274868 eval).** Our own CNN, same Li et al. recipe, seed 12 (`train_spectrogram_cnn.py
  --out-dir ../artifacts/cgan/baselines/surrogate_s12`; the trainer refuses to write a non-default seed
  over the deployed detector); it replicates the deployed one (99.94 % at +10 dB). Random-init CNN β 1/10
  trained against it (`train_gan.py --detector-dir`), evaluated against the DEPLOYED CNN at matched BER
  3e-4: 30 dB β 1 / 10 grey 0.104 / 0.107 vs white 0.078 / 0.113; 15 dB 0.056 / 0.049 vs 0.047 / 0.039 —
  **no transfer gap** (inside ±0.03 run-to-run); JSR −23 dB unchanged. The attacker needs the detector's
  type and recipe, not its weights. **Untested: a detector of a different architecture (black-box).**
- **run005 — the generator CHOOSES its power and is trained on FRAME damage (2026-09-27, user "(a)+(b)").**
  `train_gan.py --init random --power learned --jsr-cap 0 --jsr-init -40 --damage per` — a learned global
  gain under a hard per-frame cap (`attacks.learned_power`, `channel.receive(..., scale=False)`), exact
  log E[PER] (`attacks.log_expected_per` / `_log_mean_per`). Tasks 0, 1–4 (power 1s β 1–1000), 13–16
  (CNN β 1–1000). Evaluated at its OWN power by `outlier_alarm.py` (8192 frames, thresholds from
  run003_per). Jobs 2274914 verify · 2274915 smoke · 2274916 train (+2274928 task 16) · 2274933 eval.
  (First attempt 2274904 went NaN: `torch.where` differentiated the unused exact branch, 0·∞ — fixed with
  a safe dummy input; training now fails fast on a non-finite loss. Eval 2274929/2274931 failed on smoke
  checkpoints — `snr_ablation.generators` now skips them — and on an NFS race.)

  | 30 dB, own power | JSR | PER | CNN P(det) | power 1s P(det) |
  |---|---|---|---|---|
  | β = 0 | −3.3 | 0.79 | 1.00 | 1.00 |
  | power 1s β 1–1000 | −3.1…−4.2 | 0.93–0.99 | 1.00 | 1.00 |
  | CNN β 1 / 10 | −1.8 / −2.2 | 0.43 / 0.39 | 1.00 | 1.00 |
  | **CNN β 100** | **−4.0** | **0.35** | **0.000** | 1.00 |
  | CNN β 1000 | −26.2 | 0.00 | 0.25 | 0.59 |

  **With free power the jammer goes loud and is caught**, except that CNN β 100 is a total CNN blind spot
  at 30 dB (at 15 dB the CNN catches it, 0.9999; power and kurtosis always 1.00). Frames per extra alarm
  vs one-sided power 0.8–1.04 — the ≈ 1 floor. **Partly a loss artifact:** the sigmoid detection term
  saturates, so once log E[PER] (≈ −3900 at −40 dB) has pushed power past the detector's transition no
  β pulls it back (training logs: every power-target run at soft P(det) 1.0 by step 1000). Fix = D5 (§3.4).
- **Linear view — run003 over the whole damage range (2026-09-28; `shadow_figures.py --linear 0`, data
  `snr_ablation/shadow_run003/shadow_0.json` = the ideal link, 30 dB, deployed detectors).** On a linear
  BER axis every non-genie jammer lies flat on BER ≈ 0 and rises only at P(det) → 1; the genie is a
  vertical line at the FAR up to BER 1.0. **Max damage with P(det) ≤ 0.5, BER / PER:** against one-sided
  power **0 / 0 for every non-genie jammer**; against the CNN, GAN-vs-CNN (β 10) **0.0031 / 0.648**,
  Amuru 0.0028 / 0.258, noise, matched QPSK, β = 0 and GAN-vs-power 0 / 0; against kurtosis matched QPSK
  0.41 / 1.00 (kurtosis cannot see a QPSK-like jammer) and the generators 0.002–0.003 / 0.50–0.65. So at
  P(det) 0.5 the CNN-targeted generator breaks two thirds of frames under the CNN with ~1 bit error
  each, and nothing breaks under the power threshold.
- **Three limits on that gain, all load-bearing.** (i) **It lives only at low BER** — at 3e-4 the
  spread is 0.42–1.00, but by BER 3e-3 every detector reads 1.000 for every jammer. (ii) **A classical
  jammer still evades the CNN better**: Amuru pulsed sits at **P(det)_CNN = 0.421**, below every
  learned generator. **At SNR 30 dB only** — E2 found this limit holds from ~22 dB up; at 10–20 dB the
  CNN-targeted generator is the stealthier one (§3.3g 1b). The GAN wins on kurtosis and power, *not* on the detector it was trained against.
  (iii) **β has an interior optimum** — β = 1–10 produces the gain, 100–1000 erodes it, and β ≥ 1e4
  destroys it (those generators need *more* power for the same BER **and** are caught just as often).
  Over-weighting stealth produces a worse jammer, not a stealthier one.
- **The α-budget zero is the limit, not the finding.** Inside the detector's own false-alarm rate
  confirmed BER stays 0 for all 24 non-genie jammers; the CNN's stealth edge is ~−45 dB (best
  CNN-targeted generator −43.5 dB vs −46.7 dB for noise, i.e. ≤ 3 dB) while BER only becomes
  measurable near −16 dB — a **~30 dB gap** (at SNR 30 dB; the gap grows with SNR at ~0.8 dB per dB,
  §3.3g finding 4). All four detectors agree there, under the attacker's best
  case. State this as the boundary of the gain above, **not** as an impossibility headline.

**Jobs:** 2267202 verify · 2267610 train array 0–12 (task 12 `TaskProlog` transient → 2267624) ·
2267625 D1 plain eval · 2267626 eval array 0–12 (task 12 `TaskProlog` → 2267639).
**CNN round (2026-09-23):** 2268283 verify (exit 0) · 2268338 train array 13–19 (task 19 `TaskProlog`
→ 2269798) · 2269799 eval array 13–19 · 2269800 figures + table. Generators + eval
JSONs + `fig_{ber_vs_jsr,pdet_vs_jsr,frontier}.png` in `artifacts/cgan/gan/run001/`.
**Report page, matched-BER framing, all four detectors (2026-09-23, current):**
<https://claude.ai/artifact/2fRAFrMoYAfCZWPe1KvTfV> — private, **not yet shared with the supervisor**.
It supersedes the 2026-09-20 page <https://claude.ai/artifact/PBXBr7EQQ3huK7rocvSMxX>, which predates
the CNN rows and leads with the α-budget framing; do not send that one.

**Caveats / not yet done.**
- The matched-BER table above is derived from the eval JSONs. `cgan/snr_figures.py` `summary()` prints
  it at every SNR including 30 dB, from E2's re-measurement (§3.3g); `gan_figures.py` `summary()` prints
  only the α-budget table, and its `BER_REF` is 1e-3 while this table is quoted at 3e-4.
- ~~The frontier is drawn at the fixed α threshold swept over JSR, not the full ROC over all budgets.~~
  **Withdrawn 2026-09-24:** a fixed α with P(det) swept over [0, 1] *is* the view §2.7 asks for; moving
  α is the separate question D2b answers.
- `fig_frontier` / `fig_pdet_vs_jsr` draw Amuru in `#7a5195`, which fails the palette's normal-vision
  check against the CNN-targeted `#882255` (ΔE 12.6 < 15, §3.3g figures) — the two series the CNN story
  compares. E2's figures use orange; these are not yet redrawn.
- Single-round, frozen detectors, K = 1.

**Cluster note.** A `PENDING (BadConstraints)` with a ~24 h estimate was a **CPU** mismatch — the jobs
requested `--cpus-per-task=4` while the only node with a free GPU of the allowed type (tikgpu04, 2 free
TITAN RTX) had 2 free CPUs; dropping to `--cpus-per-task=2` scheduled them at once
(`submit_train_gan.sh` / `submit_eval_gan*.sh` now use 2). Migrating to INFK was re-examined and
rejected again 2026-09-21, now with measurements — see §1.5.

## 3.3g E2 — the noise ablation on the live models: 30 dB is near the WORST place to measure the gain (2026-09-23)

**E2b — on/off switching, the outlier alarm and the damage metric (2026-09-27; `cgan/outlier_alarm.py`,
jobs 2274861 smoke · 2274862 grid · 2274869 grid with PER; `artifacts/cgan/outlier_alarm/run003{,_per}/`).**
Why: once power is the jammer's choice it is free per frame, and switching matched QPSK on/off (loud
+15 dB, BER 0.45, PER 1.0, on a fraction f of frames) competes with every constant-power waveform.
Thresholds at α ∈ {0.05, 0.01, 1e-3, 1e-4} from 200k clean frames per level; realised FAR on 50k fresh
clean frames on target (1e-4 → 4e-5–1.6e-4). 512 frames per sweep point, so P(det) ≲ 2e-3 is
unresolved at α′ = 1e-4.

1. **At matched average BER 3e-4, on/off wins or ties.** It is on 0.07 % of frames: P(det) = FAR + f,
   i.e. 0.049 / 0.0014 / 0.0008 at α = 0.05 / 1e-3 / 1e-4 on either detector. run003 CNN β = 10 at
   30 dB: CNN 0.125 / 0.005 / ≤ 0.002, one-sided power 0.64 / 0.58 / 0.55. The outlier alarm alone does
   not rescue a constant-power jammer against power: it raises every frame's power.
2. **At matched frame error rate (PER, uncoded frames with ≥ 1 bit error) the CNN-targeted generator
   beats on/off against the CNN** — at every α for PER ≥ 0.1, and at α′ = 1e-3/1e-4 for every PER:
   15 dB PER 0.1: 0.047 / 0.002 / 0.000 vs on/off 0.144 / 0.101 / 0.100; 30 dB PER 0.1: 0.100 / 0.004 /
   0.002 vs 0.144 / 0.101 / 0.100; 30 dB PER 0.5: 0.25 / 0.019 / 0.003 vs 0.52 / 0.50 / 0.50. On/off must
   be loud on every frame it breaks, and each such frame trips every alarm.
3. **Against one-sided power it depends on SNR:** at 15 dB the generator wins at the outlier alarms
   from PER 0.1 up (PER 0.1: 0.04 / 0.012 vs 0.101 / 0.100) and loses at α = 0.05; at 30 dB on/off wins
   at every α and PER (e.g. PER 0.1: 0.60 / 0.53 / 0.51 vs 0.145 / 0.101 / 0.100) — the constant power
   is visible on a clean link.
4. **So the damage metric decides the winner:** average BER rewards concentrating errors (on/off
   destroys few frames), PER rewards spreading them (the generator breaks 10 % of frames with BER 4e-4,
   i.e. ~1 bit error per broken frame). PER is the throughput loss of an uncoded link with
   retransmission. **Open:** on a link with error correction the generator's 1–2 errors per frame are
   likely correctable and on/off would win again — needs the per-frame error histogram (§4.3 E2
   follow-on 1). Paper framing (proposed, not yet decided by the user): report BER and PER frontiers
   with the on/off hull as a baseline, and state that the stealth advantage holds for frame-level
   damage against the learned detector.
5. **Damage per extra alarm — the power-free summary (user 2026-09-27: power is secondary).**
   damage / (P(det) − FAR), best over power, excess ≥ 2σ (`outlier_figures.py` `slopes`,
   `damage_per_extra_alarm.json`, `fig_damage_per_extra_alarm.png`; frontiers
   `fig_frontier_ber_vs_per_{spec_cnn,power_one_sided}.png`, `submit_outlier_figures.sh`). **Bits per
   extra alarm:** loud matched QPSK 0.47, noise 0.32, β = 0 0.23, Amuru 0.15–0.16, run003 CNN β 10 0.14
   (15 dB, CNN) / 0.026 (30 dB, and vs power). **Frames per extra alarm:** vs the CNN run003 30 (15 dB) /
   3.1 (30 dB) vs classical ≤ 4.3 / 1.5; **vs one-sided power every jammer 1.05–1.3** — the energy floor.
   Only 15 and 30 dB measured; whether the floor rises at low SNR is open (§4.2).

**E2 on run003 — the paper method's numbers (2026-09-27; which of them lead depends on the story, §3.1; jobs 2274792 + 2274823 noiseless;
`artifacts/cgan/snr_ablation/run003/`).** P(det) CNN / one-sided power at matched excess BER 3e-4 (JSR):

| SNR | β = 0 control | CNN β = 1 | **CNN β = 10** | power 2s β = 1 | kurtosis β = 10 | Amuru pulsed | noise |
|---|---|---|---|---|---|---|---|
| 5 | 0.05 / 0.13 (−22) | 0.05 / 0.07 (−24) | 0.05 / 0.07 (−24) | 0.06 / 0.09 (−22) | 0.05 / 0.17 (−20) | 0.08 / 0.22 (−17) | 0.99 / 1.00 (−4) |
| 10 | 0.10 / 0.28 (−21) | 0.06 / 0.13 (−23) | 0.05 / 0.13 (−24) | 0.06 / 0.16 (−22) | 0.08 / 0.40 (−19) | 0.11 / 0.62 (−16) | 1.00 / 1.00 (−2) |
| 15 | 0.15 / 0.55 (−21) | 0.05 / 0.23 (−23) | **0.04** / 0.25 (−23) | 0.08 / 0.31 (−22) | 0.14 / 0.71 (−19) | 0.17 / 0.86 (−15) | 1.00 / 1.00 (−2) |
| 20 | 0.32 / 0.77 (−21) | 0.06 / 0.40 (−23) | **0.07** / 0.40 (−23) | 0.16 / 0.49 (−22) | 0.47 / 0.93 (−19) | 0.25 / 0.90 (−15) | 1.00 / 1.00 (−2) |
| 25 | 0.85 / 0.87 (−21) | 0.08 / 0.54 (−23) | **0.06** / 0.54 (−23) | 0.54 / 0.66 (−21) | 0.89 / 0.98 (−19) | 0.34 / 0.92 (−15) | 1.00 / 1.00 (−2) |
| 30 | 1.00 / 0.94 (−21) | 0.10 / 0.62 (−23) | **0.11** / 0.61 (−23) | 0.97 / 0.68 (−21) | 1.00 / 0.99 (−19) | 0.45 / 0.95 (−14) | 1.00 / 1.00 (−2) |
| 35 | 1.00 / 0.96 (−20) | 0.21 / 0.65 (−23) | 0.21 / 0.63 (−23) | 1.00 / 0.72 (−21) | 1.00 / 0.99 (−19) | 0.47 / 0.96 (−15) | 1.00 / 1.00 (−2) |
| 40 | 1.00 / 0.98 (−20) | 0.35 / 0.65 (−23) | 0.36 / 0.64 (−23) | 1.00 / 0.74 (−21) | 1.00 / 1.00 (−19) | 0.46 / 0.95 (−15) | 1.00 / 1.00 (−2) |

1. **The CNN barely sees the CNN-targeted generator at any SNR up to 30 dB** (≤ 0.11, α = 0.05);
   the control is caught from 25 dB up. Largest effect of the detection term: −89 pp at 30 dB.
2. **It beats the classical Amuru pulsed jammer on both detectors at every SNR ≥ 10 dB**, at ~8 dB
   less power. The 400-step "a classical jammer evades the CNN better" limit (§3.3f (ii)) is gone.
3. **Boundary:** confirmed picks vs the CNN at 5–25 dB reach BER 4e-4–3.9e-3 at P(det) 0.065–0.075
   and fail the α + 2σ confirmation narrowly; one-sided power still flags 0.25–0.64 of frames at
   matched BER from 15 dB up. Stealthy BER inside the strict budget stays 0.
4. The 0 dB row (JSR −40 for β = 10) is the low-SNR artifact of finding 5 below, not a result.
5. 30 dB regression vs run003's eval: 5544 P(det) pairs, 6 with |z| > 3 (15 by chance), none > 4.

**E2 on run002 (2026-09-26, job 2273988; `artifacts/cgan/snr_ablation/run002/`) — supersedes the
headline numbers of the 2026-09-23 run below, which compared 400-step generators against D1.** Same
protocol (Tier 1: evaluated, not retrained; detectors re-calibrated per level). P(det) CNN / one-sided
power at matched excess BER 3e-4, JSR in brackets:

| SNR | D1 plain | β = 0 | CNN β = 1 | CNN β = 10 | power 1s β = 1 | kurtosis β = 1 |
|---|---|---|---|---|---|---|
| 5 | 0.17 / 0.86 (−11) | 0.06 / 0.10 (−23) | 0.06 / 0.09 (−24) | 0.06 / 0.09 (−23) | 0.05 / 0.10 (−23) | 0.05 / 0.11 (−23) |
| 10 | 0.67 / 1.00 (−9) | 0.06 / 0.25 (−22) | 0.04 / 0.22 (−23) | 0.06 / 0.17 (−22) | 0.07 / 0.25 (−22) | 0.06 / 0.23 (−22) |
| 15 | 0.94 / 1.00 (−8) | **0.09** / 0.45 (−22) | 0.09 / 0.41 (−23) | 0.07 / 0.26 (−22) | 0.07 / 0.44 (−22) | 0.09 / 0.47 (−22) |
| 20 | 1.00 / 1.00 (−8) | 0.14 / 0.64 (−22) | 0.12 / 0.62 (−23) | 0.12 / 0.53 (−22) | 0.15 / 0.63 (−22) | 0.16 / 0.64 (−22) |
| 25 | 1.00 / 1.00 (−8) | 0.27 / 0.75 (−22) | **0.19** / 0.72 (−23) | 0.28 / 0.59 (−22) | 0.29 / 0.74 (−22) | 0.39 / 0.75 (−22) |
| 30 | 1.00 / 1.00 (−8) | 0.59 / 0.81 (−22) | **0.39** / 0.75 (−23) | 0.55 / 0.65 (−22) | 0.58 / 0.81 (−22) | 0.67 / 0.78 (−22) |
| 35 | 1.00 / 1.00 (−8) | 0.84 / 0.82 (−22) | **0.66** / 0.83 (−23) | 0.85 / 0.69 (−22) | 0.82 / 0.81 (−22) | 0.89 / 0.83 (−22) |
| 40 | 1.00 / 1.00 (−8) | 0.96 / 0.84 (−22) | 0.87 / 0.84 (−23) | 0.97 / 0.69 (−22) | 0.95 / 0.83 (−22) | 0.96 / 0.85 (−22) |

1. **BER training does the heavy lifting at every SNR.** D1 → β = 0: −85 pp against the CNN at 15 dB
   (0.94 → 0.09), −86 at 20 dB, −41 at 30 dB; 13.9 dB less power at every level.
2. **The detector term's own gain is SNR-dependent and CNN-only:** β = 1 vs β = 0 is 0 at ≤ 20 dB
   (both near the floor), −8 pp at 25, −20 at 30, −18 at 35, −9 at 40. The CNN β = 10 generator buys
   one-sided power instead (−11 to −19 pp vs β = 0 at 15–35 dB) — single-run, ±0.08.
3. **Against power, nothing hides at matched BER:** one-sided power flags 0.25–0.45 of frames at
   10–15 dB and ~0.8 at 30 dB for every BER-trained generator.
4. Findings 4–5 of the run below (edge-vs-onset mechanism, CNN off-design at low SNR) are about the
   classical rows and the CNN, not the generators, and are not re-derived here. Figures:
   `CGAN_RUN=run002 sbatch submit_snr_figures.sh` (job 2274104).

**The supervisor's mandated primary ablation** (§B.2: *"Ablation: parameter study, increase noise and
see what happens"* + *"Noise level, ε, change exponentially"*), finally run on the models the paper
leads with. It had been discharged only on the frozen sim08 OFDM stack (§3.3c) while the whole
D-series sat at the single pinned level `link.SNR_DB = 30` — Zhou's protocol value, which §3.3d then
made load-bearing by power-controlling T to 30 dB at R so one detector calibration serves every drop.

**Setup.** 10 levels — SNR **0:5:40 dB plus a noiseless anchor** (linear in dB *is* exponential in
noise power, which is the literal mandate; the range overlaps sim08's 5–30 dB so §3.3c stays
comparable). Per level: **every detector re-calibrated on 20 000 clean frames at that level**
(`calibrate_snr.py`) — thresholds are quantiles of clean frames, so without this α stops being α and
the comparison stops being matched. **The CNN's weights stay the deployed 30 dB ones** and only its
colour scale and threshold are re-fitted (user decision; framing and limits below). Then the full
classical envelope (7 attacks) and **all 21 generators** (D1 + the 20 D2 checkpoints) over the same
66-point JSR grid, 512 frames per point, confirmation on 4096 — `baselines.measure/confirm` and
`eval_gan.sweep` reused unchanged. **Tier 1 is EVALUATION ONLY**: generators trained at 30 dB are
*evaluated* elsewhere, so this measures **transfer**, not achievability (Tier 2 gated, §3.4).
Jobs: 2270447 / 2270448 verify (incl. new §15) · 2270449 task 6 · 2270556 array 0–5,7–9 (~15–30 min
each, all parallel). Outputs `artifacts/cgan/snr_ablation/run001/` (10 JSONs + 4 figures).

**0. The 30 dB level reproduces §3.3f — the regression gate passed** (`regress_snr30.py`). P(det):
**0 of 264 points outside a 4σ binomial band in every one of the 28 rows** (7392 comparisons). BER
deviations are **unbiased scatter** (389 positive / 492 negative, mean −0.004 band-units), which is the
test that matters — a regression is one-signed. At §3.3f's own operating point (BER 3e-4) the CNN row
re-measures **1.000 → 0.838 (−16.2 pp) at 12.9 dB less power** against §3.3f's −19.8 pp at 12.8 dB: the
power saving reproduces exactly and 3.6 pp is inside the ±4 pp resolution §3.3f itself states.

**1. THE RESULT — the matched-BER stealth gain is far larger off 30 dB, and 30 dB is near its worst
end.** P(det) against the CNN at matched excess BER 3e-4, D1 plain GAN vs D2 CNN-targeted β = 10:

| SNR [dB] | 0 | 5 | 10 | 15 | 20 | 25 | **30** | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| D1 plain GAN | 0.045 | 0.122 | 0.590 | 0.936 | 0.998 | 1.000 | **1.000** | 1.000 | 1.000 |
| D2 vs CNN, β = 10 | 0.039 | 0.059 | **0.055** | **0.084** | 0.226 | 0.487 | **0.838** | 0.965 | 1.000 |
| Δ [pp] | −0.6 | −6.3 | **−53.5** | **−85.1** | −77.2 | −51.3 | **−16.2** | −3.5 | 0.0 |
| JSR D2 needs [dB] | −40.0 | −21.9 | −21.4 | −21.2 | −21.2 | −21.0 | −21.1 | −21.1 | −20.9 |

**The gain peaks at −85.1 pp at 15 dB SNR, five times the 30 dB value, and is gone by 40 dB.** (A
reduction in detectability, not undetectability: at 15 dB D2 is still caught 8.4 % of the time against
α = 5 % — finding 5 is the strict version.) At
realistic link SNRs (10–20 dB) detector-aware training cuts P(det) from 0.59–1.00 to 0.06–0.23 at
matched BER. **The D-series measured its headline where it is nearly weakest**, because Zhou's protocol
fixes SNR 30 dB — the one number in the whole setup that was inherited rather than chosen.
**The power saving is SNR-invariant: 12.3–12.9 dB across 10–40 dB**, so §3.3f's "12.8 dB less power"
is robust as stated.

**1b. §3.3f limit (ii) holds only above ~22 dB.** §3.3f carried "a classical jammer still evades the
CNN better" (Amuru pulsed 0.421 vs the CNN-targeted GAN). Across SNR at matched excess BER 3e-4:
Amuru 0.092 / 0.158 / 0.297 / 0.387 / 0.429 at 10 / 15 / 20 / 25 / 30 dB, against D2 0.055 / 0.084 /
0.226 / 0.487 / 0.838. **At 10–20 dB the learned generator is the stealthier of the two — against the
very detector it was trained on — while needing ~6 dB less power**; Amuru wins from 25 dB up; at 0–5 dB
they are within noise. One of §3.3f's three load-bearing limits is therefore a high-SNR statement.
**But only at low damage** (seen 2026-09-24 in `artifacts/cgan/snr_ablation/run001/email/email_fig1_tradeoff.png`, the whole 15 dB trade-off
curve against the CNN): at 15 dB, D2 is the less detectable of the two up to BER ~1e-3, and Amuru pulsed wins
from ~2e-3 up. At BER 1e-2 Amuru is flagged in ~0.2 of frames, D2 in ~0.99. So quote "the learned jammer
beats the classical one" with both qualifiers: SNR ≤ 20 dB **and** BER ≲ 1e-3.

**2. The stealth edge is NOT SNR-independent — the detector sharpens as the link cleans.** The loudest
JSR still inside α falls **−21 → −49 dB (power 1-sided) and −16 → −48 dB (CNN) over SNR 0 → 40 dB**,
about **−0.7 dB per dB**. This **refutes the mechanism proposed when the ablation was specified**: the
clean statistic was expected to be signal-dominated and therefore flat in JSR. It is noise-dominated —
the clean statistic's variance falls with N₀, so a cleaner link buys the *defender* precision. The
attacker's problem gets harder with SNR for a reason that has nothing to do with the victim's BER.

**3. The measured BER onset IS flat**: −2.0 to −3.1 dB at every SNR ≥ 10 dB and at the noiseless
anchor (−6.0 dB at 5 dB, −47.0 at 0 dB, where the clean floor does the work).
Above ~10 dB the clean floor is negligible, so the jammer must supply essentially the whole variance the
target BER needs, which pins the onset regardless of SNR.

**4. So the gap grows with SNR (~0.8 dB per dB) — but "the 30 dB gap IS the 30 dB SNR" is false as a
mechanism.** CNN gap: **+13.0 (5 dB) → 19.9 → 24.5 → 29.7 → 35.4 → 37.7 → 41.0 → +45.5 (40 dB)**. It is
driven by the **stealth edge** moving, not by the damage threshold moving, and it is not the diagonal.
**Two onsets have to be reported, not one** (`fig_gap_vs_snr.png` draws both): *noise parity*
(JSR = −SNR, where jammer power equals the noise floor) moves 1 dB per dB **by construction**, so a gap
measured against it tracks SNR trivially and proves nothing. Quoting one without its definition is how
the retracted "+70 % channel-aware" happened (§2.7).

**5. The ≤ α edge of the curve: open at 0–10 dB, and there only against the CNN.** *(An α-budget
statement — the left edge of the trade-off curve, not a result of its own, §2.7.)* At SNR 0 dB the
jammer costs measurable excess BER from −47 dB JSR, 26 dB below where the most sensitive detector
(one-sided power, −21 dB) first flags it. With P(det) held at the budget, stealthy damage exists at
0–10 dB and nowhere from 15 dB up: at 5 dB D2 reaches 2.9e-3 excess BER against the CNN where D1
manages one bit error. At 5–10 dB the clean link makes no errors, so this damage is real — **but
against the CNN only**, which is off-design there: per detector, D2's best damage within α at 5 dB is
2.9e-3 (CNN) / 2.3e-5 (kurtosis) / 7.6e-6 (two-sided power) / **none (one-sided power)**, and at
10 dB 4.2e-4 / 2.0e-4 / none / none. The classical power detector closes what the SOTA detector leaves
open — the §3.3d pattern again. At 0 dB the clean BER is already 2.3e-3 (§3.3c #2). Budgets below
**0.005** are not resolvable from the stored 257-point CDFs and 2048 clean frames (`B_MIN`); the
figure starts there.

**6. Kurtosis is the one detector that does not sharpen.** Its edge saturates at −20 dB from SNR 10
upward — it is a *shape* statistic, not a power one — so its gap to noise parity falls 1:1 and goes
negative above 20 dB. Against kurtosis the D2 gain is large and flat across SNR (D1 0.82–0.99 vs D2
0.07–0.11 at matched excess BER 3e-4, SNR 10–40 dB) — the one detector the GAN beats at every level.

**7. Damage follows SINR, detection follows JNR.** Re-expressed as JNR (= JSR + SNR), the stealth edges
barely move: CNN −16 → −8 dB and one-sided power −21 → −9 dB over SNR 0–40 dB (against −16 → −48 and
−21 → −49 in JSR), while the BER onset is (nearly) fixed in SINR. The two thresholds live on different
axes — findings 2–4 in one line. These are *nominal* ratios from existing data; the effective
decision-point SINR of a structured jammer needs one stored number per point (§4.3).

**8. SER adds information only where a jammer hits I and Q together.** Measured SER over the SER that
BER implies with independent axes, at points with ≥ 200 bit errors: noise 1.00, plain GAN 1.01, D2
generators 0.96–0.99, Amuru p = 0.1 0.87, genie 0.52 (a flip costs two bits per symbol error) —
identical at 15 and 30 dB. So SER is a table column, not an axis. **PER is not stored:** for uncoded
256-bit frames it would be 1 − (1 − 3·10⁻⁴)²⁵⁶ ≈ **7.4 %** at the matched point if errors were
independent — which makes "BER 3e-4" land as packet loss — and errors are known to cluster within
frames for bursty jammers (`regress_snr30.py`), so PER could reorder jammers at equal BER (§4.3).

**Figures** (`artifacts/cgan/snr_ablation/run001/`, all from `snr_figures.py` on the login node):
`fig_headline_gain_vs_snr.png` (the CNN row of the table, both halves, Amuru for reference) ·
`fig_matched_ber_vs_snr.png` (all four detectors, plus the JSR each jammer needs) · `fig_mechanism.png`
(the four stealth edges and the BER onset on one axis) · `fig_gap_vs_snr.png` (the two onsets and both
gaps vs the diagonal) · `fig_far_vs_snr.png` (FAR held at α by re-calibration; the hit rate on a fixed
jammer rising with SNR) · `fig_frontier_by_snr.png` (achievable excess BER per budget ≥ 0.005, one panel
per level — **the budget moves the FAR threshold too, so this is the ≤ FAR view §2.7 rejects**; kept
for the α question, to be replaced by fixed-α trade-off curves, §4.3) ·
`fig_spectrograms_by_snr.png` (what the CNN receives at 0/15/30 dB/noiseless; from `snr_examples.py`,
job 2271314, which re-measures the headline cells on fresh seeds: 15 dB D2 0.078, 30 dB 0.857) ·
`fig_regression_30db.png` (the gate, drawn). Palettes pass `validate_palette` (light, all pairs);
Amuru is orange here, not `gan_figures.py`'s purple, which fails the normal-vision floor against the
CNN-targeted maroon — **the §3.3f figures still carry that failing pair.**

**Report page (8 figures, 2026-09-24):** <https://claude.ai/artifact/PonQqobu2xpk52oWYfyoAB> — private,
**not yet shared with the supervisor**. Its Fig. 6 is `fig_frontier_by_snr.png` (the ≤ FAR view) —
replace it before sharing if the trade-off figures exist by then. Its HTML source is not in the repo;
to update it, read the artifact by URL, edit, and republish to the same URL; figures are served as
`fig/*.png`.

**Caveats, all load-bearing.**
- **The CNN is out of distribution off 30 dB.** Weights frozen there by decision; only the colour scale
  and threshold are re-fitted, and the scale's `vmin` moves **−14.7 dB (SNR 0) → −44.2 (30) → −80.7
  (noiseless)**. Reported as *"a detector trained at 30 dB, deployed off-design"* (§3.3c #6 precedent),
  and it is why **the noiseless anchor is unusable for the CNN**: its edge reads **+15 dB**, i.e. it
  detects almost nothing, which is an OOD artifact and not a property of any jammer. Retraining per
  level is Tier 2 and is gated.
- **At the noiseless anchor both power detectors flag every JSR on the grid** (no edge exists). With
  N₀ = 0 the clean power is deterministic and the hard JSR projection makes any jammer an exact,
  detectable shift. Correct behaviour, not a bug — and the reason the anchor is an anchor, not a result.
- **Tier 1 measures transfer, not achievability.** The 10–20 dB peak could be larger still with
  per-SNR retraining, and the +1.0 pp at 0 dB is a 30 dB-trained generator working off-design.
- Single round, frozen detectors, K = 1, and the whole sweep is the `noise` row for edges.
- **Measurement traps found and fixed here; each would have produced confident nonsense** (the code
  carries the reasoning in comments). The **stealth edge** must be the *last* JSR inside α + 2σ, not the
  first crossing of α — at low JSR P(det) sits *at* α by construction, so a crossing rule fires on noise
  (`snr_figures.stealth_edge`). And: a BER
  comparison must use the **error count capped at the frame count**, never a fixed log10 band — at
  131 072 bits/point a BER of 7.6e-6 is ONE error, and bursty jammers make the frame, not the bit, the
  independent unit; and a BER **onset** must be taken at the first grid point above target when the
  left bracket is zero, or whether a one-error point happens to land below it silently deletes whole
  levels (35 dB vanished from the figure exactly this way).

## 3.3h E3 pre-check — under fading the team gains nothing; TIMING coordination does (2026-09-26)

**A cheap transfer check before committing to any MARL build (`cgan/team_fading.py`), directed by the
user this session.** The detector is at the victim receiver (confirmed this session), so the
containment argument holds: whatever K jammers deliver at R, one jammer controlling its received
signal could deliver too, so the lossless single jammer (§3.3f) is the CEILING and no team can beat it
on the BER–P(det) plane. This measured how far a *realistic channel without jammer CSI* pulls a single
jammer below that ceiling, and how much of the loss K jammers with independent content win back.
**Generators are EVALUATED, not retrained** (transfer, like E2 Tier 1). Fading is applied on the
jammer→R links only (T→R stays clean, to isolate it from the detector-blunting effect of
legitimate-link fading), block-flat per frame, E|h|² = 1: Rician K = 10 dB (LOS drone link) and
Rayleigh (worst case). Five jammers — noise, Amuru pulsed(0.1), D1 plain GAN, the CNN-targeted D2
(β = 10), the kurtosis-targeted D2 (β = 1) — each alone and as K = 2/4 with an equal power split,
independent content, independent fades; generators also run with bursts time-aligned at R vs a random
per-jammer symbol shift. Jobs: 2273604 verify (§16, all pass) · 2273605 smoke · 2273606 array (6 tasks,
SNR 15/30 dB × lossless/rician10/rayleigh, 8–10 min each). Tables from `team_figures.py`. **No figures,
no report page yet** — numbers only, in `artifacts/cgan/team_fading/run001/` (6 JSONs, not archived).

**Regression check:** the lossless K = 1 rows reproduce E2 — D2 vs the CNN at matched BER 3e-4 gives
0.822 @ 30 dB (E2: 0.838) and 0.103 @ 15 dB (E2: 0.084).

**1. Fading HELPS a single jammer, so there is no gap for a team to recover.** At matched BER, every
faded single jammer needs less power AND is flagged less than its lossless self: D2 vs the CNN at 30 dB
**0.822 → 0.541** (Rayleigh); white noise needs **3.7 dB less power** (Rayleigh). Cause: block-flat
fading gives the jammer a random received power per frame — frame-level pulsing — and damage +
detection both concentrate in the strong frames. A single jammer could reproduce this exactly by
varying its own per-frame power; it is neither a channel nor a team benefit. **This is the per-frame-
JSR-reference artifact flagged before the run** (`team_fading.py` docstring): the lossless reference
holds JSR exactly per frame while a faded jammer does not, and equalising that is what a Methodology
sentence on the hard equality projection (§2.8) already covers. Independent-content teams average the
fades out (verified 1/K frame-power spread, `verify.py` §16d), pulling back toward the lossless numbers.
Rician K = 10 dB (the realistic drone case) barely moves anything.

**2. Timing coordination is the real finding — and it is a loss to AVOID, not a gain.** Four copies of
the CNN-targeted D2, equal total power, lossless, P(det) vs the CNN at matched BER 3e-4 (total JSR in dB):

| | 15 dB | 30 dB |
|---|---|---|
| one D2 jammer | 0.103 @ −21.0 | 0.822 @ −21.2 |
| K = 4, bursts **aligned** at R | 0.079 @ −22.8 | 0.649 @ −22.5 |
| K = 4, **random timing** | 0.268 @ −16.6 | 0.996 @ −16.3 |

- **Random timing costs ~6 dB of power and 19–35 pp of P(det).** D2's bursts sit at fixed positions in
  its segment; independent offsets smear them, raising the effective duty cycle. Amuru pulsed shows the
  same from the classical side (K = 4 raises P(det)_CNN 0.417 → 0.836 at 30 dB).
- **With alignment, four drones match or slightly beat one.** At 30 dB: −17 pp and −1.3 dB total, i.e.
  each drone transmits **7.3 dB less** than the single jammer. At 15 dB the P(det) difference is within
  the ±2.6 pp (2 SE, 512 frames) noise, at −1.8 dB total.
- **Bounded by containment:** one jammer sending the aligned sum does equally well. So the honest claim
  is *"K simple drones with timing coordination match one strong jammer at lower per-drone power"*, not
  *"coordination beats the best single jammer"*.

**Consequence for the MARL build (§3.4 D4).** The single lever a co-located detector leaves open is
**timing alignment** (delays are computable from positions; carrier phase is not, §3.3d), plus the
transmit-power saving from splitting. That is exactly the supervisor's inter-jammer-delay axis
(§B.2, §4.3). Stealth is *not* a lever here; a MARL-for-stealth run would be chasing the fading
artifact of finding 1. **The user's framing (2026-09-26): the point of MARL is that the policy
DISCOVERS these levers itself, with as little inductive bias as possible** — so the pre-check's role is
to bound what a correct policy can find (the timing curve between the smeared floor and the
containment ceiling), not to hand-engineer the answer. See §3.4 D4 and §4.2 Q12.

## 3.3i D4a — the delay-decay curve: coordination is worth ~1 symbol of timing accuracy (2026-09-26)

**The non-learned half of D4 (§4.2 Q12), transfer (no retraining).** A leader holds the victim's frame
timing and K − 1 followers get it with error N(0, σ²) symbols. The setup otherwise matches E3 lossless:
equal received split and independent content. It was run on three generators × K = 1/2/4 × SNR 15/30 dB
with 512 frames per point (jobs 2273650 verify §16–17 all pass, 2273651 array, 26–29 min per task).
Numbers only, in `artifacts/cgan/team_timing/run001/` (2 JSONs); tables from
`team_figures.py --timing`. **Regression:** the σ = 0 and shifted rows reproduce E3's aligned and
shifted rows with fresh draws, to within **±0.06 P(det) / ±0.7 dB**. That is the run-to-run noise, so
differences between neighbouring σ rows smaller than that are not findings.

The headline cell: the CNN-targeted D2 (β = 10), K = 4, P(det) vs the CNN at matched excess BER 3e-4
(total JSR in dB), with the single jammer = 0.814 @ −21.1 at 30 dB and 0.095 @ −21.1 at 15 dB:

| σ [symbols] | 0 | 0.5 | 1 | 2 | 4 | 8 | 16 | uncoordinated |
|---|---|---|---|---|---|---|---|---|
| 30 dB | 0.712 @ −21.8 | 0.784 @ −21.3 | **0.858 @ −20.3** | 0.945 @ −19.1 | 0.942 @ −19.1 | 0.981 @ −17.4 | 0.991 @ −16.7 | 0.999 @ −16.2 |
| 15 dB | 0.071 @ −22.7 | 0.094 @ −21.5 | 0.104 @ −20.7 | 0.102 @ −19.5 | 0.139 @ −18.5 | 0.189 @ −17.3 | 0.216 @ −17.1 | 0.292 @ −16.6 |

1. **The team stops matching one jammer at σ ≈ 0.5–1 symbol** (30 dB, and in JSR also at 15 dB, where
   the P(det) differences are inside the noise). Half the 5.6 dB alignment gain is gone by σ ≈ 2, and
   the curve reaches the uncoordinated floor by σ ≈ 8–16. So **coordination is worth roughly one symbol of
   timing accuracy** (1 µs at 1 MBd).
2. **More followers decay further.** K = 2 loses ~3.7 dB between σ = 0 and the floor, and K = 4 loses
   ~5.6 dB. Each follower adds its own smear.
3. **The same shape holds for every generator/detector pair that is not saturated**: power at 15 dB
   (K = 4: 0.267 → 0.765), kurtosis, and the kurtosis-targeted D2. The plain GAN is flagged ≈ 1.0 by the
   CNN at 30 dB, so only its JSR moves there (−10.9 → −9.4 dB).

**Q12's knee rule, applied: the transmit advance δ_k is NOT dead weight.** The knee (σ ≈ 1–2) is
below the box's geometric delay spread. An *uncompensated* team on the 50 test drops is offset from its
leader by **1.1 symbols rms** (median 0.8, max 2.4; from `test_drops.json` positions, 1 MBd). That is
right at the crossover, so compensating propagation delay decides whether the team beats the single
jammer or loses to it. D4b keeps both actions (u_k and δ_k).

**Received powers are unequal but comparable, so timing still matters with real geometry.** D4a used an
equal received split. On the 50 test drops (free-space gains, equal transmit power), the strongest
drone's share of the team's received power has a median of **0.51** at K = 4 (IQR 0.41–0.68), and the
second-strongest drone is a median **2.3 dB** weaker (IQR 1.0–6.0). At K = 2 the share is 0.70 and the
gap 3.8 dB. A weak drone smears little, so the geometric version of the curve should decay more gently
than the table above. **Measured by D4b's heuristic arm: it does, by about 2 dB at σ = 16 and at the
floor, with no difference at σ ≤ 1 (§3.3j finding 4, with an implementation caveat).**

## 3.3j D4b — the learned policy finds the power lever, not delay compensation (2026-09-26, PAUSED)

**STATE: PAUSED by the user on 2026-09-26, pending the GAN rework (§3.3f, "Imitation start and training
budget").** Every D4b number below uses the frozen **`run001/task14_G.pt`** (the CNN-targeted D2 generator
at β = 10, warm start, 400 steps). That generator is now known to be under-trained, and a 4000-step cold
start produces a different jammer. The timing sensitivity that D4a and D4b measure comes from the
waveform's burst structure, so **E3, D4a and D4b are all conditional on the old generator**. Before any
of them is quoted, re-run them on whichever generator the rework settles on. Nothing D4b-related is in
the paper (user rule).

**What was built (first cut: K = 4, SNR 15/30 dB, β ∈ {0, 10}, own-geometry arm only; user choices
2026-09-26).** `cgan/team_policy.py` implements Q12 (§4.2):
- **Policy:** one shared MLP (4 → 64 → 64 → 2, tanh). It maps each drone's position relative to R
  (divided by the box) and log1p(σ_k) to a power fraction u_k = sigmoid and a transmit advance
  δ_k = 6·sigmoid symbols. The leader is drone 0, with σ_k = 0; followers have σ_k = σ.
- **Training:** direct gradient of the D2 loss −(log E[BER] − β·soft P(det)_CNN) through a new
  **differentiable FFT fractional delay** (`frac_delay`, a phase ramp on the padded stream) and a
  √u amplitude. Random drops come from `scene.draw_positions` with analytic LOS gain and delay; σ is drawn
  from the D4a grid, and the full-power total received JSR is drawn from (−30, −10) dB.
- **Budget axis:** T_full, the full-power total received JSR, is split across drones in proportion to
  their gain (equal transmit power). At u = 1 the team's nominal total is exactly T_full, and at K = 1 it
  reduces to the D-series jammer.
- **Evaluation:** pooled over the 50 test drops, matched excess BER 3e-4, 512 frames per point.
  Four arms: *learned* · *heuristic* (δ = τ, u = 1) · *uncoordinated* (no shared clock: a uniform
  offset over the 128-symbol period, u = 1) · *ceiling* (one D2 jammer at the team total).
- **Checks:** `verify.py` §18 has 10 checks. frac_delay equals an integer roll and carries gradient;
  geometry equals d/c and free space; heuristic K = 1 equals the single jammer; u = 1 gives exactly
  T_full; both policy outputs receive gradient. They pass in the team-only job 2273734 and in two
  full-suite jobs, 2273958 and 2273984.

**Result (final variant, jobs 2273939 train / 2273943 eval).** CNN P(det) @ total JSR [dB] at matched
excess BER 3e-4 (tables: `python team_figures.py --policy`):

| 30 dB, β = 10 | σ = 0 | σ = 1 | σ = 4 | σ = 16 |
|---|---|---|---|---|
| heuristic (δ = τ, full power) | **0.705 @ −22.2** | 0.855 @ −20.4 | 0.920 @ −19.5 | 0.959 @ −18.9 |
| learned | 0.794 @ −21.0 | 0.828 @ −20.5 | 0.908 @ −19.3 | 0.956 @ −18.7 |

For reference at 30 dB: the ceiling is 0.854 @ −21.1 and the uncoordinated floor is 0.982 @ −18.2.

| 15 dB, β = 10 | σ = 0 | σ = 1 | σ = 4 | σ = 16 |
|---|---|---|---|---|
| heuristic | 0.072 @ −22.7 | 0.100 @ −20.7 | 0.127 @ −19.4 | 0.153 @ −18.9 |
| learned | 0.080 @ −20.8 | 0.098 @ −20.4 | 0.143 @ −19.3 | 0.152 @ −19.1 |

For reference at 15 dB: the ceiling is 0.097 @ −20.8 and the uncoordinated floor is 0.202 @ −18.4.

1. **The delay-decay curve reproduces with real geometry.** The heuristic rises from 0.705 at σ = 0 to
   0.959 at σ = 16 (30 dB), and at σ = 0 it slightly beats the single-jammer ceiling (−15 pp, −1.1 dB),
   as in §3.3h. The heuristic arm is stable across four evaluations (0.673–0.705 at σ = 0, 30 dB).
2. **The policy learns the power lever: u → 0.99 everywhere.** Full power is best and no drone should go
   quiet. With a shared σ and a per-drone cap that is the expected answer, and learning finds it.
3. **The policy does not learn delay compensation.** δ does not track τ in any of the four training
   variants (below): (δ − τ) rms is 1.5–3.4 symbols, and δ ends near one constant per policy. A constant
   δ leaves the drones' geometric spread (1.1 symbols rms) uncompensated. That is why learned loses to
   the heuristic at σ = 0 (−9 pp and −1.2 dB at 30 dB β = 10; −11 pp at β = 0) and matches it once
   σ ≥ 1 swamps the geometric spread. **"Learned ≈ an uncompensated team" is inferred, not measured:
   no δ = 0 arm was run.**
4. **With real geometry the curve decays more gently at large σ, as §3.3i predicted.** The heuristic
   arm does not depend on β, so the β = 0 and β = 10 files are two independent measurements of it.
   Compared with D4a's equal split (spec_cnn_b10, K = 4):
   - σ ≤ 1: equal within the ±0.7 dB run-to-run noise.
   - σ = 16: −18.9 / −19.2 dB against −16.7 at 30 dB, and −18.6 / −18.9 against −17.1 at 15 dB,
     so about 2 dB gentler.
   - Uncoordinated floor: −18.2 against −16.2 dB. At 15 dB it is also less detectable, 0.19–0.20
     against 0.29.

   This fits the unequal-power argument: the strongest drone carries a median 51 % of the received
   power, and a weak drone smears little. **Caveat:** D4b applies delays with the FFT `frac_delay`,
   D4a with integer crops plus Sionna sinc taps. The two paths are shown equivalent only at K = 1, on
   BER, with the perfect generator (§18d). So part of the 2 dB could be implementation rather than
   geometry.

**Training variants.** All four wrote to the same filenames, so only the last one is on disk.

| variant | jobs (train / eval) | δ outcome at σ = 0 (30 dB β = 10) |
|---|---|---|
| unbounded δ = 8·raw, 400 steps, batch 32 | 2273737 / 2273738 | δ mean 0.62 vs τ mean 1.74; rms error 1.75; u ≈ 0.87 |
| unbounded, 2000 steps | 2273906 / 2273907 | **diverged**: δ mean 5.6, rms error 5.9 (−16.9 at σ = 16) |
| bounded δ = 6·sigmoid, 2000 steps | 2273924 / eval 2273925 cancelled | collapsed to a bound: δ ≈ 0.4 (or ≈ 5.8 in other tasks) |
| bounded, **ACCUM = 8** (effective batch 256), 600 updates | **2273939 / 2273943** | δ mean 5.04, rms error 3.39; u ≈ 0.99 — **on disk** |

**The δ diagnostic** (`team_policy.py --mode diag`, heuristic δ = τ plus an offset Δ, σ = 0,
T_full −20 dB, 30 dB, β = 10):
- *Common shift of all drones* (job 2273922, 512 drops; job 2273923, 4096 drops): the loss is flat to
  within ~0.3 of ~15.4 and does not have its minimum at Δ = 0. It is slightly lower at Δ = +4 to +8,
  apparently a preference of the frozen generator for a certain phase of its periodic segments.
- *Follower 1 alone* (job 2273923, 4096 drops): **a clean minimum at Δ = 0** (loss 15.37), rising
  smoothly to 15.9 by |Δ| = 2. The per-drone δ = τ optimum therefore exists, but it is only ~0.6 loss
  units deep.
- At batch 32 the soft-P(det) noise is about √(0.8·0.2/32)·β ≈ 0.7 loss units, larger than that well.
  **Raising the effective batch to 256 did not fix δ, so noise is not the whole explanation — the cause
  is open** (§4.2 Q12).

**Artifacts:** `artifacts/cgan/team_policy/run001/{snr15,snr30}_{b0,b10}{_policy.pt,.json}` (final
variant) plus `snr15_b0_smoke*` from smoke jobs 2273735/2273736. To inspect what a policy does,
rebuild the MLP from the `state_dict` (strip the `net.` prefix), feed it `obs_of` on the test-drop
positions, and compare δ = 6·sigmoid(out₁) with τ = d_kR/300 m (symbols). The script used on
2026-09-26 was session scratch and is not in the repo.

**To resume (after the GAN rework):** point `team_policy.D2_GEN` at the new generator. Then re-run E3
and D4a on it first (`team_fading.py`), because both set the scale of what coordination can buy, and
only then return to the δ question.

## 3.3k S1 — shadowing on the victim's link blinds the naive energy detector, not a gain-aware one (2026-09-28)

**Why (user, 2026-09-27): "make the env more realistic but easier for the jammer to hide, without MARL
first".** The question was whether finding 1 of §3.1 (the energy floor) is only a property of the ideal
link. The arithmetic behind the floor:
- **The energy detector's margin is 0.3 of ONE symbol's energy.** At 30 dB, clean frame power has mean
  0.10012 and its α = 0.05 threshold is 0.10036 (+0.24 %, std 0.14 %; `baselines/snr/snr_30/
  thresholds.json`). A frame is 128 unit-energy symbols over 1280 samples, so the margin is
  0.00024 × 1280 ≈ 0.3 symbol energies, i.e. a mean shift of −26 dB JSR. The spread is the signal × noise
  cross term (std √(2·N0·128)/1280 = 1.4e-4, as measured): QPSK on Nyquist pulses over h = 1 has an
  almost deterministic frame energy.
- **One bit flip costs more than that.** Pushing one axis past its decision boundary (1/√2 away) costs
  ≥ 0.5 symbol energies (matched pulse, right direction, no noise). A symbol-blind push that flips a bit
  opposes the signal and LOWERS frame energy (≈ −0.5); the equally likely opposite push raises it ≈ +1.5
  and trips the alarm — ≈ 1 alarm per broken frame, the measured 1.05–1.3 (§3.3g E2b). Back-of-envelope:
  it ignores noise, pulse overlap and complex phase.
- **Measured:** run003's power-1s β = 10 generator (30 dB, 131k bits per point) has BER 0 at every JSR
  ≤ −23 dB while one-sided power flags it 0.27 (−32 dB), 0.55 (−26), 0.65 (−23); the first error is at
  −22 dB (P(det) 0.69). **The generator does go under the threshold; nothing breaks there.**
- **Why sim01/03b/04 looked easy — the detector and the information, not the method.** sim01's threshold
  was hand-set at `mean|rx|² > 3.0` with clean power 1.0 (a +3 dB JSR budget, 29 dB looser than a
  FAR-calibrated one); sim03b/sim04 had no energy detector (kurtosis only, hand-set −1.0) and were genies.
  The recheck (§A.6) and sim08 m1b (§A.7) already showed both halves: lossless → energy kills stealth;
  fading + finite SNR → stealth revived. **An NSF-vs-GAN comparison was asked for and not run:** run003
  is already sim03b's method class (direct gradient through the detector), so it would compare
  architectures (§2.9), and the arithmetic above predicts the same floor for any symbol-blind generator.
- **The SNR wall** (Tandra & Sahai, IEEE JSTSP 2008): with clean-power uncertainty of x dB, energy
  detection is impossible below 10·log₁₀(ρ − 1/ρ), ρ = 10^(x/10): x = 0.05 / 0.1 / 0.3 / 1 dB → −16.4 /
  −13.4 / −8.6 / −3.3 dB JSR. The run003 generators work at −23 dB. A real receiver's AGC also removes
  raw frame energy; multi-user links use normalised checks (signal strength vs PDR, Xu et al. MobiHoc
  2005, or post-decision residuals), which remove the legitimate power but hit the wall on the noise floor.

**Design (built 2026-09-27; `verify.py` §19).**
- **Channel:** `Link.shadow_db` = std [dB] of a per-frame log-normal POWER gain on the victim's own link,
  normalised to unit mean power, real and positive, so the sign decisions need no channel estimate and
  clean BER at 30 dB stays ≈ 0 (the matched-BER metric keeps working). Applied in `attacks.frames` only;
  σ = 0 draws nothing, so every earlier result is bit-identical. Jammers keep their JSR against the MEAN
  signal power; the genie is handed the faded symbols (it knows the channel by definition). Rayleigh /
  Rician fading is the second step, not built: it needs an equaliser and makes clean BER non-zero.
- **Grid** `snr_ablation.SHADOW_GRID_DB` = {0, 0.01, 0.03, 0.1, 0.3, 1, 3} dB at 30 dB SNR. A +1.64σ fade
  is exp(1.645·s − s²/2) − 1 of frame power, s = σ·ln10/10 (unit-mean normalisation): +0.4 % at
  0.01 dB (≈ the ideal-link margin), +3.8 % at 0.1 dB, +142 % at 3 dB (≈ sim01's +200 % budget) — the
  measured naive-power margins are +0.44 / +3.78 / +141.9 %. 3 dB is still mild next to real small-scale
  fading.
- **Defender per σ:** Li's CNN **retrained** on shadowed frames, same recipe and seed
  (`train_spectrogram_cnn.py --shadow-db σ --out-dir ../artifacts/cgan/baselines/shadow/shadow_<σ>`),
  every threshold re-calibrated on 20k shadowed clean frames, plus **`power_csi`** — one-sided power minus
  (g² − 1)·N/len, the energy detector that knows its own link gain (a coherent receiver estimates it;
  here it is handed the true gain — see the caveat under the result). The noise LRT is dropped at σ > 0: its QPSK mixture assumes unit
  gain. Kurtosis is scale-free, so shadowing should barely move it.
- **Tier 1 — evaluation (done 2026-09-28):** run003's 21 generators + the classical envelope, not retrained:
  `sbatch --array=0-6 submit_snr_ablation.sh --axis shadow --run shadow_run003 --gan-run run003` →
  `artifacts/cgan/snr_ablation/shadow_run003/shadow_<σ>.json`. `baselines.measure` now also stores the
  frame error rate (`per`, `frame_errors`), so frames per extra alarm comes from the same JSON. σ = 0 is
  the deployed defender, i.e. E2's 30 dB row — the regression check.
- **Tier 2 — retraining the generator under shadowing: built, NOT run** (`train_gan.py --shadow-db σ
  --detector-dir ../artifacts/cgan/baselines/shadow/shadow_<σ>`). Tier 1 says it would not inform: see
  below.

**Result — Tier 1 (jobs 2274946–2274952; `cgan/shadow_figures.py`).** 30 dB, α = 0.05, 512 frames per
point. Frames broken per extra alarm (E2b's power-free unit; ≈ 1 is the floor):

| σ [dB] | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| naive power — CNN-targeted GAN (β 10) | 1.04 | 1.06 | 1.23 | 4.5 | 19 | 32 | 35 |
| naive power — Amuru pulsed (p 0.1) | 1.04 | 1.06 | 1.06 | 1.06 | 2.5 | 24 | 35 |
| naive power — white noise | 1.04 | 1.06 | 1.06 | 1.06 | 1.06 | 1.05 | 5.2 |
| **gain-aware `power_csi` — every jammer** | 1.04 | 1.06 | 1.06 | 1.05 | 1.05 | 1.04 | 1.05 |
| retrained CNN — CNN-targeted GAN (β 10) | 3.6 | 5.6 | 6.1 | 5.0 | 8.2 | 6.2 | 8.0 |
| retrained CNN — Amuru pulsed | 1.43 | 1.30 | 1.25 | 1.38 | 1.39 | 1.72 | 1.49 |

P(det) of the CNN-targeted generator (β 10) at matched excess BER 3e-4, naive power / `power_csi` / CNN:
σ 0: 0.63 / 0.63 / 0.13 · 0.03: 0.28 / 0.61 / 0.08 · 0.1: 0.12 / 0.60 / 0.09 · 0.3: 0.07 / 0.57 / 0.08 ·
1: 0.06 / 0.62 / 0.07 · 3: 0.07 / 0.50 / 0.06 (FAR 0.04–0.06 throughout).

1. **The naive energy detector goes blind where the SNR wall says:** 4.5 frames per extra alarm at
   0.1 dB, 19 at 0.3 dB; by 1 dB every shaped jammer, classical included, is past it (24–35), and at 3 dB
   even white noise gets 5.2. The prediction (collapse between 0.03 and 0.3 dB) held.
2. **A detector that knows its own link gain is unmoved: 1.04–1.06 for every jammer at every σ.** The
   floor never needed the absolute clean power, only the signal's energy, which the receiver knows once
   it knows its channel. **Against such a defender, shadowing opens no hiding place.**
3. **The CNN stays the weak link at every σ.** CNNs retrained on shadowed frames (validation accuracy
   0.970–0.983, AUC 0.994–0.997; deployed 0.960 / 0.993) flag the CNN-targeted generator — trained
   against the deployed ideal-link CNN — on 6–9 % of frames at matched BER, 5–8 frames per extra alarm.
   The evasion transfers across a retrain on a different channel, as run004 found across seeds.
4. Kurtosis does not move (scale-free); its blind spot to QPSK-like jammers is there at every σ,
   including 0 (matched QPSK: 20–48 frames per extra alarm).
5. Confirmed stealthy BER inside the strict α budget stays ≈ 0 for `power_csi` and the CNN at every σ.
6. **Regression:** σ = 0 reproduces run003's evaluation. `regress_snr30.py` (pointed at `shadow_0.json`)
   flags 2 P(det) points (0.098, 0.100 vs 0.088) and 1 BER point (1.04 band units) of 859, all inside the
   √2-corrected tolerance of the known gate bug (§3.3f); BER scatter unbiased (397/859 positive).

**Caveat on `power_csi`: it is handed the TRUE gain.** A coherent receiver estimates it. A
decision-directed estimate from the frame's own symbols is unbiased against a symbol-blind jammer, and its
error is the same noise projection that makes up the clean spread, so the residual detector it yields
should be at least as sensitive. **Argued, not measured** — the obvious next check is a detector built
from r and z alone (eval-only).

**What S1 means.** The user's intuition holds for a threshold detector that does not normalise by its own
channel and fails for one that does. **Tier 2 would not inform:** against naive power the untrained
generator already gets 35 frames per alarm, against `power_csi` the 0.3-vs-0.5 symbol-energy arithmetic
is unchanged by shadowing, and against the CNN the generator already evades.

**Jobs:** 2274945 verify (full suite incl. §19, exit 0) → 2274946–2274951 CNN retrains σ = 0.01, 0.03,
0.1, 0.3, 1, 3 → 2274952 array 0–6 (all exit 0; 18–26 min per level, ≤ 2.2 GB).

## 3.3l S2 — the listening jammer: synchronous arrival buys nothing measurable (2026-09-28, DONE)

**Why (user, 2026-09-28):** the attacker may use everything a passive third party could learn in a
listening phase before the attack. **It enters the system model only if it improves the jammer's
results** ("otherwise we rewrite the sysmodel for nothing").

**What listening can and cannot give** (the boundary, agreed with the user):
- **Learnable:** signal format (modulation, pulse, rate, band); T's symbol and frame timing; T's power
  and traffic pattern; the T→J channel; J→R by reciprocity if R ever transmits; approximate positions
  (so the arrival time at R); the defender's visible reactions.
- **Not learnable:** future data symbols (i.i.d.); the T→R channel, hence the carrier phase at R
  (λ = 12.5 cm at 2.4 GHz); the detector's weights. Adding the first two gives the genie.
- **Detector knowledge is not from listening:** the grey-box assumption (run004) is Kerckhoffs's
  principle — the detector's design and training recipe are public, the trained weights secret.

**Design — perfect timing only** (user: sweep a timing error only if perfect sync helps):
`Link.jammer_sync` → `channel.async_draw` returns offset 0 (R's symbol grid) and a still uniform phase.
The offset is drawn and then zeroed, so the channel draws of a synchronous run are paired draw for
draw with an asynchronous one of the same seed (same bits, noise, phases, pulse masks). Four generators
retrained with synchronous arrival, run003 recipe (random init, 4000 steps): tasks 0 (β = 0), 2 (power
β 10), 13 / 14 (CNN β 1 / 10) → `artifacts/cgan/gan/sync003/` (net_scratch, symlinked). Evaluated at
30 dB against the deployed detectors with seed 3000, as `shadow_run003/shadow_0.json` (the async
baseline) was: `snr_ablation.py --axis shadow --task 0 --sync --run sync_run003 --gan-run sync003`.
**The evaluation is paired only up to the first confirmation pass** — how many re-measurements it makes
depends on the picks, so later attacks see different draws; the comparison is statistical. `verify.py` §20:
default still async; sync zeroes the offset and keeps the phases; a synchronous matched-QPSK jammer
matches the noiseless closed form BER = arccos(1/√(2·JSR))/π.

**Success test (decided before the run):** synchronous generators beat run003 at matched damage —
lower P(det) at matched BER / PER, more frames per extra alarm, or more damage at P(det) ≤ 0.5 — against
the CNN or the power threshold. **Expected:** against energy no change (≈ 1 frame per alarm: timing
reaches the 0.5-symbol-energy flip bound but gives no sign knowledge); against the CNN open.

**Result — FAILS the success test** (jobs 2275790 verify §20 · 2275791 train / 2275792 eval r0 ·
2275891–2275899 sync r1–r3 + async β 10 re-evaluations · 2275934–2275939 async β 1 seeds; all exit 0).
30 dB, α = 0.05, 512 frames per point, `cgan/probe_readout.py s2`. The CNN-targeted arms, 4 seeds each
(sync: `sync003{,_r1,_r2,_r3}`; async: run003 r0 + `cold002_4k_r1–r3` for β 10 and `async003_r1–r3`
for β 1, the same recipe, all re-evaluated in the same harness with seed 3000), mean ± std:

| | β 1 async | β 1 sync | β 10 async | β 10 sync |
|---|---|---|---|---|
| CNN P(det) at matched excess BER 3e-4 | 0.118 ± 0.015 | 0.134 ± 0.078 | 0.149 ± 0.016 | 0.172 ± 0.078 |
| CNN P(det) at matched excess PER 0.1 | 0.128 ± 0.022 | 0.135 ± 0.080 | 0.159 ± 0.021 | 0.172 ± 0.078 |
| CNN P(det) at matched excess PER 0.5 | 0.246 ± 0.037 | 0.194 ± 0.057 | 0.405 ± 0.114 | 0.390 ± 0.234 |
| one-sided power P(det) at matched excess PER 0.5 | 0.730 ± 0.016 | 0.703 ± 0.093 | 0.804 ± 0.131 | 0.650 ± 0.015 |
| CNN warden error ξ at PER 0.1 (threshold-free) | — | — | 0.750 ± 0.055 | 0.710 ± 0.154 |
| JSR at excess BER 3e-4 [dB] | −23.0 ± 0.0 | −23.3 ± 0.2 | −22.8 ± 0.3 | −23.2 ± 0.2 |

(ξ needs stored statistic distributions, kept only for β 10.) The first write-up quoted frames per extra
alarm (CNN 3.5 ± 1.2 vs 5.5 ± 2.2 for β 1; power 0.95–1.05); that unit is dropped (§3.4 Track 2,
"Measures") and the verdict does not depend on it.

1. **Against power, nothing:** P(det) at matched PER within the seed spread (β 10 sync lower, 0.65 ±
   0.02 vs 0.80 ± 0.13, driven by one async seed at 0.998) — the prediction (timing gives no sign
   knowledge) held.
2. **Against the CNN, nothing measurable.** Every difference sits inside the seed spread, on every
   measure, threshold-free ξ included. Damage at CNN P(det) ≤ 0.5 is the same (max PER 0.46–0.66 both
   ways). What synchronous training does change is the **seed-to-seed spread** (CNN P(det) at matched BER
   std 0.078 vs 0.015–0.016). **The first seed alone looked like a win** (β 10 at matched BER: 0.073 vs
   0.125) and did not replicate (r1–r3: 0.205, 0.256, 0.155) — the reason for the seeds.
3. **Synchronous arrival costs the classical matched jammer 3.6 dB** (matched QPSK reaches excess BER
   3e-4 at −3.0 dB instead of −6.6 dB): on R's grid it has no pulse-tail peaks, so nothing flips below the
   −3 dB geometric bound (§3.3b/C2's async finding, from the other side). The untargeted sync generator
   (β = 0, one seed) also needs ≈ 2 dB more (−18.1 vs −20.3 ± 0.4) and the CNN-targeted sync generators
   saturate lower when loud (BER 3.9e-3 vs 1.8e-2 at +10 dB) — both outside their training band.
4. Confirmed stealthy BER inside the α budget stays 0 for every non-genie jammer, sync or not.

**Consequence (the user's rule):** listening does not enter the system model, and the timing-error sweep
(conditional on perfect sync helping) is not run. The async assumption stays, now with a measured reason:
knowing T's symbol clock does not help a symbol-blind jammer at matched damage.

## 3.3m S4 — trained at 15 dB: energy flags the best generators on ≈ 25 % of frames at PER 0.1, trained at either SNR (2026-09-28)

**Why:** §3.1 finding 1 was a 30 dB result. At 15 dB the α = 0.05 energy margin is 1.66 symbol energies,
more than the ≥ 0.5 a bit flip costs, and the 15 dB "≈ 1 frame per alarm" had only been measured on
generators trained at 30 dB (E2 is transfer). **Question:** does a generator TRAINED at 15 dB beat it?
**Prediction (§3.3k sign argument):** no — a symbol-blind flip is as likely to raise the energy.

**Design (decided with the user 2026-09-28).** Defender = the CNN **retrained at 15 dB** (Track 1's D6
round-0 defender `baselines/arms/snr15_r0`, job 2275915_0: the deployed recipe with `--snr-db 15`) plus its
own 20k-frame calibration, for training and for both evaluations. Bands by `train_gan`'s rule (straddle
the target's own transition), re-measured against that CNN on the baseline run: its noise transition
(P(det) 0.1 / 0.5 at −27 / −21 dB) sits +8 / +9 dB above the deployed CNN's at 30 dB, so the CNN band
moves (−48, −16) → **(−39, −7)**; power (−32, 0) → **(−24, 0)** (its transition moved +8 dB); β = 0 keeps
(−32, 0). Generators: run003 recipe (random init, 4000 steps, run003's seeds),
tasks 0 / 2 / 13 / 14, `train_gan.py --snr-db 15 --detector-dir <snr15_r0>` → `gan/snr15_003/`.
Evaluation `snr_ablation.py --task 3 --detector-dir <snr15_r0>`: run003 → `snr15_run003_base/`, the new
generators → `snr15_run003/`. `verify.py` §21: flag off keeps 30 dB; one training step at 15 dB replayed
draw for draw (log E[BER] −107.89, the 30 dB value on the same frames −3068); the 15 dB defender honours
α on fresh 15 dB frames (FAR 0.044–0.057).

**Result (jobs 2275923 verify · 2275924 baseline · 2275925 / 2275926 / 2275974 training · 2275975 eval,
all exit 0; `cgan/probe_readout.py s4`, `cgan/probe_report.py`).** 15 dB, α = 0.05, 512 frames per point,
FAR power 0.048 / CNN 0.045, one seed per generator. Fraction of frames flagged at matched excess PER
(the measures of §3.4 Track 2, "Measures"), trained at 30 dB (run003) / at 15 dB:

| 15 dB, CNN retrained at 15 dB | β 0 | power β 10 | CNN β 1 | CNN β 10 |
|---|---|---|---|---|
| one-sided power, PER 0.1 | 0.61 / 0.31 | 0.39 / 0.37 | 0.23 / 0.26 | 0.26 / 0.27 |
| one-sided power, PER 0.5 | 0.97 / 0.56 | 0.85 / 0.67 | 0.47 / 0.50 | 0.49 / 0.57 |
| CNN, PER 0.1 | 0.31 / 0.11 | 0.27 / 0.09 | 0.08 / 0.07 | 0.07 / 0.11 |
| JSR at excess BER 3e-4 [dB] | −20.4 / −22.3 | −20.3 / −21.0 | −23.2 / −23.0 | −23.1 / −22.4 |

Classical envelope against power at PER 0.1 / 0.5: white noise and matched QPSK 1.00 / 1.00, Amuru p 0.1
0.88 / 0.96. For comparison at 30 dB (run003, deployed defender): the CNN-targeted generators are flagged
by power on 0.61–0.63 at PER 0.1 and 0.72–0.73 at PER 0.5.

1. **Against energy at 15 dB the best generators are flagged on ≈ 25 % of frames at PER 0.1 and ≈ 50 % at
   PER 0.5, trained at either SNR** — far less than at 30 dB (≈ 63 % / 72 %). The larger margin (1.66
   symbol energies) does help a jammer at 15 dB; training there does not push the best value further.
   A 1 %/1 % sequential test on the energy alarms still decides after ≈ 1.8 frames are broken (CNN β 10).
2. **Training at 15 dB lifts the untargeted and energy-targeted generators to that level** (β 0: 0.61 →
   0.31 at PER 0.1, 0.97 → 0.56 at PER 0.5) through a 0.7–1.9 dB lower damage onset; the CNN-targeted
   ones were already there.
3. **The retrained 15 dB CNN is still the weak link.** run003's CNN-targeted generators, trained against
   the deployed 30 dB CNN and never this one, are flagged on 7–8 % of frames at PER 0.1: the evasion
   transfers across a retrain at another SNR, as it did across seeds (run004) and shadowing (S1).
   **Single seed:** the untargeted / energy-targeted generators trained at 15 dB are also far less visible
   to it (0.31 → 0.11, 0.27 → 0.09), unreplicated.

**Corrected 2026-09-29:** the first write-up of S4 (same day) said "the floor holds: 1.05–1.12 frames per
extra alarm, training buys effectiveness, not a better trade". That unit's best value over JSR is the
saturated corner 1/(1 − FAR) = 1.05 for all four arms trained at either SNR, so it hid finding 2 above.

**What S4 means for finding 1:** at 15 dB an energy detector that knows its noise level still catches
the best generator within a couple of broken frames, but flags only ≈ a quarter of the frames it breaks
at PER 0.1 — much weaker than at 30 dB. S5 (§3.3n) shows what happens when it does not know the noise.

## 3.3n S5 — noise uncertainty: an honest CFAR detector loses the energy floor at 15 dB (2026-09-28)

**Why (user):** "if noise goes up, the FAR should also be higher". The defender does not know each frame's
noise level: the noise variance is scaled per frame by a log-normal factor, std σ_N dB, unit mean
(`Link.noise_unc_db`). **Two defenders on the same frames:** (i) **naive** — calibrated at the nominal
noise, so its realised FAR rises; (ii) **honest CFAR** — re-calibrated on clean frames drawn with the
uncertainty, so FAR = α and the thresholds widen (the SNR wall, §3.3k). **Decided with the user
2026-09-28:** CFAR keeps the naive colour scale and CNN weights (frozen at 30 dB) and re-derives only its
thresholds, two-sided centres and LRT clean parts (`calibrate_snr.cfar_defender`), so both defenders see
identical statistics and only the thresholds differ. Grid σ_N ∈ {0, 0.5, 1, 2} dB × SNR {30, 15} dB; run003's
21 generators + the classical envelope, evaluated not retrained. `snr_ablation.py --axis noise` →
`snr_ablation/noise_unc_run003/snr_<snr>_unc_<σ>.json`, CFAR calibrations in `baselines/noise_unc/`.
`verify.py` §22: factor law; σ_N = 0 draws nothing; BER = mean_f Q(c0/√(N0 u_f)) and the decision-point
noise tracks N0 u_f; at 15 dB, 2 dB the CFAR defender holds α on fresh frames while the naive one-sided
power FAR rises to 0.22.

**Result (jobs 2275921 verify · 2275922 array 0–6 + 2276029 task 7, resubmitted after a `TaskProlog`
failure; all exit 0; `cgan/probe_readout.py s5`).** α = 0.05, 512 frames per point.

Realised FAR on clean frames, naive / CFAR, and the one-sided power threshold's margin over the clean mean:

| σ_N [dB] | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| 30 dB: one-sided power | 0.059 / 0.059 | 0.048 / 0.045 | 0.051 / 0.045 | 0.067 / 0.048 |
| 30 dB: CNN | 0.037 / 0.037 | 0.046 / 0.046 | 0.060 / 0.046 | **0.111** / 0.051 |
| 30 dB: power margin, CFAR | +0.23 % | +0.23 % | +0.24 % | +0.27 % |
| 15 dB: one-sided / two-sided power | 0.047 / 0.050 | 0.077 / 0.071 | 0.141 / 0.177 | **0.229 / 0.419** |
| 15 dB: CNN (naive) | 0.056 | 0.064 | 0.118 | 0.203 |
| 15 dB: power margin, CFAR | +1.25 % | +1.49 % | +2.08 % | +3.74 % |

One-sided power at 15 dB, at matched excess PER 0.1: fraction of frames flagged, and frames broken before
a 1 %/1 % sequential test on the alarms decides — honest CFAR (naive in brackets):

| σ_N [dB] | 0 | 1 | 2 | frames broken first, σ_N 0 → 1 → 2 |
|---|---|---|---|---|
| white noise · matched QPSK | 1.00 | 1.00 | 1.00 | 0.15 → 0.15 → 0.16 |
| Amuru p 0.1 | 0.86 | 0.75 (0.85) | 0.52 (0.82) | 0.2 → 0.3 → 0.5 (0.6) |
| GAN β 0 | 0.59 | 0.34 (0.56) | **0.12** (0.50) | 0.4 → 1.1 → **15** (2.5) |
| GAN power β 10 | 0.43 | 0.26 (0.43) | **0.12** (0.42) | 0.7 → 1.8 → **14** (5.3) |
| GAN CNN β 10 | 0.25 | **0.14** (0.28) | **0.09** (0.32) | 1.9 → **7.2** → **57** (21) |

The threshold-free warden error ξ = min(P_FA + P_MD) of one-sided power for GAN CNN β 10 at PER 0.1 rises
0.77 → 0.80 → 0.89 (σ_N 0 → 1 → 2), identical for both defenders by construction: the loss is in the
statistic, not in how the threshold is set.

1. **At 30 dB nothing changes for energy** (CFAR margin 0.23 → 0.27 %): the noise is 0.1 % of the received
   power. As predicted.
2. **At 15 dB the naive defender pays in false alarms:** FAR 0.08 / 0.14 / 0.23 (two-sided up to 0.42) at
   σ_N = 0.5 / 1 / 2 dB. It still flags the jammers on about the same fraction of frames, but its alarms
   carry little evidence: 21 frames broken before a sequential test decides at ±2 dB (CNN β 10).
3. **At 15 dB the honest CFAR defender lets learned jammers under.** Keeping FAR = α widens the power
   margin to +2.08 / +3.74 % of frame power (≈ −17 / −14 dB as a JSR mean shift). The generators do their
   damage at −20 to −23 dB, below that, and are flagged on 9–12 % of frames at PER 0.1 (±2 dB); white noise
   and matched QPSK need > −7 dB, above it, and are caught as before. **This is the SNR wall of §3.3k
   measured, for the detector's uncertainty about the noise rather than the gain.**
4. **The CNN is noise-sensitive even at 30 dB:** its naive FAR doubles (0.111) at 2 dB, where power's does
   not move — the spectrogram shows the out-of-band noise floor directly. Its CFAR version keeps α.
5. Evaluation only: the generators were trained with known noise at 30 dB. A generator trained against
   the CFAR defender (Tier 2) could only do better; not run.

**Regression.** σ_N = 0 draws nothing. S5's 30 dB, σ_N = 0 level uses seed 3000 like `shadow_run003/
shadow_0.json` but did NOT reproduce it bit for bit on a TITAN RTX (`tikgpu04`); re-run pinned to an RTX
2080 Ti like the original (job 2275979, `snr_ablation/noise_unc_regress/`, clean + noise attack only), it
**reproduces clean FAR, clean statistics, the noise sweep and its confirmation pass exactly.** So the code
path is unchanged, and **CUDA random numbers differ between GPU models**: "same seed = paired" holds only on
the same card type. Every cross-run pairing claim in this README is statistical unless both jobs ran on
the same GPU model.

**What S5 means for finding 1 (third qualifier after S1 and S4):** the energy floor needs a defender that
knows the signal's energy (S1) AND its own noise level to within ≈ ±0.5 dB at 15 dB — or high SNR, where
the noise is too small to matter. A defender that does not know the noise has two choices, both losing:
keep its threshold and pay in false alarms, or widen it and let learned jammers under.


## 3.3o D6 round 1 — the CNN's gap was a training-range gap at 30 dB, not at 15 dB (2026-09-28)

> **⚠ Corrected 2026-09-29 (metric).** This section first led with "frames broken per extra alarm" and
> said arm A "brings every generator to 1.05, the energy floor". The user rejected that unit (it is our
> own construction, §4.3 "Metric"), and 1.05 is its saturated corner 1/(1 − α) — reached whenever a
> jammer ends up breaking and flagged on every frame — not a floor. Restated below in P(det) at matched
> damage with the energy detector on the same frames: at 30 dB the retrained CNN draws level with energy
> on the held-out generators but stays at half of energy on the round-0 attackers. The report page of
> 2026-09-28 (<https://claude.ai/artifact/KQXwSSavTTy1uFECTVseGT>) still uses the old unit.

**Why.** Adaptation cost is the supervisor's mandated headline (§2.9), and the 2026-09-28 email promised
D6. **User scope (2026-09-28): round 1 only** — *"start by retraining the CNN against our model and show
how it impacts it, im pretty sure its overall accuracy will drop, even if a bit, its a finding"*. The
prior is Phase 0.5 (§A.6): a CNN retrained on in-band jammers closed its blind spot at 99.8 → 90.5 %
accuracy and FAR 0 → 3.8 % (frozen OFDM stack, uncalibrated, "jammed" labels even at BER ≈ 0). Round 2
(the attacker retrains against the new CNN) waits.

**Design (user decisions of 2026-09-28 in italics).**
- **Defenders**, all Li et al.'s recipe and seed 11 (`train_spectrogram_cnn.py`), each calibrated on its
  own 20k clean frames:
  - **r0** — the CNN the attackers were trained against: the deployed one at 30 dB; at 15 dB the same
    recipe trained at 15 dB (*every CNN is retrained at 15 dB*, so 15 dB compares like with like);
  - **A, attacker-agnostic** — Li's four types with the jammed frames' JSR widened from U[−20, +10] to
    U[−35, +10] dB (`--jsr-range -35 10`), no generator frames;
  - **B, attacker-aware** — A + the round-0 attackers (run003 CNN β 1 / 10, tasks 13/14) as a fifth
    jammed type (`--extra-gens`): *204 frames in total, split evenly, same widened range*.
- **Jammers** (`arms_eval.GENS`) by role: **seen** = the two round-0 attackers (arm B's training set);
  **twin** = `cold002_4k/task14`, `run004_grey/task13,14`, trained from the same init seed as a seen one
  (relative weight distance 0.12–0.15 to it vs 0.22–0.32 between independent seeds, asserted in verify
  §23); **held-out** = `cold002_4k_r1–r3/task14`, the independent β = 10 seeds (*keep all, split the
  rows*); the β = 0 control; noise, matched QPSK, Amuru p 0.1.
- **Protocol:** E2/S1's — 30 and 15 dB, JSR −50…+15 dB, 512 frames per point, α = 0.05; P(det) read at
  matched damage (excess BER 3e-4, and PER 0.1) by interpolation over the JSR sweep. Cost side: Li's own
  classes on fixed frames — P(det) at α vs JSR per type (256 frames per point) and Li's class mixture
  (4 × 1578 frames) at the original and at the widened range. **Paired:** each jammer is re-seeded by
  position, so every defender at one SNR sees the same frames — checked, 0/792 sweep points differ in bit
  or frame errors at either SNR. No attacker retrain, no confirmation pass.

**Result — the gain side** (`arms_figures.py`; the energy column is one-sided power on the same frames).
P(det) per frame at matched excess BER 3e-4; matched PER 0.1 gives the generators' numbers within
±0.03 (Amuru up to +0.07):

| 30 dB | CNN r0 (deployed) | CNN A (range) | CNN B (range + attacker) | energy |
|---|---|---|---|---|
| seen (run003 β 1, β 10) | 0.09–0.11 | 0.30–0.34 | 0.47–0.48 | 0.57–0.63 |
| twins (3) | 0.10 | 0.29 | 0.44–0.45 | 0.58–0.65 |
| **held-out seeds (3)** | **0.16–0.18** | **0.58–0.74** | **0.76–0.83** | 0.60–0.81 |
| Amuru pulsed p 0.1 | 0.44 | 0.40 | 0.38 | 0.94 |
| β = 0 control, noise, matched QPSK | 1.00 | 1.00 | 1.00 | 0.94–1.00 |

| 15 dB | CNN r0 (trained at 15 dB) | CNN A | CNN B | energy |
|---|---|---|---|---|
| seen (β 1, β 10) | 0.06–0.07 | 0.06 | 0.06 | 0.25–0.26 |
| twins (3) | 0.06–0.07 | 0.05–0.06 | 0.04–0.07 | 0.24–0.27 |
| **held-out seeds (3)** | **0.05–0.08** | **0.07–0.09** | **0.06–0.09** | 0.22–0.43 |
| Amuru pulsed p 0.1 | 0.23 | 0.17 | 0.15 | 0.79 |
| β = 0 control | 0.29 | 0.30 | 0.32 | 0.56 |

Damage at P(det)_CNN ≤ 0.5, BER / PER — 30 dB: held-out 0.0015–0.0024 / 0.39–0.56 under r0, 0 / 0 under
A and B; seen β 10 0.0027 / 0.60 → 0.0007 / 0.19 → 0.0004 / 0.11. 15 dB: every generator 0.004–0.005 /
0.73–0.79 under all three CNNs.

**Result — the cost side.** Li et al.'s classes, the same frames for every defender at one SNR; argmax
is Li's metric, "at α" the calibrated operating point everything else here uses:

| | 30 dB r0 / A / B | 15 dB r0 / A / B |
|---|---|---|
| accuracy, argmax, Li's range [−20, +10] dB | 0.962 / 0.972 / 0.969 | 0.936 / 0.916 / **0.878** |
| FAR, argmax | 0.002 / 0.005 / 0.030 | 0.013 / 0.068 / **0.184** |
| detection rate at α = 0.05 | 0.978 / 0.975 / 0.972 | 0.928 / 0.889 / **0.881** |
| AUC | 0.994 / 0.993 / 0.992 | 0.978 / 0.965 / 0.963 |
| detection rate at α, widened range [−35, +10] dB | 0.889 / 0.934 / 0.936 | 0.679 / 0.646 / 0.638 |
| JSR [dB] where P(det) ≥ 0.5: barrage · tone · pulse comb · protocol-aware | r0 −30 · −34 · −32 · −24; A −34 · −36 · −34 · −28; B −36 · −36 · −36 · −28 | r0 −20 · −20 · −20 · −12; A −20 · −16 · −22 · −8; B −20 · −16 · −20 · −8 |

1. **At 30 dB the CNN's weakness was largely its training range.** One retrain that only extends Li's
   jammer range down to −35 dB — never shown a generator — raises the CNN's P(det) at matched damage on
   the held-out seeds from 0.16–0.18 to 0.58–0.74, level with the energy detector on the same frames
   (0.60–0.81). Seeing the attacker (B) adds 9–19 pp. **The round-0 attackers and their twins stay below
   energy even after B** (0.44–0.48 vs 0.57–0.65), so the retrain narrows the gap but does not close it for
   the attackers that were trained against r0. The deployed CNN had seen no jammed frame below −20 dB;
   the attackers work at −23.
2. **At 15 dB nothing moves.** Under every defender — B included, trained on the attackers' own frames —
   the CNN flags the generators on 0.04–0.09 of frames at matched damage, against 0.22–0.43 for energy.
   B could not even learn its extra class: validation accuracy on the generator frames 0.52, i.e. chance
   (0.70 at 30 dB). At −23 dB JSR the generator sits 8 dB under the 15 dB noise floor. **§3.1 finding 2
   holds at 15 dB, as a limit of this CNN rather than of its training data.**
3. **The cost the user predicted is real at 15 dB and negligible at 30 dB.** At 15 dB the retrained CNNs
   lose 4–5 pp of detection on Li's own classes at α (0.928 → 0.889 / 0.881); argmax accuracy falls
   0.936 → 0.916 / 0.878 and argmax FAR rises 0.013 → 0.068 / 0.184 — the Phase 0.5 pattern, and **most
   of it comes from widening the range (A), not from the attacker's frames (B)**. Mechanism, from the
   training reports: at 15 dB a −35 dB frame is 20 dB under the noise, so many frames labelled "jammed"
   are indistinguishable from clean ones — label noise, Phase 0.5's own caveat (validation accuracy on
   clean frames 1.00 → 0.96 (A) → 0.73 (B)). At 30 dB −35 dB is 5 dB under the noise and learnable:
   argmax accuracy does not drop (0.962 → 0.972 / 0.969), detection at α moves −0.3 / −0.6 pp, and B's
   argmax FAR rise (0.002 → 0.030) is absorbed by the calibrated threshold.
4. **A retrain moves the blind spot as well as shrinking it.** At 30 dB both retrained CNNs flag Amuru
   pulsed more at low JSR and **9–20 pp less at −12…−6 dB**, where it breaks 47–100 % of frames (matched-
   damage P(det) 0.44 → 0.40 / 0.38; its PER at P(det) ≤ 0.5 jumps 0.08 → 0.70 / 0.84, a knife-edge
   read-out — the −10 dB point sits at P(det) 0.50 / 0.49). At 15 dB matched QPSK falls 0.95 → 0.78–0.80
   and Amuru 0.23 → 0.17 / 0.15 at matched damage. Energy flags Amuru at 0.79–0.94 throughout.
5. **Roles.** The twins track the seen generators under A as well, which saw neither, so "twin" measures
   generator similarity, not exposure. The held-out seeds were weaker evaders to begin with (0.16–0.18 vs
   0.09–0.11 under r0) and are the ones caught hardest after the retrain.

**What it means for the story — the user's call, not decided.** "The learned detector is the weak link"
becomes conditional on SNR: at 30 dB one attacker-agnostic retrain removes most of it (at no measurable
cost) though the round-0 attackers keep an edge; at 15 dB it holds against every retrain tried, and
retraining costs the CNN 4–5 pp on the jammers it was built for. That is the adaptation-cost measurement
the supervisor asked for (§2.9). 15 dB is the defensible operating point for a QPSK UAV link (§4.2 Q13),
and the next experiment (§3.4 "Next experiment") repeats D6 there with 15 dB-trained attackers and an
energy detector under noise-level uncertainty.

**Caveats.**
- **One retrain per arm, no seed replicate.** Retrain noise from existing data: S1's Li-range retrains on
  an ≈ ideal link (σ ≤ 0.03 dB, same recipe and seed, §3.3k) flag the CNN-targeted generator at 0.08–0.09
  at matched BER against the deployed 0.13 — far below finding 1's shift. A same-day Li-range control at
  30 dB (evaluate `surrogate_s12`, one job) was not run.
- At 30 dB r0 is the deployed CNN (trained 2026-09-17); A and B are fresh retrains.
- The widened range was chosen for 30 dB. At 15 dB a range scaled to the noise (JNR ≥ −5 dB, i.e. JSR ≥
  −20 dB) is Li's own range, i.e. r0 — which catches nothing either, so finding 2 does not hinge on it.
- The attackers were trained at 30 dB; the 15 dB rows are transfer, as in E2. S4 (§3.3m) retrains the
  attacker at 15 dB against D6's 15 dB r0 CNN.
- Paired evaluation needs one GPU model (CUDA's random streams differ between models, §C.4):
  `submit_arms_eval.sh` pins `titan_rtx`.
- `arms_figures.py` still prints the rejected frames-per-alarm columns and draws them in `fig_gain.png`;
  replace them with energy P(det) at matched damage before any figure is reused.

**Figures** (`arms_figures.py --figs`): `artifacts/cgan/baselines/arms/eval/round1/fig_gain.png` (per
jammer and defender: P(det) at matched BER, and the rejected frames-per-alarm panel) and `fig_cost.png`
(P(det) vs JSR on Li's four types). Report page (2026-09-28, private, old metric — see the note at the
top): <https://claude.ai/artifact/KQXwSSavTTy1uFECTVseGT>.

**Storage:** the five retrained CNNs and their LRT caches (180 MB) live on net_scratch,
`/itet-stor/rrahman/net_scratch/cgan_arms/<defender>/`, symlinked from `artifacts/cgan/baselines/arms/`
(home quota, §C.4); thresholds, training reports, eval JSONs and figures are in git. Merged to main as
79375cb (2026-09-29).

**Jobs:** 2275914 verify (full suite, exit 0) · CNN retrains 2275915 tasks 0, 2–4 (task 1 cancelled: its
tikgpu06 GPU was thermally throttled, §C.4 → 2275949) · 2275958, 2275950 verify `--arms-only` after
training (exit 0) · evals 2275916 (30 dB r0), 2275960 (30 A), 2276002 (30 B), 2275959_3 / _4 (15 r0 / A),
2276037 (15 B; 2275959_5 was a `TaskProlog` transient), all on one TITAN RTX node. Earlier: 2275902 /
2275903 failed §23's first scale check (bit-exact, too tight), 2275904 cancelled with them; 2275953 the
GPU-model RNG diagnostic.

## 3.3q Final experiment — six environments: the CNN-targeted GAN gains nothing over the literature baselines; the realism factors split the detectors (2026-09-30)

**What ran.** §3.4 "Final experiment", as specified, in `final/` (files: §1.3). Six environments at our
15 dB (Es/N0 24 dB); per environment Li's CNN retrained + every detector CFAR-calibrated on 20k clean frames
of that environment, automatic bands, 8 generators (control ×3, `cnn_b10` ×3, `energy_b10`,
`kurtosis_b10`; run003 recipe), and 22 attackers × JSR −50…+15 dB × 512 frames (the random push added the
same day, below). Everything on TITAN RTX.
Figures and tables: `artifacts/final/<env>/` (`fig1_vs_jsr`, `fig2_damage_vs_pdet`, `table.md`,
`summary.json`) and `artifacts/final/compare/` (`fig_compare_pdet`,
`fig_compare_gains`, `fig_dc_share`, `compare.md`, `compare.json`). Report page (private):
<https://claude.ai/artifact/MDtg7r4sLU8rJ8ZwopBEGd>; its source is `artifacts/final/report/index.html` (figures
published as `figs/…`, per-environment tables built in the browser from `data/<env>_summary.json` =
`artifacts/final/<env>/summary.json`).

**Gates.** `verify.py` 97/97 and the regression gate vs S4 (0/2178 P(det) pairs beyond the √2-corrected
4σ, mean z −0.005; BER 44 % positive, unbiased; clean FAR within z ±0.7), job 2277992, exit 0, before
any training; rerun after the random push was added, 100/100 and the gate passes (job 2278412). Every
environment's realised clean FAR on 4096 fresh frames: 0.043–0.057.
Base's CNN reproduces S4/D6's 15 dB r0 (validation accuracy 0.922 vs 0.932, AUC 0.977 vs 0.978, Li's four
types at P(det) 0.5 at the same JSRs); base's bands are S4's (CNN (−39.0, −7.0), energy (−25.1, 0) vs
power (−24, 0)).

**Two README premises were wrong and are corrected (verify §2):**
- *"Energy and full-band power transition within ≈ 1 dB at 30 dB, so power's band reference serves for
  energy"* (§3.4 decision 3, bands step): true for an in-band jammer (async matched QPSK: 1.4 dB, predicted
  1.37), **false for white noise, the probe the band rule uses: 10.3 dB** (predicted 10 log10(1280/128) =
  10). The matched filter passes 1/10 of a white jammer's frame energy while the clean signal × noise spread
  is the same. `bands.py` therefore measures energy's shift on energy's own statistic (30 dB reference:
  energy −16.9, power −27.2, kurtosis −13.6, CNN −29.9 dB).
- *"Energy's noise share ≈ 1.35/8 of full-band power's"*, *"≈ 6× less noise"*: it is **(N/len + N0)/(c0² + N0)
  ≈ 1/10** (measured 0.1035). The MF's noise-equivalent bandwidth is the symbol rate, 1/8 of the band, and
  the frame's pulse tails (1280 samples for 128 symbols) add another 1.25×.

**Result — P(det) at excess PER 0.1** (mean ± std over 3 seeds for `cnn_b10` and the control; `summary.json`
has every attacker, both damage levels and every detector):

| | base | noise | noise_1db | fading | fading_k28 | both |
|---|---|---|---|---|---|---|
| **CNN** — GAN vs CNN (β 10) | 0.107 ± 0.018 | 0.103 ± 0.036 | 0.052 ± 0.016 | 0.097 ± 0.012 | 0.132 ± 0.045 | 0.128 ± 0.067 |
| **CNN** — control (β 0) | 0.171 ± 0.106 | 0.055 ± 0.006 | 0.057 ± 0.012 | 0.058 ± 0.007 | 0.075 ± 0.018 | 0.052 ± 0.008 |
| **CNN** — classical envelope (min) | 0.102 (Amuru p 0.02) | 0.114 | 0.098 | 0.092 | 0.096 | 0.108 |
| **CNN** — Zhou's CGAN | 1.000 | 1.000 | 0.984 | 0.999 | 1.000 | 0.990 |
| **energy** (naive) — GAN vs CNN | 0.238 | 0.250 | 0.224 | 0.043 | 0.060 | 0.060 |
| **energy** — classical envelope | 0.147 (on/off) | 0.150 | 0.151 | 0.042 | 0.050 | 0.042 |
| **energy with CSI** — GAN vs CNN / control | — | — | — | 0.248 / 0.243 | 0.236 / 0.255 | 0.452 / 0.224 |
| kurtosis — GAN vs CNN / envelope | 0.109 / 0.141 | 0.087 / 0.130 | 0.097 / 0.129 | 0.113 / 0.136 | 0.074 / 0.140 | 0.377 / 0.132 |
| energy CFAR threshold over clean mean | +1.29 % | +1.29 % | +1.31 % | +60.7 % (CSI +1.27) | +9.5 % (CSI +1.28) | +60.4 % |

At excess BER 3e-4 the classical envelope is the on/off jammer at the FAR on every detector in every
environment (it jams ≈ 1 frame in 1500), so no attacker can beat it there; the attackers differ only at
PER 0.1. The CNN-targeted GAN's gains against the CNN (Δ P(det), pp, ± SE over seeds, PER 0.1):

| | base | noise | noise_1db | fading | fading_k28 | both |
|---|---|---|---|---|---|---|
| over the classical envelope | +0.5 ± 1.0 | −1.1 ± 2.1 | **−4.6 ± 0.9** | +0.6 ± 0.7 | +3.6 ± 2.6 | +2.0 ± 3.9 |
| over Zhou's CGAN | −89.3 | −89.6 | −93.2 | −90.1 | −86.8 | −86.2 |
| over the damage-only control | −6.4 ± 6.2 | +4.8 ± 2.1 | −0.5 ± 1.2 | +4.0 ± 0.8 | +5.7 ± 2.8 | +7.6 ± 3.9 |
| room: genie flip over the envelope | −4.9 | −6.3 | −4.9 | −4.1 | −5.4 | −6.4 |

1. **Against the CNN the CNN-targeted GAN has no meaningful gain over the literature baselines, in any
   environment.** Over the classical envelope it is within ±4 pp except `noise_1db` (−4.6 ± 0.9 pp), where it
   reaches the CNN's FAR floor and the envelope stays at 0.098 — but the control and the energy/kurtosis
   GANs reach 0.057–0.078 there too, so the gain belongs to learned waveforms in general and to the CNN
   losing sensitivity, not to the detector term. The room is small everywhere: the genie flip is only 4–6 pp
   below the envelope. **The extended Amuru family closes the gap:** p 0.02 / 0.01 (new in this run) reach
   0.09–0.12 on the CNN; p 0.1 alone (0.32 at base) would have suggested a large gain.
2. **The detector term buys nothing.** The control is as hard or harder for the CNN to detect as `cnn_b10`
   in every environment (+4 to +8 pp in the control's favour in four of six); base's −6.4 pp is one control
   seed at 0.277 (the others 0.171, 0.066).
3. **Zhou's CGAN is the only baseline the GAN beats by a lot** (−86 to −93 pp everywhere, largest at
   `noise_1db`). It imitates the clean waveform and is detected like matched QPSK (≈ 1.0): a weak baseline.
4. **Noise-level uncertainty hurts the CNN, not energy.** Energy after the MF keeps its CFAR threshold at
   +1.3 % up to σ_N 1 dB and flags the generators at 0.22–0.25 as at base; the CNN's white-noise transition
   rises +2.6 / +4.9 dB and it flags every learner at ≈ α at 1 dB (its argmax FAR on Li's classes 0 → 0.032).
5. **Fading blinds the naive energy detector, not the CNN.** At K 12 dB every learned and Amuru jammer is at
   ≈ α on naive energy; at K 28 dB still 0.05–0.08 (barrage 0.42, tone 1.0). Energy with the true gain is
   unmoved (0.24–0.26 for the generators, S1's finding with Rician fading). The CNN is unmoved by fading.
   Kurtosis is unmoved everywhere (scale-free).
6. **On energy at base/noise, on/off beats every generator** at PER 0.1 (0.147–0.151 vs 0.22–0.38).
7. **DC share:** every learner 3e-4–1.5e-2 (tone 1, matched QPSK 0.008, white noise 1e-3). No generator is
   the supervisor's constant vector.
8. **Against Amuru's best p alone (the figures' baseline since 2026-09-30) the verdict is the same.** At PER
   0.1 on the CNN it IS the classical envelope in every environment. At BER 3e-4, where on/off made the
   envelope unbeatable, the CNN-GAN's gain over Amuru's best p is −0.1 ± 1.1 / +0.4 ± 1.7 / **−4.0 ± 0.8**
   / +1.5 ± 0.7 / +4.2 ± 2.2 / +1.1 ± 3.7 pp (base … both). The best p is 0.01 or 0.02 against the CNN and
   kurtosis in every environment: the edge of the family, so p < 0.01 might do better still (not run).
9. **The random push** (`random_push_e0.1`, user 2026-09-30: *"should've been random push, we don't know the
   signal either"*): the genie push's timing and carrier phase without its data, +1.1·u on ⌊δN⌋ symbols,
   δ = JSR/1.21, u uniform QPSK independent of s, not scaled by the fading gain. verify: BER 0.03693 vs
   0.03688 predicted, SER 0.0567 vs 0.0564, one-sided energy flags it (0.98) at −10 dB. **It is the most
   damaging attacker per dB** (BER ≈ JSR/2.4 vs the genie flip's JSR/4) and **the CNN barely sees it**: at
   −20 dB, its first damaging point, P_det(CNN) = 0.082 / 0.061 / 0.074 / 0.047 / 0.047 / 0.062, below the
   CNN-GAN at PER 0.1 in five of six environments. Energy flags it 0.41–0.44 without fading (CSI 0.42–0.44),
   kurtosis 0.23–0.29. **But it overshoots:** the hard per-frame power cap means whole symbols, nothing below
   −20.2 dB and PER 0.55–0.72 at −20 dB, so every matched-damage number of it is read at that first point and
   overstates its P_det at PER 0.1 (diluted over frames like on/off, energy at base would be ≈ 0.10 — our
   construction, not computed). It needs synchronisation no asynchronous attacker has: a reference, not a
   jammer.

**§3.4's predictions, blunt:** "at base the detector term may buy nothing" — held, everywhere. "Large gain over
Zhou" — held. "Large gain over the classical envelope (Amuru p 0.1: energy 0.88, CNN 0.23)" — **failed**: p 0.1
is where predicted (0.80 / 0.32), but the envelope with p down to 0.01 and on/off is level with the GAN.
"Energy after the MF shrinks S5's effect" — held (margin +1.29/1.29/1.31 % vs S5's full-band 1.25/1.49/2.08 %).
"Naive energy near-blind under fading even at K 28; energy_csi unmoved; the CNN open" — held; the CNN is unmoved.

**Where is the gain against the CNN largest?** At `noise_1db` on all three gains, and nowhere by much. The
system-model choice stays by literature realism (`both`, §3.4 "Open"): no environment favours the GAN by more
than 5 pp. What holds robustly is the detector side (findings 4–5).

**Caveats.**
- **The band rule can starve a generator of low-power training.** Where the target only reacts at high JSR
  the band follows it up: `energy_b10` under fading trained on (−8, 0) dB (`fading_k28`: (−16, 0)) and reaches
  PER 0.1 only at −10 / −13 dB, flagged 1.00 / 0.25 by the CNN. In `both`, 2 of 3 `cnn_b10` seeds (band
  (−35.8, −3.8)) ended with a damage onset at −11 / −12 dB instead of −21; seed r1 (0.054 on the CNN) ties the
  control. Training outcomes, not crashes; nothing was rerun (it would change the recipe).
- One CNN per environment (seed 11), one seed for `energy_b10` / `kurtosis_b10`; `energy_csi` is handed the
  true gain; fading and noise level drawn independently per frame (changes no per-frame number).
- On/off is analytic; its silent frames are charged the measured clean FAR, not exactly α.
- The figures (redrawn 2026-09-30) draw six attackers per detector column, Amuru as its single best p at PER
  0.1 (`best_p`); the envelope, all seven p and every other attacker are in the tables only. The genie
  push is table-only, replaced in the figures by the random push.

**Jobs** (all exit 0 unless noted): 2277992 verify + gate · smoke chain on `both` 2277993–2277996 (2277995_7
`TaskProlog` → 2278004, eval 2278005) · chains: base 2278006 / 2278007 / 2278008 / 2278009; noise 2278010 /
2278011 / 2278012 / eval 2278063; noise_1db 2278014 / 2278015 / 2278016 / 2278064; fading 2278018 / 2278019 /
2278020 / 2278069; fading_k28 2278022 / 2278023 / 2278024 / 2278076; both 2278026 / 2278027 / 2278028 /
2278084. **Failed: 2278012_7, 2278016_7, 2278020_7, 2278024_7, 2278028_7** (every kurtosis generator but
base's): `TaskProlog failed status=1` at 0 s, always the last task of an eight-task burst (the node TaskProlog
runs `nvidia-smi` + an `scontrol` comment update); resubmitted unchanged as 2278057, 2278059, 2278068,
2278075, 2278083, all fine; the waiting evaluates 2278013 / 17 / 21 / 25 / 29 were cancelled and resubmitted.
Random push (2026-09-30): verify + gate 2278412, then `evaluate.py --only random_push_e0.1` per environment
2278413 (base) … 2278418 (both), chained `afterok`, ≈ 21 s of sweep each.
CNN retrain 8.5–9 min, generator 1.7 min (analytic) / 6.5 min (CNN), evaluate 6.5 min, verify 6.6 min.
Checkpoints (2.0 GB) on net_scratch `final/<env>/`, symlinked.

## 3.4 The plan (20 days) — re-cut 2026-09-12 for the pivot

### Final experiment — one pipeline, six environments (`final/`; DECIDED 2026-09-29, BUILT AND RUN 2026-09-30 — results §3.3q)

> **STATUS 2026-09-30 (session B done).** `final/` built (§1.3); verify 100/100 + the regression gate vs S4
> pass (job 2278412, after the random push was added); all six chains complete, figures + cross-environment
> comparison drawn (§3.3q; redrawn the same day: stacked damage/P_det vs JSR, six attackers per detector). Blunt
> outcome: the CNN-targeted GAN has no gain over the classical envelope against the CNN beyond 4.6 pp
> (`noise_1db`) and the detector term buys nothing; the realism factors split the detectors (noise
> uncertainty → the CNN, fading → naive energy). Two premises below were wrong and are corrected in
> place (marked *corrected 2026-09-30*): energy and full-band power do NOT transition alike for white
> noise (10 dB apart), and energy's noise share is ≈ 1/10 of power's, not 1.35/8. The rest of the spec is
> what ran. **Open:** the system-model choice and what goes into the paper (Wed 30.9 read-out; §4.2 Q14).

**Why (user, 2026-09-29):** put the results into *one* experiment. Every attacker runs against every
detector, with jammers and detectors retrained at one realistic operating point. The same comparison is
then repeated under noise-level uncertainty, under fading, and under both. It answers the supervisor's
2026-09-29 points (§B.1):
- P_det is specified (§2.7).
- Outperformance is quantified against literature baselines.
- His naive constant-vector attack is in the baseline set.
- Channel uncertainty and time variance are the realism axes.
- The modulation is fixed.

**The question:** in which environment is the CNN-targeted GAN's gain over the baselines, *against
the CNN*, largest? And what does each realism factor do to each detector?

**Schedule (user):** sessions A and B (below) run overnight 29/30.9, and each ends in a report. Wed 30.9
is the read-out with the user. Thu–Fri are for writing (meeting Thu 17:00).

**Not part of it:**
- **D6 (the defender retrains on the attacker) is PARKED.** User: *"we keep the D6 part for later, it
  really depends on how this goes and i want to make a clear distinction here"*. Its 15 dB spec is the
  next subsection. When it resumes, it runs on `final/`'s pipeline.
- **The noise LRT is dropped.** It is exact only for barrage, and the supervisor was not convinced by
  the detector model.
- **The D2a shaped control is left out** (user, 2026-09-29). It is optimised separately at each JSR,
  ≈ 33 CMA-ES runs per detector per environment, and it answered a different question: shaping vs a
  GAN (§3.3e).
- ξ, the SPRT detection delay and the naive (non-CFAR) defenders are not reported.

**Decisions (user, 2026-09-29):**
1. **Operating point.** Our SNR 15 dB = **Es/N0 24 dB** (Eb/N0 21 dB). `cgan/`'s SNR is power per
   sample over the 8× simulated band ([cgan/link.py:31-33](cgan/link.py#L31-L33), sps 8), so
   Es/N0 = SNR + 9.03 dB. The paper states Es/N0. The literature case is a fixed-QPSK UAS control link
   with link margin (§4.2 Q13). The adaptive-modulation argument goes: at Es/N0 24 dB the CQI mapping
   would pick 64QAM.
2. **P_det** is defined in §2.7. Under the CFAR rule, every detector is calibrated on 20k clean frames
   drawn from the environment it is evaluated in.
3. **Energy is measured after the matched filter.** The statistic is `energy` = mean_k |z_k|², the mean
   received symbol energy.
   - Why: the full-band power (`cgan` `detectors.power`) also collects the noise outside the signal's
     band. Relative to the signal, full-band power holds ≈ 10× the noise the MF output does: the MF's
     noise-equivalent bandwidth is the symbol rate (1/8 of the band), and the frame's pulse tails
     (1280 samples for 128 symbols) add 1.25× (measured 0.1035, verify §2; *corrected 2026-09-30 — this
     bullet first said "1.35/8, ≈ 6×"*).
   - Consequence: noise-level uncertainty barely moves energy's CFAR threshold (+1.29 / 1.29 / 1.31 % at
     σ_N 0 / 0.5 / 1 dB; S5's full-band power: +1.25 / 1.49 / 2.08 %, §3.3n, §3.3q).
   - The clean spreads coincide (the signal × noise cross term), **but the jammer side does not** (*corrected
     2026-09-30; this bullet first said "the two nearly coincide"*): an in-band jammer is seen alike (async
     matched QPSK: energy's transition 1.4 dB above power's), a white jammer is not (10.3 dB above: the MF
     passes 1/10 of its frame energy). §3.3q.
   - The one-sided form is the detector the energy-targeted GAN trains against. The **two-sided form is
     evaluated but not trained against** (added at r2c 2026-09-29). It is the same statistic, and the
     genie *push* lowers received power: only a two-sided test sees it (§3.3d).
   - The CNN keeps Li et al.'s full-band spectrogram. Kurtosis stays on r (it is scale-free).
4. **The victim-link factor is per-frame Rician block fading, not S1's log-normal shadowing.**
   - Shadowing is slow. It decorrelates over tens of metres, i.e. thousands of frames, and the
     receiver's AGC tracks it. The frame-to-frame variation is small-scale fading.
   - **K = 12 dB is the main value, 28 dB the extra level.** Matolak & Sun's NASA air–ground
     measurements of UAS control-link (CNPC) channels, in the two CNPC bands, gave K ≈ 12–13 dB in
     L-band (~970 MHz) and ≈ 27–31 dB in C-band (~5 GHz), over urban, hilly and sea terrain (§4.2 Q13).
   - Frame-power std ≈ 1.5 dB at K = 12 dB and ≈ 0.24 dB at K = 28 dB.
   - Amplitude only, at unit mean power. The receiver knows its phase (System Model), so there is no
     equaliser and clean BER stays ≈ 0.
   - **Per-frame P_det and damage depend only on the per-frame marginal of the gain.** Drawing it
     independently per frame therefore changes no reported number, only multi-frame reasoning.
     Consecutive frames are in fact correlated: the coherence time is milliseconds, and a frame is
     128 µs at 1 MBd.
   - Energy is bracketed by two detectors. `energy` is calibrated on faded clean frames. `energy_csi` is
     handed the true gain and subtracts (g² − 1)·c0², as `power_csi` did. A real receiver lies between
     them, so no gain estimator is built.
5. **Noise-level uncertainty: σ_N = 0.5 dB is the main value, 1 dB the extra level.** It is a per-frame
   log-normal factor on the noise variance, as in S5.
   - 0.5 dB: a calibrated receiver (Tandra & Sahai's ±1 dB ≈ σ 0.5; Nikonowicz et al.'s 0.5 dB RMSE on
     real ISM traces).
   - 1 dB: in-band interference, which has to be argued in the paper.
   - Sources: §4.2 Q13.
6. **The figure set is frozen**, identical for every environment and detector (§2.7): (1) damage vs JSR,
   (2) P_det vs JSR, (3) damage vs P_det, (4) one table of P_det at excess PER 0.1 and excess BER 3e-4.
7. **Seeds.** 3 for the CNN-targeted GAN and 3 for the damage-only control; 1 each for the energy- and
   kurtosis-targeted GANs.
   - The control's 3 seeds were added at r2c 2026-09-29, after the user said *"showing the damage only
     GAN could still be interesting"*.
   - "What the detector term buys" is the difference between those two generators, and S2 showed one
     seed can mislead.
8. **Every implemented baseline stays in** (user: *"we can always decide which to leave out for
   paper"*).
9. **Folder** `final/`.

**Open, for after the run:** which environment becomes the paper's system model. The recommendation
(2026-09-29) is to choose by literature realism (`both`) and report the other five as ablations, not to
pick the one that favours the GAN. The user's question stands: where the GAN's gain against the CNN is
largest.

**Environments** (`final/env.py`). All are at our SNR 15 dB = Es/N0 24 dB. Run all six regardless of
how the others come out: an interaction can flip a sign.

| env | σ_N | victim-link fading | role |
|---|---|---|---|
| `base` | 0 | none | the ideal link at the operating point |
| `noise` | 0.5 dB | none | noise-level uncertainty, calibrated receiver |
| `noise_1db` | 1 dB | none | the same with in-band interference (extra level) |
| `fading` | 0 | Rician K 12 dB | L-band fading |
| `fading_k28` | 0 | Rician K 28 dB | C-band fading (extra level) |
| `both` | 0.5 dB | Rician K 12 dB | the combination: the candidate system model |

**Attackers** (every environment):

| role | attacker | code | note |
|---|---|---|---|
| upper bound | genie flip, `omniscient` η = 1 | `attacks.omniscient_rx` | received law = clean law, BER = δ, invisible to every test; in every figure (§2.9) |
| reference | genie push, `omniscient` η = 0.1 | same | cheapest per bit; lowers received power, so only two-sided energy and kurtosis see it (§3.3d). Table only since 2026-09-30 |
| reference | random push, `random_push` η = 0.1 (added 2026-09-30) | `attacks.random_push_rx` | the genie push without the data: +1.1·u, u random QPSK, on ⌊δN⌋ symbols; knows timing and carrier phase only. BER ≈ δ/2, raises received power. Drawn in the figures in place of the genie push (§3.3q finding 9) |
| classical | barrage `noise` (= Li's `barrage`) | `attacks.noise_tx` | |
| classical, literature | Amuru pulsed, p ∈ {1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01} | `attacks.pulsed_tx` | p = 1 is matched QPSK (Zhou's "optimal"); p = 0.25 is Li's `protocol_aware`, so that type is not run twice. In graph 3 it is ONE line, the envelope over p: needed before any "beats Amuru" claim (§4.3) |
| classical | on/off: matched QPSK at +15 dB on a fraction f of frames | analytic, from the p = 1 point at +15 dB, as E2b (§3.3g) | P_det = f·P_det(+15) + (1 − f)·α, damage = f·damage(+15). Graphs 1–2 plot against mean JSR +15 + 10 log10 f. Wins on average BER (§2.7) |
| the supervisor's naive baseline | constant IQ vector = Li's `tone` (constant complex baseband, phase uniform per frame) | `attacks.li_tx("tone")` | the CNN is trained on this type; say so. **Diagnostic** for *"how do we know the learner does not learn exactly this?"*: each generator's per-frame DC share \|mean_n j\|² / mean_n \|j\|² (1 for `tone`) |
| classical | Li's pulse comb | `attacks.li_tx("pulse_comb")` | impulses every 64 samples |
| literature generator | Zhou's plain CGAN `artifacts/cgan/run001_G.pt` | evaluated, not retrained | trained to imitate the clean waveform, with no link or detector, so independent of SNR |
| learned control | GAN, damage only (β = 0; `cgan` task 0) | trained per env, **3 seeds** | isolates what the detector term buys |
| learned | GAN vs `energy` one-sided, β 10 (`cgan` task 2, target → the new energy) | 1 seed | |
| learned | GAN vs kurtosis, β 10 (`cgan` task 10) | 1 seed | |
| **learned, headline** | **GAN vs CNN, β 10 (`cgan` task 14)** | **3 seeds** | |

The recipe is run003's: `--init random --steps 4000`, α = 0.05, trained white-box against *that
environment's* CNN and calibration, with that environment's σ_N and K applied to the training frames.
In `final/train_gan.py` the tasks are named `control`, `energy_b10`, `kurtosis_b10` and `cnn_b10`
rather than numbered.

**Detectors** (every environment):
- `energy`, one-sided, after the matched filter;
- `energy_2s`, two-sided, evaluated only;
- kurtosis, two-sided, on r;
- the CNN, retrained per environment on that environment's frames. Li et al.'s recipe, seed 11,
  jammed-frame JSR [−20, +10] dB (Li's own range; at our 15 dB it is scaled to the noise, §3.3o caveat);
- `energy_csi`, in the three fading environments only.

**Pipeline per environment** (one `--dependency=afterok` chain; the six chains run in parallel):
1. **`train_cnn.py --env E`** → `artifacts/final/E/cnn/`: `detector_spec.pt`, `thresholds.json` (every
   detector's CFAR threshold and clean quantiles on 20k clean frames of E), and the training report.
   ≈ 10 min.
2. **`bands.py --env E`** → `bands.json`, the JSR band each target is trained on.
   - The rule is S4's (§3.3m), made automatic. Take the 30 dB band (`cgan/train_gan.JSR_BANDS`: CNN
     (−48, −16), others (−32, 0)) and shift it by how far that target's transition moved. The transition
     is the JSR where white noise reaches P_det 0.5, measured against E's defender vs the deployed 30 dB
     one. Cap the upper edge of analytic targets at 0 dB. The control keeps (−32, 0).
   - Energy's 30 dB reference is energy's own transition, calibrated on 20k clean 30 dB frames (−16.9 dB),
     not power's (−27.2 dB): for white noise the two are 10 dB apart (*corrected 2026-09-30; this bullet
     first said they transition within ≈ 1 dB, so power's reference would serve*). The shift is what the
     rule uses, and it came out as S4's (base: (−25.1, 0) vs S4's (−24, 0)). §3.3q.
   - **Known weakness (found 2026-09-30):** where the target only reacts at high JSR the band follows it up
     and the generator never trains at low power (fading: energy band (−8, 0) → a loud generator). §3.3q
     caveats; §4.2 Q14.
   - Why it matters: trap 1 of §3.1. A band outside the target's transition gives the detector term
     no gradient.
3. **`train_gan.py --env E --task T --rep r`**: 8 generators (control × 3, `cnn_b10` × 3, `energy_b10`,
   `kurtosis_b10`). Output goes to `artifacts/final/E/gan/`, with checkpoints on net_scratch
   `/itet-stor/rrahman/net_scratch/final/E/`, symlinked (home quota). ≈ 7 min each, in parallel.
4. **`evaluate.py --env E`** → `artifacts/final/E/eval.json`, pinned `--constraint=titan_rtx` (§C.4).
   ≈ 20–40 min.
   - Every attacker over JSR −50…+15 dB in 1 dB steps, 512 frames per point.
   - Per point: BER, SER, PER (with the clean values of E, for the excess), P_det of each detector at
     α = 0.05, and the DC share.
   - No confirmation pass: no pick at α is reported.
5. **`figures.py --env E`** (login node) produces the outputs of decision 6 (since 2026-09-30 graphs 1 and 2
   are one stacked figure, and the graphs draw six attackers per detector; §1.3).
   **`figures.py --compare`** gives, per detector, every attacker's P_det at matched damage across the
   six environments, plus three gains of the CNN-targeted GAN (mean ± std over seeds):
   - over the classical envelope: the minimum over barrage, the Amuru envelope, on/off, `tone` and the
     pulse comb;
   - over Zhou's CGAN;
   - over the damage-only control, i.e. what the detector term buys.

**What goes into `final/` (session B).** Code is copied, never imported from `cgan/`: both folders have
`link.py`, and flat sibling imports collide. Each copy's first line reads
`# copied from cgan/<file> at 68c97a4 (2026-09-29), pruned`. Keep the mitsuba `set_variant` preamble
wherever Sionna is imported.

| file | from | keep / change |
|---|---|---|
| `env.py` | new | the six environments, N_SYM = 128, artifact and net_scratch paths |
| `link.py` | `cgan/link.py` | `Link`, `noise_scale`; `shadow_gain` → `fading_gain(K)` (Rician amplitude, unit mean power); drop the Zhou-calibration helpers |
| `channel.py` | as is | async delay and phase |
| `attacks.py` | `cgan/attacks.py` | noise, pulsed, omniscient, `li_tx`, `gan_tx`, `frames`, error counts, `log_expected_ber` / `_per`; drop shaped, learned power and the team `rx` hook |
| `detectors.py` | `cgan/detectors.py` | + `energy` (mean \|z_k\|²) and `energy_csi`; kurtosis, `two_sided`, spectrogram + CNN, `calibrate`, `p_detect`, `soft_pdet`, `stat_quantiles`; keep `power` (full band) for the regression gate only; drop the LRT |
| `defender.py` | `Defender` and `measure` from `cgan/baselines.py` | detectors `energy`, `energy_2s`, `kurtosis`, `spec_cnn` (+ `energy_csi`) |
| `models.py` | as is | `Generator`, `load_generator` |
| `train_cnn.py` | `cgan/train_spectrogram_cnn.py` | `--env`; σ_N and fading on training AND calibration frames; the new thresholds; drop the LRT parts |
| `train_gan.py` | `cgan/train_gan.py` | `--env`, named tasks, bands from `bands.json`; drop sync, learned power, shadow |
| `bands.py`, `evaluate.py`, `figures.py` | new; `matched_per` & co from `cgan/probe_readout.py` | as above |
| `verify.py` + `submit_*.sh` | new; S5's checks from `cgan/verify.py` §22 | below; `submit_env.sh E` chains steps 1–4 |

**`final/verify.py`** is the gate for everything:
- closed forms: clean BER = Q(√(Es/N0)) with Es/N0 = SNR + 9.03 dB; the white-noise BER closed form;
- `energy`: clean mean = c0² + N0; noise share (N/len + N0)/(c0² + N0) ≈ 1/10 of full-band power's; at
  30 dB, `energy`'s white-noise transition 10 log10(1280/128) = 10 dB above `power`'s and its async
  matched-QPSK transition ≈ 1.4 dB above (*as built and passed; the spec first said "1.35/8" and "within
  ≈ 1 dB", both wrong, §3.3q*);
- Rician: unit mean power; K → ∞ gives g = 1; the empirical K matches; clean BER ≈ 0 at K = 12 dB;
- σ_N: S5's factor law, and σ_N = 0 draws nothing;
- CFAR: each detector's realised FAR on fresh clean frames of every environment within binomial
  tolerance of α;
- `bands.py` at `base` returns ≈ S4's bands (CNN (−39, −7), power (−24, 0));
- `tone` has DC share 1; the analytic on/off matches a simulated on/off at one f; the genie flip has
  BER = δ and P_det ≈ α on every detector; two-sided energy flags the genie push; the random push matches
  its BER and SER closed forms and one-sided energy flags it (added 2026-09-30).

**Regression gate, before any training.** Use the copied code with full-band `power`. Evaluate S4's four
generators (`artifacts/cgan/gan/snr15_003`) against S4's defender (`artifacts/cgan/baselines/arms/
snr15_r0`), and compare with `artifacts/cgan/snr_ablation/snr15_run003/snr_snr_15.json` on
one-sided power, kurtosis and the CNN:
- P(det) within the √2-corrected 4σ binomial band;
- BER compared as error counts, as `cgan/regress_snr30.py` does.

The gate is statistical: it is bit-exact only when both jobs run on the same GPU model (§3.3n). **No
environment runs on a copy that fails the gate.**

**Sessions** (user: one fresh session each, overnight, each ending in a report):
- **A — literature and paper text** (in parallel with B, and not blocking it: every value is decided).
  - Verify every value above against the originals: Es/N0 24 dB, σ_N 0.5 / 1 dB, K 12 / 28 dB, α 0.05.
  - Write the Experiment Setup paragraph that argues them, with citations, into
    `paper_drafts/sec_setup_operating_point.tex`, with the bib entries in `paper_drafts/refs_new.bib`.
  - Draw explanatory figures (numpy only, login node) into `artifacts/final/literature/`.
  - Update §4.2 Q13 and publish a report page.
  - If a source contradicts a decided value, it goes at the top of the report. A never edits `final/`.
- **B — build and run.** `final/`, verify, the regression gate, the six chains, all figures, the
  cross-environment comparison, the results in a new §3.3q (§3.3p is reserved for S3), the §A.0 rows,
  and a report page.
- **Wed 30.9, with the user:** both reports, the system-model choice, and what goes into the paper.

**Predictions, stated before running (from transfer data at our 15 dB; blunt)** — *checked 2026-09-30 in
§3.3q: all held except the large gain over the classical envelope, which failed (the envelope with Amuru
p down to 0.01 and on/off is level with the GAN).*
- **At `base` the detector term may buy nothing against the CNN.** In S4's 15 dB-trained generators
  against the 15 dB CNN, at PER 0.1, the damage-only control and CNN β 10 are *both* flagged on 0.11 of
  frames (one seed each, §3.3m). The large gains should be the other two:
  - over Zhou's CGAN: E2 run002 D1 at 15 dB, CNN 0.94 / power 1.00 at matched BER, §3.3g;
  - over the classical envelope: Amuru p 0.1 flagged by energy on 0.88 at PER 0.1 (§3.3m), and by the
    15 dB CNN on 0.23 at matched BER (§3.3o).
- **`energy` after the matched filter should shrink S5's noise-uncertainty effect.** §3.3n's numbers
  used full-band power, so at σ_N 0.5 dB the effect may be small.
- **Under fading, the naive `energy` should be near-blind even at K = 28 dB.** Its clean margin at our
  15 dB is ≈ 1.25 % of frame power (≈ 0.05 dB), against a fading spread of 0.24–1.5 dB. `energy_csi`
  should be unmoved. Only the CNN's fate is open.
- One CNN per environment, with no retrain seed (the §3.3o caveat).

### Background batch 2026-09-28 — five experiments, two tracks (TODO, for fresh sessions)

**Why (user, 2026-09-28):** run all five today in the background, then decide how to continue; the user
drafts Related Work and the Methodology meanwhile. **The experiments are disjoint — keep their results
apart.** S3 was discussed and cut down to one per-symbol detector (W = 1), now in Track 2.

**Isolation rules (both tracks):**
- Every new behaviour sits behind its own flag, default off, and flag-off must reproduce the current
  numbers (as `shadow_db = 0` and `jammer_sync = False` do).
- Own run names and artifact folders, own `verify.py` section, own results subsection, own §3.4 row, own
  §A.0 row. Never write into another experiment's folder or reuse its JSON. **Pre-assigned numbers:**
  verify §21 = S4, §22 = S5, §23 = D6, §24 = S3; README §3.3m = S4, §3.3n = S5, §3.3o = D6, §3.3p = S3
  (§3.3l = S2 exists).
- Generator checkpoints on net_scratch (`/itet-stor/rrahman/net_scratch/cgan_gan/<run>`), symlinked from
  `artifacts/cgan/gan/<run>` (home quota, §C.4).
- `sbatch submit_verify.sh` exit 0 gates every array (`--dependency=afterok`). It runs the whole suite,
  so a failure inside the OTHER track's section is not yours to fix: stop and tell the user.
- `verify.py` and `README.md` are shared by both tracks: re-read right before each edit and keep edits
  local to your own section. File ownership: **Track 1** `train_spectrogram_cnn.py` + any new `arms_*.py`;
  **Track 2** `link.py`, `channel.py`, `attacks.py`, `train_gan.py`, `snr_ablation.py`, `calibrate_snr.py`,
  `detectors.py`, `baselines.py`.
- Report at matched damage AND damage at P(det) ≤ 0.5, plus frames per extra alarm (§2.7).

**Track 1 — D6, the arms race (promised to the supervisor). ROUND 1 DONE 2026-09-28 ([§3.3o](#33o-d6-round-1--the-cnns-gap-was-a-training-range-gap-at-30-db-not-at-15-db-2026-09-28)):
at 30 dB one attacker-agnostic retrain (arm A) lifts the CNN on held-out generators to the energy
detector's level (P(det) at matched BER 0.17 → 0.58–0.74), the round-0 attackers stay below energy; at
15 dB no retrain moves the CNN (≤ 0.09) and retraining costs 4–5 pp on Li's classes. Round 2 not started (user: round 1
first). User decisions 2026-09-28: arm B = 204 generator frames in total at the widened range; held-out
rows kept, same-init "twins" split from the independent seeds; every CNN retrained at 15 dB too.**
Spec as written before the run: the D6 row below (arms A and B,
fictitious play, held-out generators, 2 rounds). Ideal link, 30 dB: S1 showed the CNN is the weak link
at every σ, so no shadowing. Defaults to confirm with the user if in doubt: round-0 attackers = run003
CNN β 1 / 10 (tasks 13 / 14); held out = the β = 10 seed runs `cold002_4k{,_r1,_r2,_r3}/task14_G.pt` and
`run004_grey`; arm A = Li's four types with `JSR_RANGE_DB` widened to (−35, +10); arm B = arm A plus the
round-0 generators' frames as a fifth jammed type. A retrained CNN is evaluated with
`baselines.Defender(L, thr_dir=d, cnn_path=d/"detector_spec.pt")`, as S1 does. Outputs
`artifacts/cgan/baselines/arms/…`, generators `gan/arms_r1/`. **Question:** does one retraining close the
gap — arm A without ever seeing the attacker, arm B with — and does the attacker reopen it in round 2?

**Track 2 — four probes, each isolated.** S2, S4 and S5 were run on branch `track2-s2s4s5` (a git
worktree, `/home/rrahman/BT-track2`) and merged into `main`'s working tree on 2026-09-29, after Track 1;
S3 was added to Track 2 by the Track 1 session while they ran and is **not built**.
- **S2 — the listening jammer (synchronous arrival). DONE 2026-09-28: fails its success test** —
  nothing measurable against power or the CNN at matched damage, 4 seeds each side (§3.3l). Not in the
  system model; no timing-error sweep.
- **S4 — train at 15 dB (E2 Tier 2).** Question: does a generator TRAINED at 15 dB beat ≈ 1 frame per
  alarm against power there? The α = 0.05 margin at 15 dB is 1.66 symbol energies, more than the ≥ 0.5 a
  flip costs (§3.1 finding 1); the sign argument (§3.3k) predicts it still pays ≈ one alarm per broken
  frame. **Decided with the user 2026-09-28:** the defender is the CNN **retrained at 15 dB** — Track 1's
  round-0 defender `baselines/arms/snr15_r0` (job 2275915_0, deployed recipe with `--snr-db 15`; read
  from Track 1's worktree while it existed, now in `main`'s artifacts) — with its
  own calibration, for training AND both evaluations; bands by the stated rule (straddle the target's own
  transition), re-measured against that CNN: power (−24, 0), CNN (−39, −7) (its noise transition sits
  +8/+9 dB above the deployed CNN's at 30 dB), β = 0 unchanged (−32, 0). Code: `train_gan.py --snr-db`
  (+ `--detector-dir` loaded at that SNR), `snr_ablation.py --detector-dir`; verify §21. Generators
  tasks 0, 2, 13, 14, `--init random --steps 4000` → `gan/snr15_003`; baseline = run003 at 15 dB,
  `snr_ablation/snr15_run003_base`. **DONE 2026-09-28: energy flags the best generators on ≈ 25 % of
  frames at PER 0.1 at 15 dB (≈ 63 % at 30 dB), trained at either SNR; training at 15 dB lifts the
  untargeted generator to that level** (§3.3m; the first write-up's "floor holds" was an artefact of the
  dropped unit).
- **S3 — per-symbol detector (W = 1), evaluation only. NOT BUILT (2026-09-29).** Spec: the S3 row below. Code: a `symbol_error`
  statistic in `detectors.py` (max over the frame's symbols of the distance from z_k to the nearest QPSK
  point), added to the `Defender` only when its thresholds file carries it (the `power_csi` pattern), a
  calibration on 20k clean frames into its own folder (`artifacts/cgan/baselines/symerr/`), and a flag in
  `snr_ablation.py` that loads that defender. Evaluate run003 + the classical envelope at 30 dB (and 15 dB
  if cheap); report P(det) per frame at matched damage, damage at P(det) ≤ 0.5, frames per extra alarm,
  next to the energy detector and the CNN. Retraining generators against it is a later step, not now.
  Track 2 owns `detectors.py` and `baselines.py` for this. Numbers: verify §24, README §3.3p.
- **S5 — noise uncertainty (user: "if noise goes up, the FAR should also be higher").** The defender
  does not know each frame's noise level: per frame the noise variance is scaled by a log-normal factor
  with std σ_N dB (unit mean). Two defenders on the same frames: **(i) naive** — thresholds calibrated at
  the nominal noise; report its REALISED FAR (it rises) and damage vs P(det); **(ii) honest CFAR** —
  thresholds calibrated on frames with the uncertainty, so FAR = α and the threshold widens (the SNR wall,
  §3.3k). Grid σ_N ∈ {0, 0.5, 1, 2} dB × SNR {30, 15} dB; run003 evaluated, not retrained; CNN weights
  frozen. **Decided with the user 2026-09-28:** the CFAR defender keeps the naive colour scale and
  re-derives only its thresholds, two-sided centres and LRT clean parts, so both defenders see identical
  statistics and only the thresholds differ. Code: `Link.noise_unc_db` (applied in `attacks.frames`),
  `calibrate_snr.cfar_defender` → `baselines/noise_unc/<level>_unc_<σ>/`, `snr_ablation.py --axis noise`;
  verify §22. Expected: negligible at 30 dB (noise is 0.1 % of the received power, so ±1 dB moves it
  ±0.03 % against a 0.24 % margin); large at 15 dB (±1 % against 1.25 %). **DONE 2026-09-28: at 15 dB the
  naive defender's FAR reaches 0.23 and its alarms carry little evidence; the honest CFAR one lets the
  learned jammers under its widened threshold (CNN β 10 flagged on 14 / 9 % of frames at PER 0.1, 7 / 57
  frames broken before a sequential test decides, at ±1 / ±2 dB); nothing at 30 dB** (§3.3n).

**Measures (2026-09-29, after the user rejected "frames per extra alarm").** That unit is ours (E2b,
§3.3g) and in no paper we found; its best-over-JSR value is mostly the saturated corner 1/(1 − FAR) = 1.05
and it is a maximum over noisy ratios (S2's first seed: 14.4, not replicated). Track 2's results are now
stated in what the literature reports: the damage-detection trade-off (PER vs per-frame P(det) at α,
stealthy-jamming papers), **P(det) at matched PER 0.1 / 0.5** (the main number), the warden's detection
error ξ = min(P_FA + P_MD) (covert communication, Bash et al. JSAC 2013; threshold-free, only where
statistic quantiles were kept), and detection delay — frames broken before a 1 %/1 % Wald SPRT on the
per-frame alarms decides (our combination of standard parts; an interpretation of P(det) at matched PER,
not an independent result). Code: `cgan/probe_readout.py`, `cgan/probe_report.py` →
`artifacts/cgan/probes/report_data.json`. **Not yet applied to E2, E2b or §3.3f**, which still quote
frames per extra alarm; S1 was re-measured for the report (its verdict holds).

**Report (2026-09-29):** <https://claude.ai/artifact/LuFXpEu5cW4SwcPPfwqVu7> (private) — S1, S2, S4, S5
broken down with figures, the measure change, and the continuation options.

**User decisions 2026-09-29, all carried out:** (1) §3.1 (finding 1 and the story) rewritten at merge
time — done; (2) the S5 finding recorded — the 2026-09-28 email's "a plain energy detector isn't fooled"
is false at 15 dB with ±1–2 dB noise uncertainty, and finding 1 holds only for an energy detector that
knows its link gain and its noise level (§3.1); (3) how to continue is decided in a new session — the
catalogue is §4.3 "Future experiments after the 2026-09-28 batch"; (4) merged after Track 1 — done
2026-09-29 into `main`'s working tree, uncommitted (the S4 CNN `arms/snr15_r0` is in `main`'s artifacts;
merged suite verify job 2277747, all 23 sections, exit 0). The literature measures are settled in §2.7.

**Once all five are in:** one comparison with the user of what each changes, then decide the system
model and the paper's story (§4.1 #0c).

### D6 at the realistic operating point: 15 dB, 15 dB-trained attackers, CFAR energy (PARKED 2026-09-29 — after the final experiment)

**Parked (user, 2026-09-29):** *"we keep the D6 part for later, it really depends on how this goes and i
want to make a clear distinction here. but pls do remark it somewhere we will get there eventually."*
It is kept separate from the final experiment above. When it resumes:
- it runs on `final/`'s pipeline, in the environment the final experiment settles on;
- the defender arms are §3.3o's (r0 / A / B′), and the stopping rule is the supervisor's
  2026-09-29 question.

The proposed rule: stop the alternation when a defender retrain no longer moves P_det at matched
damage beyond the seed spread. At our 15 dB that already happened in round 1 (§3.3o). The spec
below predates `final/` and the matched-filter energy detector.

**Why (user, 2026-09-29):** 15 dB is the defensible operating point for a QPSK UAV link (§4.2 Q13), so
redo damage vs P(det) against the baselines there, retrain the CNN on the attacker, and measure whether
adaptation costs. Most pieces exist: E2's baselines at 15 dB; S4's attackers trained at 15 dB
(`gan/snr15_003`, tasks 0 / 2 / 13 / 14, trained against D6's 15 dB r0 CNN); D6's 15 dB defenders r0 / A /
B (§3.3o); S5's noise-level uncertainty and CFAR calibration (Track 2 code, §3.3n). What does not exist is
all of it at one operating point.

**Design (defaults — confirm with the user before building):**
- SNR 15 dB; noise-level uncertainty σ_N ∈ {0, 0.5, 1} dB (Q13's literature range; 2 dB only together
  with the in-band-interference argument). Naive and CFAR energy on the same frames at each σ_N.
- Attackers: S4's four 15 dB generators as round 0; run003 (30 dB-trained) as the transfer reference;
  D6's held-out seeds; the classical envelope with pulsed p ∈ {0.1, 0.05, 0.02, 0.01} (§4.3 "Supervisor
  2026-09-29": the Amuru envelope over p is needed before any "beats Amuru" claim).
- Defenders: CNN r0 and A at 15 dB (exist, `artifacts/cgan/baselines/arms/snr15_{r0,A}`); **B′ = A + the
  S4 CNN-targeted generators' frames** — one new retrain, `train_spectrogram_cnn.py --snr-db 15
  --jsr-range -35 10 --extra-gens ../artifacts/cgan/gan/snr15_003/task13_G.pt
  ../artifacts/cgan/gan/snr15_003/task14_G.pt --out-dir ../artifacts/cgan/baselines/arms/snr15_B2`.
- Metrics (literature-grounded, user 2026-09-29): the full damage-vs-P(det) trade-off (BER and PER, log
  and linear), P(det) at matched damage (BER 3e-4, PER 0.1), P_FA + P_MD where a covert-comm reading
  helps, detection delay (frames to first alarm) as the sequential read-out; the cost side as in §3.3o.
  No frames-per-extra-alarm.
- Code: `arms_eval.py` gains the S4 generators in `GENS`, a pass-through to Track 2's noise-uncertainty
  flag and its CFAR thresholds, and new tasks; `arms_figures.py` drops its frames-per-alarm columns and
  panel for energy P(det) at matched damage. Evals pinned to `titan_rtx` (pairing, §C.4).
- Cost: one CNN retrain (~10 min) + 3 defenders × 3 σ_N evals (~5 min each) ≈ 1 h with the queue.

**Question:** at the realistic operating point, does the learned detector (a) miss the attacker, (b) fail
to recover by retraining, (c) pay for trying — and does CFAR energy still hold at σ_N 0.5–1 dB?
**Open for the user:** round 2 (the attacker retrains against B′ and against CFAR energy); whether the
detection-delay read-out is wanted; whether 2 dB stays in the grid.

### Exploratory CGAN track — ACTIVE since 2026-09-14

Goal, success bar and library decisions: [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).
All code goes in `cgan/`, following M0's layout conventions: flat sibling modules, run from inside the
directory, `verify.py` as the test suite, outputs in `artifacts/cgan/runNNN*`, SLURM logs in
`cgan/runs/`. **Every item runs via sbatch** (Sionna). The coordination compute track further down is
paused while this runs.

| # | Item | Output | Gate |
|---|---|---|---|
| **C0** | **DONE 2026-09-14.** Fig. 6 pixel-digitised by script: a 400-dpi render; axes least-squares-calibrated on the plot's own gridlines (residuals < 1 px; the FEC line independently reads log₁₀ BER = −2.998 vs −3); markers located by legend colour; overlay checked by eye. **The data points are at JSR −10:2:10 dB (11 points); the axis ticks at −10:2.5:10 are not the data grid.** Reading uncertainty ≈ ±0.04 decade / ±0.07 dB where visible. At JSR ≥ 0 dB the curves overlap: 3 Noise markers are fully hidden under GAN (recorded as `null`) and 3 other markers are partly hidden (flagged). What the numbers show is recorded in §4.2 Q7. | `cgan/digitise_fig6.py` → `cgan/paper_fig6.json`, `artifacts/cgan/c0_fig6_digitised.png` | — |
| **C1** | **DONE 2026-09-14 (noise + optimal; `gan` lands with C3).** Verified by C4 (job 2259149). **Sionna QPSK waveform link + the three jammers.** Link: `BinarySource` → Gray QPSK `Mapper` → `Upsampling(sps)` → pulse filter → + AWGN (SNR 30 dB) + jammer → matched filter → `Downsampling` → hard decision → BER. Jammers: `noise` (white complex Gaussian), `optimal` (same modulation and filter, random symbols), `gan(G)`. JSR = mean per-sample jammer power / signal power at the RX input, imposed by a hard power projection. Includes the closed-form `ber_noise_jammer` reference. | `cgan/link.py`, `cgan/jammers.py` | — |
| **C2** | **DONE 2026-09-14 (job 2259157, supersedes 2259150). Checkpoint passed: the link was chosen as stated assumptions, not fitted values — sps 8, RRC 0.35, white noise, async optimal (§2.10); `calibration.json`'s `selected` is deliberately `null`, the decision lives in `link.py` `LINK`.** With those assumptions: Noise crosses 1e-3 at −0.67 dB, Optimal at −6.00 dB (async rrc0.35, RMS 1.44 dec). What the calibration found: **Noise:** no link matches the paper's curve everywhere. The best constant-gain Q-function leaves RMS 0.45 decades. **sps 4** is RMS-best (0.79 dec) but crosses 1e-3 at −3.75 dB vs the paper's −1.32. **sps 8** is 1e-3-best (−0.67 dB, 0.64 dB off) but RMS 1.51 dec, too steep at low JSR. The gain that hits the paper's crossing exactly is 8.49 dB ≈ sps 7.06. The pulse is **not identified** (white noise is pulse-independent), and in-band noise fits far worse. **Optimal:** `locked` and `random_phase` produce **zero errors below −2 / −4 dB** — a geometric impossibility for the paper's low-JSR points, not a tuning problem. Only **`async`** produces errors there, via the pulse's tails at off-Nyquist instants. Best is **async, RRC β = 0.25**: crossing −6.43 dB (sps 4) / −6.35 dB (sps 8) vs the paper's −7.06, RMS 0.98 / 1.01 dec. It still falls ~1.5 decades too fast below −6 dB (paper 4.4e-5 at −10 dB; ours < 1.5e-7). **Gap at 1e-3, Noise − Optimal:** paper 5.75 dB; sps 8 gives 5.68 dB; sps 4 gives 2.69 dB. | `cgan/calibrate.py` → `artifacts/cgan/calibration.json` + `c2_calibration.png` | Checkpoint passed 2026-09-14; the bar is report-only (§2.10). **Calibrate the unstated link parameters.** (1) Fit the closed-form noise curve to the digitised Noise points (BER ≥ 1e-6) over sps ∈ {2,4,8,16} × pulse ∈ {rect, RRC β ∈ {0.25,0.35,0.5}} × noise band ∈ {full fs, in-band}. (2) On the fitted link, Monte-Carlo the `optimal` jammer under three synchronisation variants — {symbol+phase-locked, symbol-synchronous with random phase, asynchronous} — and pick the best match to the digitised Optimal curve. The same assumption then governs how the GAN training data is cut. | `cgan/calibrate.py` → `artifacts/cgan/calibration.json` + figure | **⛔ STOP: report both fits' residuals to the user before C3.** If no variant brings Optimal within ~1 dB, the success bar needs revisiting — and that is itself a finding about the paper. |
| **C3** | **Generator, discriminator, losses** per Zhou Figs. 2–3. Every unstated choice is recorded in §4.2 Q7. | `cgan/models.py`, `cgan/losses.py` (pure torch) | **DONE 2026-09-14** (job 2259280). `models.py` + `losses.py`, plain torch. |
| **C4** | **DONE 2026-09-14 — 94/94 checks pass, job 2259280** (link + models + losses); **121/121 after the 2026-09-15 revision, job 2260603** (one `desync` model for both structured jammers, a perfect-generator check, the reference-backed losses). Two real link defects were caught and fixed at source, not by loosening a tolerance: **(a)** span-8 RRC truncation ISI of 3.6 % flipped zero-margin symbols (job 2259141) → span 32 (§4.2 Q7); **(b)** JSR was averaged over the filter tails, over-driving short frames by up to 1.76 dB (job 2259142) → power ratios defined over the symbols' active window (`Link.active`). **Test suite.** Clean BER vs the Q-function reference; noise-jammer BER vs the closed form (which validates the processing gain C2 relies on); realised JSR within 0.05 dB of the request for all three jammers; a phase-locked optimal jammer at zero noise flips exactly past the amplitude margin; model shapes (G → (B,2,1024) in [−1,1]; D features 32768; score (B,1); logits (B,n_classes)); classifier loss ≡ 0 when n_classes = 1; STFT and I/Q losses ≡ 0 on identical batches; exact normalisation round trip. The link checks are written with C1 and extended at C3. | `cgan/verify.py` + `submit_verify.sh` | exits 0; ran before C5/C6 |
| **C5** | **DONE — two runs, the discriminator wins in both (§3.3b).** **run001** (2026-09-14, job 2259289, ~12 min): non-saturating BCE + GP, n_critic 1, Adam β (0.5, 0.999), all loss weights 1, MSE feature matching, log1p STFT, peak headroom 1.0 — *the current code no longer reproduces it*. Final D(G(z)) ≈ 0.3 %. **run002** (2026-09-15, job 2260624, 80 min): the reference-backed default recipe (§4.2 Q7) — WGAN-GP, n_critic 5, β (0.5, 0.9), weights feat 2 / STFT 45, headroom 0.95; critic gap stalls at ≈ 57 from iteration ~3k. Both: 10,000 iterations, batch 128, Adam 2e-4; `--adv nsgan` selects the run001-style adversarial loss. | `cgan/train_cgan.py` + `submit_train.sh` → `artifacts/cgan/run00N_G.pt`, `_losses.{json,png}` | done |
| **C6** | **DONE 2026-09-15 — reproduction NOT established (§3.3b).** The 2026-09-14 "match" (job 2259409: Noise−GAN 4.34, GAN−Optimal +1.01 dB vs paper 4.44 / +1.31) scored the GAN locked and Optimal async. Re-evaluated with one sync model for both (run001 job 2260623; run002 job 2260806, after 2260778 lost its GPU): run001 GAN−Optimal **−1.30 dB async / −4.47 dB locked** — the GAN beats "optimal", so the paper's ordering fails — while Noise−GAN 6.66 / 4.38 dB keeps the "GAN beats noise by 2–6 dB" claim. run002 has no 1e-3 crossing in the grid (BER 3.6e-3 already at −10 dB). **Evaluate against Zhou's protocol** (SNR 30 dB, JSR −10:2:10 dB — the grid of Fig. 6's markers — 10⁴ symbols × 100 trials), **extended** to ≥ 100 bit errors or 10⁹ bits so the low-BER points are real. Overlay on the digitised Fig. 6. Report JSR at BER 1e-3 for each jammer, and the two gaps next to the paper's 1.31 / 4.44 dB (report-only, §2.10). | `cgan/evaluate.py` + `submit_eval.sh` → `artifacts/cgan/run00N[_tag]_{ber_vs_jsr.{json,png},implied_gain.png,iq.png}` | done. **Step 1 closed 2026-09-16 as a partial reproduction, run001-async as step 2's reference (§2.10). Next: the statistical detector (§4.2 Q8)** |

With no pass/fail bar, hyperparameters are iterated **only** by explicit user decision after seeing
C6. Each run gets a row in [A.0](#a0-run-index).

### Detector-suite characterisation study — PROPOSED 2026-09-17; D0/D1/D2a/D2 + E2 ALL BUILT AND EVALUATED (2026-09-24)

The user's chosen staging (2026-09-17). Shared spine for options A and B; MARL and CWT are gated
"only if it bites". Same conventions as the C-series (run from `cgan/`, sbatch, matched detectability,
NP on the axes). Rationale and design constraints: §2.10 (STATE 2026-09-17). **STATE 2026-09-24:**
- D0 (baselines), D2a (learned control), D1 (plain GAN) and D2 vs **all four** detectors are done
  ([§3.3f](#33f-cgan-under-detection--d1-plain-gan-and-d2-detector-aware-gan-all-four-detectors-2026-09-20-cnn-2026-09-23)), and so is E2, the noise ablation of D1/D2 ([§3.3g](#33g-e2--the-noise-ablation-on-the-live-models-30-db-is-near-the-worst-place-to-measure-the-gain-2026-09-23)). The user directed all as preparation, not gated on the supervisor.
- **The single-jammer compute is complete.** The paper draft is being finished (user, due Sunday
  2026-09-27); the extend-vs-MARL fork resolved toward **MARL (D4) as the next experimental direction,
  built in the background** (user, 2026-09-26). The E2 follow-ons (§4.3) are not being pursued now.
  *(MARL part superseded 2026-09-28: proposed out of the ICC paper.)*
- **STATE 2026-09-26: E3 pre-check done ([§3.3h](#33h-e3-pre-check--under-fading-the-team-gains-nothing-timing-coordination-does-2026-09-26)); D4 promoted to active-background, D4a and a D4b first cut done, then all of D4 PAUSED for the GAN rework ([§3.3j](#33j-d4b--the-learned-policy-finds-the-power-lever-not-delay-compensation-2026-09-26-paused)).**
  Nothing MARL enters the paper until MARL results exist (user); if they are good, ~5 days remain to fold
  them into the draft before ICC (2026-10-02).
- ~~The *framing* of D2 onward still waits on the direction email (§4.1 #0b).~~ The email went out
  2026-09-28 (§4.1 #0c).
- **STATE 2026-09-27:** the method changed to random init, 4000 steps (run003, §2.10); grey-box
  (run004), E2b (on/off, outlier alarm, PER, damage per extra alarm) and learned power (run005) done
  (§3.3f/§3.3g). D5 and D6 were discussed the same evening (rows below): D5 demoted, D6 after S1.
- **STATE 2026-09-28:** S1 Tier 1 done (§3.3k). The email proposed D6 to the supervisor as the one
  remaining experiment ("Mon") and MARL (D4) out of the ICC paper; the story is proposed, not agreed
  (§4.1 #0c). **Evening:** S2 built (§3.3l); D6, S2, S3, S4 and S5 specified as the background batch at the
  top of §3.4, for two fresh sessions; S3 was cut to one per-symbol detector (W = 1) and added to Track 2.

Build order:

| # | Item | Note |
|---|---|---|
| **D0 = "baselines"** | **DONE 2026-09-17 on the 3-D waveform link ([§3.3d](#33d-classical-baselines-on-the-3-d-waveform-link--the-pre-gan-envelope-2026-09-17)).** Attackers barrage / pulsed-QPSK(p) / omniscient(η); detectors power (1/2-sided), kurtosis, spectrogram-CNN, and the exact noise NP test — on the waveform layer, K = 1–4. The user built it as GAN-prep, not gated on the supervisor. *(An M0 exact-NP panel was folded into the waveform NP-on-noise row rather than run separately.)* | The envelope exists: floor (barrage), smart-classical (pulsed-QPSK), genie ceiling (omniscient), optimal warden (NP on noise). Every realisable attack is at the floor. |
| **D1** | **DONE 2026-09-20 ([§3.3f](#33f-cgan-under-detection--d1-plain-gan-and-d2-detector-aware-gan-all-four-detectors-2026-09-20-cnn-2026-09-23)).** Plain GAN (`run001_G`) on the BER–P(det) plane: stealthy BER 0 against every detector — on the floor with noise/pulsed. The honest "detectability of a reproduced GAN jammer". | `eval_gan.py`; `attacks.gan_tx` + `gan` spec. |
| **D2a** | **DONE 2026-09-19 ([§3.3e](#33e-learned-shaped-noise-control-d2a--shaping-buys-effectiveness-not-stealth-2026-09-19)).** Learned control tier: shaped-noise jammer (48 parameters), CMA-ES, black-box score-based, one per detector, the same objective D2 will use. User decision 2026-09-18 (§2.10). | Stealthy BER 0 against every detector. The CNN is evaded only out of band, where nothing reaches the victim; the effectiveness gain is 15.9 dB. D2 must beat this, not the fixed rows. |
| **D2b (later)** | **FAR ablation** (user, 2026-09-19). α = 0.05 *per frame* is arbitrary: at 1 MBd and 128-symbol frames it is about 390 false alarms per second, and a real receiver would run far lower α. Plan: α ∈ {1e-3, 1e-2, 0.05, 0.1}. **Evaluation first:** re-measure the existing fixed and learned jammers, storing per-frame statistics so any α (and the ROC) comes for free; `thresholds.json` holds only α ∈ {0.01, 0.05}, from 20k clean frames, so α = 1e-3 needs ≥ 100k. **Retrain the learned tier only if the ranking changes with α**; its objective uses the α = 0.05 threshold. **Half-built by E2 (2026-09-23):** `detectors.stat_quantiles` + `baselines.measure(..., keep_stats=True)` store each statistic's 257-point CDF, which yields P(det) at ANY α without per-frame dumps (tens of KB, not hundreds of MB); `snr_figures.fig_frontier` already reads the full ROC off it. E2 stores it for two tags only — widening the tag list is the rest of D2b. | Expected: power's "first errors only at P(det) ≥ 0.83" is robust to α, because at −16 dB the mean shift is far above the clean spread. The CNN rows are the ones likely to move. |
| **D2** | **DONE — all four detectors ([§3.3f](#33f-cgan-under-detection--d1-plain-gan-and-d2-detector-aware-gan-all-four-detectors-2026-09-20-cnn-2026-09-23)).** Analytic 2026-09-20, CNN 2026-09-23. White-box direct-gradient generator (warm-started from `run001_G`), objective `−(log E[BER] − β·P_det_soft)`, one per (detector, β); the CNN is differentiated through a straight-through viridis LUT — the deployed weights, no surrogate. **Confirmed stealthy BER 0 against every detector**, ≤ 3 dB stealth edge over noise, ~30 dB gap to where BER bites (at 30 dB SNR; it grows with SNR, §3.3g). No better than D2a's 48-parameter control. | `train_gan.py`; per-target `JSR_BANDS` and β range matter — see §3.1 traps. |
| **E2 = the noise ablation** | **DONE 2026-09-23 ([§3.3g](#33g-e2--the-noise-ablation-on-the-live-models-30-db-is-near-the-worst-place-to-measure-the-gain-2026-09-23)).** The supervisor's mandated primary ablation (§B.2), run on the live models: SNR 0–40 dB + noiseless, all 21 generators, detectors re-calibrated per level, CNN weights frozen at 30 dB, generators evaluated not retrained (transfer, not achievability). The gain peaks at −85.1 pp at 15 dB vs −16.2 pp at 30 dB. **Follow-ons (one rerun for PER / SINR / AUC, fixed-α trade-off figures, Tier 2 retraining) parked by the user 2026-09-24 — §4.3.** | `calibrate_snr.py`, `snr_ablation.py`, `regress_snr30.py`, `snr_examples.py`, `snr_figures.py`, `verify.py` §15. An eval is 35 s; the grid runs in ~20–30 min wall as a 10-task array. Setup, traps and caveats: §3.3g. |
| **E3 pre-check** | **DONE 2026-09-26 ([§3.3h](#33h-e3-pre-check--under-fading-the-team-gains-nothing-timing-coordination-does-2026-09-26)).** Transfer check (no retrain): faded jammer→R links, K = 1/2/4, aligned vs random timing, five jammers. **Fading helps a single jammer (no gap to recover); the only real lever is TIMING alignment** — aligned K = 4 ≈ one jammer at 7.3 dB less per drone, random timing loses ~6 dB. Bounds what D4 can find. | `cgan/team_fading.py`, `submit_team_fading.sh`, `team_figures.py`, `verify.py` §16. Numbers only, no figures. |
| **S1 — shadowing ablation — Tier 1 DONE 2026-09-28** | **Per-frame log-normal shadowing on the victim's link, σ = 0–3 dB, before any MARL** (user: "more realistic but easier for the jammer to hide"). CNN retrained per σ + a gain-aware energy detector `power_csi`; run003 evaluated, not retrained. **Result:** the naive energy detector goes blind as the SNR wall predicts (CNN-targeted generator: 1.04 → 4.5 → 19 → 35 frames per extra alarm at σ = 0 / 0.1 / 0.3 / 3 dB), the gain-aware one stays at 1.04–1.06 for every jammer at every σ, and the retrained CNN stays the weak link (5–8). Tier 2 (retrain under shadowing) built, not run — it would not inform. Open: `power_csi` uses the true gain; a decision-directed version is argued, not measured. [§3.3k](#33k-s1--shadowing-on-the-victims-link-blinds-the-naive-energy-detector-not-a-gain-aware-one-2026-09-28). | `link.shadow_gain`, `detectors.power_csi`, `train_spectrogram_cnn.py --shadow-db`, `snr_ablation.py --axis shadow`, `train_gan.py --shadow-db`, `shadow_figures.py`, `verify.py` §19. |
| **S2 — the listening jammer (synchronous arrival) — DONE 2026-09-28: fails its success test** | The attacker uses what a passive listener could learn; here, T's symbol timing plus geometry → it lands on R's symbol grid, phase still random, symbols unknown. Perfect timing only; four generators retrained synchronously (run003 recipe), evaluated paired with the async baseline. **Adopted into the system model only if it beats run003 at matched damage** (user). [§3.3l](#33l-s2--the-listening-jammer-synchronous-arrival-buys-nothing-measurable-2026-09-28-done): nothing measurable at matched damage, 4 seeds each side, so it stays out. | `Link.jammer_sync`, `channel.async_draw`, `train_gan.py --sync`, `snr_ablation.py --sync`, `verify.py` §20. |
| **S3 — a per-symbol detector (W = 1) — DECIDED 2026-09-28, TODO (Track 2)** | **User decision:** only W = 1, nothing in between ("simplify where possible"); no window sweep, no separate undetected-damage metric. The question stays per frame: *if a jammer attacks a frame, how probable is it that the frame is detected* — now by a detector that inspects every symbol. Statistic: per symbol, the distance of the matched-filter sample z_k to the nearest QPSK point; per frame, the MAX over its 128 symbols (flagged if any symbol looks wrong). Threshold: the (1 − α) quantile of that max over clean frames, so the frame FAR is 0.05 like every other detector (128 tests at 5 % each would flag 99.9 % of clean frames). Why it matters: the one-sided energy detector's alarms fall mostly on the frames where the jammer FAILED (§2.7); a clean symbol's error spreads ~0.016 symbol energies, a blind push that breaks a bit leaves it near the boundary (~0.5), so this detector should see the damage itself. The genie stays invisible (it lands ON another point). The CNN stays per frame (it needs an image). | Not built. See the batch, Track 2. |
| **D5 — DEMOTED 2026-09-27 (discussion)** | Was: learned power with a non-saturating detection term, β·mean softplus((ψ−τ)/s). **Not built, for three reasons.** (i) softplus = −log(1−σ) is unbounded — against power it is a linear POWER PENALTY (§2.8 forbids), and it charges a detected frame for its loudness, which steers away from on/off, the average-BER winner (§3.3g E2b); the objective changes, not just its gradient. (ii) Saturation is half the diagnosis: log E[PER] ≈ −3900 at −40 dB, and Adam's step on one scalar log-gain follows whichever gradient is larger, so any bounded detection term loses on the way up. (iii) Both expected outcomes are already known (≈ 1 frame per alarm vs energy; run005 CNN β = 100 already 0.000). **Replacement, eval-only:** run003's power-targeted generators (tasks 1–4) through `outlier_alarm.py --gens …` — no energy-trained generator was in E2b. If D5 is ever built: anneal the sigmoid width instead. | Not built. |
| **D6 — ROUND 1 DONE 2026-09-28 ([§3.3o](#33o-d6-round-1--the-cnns-gap-was-a-training-range-gap-at-30-db-not-at-15-db-2026-09-28)); round 2 open** | **Round 1 result (P(det) at matched damage):** at 30 dB widening Li's JSR range to −35 dB (arm A, never sees the attacker) lifts the CNN on the held-out seeds from 0.16–0.18 to 0.58–0.74, level with energy on the same frames; seeing the attacker (B) adds 9–19 pp, but the round-0 attackers stay below energy (0.44–0.48 vs 0.57–0.65); no measurable cost. At 15 dB nothing moves (CNN ≤ 0.09 under every defender, energy 0.22–0.43) and retraining costs 4–5 pp of detection on Li's own classes, mostly from the range widening. **Arms race: the defender retrains** — the supervisor's mandated headline (§2.9: *"adaptation cost is the headline claim, not 'the jammer evades the CNN'"*), which the §3.1 candidate story is not. **Two design changes from the discussion.** (i) **A training-range confound:** the CNN saw jammed frames only at JSR U[−20, +10] dB, 204 per type (`train_spectrogram_cnn.py` `JSR_RANGE_DB`); the headline generator works at −23 dB. So round 1 has two defender arms — **A, attacker-agnostic:** Li's four types with the range widened to ≈ [−35, +10], no generator frames; **B, attacker-aware:** A + the generator's waveforms as a fifth jammed type. A closing the gap = a training-range gap, fixed without seeing the attacker. (ii) The defender trains on ALL earlier generators (fictitious play, no cycling), is scored on HELD-OUT generators (the five β = 10 seeds, the two grey-box ones), one number per round = frames per extra alarm vs that round's CNN at 15/30 dB with the energy floor as a line, 2 rounds. ~~Run it where the naive energy detector is blind (an S1 level).~~ *(Dropped 2026-09-28: S1 found no σ where a gain-aware energy detector goes blind, so no such regime exists for a sensible defender; D6 measures the CNN's adaptation on its own terms, with the energy floor drawn as a line.)* | `train_spectrogram_cnn.py --jsr-range / --extra-gens / --snr-db`, `submit_arms_cnn.sh`, `arms_eval.py` + `submit_arms_eval.sh`, `arms_figures.py`, `verify.py` §23. |
| **D4 (PAUSED 2026-09-26; proposed OUT of the ICC paper 2026-09-28 → thesis/TWC)** | **MARL coordination: the delay-decay curve, learned with minimal inductive bias. Design SETTLED 2026-09-26 in [§4.2 Q12](#42-open-technical-questions).** The x-axis is the inter-jammer timing error σ (the supervisor's axis, §B.2). Each drone sees only its own geometry and σ_k and acts with (u_k = fraction of a per-drone power cap, δ_k = transmit advance); the D2 waveform is frozen. One shared MLP, CTDE, **direct gradient through the differentiable link**, MAPPO only as fallback. **Never policy-gradient RL over raw IQ** (§A.5, §2.8). Arms at matched detectability: learned · geometric heuristic · uncoordinated · single-jammer ceiling. **D4a DONE 2026-09-26 ([§3.3i](#33i-d4a--the-delay-decay-curve-coordination-is-worth-1-symbol-of-timing-accuracy-2026-09-26))**: the knee is at σ ≈ 1 symbol, below the geometric spread, so δ_k matters. **D4b first cut DONE, then PAUSED ([§3.3j](#33j-d4b--the-learned-policy-finds-the-power-lever-not-delay-compensation-2026-09-26-paused))**: the policy learns u → 1 but not δ = τ, and every D4 number is conditional on the under-trained `run001/task14_G.pt`. | `team_fading.py --sigmas`, `verify.py` §17–18, `team_figures.py --timing/--policy`, `team_policy.py` + `submit_team_policy.sh`. Nothing goes into the paper until results exist. |

MVP that already makes the point: **D0 + D1** (attackers {barrage, pulsed-QPSK, GAN} × detectors
{energy, kurtosis, CNN, NP}, single jammer, one layer). Add D2, then D3/D4 only on evidence.

### Coordination track — PAUSED 2026-09-14 (kept as written 2026-09-12)

**What changed.** The old plan was ordered around G1 as the gate, because G1 decided whether a
*stealthy* attacker had headroom. That question is retired (§2.1), so **G1 is demoted from gate to
optional** and the critical path now runs through the multi-jammer extension, which was previously
last. G0 and G2–G4 are unchanged in content; several changed in priority.

**Everything below is contingent on §4.1 #0** — his sign-off on the pivot. Items marked ⛔ commit
days of work to one branch of that fork and **must not start before his reply.**

**Compute track.**

| # | Item | Cost | Gate |
|---|---|---|---|
| **G0** | **Add the omniscient reference (`counter_flip`) to `fig_detectors` and `fig_stealth_vs_sigma`** in `m0/figures.py` (L162, L202 — their attack lists omit it). His mandate is *every* results figure, and `sec:system:objective` states the convention in print. | ~15 min, CPU | **Do now.** Unaffected by the pivot; needed before any figure regeneration. |
| **G6** | **M1 — multi-jammer coordination under a delayed inter-jammer link. THE LEAD EXPERIMENT (E3).** N_J ≥ 2 jammers, per-link gain/phase, superposing at the victim. Three arms: coordinated · N_J-independent · single jammer at equal *total* power, at matched power **and** matched detectability. **The sweep is the inter-jammer delay / phase error** (§2.3 RQ1, his 2026-09-12 note) — the deliverable is the gain-decay curve between the independent floor and the perfect-coordination ceiling. Direct/surrogate gradient as in sim04; PettingZoo only if an off-the-shelf MARL algorithm is genuinely needed. | ~4–5 days, mostly new code | ⛔ **after §4.1 #0.** Spec it on paper meanwhile. |
| G5 | **Conditional generator on M0**, direct gradient. Under the pivot its job is no longer "beat the closed form" but **be the policy class G6 coordinates** — so it is now a *component* of G6 rather than a separate result, and can be built single-jammer-first as a de-risking step. | ~2–3 days | ⛔ merge into G6's schedule |
| G3 | **Power-budget ablation**, log grid. Cheap, and it is the axis the coordination gain is read against (equal *total* power is the whole comparison). | ~½ day | independent — **promoted**, do while waiting on #0 |
| G2 | **E2 — noise ablation.** Largely already produced by the 8-σ E1 sweep; needs the dual-axis figure and the matched-P(det) companion — **and the missing σ = 0 anchor + a proper log grid** (§3.2). Still his mandated primary ablation, so it survives the pivot intact. | ~½ day | independent, do while waiting |
| G4 | **Structure ablation** (§4.2 Q2) — iid-uniform → non-uniform priors → correlated → pilots. **Weakened by the pivot:** it was there to find the single-link learner a job, and RQ1 now says the learner's job is coordination (Q2 items 3–4), which G4 does not test. Keep only if time allows. | ~1 day | **demoted** |
| G1 | **The covertness-constrained optimality program** (§2.8). ⚠ Prior art (§2.1) — instantiates the CPS stealthy-FDI program. | ~1 h compute, ~½ day to write | **demoted from gate to optional.** Run only if the stealth axis is still being reported quantitatively. |
| G7 | **Adaptation-cost rounds R0/R1/R2** (RQ2). Tooling exists; in M0 it gains the distance-from-`D_NP` reference E1 already measured. | ~2 days | **depends on how he breaks the RQ1/RQ2 tension** (§4.1 #0) |

**Writing track (starts now, no compute).**
- **Experiment History: the sim00–08 record** — condensed to one ~1-column summary,
  `paper_drafts/sec_exphist.tex`, **not yet pasted** (§3.5).
- Repair `paper_drafts/sec_system_model.tex` (§Cooperative — the CLT argument, see §4.2); fold the
  §2.8 decisions into §Defender Model and §Generative Attack Policy. **This is the M0 draft, and since
  2026-09-24 it is thesis material only.**
- **`paper_drafts/sec_system_model_waveform.tex` is the paper's System Model** (the user picked it over
  the M0 draft on 2026-09-24). It describes the geometry-free waveform system the D-series ran on; its
  contents and state are in §3.1 ("System Model draft"). Not in Overleaf.
- Related Work is drafted (`paper_drafts/sec_related.tex`, 1081 words ≈ 1.93 columns; the
  1253-word thesis version is `sec_related_long.tex`). **Not yet pasted into Overleaf** — `main.tex`
  still carries all three legacy blocks (`Related Works` L72, `Literature Review` L233,
  `Old Related Works` L309).
- Likewise `sec_system_model.tex` is **not yet in Overleaf**; `main.tex` §System Model (L107) still
  describes the K-subcarrier / TDL / N_J-jammer setting **that no working experiment supports**.

**Rough calendar, re-cut 2026-09-12 (20 days).** Sep 12–13: §4.1 #0 email + read the three prior-art
papers + G0 + spec G6 on paper. Sep 14–15: G2 + G3 while waiting on his reply; paste the condensed
Experiment History (§3.1 item 2). Sep 16–22: G5-as-component then G6. Sep 23–26: G6 results, figures,
write-up. Sep 27–Oct 2: lock, polish, buffer. **Results lock Sep 27** — one day earlier than the old
plan, because the lead experiment is now the one that does not exist yet.

**Risks, in order — reordered after the pivot.**
(i) **He does not agree with the pivot**, or breaks the RQ1/RQ2 tension toward RQ2. Mitigated only by
asking early, which is why #0 is the next action and not a background task. **This is now the top
risk and it is a scheduling risk, not a technical one.**
(ii) **G6 is new code on a 20-day clock**, and the multi-user/multi-jammer extension has never been
written for M0. sim04 is the precedent (two coordinated agents, direct gradient, worked) but it is
frozen sim-stack code, not M0 code — treat it as a design reference, not something to lift.
(iii) **The coordination gain turns out to be small.** sim04's honest number at matched total power
was ~15%, not the 2× the draft once claimed (§A.3). A 15% gain is still a result, but it must be
reported at matched detectability too, or it is the m1 mistake again (§2.7).
(iv) Scope creep against his explicit "simplify, as much as possible" and the 2–3-experiment quota —
**the pivot is permission to change direction, not permission to add layers.** M1 as specified in
§2.4 is the sanctioned extension; nothing beyond it.

> **The fallback, in his own words, banked 2026-09-12.** If G6 does not land in 20 days, the
> single-agent case is *"already a problem in itself. even understanding the first would be good"*
> (§B.2). So the degraded outcome is a paper on the single-agent generative attacker with the
> coordination result as future work — **not** a scramble back to the retired stealth headline.
> Decide this by **Sep 22**: if M1 does not run end-to-end by then, take the fallback rather than
> spending the buffer.

## 3.5 The Overleaf appendix — "Experiment History"

**State (2026-09-27): `sec_exphist.tex` is ~1,250 words of prose (~2.2 columns), still not pasted.**
Added 2026-09-27: "Imitation as initialization" (the warm start + 400-step budget, now history) and six
paragraphs written by a subagent from this README — minimal model / optimal detector (framed as a
square-root-law calibration, not a finding) · reproducing Zhou's CGAN · classical baselines on the
waveform link · the shaped-noise control · the first (400-step) noise ablation · timing among several
jammers (**labelled as the old 400-step generators, user's decision**; its `%` comment notes it may move
to the main text/Future Work). Each carries a `%` README-source comment. Open in the file: two
`\cite{TODO-key}` (square-root law, CMA-ES — only in `paper_drafts/refs_new.bib`, not in Overleaf's
`refs.bib`); the older "Learned detection" paragraph's explanation (a jammed spectrogram is a clean one at
lower SNR) now conflicts with §3.3d's per-image contrast stretch — not yet reconciled; the closing
sentence ("attacks are parameterised in low dimension rather than raw IQ") contradicts the raw-IQ
generator method — proposed fix "not trained by policy gradient over raw IQ", awaiting the user. The
2026-09-15 state follows.

**State (2026-09-15): condensed to one summary section, `paper_drafts/sec_exphist.tex`, not pasted.**
The user decided the 10-part appendix was far too long for a 6-page conference paper: its job is only
to *summarise* what was done, and experiments that carry a claim get written out in full in their own
sections. Parts 9 (verdicts + traceability table) and 10 (What the History Determines) are **dropped**;
parts 1b and 5–8 were folded into the summary and their per-part drafts deleted (recoverable from git,
commit `4f6a1d5`, if the thesis wants the long versions).

`sec_exphist.tex` is ~610 words of prose (~1.1 IEEE columns), no table, no figure: a two-sentence
opener with the reward in one inline formula, then one italic-headed paragraph per step (statistical
detectors · two cooperating jammers · learned detection · policy gradient over raw IQ · detector
characterisation · realistic channel), and a one-sentence close naming the three choices the history
fixed (low-dimensional action, matched detectability under the FAR budget, judge against the optimal
detector). It **replaces the entire `\section{Experiment History}` in Overleaf** (`main.tex` L534– at
`b9620a0`), including the user-written parts 2–4, and already carries the corrections those parts owed:
2 legitimate TX→RX pairs in the first experiment; two-agent numbers from job 99211 (**BER ≈ 0.24,
per-agent power ≈ 0.62, det 2–8%**, not 0.33 / 0.45 / 3–10%); the gain at matched total power is ~15%,
not 2× ([A.3](#a3-sim04--sim04b--a-coordinated-solution-exists-and-is-gradient-reachable)); and the
centralised-optimiser caveat. Header comment in the file lists the job IDs behind every number.

Still valid from the old plan: **do not name simulations in the prose**; negative results are stated
with their mechanism, never apologised for. It has no `\ref`s (the user replaced the `sec:system` ref
with "the current model"); its only citation is `9707819` (Li et al.), already in `refs.bib`. **The
user is converting the italic paragraph heads to `\subsection`s** — as of 2026-09-15 only "Statistical
detectors" is converted; keep the rest consistent with whichever the user settles on. Unrelated Overleaf hygiene still owed: remove `\nocite{*}`, and Experiment History sits
behind two scratch appendices (`Literature Review`, `Old Related Works`) full of `\adm{}`/`\rar{}`
notes. Live ref last read: `overleaf/main` @ `b9620a0`; **`git fetch overleaf` fails from this repo**
(no GitHub credentials), so it may lag the real Overleaf head.

## 3.6 Explicitly NOT doing

- **Extending sim06/07/08 in any direction** — no further OFDM/fading/CNN sweeps, no
  matched-detectability follow-ups. Appendix material, finished, and the *opposite* of "simplify".
- **RLlib.**
- **Black-box PPO over raw IQ** — falsified.
- **Channel-aware *subcarrier selection* as a lever** — refuted at matched detectability, and there
  are no subcarriers to select in M0.
- **Further characterization sweeps.** Characterization has already done its job: it defined the
  target (the residual region) and the metric (matched detectability). More of it is the main way
  left to waste the remaining hours.
- **Stealth as the attacker's objective** — retired 2026-09-12 (§2.1). Detectability stays as the
  axis attackers are *compared* on; it is not what anything optimises, and no headline rests on it.
  **This does not license reintroducing it later in the guise of "just one more frontier sweep".**
  *2026-09-14: reopened as a conditioning variable on the exploratory CGAN track only, by explicit
  user decision, with the square-root-law caveat carried over —
  [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).*
- **Any novelty claim about the impossibility result** without first reading Bash (JSAC 2013), Li
  (TIFS 2016/2020) and Csiszár–Narayan. The result is correct; the claim of priority is not.

> **What the pivot does NOT reopen.** The 2026-09-12 change of direction is a return to bet (a),
> cooperation — **not** to bets (b) and (c) (§1.1). PPO/MAPPO over raw IQ with black-box access
> stays falsified (sim06/06b/07, [A.5](#a5-sim06-jammer--06b--07--the-untrainability-result)); actions
> stay low-dimensional perturbation *parameters*; the training method stays direct/surrogate
> gradient. "Cooperative multi-agent generative" describes the *problem*, not a licence to re-run the
> algorithm that failed. If G6 seems to need MARL, re-read A.5 before writing any of it.

**Deferred refinements, not on the critical path:** BER-thresholded in-band labels + threshold
calibration for the m2 detector; extending the sim08 suite with more classical detectors
(kurtosis/GLRT/pilot-variance).

---

# PART 4 — OPEN QUESTIONS & IDEAS

## 4.1 Blocking / needs supervisor input

| # | Item | Why it blocks |
|---|---|---|
| **0** | ~~**Sign-off on the pivot** to coordination as RQ1 (2026-09-12)~~ **SUPERSEDED 2026-09-28 — never sent.** The 2026-09-28 email (#0c) replaced it: coordination is proposed out of the ICC paper (thesis/TWC), the literature collision (§2.1) was disclosed in one line, and the RQ1/RQ2 tension is resolved toward the single-jammer study + D6 (his adaptation question). | — |
| **0b** | ~~**Sign-off on retiring stealth entirely + the GAN/MARL replacement** (2026-09-16; options A/B 2026-09-17)~~ **SUPERSEDED 2026-09-28 — never sent.** The 2026-09-28 email (#0c) went with option B (the detector-gap measurement, generator as instrument), dropped A's MARL from the paper, and introduced the CGAN track, which he had not seen. | — |
| **0c** | **⚠ 2026-09-28 — email SENT; awaiting his reply.** It proposes (content in §B.1): the §3.1 story; MARL out of the paper, D6 as the one remaining experiment; a Friday 2026-10-02 submission with the full draft to him Wednesday evening. It asks: OK to submit this scope on Friday? a short call Mon/Tue? move to the separate paper Overleaf he planned? **Correction owed in the Wednesday draft:** the energy-floor claim needs the gain-aware qualifier (§3.1, S1). | His answer decides the paper's scope; the week's plan (§3.1) runs meanwhile. If he wants MARL in, one small experiment fits, likely negative. |
| 1 | **Supervisor of record for the ETH registration** | D-INFK professor requirement; may need Di Maio as co-supervisor. Needed on the myStudies form. |
| 2 | **Title, start date, end date, task description** | All four gate registration. Open since 2026-09-01. |
| 3 | ~~**Feedback on the proposal** (handed over 2026-09-01)~~ **RESOLVED 2026-09-12 — it arrived.** Nine inline `\adm{}` comments, transcribed in [B.2](#b2-his-verbatim-points-and-what-each-changed); consequences propagated to §2.1, §2.3, §2.4, §2.9, §3.4 and §4.3. Three prose fixes remain owed on the proposal itself (§B.3). | — |
| 4 | **Which 2–3 experiments go in the main paper** | His call. Candidates **changed 2026-09-12**: E3 (coordination) now leads, E2 (noise ablation) is his mandated primary ablation, E1 demoted to calibration/appendix (§3.3). Ask as part of #0. **E2 is now DONE on the live models (§3.3g, 2026-09-23)** — and it is a stronger candidate than when it was proposed, because it does not merely ablate: it relocates the headline (the matched-BER gain is 5× larger at 15 dB SNR than at the 30 dB the D-series reported) and supplies the mechanism (the stealth edge falls with SNR). **Proposed 2026-09-28 (email):** the CNN result against its β = 0 control over SNR (run003, E2), the energy floor as damage per extra alarm (E2b, with S1's qualifier), grey-box transfer (run004) and D6; E3/D4 out. |
| 5 | **Single-round evasion on a frozen detector, or fully co-adaptive?** | His 2026-08-03 phrasing leans co-adaptive (*"one can always fine-tune a defender on an attacker and vice versa"*) but was never a direct answer. RQ2 assumes round-based-and-offline. **Proposed 2026-09-28:** round-based co-adaptation, measured by D6 (§3.4). |
| 6 | ~~**Is he comfortable leading with detector characterization as the solid core** and the cooperative learned jammer as the high-upside extension?~~ **MOOT as posed, 2026-09-12.** The pivot answers it the other way round: the cooperative jammer *is* the core and characterisation is appendix. Still worth flagging in #0 that this is a reversal of the mid-July proposal he never answered. |
| 7 | **His September availability / feedback cadence** | Asked twice, still unanswered. Short frequent rounds >> one large end-of-block review. |
| 8 | ~~**Inter-jammer coordination assumption** — shared backhaul / shared clock only / fully independent?~~ **ANSWERED 2026-09-12, and reframed.** He does not pick from that menu; he says the interesting thing is that the inter-jammer link *"introduces desynchronization"* through **communication delay** (§B.2). So the assumption is a link with a delay parameter, and the delay is swept rather than assumed (§2.3 RQ1, §2.4). Residual open question, much smaller: what delay *range* is realistic — worth one line in the reply to #0. | — |

**When his corrections come back — the four things most likely to be challenged**, recorded so the
reasoning does not have to be reconstructed:
1. ~~**The gap claim** (§2.2 point 5)~~ — **no longer a risk, because we withdrew it first**
   (2026-09-12, §2.1). It would have been the first thing challenged; it is now the first thing
   disclosed. Raising it ourselves in #0 is worth more than defending it would have been.
2. **The dropped countermeasure RQ** — the answer is the cost argument (§2.3); the cheap fallback is
   time-to-first-detection (§4.3), already specified and needing no new machinery.
3. **RQ2 vs the bachelor timeline.** It is last in the plan and depends on a working attacker; if the
   schedule tightens, **this** is the item to renegotiate — not the structure ablation, which gates
   everything else.
4. **Whether "protocol-aware" over-promises.** The Introduction positions protocol-aware jamming as
   literature context, but our method is signature-shaped interference. Defensible
   (protocol-deterministic fields are exactly what survives scrambling) but he may read it as a claim
   about our method.
5. **"Isn't this just a combination of existing methods?"** (own concern, not his — but it will be
   asked. **Sharpened 2026-09-12: the literature check is the concrete instance of exactly this
   objection landing**, so treat it as demonstrated rather than hypothetical.) The honest answer is
   yes, and that is fine **as long as the paper leads with the problem and the findings, not the
   architecture.** What justifies it: the systematic **ablation trail**
   showing *why* each simpler alternative structurally fails (Gaussian → GMM → GAN → flow, each with
   a mechanism, not a preference); the **problem formulation** itself; and whatever the attacker
   actually discovers. Comparable precedent exists at solid venues. The bar is whether the
   combination produces insight the parts alone could not — which is exactly why the negative results
   in [Appendix A](#appendix-a--experiment-history) are load-bearing rather than embarrassing.

## 4.2 Open technical questions

**Opened 2026-09-27 (not yet numbered):**
- **Damage after error correction.** run003 breaks frames with ~1 bit error each (PER 0.1 at BER 4e-4);
  any FEC would repair them and on/off would win again. Needs the per-frame error histogram (§4.3 E2
  follow-on 1) or FEC in the link (Sionna LDPC) — decides whether "frames per alarm" survives a real link.
- **Black-box transfer.** run004 transferred between two CNNs of the same architecture and recipe; a
  detector of a different architecture is untested.
- **Is the energy floor SNR-dependent? — ANSWERED 2026-09-29 (S4, S5): yes, and the "floor" wording is
  withdrawn.** With known gain and noise the best generator is flagged by energy on 63 % of frames at
  PER 0.1 at 30 dB and 26 % at 15 dB, whether trained at 30 or 15 dB (§3.3m); at 15 dB an unknown noise
  level of ±1–2 dB drops it to 9–14 % for an honest detector (§3.3n). Below 15 dB is unmeasured with
  trained generators.

**Q1 — Does M1 still have a target? — ANSWERED 2026-09-12, and not by G1.**
The question was whether a *stealthy* learned attacker had headroom over the closed form, with G1 as
the gate that would decide. **The literature check answered it first and differently: there is no
headroom worth chasing on the stealth axis, and the fact that there isn't is published** (§2.1). So
the "headroom ≈ 0" branch below is the one that obtains — but its stated consequence has changed too.

The old branch said: with no headroom, the generator's job becomes rediscovering the closed form and
the claim becomes the learned-vs-optimal detector gap. **That is still a true claim and it is
surviving item 1 in §2.1.** What the pivot adds is that it is not the only option — M1 does not have
to rest on the V>1 mechanism alone, because **coordination itself is the target** (§2.3 RQ1): N_J
jammers splitting power and phase has no closed form to be beaten by, so the "a learner can at best
rediscover the closed form" objection simply does not apply there.

*Original two-branch text kept below because G1 is still optional and its output is still readable
against it:*
- **Headroom exists** → the generator is chasing a quantified gap, and we can report how much of it a
  learned policy recovers. Strictly better than "our generator got BER X".
- **Headroom ≈ 0** → the generator's job changes from *beating the closed form* to *rediscovering it
  without the genie's information*, and the paper's claim becomes about the **learned vs optimal
  detector gap** (adaptation cost) rather than attack effectiveness. Still a paper; different
  headline.

**Q2 — Does the learner have a job at all? (§13.1 of the old checklist; G4 answers it.)**
His sharpest challenge: *"Is it a valid assumption that all legitimate symbols are equally spread?
→ RL shines when it can find something"* + *"scrambling makes the transmitted sequence look
statistically random"*. Read together: if the payload is iid-uniform — **and real systems scramble
precisely to guarantee that** — there is no payload structure to discover, the minimum-energy attack
is closed form, and **a learner can at best rediscover it.** E1 confirms the closed form is already
at the limit for the single link.

Not fatal, but it *relocates* the contribution. The structure that **survives scrambling** is:
1. **Protocol-deterministic structure** — preamble, pilots, guard/DC nulls, control signalling.
   Exactly the "protocol-aware attack" the Introduction claims, so claim and method finally line up
   — and an argument for keeping *pilots* in M0 even though nothing else survives the simplification.
2. **The detector's decision surface** — signature-shaping searches the *defender's* model, not the
   payload distribution. Symbol statistics are irrelevant to it.
3. **Channel and geometry** — per-link gains and phases (the M1 spatial step).
4. **Coordination** — how N_J jammers split power and phase so their perturbations add at the victim
   while each stays under threshold. No closed form; a genuine joint optimization.

**Turn the objection into an experiment (G4, cheap in M0):** sweep *the amount of exploitable
structure* — iid-uniform → non-uniform symbol priors → correlated/unscrambled → pilots present — and
show the learned attacker's advantage over the closed-form boundary attack **appear exactly as
structure appears**. That answers his question with a curve instead of a paragraph. **Record the
honest risk now:** if the advantage never appears, the single-link case is *solved by the closed
form* and the entire learning contribution lives in (3)+(4) — which is also the destination he cares
most about.

**Q3 — The CLT/Gaussianization coordination story is probably dead. Repair `sec_system_model.tex`
before it goes to him.** §Cooperative currently argues that independent per-agent generators drive
`d = Σ d_k` Gaussian, degenerating `D_NP` toward an energy detector. But E1 measured the Gaussian
perturbation as the **least effective attack in the ladder** (excess BER 0.0007 at σ = 0.5), so that
mechanism plausibly reduces to *"coordinate to be useless"*. The **second** mechanism in that
subsection — **V > 1 receivers**, where concavity of error rate in perturbation power makes spreading
beat concentration at matched worst-case detectability — survives, and it is where
`valianti2024cooperative`'s coupling actually transfers. It needs a multi-user extension that does
not exist yet. **Either repair the paragraph or cut it to the V>1 argument alone.**

**Q4 — The scoping fork — RESOLVED 2026-09-12: cooperation is the DESTINATION.**
The fork was whether multi-agent cooperation is the destination (the thesis is about *cooperative*
jamming; single-agent signature-shaping is a stepping stone) or the garnish (the thesis is about
*learned evasion*; cooperation is an extension). **Resolved toward destination**, because the
evasion branch is what §2.1 retired — so the fork collapsed rather than being decided on its merits.
Two things corroborate the landing: the registered title already says "Cooperative Multi-Agent", and
Di Maio's own stated destination is *"the optimal multi-jammer coordination against one or more
mobile victims"* (§B.2). **Pending his sign-off (§4.1 #0), which is the only thing that could
reopen it.**

**Q5 — E1 hygiene, cheap to fix (§3.2, §3.3).** Add the σ = 0 anchor; put the σ grid on a proper log
spacing; re-caption anything quoting the `boundary_genie` row to say it is the ρ = √2 permutation
degeneracy; fix the meaningless `power` column for the `counter_*` attacks.

**Q6 — Related Work inclusion calls.**

> **⚠ FIVE MANDATORY ADDITIONS, 2026-09-12 (§2.1). These are not optional and not "nice to have" —
> they are the prior art the retired headline collided with, and a paper in this area that omits
> them is not defensible.** None is in `refs.bib` yet; none has a key yet.
> 1. **Bash, Goeckel & Towsley**, "Square root law for communication with low probability of
>    detection on AWGN channels", **IEEE ISIT 2012**, pp. 448–452, DOI `10.1109/ISIT.2012.6284228`.
>    Extended journal version: IEEE JSAC 31(9), 2013. **Checked 2026-09-12: the key
>    `bash2013limits`, used throughout this README, is NOT in the live `paper/refs.bib`** — it
>    exists only in the staging file `paper_drafts/refs_new.bib`. The live bibliography has **no
>    covert-communication entry at all**. So every README sentence citing `bash2013limits` (§2.8's
>    divergence row, §2.7's stealth-budget convention) currently points at nothing in the paper.
> 2. **Csiszár & Narayan**, arbitrarily-varying-channel symmetrizability (1988) — the general form of
>    the constellation-symmetry result.
> 3. **Li et al. (Tongtong Li's group), disguised jamming** — CDMA: IEEE TIFS 2016,
>    DOI `10.1109/TIFS.2016.2585089`; OFDM/SP-OFDM: IEEE TIFS 15, 2020. **The nearest neighbour of
>    all, and the one to read first.**
> 4. **Amuru & Buehrer**, "Optimal jamming strategies in digital communications — impact of
>    modulation", **IEEE GLOBECOM 2014**, pp. 1619–1624; extended IEEE TIFS 10(10), 2015,
>    pp. 2212–2224. The closed-form effectiveness half of our attacker ladder.
> 5. **The stealthy-FDI / CPS line** (one representative citation is enough) — for G1's divergence
>    program and the Chernoff–Stein bridge.
>
> **Also owed: a Related Work paragraph that positions us against these**, not around them. The
> honest positioning after the pivot is: these establish what a *stealthy* attacker cannot do; we ask
> what a *coordinated* one can, and use their bound as the instrument (§2.3 RQ1).

The two long-open inclusion calls are RESOLVED. Include both.
Settled in `paper/Sources_And_Evaluation.md`; recorded here so they are not reopened.
- **Hameed, György, Gündüz, "The best defense is a good offense"** (IEEE TIFS, vol. 16,
  pp. 1074–1087, 2021, DOI `10.1109/TIFS.2020.3025441`, key `hameed2021offense`) — **include.** The
  cleanest published instance of the *dual* objective we adopt: perturb symbols so a learned
  classifier fails while the intended receiver still decodes. Structurally identical to "maximize BER
  subject to a detectability budget", with the two objectives swapped in sign. *Differs:* covert
  comms, not disruption — the perturbation protects a friendly link rather than destroying a hostile
  one. `sec_related.tex` already lists it under "never cut", as part of the evasion lineage with
  `delvecchio2020spectral`.
- **Ziemann & Metzler, "Adaptive LPD radar waveform design with generative deep learning"** — **include.**
  ⚠ **Update the citation:** it is no longer arXiv-only. Now *IEEE Transactions on Radar Systems*,
  vol. 3, pp. 417–429, **2025**, DOI `10.1109/TRS.2025.3542283`. **This kills the original objection**
  ("arXiv-only, cross-domain, might distract"). It is the strongest existing validation of exactly our
  paradigm — a generative model producing waveforms simultaneously effective and statistically
  indistinguishable from the background, trained against a critic. Cite as cross-domain corroboration
  in one sentence.
- ⚠ **Also correct while you are in `refs.bib`:** the flow-policy citation `ward2019nf_rl` is a
  **workshop paper, not peer-reviewed proceedings**, and its third author is **Bose, not "Bhatt"**.
  The recommended peer-reviewed replacement making the same claim is Mazoure, Doan, Durand, Pineau &
  Hjelm, "Leveraging exploration in off-policy algorithms via normalizing flows", **CoRL 2020**, PMLR
  vol. 100.

**Decided, for the record: PyJama is DROPPED** (Ulbricht/Marti et al., SPAWC 2024, arXiv:2407.15473,
ETH Zurich IIP). A differentiable jamming library on Sionna using SGD for power allocation over an
OFDM grid. It shares almost none of our axes — no waveform synthesis, no RL, no multi-agent, no
stealth objective — so it would be a row of "No" across every table column. Its only connection is
"also uses Sionna", which is tooling, not a contribution. If mentioned at all it belongs in
Methodology when introducing Sionna, not in Related Work. *(It is also Sionna 0.x + TensorFlow, so it
was never usable as a dependency; the Clancy-2011 pilot-nulling strategy it builds on is ~20 lines
from scratch if ever wanted.)*

**Q7 — Zhou 2025: what the paper leaves unstated, and what we use instead (CGAN track, §2.10).**
Filled in as each value is fixed; "calibrated" rows are decided by §3.4 C2. Every row marked *default*
is a choice of ours and must be called one if these results are ever written up.

| Parameter | Paper | Ours | Status |
|---|---|---|---|
| Segment length | **1024** in Figs. 2–3; **1200** in the §II-C text | 1024 | fixed — the figures are self-consistent (256 × 128 = 32768 flattened) |
| D feature map | text: (256, 1, **28**) | (256, 1, **128**) | fixed — "28" is a typo; Fig. 3 flattens to 32768 |
| Batch size | "126" as rendered in the figure labels | 128 | fixed — almost certainly 128 at low resolution |
| Samples per symbol | — | **8** | **assumed** (C2 checkpoint, §2.10). The calibration grid was {2, 4, 8, 16}; no value fits the whole Noise curve |
| Pulse shape | "identical filtering" to the target, unspecified | **RRC β = 0.35** | **assumed** (C2 checkpoint). β = 0.25 fits the Optimal curve slightly better and was deliberately not chosen |
| RRC filter span | — | **32 symbols** | fixed (C1). Span 8 left summed truncation ISI of 3.6 % of the symbol amplitude (β = 0.35), which flipped symbols in the zero-noise geometry checks (verify job 2259141: BER 1.3e-3 where 0 is predicted). Span 32 → 0.3 %. The link must not create errors the paper's model does not have. `verify.py` now asserts ISI < 0.5 %. |
| Noise-jammer bandwidth | — | **full sample band** (white) | **assumed** (C2 checkpoint); the in-band alternative fits far worse |
| JSR / SNR definitions | SNR 30 dB, JSR −10…10 dB, no definitions | mean per-sample power ratios at the RX input | default |
| JSR grid | not stated; axis ticks every 2.5 dB | **−10:2:10 dB**, read off the marker positions | fixed (C0) |
| Receiver | — | coherent matched filter, symbol-time sampling, per-axis hard decision | default |
| Jammer synchronisation (Optimal **and GAN**) | "identical filtering and modulation parameters"; nothing on timing or phase | **asynchronous**: uniform timing offset in [0, sps) samples + uniform carrier phase, both per frame, for both structured jammers (`jammers.desync`) | **assumed** (C2 checkpoint; extended to the GAN 2026-09-15, §2.10). The only variant that produces errors below −3 dB JSR, as the paper's curve does. **run001 scored the GAN locked** — see §3.3b |
| Adversarial loss | BCE equations (1)–(6) **and** "gradient penalty" in the D-loss list, with an InstanceNorm critic | **WGAN-GP** (λ = 10, n_critic 5, Adam β = (0.5, 0.9)); `--adv nsgan` (eqs. 4/5 + GP) as ablation | **from ref. [7]** (2026-09-15): Saarinen & Koivunen 2020 is a *conditional WGAN-GP* (paper paywalled; per Aalto follow-ups and MathWorks' port of it), which is also why the critic uses InstanceNorm. Eqs. (1)–(6) are textbook background from [4]. run001 used nsgan, n_critic 1, β = (0.5, 0.999) |
| G conv blocks | Conv1d–BN–LeakyReLU–Dropout–MaxPool ×3, sizes unstated | channels 64/128/256, kernel 5, pool 2, dropout 0.3 | default |
| D conv blocks | Conv2d–(InstanceNorm)–LeakyReLU–MaxPool ×3, sizes unstated | (1×5) kernels, (1×2) pools, 64/128/256 channels | default |
| "Time-frequency discrepancy" loss | STFT of generated vs target (Fig. 1) | L1 between log(clamp(batch-mean \|STFT\|, 1e-5)) of real and generated — unpaired, since z is random | **from ref. [5]** Fre-GAN / HiFi-GAN's spectrogram compression (2026-09-15). run001 used log1p(\|S\|²), which barely penalises an out-of-band floor |
| Feature-matching loss | named only | L1 between batch-mean final D features (the one feature output Fig. 3 exposes) | **L1 from ref. [5]**, batch means from Salimans et al. 2016 (no paired real sample). run001 used squared L2 |
| "I/Q distribution distance" loss | named only | sorted-sample 1-D Wasserstein on the I, Q and \|x\| marginals | default — no reference defines it |
| Loss weights, optimiser, LR | — | G: adv 1, feat **2**, STFT **45**, I/Q 1; D: adv 1, GP 10, cls 1; Adam 2e-4 | **feat/STFT weights from ref. [5]** (λ_fm = 2, λ_mel = 45); rest default. run001: all weights 1 |
| Training data | "target signal", normalised, inverse-normalised on output (cites [5]) | clean QPSK waveform segments ÷ a global peak scale, largest sample → **0.95**; G output × that scale | **headroom from ref. [5]**'s loader (`normalize(audio) * 0.95`); run001 mapped the peak to 1.0, Tanh's asymptote. Irrelevant to BER: JSR is imposed by projection |
| Ref. [6] GAN-TTS (cited for D's "multitask capability") | — | nothing taken | conditional + unconditional random-window discriminators; with one class there is nothing to transfer |
| Iterations / test protocol | 10,000 iterations; 10⁴ symbols × 100 trials | same; plus trials until ≥ 100 bit errors or 10⁹ bits | stated + extended |

**Properties of the paper that shape both steps**, recorded so they are not rediscovered:
- **(i) BER is not in any loss term.** The generated jammer is effective only because it imitates the
  target modulation, so "conditioning on stealth" gets no help from the paper's objective.
- **(ii) Parts of Fig. 6 cannot be Monte-Carlo output from the stated protocol.** The digitised Noise
  curve reaches **8.9·10⁻¹¹ at −10 dB and 4.6·10⁻⁸ at −8 dB**; ≈2·10⁶ bits cannot resolve either value.
- **(iii) The top of the GAN curve exceeds BER 0.5** — 0.53 at +8 dB and 0.54 at +10 dB (C0; the
  markers sit visibly above the 0.5 level, not a reading error). A jammer that is independent of the
  payload cannot push Gray-QPSK BER past 0.5, so either the GAN jammer was correlated with the
  transmitted data, or these points were not measured as described. Its plateau near 0.5 is also what a
  **locked** jammer gives (0.50), while the paper's Optimal plateau (0.38 at +10 dB) matches **async**
  (0.39, C2). So Zhou's GAN may have been scored synchronised and their Optimal not — the same
  mismatch run001 had (§2.10, §3.3b).
- **(iv) Their own "optimal" is not an upper bound in their own figure.** GAN > Optimal at every
  JSR ≥ +2 dB, and Noise ≈ Optimal from +4 dB.
- **(v) The Noise curve is not a single Q-function** (a check on the digitised points, ahead of C2).
  Inverting BER = Q(√(g·10^(−JSR/10))) point by point gives an implied processing gain **g = 6.10,
  6.56, 6.70, 7.48, 8.22, 8.87, 5.45, −0.03 dB** at JSR −10, −8, …, +4 dB. It climbs steadily by
  2.8 dB and then collapses, with BER jumping 25× between 0 and +2 dB. A white-noise jammer through a
  fixed matched filter has a constant g. So C2's closed-form fit is expected to leave a systematic
  residual, and C2 must report it rather than hide it inside the parameter choice.
  **Extended 2026-09-15 to all three curves** — this is *why the figures look different*
  (`evaluate.py` → `artifacts/cgan/run001_async_implied_gain.png`, same inversion with the link's
  noise term, BER = Q(√(1/(1/(sps·SNR) + JSR/g)))). The paper's **Optimal** implies an almost flat
  g = 1.9–3.1 dB over −10…+2 dB and its **GAN** 2.7–4.0 dB over −10…0 dB: the signature of
  *Gaussian-like* interference, whose tails give errors at any JSR, hence smooth low-JSR slopes. Ours
  cannot do that: noise is flat at exactly 9.03 dB (the closed form, which validates the diagnostic),
  while async Optimal falls 3.6 → 0.3 dB over −6…−2 dB and is zero-error below. A same-pulse QPSK jammer
  through a Nyquist matched filter is *bounded*: it flips nothing until √JSR·Σ|g_k| > 1/√2, ≈ −7.7 dB
  for RRC 0.35 async, and SNR 30 dB noise is too weak to smooth that edge — hence our cliff. No sps,
  pulse or sync in C2's grid changes this, and no reference defines the jammer, so the low-JSR shape
  of Fig. 6 is **not reproducible under a stated physical model**; it is not a GAN problem.
- **(vi) The paper names two different noise baselines.** §III: "conventional Gaussian noise
  interference"; §IV (conclusion): "traditional frequency-modulated noise jamming". We use Gaussian
  (the experiment section's wording).
- **(vii) The references define none of the link.** [7], cited for the "optimal" baseline, is a
  radar-waveform GAN paper; [5] and [7] do inform the training recipe (Q7 table); [1], [2], [4] are
  background, [6] is not transferable to one class, and [3] is a garbled citation (Mirza & "Osindero",
  invented venue).

What (ii)–(v) mean for the comparison: it is read at **JSR at BER 1e-3**, the region the paper's
protocol can resolve and where no curve is occluded. Matching the whole figure was already ruled out.

**Paper values at BER 1e-3** (linear interpolation in log₁₀ BER between the bracketing digitised
points; exact values from `calibration.json`): **Optimal −7.06 dB, GAN −5.76 dB, Noise −1.32 dB.**
- Noise − GAN = **4.44 dB**, inside the abstract's "2–6 dB".
- GAN − Optimal = **1.31 dB**, which fails the original ≤ 1 dB bar by 0.3 dB, well beyond the
  ±0.07 dB reading error. **That is why the bar became report-only** (§2.10, decided at the C2
  checkpoint).
- Noise − Optimal = **5.75 dB**, the gap the C2 link choice is judged on.

**Q8 — Which statistical detector for step 2? OPEN — to be discussed with the user before step 2.**
Not decided. Candidates to bring to that discussion:
- **kurtosis / higher-order cumulants** — the detector of sim02–04, cheap and classical;
- **a cyclostationary (spectral-correlation) test** at the symbol-rate cycle frequency — the natural
  test against a jammer that imitates the target's modulation, since it looks at exactly the structure
  the CGAN copies;
- **a GLRT / likelihood-ratio test** on matched-filter outputs — the waveform-level descendant of M0's
  NP test; needs a stated jammer model;
- **eigenvalue-based detectors** (max–min eigenvalue ratio) — blind, no noise-power knowledge needed.

**RESOLVED 2026-09-17 — kurtosis, and it is built (§3.3d).** The baseline set uses **kurtosis** as the
statistical detector (cheap, classical, the sim02–04 detector; `cgan/detectors.py`), with one- and
two-sided **power**, the **spectrogram-CNN** (Li et al., retrained), and the exact **noise NP test** as
the ceiling row. The cyclostationary and GLRT options stay on this list as later additions if the
frontier warrants; the CWT-CNN (Zhang & Krunz, Q10) is the gated extension.

The power threshold (detector i) follows M0's convention: larger statistic = more suspicious,
threshold set from the empirical (1−α) quantile of clean segments (`m0/detectors.py` `calibrate`).

**Q9 — Zhou's "optimal" jammer is not optimal in the Amuru–Buehrer sense.** Zhou's reference is a
matched-modulation waveform; the BER-maximising jammer under an average-power constraint (Amuru &
Buehrer, §2.1) is generally *pulsed* at low JSR. For step 1 we reproduce Zhou's definition, because
that is the claim under test. **For step 2, decide which ceiling a stealth-conditioned generator is
measured against** — this matters as soon as effectiveness at matched detectability is reported.

**Q10 — Zhang & Krunz is an adaptation, not a drop-in baseline.** The original classifies clean vs
preamble / pilot / interleaving jamming on 802.11ac OFDM (20 MHz, TGac-B channel, SNR 20 dB,
400-sample windows = 5 OFDM symbols, scalogram input 400 × 100). Single-carrier QPSK has none of
those attack classes. What transfers is the **representation (Morlet CWT, f_b = 2, f_c = 1) and the
network (DCNN₁)**, retrained on our clean-vs-jammed segments over a mixture of JSRs. Report it as
*"the Zhang & Krunz detector architecture, retrained"*, never as their detector. `pywt` is not in the
venv, so the CWT will be written in torch.

**Q11 — Which training recipe does step 2 retrain with? LARGELY MOOT for D2 as built (2026-09-20, §3.3f).**
D2 does **not** re-run Zhou's adversarial GAN training — it warm-starts `run001_G` and fine-tunes by
**direct gradient** on the detector-aware objective, so "which GAN recipe" no longer gates it. The note
below stands only if a *GAN-adversarial* retrain is ever wanted. Step 1 closed on `run001_G.pt` (§2.10),
but that recipe is no longer in the code: `train_cgan.py`
defaults to run002's (WGAN-GP, n_critic 5, feat 2 / STFT 45, L1 feature matching, log-clamped STFT,
headroom 0.95), which produced a worse imitation (§3.3b). `--adv nsgan --n-critic 1 --w-feat 1
--w-stft 1` restores run001's adversarial loss and weights, **not** its MSE feature matching, log1p
STFT or headroom 1.0. Two things are unknown:
- **Why run002 failed** — the adversarial loss or Fre-GAN's weights. The parked ablation (option (b),
  ~1.5 h): WGAN-GP with all weights 1.
- **Whether the D-wins-outright state matters for step 2.** Both runs end with the discriminator
  separating real from generated (§3.3b item 4); a stealth term added to G's loss competes with a
  discriminator that is already not being fooled.
Before retraining: either make run001's loss definitions selectable again or run the ablation, and
decide with the user.

**Q12 — MARL coordination design (D4). SETTLED 2026-09-26 (user picked the claim and the observation;
the action space was delegated and is defined here). One sub-question stays open (last bullet).**
The E3 pre-check (§3.3h) bounds the problem: with the co-located detector the only levers are **timing
alignment** and the **power split**, the ceiling is the containment bound (one jammer sending the aligned
sum) plus Amuru Thm 4, and stealth is not a lever. The user's directive is **minimal inductive bias**: the
policy should find those levers itself.
- **Claim: the delay-decay curve (option ii).** The x-axis is the timing error σ [symbols] that the
  inter-jammer link introduces (the supervisor's axis, §B.2). A **leader** holds the victim's frame
  timing, as the single aligned D2 does. Each **follower** receives that timing over the inter-jammer link
  with error ε_k ~ N(0, σ²). Grid σ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64} plus *uncoordinated*
  (uniform over the 128-symbol burst period). The y-axis is P(det) at matched excess BER 3e-4 and the
  total JSR needed, both reported. **Arms:** learned team · geometric heuristic (compensate its own
  propagation delay exactly, full power) · uncoordinated · single jammer at the team's received total
  (the ceiling). The rejected options: (i) "K drones ≈ one strong jammer" is already shown by hand in
  E3; (iii) a separated warden needs the parked geometry layer and does not fit before ICC.
- **Observation (user): each drone's own geometry only.** Its position relative to R (3-D, normalised by
  the box) and its own sync-error level σ_k, which is local knowledge of its own link quality. Nothing about
  teammates. One shared-parameter MLP maps observation → action. **CTDE:** trained centrally by direct
  gradient over random drops (`scene.draw_positions`), executed per drone. Evaluated on the 50 fixed
  test drops.
- **Actions, two scalars per drone:**
  1. **u_k ∈ [0, 1]** — the fraction of a **per-drone** hard cap P_J/K that the drone transmits
     (sigmoid, an inequality: a drone may go quiet). The cap is the environment constraint; power never
     enters the loss (§2.8). The cap is per drone *on purpose*: with a free split of a shared total, the
     team puts all the power on one drone and reaches the ceiling without coordinating, which measures
     nothing.
  2. **δ_k** — a continuous transmit advance [symbols] that compensates its propagation delay
     τ_k = d_kR/c. It arrives at offset τ_k − δ_k + ε_k, plus the uniform sub-symbol async that every
     jammer has (§3.3d). The heuristic sets δ_k = τ_k.

  **Excluded, with the reason:** the *waveform* is the frozen CNN-targeted D2 (β = 10) with fresh z per
  drone. That keeps coordination separate from the §3.3f waveform result; conditioning the generator
  per drone is a possible v2. *Position/movement* is out: with a co-located detector, position acts
  only through g_kR and τ_k, which u_k and δ_k already span at R, and mobility is §4.3. *Carrier phase*
  cannot be computed (§3.3d).
- **Objective:** `−(log E[BER] − β·soft P_det,CNN)`, β ∈ {0, 10} as in D2 (§3.3f), averaged over drops,
  σ from the grid and a budget band. MAPPO/PettingZoo is a fallback only (§2.9); re-read §A.5 before
  writing any RL.
- **What learning can find beyond the heuristic:** (a) δ from geometry, without being told that delay is
  proportional to distance; (b) at large σ, going quiet on badly synced or weak drones to stop smearing,
  which the always-full-power heuristic cannot do. If it only rediscovers the heuristic, report that as
  the minimal-bias result, not as a failure.
- **Staging.** **D4a (no training)** is the heuristic/floor/ceiling curve by transfer: equal received
  split, lossless, K = 2/4, 15 and 30 dB. It is built as a σ timing mode in `team_fading.py`
  (`--sigmas`, `--art ../artifacts/cgan/team_timing`), checked by `verify.py` §17, and tabulated by
  `team_figures.py --timing`. It locates the curve's knee. **Decision rule:** if the knee sits above the
  box's geometric delay spread (≤ 4.7 symbols: the 1,422 m diagonal at 1 MBd), δ_k is dead weight and the
  learned lever reduces to u_k. **Applied 2026-09-26 (§3.3i): the knee is at σ ≈ 1 symbol and
  uncompensated test drops sit at 1.1 symbols rms, so δ_k stays.** **D4b** is the policy (§3.3j).
- **D4b risks:** loss vs δ is periodic and non-convex (bursts), and whole-sample shifts are not
  differentiable, so δ needs a differentiable fractional delay (an FFT phase ramp on a padded stream;
  `channel.py`'s sinc taps reach only 8 samples). Training with σ > 0 jitter smooths the landscape.
  **The fractional delay is built (`team_policy.frac_delay`) and the risk materialised (§3.3j).**
- **OPEN — why the policy does not learn δ = τ (§3.3j).** A one-drone δ = τ minimum exists (~0.6 loss
  units), and raising the effective batch from 32 to 256 did not help, so gradient noise is not the whole
  story. Candidates, none tested: (i) a common shift of all δ is almost free, and the frozen generator
  slightly prefers some common offset, so the optimiser spends δ on that; (ii) an MLP has to learn a
  Euclidean norm from raw position through a shallow well; (iii) credit is diluted over K drones. The
  options the user was offered and has not yet chosen between: accept it and add an explicit δ = 0 arm
  (~2 min of eval) · give the policy its distance to R (the inductive bias the user wants to avoid) ·
  per-drone credit assignment or δ-only training · move on to the teammate arm or K = 2.
- **Second arm: teammate information (user, 2026-09-26).** Besides its own observation, each drone
  also sees its teammates' relative positions and σ_j, pooled over teammates (a mean over a shared
  embedding, so drone order does not matter). For timing this should not help, since each drone aligns
  to R's frame and needs only its own τ_k. For power it can: at large σ, deciding *which* drones go
  quiet is a joint decision that own information alone cannot settle. **The gap between the two arms at
  large σ is the value of communicating over the inter-jammer link.** If the gap is zero, that is a
  finding too: coordination needs a shared clock, not shared state. Own-geometry stays the main arm.
- **OPEN — how a link *delay* maps to the timing error σ.** The realistic delay range is unknown (§4.1
  #8), so σ is swept rather than derived. A stated assumption is owed in the Methodology if D4 reaches
  the paper.

**Q13 — Which SNR, noise-level uncertainty and fading are realistic for a QPSK UAV link? DECIDED
2026-09-29 (user); VERIFIED 2026-09-30 (session A of the final experiment, §3.4).** The values: our SNR
15 dB = Es/N0 24 dB · σ_N 0.5 dB main, 1 dB extra · Rician K 12 dB main, 28 dB extra · α = 0.05.
**No source contradicts a decided value.** Every source below was read in full text unless it is marked
otherwise; the page of each cited passage is given. Paper text: `paper_drafts/sec_setup_operating_point.tex`
(its bib entries are in `paper_drafts/refs_new.bib`). Figures and their script (numpy, login node):
`artifacts/final/literature/`. Local copies of the PDFs: `source_papers/` (git-ignored). The report page
is linked at the end of this entry. User, 2026-09-29: *"as long as we can argue that 15 dB is realistic,
that is good, especially with higher dB you would use more fine grained modulation schemes"*.
- **What the 2026-09-29 notes had wrong (corrected here; do not quote the old versions).**
  - The 2024 *Ad Hoc Networks* survey does NOT recommend CNPC SNR > 20 dB. Qazzaz et al., "Non-Terrestrial
    UAV Clients for Beyond 5G Networks: A Comprehensive Survey", Ad Hoc Netw. 157:103440, 2024, was read in
    full (the open copy at <https://eprints.whiterose.ac.uk/209662/>). It gives no SNR value at all, and the
    string "dB" does not occur in it. Dropped.
  - The Tandra & Sahai quotes came from the Berkeley preprint. The published JSTSP text differs: "residual
    uncertainty … even after run-time calibration" is not in it, and "device level" is missing from the
    Fig. 4 caption. The "Fig. 5" of the preprint is Fig. 8 of the paper. Cite the published pages below.
  - 960–977 MHz is not a WRC-12 allocation. WRC-12 added a new AM(R)S allocation at 5030–5091 MHz for
    line-of-sight CNPC. 960–977 MHz is the portion of the existing 960–1164 MHz AM(R)S allocation that is
    *proposed* for CNPC.
  - Khawaja et al.'s K-factors are running text (Sec. IV-C), not a table. The "near-urban 13.1 / 28.7 dB"
    snippet is in none of the four papers; Part III gives 12.0 / 27.4 dB. Dropped.
- **Two weak spots in the framing, not in the values** (stated in the paper text, top of the report).
  - **CNPC prototype radios use GMSK, not QPSK.** Bishop et al., NASA/CR-2017-219380, 2017, §3.1–3.5 and
    Tables I–II (pp. 2–3): GMSK with BT 0.2, 1 bit/symbol, 103,500 or 138,000 symbols/s in 90 or 120 kHz.
    RTCA DO-362 is waveform-agnostic (public description; the standard itself was not read, it is
    paywalled). So "fixed modulation" is supported and "QPSK" is our stand-in. The draft says so. MSK is
    offset-QPSK with half-sine pulses, and GMSK is a smoothed MSK.
  - **The K-factors come from a manned aircraft high above the ground.** The aircraft was NASA's piloted
    S-3B Viking at 500–1000 m above ground and 75–92 m/s, with a 20 m ground mast and 5 MHz (L) / 50 MHz (C)
    sounders (Part I pp. 29–31 and 35; Part III p. 6617). A low-flying small UAS (lower elevation angle; Khawaja
    p. 2373: K grows with elevation) and a narrower-band receiver (it cannot resolve multipath off the LOS
    tap) would both see a *lower* K. K = 12 dB is therefore not the defender's worst case. Airframe
    shadowing "strongly affects" K (Part II p. 1919) and is not modelled.
- **Units.**
  - Our SNR is power per sample over the 8× band, so Es/N0 = SNR + 9.03 dB and Eb/N0 = Es/N0 − 3.01 dB.
    Our 15 dB is therefore Es/N0 24.03 dB and Eb/N0 21.02 dB, and our 30 dB is Es/N0 39 dB.
  - Kakar's 14 dB is SNR in the channel bandwidth. That equals Es/N0 when the noise bandwidth equals the
    symbol rate, and is within 1.3 dB of it for the RRC-0.35 occupied bandwidth. It is ≈ our 5 dB.
  - A ±x bound is not a log-normal std. The equivalents of ±1 dB are:
    - σ 0.50 dB, reading the bound as ±2σ (95.4 % of frames inside);
    - σ 0.58 dB, the std of a uniform draw on ±1 dB (Shellhammer & Tandra's and He et al.'s bounded
      model);
    - σ 0.61 dB, Sonnenschein & Fishman's tail rule at a 0.05 tail (0.32 dB at their 10⁻³ tail);
    - σ 0.41 dB, the RSS of Shellhammer & Tandra's three components. They add them linearly, to 0.70 dB.

    σ_N 0.5 dB sits in the middle of that range.
- **SNR: Es/N0 24 dB = a fixed-modulation control link 10 dB inside its range-edge requirement.**
  - **Kakar, M.S. thesis, Virginia Tech, 2015, Table 4.3 (p. 41), re-read.** A LoS budget for MAV/SUAV C2
    links (QPSK-OFDM, 1 / 5 GHz, 12.5 / 37.5 kHz channels): RX SNR 14 dB = 6 dB theoretical minimum + 2 dB
    implementation loss + 6 dB aviation safety margin, excess margin 0 dB, at 30.0 / 23.8 km (1 GHz) and
    10.1 / 7.1 km (5 GHz). The 6 dB "theoretical minimum" is not derived in the thesis (p. 40). It is a
    thesis, not a standard. Handle <http://hdl.handle.net/10919/53512>.
  - **The argument.** 24 dB is 10 dB above that range edge: 1/√10 ≈ 0.32 of the range in free space.
    Uncoded QPSK has clean BER 2.7·10⁻⁷ at 14 dB and 3·10⁻⁵⁷ at 24 dB, so every counted error is the
    jammer's.
  - **Shalkhauser et al., NASA/TM-2017-219379, 2017, p. 28** (CNPC prototype flight test). At 100 nmi with
    partial terrain obstruction, excess path losses of 18.3–20.4 dB caused the random frame losses, so
    ≈ 20 dB of margin over free space would have avoided them. The report says this "compares favorably
    with" DO-362 App. L and is "likely a far too conservative approach for many flight situations". It
    rests on one flight segment. In the bib, but not in the draft text; kept for a rebuttal.
  - **Adaptive-modulation argument: dropped.** It argues against us: Wang & Abdelhadi, arXiv 1507.07159,
    Eq. (1) and Table I, re-read: CQI = 0.5223·SNR + 4.6176, and QPSK is CQI 1–6, i.e. SNR < 4.56 dB. An
    adaptive link at Es/N0 24 dB would run 64QAM.
  - Lin et al., IEEE Commun. Mag. 56(4), 2018, Sec. IV and Fig. 5 (read 2026-09-29): cellular aerial UEs
    are interference-limited, with a median SINR of ≈ 3–5 dB. That is a different link (LTE, not CNPC). Use
    it only to argue that UAV links can be interference-limited (the 1 dB level).
  - What the literal, unconverted reading would have meant: E2 at our 5 dB (Es/N0 14 dB) flags every
    structured jammer on ≤ 0.08 of frames (CNN) and ≤ 0.22 (one-sided power) at matched BER (§3.3g).
- **Noise-level uncertainty: σ_N 0.5 dB (calibrated receiver), 1 dB (in-band interference).**
  - **Shellhammer & Tandra, IEEE 802.22-06/0134r0, July 2006 (public .ppt, read in full), slides 4–9 and
    11.** This is where the "1 dB" comes from:
    - temperature +20 K gives 0.28 dB, LNA gain at 0.01 dB/°C gives 0.2 dB, and a 1 ms calibration
      estimate has std 0.22 dB;
    - together "at least 0.7 dB", then "Rounding up … Noise Uncertainty 1 dB" without interference;
    - "With interference the noise uncertainty may be much larger";
    - the range is N̄ ± Δ, with a uniform prior on the PSD;
    - conclusions: "may be larger than the 1 to 2 dB used in these simulations".
    - This is an engineering estimate for a TV-band sensor, not a measurement.
  - **Tandra & Sahai, IEEE JSTSP 2(1), 4–17, 2008.**
    - p. 7: σ² ∈ [σ_n²/ρ, ρσ_n²], x = 10 log₁₀ ρ, and Fig. 4 marks x = 1 dB (SNR wall −3.3 dB).
    - p. 13, Fig. 8: "ρ = 10 (to account for potentially significant interference)", which noise
      calibration reduces to λ = 10^0.1, i.e. 1 dB.
    - p. 5: background noise = thermal + leakage + aliasing + quantization + interference.
  - **Tandra & Sahai, DySPAN 2008, Fig. 2:** x = 1 dB with P_FA = P_MD = 0.1.
  - **Sonnenschein & Fishman, IEEE TAES 28(3), 654–660, 1992.**
    - p. 658: a log-normal noise-level error, "standard deviation of σ dB". The example σ = 1 dB gives
      "±3 dB" at the 10⁻³ tails.
    - p. 659 names the sources: antenna temperature varies "as a function of weather, pointing angle, and
      frequency" (an airborne antenna changes attitude), plus finite averaging and nonstationary noise.
  - **He, Yan, Zhou & Lau, IEEE Commun. Lett. 21(4), 941–944, 2017.**
    - p. 942: sources "temperature change, environmental noise change, and calibration error"; the bounded
      model is log-uniform, the unbounded one log-normal (citing Sonnenschein & Fishman).
    - p. 944: σ_Δ ∈ {0.5, 1, 1.5} dB (Fig. 1), swept over 0–2 dB (Fig. 2).
  - **Nikonowicz, Mahmood, Sisinni & Gidlund, IEEE TIM 68(1), 105–115, 2019** (the journal version of arXiv
    1711.05642). Abstract p. 105 and Sec. VII: on industrial ISM noise measured with a USRP, the best
    estimator reaches < 0.5 dB RMSE (std 0.075 dB).
  - **The argument.** σ_N 0.5 dB keeps 95 % of frames inside the ±1 dB bound, and it equals a good
    estimator's error.
  - **Weak spots.** No source measures this for a UAS receiver. "1 dB = in-band interference" is our
    assignment (Shellhammer & Tandra's "1 to 2 dB" with interference, Tandra & Sahai's ρ = 10 case). The
    model is S5's and `final/link.noise_scale`'s: unit mean, per frame.
- **Victim-link fading: Rician K 12 dB (L-band) and 28 dB (C-band).**
  - **Matolak & Sun, IEEE TVT 66(1), 26–44, 2017 (Part I, over water).**
    - Table V (p. 37), mean K over sea 12.5 (L) / 31.3 (C) dB and over fresh water 12.8 / 27.3 dB;
      medians 12.7 / 31.0 and 12.9 / 27.0.
    - Sec. VI (p. 41): "mean K-factors ∼12 dB for L-band and 27–30 dB for C-band".
    - The stationarity distance is 15 m, and the fading is "unfiltered (memoryless) Rician".
  - **Sun & Matolak, IEEE TVT 66(3), 1913–1925, 2017 (Part II, hilly and mountainous).** Abstract (p. 1913)
    and p. 1919: 12.8 (L) / 29.4 (C) dB, K "almost independent of environment" because LOS is almost
    always present.
  - **Matolak & Sun, IEEE TVT 66(8), 6607–6618, 2017 (Part III, suburban and near-urban).**
    - Table V (p. 6613): near-urban median 12.0 (L) / 27.4 (C) dB; suburban medians 13.2–14.7 / 27.5–29.6.
    - The K spread around the fit is σ_Y 0.7–2.3 dB. Minima are far lower (−86 dB in L-band, i.e. LOS
      blockage).
  - **Khawaja et al., IEEE COMST 21(3), 2361–2391, 2019, Sec. IV-C (p. 2373).** Reports 12 / 27.4 ("urban";
    Part III says near-urban median), 12.8 / 29.4 and 12.5 / 31.3 dB. It says K "does not strongly depend
    on the GS environment", and that C > L partly because the C-band sounder bandwidth was larger.
  - **Bands.**
    - Kerczewski et al., IEEE Aerospace Conf. 2015, abstract: WRC-12's 5030–5091 MHz plus 960–977 MHz of
      the existing L-band allocation.
    - Bishop et al., NASA/CR-2017-219380 §3.1 (p. 2): ITU-R M.2171 set the requirement at 34 MHz.
    - Matolak Part I (p. 27): "tentatively granted"; the maximum UAS link range is ≈ 129 km.
  - **The argument.** 12 dB is the lowest L-band median, hence the strongest measured fading, and 28 dB lies
    inside the C-band range 27.3–31.3 dB.
  - **Why one draw per frame is enough.** P_det and damage are per-frame averages, so they depend only on
    the per-frame marginal of g. For the energy detector block fading is also the hard case: fading within
    a frame would be averaged out.
  - **Coherence time vs frame.** f_D = v·f_c/c, T_c = 0.179/f_D … 0.423/f_D. §3.4 decision 4's "the
    coherence time is milliseconds" holds only for L-band at small-UAS speeds:

    | band, speed | T_c | frames of 128 symbols at 1 MBd | at the CNPC radio's 103.5 kBd |
    |---|---|---|---|
    | L-band, 33 m/s (Kakar's SUAV, 120 km/h) | 1.7–3.9 ms | 13–31 | 1.3–3.2 |
    | C-band, 33 m/s | 0.32–0.75 ms | 2.5–5.9 | 0.26–0.61 |
    | the measurement aircraft, 90 m/s, C-band | 0.12–0.28 ms | 0.9–2.2 | — |

    The diffuse share of power is 5.9 % at K 12 dB and 0.16 % at 28 dB, so fading within a frame barely
    changes the frame's power.
  - **In numbers** (closed form / 2·10⁶ draws, `literature_figures.py`):
    - frame-power std 1.57 dB at K 12 dB (5th / 95th percentiles −3.04 / +2.05 dB) and 0.245 dB at 28 dB
      (−0.41 / +0.39 dB);
    - the post-matched-filter `energy` detector's clean margin at our 15 dB is exactly +1.29 % = 0.056 dB
      (noncentral χ², N = 128, α = 0.05). S5 measured +1.25 % for full-band power (§3.3n).
- **α = 0.05 — there is no standard value for jamming detection; say so.**
  - Stevenson et al., IEEE Commun. Mag. 47(1), 2009, p. 137 (IEEE 802.22): "The probability of detection is
    0.9, while the probability of false alarm is 0.1", per 2 s sensing decision.
  - Arcangeloni, Testi & Giorgetti, IEEE ISSE 2024, p. 6: learned (VAE) jamming detector at "Pfa = 0.05".
  - Mehrabian & Kaddoum, arXiv 2507.20504 (2025): p_fa = 0.01. Tandra & Sahai, DySPAN 2008: 0.1.
  - The draft's wording: equal α for every detector, so they are compared at equal false-alarm rates. It
    does not claim the ranking holds at every α (that is D2b, §3.4).
  - Weak spot: per frame, α = 0.05 is ≈ 390 false alarms/s at 1 MBd (≈ 40/s at 103.5 kBd). A deployed
    monitor would integrate frames.
- **What "CFAR" means here** (explained to the user 2026-09-29): the honest energy detector's threshold
  is calibrated on clean frames that include the noise-level variation, so its false-alarm rate stays at α
  and it gives up sensitivity instead — the NP convention every detector here already follows. It is a
  fixed-threshold CFAR; an adaptive CFAR that estimates the noise per frame (e.g. from the unused band
  edges of the RRC 0.35 spectrum) would be stronger, and a reviewer may ask for it.
- **Report page** (session A, 2026-09-30; private): <https://claude.ai/artifact/Tci2S5xPvXAzAVnVj8PygF>. It has
  every source with its quoted passage, the four figures with captions, the judgement calls and the drafted paragraph.

**Q14 — Open after the final experiment (2026-09-30, §3.3q). None blocks the read-out; each is the user's call.**
- **The band rule can starve a generator of low-power training.** It shifts the band with the target's
  white-noise transition. Where that transition moves far above the damage onset (naive energy under
  fading: band (−8, 0); `both`'s CNN band (−35.8, −3.8) for 2 of 3 seeds), the generator ends up damaging
  only when loud (PER 0.1 at −10 to −12 dB instead of −21). Options: (a) report as is (done); (b) extend
  the lower edge to the control's damage onset minus ≈ 5 dB and retrain only the affected generators
  (`fading`/`fading_k28`/`both` `energy_b10`, `both` `cnn_b10_r0/r2`; ≈ 30 min on the cluster). (b) is a recipe
  change, so it needs the user's go.
- **The paper's claim.** The CNN-targeted GAN has no gain over the classical envelope against the CNN
  (≤ 5 pp of room exists at all) and the detector term buys nothing; the robust finding is the detector
  side (σ_N → CNN, fading → naive energy). Which of these the paper leads with is open.
- **`energy_csi` is handed the true gain.** A decision-directed gain estimate from the frame's own symbols
  would place a real receiver between naive and CSI energy (S1's untested argument, §3.3k). One eval-only
  job on the existing generators.
- **Adaptive CFAR** (noise estimated per frame from the RRC band edges, Q13 above) would be a stronger
  energy detector under σ_N; not needed for the MF energy detector, whose threshold σ_N barely moves.

## 4.3 Ideas on the shelf — specified, not adopted

### Future experiments after the 2026-09-28 batch — catalogue for the next discussion (2026-09-29)

**2026-09-29: the user chose the final experiment (§3.4) instead of picking from this list.**
- **Absorbed into it:** A2 (the `noise` environment trains its generators against its CFAR-calibrated
  defender) · A3 (the matched 15 dB column, for every target) · A4 (seeds, 3 for the CNN-targeted GAN)
  · B4 (the CNN retrained under noise uncertainty) · C3 (the Amuru envelope down to p = 0.01) · C4
  (evaluations pinned to one GPU model) · the supervisor's constant-vector baseline (as Li's `tone`)
  · B3 in a new form (Rician fading instead of shadowing, crossed with noise uncertainty).
- **Parked:** A1 (D6 at 15 dB, §3.4).
- **Not in it:** B1, B2 (energy is bracketed by naive and true-gain versions instead), B5, B6/S3, C1,
  C2, C5.

Everything that could follow D6 / S2 / S4 / S5, in one list, for the user to choose from (the user also
has an idea of their own to bring). Measures for all of them: §2.7. Costs are GPU wall time with an idle
queue. Items specified elsewhere are only pointed to.

**A. At the defensible operating point, 15 dB (§4.2 Q13) — where the conclusions changed.**
- **A1. D6 at 15 dB with 15 dB-trained attackers and CFAR energy** — fully specified in §3.4 "Next
  experiment" (Track 1's session, user go-ahead pending). The lead candidate: it puts S4's generators,
  S5's noise uncertainty and D6's retrained CNNs at one operating point. ~1 h.
- **A2. S5 Tier 2: train generators against the honest-CFAR energy detector** (15 dB, σ_N 0.5 / 1 dB).
  S5 evaluated generators trained with known noise at 30 dB; a generator trained against the widened
  threshold measures how much of the opening S5 found is left on the table. Needs a `--noise-unc-db` flag
  in `train_gan.py` (`Link.noise_unc_db` + `calibrate_snr.cfar_defender` exist) and verify checks.
  4 generators × ~7 min + 1 evaluation (~25 min). Only an upper bound on the attacker unless A3 comes too.
- **A3. The full matched 15 dB column.** Jammer AND detector trained at 15 dB for the whole comparison
  (user, 2026-09-29: "makes sense to train both on the same"): the rest of run003's 20 tasks at 15 dB
  against `arms/snr15_r0` (S4 did 4). 16 × ~7 min + one evaluation.
- **A4. Seeds for S4.** S4 is one seed per generator; its side result (the untargeted / energy-targeted
  generators trained at 15 dB are far less visible to the CNN, 0.31 → 0.11, 0.27 → 0.09 at PER 0.1) is
  unreplicated. 3 seeds × 4 tasks × ~7 min + 3 evaluations.

**B. Defender realism — does finding 1's condition hold for a real receiver?**
- **B1. A noise-estimating energy detector** (S5's missing defender). Estimate each frame's noise floor
  from the out-of-band part of the spectrum (the simulated band is 8× the signal's) or from guard
  intervals, and normalise frame power by it. Decides whether S5's loss is real for a sensible receiver
  or an artefact of a detector that ignores information it has; the CNN's naive FAR doubling at 30 dB
  (§3.3n) shows the information is there. Evaluation only; new statistic + calibration + verify section.
- **B2. Estimated rather than true gain in `power_csi`** — see "Supervisor 2026-09-29 follow-ons",
  channel uncertainty (i), below. The same question as B1 for S1.
- **B3. Shadowing and noise uncertainty together** (S1 × S5), against the gain-aware + noise-estimating
  detector of B1/B2. Both flags exist; evaluation only once B1/B2 exist.
- **B4. A CNN retrained under noise uncertainty.** S5 kept the 30 dB weights; retrain on uncertain frames
  (`train_spectrogram_cnn.py` + `Link.noise_unc_db`, needs a flag there) and re-evaluate. Fits D6's
  retrain arms. ~10 min per CNN + evaluation.
- **B5. A real sequential detector as a defender.** §2.7's detection delay assumes the defender pools
  per-frame alarms optimally (Wald SPRT); build CUSUM on the energy statistic itself (Page's test, the
  jamming-detection literature's sequential tool) and measure detection delay at a fixed mean time between
  false alarms directly. Also answers "Time-to-first-detection" below. Evaluation only, CPU-cheap from
  stored statistics if per-frame statistics are kept.
- **B6. S3, the per-symbol detector (W = 1)** — specified and unbuilt (§3.4 batch, Track 2; verify §24,
  README §3.3p reserved).

**C. Measurement hygiene (read-outs, cheap).**
- **C1. Re-measure the older results in §2.7's measures:** E2 (§3.3g), E2b (on/off, §3.1 finding 3),
  §3.3f (run003 / run004 / run005). JSONs exist; read-out only (`probe_readout.py` functions), ~half a day.
- **C2. Store full statistic quantiles for every generator** (`snr_ablation.py --stats-for` all tags) so
  the threshold-free ξ is available everywhere, not only for CNN β 10. ~20–30 min per evaluation sweep.
- **C3. The Amuru envelope over p ∈ {0.05, 0.02, 0.01}** — in the supervisor follow-ons below; needed
  before any "beats Amuru" claim.
- **C4. Pin the GPU type of evaluations that are compared by seed** (CUDA random numbers differ between
  GPU models, §3.3n "Regression"): `--constraint=titan_rtx` (A1 already does) or state pairing as
  statistical.
- **C5. `verify.py` §15 rewrites E2's calibration caches** — E2 follow-on 4 below; a scratch folder fixes it.

**Closed — do not re-propose without new evidence.** The timing-error sweep for the listening jammer
(S2 failed, §3.3l); an S2 control with async-trained generators arriving in sync (moot after S2);
"damage per extra alarm" (withdrawn, §2.7); widening the training band to be loud (run003_wide, below).

### Earlier shelf items

- **Supervisor 2026-09-29 follow-ons (§B.1) — noted, not scheduled unless marked.**
  - **Pulsed baseline below 10 % duty — needed before any "beats Amuru" claim. → In the final
    experiment (§3.4), 2026-09-29.** `pulsed_qpsk(p)` ran only
    at p ∈ {1, .5, .25, .1} (§3.3d), while the CNN-targeted generator puts 97 % of its power in 5 % of the
    symbols (§3.3f). Evaluate p ∈ {0.05, 0.02, 0.01} (eval only) and draw Amuru as ONE line, the envelope
    over p. This is his "does the learner just learn a simple strategy?" question in its dangerous form.
  - **Random-direction / constant IQ vector attack (user: study it). → In the final experiment (§3.4),
    2026-09-29, as Li's `tone`, which is exactly this: a constant complex baseband vector with its phase
    uniform per frame, i.e. a uniformly random direction rather than one of four. The CNN is trained on
    it. A DC-share diagnostic runs on every generator.** The jammer picks one of four
    mutually perpendicular directions, independent of the symbol, with enough power to cross the boundary.
    Diagonal directions (|d| ≥ 1, JSR ≥ 0 dB): the push is aligned with s w.p. 1/4 (no error), opposite
    w.p. 1/4 (2 bit errors), perpendicular w.p. 1/2 (1 bit error), so **SER 0.75, BER 0.5** (the link
    carries nothing) and PER → 1 for any multi-symbol frame. Axis directions (|d| ≥ 1/√2, JSR ≥ −3 dB):
    BER 0.25, SER 0.5. Held constant over a frame, every symbol lands in the same quadrant. It is M0's
    `boundary_blind` family and a baseband tone; slot sync buys it nothing because it does not vary in
    time. Prediction: caught by energy at any power that does damage (+0.5 to +1 symbol energy per symbol
    against a 0.3-per-frame margin at 30 dB). Worth measuring below the flip threshold at 15 dB, where
    "pushing the Gaussian centre" gives graded damage.
  - **DC-offset removal as the defense against it.** Subtract the frame's sample mean before detection
    and decision. Direct-conversion receivers already do this against LO leakage, and scrambled QPSK is
    zero-mean, so the mean is the attack. One line in the link, eval only. Caveat: a real jammer has a
    carrier offset, so the vector rotates (a tone at Δf ≠ 0) and needs a notch rather than DC removal. Our
    link has a per-frame phase but no CFO. As a by-product it shows whether the learned jammer has any
    constant component.
  - **Channel uncertainty / time variance, beyond S1 (shadowing) and S5 (noise level).** Each behind a
    flag (default off = current numbers), defender re-calibrated per condition, run003 evaluated (Tier 1),
    retrained only if Tier 1 shows an opening. In order of value: (i) **estimated rather than true gain in
    `power_csi`** (decision-directed or pilot LS with estimation error), which tests whether S1's
    gain-aware floor survives realistic CSI; (ii) **Rayleigh/Rician block fading on the victim link + a
    pilot-based equaliser** (S1's unbuilt second step; clean BER > 0, so the excess-BER metric carries
    it); (iii) **Doppler within a frame** (time-varying channel, Sionna TDL with `max_speed`), which smears
    the spectrogram the CNN reads; (iv) **CFO / phase noise between jammer and victim**, which matters for
    the constant-vector attack above.
  - **Detection as localization (his reframing; to be discussed with the user).** The defender estimates
    the jammer's position from RSS, AoA (antenna array) or TDoA (several receivers). Candidate measures:
    localization error (mean / CDF), P(error < r) as the analogue of P(det) ("close enough to
    neutralize"), and the **Cramér–Rao bound** as the analogue of the NP reference. For TDoA,
    var(τ̂) ≥ 1/(8π² β_rms² E/N0), so localizability depends on the jammer's received energy and RMS
    bandwidth. Both are computable from existing waveforms without building a localizer. None of the
    current detectors localize, and coordination matters here (a team presents overlapping sources), which
    is why it goes with MARL for the thesis/TWC. Literature to read and verify first (named from memory):
    Liu, Xu, Chen, Liu, "Localizing jammers in wireless networks", PerCom 2009; Pelechrinis et al.,
    "Lightweight jammer localization in wireless networks", GLOBECOM 2009; the GNSS jammer-localization
    line (TDoA/AoA).
  - **Metric — SETTLED 2026-09-29 (§2.7: literature-grounded measures only; "damage per extra alarm"
    withdrawn).** Background kept for the Related Work: the literature's convention is
    detection probability at a fixed false-alarm rate (Neyman–Pearson / spectrum sensing: Kay 1998, Axell
    et al. 2012; covert communications: Bash et al. 2013). Jamming-detection papers report accuracy / FAR
    at one operating point (Li et al. 2022; Zhang & Krunz 2023). Adversarial RF attacks report classifier
    accuracy against perturbation power (Sadeghi & Larsson 2019), and Flowers et al. 2020 add BER. "Frames
    per extra alarm" has no precedent found. The proposed replacement is P(det) at α read at matched damage
    (BER and PER), with the energy floor stated as "P(det) ≈ α + PER".

- **"King GAN" (working title; user, 2026-09-28) — parked until the system model is clean.** One
  conditional generator trained against ALL detectors at once, conditioned on a vector of flags saying
  which detectors are active (the unused conditioning path of Zhou's cGAN), each active detector
  contributing its own `−β·soft P(det)` term; evaluated against each detector separately. It would
  also let the generator find its own power against the power threshold (the user's answer to "why
  JSR", 2026-09-28): power as a learned parameter, with detection as the only thing pushing it down.
  Lessons that apply: run005's learned power went loud because a saturating detection term cannot
  outweigh log E[damage] once past the transition (§3.3f run005, §3.4 D5).

- **E2 follow-ons — parked by the user 2026-09-24 ("keep in mind for future"; §3.3g).** In order of
  cost:
  1. **One rerun of `snr_ablation.py` (~30 min wall, same array)** storing three more scalars per sweep
     point: the frame-error count plus a per-frame error histogram (**PER** for uncoded 256-bit frames —
     FEC is out of scope per §2.9, the histogram leaves "PER under a t-error code" open without a
     rerun); the **effective decision-point SINR** E|s|²/E|z−s|² (how much of a structured jammer
     survives the matched filter); and **AUC per detector** from the stored clean CDF — a
     threshold-free detectability that removes α altogether. `regress_snr30.py` gates it as before.
  2. **The trade-off figures §2.7 asks for:** per level and detector, the whole BER-vs-P(det) curve
     over [0, 1] at fixed α, the envelope, FAR dotted, matched BER at P(det) 0.05 / 0.1 / 0.25 / 0.5;
     BER against SINR (Gaussian rows collapse onto the closed form; each structured jammer's offset is
     its damage per unit interference) and P(det) against JNR; SER as a column. Replaces the E2
     report's Fig. 6. Items 2's non-PER/AUC parts need no rerun.
  3. **Tier 2** — retrain D1 and the headline D2 generators (ideally the CNN too) at 10 / 15 / 20 dB
     to turn transfer into achievability; re-check `train_gan.JSR_BANDS` at each level first (§3.1 trap 1).
     **Done at 15 dB by S4** (§3.3m: tasks 0 / 2 / 13 / 14, one seed, CNN retrained at 15 dB); 10 / 20 dB
     not run. Item 1's PER is stored at every point since S1; its AUC is available as ξ where quantiles
     were kept (§2.7); SINR is not done.
  4. Housekeeping: `gan_figures.py`'s failing Amuru colour and its `BER_REF` 1e-3 vs §3.3f's 3e-4
     (§3.3f caveats); `regress_snr30.py`'s P(det) tolerance is 4σ of one
     estimate, not of a difference (should be √2 × 0.088 = 0.125, §3.3f run002). And `verify.py` §15
     rewrites E2's calibration caches (`artifacts/cgan/baselines/snr/{snr_0,snr_10,snr_30,noiseless}/
     thresholds.json`) at 1024/4096 clean frames, so they show as modified after every verify run; the
     sweep redoes them at 20k on next use (`calibrate_snr`'s `n_clean` check). **Revert, don't commit:**
     `git checkout -- artifacts/cgan/baselines/snr/`. A fix is to point §15 at a scratch folder.
- **Damage when loud — a power-conditioned generator (2026-09-26, offered, not chosen).** The cold-start
  CNN generator is pulsed and saturates near BER 3 % because training never sees JSR above −16 dB, and
  the best duty cycle grows with power (Amuru). Options offered to the user: (1) widen the training
  band to +10 dB (one line; one waveform then compromises); (2) condition G on its power budget via
  the inactive conditioning path (principled; assumes the jammer knows its received JSR, i.e. its
  channel gain); (3) report the stealthy and loud regimes as separate operating points. ~5 min per
  generator to train. **2026-09-27: (1) TESTED AND REJECTED** (`run003_wide`, cold, CNN β 1/10, band
  (−48, +10) dB, jobs 2274758/2274759, one seed): BER at +10 dB unchanged (0.019 vs 0.018) and stealth
  lost (CNN P(det) at matched BER 3e-4: 0.213 vs 0.105 for β = 10). Likely cause (untested): at weak
  JSR log E[BER] is in the thousands, so those steps dominate the gradient and the high-JSR samples
  barely register. **Decision (user asked for a recommendation): (3) for the paper** — once every
  detector flags every frame, stealth is moot and the best loud jammer is matched QPSK; (2) stays the
  principled fix, future work.
- **Detector-conditioned generator (user, 2026-09-26; decide after MARL).** Zhou's conditioning path
  (label embedding into G, auxiliary classifier out of D) is inert today: QPSK is the only class, so the
  label is constant and `losses.classification_loss` is exactly 0. Re-purpose the label as **which
  detector is the target** (power 1s/2s, kurtosis, CNN) and train **one** detector-aware generator for
  all four, instead of one per (target, β) as in D2 (§3.3f). It tests whether detector-aware training
  *generalizes* across detectors rather than overfitting one statistic. Evaluate it against each
  single-target D2 generator at matched BER. Held until it is clear whether the MARL direction is
  pursued. The methodology draft carries a `\rar{}` note on this
  (`paper_drafts/sec_methodology.tex`, §method:plain).
- **Time-to-first-detection.** The cheap countermeasure-facing result: measure the **trigger**
  instead of the reaction. Computable from the per-frame P(det) we already produce — no
  countermeasure, no mobility, no throughput model. It also fixes a known weakness of our own
  reporting (a threshold like P(det) ≤ 0.5 is not operational stealth), turning a hand-written caveat
  into a reported number. Reconsider if a defense-facing result is ever wanted. **2026-09-29:** its
  read-out now exists as §2.7 measure (4), detection delay under a sequential test on the alarms; a real
  sequential defender is item B5 of the catalogue at the top of this section.
- **ACK/NACK as the execution-time observation** (§2.8, CTDE). If any execution-time adaptivity is
  wanted, this is the channel to model — one line in the system model, and if cheap, an
  observation-space ablation (blind vs ACK-aware).
- ~~**Desync / realism axis.**~~ **PROMOTED OFF THE SHELF 2026-09-12 — it is now RQ1's swept
  variable** (§2.3), not an optional realism garnish. Recorded here only so the trail is visible:
  it sat on this shelf from the 2026-08-03 email, where he wrote *"introducing some
  desynchronization … will make the attacker more realistic and weaker, which is good for the paper,
  especially if BER is high and detection rate is low."* His 2026-09-12 note gives it a **cause**
  (inter-jammer communication delay) and therefore a reason to be the x-axis rather than a
  robustness check. Per-jammer CFO, timing and residual phase error are the implementation. It still
  also substantiates the counter-signal-vs-boundary robustness claim (§2.5) as a by-product.
- **Scenario-size ablation:** #jammers, **#legitimate users**. The latter is new — the model is
  1 TX → 1 RX today, so it needs a multi-user extension first (and it is the same extension the V>1
  coordination mechanism needs — see Q3).
- **Victim mobility.** His *"most interesting investigation"*. Stretch; only after M1 works.
- **Jammer localization as a second detector modality.** *"A form of detection is to leak information
  on the position of the jammer(s) so that a defender can physically neutralize them."* Out of scope
  for the thesis core; **name it in the Threat Model as an out-of-scope defender capability and
  future work.**
- **Real hardware.** He offered it: *"Real hardware to implement this method is available, if you'd
  like to experiment later on."* Future work; worth asking what hardware, in case a small validation
  is cheap.
- **A boundary attack that is optimal for where detection happens.** Detection happens at the RX on
  the composite signal *before* equalization, so what the detector sees is the perturbation at
  magnitude `|d·h_tx|`. Minimizing that favours subcarriers with **small `|h_tx|`** — a *different*
  criterion from m1's `|h_jam/h_tx|`, which maximized damage per unit *transmit* power. **So the
  matched-detectability refutation killed one specific criterion, not the idea that channel knowledge
  helps.** Parked because M0 has no subcarriers. **Now runnable (2026-09-16):** the OFDM stack is
  being revisited through `sim08_ablation/`, so this is a new `sparse_*` selection rule in
  `ablation.py`'s own `build_jams` — never in the frozen `frontier_channel.py`. §3.3c finding 3 also
  gives it a concrete reason: detection tracks total *received* jammer energy, and received energy on
  subcarrier n scales with `|h_jam[n]|` while damage scales with `|h_jam[n]/h_tx[n]|`. The ratio
  damage / detectability is therefore `1/|h_tx[n]|` — rank by small `|h_tx|`, not by m1's
  `|h_jam/h_tx|`. Still a genie (needs the victim's own channel). Two further items of his live in the
  same bucket: *"a well-crafted adversarial signal could also disrupt multiple subcarriers
  simultaneously"* (a joint multi-subcarrier attack) and the simpler per-subcarrier isolated problem he
  thought could be interesting on its own.
- **Time-domain jammer structure on the sim08 suite** (specified 2026-09-16, not run). §3.3c covered
  exactly one jammer family — held random-phase tones — so its finding 3 ("detection is set by total
  power, not spread") is only established *within* that family. The frozen CNN was trained on held
  in-band jammers plus classical time-domain ones, never on duty-cycled in-band jamming. Candidates, in
  order of expected value: **duty cycle** (active on a fraction of the 14 OFDM symbols, total energy
  matched — the classic energy-detector evasion); per-OFDM-symbol **hopping** (`sparse_hopping` exists
  in frozen `frontier/frontier_sweep.py` as a template); a **modulated** QPSK-like waveform per
  resource element. Same harness, same confirmation pass, reported at matched detectability.

## 4.4 Known limitations to keep visible

- **No results from a lossless channel make scientific claims.** sim06/07's lossless channel is a
  controlled simplification for isolating observation-model and training-algorithm effects.
- **Single-round, frozen detector** (§2.1). RQ2 partially buys this back but does not make it an
  arms race.
- **The m2 in-band training labels are not BER-thresholded** (in-band samples labelled "jammed" even
  when BER ≈ 0), which inflates FAR. A calibrated-threshold version would sharpen the exact numbers;
  the qualitative result is robust.
- **M0 has no pilots**, which is exactly the structure Q2 says the learner would need. Keeping pilots
  is the one argued exception to the simplification.
- **E1's σ grid is incomplete** vs the design (§3.2).
- **The novelty check behind the 2026-09-12 pivot was web search, not a systematic IEEE Xplore
  sweep** (§2.1). The positive findings (what prior art exists) are solid — those papers were
  located and identified. The negative findings (what does *not* exist, i.e. §2.1's four surviving
  claims) are **weak evidence and must not be quoted as "no prior work exists"** in print until the
  three key papers have been read in full.
- **The multi-jammer extension G6 depends on does not exist yet**, and sim04 — the only precedent —
  is frozen sim-stack code, not M0 code, and was **centralised-execution** (one optimizer over both
  agents' parameters), so it shows a coordinated solution is gradient-reachable, **not** that
  decentralised agents find it (§A.3). RQ1 must not overclaim past that.
- **Every model before `cgan/` placed the jammer on the victim's own symbol grid** (found
  2026-09-14, §2.10): M0 as one complex number per symbol, sim06–08 as one value per OFDM resource
  element. That is an unstated **perfect time-synchronisation** assumption. `cgan/`'s C2 shows it is
  not innocuous: a grid-aligned jammer cannot cause any bit error below −3 dB JSR at 30 dB SNR, while a
  time-offset one can. E1's numbers are correct *for a synchronised jammer* and must be quoted with that
  qualifier.
- **E2's detector is out of distribution off 30 dB** (§3.3g). The spectrogram CNN's weights are frozen
  at the level it was trained on and only its colour scale and threshold are re-calibrated per SNR, so
  every E2 row except 30 dB is *"a detector trained at 30 dB, deployed off-design"*. The scale's `vmin`
  moves −14.7 dB (SNR 0) → −44.2 (30) → −80.7 (noiseless), and at the **noiseless anchor the CNN is
  effectively blind** (edge +15 dB) — an OOD artifact that must not be read as a jammer property.
  Retraining per level is the fix (Tier 2, §3.4). The analytic detectors carry no such caveat.
- **E2 Tier 1 measures transfer, not achievability** (§3.3g): the generators were trained at 30 dB and
  evaluated elsewhere, so the run002 numbers at 15 dB (CNN 0.09 at matched BER) are what a 30 dB-trained attacker achieves off-design. Per-SNR
  retraining could only raise it, but that is an argument, not a measurement.
- **D2a's learned family is Gaussian only** (§3.3e):
  - it cannot hold kurtosis at the clean value, so "kurtosis catches every effective learned jammer"
    is a statement about this family, not about all jammers. A constant-envelope (tone/noise)
    family was offered on 2026-09-18 and not chosen;
  - the training grid is 4 dB in JSR, at α = 0.05 only, with 256 frames per fitness evaluation;
  - no grid point below −40 dB, so "no stealthy point against power" means ≥ −40 dB.
- **The realistic range of inter-jammer delay is unknown to us** (§4.1 #8). RQ1's x-axis is
  therefore currently in arbitrary units — symbol periods, or radians of residual phase error. The
  decay *curve* is meaningful without it, but any sentence of the form "at realistic delays the gain
  is X" needs a number we do not have. Ask him, or report the axis normalised and say so.

---
---

# APPENDIX A — Experiment history

The falsification record. The paper's Experiment History (§3.5) is a ~1-column summary condensed from
it; this appendix is the long form, kept for the thesis. Findings and mechanisms only — the per-run debugging chronology has been compressed
to one line per class of bug.

## A.0 Run index

Convention for `artifacts/simXX/`: `runNNN.png` (training curves) · `runNNN_iq.png` (IQ scatter) ·
`runNNN_model.zip`/`.pt` (saved model) · `runNNN_slurm_<jobid>.out/.err` · `tb/runNNN/`
(TensorBoard; view with `tensorboard --logdir artifacts/simXX/tb --port 6006` + VSCode port
forwarding — the forwarded port appears in the Ports tab, no manual `ssh -L` needed).

| Sim | Runs | Method | Headline | Job(s) |
|---|---|---|---|---|
| 00 | — | none | BER 0→0.5, power 1.0→51, 0→2 users detect when jammer turns on | — |
| 01 | 001–007 | PPO, Gaussian | best at N=16, β=0.5: BER 0.19, det 8–10%, power 2.2. N=512 fails (action space too large) | — |
| 02 | 001–002 | PPO, Gaussian | det flat 100%, kurtosis stuck ~−0.25 at every setting | — |
| 03 | 001–004 | PPO, NSF | det flat 85–100%, kurtosis ~−0.25. N=16 too noisy; N=128 fixed the estimator, not the outcome | 93396/93407/93423 |
| 03b | 001–008 | **direct gradient, NSF** | **best pre-sim04 result: kurt −1.30, BER 0.17, det ~0%** | 98432/98441/98470/98478/98483/98494/98509/98516 (run008 = 98516) |
| 03c | 001–009 | PPO, GMM | closed. run009 (1M steps): BER 0.05 *declining*, det ~80%, kurt pinned at −1.0 | 98764–98777 |
| 04 | 001–007 | direct gradient, 2 agents | run001 BER 0.24 @ total power 1.25; run007 BER 0.65 @ total power 4.0 | 99211/99245/99825/99835/100037/100040/100041 |
| 04b | — | Sionna on GPU | validation only | — |
| 05 | — | EfficientNet-B0 on flat QPSK | **78.9%** — detector fails without OFDM | — |
| 06 | det 002; jam 001–003 | CNN + MAPPO | detector **99.79%** ✓; jammer fails, P(jam)≈0.999 flat | — |
| 06b | 001 | MAPPO in 2D | stealth solved (P(jam)≈0.003) but per-SC BER plateaus 0.35; no input correlation | — |
| 07 | 001–005 | blind causal MAPPO | dead end; entropy bit-for-bit flat (~178.4) across all runs | 101622/101632/101650/101657/101817 |
| frontier | Phase 0 | inference sweep | **detector is an out-of-band-emission detector** | 101860 |
| frontier | Phase 0.5 | retrain + re-sweep | blind spot closes; acc 99.8→90.5%, FAR 0→3.8% | 101866 |
| frontier | recheck | complex-STFT + energy | out-of-band survives; **energy detector kills lossless stealth** (max stealthy BER 0.42→0.005) | 102115 |
| 08 | m1 | TDL fading sweep | SNR-independent BER floor; channel-aware "+70%" (later refuted) | 101870 |
| 08 | m1b | + energy detector | **stealth revived on the realistic channel**, BER 0.20–0.24 at P_energy ≤ 0.5 | 102305 |
| 08 | m2-det | channel-valid CNN | acc 94.3%, DR 91.2%, FAR 2.3%, F1 0.94 | 102316 |
| 08 | m2-suite | CNN ∨ energy per-sample | **suite ≡ CNN**; residual stealthy region BER 0.065–0.11 | 102319 |
| 08 | dense | 9 powers × 11 n_active, B=512 | **refutes channel-aware > blind at matched detectability** | 102390 |
| **M0** | E1 | 8-σ array + NP-optimal | **no realizable stealthy attack** (§3.3) | 2243867, 2243879 |
| **cgan_learned** | run001 (D2a) | shaped noise (48 params), CMA-ES, black-box score-based, per detector | **stealthy BER 0 against every detector**; CNN evaded to −20 dB only out of band (BER 0); shaping +15.9 dB effectiveness; power not waveform-blind (−0.06 near FAR, −0.19 at −20 dB); §3.3e | 2267049, 2267050, 2267051, 2267090 |
| **cgan** | run001 | CGAN (Zhou 2025), NS-GAN+GP, weights 1 | **ordering not reproduced**: with one sync model GAN−Optimal −1.30 dB async / −4.47 locked (paper +1.31); Noise−GAN 6.66 / 4.38 dB. The first-reported 4.34 / +1.01 scored GAN locked vs Optimal async (withdrawn); §3.3b | 2259289, 2259409, 2260623 |
| **cgan** | run002 | CGAN, reference-backed recipe (WGAN-GP, feat 2 / STFT 45) | worse imitation: EVM 0.70, no 1e-3 crossing in grid (BER 3.6e-3 at −10 dB), plateau 0.31; critic gap stalls ≈ 57; §3.3b | 2260624, 2260806 |
| **cgan_snr** | run001 | **E2**: SNR 0:5:40 dB + noiseless anchor × 7 classical attacks × all 21 generators, detectors re-calibrated per level (CNN weights frozen at 30 dB) | **matched-BER gain vs the CNN peaks at −85.1 pp at 15 dB SNR, 5× the −16.2 pp at 30 dB, gone by 40 dB**; stealth edge falls ~0.7 dB per dB while the BER onset is flat, so the gap grows with SNR; 30 dB reproduces §3.3f; power saving SNR-invariant at 12.3–12.9 dB; §3.3g | 2270447, 2270448, 2270449, 2270556, 2271314 |
| **cgan_gan** | run002 | **D2 re-run at 4000 steps** (all 20 tasks, warm start, run001 recipe otherwise) + D1 re-evaluated | **β = 0 (BER only) does most of it**: BER 3e-4 at −22.1 dB (D1 −8.2), CNN 0.579; CNN term adds −17 pp (β = 1: 0.413); power/kurtosis targets no better than β = 0; §3.3f | 2273984 (verify), 2273985, 2273986, 2273987 |
| **cgan_gan** | cold001 · cold002_4k · warm002_4k · {cold,warm}002_4k_r1–r3 | **imitation-start / training-budget ablation**, tasks 0/9/14 (seeds: 0/14), random vs warm init, 400 vs 4000 steps; on net_scratch via symlinks | **400 steps under-trains; the imitation start trades stealth for damage capacity** — cold CNN β = 10: 0.146 ± 0.024 (4 seeds) vs warm 0.480 ± 0.074, pulsed (97 % power in 5 % of symbols), BER saturates ~3 %; §3.3f | 2273958–2273960, 2273967–2273970, 2273989–2274000 |
| **cgan_snr** | run002 | **E2 on the run002 generators**, same protocol as run001 | **D1 → β = 0: −85 pp vs the CNN at 15 dB, 13.9 dB less power at every SNR; the detector term (CNN β = 1) adds −8 to −20 pp at 25–35 dB only**; 30 dB reproduces run002's eval (two-sample z, none > 4); §3.3g | 2273988, figures 2274104 |
| **cgan_gan** | run003 · run003_wide | **THE PAPER METHOD: random init, 4000 steps, all 20 tasks** (+ wide-band arm: CNN β 1/10, band (−48, +10) dB) | **CNN β = 10: CNN 0.113 at matched BER 3e-4 / 30 dB vs control 0.999; best generator on every detector**; wide band: no loud-regime gain, stealth lost (0.213); §3.3f, §4.3 | 2274753–2274759, 2274789–2274792, 2274822 |
| **cgan_snr** | run003 | **E2 on run003** | **CNN β = 10 ≤ 0.11 CNN P(det) at matched BER over 0–30 dB; control 1.00 at 30 dB; beats Amuru on both detectors ≥ 10 dB**; §3.3g | 2274792, 2274823, figures 2274829/2274828 |
| **cgan_outlier** | run003 · run003_per | **E2b**: outlier alarm α′ ∈ {1e-3, 1e-4} (200k clean frames/level) + frame error rate, 15/30 dB, noise/pulsed/β=0/CNN β 1,10 | **average BER: on/off switching wins or ties; PER: the CNN-targeted generator beats on/off vs the CNN at every α (PER ≥ 0.1); vs power it wins only at 15 dB outlier alarms**; §3.3g E2b | 2274861, 2274862, 2274869, figure via `submit_outlier_figures.sh` |
| **cgan_gan** | run004_grey | **grey-box**: surrogate CNN (same recipe, seed 12) → random-init CNN β 1/10 trained against it → eval vs the deployed CNN, 30/15 dB | **no transfer gap** (30 dB β = 10: 0.107 grey vs 0.113 white); §3.1 | 2274864–2274868 |
| **cgan_gan** | run005_learned | **learned power (cap 0 dB, start −40 dB) + frame damage**, random init, tasks 0 / 1–4 / 13–16, eval at own power | **goes loud and is caught**; CNN β 100 = CNN blind spot at 30 dB (PER 0.35, CNN 0.000, power 1.00); energy floor ≈ 1 frame per alarm; sigmoid detection term saturates → D5; §3.3f | 2274914–2274916, 2274928, 2274933 |
| **cgan_shadow** | shadow_run003 | **S1 Tier 1**: per-frame log-normal shadowing on the victim's link, σ ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3} dB at 30 dB; CNN retrained per σ + `power_csi`; run003's 21 generators + classical envelope, evaluated not retrained | **naive energy blind by 0.1–1 dB** (CNN-targeted GAN 4.5 → 35 frames per extra alarm), **gain-aware energy 1.04–1.06 at every σ**, retrained CNN still the weak link (5–8); σ = 0 reproduces run003; §3.3k | 2274945 verify · 2274946–2274951 CNN · 2274952 |
| **cgan_arms** | round1 | **D6 round 1**: the CNN retrains, the attackers stay frozen — defenders r0 (deployed; 15 dB: retrained at 15 dB) / A (Li's range widened to −35 dB) / B (A + run003 CNN β 1/10 frames), 30 and 15 dB; 9 generators (seen, same-init twins, 3 held-out seeds) + 3 classical, paired frames | **30 dB: one retrain without the attacker (A) lifts the CNN on held-out generators 0.17 → 0.58–0.74 at matched BER (energy 0.60–0.81), round-0 attackers stay below energy; no cost. 15 dB: nothing moves (≤ 0.09) and retraining costs 4–5 pp on Li's classes**; §3.3o | 2275914 verify · 2275915/2275949 CNN · 2275950/2275958 verify · 2275916, 2275960, 2276002, 2275959, 2276037 eval |
| **cgan_sync** | sync003{,_r1–r3} · async003_r1–r3 · evals sync_run003{,_r1–r3}, async_run003_r1–r3, async_b1_run003_r1–r3 | **S2 listening jammer**: every jammer on R's symbol grid, phase still random; run003 recipe retrained synchronously (tasks 0/2/13/14; seeds r1–r3 for 13/14) + async β 1 seeds as the control; 30 dB, deployed detectors | **fails its success test**: power and CNN P(det) at matched PER inside the seed spread (CNN β 1 at PER 0.1: 0.135 ± 0.080 sync vs 0.128 ± 0.022 async; ξ unchanged); the first seed's "win" did not replicate; sync costs matched QPSK 3.6 dB; §3.3l | 2275790 verify · 2275791/2275792 · 2275891–2275899 · 2275934–2275939 |
| **cgan_snr15** | snr15_003 · evals snr15_run003, snr15_run003_base | **S4 (E2 Tier 2)**: run003 recipe trained at 15 dB against the CNN retrained there (Track 1's `arms/snr15_r0`), bands re-derived (power (−24, 0), CNN (−39, −7)); tasks 0/2/13/14, one seed | **energy flags the best generators on ≈ 25 % of frames at PER 0.1 at 15 dB (≈ 63 % at 30 dB), trained at either SNR**; training at 15 dB lifts β 0 from 0.61 to 0.31; retrained CNN still the weak link (7–8 %); §3.3m | 2275923 verify · 2275924 · 2275925 · 2275926 · 2275974 · 2275975 |
| **cgan_noise_unc** | noise_unc_run003 (+ noise_unc_regress) | **S5**: per-frame log-normal noise-variance factor σ_N ∈ {0, 0.5, 1, 2} dB × {30, 15} dB; naive vs honest-CFAR defender on the same frames; run003 + classical, evaluated | **30 dB: nothing for energy; 15 dB: naive FAR → 0.23 (two-sided 0.42), CFAR lets learned jammers under** (CNN β 10 flagged on 0.25 → 0.14 → 0.09 of frames at PER 0.1, 2 → 7 → 57 frames broken before a sequential test decides, σ_N 0 / 1 / 2 dB); CUDA RNG differs across GPU models; §3.3n | 2275921 verify · 2275922 · 2276029 · 2275979 |
| **sim08_abl** | run001 | noise × power × #jammers on the frozen sim08 suite | **stealthy BER 0.005–0.016 at every Eb/N0 ≥ 15 dB** (×1.4→×109 the floor); detection set by *total* power; more jammers = louder, no matched-detectability gain; suite ≡ CNN (26/4914); §3.3c | 2261123, 2261126, 2261146, 2261173, 2261174 |
| **cgan_team** | run001 | **E3 pre-check**: K = 1/2/4 jammers, jammer→R fading (lossless/rician10/rayleigh) × SNR 15/30 dB, aligned vs random timing, 5 jammers; transfer (no retrain) | **fading helps a single jammer (frame-level pulsing artifact), so no gap for a team; the only lever is TIMING** — aligned K = 4 ≈ one jammer at 7.3 dB less per drone, random timing loses ~6 dB / 19–35 pp; lossless K = 1 reproduces E2; §3.3h | 2273604, 2273605, 2273606 |
| **cgan_team_timing** | run001 | **D4a delay-decay curve**: lossless, K = 1/2/4, SNR 15/30 dB, three generators (spec_cnn_b10, plain_run001, kurtosis_b1); followers' timing error σ ∈ {0 … 64} symbols + uncoordinated; transfer (no retrain) | **the team stops matching one jammer at σ ≈ 0.5–1 symbol**; half the gain is gone by σ ≈ 2 and it reaches the floor by 8–16; an uncompensated team in the test drops sits at 1.1 symbols rms, so δ matters; σ = 0 and shifted reproduce E3 within ±0.06 / ±0.7 dB; §3.3i | 2273650 (verify §16–17), 2273651 |
| **cgan_team_policy** | run001 | **D4b learned policy**: shared MLP → (u_k, δ_k) per drone, direct gradient through `frac_delay`, K = 4, SNR 15/30 dB × β 0/10; four training variants, only the last on disk (bounded δ, effective batch 256, 600 updates); pooled over the 50 test drops | **learns u → 1, not δ = τ**: loses to the δ = τ heuristic at σ = 0 (30 dB β 10: 0.794 vs 0.705 CNN P(det), −1.2 dB) and matches it for σ ≥ 1; a one-drone δ = τ minimum exists but is ~0.6 loss units deep; conditional on the under-trained `task14_G`; **paused**; §3.3j | 2273734 (verify §16–18), 2273735/36 (smoke), 2273737/38, 2273906/07, 2273924/25, 2273922/23 (diag), **2273939/43** |
| **final** | verify · regress | **`final/` test suite + regression gate vs S4**: closed forms, energy after the MF, Rician, σ_N, CFAR in all six envs, bands at base, baselines, one training step per target; S4's 4 generators + 7 classical vs `arms/snr15_r0` | **97/97 (100/100 with the random push, job 2278412); gate 0/2178 P(det) pairs beyond the √2-corrected 4σ**; corrects two §3.4 premises (energy vs power white-noise transition 10.3 dB apart; noise share ≈ 1/10); §3.3q | 2277992 |
| **final** | `_smoke/both` | end-to-end smoke of the chain (2 epochs, 20 steps, 4 JSR points) | pipeline runs; outputs deleted after the check | 2277993–96, 2278004/05 |
| **final** | base · noise · noise_1db · fading · fading_k28 · both | **the final experiment**: per env CNN retrain + CFAR thresholds → bands → 8 generators (control ×3, cnn_b10 ×3, energy_b10, kurtosis_b10; run003 recipe) → 21 attackers × JSR −50…+15 × 512 frames, 15 dB (Es/N0 24 dB), TITAN RTX; + the random push (22nd attacker, 2278413–18) | **CNN-targeted GAN vs the CNN at PER 0.1: within ±4 pp of the classical envelope except noise_1db (−4.6 ± 0.9); no gain over the damage-only control; −86…−93 pp over Zhou's CGAN; σ_N blinds the CNN, fading blinds naive energy (CSI energy unmoved)**; §3.3q | chains 2278006–2278029 (evaluates 2278009, 2278063, 2278064, 2278069, 2278076, 2278084); TaskProlog reruns 2278057/59/68/75/83 |

**The attacker's objective at each step** — re-read from the code 2026-09-10, because the Overleaf
appendix states it nowhere and two findings below are properties of the objective, not of the
attacker. RL form `R = BER − β·D − γ·P − 0.05`; direct-gradient form `L = L_BER + λ·D + γ·P`, where
`L_BER` is BCE between the demapper LLRs and the *complement* of the tx bits.

| Sim | Detector | Detection term `D` | β/λ | γ | Optimiser | Source |
|---|---|---|---|---|---|---|
| 01 | power threshold | `1[flagged]` | 3.0 | — | PPO | `jammer_env.py:10,72` |
| 02 | kurtosis | `1[flagged]` | 0.5 | — | PPO | `jammer_env.py:10,74` |
| 03 | kurtosis | `max(0, kurt−τ)` | 2.0 | 0.02 | PPO | `jammer_env.py:9,10,80` |
| 03c | kurtosis | `max(0, kurt−τ)` | 2.0 | 0.02 | PPO (GMM) | `jammer_env.py:9,10,80` |
| 03b | kurtosis | `relu(kurt−τ)` | 2.0 | 0.02 | direct gradient | `train.py:31,32,234,249` |
| 04 | kurtosis | `relu(kurt−τ)` | 2.0 | 0.02 (on **Σ** powers) | direct gradient, joint | `train.py:31,32,288,290` |
| 06 jam | CNN | `−log(1−p̂+ε)` | 0.3 (linear warmup) | 0.1 | MAPPO | `train_jammer.py:50,51,321` |
| 07 | CNN | `−log(1−p̂+ε)` | 0.3 (linear warmup) | 0.05 | MAPPO | `simulation07:68,69,437` |
| **M0** | NP / CNN / energy | flagged frames, β swept | — | **hard budget, not a term** | — | `attacks.py:23` |
| **cgan D2a** | power 1/2-sided, kurtosis, CNN (one each) | hard flag rate (+1e-6 score tie-break) | 0 / 0.01 / 1 | **per-frame equality projection, not a term** | CMA-ES, black-box | `cgan/train_shaped.py` `fitness` |
| **cgan D2** | power 1/2-sided, kurtosis, CNN (one each) | `soft_pdet` = mean sigmoid((stat−thr)/σ_clean) at the α = 0.05 threshold | 0 · {1,10,100,1000} · CNN also {1e4,1e5,1e6} | **per-frame equality projection, not a term** | Adam, white-box direct gradient | `cgan/train_gan.py` `train` |

**Two findings only visible in this table**, each recorded where it is used: the **log barrier is the
mechanism of the untrainability result** ([A.5](#a5-sim06-jammer--06b--07--the-untrainability-result)),
and **γ = 0.02 never binds**, which is why run007's power blew up
([A.3](#a3-sim04--sim04b--a-coordinated-solution-exists-and-is-gradient-reachable)). The M0 row is the
correction: the supervisor's mandate (§2.8) is precisely a verdict on this table.

## A.1 sim00–01 — the measurement chain, and the limits of a power threshold

**sim00** (`simulation00/baseline_lossless.py`) is purely observational: 2 legitimate QPSK TX→RX
pairs, 1 jammer silent for t = 0–4 then max-power Gaussian for t = 5–9, lossless `rx = tx + jam`,
detection by `mean|rx|² > 3.0`. Result: BER 0 → ≈0.5, power ≈1.0 → ≈51.0, 0 → 2 users detecting,
exactly at the switch. **Establishes that the measurement chain is trustworthy** — BER, power and
detection all move when and only when they should.

**sim01** adds a PPO jammer against the same power threshold. Best configuration (N=16, β=0.5): BER
converged to ~0.19, detection fell 45% → 10%, jammer power settled at ~2.2 — just under the
threshold. That is near-optimal: the TX contributes ~1.0 to received power, threshold 3.0 leaves a
jammer budget ≈2.0, and the theoretical max BER at that power is ≈ Q(√0.5) ≈ 0.24. **Establishes
that against a power threshold, power tuning is the only strategy available** — the IQ scatter stayed
a featureless Gaussian cloud throughout. N=512 (runs 005–007) failed outright: the action space is
too large to learn, and at β=3 the agent abandoned stealth entirely.

## A.2 sim02–03c — what a policy distribution can and cannot represent

**sim02** replaces the power threshold with a **kurtosis detector** (QPSK has excess kurtosis ≈ −2,
Gaussian noise 0; flag if `kurt(rx) > −1.0`). Detection stayed flat at 100% regardless of training and
kurtosis stuck around −0.25. **A diagonal-Gaussian policy can only produce Gaussian-shaped IQ clouds
— it is *structurally incapable* of QPSK-like sub-Gaussian statistics, no matter how training
proceeds.** This is the first structural (rather than tuning) negative result on the ladder, and it
motivates every policy-distribution upgrade that follows.

**sim03** swaps SB3's diagonal-Gaussian action head for a **Neural Spline Flow** (zuko), conditioned
on the PPO MLP latent, giving exact `log_prob` via change-of-variables plus a Monte-Carlo entropy
estimate — a drop-in for everything PPO needs. Outcome unchanged: detection flat 85–100%, kurtosis
stuck near −0.25. Two fixes were tried and neither moved it: replacing the **binary** detection
penalty with a **continuous** kurtosis-excess penalty (`β·max(0, kurt − thresh)`), so the agent gets a
gradient proportional to how far above threshold it is rather than a flat step; and raising N from 16
to 128, since N=16 was too noisy for the kurtosis estimate to give a usable signal at all. N=128 fixed
the estimator noise but not the outcome. *Performance note:* `zuko.flows.NSF` defaults to
fully-autoregressive (`passes=None`), needing `action_dim` sequential hypernetwork calls per
transform (~20 s/step on CPU). Setting `passes=2` (coupling-style, RealNVP-like) gave a **~44×
speedup** while remaining an exact-likelihood flow. **Establishes that the flow is expressive enough;
PPO is what fails to move it.**

**sim03b** removes RL entirely: the same NSF generates jam symbols and is trained by **direct
backprop** through a fully differentiable loss
(`soft_BER + λ·relu(kurt(rx) − thresh) + γ·power`). Fair-comparison constraint: kurtosis is **not**
in the observation, so the generative model's only advantage over PPO is that its loss is
differentiable, not extra information. **This is the best result of the whole pre-sim04 ladder:
kurtosis ~−1.30, BER ~0.17, detection ~0%** — a non-trivial local optimum found where PPO found
nothing.

Three bug classes were fought and fixed along the way, each worth one line: (i) `demapper(no=1e-10)`
saturates the LLRs so `binary_cross_entropy_with_logits` has ~zero gradient — the system collapsed to
"do nothing"; fixed with `no=1.0` and a tighter clamp. (ii) NSF rational-quadratic splines can
extrapolate to huge values, overflowing the kurtosis `m4/m2²` ratio to NaN, which then permanently
poisons the weights; fixed with `nan_to_num` **before** clamping (`clamp(nan)` is still `nan`), grad
clipping, and skipping non-finite steps. (iii) Outputs are overwritten in place at every checkpoint —
**there is no per-checkpoint history on disk.**

*The theoretical optimum, derived and never reached:* `jam = −2·tx` gives `rx = −tx`, i.e. BER 1.0
and kurtosis exactly −2 (statistically indistinguishable from clean), for jam power 4. Runs sat far
from it — a large basin-of-attraction gap, because both the kurtosis-relu and the power penalty push
toward small power early. *Caveat on that optimum:* a deterministic full inversion is
informationally equivalent to BER 0 for an adversary who knows the pattern, so "BER = 1" is a
property of this loss formulation, not necessarily a win against an adaptive receiver.

**sim03c** tried a per-symbol **GMM** (K=8) action head with PPO — a single-feedforward alternative
to NSF. **Closed, negative.** Nine runs systematically ruled out every PPO-mechanics knob (std clamp,
target_kl, LR, entropy coefficient, removing target_kl entirely). run004 (bias-initializing `log_std`
so the jammer starts near-silent) produced a large one-time jump, but runs 005–008 showed it to be a
**dead local optimum**: run008 (`target_kl=None`) produced 10× more gradient updates with
`approx_kl` ~1.5 and `clip_fraction` ~0.87 — massive raw parameter movement — yet every macro
statistic stayed **bit-for-bit identical** to run004. run009 (1M steps, 5–20× longer than anything
else) confirmed it: BER ≈ 0.05 and *declining*, all K=8 components collapsed to a single isotropic
Gaussian.

**Diagnosis: GMM permutation symmetry.** With K components per symbol, gradient steps can
substantially relabel/reshuffle individual mixture components without changing the *marginal
distribution* actually sampled — the optimizer's movement budget is absorbed by the symmetry instead
of reshaping the output. Combined with `MixtureSameFamily`'s non-reparameterized (score-function)
gradients being high-variance for overlapping components, PPO+GMM cannot make directed progress.
**Why NSF + direct gradient did better:** a flow is a *bijective* transform (no permutation symmetry)
and direct-gradient training uses *reparameterized* sampling — low-variance pathwise gradients from
loss straight to distribution parameters. Neither property holds for GMM+PPO.

**Architecture-independent side-finding: the reward "cliff".** From
`reward = ber − β·max(0, kurt − thresh) − 0.05 − γ·power`: `jam = 0` gives kurt = −2 → no penalty →
reward −0.05. Default init (std ≈ 1, power ≈ 2) starts at kurt ≈ 0 — **already past the penalty
cliff**, i.e. worse than doing nothing. As power rises from 0, reward *increases* until kurt crosses
the threshold, then falls off sharply. The true optimum sits **at the cliff edge**. Worth carrying
into any reward design, regardless of architecture.

## A.3 sim04 / sim04b — a coordinated solution exists and is gradient-reachable

**sim04** extends sim03b to two jammers sharing one lossless channel
(`rx = tx + jam₁ + jam₂`), trained jointly from a single shared differentiable loss — **centralized**
direct gradient, not MARL. Each agent has its own NSF encoder+flow; one optimizer backpropagates
through both, so each agent's gradient already accounts for the other's contribution.

*Why two agents:* the single-jammer optimum is `jam = −2·tx` (power 4); with two agents the
equivalent is `jam₁ = jam₂ = −tx` (power 1 each) — the same `rx = −tx` at half the per-agent power,
easier for the optimizer to find and avoiding the instability region that plagued sim03b.

**Two valid operating points, not broken-vs-fixed:**
- **run001** (job 99211, killed by the time limit at step 12 400): **BER ≈ 0.24 at total power ≈ 1.25**
  (p1 ≈ 0.60, p2 ≈ 0.65), detection 2–8%, kurtosis ≈ −1.25.
  **Corrected 2026-09-10 from the job log** — an earlier version of this line read "BER ≈ 0.35 at
  total power ≈ 0.9", which pairs a step-50 *untrained* transient (BER 0.357 at det = 1.000) with a
  step-8000 power reading. **The "2× sim03b's best" claim does not survive.** At *matched total
  power* ≈ 0.97 run001 sits at BER ≈ 0.195, det ≈ 3%, kurt ≈ −1.24 (step ≈ 8900), against sim03b
  run008's single-agent BER 0.170 at power 0.965, det 3.1%, kurt −1.31 (job 98516) — a ~15% relative
  gain, not a doubling. This is the same matched-vs-unmatched comparison error that later cost the
  "+70% channel-aware" headline (§2.7).
- **run007** (truncated at 33k/100k steps, BER still rising): **BER ≈ 0.65 at total power ≈ 4.0**,
  detection ≈ 0.00, received kurtosis ≈ −1.65.

**Read run007 honestly: the gain was bought with POWER, not strategy.** 3.2× run001's total power for
2.7× the BER. Total power ≈ 4.0 is exactly the power of the omniscient counter-signal `jam = −2·tx`, and
BER climbing past 0.5 toward 1.0 is the signature of *inverting* the constellation, not merely
disturbing it. The two-agent design argument predicts the same received signal at total power **2** —
run007 used double that, i.e. **it did not find the efficient split.** Cause: `GAMMA = 0.02` on the
power term is negligible against BER gains, so nothing constrained power growth — the objective
rewarded a jammer for doubling its power to gain a hundredth of BER, so **this is an
unconstrained-power result** and must be labelled as one (constants in [A.0](#a0-run-index)). **This is a concrete
instance of the proxy-reward problem he told us to delete, and a direct argument for the hard power
budget** — a self-diagnosed flaw turned into a finding.

**The clearest evidence of actual coordination is the equal power split (2.0 / 2.0)**, visible in
`run007.png` panel 4, *not* in the IQ scatter.

**Write the structure claim about `rx`, not the agents.** An earlier version of this note said "both
agents independently converged to a 4-cluster QPSK-like IQ structure"; `run001_iq.png` does not
support that. The **individual** jammer distributions are concentrated at the *origin* with four-fold
symmetry; it is the **received** signal that is four-clustered (and hence indistinguishable from
clean QPSK to the kurtosis detector).

**Why sim04 "worked" and the GAN does not (user question 2026-09-27, figure
`artifacts/cgan/iq/fig_iq_sim04_vs_gan.png`, `cgan/iq_figures.py`, 30 dB, deployed defender).** It is
information, not method: the genie flip (our `omniscient`, η = 1 — sim04's mechanism) leaves the
received constellation identical to clean (BER 0.25, every detector at its FAR 0.05–0.06), while a
symbol-blind matched-QPSK jammer at sim04's power (JSR +1 dB) does the same BER 0.23 and is caught
100 % by all four detectors. The run003 CNN-targeted GAN is a burst jammer: almost every symbol stays on
its QPSK point and 1–3 % are blown off-axis, so BER caps near 2 % (4.0e-4 at −23 dB, 1.0e-2 at 0 dB,
1.8e-2 at +10 dB; at −23 dB power 0.57 / kurtosis 0.10 / CNN 0.15). sim04 also measured detection
differently: kurtosis only, hand-set threshold −1.0 (not FAR-calibrated), no energy detector, and a
soft power penalty that let it run at JSR ≈ +1 dB.

**Two caveats that must travel with this result:** (i) kurtosis detector, lossless channel, genie
observation ⇒ this is a **mechanism** result, not a stealth result. (ii) **Centralised execution** —
one optimizer over the union of both agents' parameters means coordination was *maximal*, handed over
by the optimizer. What is absent is decentralisation. So it shows a coordinated solution **exists and
is gradient-reachable**, *not* that decentralised agents could find it. That matters, because
"cooperative multi-agent" is in the thesis title.

*Historical note on the run005/006 LLR sign bug:* it was **introduced** in run005 when Sionna was
removed from the training loop (the handcrafted demapper used the opposite LLR convention, so the
optimizer rewarded *correct* decoding). Runs 001–004 used Sionna's demapper and were **correct** —
just slow and truncated. The performance work that motivated removing Sionna (large batch, pure
PyTorch ops, `torch.compile`) took throughput from ~248 to ~39k samples/s.

**sim04b** repeats sim04 with Sionna's `BinarySource`/`Mapper`/`Demapper` on GPU
(`sn.config.device = "cuda:0"`, set before module creation) to confirm Sionna-on-GPU is viable for
the OFDM work ahead. Validation only.

## A.4 sim05–sim06 (detector side) — spectrograms need OFDM

**sim05** tried to train the Li et al. spectrogram CNN (EfficientNet-B0) on the **flat QPSK** channel.
**It failed: 78.9% validation accuracy** (vs the paper's 99.79%), with massive overfitting (train
99.7%, val stalled ~75%).

Cross-evaluating jammers against it is what carries the diagnosis (and is the table the Overleaf
appendix A.5 should use instead of a confusion matrix):

| Jammer | Detection rate | Verdict |
|---|---|---|
| Clean | 2.5% | FAR — low, good |
| Barrage | 2.5% | undetected — same as clean |
| Single-tone | 100.0% | detected (spectral spike) |
| Successive-pulse | 97.0% | detected (periodic pattern) |
| Protocol-aware | 1.0% | undetected |
| MARL (sim04) | 1.0% | undetected |

**Root cause: flat QPSK has no time-frequency structure.** Spectrograms of "QPSK + Gaussian noise"
are indistinguishable from "QPSK at a different SNR", so the CNN only ever learned to detect spectral
lines and periodic impulses. **Establishes that spectrograms require OFDM for the CNN detector to be
meaningful** — which is why sim05/06/07's original roadmap was merged into one sim06.

> **Refined 2026-09-17 (§3.3d).** A second cause was found when the Li et al. CNN was rebuilt for the
> baselines: the frozen sim05/06 code stretched **each** spectrogram to its own 2–98 % contrast, which
> removes absolute power and so blinds it to any jammer that lifts the whole band evenly (barrage in-band).
> With a **fixed** dB colour scale anchored to clean data (Li et al.'s own waterfall convention), the
> retrained CNN detects barrage and even a pulse-shaped QPSK jammer on the *single-carrier* link
> (`cgan/detectors.py`). So "spectrograms need OFDM" was partly a normalisation artifact; the honest
> statement is that a per-image stretch, not the single carrier, was hiding the wideband jammers.

**sim06, phase 1** puts the same detector on a 64-subcarrier 802.11a-like OFDM chain (Sionna
`ResourceGrid`: FFT 64, CP 16, 52 effective SCs (6+5 guard + DC null), Kronecker
pilots on symbols 2 and 11, 14 OFDM symbols/frame, 1120 samples/frame), trained on clean + 4 classical
jammers with Li et al.'s hyperparameters:

| Metric | sim05 (flat QPSK) | sim06 (OFDM) | Li et al. (real SDR) |
|---|---|---|---|
| Accuracy | 78.9% | **99.79%** | 99.79% |
| DR | 59.1% | **99.59%** | 99.79% |
| FAR | 2.07% | **0.00%** | 0.03% |
| F1 | 0.73 | **0.998** | 1.00 |

**Matches the published result exactly** — confirming OFDM structure was the missing piece, and
giving the project a credible SOTA defender to attack.

## A.5 sim06 (jammer) / 06b / 07 — the untrainability result

**This is the negative result Di Maio asked about directly, and the one to lead with the
*action-parameterisation* explanation rather than the reward formula.**

**sim06 phase 2** trains 2 cooperative NSF jammers with **MAPPO** (CTDE: centralized critic sees both
agents' obs+actions, actors use local obs only) against the frozen CNN, injecting in the frequency
domain per subcarrier. Reward `BER − β·P(jammed) − γ·power`. **All three runs failed identically**,
with `P(jam) ≈ 0.999` throughout: run001 (β linear, entropy 0.01) diverged power 4→33 with entropy
saturating at its ceiling; run002 (log-shaped β) amplified the penalty magnitude but it was still a
**constant across every batch element**, so nothing differentiated; run003 (entropy 0, β warmup)
stopped the power blowup but the policy **froze completely** — BER, entropy and power unchanged for
290 iterations.

**The diagnostic that explains it** (`probe_1sc.py`): the CNN is trivially fooled by single-subcarrier
jamming even at power 16 (P(jam) ≈ 0.007) but detects **all-subcarrier broadband noise instantly,
even at power 0.01 per SC** (P(jam) = 1.000). The NSF initializes to ~N(0,1) across all 128 dims —
textbook broadband noise. **Every action the agent ever tried was detected with P(jam) ≈ 0.999, so
there was never any reward differentiation to learn from.**

**sim06b** reduces the action space to **2 real dimensions** (one target subcarrier) to test whether
this is a dimensionality problem. It is not. Stealth was solved (P(jam) ≈ 0.003 throughout — a real,
varying gradient existed) but **waveform learning still failed**: per-SC BER plateaued at 0.35 against
a theoretical optimum of 1.0, with Gaussian blobs and no input correlation. **Establishes the
mechanism: a scalar frame-level reward fundamentally cannot teach an input–output correlation.** PPO
would have to randomly sample an action near `−2·tx`, get a high reward, and reinforce that specific
input-dependent direction — and isotropic exploration does not find that.

**sim07** changes exactly one axis from sim06 — the observation model — and locks in a **black-box,
score-based** threat model (the jammer sees only the scalar `P(jammed)`; the `@torch.no_grad()` in
`detect()` is load-bearing, which is what forces RL rather than direct gradient). The jammer observes
`tx[t−1]`, not `tx[t]`. For iid QPSK that carries **zero** information about `tx[t]`, so the
cancellation shortcut becomes mathematically unreachable and the agent must learn a **blind waveform
distribution** — the NSF is essentially unconditional. *(Exception: OFDM symbols 2 and 11 are pilots,
so at t = 3 and t = 12 the observation is a known pattern — any energy concentrated there would be
**protocol-aware jamming discovered through learning**, tracked via a `pilot_power_ratio` metric.)*

Five runs, each fixing the previous one's mechanical failure, and all five failing identically:

| run | change | outcome |
|---|---|---|
| 001 | baseline | P(jam) pinned 0.999; entropy hit its ceiling; power diverged 4→33 |
| 002 | + hard power cap, entropy → 0 | power fixed at 2.00, but P(jam) still 0.999 and every metric bit-for-bit flat |
| 003 | + top-K=8 subcarrier masking | first movement (P(jam) 0.985–0.996) but entropy varied <0.05% over 190 iterations |
| 004 | top-K 8→1 | P(jam) 0.35–0.44 — *not* the ≈0.0002 the static probe predicted |
| 005 | + temporally-held frame action | final: BER ≈ 0.013, P(jam) ≈ 0.3 (pure sampling noise), entropy flat at ~178.4 |

**The spectrogram probe root-caused the sparsity threshold directly:** a sharp, **power-independent**
cliff between 3 and 4 simultaneously active subcarriers (1–3 SCs → P(jam) ≤ 0.012 at any power;
4 SCs → 0.9955; 64 SCs → 0.9999 even at power 0.01/SC). The CNN keys off the *number* of anomalous
spectrogram rows, not their magnitude. Plausibly because its training set contained only "1 active
tone" and "all tones" as sparse/dense extremes, so it generalizes conservatively across the unseen
middle. **run004's discrepancy was frequency hopping:** the NSF resamples at each of the 14 OFDM
symbols, each potentially targeting a different top-1 subcarrier, and the frame-level spectrogram
aggregates all 14 → looks multi-tone. Confirmed by a controlled probe (`hopping_2sc` at power 1 gives
P(jam) = 0.4633, matching the live run exactly, vs ≈0.0002 for a temporally-fixed single SC). **Not a
learning failure — a structural mismatch between the per-symbol rollout and the frame-level
spectrogram.** run005 fixed it by sampling once per frame and holding.

**Two structural causes, not tuning:**
1. **Reward variance across the batch ≈ 0** → normalized advantages are pure noise → **PPO has no
   gradient.** Same wall as sim06/06b: a scalar frame-level reward carries no per-dimension
   information. **The reward shape is why** (added 2026-09-10, from the objective table in
   [A.0](#a0-run-index)): the detection penalty is the log barrier `−log(1−p̂+ε)`, and at `p̂ ≈ 0.999`
   that is enormous but very nearly *constant across the batch* — PPO normalises advantages within the
   batch, so a large constant penalty carries exactly as much information as no penalty at all. Stated
   this way it is a property of the objective, not a training failure, which is the form to write it
   up in.
2. The K=1 sparsity sim07 needed (to get under the detector's apparent cliff) **caps BER at
   ~2/52 ≈ 0.04** — too low to matter.

**Refined conclusion, and the one to lead with:** the root cause is the **action parameterization** —
raw IQ, very high-dimensional — not the reward formula. This is what motivates low-dimensional
perturbation *parameters* with surrogate gradients, and it is why a reward-engineering fix was never
going to work. He accepts the result on its own terms: *"showing in what cases it is hard to beat is
already a small result."* Under the adaptation-cost framing it is more than that — **direct evidence
that attacker adaptation by that route is prohibitively expensive.**

*Honest note on the inductive bias:* top-K masking is a real architectural prior — it presupposes the
solution is sparse rather than letting the agent discover it. Justified as a *feasibility check*,
because an NSF initialized to ~N(0,1) across 128 dims has no natural pathway to sparse samples
(concentration of measure means no batch element looks meaningfully different), so the policy gradient
is provably flat in the unmasked regime. The clean way to report it is as a documented finding:
*"unconstrained continuous RL cannot discover sparse evasive strategies from broadband initialization
without a structural prior."*

## A.6 Phase 0 / 0.5 / recheck — detector characterisation and errata

**Phase 0** (`frontier/frontier_sweep.py`, job 101860) is a pure-inference sweep — no RL — over
(active subcarriers × power × structure) through the frozen sim06 detector, recording **both** P(det)
and BER. Motivation: before spending more compute on RL, bound analytically what *any* jammer could
achieve at each stealth level.

**Headline: the "99.79% SOTA detector" is essentially an OUT-OF-BAND-EMISSION detector, not a jamming
detector.** The decisive control holds power and BER fixed and differs only in whether the 12 guard/DC
null bins are touched:

| Strategy | n_active | power | P(detect) | BER |
|---|---|---|---|---|
| broadband **in-band** (52 eff. SCs) | 52 | 8 | **0.011** | 0.360 |
| broadband **out-of-band** (all 64 bins) | 52 | 8 | **0.999** | 0.363 |
| broadband **out-of-band** | 52 | 0.03 | **0.9999** | 0.000 |

It fires at ~1.0 on *any* out-of-band energy — even at power 0.03 causing **zero** BER — and is
near-blind to in-band interference causing BER 0.36. **This retroactively explains two earlier
"walls"**: sim06's "broadband → 0.999 even at power 0.01" (that probe injected into guard/DC bins,
i.e. a spectral-mask violation) and sim07's "4-subcarrier cliff" (its top-K over the full 64-bin grid
was selecting guard/DC bins as it activated more subcarriers). **The earlier conclusion that
"stealth–effectiveness tension is fundamental / BER capped at 0.04" was WRONG — an out-of-band-leakage
artifact.** Temporal coherence (held vs hopping) turned out to be a minor axis; **in-band vs
out-of-band is the dominant one.**

**Phase 0.5** (`frontier/retrain_detector_inband.py`, job 101866) retrains a fresh EfficientNet on
clean + 4 classical + in-band jammers, then re-sweeps. **The blind spot closes, but only at a real
cost:**

| | Original | Retrained (+ in-band) |
|---|---|---|
| Overall accuracy | 99.79% | **90.5%** |
| FAR | 0.00% | **3.81%** (up to ~50% on near-zero-power in-band perturbations) |
| in-band broadband n=52 pwr=8 → P(det) | 0.011 | **0.995** |
| in-band DR (fresh jammers) | ~0% | **68%** — a third still evade |
| max stealthy BER at P(det) ≤ 0.50 | 0.419 | **0.020** (sparse, n=3) |

Even the omniscient `jam = −2·tx` (statistically clean `rx = −tx`) is now flagged at ≈0.15.
**Establishes an intrinsic trade-off, not just a training-data gap: a spectrogram CNN cannot catch
spectrally-compliant in-band interference without sacrificing false-alarm rate and clean accuracy.**
*Caveat:* a quick 50-epoch naive retrain, with in-band samples labelled "jammed" even at BER ≈ 0,
which inflates FAR. A careful version (BER-thresholded labels, threshold calibration) would sharpen
the exact numbers; the qualitative result is robust.

**Recheck** (`frontier/recheck_suite.py`, job 102115) — prompted by scrutiny of the Li et al.
replication (their repo confirms EfficientNet-B0 is one of their four CNNs, but they use **real
over-the-air SDR data**, train from scratch, and their spectrogram generation is undocumented). Two
faithfulness fixes: the detector spectrogram was **real-part-only** — a bug — corrected to the
standard **complex two-sided STFT** and retrained (val acc 99.8%, FAR 0%); and an **energy detector**
(mean received power vs a clean-calibrated 1% FAR threshold) was added so the frontier is evaluated
against the **detector SUITE**, not the CNN alone.

- **(A) The out-of-band finding SURVIVES the complex-STFT fix — it was real.** Same power and BER
  (0.362): in-band P(CNN) = **0.086** vs out-of-band P(CNN) = **1.000** (out-of-band is 1.000 at every
  power).
- **(B) The energy detector DEMOLISHES the "stealthy BER 0.42" claim.** Against the suite (CNN OR
  energy), max stealthy BER: P(det) ≤ 0.05 → **none**; ≤ 0.10 → **0.000**; ≤ 0.50 → **0.005**.
  In-band jamming that flips bits necessarily raises power, and a trivial power meter catches it. The
  original figure was an artifact of evaluating against the **CNN alone**. **"SOTA CNN is blind →
  jammer wins" is dead as stated.**

**Crucial caveat, and what kept sim08 alive:** this is the **noiseless lossless** channel, where clean
power is a razor-sharp constant so any added power trips the threshold. On a **realistic** channel
(finite SNR + fading) the threshold is looser and a low-power jammer can hide **under the noise
floor**.

## A.7 sim08 — the realistic channel, and the honest metric

**m1** (`simulation08/channel.py`, `frontier_channel.py`, job 101870). `MultiLinkChannel` uses Sionna
`tr38901.TDL` (model C, 100 ns delay spread, 5.2 GHz) via `GenerateOFDMChannel` for an independent
per-subcarrier frequency response per link (TX→RX and each jammer→RX), plus per-link average path gain
and AWGN at a target Eb/N0. The jammer transmits through its **own** channel `h_jam`; the RX does
perfect-CSI ZF equalization of the TX link, so post-equalization the effective interference on
subcarrier n is `(h_jam[n]/h_tx[n])·jam[n]`.

1. **The channel is physically correct** — clean BER floor waterfalls 0.088 (5 dB) → 0.0003 (30 dB).
2. **A sparse in-band jammer imposes an SNR-INDEPENDENT BER floor** — ~0.05–0.07 at n=8 across all
   SNR; broadband ~0.34. **The jammer wins in the high-SNR regime because the impairment is
   interference, not noise**, so more transmit power cannot fix it. *(This result does not depend on
   any stealth threshold and still stands.)*
3. **Channel-aware beat blind subcarrier selection by up to +70% BER at equal power** — *later
   refuted, see below.*

**m1b** (job 102305) folds in the per-SNR energy detector (calibrated on **faded** clean frames to 1%
FAR). **Stealth is revived on the realistic channel:** a jammer stays stealthy (P_energy ≤ 0.5) while
causing **BER ≈ 0.20–0.24 across all SNRs** — versus 0.005 max on the lossless channel. At 30 dB that
drives a near-error-free link to BER 0.20 while hiding under the power meter. **The noise floor and
fading provide cover the lossless channel could not.**

**m2** completes the suite. **(1)** `retrain_detector_channel.py` (job 102316) retrains
EfficientNet-B0 on complex-STFT spectrograms of signals through `MultiLinkChannel`, Eb/N0 drawn
uniformly in [5,30] dB per mini-batch, classes = faded clean (0) vs faded + classical time-domain
jammer or faded + in-band frequency-domain jammer through its own `h_jam` (1). Result: **held-out
faded-mix accuracy 94.3%, DR 91.2%, FAR 2.3%, F1 0.94** — a genuine clean/jammed boundary on the
faded channel, where the lossless-trained detector had ~13% FAR and no separation at all. Per-SNR on
fresh samples: clean FAR 10% @ 5 dB → 1% @ 30 dB; in-band DR 88% @ 5 dB → 76% @ 30 dB (it still
misses ~24% of in-band jammers at high SNR — a residual blind spot).

**(2)** The full-suite frontier (job 102319) folds the CNN and energy detector into a per-sample suite
(CNN hit OR energy hit). Two findings:

- **The suite ≡ the CNN on the realistic channel.** Across the whole sweep the energy detector catches
  **0** configs the CNN misses. Any jammer that evades the CNN (sparse, thin) also evades the power
  meter. This is the **mirror image** of the lossless recheck, where the energy detector subsumed the
  CNN — so **the expensive CNN earns its keep specifically on realistic channels.**
- **The stealthy-effective region shrinks but survives.** A residual sparse ~16-subcarrier, power ≈ 1
  jammer stays under both detectors (p_cnn ≈ 0.40–0.47, p_energy ≈ 0.02). Cost to the defender: the
  CNN's clean FAR is ~20% at 5 dB, 1–3% above.

**matched-detectability** (`matched_detectability.py` + dense re-sweep, job 102390) — **and this is
where the m1 motivation dies.** The m1 "+70%" was measured at matched jammer **config** (same
n_active, same power). But concentrating power on high-gain subcarriers raises BER **and** the
spectrogram/energy signature together — channel-aware is *louder*. The only fair comparison holds what
the defender sees (suite P(detect)) fixed. The tool builds, per strategy per SNR, the achievable
frontier `BER*(β) = max BER over configs with p_suite ≤ β` — a monotone step function of the
detectability budget — and compares blind vs channel-aware. The coarse m2 grid gave wildly swinging
gains (+5, −24, −14, −57, −6%), clearly grid noise, so the sweep was densified to **9 powers × 11
n_active at B=512**, written to a separate directory so the canonical m2 figures stayed intact.

**Result: the channel-aware advantage collapses to ≈0 at matched detectability** — dense gain
−4.6, +1.5, −1.8, +6.2, +0.0% across 5→30 dB. The two achievable frontiers sit essentially on top of
each other at every SNR. **Genie channel-aware ≈ blind once detectability is matched; the "+70%" was
a matched-config artifact.** Three nuances: (a) a faint edge (~+6%) survives only at 20–30 dB in the
mid-detectability band; (b) the residual stealthy region is confirmed and slightly *larger* than m2
reported (the finer grid finds better configs); (c) "suite ≡ CNN" softens at 30 dB, where the energy
detector catches one thin-but-loud config the CNN misses.

> **THE HONESTY CORRECTION — the most important line in this appendix.** The same sweep discredits the
> `P(det) ≤ 0.5` "stealthy BER 0.065–0.11" headline. That threshold **is not operational stealth** — a
> jammer caught half of every frame is caught within a few frames, and it contradicts m1's own
> *persistent* SNR-independent floor. At the honest stealth budget = **the suite's own clean
> false-alarm rate** (~0.12 @ 5 dB → 0.03 @ 30 dB), **no stealthy-effective jammer exists at
> 5–15 dB**, and at 20/30 dB only BER **0.011 / 0.004** (~3× / ~11× the floor) — one to two orders of
> magnitude below the ≤0.5 figure. **The "jammer beats the suite" story is real but MODEST; the strong
> version was the loose threshold talking.** The detector-characterization results (suite ≡ CNN, the
> out-of-band finding, the m1 floor) are unaffected — they do not depend on a stealth threshold.

**What this leaves standing, and what it killed.** Standing: the detector characterization — the CNN
is an out-of-band detector; closing the in-band blind spot costs FAR/accuracy; a channel-valid
retrain + energy meter close most of it on realistic channels with the roles flipping; a sparse
jammer imposes an SNR-independent BER floor. Killed: **channel-aware subcarrier selection as a lever**
(the genie extracts no matched-detectability gain from it, so "learn the channel-aware genie" was
never the plan) and the strong form of the stealth claim. Independently confirmed by his own note:
*"selecting the optimal subcarrier is a proxy problem on the way to the true problem of maximizing
BER while minimizing detection probability."*

## A.8 What the ladder means under the adaptation-cost framing

Almost nothing is wasted, **including the failures** — most of what exists already *is* adaptation-cost
data:

- **Phase 0.5** — closing the CNN's in-band blind spot costs accuracy 99.8 → 90.5% and FAR 0 → 3.8%.
  **Defender adaptation cost, round 1**, already measured.
- **m2** — the channel-valid retrain buys a genuine faded-channel decision boundary but pays ~20% FAR
  at 5 dB. **Defender adaptation cost on the realistic channel.**
- **matched-detectability** — the genie extracts ≈0 gain from channel-aware selection at equal
  detectability. **The attacker's cheap adaptation lever is already exhausted.**
- **sim06/06b/07** — black-box RL over raw IQ is structurally untrainable. Under the old framing this
  was an embarrassing dead end filed as an "ablation"; under the new framing it is **direct evidence
  that attacker adaptation by that route is prohibitively expensive** — i.e. a *contribution*.
- **E1** — the learned detector catches 11% of what the NP-optimal one catches 84% of, at σ = 0.2.
  **The remaining adaptation budget, as a number.**

So the R0/R1 rounds are largely **already banked**; the genuinely new compute for RQ2 is narrower than
it looks — a fresh retrain round using the best current attacker as input, then R2's re-optimization
against it.

## A.9 Frozen code inventory

Kept for provenance; **do not extend any of it.**

- `frontier/frontier_sweep.py` + `submit.sh` — Phase 0 (lossless). Contains `build_jam` (jammer
  families) + `detect_chunked`, reused everywhere downstream.
- `frontier/retrain_detector_inband.py` + `submit_phase05.sh` — Phase 0.5 retrain + re-sweep.
- `frontier/recheck_suite.py` + `submit_recheck.sh` — complex-STFT retrain + energy-detector suite.
  Energy detector = `frame_power()` vs a clean-calibrated threshold.
- `frontier/spectrogram_figure.py` + `submit_fig.sh` — the in-band vs out-of-band figure
  (`artifacts/frontier/inband_vs_outofband.png`, the best figure in the project).
- `simulation06/{ofdm,detector,jammer,train_detector}.py` — OFDM chain, detector (complex STFT),
  classical jammers.
- `simulation08/channel.py`, `frontier_channel.py` (per-SNR energy detector + per-sample CNN∨energy
  suite; takes `--powers`/`--n-active` overrides and writes a per-SNR incremental `results.json`
  checkpoint so a wall-kill cannot lose a sweep), `submit.sh`, `submit_frontier.sh`,
  `submit_frontier_dense.sh`.
- `simulation08/retrain_detector_channel.py` — the channel-valid CNN (m2).
- `simulation08/matched_detectability.py` — pure post-processing, no GPU.

**`sim08_ablation/` (2026-09-16, live) reads this stack but does not modify it** — it imports
`channel.MultiLinkChannel` and `frontier_channel`'s `build_jam` / `frame_power` / `calibrate_energy` /
`detect_chunked` plus `simulation06/`'s OFDM chain and detector, and adds only what the frozen
`evaluate()` cannot express: N_J > 1 jammers (it applies jammer 0 only, because `chan.apply` zips the
jammer list), a noiseless anchor, and the received JSR. Its `verify.py` re-measures frozen dense-sweep
points (job 102390) as a regression check. Results: §3.3c.

**Detector checkpoints** — note which spectrogram representation each was trained on. Only the sim08
and m0 rows are on the cluster; the sim06 and frontier rows are in git only (§1.3 note):

| Path | What |
|---|---|
| `artifacts/sim06/detector/run002_best.pt` | original **real-part** STFT — superseded *(earlier notes gave this path as `simulation06/artifacts/...`, which does not exist)* |
| `artifacts/sim06/detector/run003_best.pt` | **complex-STFT, lossless-trained** — the corrected lossless CNN |
| `artifacts/frontier/detector/run001_best.pt` | Phase 0.5 in-band-augmented (real-part era) |
| `artifacts/sim08/detector/run001_best.pt` | **complex-STFT, faded-channel-trained** — the channel-valid CNN; **the one to use on a realistic channel** |
| `artifacts/m0/detector/dl_sigma*.pt` | M0 learned IQ-histogram detectors, one per σ |

---

# APPENDIX B — Supervisor record

## B.1 Correspondence log

- **Rahul's update email** (~mid-July 2026): reported the sim06/06b/07 negative result, the three
  characterization findings, and asked two open questions — (a) is a single-round evasion attack on a
  frozen detector an acceptable contribution, or does he want a co-adaptive setting; (b) is he
  comfortable leading with detector characterization as the solid core and the cooperative learned
  jammer as the high-upside extension. Also flagged 3+ weeks without a reply, and candidly asked
  whether the drift toward detector characterization is still publishable.
- **Di Maio's reply (2026-08-03)** — reframed the thesis. Answers (a) implicitly: *"one can always
  fine-tune a defender on an attacker and vice versa … this adaptation is very expensive"* — i.e. he
  wants the **round-based / offline co-adaptive framing** with cost-of-adaptation as the headline.
  (b) was not answered directly. Point-by-point consequences are folded into §2.8, §2.9 and B.3.
- **Rahul's reply (2026-08-17,** delayed by the first exam block, acknowledged as such): proposed
  meeting the week of 17–21 Aug and registering that same week → landed on Fri 21 Aug. Stated
  availability: 15–20 Aug full time; **22–27 Aug second exam block, no thesis work**; 1–14 Sep 100% on
  the paper. Asked directly how available he is 1–14 Sep, requesting short/frequent feedback rounds
  over one large end-of-block review — **still unanswered.** Gave short answers to each feedback point.
- **Meeting 2026-08-21** — the simplification mandate and the two questions that threaten the RL
  framing. Notes transcribed 2026-09-01; consequences are throughout Part 2, and the two questions are
  §4.2 Q2 and §2.8 (CTDE).
- **2026-09-01** — registration proposal handed over for correction.
- **2026-09-12 — HIS PROPOSAL FEEDBACK ARRIVED.** Returned as inline `\adm{}` comments in
  `proposal/proposal.tex`; nine substantive, transcribed verbatim into [B.2](#b2-his-verbatim-points-and-what-each-changed).
  **The headline of the feedback is his "important:" note on Methodology** — the contribution as
  written reads as *"the application of some neural architecture"*, and he names where the novelty
  should come from instead: **how attackers and defenders interact, coordinate and synchronise
  within themselves.** Read together with his inter-jammer-delay note, that is a direct steer to the
  coordination axis, arriving independently of, and on the same day as, our own literature-driven
  pivot (§2.1). He has **not** seen the literature collision — that is still ours to disclose
  (§4.1 #0).
  *(Two of the three proposal fixes B.3 was tracking are also visibly done in the returned file: the
  `xcolor` package and the Introduction's closing line.)*

- **2026-09-24 — an update email was drafted and never sent** (reframing, the D2 result, the MARL
  question, a full draft promised for 2026-09-27).
- **2026-09-28 — Rahul's update email, SENT (night of 2026-09-27/28): his first news since 2026-09-12.**
  Content as drafted in-session (the user may have edited it before sending):
  - Apology for the delay (ill ~5 days).
  - Why the question moved: effective + undetectable jamming is bounded (square-root law, disguised
    jamming), so the study compares detectability at equal damage. Introduces the CGAN track.
  - What was done since the proposal, framed as his requests: Zhou's CGAN reproduced and used as the
    generator; classical baselines (noise, pulsed, omniscient) vs four detectors; training on his
    BER − β·detections reward; his noise ablation (SNR 0–40 dB); grey-box, on/off, learned power, a first
    multi-jammer study.
  - Results: the CNN flags the detector-trained jammer on ~14 % of frames at matched BER 3e-4 / 30 dB
    (5 runs) vs 100 % for the damage-only control; grey-box transfers with no gap; "a plain energy detector
    isn't fooled", ≈ 1 broken frame per extra alarm for every jammer vs 30 against the CNN at 15 dB
    (classical 4); limits: low damage only, and BER vs frame errors decides the comparison with on/off.
  - Story: learned detectors are the weak link, energy detection the floor. Closest prior work:
    adversarial attacks on RF classifiers (Sadeghi & Larsson; Flowers et al.) — there the attacker hides
    its own transmission, ours breaks someone else's link while that link's detector watches.
  - MARL out of the paper, citing his "even understanding the first would be good"; D6 instead, his
    adaptation question; one small MARL experiment offered if he insists, likely negative.
  - Draft: System Model core stable but partly decision-dependent (15 min on assumptions, not wording);
    Intro and Related Work still the proposal's — skip.
  - Plan: Mon D6 · Tue–Wed writing · Wed evening full draft · Thu his comments · Fri 2.10 submit.
  - Asked: OK to submit this scope Friday? a call Mon/Tue? move to the separate paper Overleaf?
  - Attached: `artifacts/cgan/snr_ablation/run003/email/email_fig1_cnn_vs_control.png`,
    `…/email_fig2_frames_per_alarm.png`.
  - **Not in it:** S1 and the gain-aware qualifier on the energy floor (§3.1) — owed in the Wednesday draft.
- **2026-09-29 — his reply.** Meeting **Thu 2026-10-01 17:00, his office**: explain the experiments and
  the math formulation in detail. **The user keeps Fri 2026-10-02 as the target**: convince him by Thursday
  that the current results are worth submitting, and fold in as many of his points as possible meanwhile.
  His points, condensed:
  - MARL negative → single-jammer paper, **if** it has "outperformance quantification" and results "strong
    and novel compared to the literature".
  - Coordination later, reframed: detection means **localization**. Jammers do not know each other's
    positions, beamforming leaves a spatial error margin, and coordination should lower the probability of
    being localized. Presence detection (a jammer vs honest interference) is "less pressing"; honest nodes
    either comply with interference reduction or do not care about being localized.
  - Naive baseline: sync to the victim's slots, compensate the travel time, add a small constant IQ vector
    that pushes the symbol's Gaussian centre over the boundary. What are its limits, how does a defender
    stop it, and how do we know the learned attacker does not learn exactly this?
  - Fixed modulation (ablate or fix). "When to stop" the attacker–detector alternation, and how to claim
    outperformance at all.
  - The draft: the intro is still related work (focus on challenges); uniform symbols is a good assumption
    (rules out symbol prediction); the attacker model looks right; channel uncertainty / time variance would
    make it more challenging; not convinced by the detector model; unclear whether the paper is attack or
    defense; P_det not specified.
  - Kurtosis: an attacker could keep the kurtosis unchanged by forcing each symbol onto another legitimate
    one ("a new idea"). A heuristic kurtosis detector is underwhelming; wants a learned one.
  - **Reading (user + session):** he most likely skimmed the draft. The CNN is in it
    (`paper_drafts/overleaf.tex` L231–241), but the Detector Model gives kurtosis a formula and the longest
    sentence and lists it before the CNN, and the Experiment History is kurtosis-heavy (sim02–04). His
    symbol-jump attack is our genie flip (T0; disguised jamming; sim04 converged to it), which his own
    uniform-symbol point rules out. **Decided by the user:** attack paper; kurtosis is stated as a
    baseline, Li et al.'s CNN as the main detector, the aim as motivating better and proactive defenses;
    localization stated as out of scope for ICC. Follow-ons: §4.3 "Supervisor 2026-09-29".

*(A note for the record: earlier drafts of the project notes speculated about a "Thu 13 Aug" meeting,
picked up from a date we had proposed to ourselves. That meeting never happened; **21 Aug is the real,
agreed slot**, and it happens to land on the date we had independently set as the "show him something"
target.)*

## B.2 His verbatim points and what each changed

Quotes preserved because the wording matters. Consequences already actioned are marked ✅.

| His point | Consequence |
|---|---|
| *"we will probably include a subset of those results … complementary results in the appendix"* → **(mtg) "Experiments 2,3 strongest add"** | ✅ Hard quota: **2–3 experiments** in the main paper; the entire characterization arc is appendix and **DONE**. Further sweeps have *negative* expected value. |
| *"the most agnostic reward for the attacker is BER − beta*detections. The other aspects should not be relevant for the reward and be controlled by the environment"* | ✅ §2.8. Delete every proxy term; power becomes a hard environment constraint. |
| *"the setup reminds a bit of GANs … one can always fine-tune a defender on an attacker and vice versa. **The core contribution is to show that this adaptation is very expensive**"* | ✅ The new headline claim, and RQ2. |
| *"this can only happen at training time: there are no ground-truth labels at execution time"* | ✅ Justifies the frozen-detector evaluation; the arms race is round-based and offline. |
| *"jammers need to synchronize with the victim's preamble … introducing some desynchronization due to cheap hardware will make the attacker more realistic and weaker, **which is good for the paper**"* | Realism axis (§4.3). He *wants* the attacker handicapped. |
| *"it is important to clearly formulate the system model and both the defender and thread [threat] models"* — **said twice** | The System/Threat model is the top **written** deliverable. Drafted in `paper_drafts/sec_system_model.tex`; **not yet in Overleaf** (§3.4). |
| *"the shapes do not seem the most energy-optimal … most of the points under attack … around the symbol classification boundary … minimal-energy alteration … (symbol error rate could also be a possible metric)"* | ✅ The boundary attack and SER. In M0 this is `boundary_genie`/`boundary_blind` (§2.5). |
| *"selecting the optimal subcarrier is a proxy problem on the way to the true problem of maximizing BER while minimizing detection probability"* | ✅ Independently confirms the matched-detectability verdict. |
| *"a form of detection is to leak information on the position of the jammer(s) so that a defender can physically neutralize them"* | Out of scope; **name it in the Threat Model** as an out-of-scope defender capability + future work (§4.3). |
| *"consider PettingZoo and BenchMARL … RLlib is famous for being too complex … I would avoid it"* | ✅ §2.8. |
| *"I did not fully get why the MAPPO jammer can't be trained against the CNN detector … showing in what cases it is hard to beat is already a small result"* | Owed a crisp write-up as a **result, not an excuse** — [A.5](#a5-sim06-jammer--06b--07--the-untrainability-result). Still open: characterize *in which cases* it is hard to beat (which detectors/regimes). |
| *"train both attacker and defender jointly, then pick one side … if performance becomes too extreme (e.g., always stealth, high BER) then **relax assumptions** … until the performance gap between your method and the baselines increases"* | The tuning protocol, handed to us. Pick the **attacker** side, as he suggests. Also: **run the baselines** — he put it in parentheses as an assumption, so it is not optional. |
| *"the most interesting investigation will still be the optimal multi-jammer coordination against one or more mobile victims"* | The destination: multi-agent + **victim mobility** (§4.3). **After the 2026-09-12 pivot this is no longer the distant destination but the actual next experiment** (G6/E3, §3.4) — the strongest single piece of evidence that the pivot moves *toward* his stated preference rather than away from it. Lead with it in §4.1 #0. |
| **(mtg)** *"Priority: simplify, as much as possible — single subcarrier, one channel"* | ✅ M0. sim06–08 frozen. |
| **(mtg)** *"First thing to add: spatial, after solving single, no noise, no prop"* | M1 is the only sanctioned extension. **"No noise, no prop" is literal** (confirmed): σ = 0, no propagation delay — which is why σ = 0 is the *anchor* of the noise sweep, not the operating point. |
| **(mtg)** *"Intro: bit-error recovery 'out-of-scope' → motivate importance, we assume it's handled by another model"* | §2.9. |
| **(mtg)** *"Impossible to beat baseline: omniscient jammer — show in results"* | ✅ In the baseline envelope, and it must appear **in the figures**. |
| **(mtg)** *"Is it a valid assumption that all legitimate symbols are equally spread? → RL shines when it can find something"* + *"scrambling makes the transmitted sequence look statistically random"* | **The sharpest challenge of the meeting** — §4.2 Q2. Turned into an experiment (G4). |
| **(mtg)** *"At decentralized execution: jammer is 'deaf' to rewards, maybe ACKs"* | ✅ §2.8 (CTDE). |
| **(mtg)** *"How is 'counter signal' not viable: add vector in random direction in I/Q plot"* | Both halves required: **motivate away** *and* **run as a baseline** (§2.5). |
| **(mtg)** *"Ablation: parameter study, increase noise and see what happens (less detection e.g.)"* + *"Noise level, ε, change exponentially"* | The **primary** ablation, on a log grid. He has predicted the direction. ⚠ E1's grid is not yet compliant (§3.2). |
| **(mtg)** *"Scenario, e.g. #jammers, #legitimate users"* | Second ablation axis. **#jammers done on the sim08 stack 2026-09-16** (N_J ∈ {1,2,4}, uncoordinated, equal total power — no gain at matched detectability, §3.3c). **First mention of multiple legitimate users** — the model is 1 TX → 1 RX today, so that half is still open. |
| **(mtg)** *"Possibly double axes, BER/detection → show trade-off; no attackers / ground-truth attacker"* | ✅ The prescribed figure format, §2.7. |
| **(mtg)** *"Put as much info as possible in Overleaf"* | The document is the working record. Assumption table, baseline table and ablation list go in **now**, as stubs if necessary. |
| *"Real hardware to implement this method is available, if you'd like to experiment later on."* | Future work (§4.3). |
| **(mtg, in `main.tex`)** *"citations should be such that text is also equally readable if removed"* | IEEEtran style: bracket **before** all punctuation with a `~` tie (`...adaptation~\cite{key}.`); group multiples as `\cite{a,b}`; keep the number out of the grammar. |

**Proposal feedback, 2026-09-12** — inline `\adm{}` comments on `proposal/proposal.tex`. Kept in a
separate table because they are one document's review, and because three of them change the
direction rather than the prose. Quoted verbatim.

| His point | Consequence |
|---|---|
| **⚠ *"important: the main contrib now seems the application of some neural architecture for this problem. discussing a novelty in the neural architecture or in how defenders and attackers interact, coordinate, and syncronize within themselves could strengthen the contribution's novelty"*** | **The most consequential comment in the set, and he flagged it himself.** It is §4.1 item 5 ("isn't this just a combination of existing methods?") arriving from the supervisor. He offers two escapes and we take the second: novelty in **coordination and synchronisation**, not in architecture. This is what sharpens §2.3 RQ1 from "coordination helps" to "coordination *under a realistic inter-jammer link*". |
| **⚠ *"I'd add somewhere here that one big issue in general is the inter-jammer or inter-node communication delay, which introduces desynchronization, and therefore makes it hard for jammers to react to honest symbol, and hard for honest node to defend against jamming symbols"*** | **Answers §4.1 #8**, which asked what the inter-jammer coordination assumption should be: not "shared backhaul" vs "independent" but a **link with delay**, and the delay is the interesting variable. Also independently motivates the desync axis (§4.3), which was on the shelf. Note the symmetry he draws: delay hurts the *defender* too. |
| **⚠ *"highlight in intro the transition from single-agent jamming, which is already a problem in itself, to multi-agent cooperative jamming. even understanding the first would be good"*** | Structural instruction for the Introduction — show the single→multi transition explicitly. **The second clause is a safety net worth banking:** he considers the single-agent case alone a worthwhile result. If G6 slips (§3.4 risk ii), that is his own words authorising the fallback. |
| *"doesn't the proposed method also use some form of prediction of activity? i.e., jammer predicts distribution of honest codewords, and defender predicts jammer presence?"* | A consistency challenge to the Introduction's "defenders are merely reactive" criticism: our attacker predicts too, so the criticism as phrased cuts both ways. **An independent second reason §2.2's framing fails**, unrelated to the square-root-law reason (§2.1). |
| *"would this motivate 'proactive' (i.e., always spend a certain amount of resources to be resilient at any time, i.e., uniformizing the codeword distribution) or 'adaptive' (i.e., increase budget, e.g., stronger modulation, when defender detects a higher probability that the channel is being jammed)?"* | Asks what defense the gap argument actually motivates, and supplies the taxonomy. **Note the convergence:** "uniformizing the codeword distribution" is essentially the shared-randomness defence from the disguised-jamming literature (SP-OFDM, §2.1) — he arrived at the published countermeasure independently. Worth telling him. |
| *"regarding channel, we can also assume a single channel to simplify the scope and we can always extend the proposed method in the future if the paper is accepted for a journal extension, e.g., IEEE Transaction on Wireless Communications"* | ✅ Confirms M0's single channel is the right cut, and re-states the simplify mandate unprompted. **New information: he has a journal-extension path in mind (TWC).** That reframes anything cut for the 6-page ICC limit as deferred rather than discarded. |
| *"Beyond conventional broadband jamming \adm{cite?}"* | Citation owed for broadband jamming in the Introduction. |
| *(commented out) "cite"* on the ML-in-PHY sentence | Already discharged — `pirayesh2022jamming` is now on that sentence. |
| *(commented out) "replace 'obvious' with more precise term: detected? if so, how are detected?"* | Already discharged — the text now reads "without producing the wide-band energy signature that conventional detectors sense". |

**Where the two sources conflict, the meeting (2026-08-21) wins.** Where the meeting was silent — the
adaptation-cost headline, reward = BER − β·det, PettingZoo/no-RLlib, the desync axis, multi-jammer
coordination against mobile victims as the destination — **the July email still stands.**

## B.3 Remaining checklist

Everything not already covered by Part 2 (settled) or §4.1 (needs his input).

**Written deliverables**
- [ ] **System & Threat Model into Overleaf.** The paper takes the waveform draft,
      `paper_drafts/sec_system_model_waveform.tex` (user pick 2026-09-24, §3.1). The M0 draft,
      `paper_drafts/sec_system_model.tex`, is thesis material. **The waveform model is now in the
      Overleaf draft**; the user's hand-saved copy of it is `paper_drafts/overleaf.tex` (2026-09-27 22:16).
      **Review 2026-09-28 (line numbers of that copy).** Stable: link and channel (L133–156), the
      detector bank with per-detector calibration (L231–240), the hard power constraint (one sentence,
      L192–197 — keep it, power is otherwise secondary, §2.7), tiers T0–T2, offline round-based adaptation.
      **Stale — fix before the Wednesday draft:** the title (L17, "Cooperative Multi-Agent … UAV
      Networks"); L125 "remaining below its detector" (the retired stealth objective, contradicting L262's
      frontier); L209–210, imitation vs detector-aware and "learns the victim's modulation" (random init
      since 2026-09-27, and run003 puts 97 % of its power on 5 % of the symbols); L130 "co-adaptive defender
      is future work" (contradicts RQ2 at L76 and D6); the threat model (L248–251) has no grey-box level,
      and run004 is the realistic attacker; the Related Work table (L117). **Decision-dependent — mark, do
      not guess:** single jammer vs team Σⱼ (MARL); frozen vs retrained defender (D6); the damage metric
      (L258 is BER, the frames-per-alarm figure is PER — define both if it goes in); the NP test (L240)
      only if Results uses it. Small: `\cite{TODO-key}` (L348), mixed `\cite`/`\citeparttwo` (L310).
      Must pin down (the thesis-level list; CTDE and inter-jammer coordination leave the ICC paper with
      MARL): where detection happens (victim RX, composite pre-equalization frame);
      what the detector observes (STFT spectrogram + mean frame power; no CSI, no runtime labels); the
      **assumption tiers** (genie / realistic / blind) for the attacker *and* for the honest policy;
      the **CTDE split**; **inter-jammer coordination** and what it costs in hardware; the power budget
      as a hard constraint; the **counter-signal non-viability** argument; jammer localization as
      out-of-scope. Repair the CLT paragraph first (§4.2 Q3).
- [ ] **Related Work into Overleaf** — replace all three legacy blocks with `sec_related.tex`. Move
      the CTDE (L256) and mobility-evasion (L262) passages to a thesis-only appendix rather than
      deleting them; he asked that nothing suggested be thrown away.
- [ ] **Rewrite `main.tex` §System Model around M0**, with OFDM/fading/multi-antenna as *extensions*.
      This also repairs a live mismatch: it currently describes a system **no working experiment
      supports.**
- [ ] **Intro:** the bit-error-recovery scoping paragraph, and the rebuild around the detection-gap
      argument (§2.2).
- [ ] **Methodology:** the scope sentence about the dropped countermeasure RQ (§2.3).
- [ ] **Move the assumption table, baseline table and ablation list into Overleaf now**, as stubs.
- [ ] **Produce an explicit triage table:** each result → main paper / appendix / dropped.
- [ ] **Write up the MAPPO untrainability as a result, not an excuse**, and characterize *in which
      cases* it is hard to beat. *(2026-09-15: stated as a result with its mechanism in one paragraph of
      `paper_drafts/sec_exphist.tex`; the "in which cases" characterisation is not in the paper.)*
- [ ] **Paste the condensed Experiment History** — §3.5.

**`refs.bib` surgery** (plan in `paper_drafts/refs_patch.bib` + `refs_new.bib`, 40 entries; do the
edits **in Overleaf**)
- [x] Li et al., IEEE Access 2022 (spectrogram detector) added 2026-09-01 — backs "state-of-the-art
      learned detector" in RQ1 and is the paper we replicated.
- [ ] Deletions he flagged: `electronics14163307` (MDPI, weak venue), `tong2025wirelessagent` (LLM
      agents, tangential), `djuhera2025r` (R-SFLLM, out of scope), `Nguyen2025_MARL_UAVRelay`
      (preprint adding nothing) + 3 others — full list in `refs_patch.bib` Part A.
- [ ] Key renames assumed by `sec_related.tex`: `jamming_survey_2024` → `pirayesh2022jamming`,
      `article` → `zhang2025cooperative`, `11302544` → `leuenberger2025proactive`.
- [ ] Optional if tooling is cited: Sionna (Hoydis et al., arXiv:2203.11854), PettingZoo (Terry et
      al., NeurIPS 2021).
- [ ] Resolve the two inclusion calls — §4.2 Q6.
- [ ] **⚠ Add the five mandatory prior-art entries (§4.2 Q6 box), 2026-09-12.** Bash ISIT 2012 /
      JSAC 2013 · Csiszár–Narayan 1988 · Li et al. TIFS 2016 + TIFS 2020 · Amuru–Buehrer GLOBECOM
      2014 / TIFS 2015 · one stealthy-FDI CPS reference. **Verified 2026-09-12: `bash2013limits` is
      absent from the live `paper/refs.bib`** (it is only in `paper_drafts/refs_new.bib`), and the
      live bibliography has no covert-comms entry at all — so all five are genuinely new additions,
      not renames.

**Proposal** — **two files on disk, both kept; Overleaf is truth and both will drift from it.**
`proposal/proposal_reviewed_2026-09-12.tex` is **his annotated copy**, saved verbatim with the
`\adm{}` comments intact — *do not clean the comments out of it*, it is the primary record of the
review. `proposal/proposal.tex` is the older, longer pre-feedback draft (262 lines, no annotations,
no Experimental Setup / Evaluation sections) and holds the five-title-option record in its header
comment. **Apply fixes to the reviewed file.**
- [x] `\usepackage{xcolor}` — **done**, visible in the returned 2026-09-12 file (now with a
      `\ifshowcomments` switch to hide the annotations for submission).
- [x] Fix the Introduction's closing line — **done**; it now reads *"against classical and
      non-cooperative baselines on the effectiveness–detectability plane"*, relocated into
      §Experimental Setup.
- [ ] Minor, **still owed** (both survive in the returned file): spurious comma in "triggered by
      detection, cannot be"; *"Coordination because…, generative because…"* is a sentence fragment —
      join with a colon or dash.
- [ ] **From his 2026-09-12 comments** (§B.2): add the missing broadband-jamming citation; add the
      single→multi-agent transition paragraph to the Introduction; add the inter-jammer
      delay/desynchronisation passage; close the Introduction on his proactive-vs-adaptive defender
      question.
- [ ] **Rewrite RQ1 and RQ2 in the proposal to match §2.3.** The submitted text still has the
      stealth-framed RQ1 ("remaining below the detection threshold") and gives RQ2 equal billing.
      **Gate this on §4.1 #0** — do not resubmit a proposal that changes direction before he has
      agreed to the change.

**Experiments** — the compute track is §3.4; the ablations he named are §4.3 and G2–G4.

---

# APPENDIX C — Engineering notes

## C.1 Artifacts convention

All training outputs go to `artifacts/simXX/`, **not** `simulationXX/runs/`. Each `train_*.py` should
write `runNNN.png` (+ `runNNN_iq.png` if applicable), save the model whenever the run is good enough
to reuse, and get a row in the run index ([A.0](#a0-run-index)). This keeps every run's plots, model
and hyperparameters discoverable in one place, and makes it possible to load a model trained in one
simulation and evaluate it in another.

## C.2 Sionna gotchas (apply to all simulations)

- Sionna returns **PyTorch tensors** — use `.abs().pow(2).mean()`, not `np.mean(np.abs(...))`. Call
  `.numpy()` before passing to numpy ops.
- `Demapper` needs a noise variance `no`. Pass `1e-10` for the lossless case (not 0) **only when the
  LLR is used non-differentiably** (e.g. just for `hard_decisions`/BER bookkeeping). With
  `no=1e-10`, LLRs blow up to ±∞ for any nonzero rx−tx deviation.
- **If the LLR feeds a differentiable loss**, use `no=1.0` and a tighter clamp (e.g. `(-10,10)`).
  `no ≈ 0` saturates the LLR and kills the gradient — this is exactly what trapped sim03b run001.
- `Mapper` output shape is `(N, 1)` — always `.squeeze()` to `(N,)` before arithmetic.
- `sn.utils.PlotBER.simulate()` is for Eb/N0 sweeps only — not for timestep loops.
- Set `sn.config.device = "cuda:0"` **before** creating any Sionna module. Sionna 2.x modules inherit
  from `torch.nn.Module` and carry `torch.compile` guards.
- `import sionna` fails **on the login node** (missing `libLLVM.so`, and the login CPU lacks `fma` so
  DrJit's LLVM fallback shuts down). Fine on compute nodes. Never compute on the login node anyway.

## C.3 GPU viability, in general

The lesson from sim04, which generalizes: tiny networks on GPU are slower than CPU because
kernel-launch overhead dominates and library round-trips force CPU syncs. Three changes flipped it:
**large batch** (amortises launch overhead), **removing library calls from the training loop** (pure
PyTorch ops run natively on GPU with no transfers), and **`torch.compile`** (fuses many small
sequential ops into fewer kernels). Together: ~248 → ~39k samples/s. *(M0 is small enough that this
does not matter — it runs on CPU in seconds.)*

## C.4 Cluster quick reference

Full detail in **[`cluster/README.md`](cluster/README.md)**; §1.5 has the summary. Day-to-day:

```bash
cd <sim dir> && sbatch submit.sh     # always submit from the sim directory
squeue --me                          # status
tail -f runs/slurm_<JOBID>.out       # watch live output
scancel <JOBID>                      # cancel
```

`SLURM_CONF=/home/sladmitet/slurm/slurm.conf` must be set on the submit host or every slurm command
fails; it is persisted in `~/.bashrc.user`. Jobs run independently — safe to close the terminal.

**A job stuck in `PENDING (BadConstraints)` is usually not a constraint problem** (found 2026-09-16).
The submit plugin adds `cpu.normal` to every job's partition list, and that partition has no GPU
features, so this reason is shown even while GPU nodes are merely full; array tasks ran with the same
label. Diagnose with evidence before changing the request:

```bash
scontrol show node artongpu03 | grep -E "CfgTRES|AllocTRES|State"   # gres/gpu and mem, allocated vs configured
sacct -j <JOBID> --format=JobID,MaxRSS,Elapsed,State                 # what a finished task actually used
```

On 2026-09-16 every GPU was allocated except two on tikgpu07, which had only ~3 GB of RAM left. A 4 GB
request could never start there; resized to 2.5 GB (measured peak 1.7 GB) it started at once. **Size
`--mem` from `sacct MaxRSS`, not by habit** — the 16 GB in older submit scripts blocks scheduling on a
busy cluster. `slurmstepd: error: TaskProlog failed` with 0 s elapsed is node-side: resubmit the task.
`NVML … GPU is lost` (artongpu01, 2026-09-15) is a failing GPU: resubmit and exclude that node.
**`cgan/submit_verify.sh` needs a 24 GB card** (`titan_rtx|geforce_rtx_3090`). Section 10's geometry
check holds ~9 GB at once and ran out of memory on an 11 GB 2080 Ti (job 2267046, 2026-09-19).

**The 48 GB cards are out of reach, so widening `--constraint` cannot rescue a full cluster**
(2026-09-21). `tikgpu08` (rtx_a6000) and `tikgpu10` (a100) are in **`disco.all`, not `disco.med`**, and
`sacctmgr show assoc user=rrahman` lists exactly one account — `disco-med`. The reachable GPU nodes are
`tikgpu02–07,09` (of which `02/03` are Pascal and unusable) plus `artongpu01–07` (2080 Ti, 11 GB). Check
partition membership before adding a feature name that looks free:

```bash
sinfo -o "%20P %25N %10T"                       # which nodes each partition actually holds
sacctmgr -n show assoc user=rrahman format=Account,Partition
```

When `tikgpu04` shows free GPUs but `CPUAlloc == CPUTot`, lowering `--cpus-per-task` does not help
either — zero free CPUs is zero. On such a day the queue estimate (`squeue --me -o "%S"`) is the honest
answer; chain the pipeline with `--dependency=afterok:<jobid>` so it runs unattended when capacity
returns, instead of waiting to submit the next stage by hand.

**CUDA random streams differ between GPU models (found 2026-09-28, D6).** The same seed and code draw
different frames on an RTX 3090 and a 2080 Ti (seed-11 colour-scale floor −44.25 vs −44.28 dB; on one
card two code paths are bit-identical, job 2275953). An evaluation that claims paired frames across jobs
must pin one card type (`cgan/submit_arms_eval.sh` pins `titan_rtx`); for anything else a re-run on
another card is statistically equivalent.

**A throttled GPU looks like a slow job (2026-09-28).** One RTX 3090 on tikgpu06 ran at 112 MHz (84 °C,
clock-event reasons 0x68: HW slowdown + SW/HW thermal), about 15× slow — a CNN retrain went from 3.5 s
to ~55 s per epoch and would have hit its time limit. Check from inside the allocation with
`srun --jobid=<JOBID> --overlap nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv`,
resubmit with `--exclude=<node>`, and tell ISG.

**Array bursts and chained dependencies (2026-09-30, `final/`).**
- **`TaskProlog failed status=1` hit the last task of an eight-task array burst** in 5 of 6 environments
  (always index 7, 0 s). `/home/sladmitet/slurm/taskprolog.sh` runs `nvidia-smi` and an `scontrol` update of
  the job comment; a race when several tasks start on one node is the plausible cause. Resubmitting the
  single task (`--array=7`) always worked.
- **An `afterok` on an array waits forever once one task failed**, and `sbatch --dependency=afterok:<id>_<t>`
  on a task that already COMPLETED is refused ("Job dependency problem"). Re-chain on the resubmitted
  task plus only the still-running tasks.
- **Dependent jobs queue behind younger arrays**: a job's age counts from when its dependency cleared.
  `scontrol top` is not permitted for users; `scontrol update jobid=<id> nice=20` on your own competing
  array is, and it reorders them.
- **The login node's `/tmp` (and with it the session scratchpad) was cleared overnight.** Keep anything a
  later session needs (e.g. a report page's source) under the repo.

## C.5 Knowledge graph (`graphify-out/`)

A queryable graph of this repo — code symbols, prose concepts and their relations — built 2026-09-20,
extended 2026-09-21, with [graphify](https://github.com/Graphify-Labs/graphify) (`~/.claude/skills/graphify/`, PyPI
`graphifyy`). Ask a question about the codebase in plain English and the skill answers from
`graphify-out/graph.json` instead of grepping; `graphify query/path/explain/affected/god-nodes` are the
explicit forms. `GRAPH_REPORT.md` is the audit trail, `graph.html` the interactive view.

**It is partial by choice.** 1970 nodes, 3977 edges, 154 communities (2026-09-29, after D6). All code
files are in (deterministic AST), as are `README.md`, `CLAUDE.md`, `cluster/README.md` and `proposal.pdf`.
Of the 90 `artifacts/*.png` figures on disk, 32 are vision-extracted (28 `cgan/`, 4 `m0/`) and **58
remain queued** (41 newer `cgan/` incl. D6's `fig_gain`/`fig_cost`, 11 `sim08/`, 5 `sim08_ablation/`,
1 `m0/`). They are manifest-unstamped, so they re-queue rather than being skipped, at ~48 k tokens each
(~2.8 M for all 58). Every live track has its code in full. The graph has 0 dangling edges and 17 self-loops (measured
2026-09-28).

**`graphify update` evicts the nodes of any non-code source that is no longer on disk**, and the CLI's
shrink guard does not stop it. On 2026-09-24 that removed the **129 nodes** from the vision-extracted
`frontier/`, `frontier_inband/` and `frontier_recheck/` figures, once they were archived (§1.3 note).
The sim01–07 figures were never extracted, so nothing was lost for them. The pre-eviction graph (1915
nodes) is kept in `BT/archive/graphify-out_pre-archive/`; `graphify-out/2026-09-24/` has a copy until
another update runs that day. To get the nodes back, restore those figures from git and that `graph.json`.

Community names are **hub-derived, not curated** — each is its community's highest-degree node. Blunt
but serviceable; `graphify label` regenerates them with an LLM if that is ever worth the tokens.

**Two commands, very different costs:**

```bash
graphify update .          # changed CODE only: AST, no LLM, no tokens, seconds — r2c step 6
/graphify . --update       # also re-reads changed docs/papers/figures: subagents, expensive
```

`graphify update` is login-node-safe for the same reason M0 is: pure tree-sitter parsing, no Sionna, no
GPU. It snapshots the old graph to `graphify-out/<date>/` and overwrites curated community names with
hub-derived ones (cosmetic; `graphify label` restores them).

`UV_TOOL_DIR`, `UV_CACHE_DIR` and the `PATH` entry are persisted in `~/.bashrc.user` — the tool venv
lives on `net_scratch`, because without them graphify's interpreter probe reinstalls the package into
the small, silently-enforced home quota (§1.5, C.4).

**Note `graphify-out/` is tracked in git** despite the `graphify-out/` line in `BT/.gitignore` — commit
`d139c42` added 131 of its files before the rule existed, and gitignore does not untrack. Left as-is by
user decision 2026-09-20. `git rm -r --cached "Tabula Rasa/graphify-out"` would untrack it if wanted.

## C.6 Project-local skill: `no-ai-slop` (`.claude/skills/no-ai-slop/`)

A writing-style skill, installed 2026-09-26 from `github.com/petergyang/no-ai-slop` at commit `000650b`
(2026-09-01, MIT). `/no-ai-slop <draft>` edits a draft; `/no-ai-slop is this slop? <draft>` names the
patterns it finds, quoting each line, and does not rewrite. Only `SKILL.md`, the `eval.md` checklist it
runs on its own output, and `LICENSE` were copied. The repo's Codex/ChatGPT manifest, logo, build script
and CI workflow were left out. The installed files are byte-identical to the source (sha256). Committed in
`558c61d`.

**Security review before install (all files, full history).** Both skill files are pure ASCII, with no
zero-width, bidi or Unicode-tag characters that could hide instructions. They contain no shell commands,
URLs to fetch, or instructions to read files or change settings. Nothing executable ships with the skill.
The repo has no symlinks, submodules or `.gitattributes`, and nothing is appended to its PNG. The only file
ever deleted from its history is an old SVG logo, which has no scripts or links. To update: re-clone,
diff against `000650b`, re-review, and copy the same three files.

**Trap for this paper:** its banned-word list includes "robust", and it trims adverbs such as
"fundamentally". In a detection paper these are often technical terms (a robust detector, a fundamental
limit), so keep them where they carry that meaning.

## C.7 Global skill: `arena` (`~/.claude/skills/arena/`)

A tournament skill, installed 2026-09-27 from `github.com/Jakeschincariol/arena-skill` at commit
`df07b8e` (2026-09-26, MIT, the repo's only commit). `/arena [--quick | --agents N] <task>` gives N
sub-agents the same task, each with a different strategy card, and runs an attack → defend → judge
bracket until one answer is left. The default is 100 agents and 595 sub-agent calls; `--quick` is 16
agents and 91 calls. It is installed under `~/.claude`, outside the repo, so git does not track it.
`SKILL.md`, `bracket.py`, `rubric.md` and `strategies.json` were copied; tests, manifests and the README
were left out. **One local change:** `disable-model-invocation: true` in `SKILL.md`'s frontmatter, so
only a typed `/arena` starts a run. Upstream it also fires on "try again" or "bad answer". Runs write
`.arena/` into the current directory, which is ignored by `BT/.gitignore`.

**Security review before install (all files).** `bracket.py` is standard library only (`argparse json
math os random re sys time`). It has no network, subprocess or `eval`, and it writes only
`.arena/<run>/` and `.arena/LATEST`. It deletes nothing. There are no hidden Unicode characters, no
symlinks and no submodules. All 27 upstream tests pass. To update: re-clone, diff against `df07b8e`,
re-review, copy the same four files, and re-add the frontmatter line.

**Trap for this repo:** "sub-agents only write inside `.arena/`" is a prompt instruction, not a sandbox.
They are `general-purpose` agents with Bash. Judges are told they may run competitor-written code in
`.arena/<run>/scratch/`, which here means the login node (no Sionna, no compute; CLAUDE.md). So run
`/arena` in the default permission mode, not auto mode. Do not run it inside a repo that already ships
an `.arena/` folder, because `bracket.py` trusts `.arena/LATEST`.
