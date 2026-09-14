# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bachelor's thesis research repo (ETH D-INFK): building a jamming attacker and evaluating it against
a *learned* jamming detector, on the effectiveness–detectability plane. Not a software product —
there is no build, no package, no test framework. Experiments are scripts submitted to a SLURM
cluster; findings live in prose.

**Since 2026-09-14 an exploratory track is active** (README §2.10): reproduce the CGAN jamming-waveform
generator of Zhou et al. 2025 on QPSK, then condition it on stealth against three detectors. The M0
coordination direction is paused, not superseded.

**`README.md` is the single source of truth** for goals, state, results and open questions. Read it
before planning anything. It is deliberately structured as: 1. whole picture · 2. goal & approach ·
3. current state · 4. open questions · then Appendix A (experiment history), B (supervisor record),
C (engineering notes).

## Rules that override normal instincts

- **`../paper/main.tex` and `../paper/refs.bib` are READ-ONLY here.** They are a git subtree of the
  author's Overleaf document. Never edit, never push. LaTeX drafts are written in
  `paper_drafts/*.tex` and pasted into Overleaf **by the user**. The `.md` files in `../paper/` are
  ours and may be edited.
- **`simulation00`–`simulation08` and `frontier/` are FROZEN.** They are appendix material. Do not
  extend, re-run sweeps, or "improve" them — the supervisor's standing mandate is *simplify, as much
  as possible*. Live work happens in `m0/` and `cgan/` only.
- **Do not create new planning/status markdown files.** A 2026-09-10 consolidation deleted six
  overlapping docs into `README.md`. Add to the relevant README section instead.
- **Never compute on the login node** — always `sbatch`. (M0 is the exception in practice: it is
  pure-PyTorch and runs on CPU in seconds.) **`cgan/` has no exception**: its link uses Sionna, which
  cannot be imported on the login node, so even `cgan/verify.py` goes through `sbatch`.
- **`source_papers/*.pdf` are IEEE-licensed ETH copies** and are git-ignored. Never commit or push them.

## Commands

All Python runs need the cluster venv; there is no system torch:

```bash
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
```

M0 modules import each other as flat siblings (`import link, attacks`), so **scripts must be run from
inside `m0/`**:

```bash
cd m0
python verify.py            # THE TEST SUITE — run this first; exit 0 = all ~50 checks pass
python verify.py -v         # per-check detail

python train_detector.py    # one learned detector per sigma -> ../artifacts/m0/detector/
python frontier_m0.py --sigma 0.2       # single-sigma sweep; --sigmas for all
python figures.py           # all figures -> ../artifacts/m0/*.png
python figure_geometry.py
```

Cluster submission (submit from the directory containing the script — scripts `cd "$SLURM_SUBMIT_DIR"`):

```bash
cd m0
sbatch submit_train.sh                  # detector training
sbatch submit_frontier.sh               # the E1 sweep, 8-task array, one sigma per task
squeue --me
tail -f runs/frontier_<JOBID>_0.out
```

`cgan/` (**being built** — README §3.4 items C0–C6 say which scripts exist yet) follows the same
layout: flat sibling modules, run from inside `cgan/`, **but every entry point is a submit script**
(Sionna):

```bash
cd cgan
sbatch submit_verify.sh      # THE TEST SUITE for cgan/ — exit 0 before any other cgan job
sbatch submit_calibrate.sh   # fit unstated link params to Zhou Fig. 6 -> ../artifacts/cgan/calibration.json
sbatch submit_train.sh       # CGAN training -> ../artifacts/cgan/runNNN_G.pt
sbatch submit_eval.sh        # BER vs JSR, noise/optimal/GAN -> ../artifacts/cgan/runNNN_ber_vs_jsr.*
tail -f runs/verify_<JOBID>.out
```

Reading the live paper draft (never touches the working tree):

```bash
cd /home/rrahman/BT
git fetch overleaf
git show overleaf/main:main.tex
git log -1 --format='%h %ad %s' overleaf/main
```

`verify.py` is the closest thing to a test suite: it checks measured quantities against closed-form
predictions (BER vs `Q(1/(σ√2))`, SER vs `1−(1−BER)²`, each attack against its geometric prediction,
each detector's realised false-alarm rate against its target). **Run it after any change to
`link.py`, `attacks.py` or `detectors.py`.** There is no way to select a single check via CLI.

## Architecture

### M0 — the live model (`m0/`)

One QPSK symbol at a time, one channel (`h = 1`), AWGN with swept σ. `y = s + d + w`, decisions by
per-axis sign test. No OFDM, no fading, no pilots. The whole point of shrinking this far: at this
size the **Neyman–Pearson optimal detector is computable in closed form**, so results read *"no
detector can do better than X"* rather than *"our CNN failed to catch it"*.

Four modules with a deliberate contract between them:

- **`link.py`** — constellation, channel, BER/SER, and the analytic references everything is checked
  against (`theoretical_ber`, `ebn0_db`, `sigma_for_ebn0_db`).
- **`attacks.py`** — `build_attack(name, sym, power, ...) -> (d, x)`. **The two returns are the key
  abstraction:** `d` is the perturbation in the *equalised* domain (what lands on the victim's
  constellation, i.e. what causes BER) and `x` is what the jammer actually *transmits* (what a
  detector sees). **The hard power budget applies to `x`, never to `d`** — that separation is what
  makes the effectiveness/detectability trade-off measurable at all. Eight tiers in `ATTACKS`,
  ordered by attacker knowledge; only the symbol-aware tiers (`boundary_*`, `counter_*`) may read
  `sym`, and the blind tiers use its shape alone.
- **`detectors.py`** — three detectors sharing one interface contract: every statistic is built so
  **larger = more suspicious**, then `calibrate(stat_clean, alpha)` sets the threshold from the
  empirical (1−α) quantile of *clean* frames, so the false-alarm rate is honoured by construction.
  The three are `energy_statistic` (one- and two-sided), `LearnedDetector` (CNN on a 2-D IQ
  histogram), and `np_statistic` (the exact LRT, built from `log_p0`/`log_p1`).
  **Invariant that must hold everywhere: `P_NP ≥ P_learned ≥ P_energy`** — an "optimal" detector
  losing to a CNN means the maths is wrong. `verify.py` asserts it.
- **`frontier_m0.py`** — the sweep, and the only place the three above meet: for each (σ, attack,
  power, duty) it records BER/SER *and* each detector's P(detect), writing one JSON per σ.

Two traps when extending this: a hard IQ histogram is **not differentiable** (soft-bin or KDE it if
a learned detector must sit inside a training loop, while `np_statistic` is differentiable as
written); and the NP test depends on the attack law, so a moving generator moves the optimal test
with it — recompute the LRT rather than learning it.

### CGAN track — the exploratory model (`cgan/`, being built)

A waveform-level QPSK link (upsampling, pulse filter, AWGN at SNR 30 dB, jammer, matched filter,
hard decisions) with three jammers — Gaussian `noise`, Zhou's matched-modulation `optimal`, and
`gan(G)` — compared on BER vs JSR. Goal, success bar and every decision: README §2.10; the plan:
§3.4 C0–C6; unstated-parameter choices: §4.2 Q7.

**Library split (decided 2026-09-14, README §2.10 Decision 0):** the **GAN is plain PyTorch**
(`models.py`, `losses.py`, `train_cgan.py` — no GAN framework, no Sionna import, so every loss term
stays visible for step 2); the **link is Sionna 2.0.1** (`sionna.phy.mapping`, `sionna.phy.signal`
filters/up-/down-sampling, `sionna.phy.channel.AWGN`, `compute_ber`). README §C.2's Sionna gotchas
apply to `link.py`. The analytic references are re-derived locally, not imported from `m0/` — both
directories have a `link.py`, and flat sibling imports would collide.

Contracts that carry over from M0: **JSR is imposed by a hard power projection** on the transmitted
jammer waveform, never by a loss term; any detector added in step 2 follows the
larger-is-more-suspicious + `calibrate(stat_clean, alpha)` convention of `m0/detectors.py`.

### The frozen stack (`frontier/`, `simulation06/`, `simulation08/`)

Read only to write up the appendix. Shape worth knowing so you can navigate it: `simulation06/`
holds the 64-subcarrier OFDM chain, the EfficientNet-B0 spectrogram detector and the classical
jammers; `frontier/` sweeps a frozen detector over a jammer grid recording both P(det) and BER
(`build_jam` + `detect_chunked` there are reused downstream); `simulation08/` adds `MultiLinkChannel`
(per-link TDL fading + AWGN + ZF equalisation) and evaluates against a per-sample CNN∨energy *suite*.

Two things about this stack are load-bearing history rather than trivia, and Appendix A explains
both: the detector spectrogram must be a **complex two-sided STFT** (it was real-part-only, a bug
that invalidated a headline), and detector checkpoints are **not interchangeable** — each is valid
only for the channel and representation it was trained on (see the checkpoint table in README A.9).

### Where outputs go

`artifacts/simXX/` — never `simulationXX/runs/`. `runNNN.png` (curves), `runNNN_iq.png` (IQ scatter),
`runNNN_model.pt`. New runs get a row in the run index (README §A.0). `runs/` inside a sim directory
holds only SLURM `.out`/`.err`.

## Cluster gotchas (ITET/TIK — full detail in `cluster/README.md`)

These silently break jobs rather than erroring usefully:

- **Do not set `--partition`** — a lua submit plugin overrides it from account membership.
- **`--gres=gpu:1`, never `--gpus=1`**, plus
  `--constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090`. `tikgpu02/03` are TITAN
  Xp (sm_61) and torch 2.9 dropped Pascal — jobs landing there die with `no kernel image is
  available`.
- **Pin CUDA 12.8** when rebuilding the venv; default torch wheels are cu13 and need driver ≥580,
  while every node here runs 535.
- `SLURM_CONF=/home/sladmitet/slurm/slurm.conf` must be exported or every slurm command fails
  (persisted in `~/.bashrc.user`).
- The home quota is small and **invisible until you hit it** (`df` reports the whole NFS export).
  Anything bulky goes on `net_scratch`.
- `import sionna` fails on the login node. M0 is sionna-free so this does not affect it; **`cgan/`
  is not**, so all of it — tests included — goes through `sbatch`.

## Source papers (`source_papers/`, git-ignored)

The CGAN track (README §2.10) is built on two papers. Read the PDF before changing anything that
claims to follow it; the known traps are listed here so they are not rediscovered.

- **`L.Zhou 2025.pdf`** — Zhou, Tan & Xu, *"Communication Jamming Waveform Generation Technology Based
  on Conditional Generative Adversarial Networks"*, ISSET 2025, pp. 373–377. Bibkey `11184988` in
  Overleaf, `zhou2025cgan` in `paper_drafts/`. **Step 1 reproduces it (QPSK only).** What we take:
  G (z ∈ ℝ⁴⁰⁰ ⊕ 16-d label embedding → 3 Conv1d blocks → Linear → Tanh → 1024 I + 1024 Q) and D
  (2-channel (1,1024) input → 3 Conv2d blocks → features (256,1,128) → scoring head + auxiliary
  classifier head); G loss = time-frequency (STFT) + adversarial + feature matching + I/Q
  distribution distance; D loss = adversarial + gradient penalty + classification; 10,000
  iterations; test at SNR 30 dB, JSR −10…10 dB, 10⁴ symbols × 100 trials; baselines Gaussian noise
  and an "optimal" same-modulation jammer. **Traps:**
  - 1024 samples in the figures vs 1200 in the text — use 1024;
  - the text's feature size (256,1,28) is a typo for (256,1,128);
  - BCE equations sit next to a "gradient penalty" term and an InstanceNorm critic;
  - most link and training parameters are unstated — README §4.2 Q7 holds the full table of what we
    chose;
  - the plotted BERs (~1e-8) are below what its own protocol can measure;
  - **BER appears in no loss term**;
  - its "optimal" jammer is not the Amuru–Buehrer optimum (README §4.2 Q9).
- **`Zhang & Kunz 2023.pdf`** — Zhang & **Krunz** (the filename misspells him), *"Detection and
  Classification of Smart Jamming in Wi-Fi Networks Using Machine Learning"*, MILCOM 2023,
  pp. 919–924. Bibkey `zhang2023detection`. **Step 2's SOTA detector baseline.** What we take:
  - complex Morlet CWT scalogram (f_b = 2, f_c = 1) of sliding I/Q windows;
  - DCNN₁: 7×7 conv32 → BN → 3×3 maxpool → 3×3 conv32 → 2×2 maxpool → 3×3 conv32 → 3×3 conv32 →
    2×2 avgpool → FC; stride 2 except the 2nd/3rd conv; ReLU; softmax; 32,292 parameters at
    400×100 input;
  - Adam at lr 1e-3, batch 128, 10–20 epochs, early stopping with patience 3;
  - trained on a mixture of SJRs.

  **Traps:**
  - it is an 802.11ac OFDM *4-class* classifier (clean / preamble / pilot / interleaving) — on
    single-carrier QPSK it is an adaptation, and must be reported as "their architecture, retrained"
    (README §4.2 Q10);
  - `pywt` is not in the venv — write the CWT in torch.

## `r2c` — ready to clear

When the user types **`r2c`**, the conversation context is about to be wiped. Bring `README.md` to a
state where a fresh instance with **no memory of this session** can resume without losing anything.
Work through all of it, in order:

1. **Fold this session's findings into the topically correct section** — results and status into
   Part 3, new uncertainties into Part 4, settled decisions into Part 2, experiment outcomes into
   Appendix A. A finding that exists only in the conversation is lost the moment context clears.
2. **Update the status markers** — §3.1, the "STATE:" lines, and any date stamps — so they describe
   now, not the last session.
3. **Correct what the session disproved.** Do not leave a superseded claim standing next to its
   replacement; fix it in place and say what changed if the old version might still be quoted
   elsewhere (e.g. in Overleaf).
4. **Delete what the session made redundant.** The README is deduplicated by policy — a new section
   that restates an old one means the old one goes.
5. **Re-validate mechanically:** internal anchors resolve, referenced file paths exist on disk,
   numbers/job IDs match the artifacts they cite. Do not trust prose that was carried forward.
6. **Report the git working-tree state** — staged, unstaged, untracked — and say plainly what is
   uncommitted. Do not commit unless asked.
7. **Close with a short handoff note in chat:** what changed, what the single next action is, what is
   blocked and on whom.

**Do not add a changelog, session log, or "recent changes" section to the README.** That is how the
duplication removed on 2026-09-10 accumulated in the first place. Findings go where they belong by
topic, not by date.

## Methodology constraints that affect what you write

These are decided; recheck README §2.7–2.8 before proposing otherwise.

- **Compare at matched *detectability*, not matched configuration.** The project has already lost one
  headline to this: a "+70% channel-aware" gain evaporated entirely when BER was compared at matched
  P(detect) instead of matched jammer config. Report both ways.
- **The stealth budget is the detector's own clean false-alarm rate**, not a loose threshold like
  `P(det) ≤ 0.5` — a jammer caught half of every frame is caught within a few frames. Report the
  whole frontier curve.
- **Attacker reward is `BER − β·detections`, nothing else.** Power is a hard environment constraint,
  never a reward penalty term. Proxy reward terms are what produced the misleading sim04 result.
- Actions are low-dimensional perturbation *parameters*, never raw IQ — raw-IQ policy-gradient RL is
  the falsified method (README §A.5).
- **CGAN-track exceptions, and their limits (README §2.10).** The `cgan/` generator outputs raw IQ and
  is trained as a GAN. That is legitimate there, because it is trained by *direct gradient through a
  differentiable discriminator*, not by policy-gradient RL on a scalar reward, and no closed-form
  density ratio exists at waveform level. Step 1 uses Zhou's losses unchanged, because it is a
  reproduction. **Neither exception applies to `m0/`**, and neither licenses PPO over IQ anywhere.
  Matched detectability, the FAR stealth budget and power-as-hard-constraint **do** apply to `cgan/`.
