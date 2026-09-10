# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bachelor's thesis research repo (ETH D-INFK): building a jamming attacker and evaluating it against
a *learned* jamming detector, on the effectiveness–detectability plane. Not a software product —
there is no build, no package, no test framework. Experiments are scripts submitted to a SLURM
cluster; findings live in prose.

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
  as possible*. Live work happens in `m0/` only.
- **Do not create new planning/status markdown files.** A 2026-09-10 consolidation deleted six
  overlapping docs into `README.md`. Add to the relevant README section instead.
- **Never compute on the login node** — always `sbatch`. (M0 is the exception in practice: it is
  pure-PyTorch and runs on CPU in seconds.)

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
- `import sionna` fails on the login node. M0 is sionna-free so this does not affect it.

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
