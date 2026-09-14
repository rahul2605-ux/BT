# Tabula Rasa — learned jamming under detection constraints

**Bachelor's thesis (ETH D-INFK), supervisor A. Di Maio.** Target: **ICC, deadline 2026-10-02.**
Last consolidated: 2026-09-12.

> **This file is the single entry point.** It is organised as:
> **[1. The whole picture](#part-1-the-whole-picture)** ·
> **[2. Goal & approach](#part-2-goal-approach)** ·
> **[3. Current state](#part-3-current-state)** ·
> **[4. Open questions & ideas](#part-4-open-questions-ideas)** ·
> then **[Appendix A: experiment history](#appendix-a-experiment-history)** (what each simulation
> falsified), **[Appendix B: supervisor record](#appendix-b-supervisor-record)**,
> **[Appendix C: engineering notes](#appendix-c-engineering-notes)**.
> Cluster operations live in **[`cluster/README.md`](cluster/README.md)** and are not repeated here.

---

# PART 1 — THE WHOLE PICTURE

## 1.1 What this is, in one page

> **STATE 2026-09-14 — exploratory track opened: [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).**
> The findings up to 2026-09-12 were judged too weak to carry the thesis. A new direction is being
> tried: **reproduce the CGAN jamming-waveform generator of Zhou et al. (ISSET 2025) on QPSK, then
> condition it on stealth** and evaluate against three detectors (power threshold · a statistical
> test, still to be chosen · the Zhang & Krunz 2023 CWT-CNN). The coordination direction described in
> the rest of this page is **paused, not superseded**. Live code for the new track goes in `cgan/`.

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
[Appendix A](#appendix-a-experiment-history).

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

## 1.3 Where everything lives

```
BT/
├── Tabula Rasa/            <- this repo's working tree
│   ├── README.md           <- you are here; the only planning document
│   ├── CLAUDE.md           <- operating rules for Claude Code (commands, contracts, gotchas)
│   ├── m0/                 <- live code: the minimal model (coordination track, paused)
│   ├── cgan/               <- live code: the CGAN exploratory track (§2.10), being built
│   ├── source_papers/      <- PDFs the CGAN track reproduces/uses (IEEE-licensed; git-ignored)
│   ├── cluster/README.md   <- cluster ops; read before submitting anything
│   ├── artifacts/          <- all outputs, one dir per simulation
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

> **If you meet a reference to a file that no longer exists**, it was folded into this README on
> 2026-09-10, when six overlapping documents were consolidated to remove ~2,400 lines of duplication:
> `SUPERVISOR_TODO.md` → [Appendix B](#appendix-b-supervisor-record) (its §§1–14 map onto B.2/B.3) ·
> `artifacts/RUNS.md` → [A.0](#a0-run-index) · `simulation03/README.md`, `simulation03c/README.md` →
> [A.2](#a2-sim0203c-what-a-policy-distribution-can-and-cannot-represent) ·
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
[§3.4](#34-the-plan-20-days-re-cut-2026-09-12-for-the-pivot) (items C0–C6); this table gets one row
per module as each lands. **Everything except `digitise_fig6.py` needs Sionna, so it runs via sbatch.**

| file | what |
|---|---|
| `cgan/digitise_fig6.py` → `paper_fig6.json` | C0: Zhou Fig. 6 pixel-digitised, axes calibrated on its own gridlines (the only login-node-safe script) |
| `cgan/link.py` | Sionna QPSK waveform link (upsample → pulse → AWGN + jammer → matched filter → sign decision), closed-form Gaussian-jammer BER, `measure_ber` with an error stopping rule, paper-curve helpers |
| `cgan/jammers.py` | `noise` (full / inband), `optimal` (locked / random_phase / async), exact JSR scaling |
| `cgan/verify.py` + `submit_verify.sh` | the test suite — **run first**; exit 0 = all pass (75 checks, job 2259149) |
| `cgan/calibrate.py` + `submit_calibrate.sh` | C2: fit the unstated link parameters to Fig. 6 → `artifacts/cgan/calibration.json`, `c2_calibration.png` |

**Frozen code** (sim06/07/08 + frontier) is inventoried in [A.9](#a9-frozen-code-inventory).

## 1.4 The paper — Overleaf is authoritative, this repo only reads

> **RULE: never write to the paper from this repo. Pull only.** The authoritative document is in
> **Overleaf**, which syncs to `github.com/rahul2605-ux/BT-Paper`, fetched here as the `overleaf`
> remote and merged into `paper/` as a git **subtree**. Edits happen in Overleaf, by the user.

The current Overleaf document **is the thesis**. Di Maio will open a separate one for the ICC paper
— his note at `overleaf/main` L463: *"VDN; QMIX are interesting info but not for paper, rather for
your thesis. To keep nice separation, I will create a new overleaf for paper and we keep this for
thesis."* Reinforced at L441 (*"do not throw away anything that I suggest not including in the
paper"*). This is why ICC's ~6-page limit does not constrain the appendix.

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
moves independently and `paper/` only updates on an explicit pull.

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
- Env: `/itet-stor/rrahman/net_scratch/bt_env`. The home quota is small and invisible until you hit
  it; keep bulky things on net_scratch.
- Workflow unchanged: `cd` into a sim dir and `sbatch submit.sh`. Never compute on the login node.

All 19 submit scripts were migrated and verified end-to-end (job 2243247 reproduced the recorded m2
numbers in 2.9 s). **M0 runs on CPU in seconds**, so compute is not currently a constraint at all.

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

- **BER and SER** (he asked for SER by name), P(detect) at a fixed FAR.
- **The stealth budget is the detector's own clean false-alarm rate**, not a loose convention. The
  project has already been burned once by reporting BER at `P(det) ≤ 0.5`: a jammer caught half of
  every frame is caught within a few frames. Report the **whole frontier curve**, not a single
  threshold.
- **Compare at matched detectability, not matched configuration. Non-negotiable.** This is the one
  trap the project has already fallen into: sim08-m1's "+70% channel-aware" evaporated entirely once
  BER was compared at matched P(detect) instead of matched jammer *config*. Report both ways; if a
  claim survives only the matched-power view, it is the m1 mistake repeated.
- **Figure format he asked for specifically:** one panel, **two y-axes** — BER (and SER) left,
  P(detect) right, against the swept parameter (σ, or power budget) — with the **no-attacker** and
  **omniscient-attacker** references drawn in. Keep the parametric **BER-vs-P(det) frontier** plot as
  the companion: the dual-axis view is what he wants to read, the frontier view is what supports
  matched-detectability comparisons. **Produce both, for the same runs.**

## 2.8 Settled method decisions

Do not re-litigate these; they are decided. Reasoning kept because it gets revisited when his
corrections come back.

> **2026-09-14:** the exploratory CGAN track reopens the "NOT a GAN" and "never raw IQ" rows **outside
> M0 only**, with the reason for each — see [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).
> Inside M0 both rows stand.

| Decision | Why |
|---|---|
| **Reward = `BER − β·detections`. Nothing else.** | His "most agnostic reward". Every proxy term (idle penalty, power penalty, kurtosis penalty) from sim01–04 is **deleted**. |
| **Power budget is a hard environment/action-space constraint**, not a reward term | His instruction, and sim04-run007 is the concrete proof: `GAMMA = 0.02` was negligible against BER gains, so nothing constrained power and it climbed monotonically to 4.0. |
| **Conditional generator + direct gradient. NOT a GAN.** | A GAN discriminator is a *density-ratio estimator* — it exists for the **likelihood-free** case. In M0 the ratio is **closed form**, so an adversarially trained discriminator would spend its budget approximating a function we can already write down. Use a reparameterised `G_θ(z; c) → d` + hard power projection, trained by direct gradient on the exact objective. That is sim03b's method — the one thing on the ladder that worked — and it drops GAN instability, mode collapse and discriminator scheduling from the risk list. `zhou2025cgan` stays in Related Work as the nearest neighbour, not as the method. |
| **The optimality gate is a *divergence*-constrained convex program** | `max BER s.t. E|d|² ≤ P, P_det^NP ≤ β` is **not convex** — the optimal test depends on π, so the constraint moves as the variable moves. Replace the detection constraint with `D(p₁‖p₀) ≤ δ` (or TV): BER is linear in π, power is linear in π, and the divergence is convex in `p₁` which is linear in π ⇒ a genuine convex program on a discretised `d`-grid. **Pinsker's inequality converts δ into a bound on *every* detector's error probability**, so the answer is detector-free and therefore a true ceiling. This is exactly the covert-communication formulation (`bash2013limits`), already cited for the stealth-budget convention — method and citation line up. **⚠ 2026-09-12: this program is prior art, not ours** — it is the standard stealthy-FDI formulation in the cyber-physical-systems literature, Chernoff–Stein included (§2.1). Still worth running as G1, but write it up as *instantiating* a known program. |
| **CTDE: the jammer is deaf to its own reward at execution** | BER is a **training-time** construct, available to the centralized critic only. A deployed jammer cannot measure the victim's BER. The executed policy observes its own waveform, its own channel estimate, and *at best* a 1-bit delayed noisy **ACK/NACK**. Not BER, not P(detect). This is the attacker-side mirror of his defender-side "no ground-truth labels at execution time", and it must be labelled as such in the threat model. |
| **PettingZoo for the multi-agent env API; BenchMARL only if an off-the-shelf MARL algorithm is genuinely needed; SB3 for single-agent baselines over the same env. Never RLlib.** | His words: *"RLlib is famous for being too complex for what we need, so I would avoid it."* Confirmed absent from the repo. Surrogate gradients stay the **primary** method; MARL is the comparison, not the default. |
| **Actions are low-dimensional perturbation *parameters*, never raw IQ** | This is what killed sim06/07. |
| **The detector is frozen; the arms race is round-based and offline** | His: *"this can only happen at training time: there are no ground-truth labels at execution time"*. |

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

Full record and open items in [Appendix B](#appendix-b-supervisor-record).

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
  becomes outage.
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
| Noise sweep = primary ablation, log grid | G2 | **not started** — needs the σ=0 anchor + a proper log grid (§3.2) |
| Untrainability written up as a *result* | Overleaf appendix A.6 (§3.5) | **drafted, not pasted** — `sec_exphist_6_untrainability.tex` |
| Novelty in coordination/synchronisation, not architecture (2026-09-12) | §2.3 RQ1's delay-decay curve; G6 (§3.4) | **not started** — this is the paper's contribution now |
| Inter-jammer delay + desync modelled and swept (2026-09-12) | M1 spec (§2.4); G6 | **not started** — M1 does not exist yet |
| Intro shows single→multi transition (2026-09-12) | Intro rewrite (§B.3) | **not started** |

The paper-side home for all of this is the appendix's closing subsection
(`paper_drafts/sec_exphist_10_determines.tex`), which states each mandate as a design constraint
derived from the experiment history rather than as an instruction — the rationale has to survive a
reader who does not know the supervision history, and has to be reusable in the findings paper.

## 2.10 Exploratory track: CGAN jamming waveforms under detection

**STATE: opened 2026-09-14 by the user. Step 1 (reproduction) in progress. Step 2 not started.**
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

**The link — decided 2026-09-14 at the C2 checkpoint, as stated assumptions (`cgan/link.py` `LINK`).**
- **sps 8.** A common simulation choice, well above the Nyquist minimum (sps > 1 + β).
- **RRC pulse, β = 0.35, span 32.** The classic default roll-off.
- **White-noise ("full"-band) jammer.** A classical barrage jammer.
- **Asynchronous, random-phase "optimal" jammer.** A jammer is not synchronised to its victim unless
  that is assumed.

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
  C2). Neither exists on a symbol grid.
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

**STATE 2026-09-14: exploratory CGAN track opened — [§2.10](#210-exploratory-track-cgan-jamming-waveforms-under-detection).
Step 1 (reproduce Zhou 2025 on QPSK) is in progress; nothing has run yet.** The user judged the
2026-09-12 findings too weak to carry the thesis. The coordination plan below is **paused, not
superseded**, and was not re-validated today. Whether the §4.1 #0 email to Di Maio was sent is not
recorded after 2026-09-12, and he has not been told about this track as far as this README records.

**Where it stands: C0, C1 and C2 are done, and the C2 checkpoint is passed** (link = stated
assumptions, bar = report-only, §2.10). **Next action: C3–C5** (GAN code, tests, training). Step 2 does not start until step 1 has
been reported against the paper's gaps (report-only, §2.10) and the statistical detector has been
chosen together (§4.2 Q8).

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

**Blocking** — see [§4.1](#41-blocking-needs-supervisor-input): thesis **registration** (open since
2026-09-01, still the longest-running item) · and **top of the list: sign-off on the pivot** (§4.1
#0). *Proposal feedback (#3) and the inter-jammer coordination assumption (#8) both closed on
2026-09-12.* Do not spend the remaining 20 days building toward a headline he has not agreed to —
though note his feedback already endorses the *direction*, so #0 is narrower than it was this
morning: it is the literature collision and the RQ1/RQ2 tension, not the change of subject.

**The single next action is [§4.1 #0](#41-blocking-needs-supervisor-input) — email Di Maio.** It is
half a day of writing at most, it unblocks everything, and every compute item below reads differently
depending on his answer. Draft it around three points: (i) the literature collision, stated plainly
with the three citations; (ii) the proposed new RQ1; (iii) the RQ1-vs-RQ2 tension he has to break
(§2.3). **Do not start G6 before that reply** — it is 4–5 days of new code committed to one branch of
the fork.

**Next session, in this order.**

| # | Do | Where | Note |
|---|---|---|---|
| 0 | **Email Di Maio: the pivot** | §4.1 #0 | *Do this first, before any other work.* Everything below is contingent on it. Open by adopting his coordination/synchronisation steer, then disclose the literature collision, then ask him to break the RQ1/RQ2 tension. |
| 1 | **Read the three prior-art papers in full** | §2.1 table | Bash JSAC 2013 · Li TIFS 2016 + 2020 · Amuru TIFS 2015. Desk work, no compute. Needed before *any* novelty claim goes in print, and needed to write the email in #0 credibly. |
| 1b | **Apply his prose fixes to `proposal/proposal.tex`** | §B.3 | Small and unblocked: broadband-jamming citation, the two surviving grammar items, the single→multi transition paragraph, the delay/desync passage. **Do not** rewrite the proposal's RQs before #0 returns. |
| 2 | **Paste the A.4 correction into Overleaf** | §3.5 fixes-owed | A wrong claim ("roughly double the single-agent result") is live in a document he may read. Unaffected by the pivot — the appendix records history, and the history did not change. |
| 3 | Rewrite the two A.3 sentences added in `13adaf5` | §3.5 fixes-owed | "it could try to predict it" is **falsified by part 6**. Misnames the assumption too. |
| 4 | Paste appendix parts 6, 7, 8, 9 (**drafted, revised 2026-09-11**) | §3.5 | Already written and compiling; this is a paste job, not a writing job. Paste `sec_system_model.tex` **first** — 5 and 10 `\ref` into it. |
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
> with its consequences propagated. Nothing in `paper_drafts/` was edited; parts 6–9 carry
> uncommitted revisions from 2026-09-11 (see §3.5). His annotated proposal was saved to
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

> **Gap vs the stated design: σ = 0 is NOT in the sweep.** The design (§2.4, and his mandate) calls
> for σ = 0 as an explicit anchor point, and the grid above starts at 0.02. The grid is also only
> log-spaced at the low end (0.02, 0.05, 0.1) and roughly linear above (0.15, 0.2, 0.3, 0.4, 0.5),
> where he asked for exponential spacing throughout. Both are cheap to fix and should be, before E1
> is quoted as answering the noise ablation. **Two things are lost by omitting σ = 0:** the cleanest
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

## 3.4 The plan (20 days) — re-cut 2026-09-12 for the pivot

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
| **C3** | **Generator, discriminator, losses** per Zhou Figs. 2–3. Every unstated choice is recorded in §4.2 Q7. | `cgan/models.py`, `cgan/losses.py` (pure torch) | after C2 sign-off |
| **C4** | **LINK HALF DONE 2026-09-14 — 75/75 checks pass, job 2259149** (rrc0.35 at sps 8, rect at sps 4). Two real defects were caught on the way, both fixed at the source rather than by loosening a tolerance: **(a)** span-8 RRC truncation ISI of 3.6 % flipped zero-margin symbols (job 2259141) → span 32 (§4.2 Q7); **(b)** JSR was averaged over the filter tails, over-driving short frames by up to 1.76 dB (job 2259142) → power ratios are now defined over the symbols' active window (`Link.active`). The model/loss half follows with C3. **Test suite.** Clean BER vs the Q-function reference; noise-jammer BER vs the closed form (which validates the processing gain C2 relies on); realised JSR within 0.05 dB of the request for all three jammers; a phase-locked optimal jammer at zero noise flips exactly past the amplitude margin; model shapes (G → (B,2,1024) in [−1,1]; D features 32768; score (B,1); logits (B,n_classes)); classifier loss ≡ 0 when n_classes = 1; STFT and I/Q losses ≡ 0 on identical batches; exact normalisation round trip. The link checks are written with C1 and extended at C3. | `cgan/verify.py` + `submit_verify.sh` | must exit 0 before any C2/C5/C6 job |
| **C5** | **Train.** 10,000 iterations, batch 128, Adam 2e-4 with β = (0.5, 0.999), non-saturating loss + gradient penalty (`--adv wgan-gp` as ablation). | `cgan/train_cgan.py` + `submit_train.sh` → `artifacts/cgan/run001_G.pt`, loss log, `run001_losses.png` | after C4 |
| **C6** | **Evaluate against Zhou's protocol** (SNR 30 dB, JSR −10:2:10 dB — the grid of Fig. 6's markers — 10⁴ symbols × 100 trials), **extended** to ≥ 100 bit errors or 10⁹ bits so the low-BER points are real. Overlay on the digitised Fig. 6. Report JSR at BER 1e-3 for each jammer, and the two gaps next to the paper's 1.31 / 4.44 dB (report-only, §2.10). | `cgan/evaluate.py` + `submit_eval.sh` → `artifacts/cgan/run00N_ber_vs_jsr.{json,png}` | after C5. **Then stop: step 1 sign-off, and choose the statistical detector (§4.2 Q8), before any step-2 work** |

With no pass/fail bar, hyperparameters are iterated **only** by explicit user decision after seeing
C6. Each run gets a row in [A.0](#a0-run-index).

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
- **Appendix: the sim00–08 experiment record** — see §3.5. The single largest piece of prose still
  owed, and it depends on nothing. **Four of the ten parts remain unwritten: 6 (untrainability),
  7 (detector characterisation + errata), 8 (realistic channel / matched detectability), 9 (the
  hypotheses-and-verdicts table).** Three more are drafted in `paper_drafts/` and **not yet pasted**:
  `sec_exphist_1b_objective.tex`, `sec_exphist_5_learned_detection.tex`,
  `sec_exphist_10_determines.tex`. Parts 1–4 are in Overleaf with fixes owed (§3.5).
- Repair `paper_drafts/sec_system_model.tex` (§Cooperative — the CLT argument, see §4.2); fold the
  §2.8 decisions into §Defender Model and §Generative Attack Policy.
- Related Work is drafted (`paper_drafts/sec_related.tex`, 1081 words ≈ 1.93 columns; the
  1253-word thesis version is `sec_related_long.tex`). **Not yet pasted into Overleaf** — `main.tex`
  still carries all three legacy blocks (`Related Works` L72, `Literature Review` L233,
  `Old Related Works` L309).
- Likewise `sec_system_model.tex` is **not yet in Overleaf**; `main.tex` §System Model (L107) still
  describes the K-subcarrier / TDL / N_J-jammer setting **that no working experiment supports**.

**Rough calendar, re-cut 2026-09-12 (20 days).** Sep 12–13: §4.1 #0 email + read the three prior-art
papers + G0 + spec G6 on paper. Sep 14–15: G2 + G3 while waiting on his reply; paste the appendix
backlog (§3.1 items 2–5). Sep 16–22: G5-as-component then G6. Sep 23–26: G6 results, figures,
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

**State (2026-09-12): 1, 1b, 2–5 in Overleaf with fixes owed · 6, 7, 8, 9, 10 drafted in
`paper_drafts/`, not pasted. Parts 6–9 carry uncommitted working-tree revisions made 2026-09-11**
(prose cut hard, verdict table 17 → 13 rows) — `git diff` before assuming the committed version is
current.

> **Pivot impact (2026-09-12): this appendix is unaffected and should still be finished.** It
> records what each experiment *established*, and the 2026-09-12 literature check changed none of
> that — it changed which forward claim the history supports, which is part 10's job, not parts
> 1–9's. **Part 10 (`sec_exphist_10_determines.tex`) is the one file that needs revision**: it
> closes by handing off to the stealth-frontier experiments, and that hand-off is now wrong. Rewrite
> its closing to hand off to the coordination question (§2.3 RQ1) before pasting.** Live ref last read: `overleaf/main` @ `b9620a0` "Experimental History:
Learned Detectors Section done". Reviewed 2026-09-10. **`git fetch overleaf` now fails from this repo**
(no GitHub credentials in the environment) — the local `overleaf/main` ref is what is readable, so it
may lag the true Overleaf head. It belongs in the thesis (§1.4), and writing it is directly responsive to a
request he made twice.

> **Numbering warning.** In this subsection, `A.n` means a **subsection of the Overleaf appendix**
> (the 10-part structure below, plus the inserted 1b). It does **not** line up with this README's own
> [Appendix A](#appendix-a-experiment-history), which is numbered independently. The source material
> is the same; the numbering is not.

Structure — `\section{Experiment History}`, at `main.tex` L534 as of `13adaf5`:
1. Scope and Reading Guide · **1b. The Attacker's Objective** (inserted 2026-09-10 between the scope
paragraph and part 2; drafted, `paper_drafts/sec_exphist_1b_objective.tex` — two equations plus the
per-step constants table, **ending at the table**) · 2. Simulation Chain and Validation (sim00, 04b) · 3. Gradient-Based
Attacks against Statistical Detectors (sim01, 02, 03, 03b, 03c) · 4. Two-Agent Cooperative Attack
(sim04) · 5. From Statistical to Learned Detection (sim05, sim06 detector side) — drafted,
`paper_drafts/sec_exphist_5_learned_detection.tex`, condensed to 418 words to match the register of
the user-written parts · 6. Untrainability
of Policy-Gradient RL over Raw IQ (sim06 jammer, 06b, 07) — drafted,
`paper_drafts/sec_exphist_6_untrainability.tex` · 7. Detector Characterisation and Errata
(Phase 0, 0.5, recheck) — drafted, `paper_drafts/sec_exphist_7_characterisation.tex` · 8. Realistic
Channel, Suite, and Matched Detectability (sim08 m1, m2, dense) — drafted,
`paper_drafts/sec_exphist_8_realistic_channel.tex` · 9. Summary of Hypotheses and Verdicts (verdict
table + the traceability table convention (a) owes) — drafted,
`paper_drafts/sec_exphist_9_verdicts.tex` · 10. **What the History Determines** — the forward-facing close, added 2026-09-10:
each of his mandates (§2.9) restated as a *design constraint derived from the history*, never as an
instruction, so the appendix hands off to the remaining experiments instead of stopping at a verdict
table. Drafted, `paper_drafts/sec_exphist_10_determines.tex`.

**Conventions settled.** (a) **Do not name simulations** ("sim04") in the prose — the reader has no
access to the code; write it as a continuous narrative, with a *small traceability table at the very
end* mapping narrative step → directory → job ID. (b) Register: **"we"**, past tense for events,
present tense for what remains true; rationale expressed as *what the previous step established*,
never as "I decided". (c) Every subsection follows the same four moves: **what the previous step left
open → what changed → what happened (numbers) → what it established**. Move 1 is the one that gets
skipped and the one the supervisor is reading for. (d) Negative results are findings with a
mechanism, never apologies — no "unfortunately", "we tried to", "we hoped". (e) **The objective is
stated once, up front, not per section** — a new opening subsection ("The Attacker's Objective") gives
the two forms and one table of per-step constants
([Appendix A.0](#a0-run-index)), and each later subsection then names only its *delta* in a clause.
Rationale: the supervisor cannot give useful input on a result whose objective is invisible, and the
table is the reproducibility artefact if this material is reused in the findings paper. **The
objective subsection ends at the table** — findings go at their point of use, not next to the
reference table. (f) **The appendix never re-defines what §System Model defines; it references.**
`sec_system_model.tex` §`sec:system:objective` already owns the final objective, the *matched
detectability* rule, the false-alarm-rate stealth budget and the omniscient-ceiling-in-every-figure
rule — two of the appendix drafts restated all four before this was caught on 2026-09-10. §System
Model is §II and the appendix is last, so the appendix is always the one that defers. It does still
own what §System Model has no reason to carry: the 64-subcarrier OFDM grid, the spectrogram
representation and the frozen detectors, none of which exist in the minimal model.

**Figure shortlist — 5 for the whole appendix, deliberately.** ~130 PNGs exist; almost all are
training dashboards, not findings.

| Figure | Section | Note |
|---|---|---|
| `artifacts/sim04/run001_iq_rx.png` | A.4 | bottom row of `run001_iq.png` (`rx` early/mid/late). The one place the finding is *visible*. **If A.4 quotes run007 numbers, re-crop from `run007_iq.png` — do not mix runs between text and figure.** |
| `artifacts/frontier/inband_vs_outofband.png` | A.7 | best figure in the project; four panels, self-explanatory. Use as-is. |
| `artifacts/sim06b/jammer/run001.png` | A.6 | **crop to the bottom two panels** (mean reward falling, policy entropy *rising*). The rising entropy is the smoking gun — the policy diffuses rather than learns. |
| `artifacts/sim08/frontier_dense/matched_detectability.png` | A.8 | crop to 2 panels (5 dB, 30 dB); five is overkill. |
| `artifacts/sim08/frontier/stealth_suite_vs_snr.png` | A.8 | optional; **not yet eyeballed** — check before committing. |

**Deliberately no figure in A.3 or A.5.** A.3's finding is about *how the gradient reaches the
policy*, which no single figure shows, and the only candidates are pictures of featureless clouds.
A.5 uses the six-row **cross-evaluation table** instead of a confusion matrix — it carries the
diagnosis.

**Fixes owed on the user's draft (reviewed 2026-09-10):**
- **Factual:** the first experiment had **2 legitimate TX→RX pairs, not 1**, and the jammer was
  silent t=0–4 then max-power t=5–9 (that on/off structure is the point). Overleaf §A.4's numbers are
  run001's and are **also wrong**: the draft says "BER ≈ 0.33 at 3–10% detection, per-agent power
  ≈ 0.45", but job 99211 ends at **BER ≈ 0.24, per-agent power ≈ 0.62** (total 1.25), det 2–8%,
  kurt ≈ −1.25. The claim "roughly double the single-agent result" must go — at matched total power
  the two-agent gain is ~15%, not 2× (see
  [README A.3](#a3-sim04-sim04b-a-coordinated-solution-exists-and-is-gradient-reachable)).
  Quote run001 and run007 as **two operating points**, not broken-vs-fixed.
- **Wording:** "until we find a meaningful result" → reads as an admission none exists; "we
  **setup**" → "set up". *(The "highest possible **inference**" → "similar interference" fix landed in
  `13adaf5`.)*
- **RESOLVED in `b9620a0`.** The A.3 caveat was rewritten correctly: it now names the differentiable
  white-box model *and* the genie observation of `tx[t]`, says which of the two fails on a real link,
  and states that predicting `tx[t]` from `tx[t−1]` is impossible for iid QPSK rather than merely hard.
  The "might not work / could try to predict it" hedge is gone. No further action.
- **Structural:** A.2 never says what it *establishes* (that the measurement chain is trustworthy —
  BER, power and detection all move when and only when they should), omits its numbers (BER 0→0.5,
  power 1.0→51, 0→2 users detecting), and omits sim04b. A.4 is missing both caveat sentences: the
  one-liner (kurtosis detector, lossless, genie observation ⇒ a *mechanism* result, not a stealth
  result) and the **centralised-execution** caveat — one optimizer over the union of both agents'
  parameters means coordination was *maximal*, handed over by the optimizer; what is absent is
  decentralisation, so it shows a coordinated solution exists and is gradient-reachable, **not** that
  decentralised agents could find it. That matters because "cooperative multi-agent" is in the title.
- **Sentence fragment** in A.4 ("An identical aggregate effect at …") — the same construction he
  already flagged in the proposal.
- **LaTeX:** `\appendices` already exists (L232 in `b9620a0`), so this is *not* owed — but Experiment
  History is the **third** appendix, behind `\section{Literature Review}` (L233) and `\section{Old
  Related Works}` (L309), both scratch material dense with `\adm{}`/`\rar{}` notes. It will render as
  "Appendix C" after ~300 lines of unfinished notes: **move it first among the appendices, or comment
  the two scratch sections out before he reads it.** Still owed: **remove `\nocite{*}` (L662 in
  `b9620a0`)** — it emits the whole `refs.bib`.
- **Cross-references:** no Experiment History subsection carries a `\label`, so nothing can be
  referenced. The drafts introduce `subsec:exphist:{untrainable,characterisation,realistic,verdicts}`;
  parts 5 and 6 both forward-reference the characterisation subsection.
- **The traceability table required by convention (a) does not exist in Overleaf.** It is drafted at
  the end of `sec_exphist_9_verdicts.tex`.
- Still-live `\rar{}` notes above the appendix: L216, L256.

**A.1b is drafted and PASTED** (`b9620a0`), lightly edited by the user on paste — the opening was
shortened and the κ definition moved out of the table caption into the prose, where it still stands.
Source: `paper_drafts/sec_exphist_1b_objective.tex`.
Inserts after the scope paragraph, before "Simulation Chain and Validation": two equations, the
definition of κ, and a `table*` of per-step constants. **It ends at the table.** The two findings the
table makes visible were deliberately moved to their point of use — the log barrier to part 6, γ = 0.02
to part 4 — and the "power is a constraint" paragraph was cut as a duplicate of
`sec:system:objective`. A header comment in the file records all three so they are not lost.

**A.5 is drafted and PASTED** (`b9620a0`), expanded slightly by the user on paste (the training
recipe — SGD at 1e-3, batch 32, 100 epochs — was restored, and the four jammer families spelled out).
Source: `paper_drafts/sec_exphist_5_learned_detection.tex`. Replaces the one-sentence stub and keeps that sentence as its opening line. **418 words
of prose** (vs 244 for the user-written part 4) after a rewrite for concision and register; the
cross-evaluation table is six rows × **two** columns, the third having duplicated the prose. Ends on
three deviations (binary head, ImageNet-pretrained, synthetic frames) plus the real-part-only STFT
erratum, which forward-references part 7 — **if part 7 is dropped or renamed that sentence dangles.**
Needs one `\ref` once the characterisation subsection has a `\label`, and it is the first place
`\cite{9707819}` (Li et al.) is used in `main.tex`.

**Parts 6–9 are drafted** (2026-09-10), all four verified to compile under `IEEEtran` with no errors:
- `sec_exphist_6_untrainability.tex` — leads with the **action parameterisation**, not the reward
  formula; the log barrier appears as the proximate mechanism only. The four-subcarrier "cliff" is
  written as an *observation* that part 7 explains away, never as a finding.
- `sec_exphist_7_characterisation.tex` — carries `\label{subsec:exphist:characterisation}`, which parts
  5 and 6 both forward-reference. **If this part is dropped or renamed, both dangle.**
- `sec_exphist_8_realistic_channel.tex` — both retractions (+70%, and the `≤0.5` stealth figure) are
  written to share one stated cause: a comparison that held the wrong thing fixed.
- `sec_exphist_9_verdicts.tex` — 17-row verdict table plus the traceability table convention (a) owes.

**Dependency to respect when pasting — verified 2026-09-10, and worse than recorded.** The System Model
*is* in Overleaf (L107–217 of `b9620a0`), but it is the **older OFDM version**, not
`paper_drafts/sec_system_model.tex`. It defines `sec:jammer`, `sec:detector`, `eq:problem` — and
**not** `sec:system:objective` or `sec:system:link`, which part 10 references three times. A test
compile confirms both come back undefined. It also does **not** define matched detectability, the
false-alarm-rate stealth budget, or the omniscient-ceiling rule, so convention (f) — the appendix
defers, never re-defines — is currently unsatisfiable. **Paste `sec_system_model.tex` first**, or part
10 has to carry those definitions itself.

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
> stays falsified (sim06/06b/07, [A.5](#a5-sim06-jammer-06b-07-the-untrainability-result)); actions
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
| **0** | **⚠ 2026-09-12, TOP PRIORITY — sign-off on the pivot.** Three things in one email: (i) the literature collision (§2.1) — E1's impossibility finding is prior art in the square-root law, symmetrizability and disguised jamming, so the stealth headline cannot stand; (ii) the proposed replacement, RQ1 = the coordination gain **as a function of inter-jammer delay** (§2.3) — which is *his own* 2026-09-12 steer, so frame it as adopting his suggestion, not as a unilateral change of subject; (iii) **the RQ1/RQ2 tension he must break** — his mandated headline is *adaptation cost*, but the pivot makes *coordination* the lead, and 20 days does not fit both. | **Blocks the entire remaining plan.** G5/G6 are 4–5 days of new code committed to one branch of this fork (§3.4). Do not start them on a guess. **His proposal feedback (§B.2) already endorses the direction** — coordination/synchronisation as the novelty axis — so only (i) and (iii) are genuinely open. That makes this email easier to write and *more* urgent, not less: the cheap half is already agreed. |
| 1 | **Supervisor of record for the ETH registration** | D-INFK professor requirement; may need Di Maio as co-supervisor. Needed on the myStudies form. |
| 2 | **Title, start date, end date, task description** | All four gate registration. Open since 2026-09-01. |
| 3 | ~~**Feedback on the proposal** (handed over 2026-09-01)~~ **RESOLVED 2026-09-12 — it arrived.** Nine inline `\adm{}` comments, transcribed in [B.2](#b2-his-verbatim-points-and-what-each-changed); consequences propagated to §2.1, §2.3, §2.4, §2.9, §3.4 and §4.3. Three prose fixes remain owed on the proposal itself (§B.3). | — |
| 4 | **Which 2–3 experiments go in the main paper** | His call. Candidates **changed 2026-09-12**: E3 (coordination) now leads, E2 (noise ablation) is his mandated primary ablation, E1 demoted to calibration/appendix (§3.3). Ask as part of #0. |
| 5 | **Single-round evasion on a frozen detector, or fully co-adaptive?** | His 2026-08-03 phrasing leans co-adaptive (*"one can always fine-tune a defender on an attacker and vice versa"*) but was never a direct answer. RQ2 assumes round-based-and-offline. |
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
   in [Appendix A](#appendix-a-experiment-history) are load-bearing rather than embarrassing.

## 4.2 Open technical questions

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
| "Optimal" jammer synchronisation | "identical filtering and modulation parameters" | **asynchronous**: uniform timing offset in [0, sps) samples + uniform carrier phase, both per frame | **assumed** (C2 checkpoint). It is also the only variant that produces errors below −3 dB JSR, as the paper's curve does |
| Adversarial loss | BCE equations (1)–(6) **and** "gradient penalty" in the D-loss list, with an InstanceNorm critic | non-saturating eqs. (4)/(5) + GP (λ = 10); `wgan-gp` as ablation | default — reconciles both statements |
| G conv blocks | Conv1d–BN–LeakyReLU–Dropout–MaxPool ×3, sizes unstated | channels 64/128/256, kernel 5, pool 2, dropout 0.3 | default |
| D conv blocks | Conv2d–(InstanceNorm)–LeakyReLU–MaxPool ×3, sizes unstated | (1×5) kernels, (1×2) pools, 64/128/256 channels | default |
| "Time-frequency discrepancy" loss | STFT of generated vs target (Fig. 1) | L1 between log batch-mean power STFTs — unpaired, since z is random | default |
| Feature-matching loss | named only | ‖E f(x) − E f(G(z))‖² on D's features | default (Salimans et al. 2016) |
| "I/Q distribution distance" loss | named only | sorted-sample 1-D Wasserstein on the I, Q and \|x\| marginals | default |
| Loss weights, optimiser, LR | — | all weights 1; Adam 2e-4, β = (0.5, 0.999); n_critic 1 | default |
| Training data | "target signal", normalised, inverse-normalised on output | clean QPSK waveform segments ÷ a global peak scale; G output × that scale | default |
| Iterations / test protocol | 10,000 iterations; 10⁴ symbols × 100 trials | same; plus trials until ≥ 100 bit errors or 10⁹ bits | stated + extended |

**Properties of the paper that shape both steps**, recorded so they are not rediscovered:
- **(i) BER is not in any loss term.** The generated jammer is effective only because it imitates the
  target modulation, so "conditioning on stealth" gets no help from the paper's objective.
- **(ii) Parts of Fig. 6 cannot be Monte-Carlo output from the stated protocol.** The digitised Noise
  curve reaches **8.9·10⁻¹¹ at −10 dB and 4.6·10⁻⁸ at −8 dB**; ≈2·10⁶ bits cannot resolve either value.
- **(iii) The top of the GAN curve exceeds BER 0.5** — 0.53 at +8 dB and 0.54 at +10 dB (C0; the
  markers sit visibly above the 0.5 level, not a reading error). A jammer that is independent of the
  payload cannot push Gray-QPSK BER past 0.5, so either the GAN jammer was correlated with the
  transmitted data, or these points were not measured as described.
- **(iv) Their own "optimal" is not an upper bound in their own figure.** GAN > Optimal at every
  JSR ≥ +2 dB, and Noise ≈ Optimal from +4 dB.
- **(v) The Noise curve is not a single Q-function** (a check on the digitised points, ahead of C2).
  Inverting BER = Q(√(g·10^(−JSR/10))) point by point gives an implied processing gain **g = 6.10,
  6.56, 6.70, 7.48, 8.22, 8.87, 5.45, −0.03 dB** at JSR −10, −8, …, +4 dB. It climbs steadily by
  2.8 dB and then collapses, with BER jumping 25× between 0 and +2 dB. A white-noise jammer through a
  fixed matched filter has a constant g. So C2's closed-form fit is expected to leave a systematic
  residual, and C2 must report it rather than hide it inside the parameter choice.

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

## 4.3 Ideas on the shelf — specified, not adopted

- **Time-to-first-detection.** The cheap countermeasure-facing result: measure the **trigger**
  instead of the reaction. Computable from the per-frame P(det) we already produce — no
  countermeasure, no mobility, no throughput model. It also fixes a known weakness of our own
  reporting (a threshold like P(det) ≤ 0.5 is not operational stealth), turning a hand-written caveat
  into a reported number. Reconsider if a defense-facing result is ever wanted.
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
  helps.** Parked because M0 has no subcarriers; relevant if the OFDM stack is ever revisited. Two
  further items of his live in the same parked bucket, for the same reason: *"a well-crafted
  adversarial signal could also disrupt multiple subcarriers simultaneously"* (a joint multi-subcarrier
  attack) and the simpler per-subcarrier isolated problem he thought could be interesting on its own.

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
- **The realistic range of inter-jammer delay is unknown to us** (§4.1 #8). RQ1's x-axis is
  therefore currently in arbitrary units — symbol periods, or radians of residual phase error. The
  decay *curve* is meaningful without it, but any sentence of the form "at realistic delays the gain
  is X" needs a number we do not have. Ask him, or report the axis normalised and say so.

---
---

# APPENDIX A — Experiment history

The falsification record. Kept because the Overleaf "Experiment History" appendix (§3.5) is being
written from it. Findings and mechanisms only — the per-run debugging chronology has been compressed
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

**Two findings only visible in this table**, each recorded where it is used: the **log barrier is the
mechanism of the untrainability result** ([A.5](#a5-sim06-jammer-06b-07-the-untrainability-result)),
and **γ = 0.02 never binds**, which is why run007's power blew up
([A.3](#a3-sim04-sim04b-a-coordinated-solution-exists-and-is-gradient-reachable)). The M0 row is the
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

**Detector checkpoints on disk** — note which spectrogram representation each was trained on:

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
| *"I did not fully get why the MAPPO jammer can't be trained against the CNN detector … showing in what cases it is hard to beat is already a small result"* | Owed a crisp write-up as a **result, not an excuse** — [A.5](#a5-sim06-jammer-06b-07-the-untrainability-result). Still open: characterize *in which cases* it is hard to beat (which detectors/regimes). |
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
| **(mtg)** *"Scenario, e.g. #jammers, #legitimate users"* | Second ablation axis. **First mention of multiple legitimate users** — the model is 1 TX → 1 RX today. |
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
- [ ] **System & Threat Model into Overleaf.** Drafted (`paper_drafts/sec_system_model.tex`) but not
      pasted. Must pin down: where detection happens (victim RX, composite pre-equalization frame);
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
      cases* it is hard to beat.
- [ ] **The appendix fixes owed** — §3.5.

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
