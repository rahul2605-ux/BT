# Tabula Rasa — learned jamming under detection constraints

**Bachelor's thesis (ETH D-INFK), supervisor A. Di Maio.** Target: **ICC, deadline 2026-10-02.**
Last consolidated: 2026-09-10.

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

A jamming attacker is built and evaluated against a *learned* jamming detector, and the question is
not "does the jammer win" but **what does the effectiveness–detectability trade-off actually look
like, and what does it cost either side to move it.**

The project began as "a cooperative multi-agent RL jammer that fools a CNN detector". That bundled
three independent bets: (a) multi-agent cooperation, (b) reinforcement learning over raw IQ,
(c) black-box access to the detector. A ladder of thirteen simulations (sim00 → sim08) **falsified
(b)+(c) as a method** — RL-over-raw-IQ with black-box access is structurally untrainable here, not
merely badly tuned. What survives, method-agnostic, is the **scientific question**: *can a learned jammer
evade a state-of-the-art learned detector while staying effective, on a realistic channel?*

After a supervisor mandate to **simplify as hard as possible** (2026-08-21), the working model is no
longer the 64-subcarrier OFDM stack but **M0**: one QPSK symbol, one channel, AWGN with swept σ.
The retreat buys something the big stack could never have: at this size the **Neyman–Pearson optimal
detector is computable in closed form**, so results read *"no detector can do better than X"*
instead of *"our CNN failed to catch it"*.

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

## 1.3 Where everything lives

```
BT/
├── Tabula Rasa/            <- this repo's working tree
│   ├── README.md           <- you are here; the only planning document
│   ├── CLAUDE.md           <- operating rules for Claude Code (commands, contracts, gotchas)
│   ├── m0/                 <- THE LIVE CODE (minimal model)
│   ├── cluster/README.md   <- cluster ops; read before submitting anything
│   ├── artifacts/          <- all outputs, one dir per simulation
│   ├── paper_drafts/       <- LaTeX sections drafted here, pasted into Overleaf by hand
│   ├── proposal/           <- registration proposal (reference draft; Overleaf is truth)
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

## 2.1 The question, and what has been falsified as *method*

The enduring question: **can a learned jammer evade a SOTA learned detector while staying effective,
on a realistic channel — and what does it cost either side to adapt?**

MARL + black-box + raw-IQ was a **method-hypothesis**, tested and largely rejected (sim06/06b/07).
Re-anchoring on the question rather than the method is why signature-shaping is a *return* to the
goal, not a drift: it pursues the same question with the method the ladder proved works
(direct/surrogate gradient, sim03b/sim04) instead of the one it proved does not.

**What this actually is, in the literature's terms: an adversarial-ML evasion attack with
wireless-physical constraints.** Signature-shaping via surrogate-gradient transfer is precisely a
transferability-based **evasion attack** against a *fixed* classifier (Papernot-style). The novelty
over vanilla adversarial ML is the constraint set: the perturbation must be a **physically
realizable** interference through the jammer's own channel, AND it must **cause BER**, not merely
flip a label. That triple — realizable + effective + stealthy — is the contribution.

**Scope caveat to state plainly:** this is a **single-round** attack on a **frozen** detector — an
evasion result, not a full adaptive arms race. Standard for the genre, but not a solved
co-adaptation game. RQ2 (§2.3) is what partially buys this back.

## 2.2 The motivating argument — the UAV detection gap

Settled 2026-09-01 while drafting the registration proposal. This is what the Introduction argues.

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

> **Honesty constraint on claim 5 — load-bearing, and the first thing a reader will test.**
> "Nobody studies stealthy jamming" is false and easy to attack: covert communication and LPI
> waveforms exist. The defensible claim is the **conjunction** — explicit stealth objective **and**
> evaluated against a *learned* detector **and** reported as a trade-off curve. Related Work must
> name the nearest neighbours (`wen2025generative` GAN-aided covert comms, `valianti2024cooperative`
> cooperative-RL jamming) and say precisely what they do not do.

The characterization work (Phase 0/0.5, m1, m2, matched-detectability) is now an *instrument* of this
argument rather than the contribution itself.

## 2.3 Research questions, as registered

- **RQ1** — can a cooperative multi-agent generative policy degrade the link while staying below a
  SOTA learned detector's threshold, and what trade-off does it achieve **relative to baselines**
  (barrage / closed-form minimum-energy / single-agent learned / omniscient cancellation as ceiling)?
- **RQ2** — **adaptation cost**, round-based and offline: frozen detector → attacker optimized
  against it → detector retrained → attacker re-optimized. What does re-closing the gap cost the
  defender, and how much does the attacker recover?
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
method, target and constraint without committing to a result. The sharper *"Breaking the Detection
Assumption: …"* is the better eventual **paper** title but promises a finding not yet measured.

## 2.4 The model — M0 and M1

Per the 2026-08-21 mandate. **M0** = the simplified base (built). **M1** = M0 + spatial, the *only*
sanctioned extension. Naming is ours; he did not name them.

| | **M0 — built** | **M1 — the one sanctioned extension** | *(sim06–08, frozen)* |
|---|---|---|---|
| Subcarriers | **1** — no OFDM grid, no IDFT, no guard/DC/pilot bins | 1 | 64-SC OFDM |
| Channel | **one** fixed realization, `h = 1` | per-jammer link gain/phase, superposing at RX | TDL freq-selective, per-link, per-frame |
| Propagation | **none** — no delay, no path loss (literal) | none; geometry enters only as per-link gain/phase | — |
| Noise | **AWGN, σ swept** | single global σ, same sweep | per-SNR Eb/N0 5–30 dB |
| Jammers | 1 | **N_J ≥ 2, coordinated** — the point of M1 | 1 (frontier) / MAPPO team (dead) |
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

| Decision | Why |
|---|---|
| **Reward = `BER − β·detections`. Nothing else.** | His "most agnostic reward". Every proxy term (idle penalty, power penalty, kurtosis penalty) from sim01–04 is **deleted**. |
| **Power budget is a hard environment/action-space constraint**, not a reward term | His instruction, and sim04-run007 is the concrete proof: `GAMMA = 0.02` was negligible against BER gains, so nothing constrained power and it climbed monotonically to 4.0. |
| **Conditional generator + direct gradient. NOT a GAN.** | A GAN discriminator is a *density-ratio estimator* — it exists for the **likelihood-free** case. In M0 the ratio is **closed form**, so an adversarially trained discriminator would spend its budget approximating a function we can already write down. Use a reparameterised `G_θ(z; c) → d` + hard power projection, trained by direct gradient on the exact objective. That is sim03b's method — the one thing on the ladder that worked — and it drops GAN instability, mode collapse and discriminator scheduling from the risk list. `zhou2025cgan` stays in Related Work as the nearest neighbour, not as the method. |
| **The optimality gate is a *divergence*-constrained convex program** | `max BER s.t. E|d|² ≤ P, P_det^NP ≤ β` is **not convex** — the optimal test depends on π, so the constraint moves as the variable moves. Replace the detection constraint with `D(p₁‖p₀) ≤ δ` (or TV): BER is linear in π, power is linear in π, and the divergence is convex in `p₁` which is linear in π ⇒ a genuine convex program on a discretised `d`-grid. **Pinsker's inequality converts δ into a bound on *every* detector's error probability**, so the answer is detector-free and therefore a true ceiling. This is exactly the covert-communication formulation (`bash2013limits`), already cited for the stealth-budget convention — method and citation line up. |
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
| Adaptation cost is the headline | G7 (R0/R1/R2); the NP−learned gap E1 already measures | **not started** — G7 is last in §3.4 |
| Only 2–3 experiments in the main paper | triage table (§B.3); candidates E1/E2/E3 | **blocked on him** (§4.1 #4) |
| Omniscient jammer in **every** results figure | `m0/figures.py` | **PARTIAL — 2 of 4.** `fig_tradeoff` (L92) and `fig_frontier` (L135) carry `counter_flip`; **`fig_detectors` (L162) and `fig_stealth_vs_sigma` (L202) do not** — their attack lists omit `counter_null`/`counter_flip` |
| Intro scopes out FEC/ARQ, then motivates raw BER/SER | Intro rewrite (§B.3) | **not started** |
| Put as much info as possible in Overleaf | assumption/baseline/ablation stubs (§B.3) | **not started** |
| Noise sweep = primary ablation, log grid | G2 | **not started** — needs the σ=0 anchor + a proper log grid (§3.2) |
| Untrainability written up as a *result* | Overleaf appendix A.6 (§3.5) | **next in the writing thread** |

The paper-side home for all of this is the appendix's closing subsection
(`paper_drafts/sec_exphist_10_determines.tex`), which states each mandate as a design constraint
derived from the experiment history rather than as an instruction — the rationale has to survive a
reader who does not know the supervision history, and has to be reusable in the findings paper.

---

# PART 3 — CURRENT STATE

## 3.1 Status line

**M0 exists, is verified, and E1 has been run. The headline is NEGATIVE.** The deadline moved from
15 Sept to **ICC, 2026-10-02** — 24 days instead of 7, a 3.4× expansion that changes the *plan* rather
than merely relaxing it: the ablations **and** the learned-attacker arc both fit, where before they
were mutually exclusive.

**Blocking and outstanding since 2026-09-01:** thesis **registration** (title, dates, supervisor of
record) and Di Maio's **feedback on the proposal**. The deadline move does not make registration less
blocking. Chase both. See [§4.1](#41-blocking-needs-supervisor-input).

**The single next action is [G1](#34-the-plan-24-days-two-parallel-tracks)** — the
covertness-constrained optimality program. ~1 h of CPU, and it is the gate: it decides whether a
learned attacker has any headroom to chase, and therefore what G5/G6 are even *for*. Everything else
in the compute track (G2–G4) is independent of it and can run alongside. The writing track — the
Overleaf appendix (§3.5) and the **five** drafted LaTeX sections still not pasted in (§3.4) — needs no
compute and is not blocked by anything.

**Next session, in this order.** Nothing here needs the cluster except item 5.

| # | Do | Where | Note |
|---|---|---|---|
| 1 | **Paste the A.4 correction into Overleaf** | §3.5 fixes-owed | *Do this first.* A wrong claim ("roughly double the single-agent result") is live in a document he may read. The rest of the owed fixes ride along. |
| 2 | Rewrite the two A.3 sentences added in `13adaf5` | §3.5 fixes-owed | "it could try to predict it" is **falsified by part 6**. Misnames the assumption too. |
| 3 | **Write appendix parts 6, 7, 8, 9** | §3.5 | 6 = untrainability, lead with action-parameterisation, and it now has the log-barrier mechanism ([A.5](#a5-sim06-jammer-06b-07-the-untrainability-result)). 7 and 8 must land or part 5's erratum sentence forward-references nothing. 9 is a table. |
| 4 | Paste `sec_system_model.tex`, **then** parts 1b / 5 / 10 | §B.3, §3.5 | Ordering is load-bearing: 5 and 10 `\ref` into it, so pasting them first leaves dangling refs. |
| 5 | Add the omniscient reference to `fig_detectors` and `fig_stealth_vs_sigma` | §2.9 coverage, `m0/figures.py:162,202` | Small, CPU-only. Do it **before** regenerating E1 figures — the paper asserts the convention in print (`sec:system:objective`). |
| 6 | G1 | §3.4 | The compute gate. Independent of 1–5. |

**Do not** re-derive the sim04 numbers from the README alone — §A.3's figures came from the SLURM
logs on 2026-09-10 and the prose that preceded them was wrong. `simulation04/runs/slurm_99211.out`
(run001) and `slurm_100041.out` (run007) are the sources.

> **Where the last working session stopped (2026-09-10, second session of the day).** No experiments
> were run and nothing in `m0/` was modified, so `verify.py` is still valid as last run. The session
> was entirely Overleaf-appendix writing (§3.5) and produced three drafts in `paper_drafts/`, none yet
> pasted. Re-reading the SLURM logs to source those drafts **disproved the sim04 two-agent headline**
> (§A.3) and produced the **per-step objective table** (§A.0), which in turn gave the untrainability
> result a mechanism. A mandate-coverage audit (§2.9) found the omniscient reference **missing from
> two of the four M0 figures**.

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

## 3.3 E1 result — the negative headline

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
"honesty correction", which measured the same thing empirically on the OFDM stack.

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
   `BER*(P) = min(P/4, 0.5)` — invisibility costs exactly 4× the power. **Do not oversell this:** it
   is a direct consequence of the symmetry, the `d = −2s` case was already known ("statistically
   clean rx = −tx"), it is knife-edged, and it needs the genie.

**The adaptation-cost reference point now exists.** At σ = 0.2, α = 0.05, `boundary_blind` at
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

## 3.4 The plan (24 days) — two parallel tracks

The writing track needs no compute and the compute track has idle time built into it.

**Compute track.**

| # | Item | Cost | Gate |
|---|---|---|---|
| **G0** | **Add the omniscient reference (`counter_flip`) to `fig_detectors` and `fig_stealth_vs_sigma`** in `m0/figures.py` (L162, L202 — their attack lists omit it). His mandate is *every* results figure, and `sec:system:objective` states the convention in print. | ~15 min, CPU | **Before any figure regeneration.** |
| **G1** | **The covertness-constrained optimality program** (§2.8). Convex, CPU, discretised `d`-grid. Produces the ceiling for *any* attacker at each (σ, P, δ). | ~1 h compute, ~½ day to write | **Do first.** Everything below reads differently depending on its answer. |
| G2 | **E2 — noise ablation.** Largely already produced by the 8-σ E1 sweep; needs the dual-axis figure and the matched-P(det) companion — **and the missing σ = 0 anchor + a proper log grid** (§3.2). | ~½ day | independent of G1 |
| G3 | **Power-budget ablation**, log grid. | ~½ day | independent |
| G4 | **Structure ablation** (§4.2) — iid-uniform → non-uniform priors → correlated → pilots. Decides *where* the learning contribution lives. | ~1 day | independent, but read with G1 |
| G5 | **Conditional generator on M0**, direct gradient, vs the closed-form frontier at matched detectability. | ~3–4 days | after G1 + G4 |
| G6 | **M1 — spatial / multi-jammer.** PettingZoo only if a MARL algorithm is genuinely needed. Justified on the V>1 mechanism (§4.2). | ~4–5 days | after G5 |
| G7 | **Adaptation-cost rounds R0/R1/R2** (RQ2). Tooling exists; in M0 it gains the distance-from-`D_NP` reference E1 already measured. | ~2 days | after G5; can precede G6 if G6 slips |

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

**Rough calendar.** Sep 8–14: G1–G4 + start the appendix. Sep 15–21: G5 + system-model repair.
Sep 22–28: G6/G7 + results write-up. Sep 29–Oct 2: lock results, figures, polish.
**Results lock Sep 28** — leave four days, not one, given how many headlines on this project have
died to a late-arriving honest metric.

**Risks, in order.** (i) G5 produces no separation from the closed form — mitigated, because G1 tells
us in advance whether separation is even *possible*, so this is a known outcome rather than a
surprise. (ii) M1's motivation is thinner post-E1 than when the proposal was written; if G1 returns
≈0 headroom, M1 rests on the V>1 mechanism alone and the multi-user extension is new code.
(iii) Scope creep against his explicit "simplify, as much as possible" and the 2–3-experiment quota —
**the extra 17 days are permission to do the *planned* work properly, not permission to add layers.**

## 3.5 The Overleaf appendix — "Experiment History"

**State (2026-09-10): 1–4 in Overleaf with fixes owed · 1b, 5, 10 drafted in `paper_drafts/`, not
pasted · 6, 7, 8, 9 unwritten.** Live ref last read: `overleaf/main` @ `13adaf5` "push latest".
Reviewed 2026-09-10. It belongs in the thesis (§1.4), and writing it is directly responsive to a
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
of Policy-Gradient RL over Raw IQ (sim06 jammer, 06b, 07) · 7. Detector Characterisation and Errata
(Phase 0, 0.5, recheck) · 8. Realistic Channel, Suite, and Matched Detectability (sim08 m1, m2,
dense) · 9. Summary of Hypotheses and Verdicts (one table: hypothesis · evidence · verdict ·
superseded-by) · 10. **What the History Determines** — the forward-facing close, added 2026-09-10:
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
- **New in `13adaf5`, and wrong as written** — the two sentences added to A.3 ("this approach might
  not work in a realistic scenario, as a jammer wouldn't have direct access to a detectors channel. At
  the other hand it could try to predict it"). The instinct is right and the caveat belongs there, but
  (i) the assumption is misnamed — direct backprop needs a *differentiable white-box model* of the
  receiver chain and the statistic, plus **genie access to `tx[t]`**, and it is the second one that
  fails on a real link; (ii) **"it could try to predict it" is falsified two subsections later** — A.6
  moves the observation to `tx[t−1]`, which for iid QPSK carries **zero** information about `tx[t]`,
  so prediction is impossible, not merely hard; (iii) register — "might not work" is a hedge where
  convention (d) wants a mechanism. Rewrite as a forward pointer to the black-box setting. Also
  `detectors` → `detector's`, `At the other hand` → `On the other hand`.
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
- **LaTeX:** add `\appendix` before L534 or the section numbers as a regular section; **remove
  `\nocite{*}` (L572 in `13adaf5`) before he reads it** — it emits the whole `refs.bib`.
- Still-live `\rar{}` notes above the appendix: L216, L256.

**A.1b is drafted** — `paper_drafts/sec_exphist_1b_objective.tex`, 2026-09-10, not yet pasted.
Inserts after the scope paragraph, before "Simulation Chain and Validation": two equations, the
definition of κ, and a `table*` of per-step constants. **It ends at the table.** The two findings the
table makes visible were deliberately moved to their point of use — the log barrier to part 6, γ = 0.02
to part 4 — and the "power is a constraint" paragraph was cut as a duplicate of
`sec:system:objective`. A header comment in the file records all three so they are not lost.

**A.5 is drafted** — `paper_drafts/sec_exphist_5_learned_detection.tex`, 2026-09-10, not yet pasted
into Overleaf. Replaces the one-sentence stub and keeps that sentence as its opening line. **418 words
of prose** (vs 244 for the user-written part 4) after a rewrite for concision and register; the
cross-evaluation table is six rows × **two** columns, the third having duplicated the prose. Ends on
three deviations (binary head, ImageNet-pretrained, synthetic frames) plus the real-part-only STFT
erratum, which forward-references part 7 — **if part 7 is dropped or renamed that sentence dangles.**
Needs one `\ref` once the characterisation subsection has a `\label`, and it is the first place
`\cite{9707819}` (Li et al.) is used in `main.tex`.

**Next in this thread:** **part 6 — the untrainability result**, which is the one he asked about
directly, and the one to lead with the *action-parameterisation* explanation, not the reward formula.
It now has a mechanism to lead with as well: the log barrier
([A.5](#a5-sim06-jammer-06b-07-the-untrainability-result)). Then 7, 8, 9.

**Dependency to respect when pasting:** parts 5 and 10 both `\ref` into `sec_system_model.tex`, which
is **itself not in Overleaf yet** (§B.3). Paste the System Model section first or those refs dangle.

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

**Deferred refinements, not on the critical path:** BER-thresholded in-band labels + threshold
calibration for the m2 detector; extending the sim08 suite with more classical detectors
(kurtosis/GLRT/pilot-variance).

---

# PART 4 — OPEN QUESTIONS & IDEAS

## 4.1 Blocking / needs supervisor input

| # | Item | Why it blocks |
|---|---|---|
| 1 | **Supervisor of record for the ETH registration** | D-INFK professor requirement; may need Di Maio as co-supervisor. Needed on the myStudies form. |
| 2 | **Title, start date, end date, task description** | All four gate registration. Open since 2026-09-01. |
| 3 | **Feedback on the proposal** (handed over 2026-09-01) | Nothing downstream is blocked on *us*, but it is the longest-outstanding item. |
| 4 | **Which 2–3 experiments go in the main paper** | His call. Candidates E1 / E2 / E3 (§2.9). |
| 5 | **Single-round evasion on a frozen detector, or fully co-adaptive?** | His 2026-08-03 phrasing leans co-adaptive (*"one can always fine-tune a defender on an attacker and vice versa"*) but was never a direct answer. RQ2 assumes round-based-and-offline. |
| 6 | **Is he comfortable leading with detector characterization as the solid core** and the cooperative learned jammer as the high-upside extension? | Asked in the mid-July email; never answered directly. |
| 7 | **His September availability / feedback cadence** | Asked twice, still unanswered. Short frequent rounds >> one large end-of-block review. |
| 8 | **Inter-jammer coordination assumption** — shared backhaul / shared clock only / fully independent? | Needed to finish the Threat Model. Good candidate to decide together rather than guess. |

**When his corrections come back — the four things most likely to be challenged**, recorded so the
reasoning does not have to be reconstructed:
1. **The gap claim** (§2.2 point 5) — the load-bearing sentence of the Introduction. Defend it as the
   *conjunction*, and name the nearest neighbours.
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
   asked.) The honest answer is yes, and that is fine **as long as the paper leads with the problem
   and the findings, not the architecture.** What justifies it: the systematic **ablation trail**
   showing *why* each simpler alternative structurally fails (Gaussian → GMM → GAN → flow, each with
   a mechanism, not a preference); the **problem formulation** itself; and whatever the attacker
   actually discovers. Comparable precedent exists at solid venues. The bar is whether the
   combination produces insight the parts alone could not — which is exactly why the negative results
   in [Appendix A](#appendix-a-experiment-history) are load-bearing rather than embarrassing.

## 4.2 Open technical questions

**Q1 — Does M1 still have a target? (the big one; G1 answers it.)**
E1 tested a *menu* of eight hand-written attack laws and found none of the realizable ones works. It
did **not** compute the best possible law. G1 does. Two outcomes, and the deadline move turned this
from a go/no-go into a **sequencing gate** — run it either way, because it determines what the
learner is *for*:
- **Headroom exists** → the generator is chasing a quantified gap, and we can report how much of it a
  learned policy recovers. Strictly better than "our generator got BER X".
- **Headroom ≈ 0** → the generator's job changes from *beating the closed form* to *rediscovering it
  without the genie's information*, and the paper's claim becomes about the **learned vs optimal
  detector gap** (adaptation cost) rather than attack effectiveness. Still a paper; different
  headline. M1 is then justified on the V>1 mechanism only.

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

**Q4 — The scoping fork, still unresolved.** Is multi-agent cooperation the **destination** (the
thesis is about *cooperative* jamming; single-agent signature-shaping is a stepping stone) or the
**garnish** (the thesis is about *learned evasion*; cooperation is an extension)? That decision — not
any further sweep — shapes M1. Note the title as registered says "Cooperative Multi-Agent", which
leans destination.

**Q5 — E1 hygiene, cheap to fix (§3.2, §3.3).** Add the σ = 0 anchor; put the σ grid on a proper log
spacing; re-caption anything quoting the `boundary_genie` row to say it is the ρ = √2 permutation
degeneracy; fix the meaningless `power` column for the `counter_*` attacks.

**Q6 — Related Work: the two long-open inclusion calls are RESOLVED. Include both.**
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

## 4.3 Ideas on the shelf — specified, not adopted

- **Time-to-first-detection.** The cheap countermeasure-facing result: measure the **trigger**
  instead of the reaction. Computable from the per-frame P(det) we already produce — no
  countermeasure, no mobility, no throughput model. It also fixes a known weakness of our own
  reporting (a threshold like P(det) ≤ 0.5 is not operational stealth), turning a hand-written caveat
  into a reported number. Reconsider if a defense-facing result is ever wanted.
- **ACK/NACK as the execution-time observation** (§2.8, CTDE). If any execution-time adaptivity is
  wanted, this is the channel to model — one line in the system model, and if cheap, an
  observation-space ablation (blind vs ACK-aware).
- **Desync / realism axis.** Per-jammer CFO, timing and residual phase error as a "cheap hardware"
  quality level. He *wants* the attacker handicapped: *"introducing some desynchronization … will
  make the attacker more realistic and weaker, which is good for the paper, especially if BER is high
  and detection rate is low."* Only meaningful once the attack is phase-coherent, and it is the
  experiment that substantiates the counter-signal-vs-boundary robustness claim (§2.5).
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
- **2026-09-01** — registration proposal handed over for correction. **Awaiting feedback.**

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
| *"the most interesting investigation will still be the optimal multi-jammer coordination against one or more mobile victims"* | The destination: multi-agent + **victim mobility** (§4.3). |
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

**Proposal** (`proposal/proposal.tex`; **Overleaf is truth**, this file is the reference draft plus the
five-title-option record — it will drift from the submitted version)
- [ ] `\usepackage{xcolor}` — the `\adm`/`\rar` macros use `\color{orange}` and error without it.
- [ ] Fix the Introduction's closing line: it still promises evaluation *"against reactive and
      proactive countermeasures"*, but that RQ was dropped. Must read *"against classical and
      non-cooperative baselines on the effectiveness–detectability plane"*.
- [ ] Minor: spurious comma in "triggered by detection, cannot be"; *"Coordination because…, generative
      because…"* is a sentence fragment — join with a colon or dash.

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
