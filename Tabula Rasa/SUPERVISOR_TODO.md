# Supervisor TODO — A. Di Maio

Exhaustive checklist of **every** actionable point from the supervisor, including the small ones.
Nothing here is paraphrased away — where the wording matters, his words are quoted.

Two sources, both live:
- **Email 2026-07-17** — sections 1–10. Written up 2026-08-03. Reasoning in `README.md` §"Supervisor steer (2026-08-03)".
- **Meeting 2026-08-21** — sections 11–13, plus items tagged **(mtg)** inside sections 1–10. Notes transcribed
  2026-09-01. Reasoning in `README.md` §"Supervisor meeting (2026-08-21)". **Where the two conflict, the
  meeting wins.**

This file is the *checklist*; the README carries the *reasoning*. Keep them in sync.

**Status key:** `[ ]` open · `[x]` done · `[~]` decided, not yet executed · `[?]` needs his input at the meeting

---

## 1 · Meeting & communication — P0, blocks everything

- [x] **Reply to the email.**
- [x] **Hold the meeting — 2026-08-21.** Notes transcribed 2026-09-01 → sections 11–13 below.
  > "I would need to understand better what you have done so far, e.g., during a meeting"
  - [x] Prepare a short summary of work to date (the ladder sim00→sim08, what each falsified).
- [ ] **Explain why the MAPPO jammer could not be trained against the CNN.**
  > "I did not fully get why the MAPPO jammer can't be trained against the CNN detector"
  - Answer: reward variance ≈ 0 across the batch → normalized advantages are noise → no gradient.
    Confirmed structural (not tuning) by reproducing it in a reduced 2D setup (sim06b).
  - [ ] **Write it up as a result, not an excuse** — he explicitly accepts it:
    > "showing in what cases it is hard to beat is already a small result"
    - [ ] Characterize *in which cases* it is hard to beat (which detectors / regimes), not just "it failed".
- [ ] **[?] Who is the supervisor of record for the ETH registration?** Needed for the myStudies form.
- [ ] **[?] Agree title, start date, end date, and task description** → gates registration. **Still open if the
      meeting did not settle it — this blocks registration, so chase it.**
- [ ] **[?] Ask about his September availability** (1–14 Sep is the paper crunch).
- [ ] **(mtg) [?] Confirm which 2–3 experiments he wants in the main paper** (see §12).

---

## 2 · Paper scope & structure

- [ ] **Triage every result into main paper / appendix / cut.**
  > "we will probably include a subset of those results in the final manuscript, but we can still
  > include those complementary results in the appendix"
  - [ ] Produce an explicit table: each result → main / appendix / dropped.
  - [~] **(mtg) HARD QUOTA: only the 2–3 strongest experiments go in the main paper**, everything else to the
        appendix. Tighter than the email's "a subset". Candidates in §12.
  - [~] Working assumption: **the characterization arc goes to the appendix** — Phase 0, Phase 0.5,
        the lossless recheck, sim08 m1, m2, matched-detectability.
  - [~] Main paper leads with the **adaptation-cost** contribution.
  - [ ] **[?] Confirm this split with him** — it is his call which subset he wants up front.
- [~] **Stop running further characterization sweeps.** They are appendix material and finished;
      more of them is the main way to waste the remaining hours.
      - [~] **(mtg) Stronger now:** stop extending sim06/07/08 in *any* direction. Further complexity is the
            opposite of his stated priority (§11).
- [ ] **Formulate the system model, the defender model and the threat model — clearly and separately.**
  > "it is important to clearly formulate the system model and both the defender and thread models"
  - He raised this **twice** → treat as the top written deliverable. Details in section 3.

---

## 3 · System / threat model — P0, he asked twice

- [ ] **Fix where detection takes place.**
  > "Fixing where the detection takes place and what information is available to the detectors is also important"
  - Currently: at the victim RX, on the composite pre-equalization time-domain frame.
  - [ ] State it explicitly in the System Model section rather than leaving it implicit in the code.
- [ ] **Fix what information the detectors have.**
  - Currently: complex two-sided STFT spectrogram + mean frame power. No CSI. No labels at runtime.
- [ ] **State the no-online-adaptation constraint.**
  > "Mind that this can only happen at training time: there are no ground-truth labels at execution
  > time to fine tune."
  - Justifies evaluating against a **frozen** detector; the arms race is round-based and offline.
- [ ] **Enumerate what the jammers must know, and make it an explicit assumption table.**
  > "requires some information on the victim position, propagation time between jammer(s) and the
  > victim, and the channel response between jammers and victim. We should make some assumptions
  > here, which will shape what information is available to the honest and malicious generative policies."
  - [ ] victim position · [ ] propagation delay · [ ] jammer→victim channel response
  - [ ] Define assumption **tiers**: genie / realistic (sync error, partial CSI) / blind.
  - [ ] Cover the **honest** policy's information set too, not only the malicious one.
- [ ] **Model preamble synchronization.**
  > "the jammers need to synchronize with the victim's preamble, to know how to phase-shift their
  > jamming signal to induce the interference at the correct symbol"
- [ ] **(mtg) State the CTDE split explicitly: the jammer is DEAF to its own reward at execution.**
  > "At decentralized execution: jammer is 'deaf' to rewards, maybe ACKs"
  - BER is available to the centralized critic at **training** time only; a deployed jammer cannot measure the
    victim's BER. This is the attacker-side mirror of his defender-side "no labels at execution time".
  - [ ] Restrict the executed policy's observation space to own waveform + own channel estimate (+ optional
        **ACK/NACK**). Not BER, not P(detect).
  - [ ] Decide whether to model ACK/NACK at all; if yes, add a blind-vs-ACK-aware observation ablation.
- [ ] **(mtg) Write the counter-signal non-viability argument into the threat model.**
  > "How is 'counter signal' not viable: add vector in random direction in I/Q plot"
  - Needs per-symbol X, exact amplitude+phase of both channels, and sample-level sync; degrades ungracefully
    under CFO/timing error (an exact-inverse operation), unlike the boundary attack which only needs the right
    half-plane. See §6 for the baseline side of the same item.
- [ ] **(mtg) Audit the iid-uniform symbol assumption** (see §13 — it is a framing risk, not a detail).
- [ ] **Model inter-jammer coordination and say what it costs in hardware terms.**
  > "In the more sophisticated coordinated attack, jammers should also coordinate among themselves.
  > This might not be trivial from the hardware perspective"
  - [ ] Shared clock? Backhaul link? Nothing? — pick one and defend it.

---

## 4 · Reward & training setup

- [~] **Attacker reward = `BER − beta·detections`. Nothing else.**
  > "the most agnostic reward for the attacker is BER - beta*detections"
- [ ] **Delete every proxy reward term.** sim01–04 all carry idle penalties, power penalties, kurtosis
      penalties.
  > "Rewards functions can be designed to maximize the end goal directly and not contain intermediate
  > proxy metrics that are expected to maximize the end goal."
- [ ] **Move everything else into the environment.**
  > "The other aspects should not be relevant for the reward and be controlled by the environment."
  - [ ] Power budget becomes a **hard action-space / environment constraint**, not a reward penalty.
- [ ] **Formulate it explicitly as a zero-sum game** between detectors and jammers.
- [ ] **Look at the GAN literature and see whether it transfers.**
  > "The setup reminds a bit of GANs, so you may have a look at that to see if this old topic could be
  > revived here. Somewhat related to the 'co-adaptive' strategy you mentioned."
  - [ ] Short lit check; if it transfers, cite it in related work; if not, say why in one sentence.
- [ ] **Train attacker and defender jointly.**
  > "I'd suggest training both attacker and defender jointly"
- [ ] **Then pick one side and make life hard for the other.**
  > "then pick one side (e.g., the attacker's should be easier) and make the attacker such that
  > training a good defender is difficult"
  - [~] Pick the **attacker** side, as he suggests.

---

## 5 · The core contribution — adaptation cost

> "The core contribution is to show that this adaptation is very expensive for an attacker or
> defender to conduct."

- [ ] **Make adaptation cost the headline claim** (not "the jammer evades the CNN").
- [ ] **Design the cost measurement.** What counts as "expensive"? Candidates: change in accuracy,
      change in false-alarm rate, training samples required, GPU-hours, how much performance the other
      side recovers.
- [ ] **Round-based protocol:**
  - [ ] **R0** — frozen detector vs the best attacker.
  - [ ] **R1** — retrain the detector on those attacks; measure what it costs.
  - [ ] **R2** — attacker re-optimizes against R1; does it recover, and at what price?
- [x] **Recognize that existing results already measure this** — reframe, don't rerun:
  - Phase 0.5: closing the in-band blind spot cost acc 99.8→90.5%, FAR 0→3.8%.
  - sim08 m2: channel-valid retrain pays ~20% FAR at 5 dB.
  - matched-detectability: the attacker's channel-aware lever yields about 0 gain → already exhausted.
  - sim06/06b/07: black-box RL over raw IQ is structurally untrainable → that route to adaptation is
    prohibitively expensive. **This turns the dead end into evidence.**

---

## 6 · Attacker design — the energy-optimal (boundary) attack

> "the shapes do not seem the most energy-optimal alteration of symbols … we could expect most of the
> points under attack to be located around the symbol classification boundary perpendicular to the
> symbols' prototypes, because that is the minimal-energy alteration"

- [ ] **Implement the minimum-energy boundary attack.** Closed-form, no training — push each symbol
      just across its nearest decision boundary instead of a random-phase shove.
- [ ] **Add symbol error rate (SER) as a metric alongside BER.**
  > "(symbol error rate could also be a possible metric)"
- [ ] **Re-plot the IQ scatter under attack** and check the points now cluster at the boundary — this is
      the visual he will look for.
- [ ] **Compare against blind / channel-aware at matched `P(suite)`**, using the existing axis.
- [~] **Continue with coordinated, phase-shifted malicious symbol shaping** — he confirmed the direction.
  > "Shaping the coordinated, phase-shifted malicious symbols from the attacker was the original
  > project's direction so it is good you proceed with it."
- [ ] **(mtg) Add the counter signal (`jam = −H0*X`, `−2H0*X`) as the "impossible to beat" CEILING baseline.**
  > "Impossible to beat baseline: omniscient jammer show in results"
  - It must appear **in the results figures**, not only in prose.
- [ ] **(mtg) Add the naive random-direction I/Q vector as the FLOOR baseline** — the thing the boundary attack
      has to beat. (See §8 for the full envelope.)
- [ ] **Consider a multi-subcarrier joint attack.**
  > "A well-crafted adversarial signal could also disrupt multiple subcarriers simultaneously"
- [ ] **Also consider the simpler per-subcarrier isolated problem** — he says it could be interesting
      on its own.
- [x] **Treat subcarrier selection as a proxy, not the goal.**
  > "selecting the optimal subcarrier is a proxy problem on the way to the true problem of maximizing
  > BER while minimizing detection probability"
  - Already confirmed independently by the matched-detectability result. Optimize the true objective.

---

## 7 · Realism — make the attacker weaker on purpose

> "Introducing some desynchronization due to cheap hardware will make the attacker more realistic and
> weaker, which is good for the paper, especially if BER is high and detection rate is low."

- [ ] **Add desynchronization to the jammers:** carrier frequency offset, timing offset, residual phase
      error — parameterized as a "cheap hardware" quality level.
- [ ] **Sweep performance vs desync level.**
- [ ] **Target the result he named:** high BER *and* low detection rate under realistic impairment.
- [ ] **Note the ordering:** desync only bites on a phase-coherent attack, so it must come *after* the
      boundary attack (section 6). A random-phase jammer is indifferent to phase error.
- [ ] **Tuning protocol — relax assumptions if results look too good.**
  > "If performance becomes too extreme (e.g., always stealth, high BER) then relax assumptions on the
  > scenario to make it more realistic until the performance gap between your method and the baselines
  > … increases."

---

## 8 · Baselines & evaluation

- [ ] **Run the baselines for comparison.** He put this in parentheses as an assumption, so it is not
      optional: "(which should also be ran for comparison)".
  - [ ] classical jammers · [ ] blind · [ ] genie channel-aware · [ ] the learned/boundary attack
- [ ] **(mtg) Fix the baseline envelope and put ALL of it on every results figure:**
  - [ ] **no attacker** (BER/SER floor *and* detector FAR — the lower reference on both axes)
  - [ ] **random-direction I/Q vector** (no knowledge — the naive floor)
  - [ ] **boundary / min-energy** (symbol + channel — closed form, no training)
  - [ ] **counter signal** (symbol + channel + perfect sync — the ceiling)
  - [ ] **learned / coordinated** (the proposed method)
- [ ] **Report the gap between our method and the baselines** — that gap is the tuning target in section 7.

---

## 9 · Multi-agent — the destination

- [ ] **Use a multi-agent-native library.**
  > "For multi-agent environments you could consider using multi-agent-native libraries."
  - [ ] Evaluate **PettingZoo** (environment API).
  - [ ] Evaluate **BenchMARL** (only if an off-the-shelf MARL algorithm is needed).
  - [x] **Do not use RLlib.**
    > "RLlib is famous for being too complex for what we need, so I would avoid it."
    Confirmed 2026-08-03: RLlib/Ray appear nowhere in the repo. Current stack is SB3 + gymnasium + zuko.
- [ ] **Optimal multi-jammer coordination — the investigation he is most interested in.**
  > "The most interesting investigation will still be the optimal multi-jammer coordination against one
  > or more mobile victims."
  - [ ] Multiple coordinated jammers.
  - [ ] **Victim mobility** — one or more *mobile* victims. Not modelled at all yet.
  - [~] **(mtg) Reachable much sooner now:** the spatial step is the *first* extension of the minimal model
        (§11, M1), not the last item of the OFDM stack. Coordination becomes the headline learning
        contribution if §13's structure ablation comes out negative.

---

## 10 · Parked / optional — raised by him, out of scope for now

- [ ] **Jammer localization as a detection modality.**
  > "Besides detecting that jamming occurred … a form of detection is to leak information on the
  > position of the jammer(s) so that a defender can physically neutralize them."
  - Name it in the threat model as an out-of-scope defender capability + future work.
- [ ] **Real hardware experiments.**
  > "Real hardware to implement this method is available, if you'd like to experiment later on."
  - Mention in future work; **[?] worth asking at the meeting what hardware, in case it is cheap
    to get a small validation.**
- [ ] **Scope caveat to state plainly:** this is a single-round attack on a frozen detector, not a full
      adaptive arms race.

---

---

## 11 · Simplify — the minimal-model ladder (mtg) — P0, this reorders everything

> "Priority: simplify, as much as possible" · "Single subcarrier" · "One channel" ·
> "First thing to add: spatial, after solving single, no noise, no prop"

His reasoning, made explicit in the meeting: **the simple simulations already do not work, so the
complicated ones certainly will not.** This is not a new direction — it is an insistence on the one the
ladder claimed to follow. Full spec table in `README.md` §"Supervisor meeting" (A).

- [ ] **Build M0 — the minimal model.** Single subcarrier (no OFDM grid, no IDFT, no guard/DC/pilot bins),
      one channel realization (start at `h = 1`), one jammer, QPSK, hard power budget.
  - [ ] **AWGN with σ as the primary swept axis** — noise is *not* optional here: at σ = 0 the clean
        constellation is four exact points, so any perturbation is detected with probability 1 and there is
        no stealth problem to study.
  - [ ] Metrics: **BER and SER** (SER is finally natural per-symbol), P(detect) at a fixed FAR.
  - [ ] Do **not** import the sim06/08 OFDM/fading machinery. Reuse the `build_jam` *structure* only.
  - [ ] Runs on CPU → the 1-GPU-job concurrency cap stops being the bottleneck.
- [ ] **Implement the detector trio for M0:** energy meter · learned classifier on the received IQ scatter ·
      **the Neyman–Pearson optimal test**.
  - [ ] **The NP test is the reason the retreat pays off.** With one subcarrier, known σ and a stated
        perturbation model, the optimal detector is computable — so P(detect) can be reported against an
        *optimal* defender instead of "whatever the CNN happened to learn". Upgrades every "the CNN is blind
        to X" claim into "**no** detector can do better than Y", and gives the adaptation-cost headline (§5)
        the reference point it currently lacks.
- [ ] **M1 — the spatial step, and the ONLY sanctioned extension after M0.** N_J ≥ 2 jammers with per-link
      gain/phase superposing at the RX. No propagation delay, no path-loss law.
  - [x] **"No noise, no prop" is literal** (confirmed 2026-09-01): σ = 0 and no propagation delay for the
        simplest case. Resolution: build exactly that, then make σ the sweep with σ = 0 as its anchor (§13).
- [ ] **Rewrite `main.tex` §System Model around M0**, with OFDM / fading / multi-antenna as *extensions*.
      This also repairs a live mismatch: the section currently describes a full K-subcarrier, TDL, N_J-jammer
      system that **no working experiment supports**.
- [ ] **Freeze sim06/07/08.** Appendix material. No further sweeps in any direction.

---

## 12 · Paper: the 2–3 experiment quota, the intro, and Overleaf (mtg)

> "Experiments 2,3 strongest add" (= put only the 2–3 strongest experiments in the main paper) ·
> "Put as much info as possible in overleaf"

- [ ] **Pick the 2–3 main-paper experiments.** Candidates, to confirm with him:
  - **E1** — the M0 trade-off frontier with the full baseline envelope (§8) in the dual-axis format (below).
  - **E2** — the noise-level ablation (§13).
  - **E3** — the spatial / multi-jammer coordination result (M1, §11).
  - [ ] **[?] Confirm the choice with him** — it is his call.
- [ ] **Everything else → appendix.** The whole sim06→08 characterization arc, as already agreed in July.
- [ ] **Intro: scope out bit-error recovery, then motivate the metric.**
  > "Intro: bit-error recovery 'out-of-scope' → motivate importance, we assume it's handled by another model"
  - [ ] State plainly that FEC / ARQ / retransmission is out of scope and assumed handled by a higher layer.
  - [ ] **Then motivate why raw BER/SER is still the right target** — it is the input any recovery layer
        receives, and pushing it past the code's correcting capability is what becomes outage.
- [ ] **The dual-axis trade-off figure — he asked for this format specifically.**
  > "Possibly double axes, BER/detection → show trade-off; no attackers / ground truth attacker"
  - [ ] One panel: BER (and SER) on the left axis, P(detect) on the right, against the swept parameter.
  - [ ] Draw in the **no-attacker** and **ground-truth/omniscient-attacker** references.
  - [ ] Keep the parametric **BER-vs-P(det) frontier** plot as the companion — the dual-axis view is what he
        wants to read, the frontier view is what supports matched-detectability comparisons. Produce both.
- [ ] **Move working material into Overleaf now, as stubs if necessary:** the assumption table (§3), the
      baseline table (§8), the ablation list (§13). He wants the document to *be* the working record.
  - Note: `paper/` in this repo is **read-only** (pull to read, never push) — these edits happen in Overleaf.

---

## 13 · Ablations, parameter studies, and the assumption that could sink the framing (mtg)

> "Ablation: parameter study, increase noise and see what happens (less detection e.g.)" ·
> "Noise level, *epsilon* change exponentially" · "Scenario, e.g. #jammers, #legitimate users"

- [ ] **Noise-level study — the primary one, and it carries the headline claim.** (Confirmed: the "epsilon"
      in the notes *is* the noise level.) Full design in `README.md` §F1.
  - [ ] **Start at σ = 0** — the simplest case he asked for — then a **log grid** up to σ ≈ 0.5. σ = 0 is an
        anchor point, not an operating point.
  - [ ] **Run every baseline (§8) at every noise level.** Target claim: *across the whole noise range, our
        attacker gets higher BER/SER and/or lower detection than the baselines.* A claim along a curve, not
        at a hand-picked point.
  - [ ] **His prediction to test:** detection rate falls as noise rises.
  - [ ] **Say plainly what happens at σ = 0:** four exact constellation points ⇒ an optimal detector flags any
        perturbation w.p. 1 ⇒ the "less detected" half of the claim cannot hold for *anyone*, and the
        comparison reduces to BER/SER at matched power. This is the cleanest demonstration that **stealth is a
        noise phenomenon** — and it matches the 2026-07-03 recheck, where the energy detector demolished
        stealth on the noiseless channel because clean power was a razor-sharp constant.
  - [ ] **Free calibration point:** any gap between the learned detector and the NP-optimal one at σ = 0 is
        pure detector suboptimality → feeds the adaptation-cost measurement (§5).
  - [ ] **Report at matched P(detect) as well as matched power.** Non-negotiable: sim08-m1's "+70%
        channel-aware" evaporated under exactly this check. If the claim survives only the matched-power view,
        it is the m1 mistake repeated.
  - [ ] Expected to reproduce the sim08-m1 stealth region in a model where the mechanism can be **derived**
        rather than merely observed.
  - [ ] Sanity on the range (unit-energy QPSK, σ per real dimension, N₀ = 2σ²): σ = 0.5 → Eb/N₀ ≈ 0 dB;
        σ = 0.1 → ≈14 dB; σ → 0 → ∞. So [0, 0.5] overlaps sim08's 5–30 dB from below and keeps the appendix
        results comparable.
- [ ] **Scenario-size ablation:** number of jammers · **number of legitimate users**.
  - Note: **multiple legitimate users is new** — the model is 1 TX → 1 RX today, so this needs a multi-user
    extension before it can be swept.
- [ ] **Attacker power-budget ablation**, log grid.

### 13.1 · The iid-uniform symbol assumption — a framing risk, not a detail

> "Is it a valid assumption that all legitimate symbols are equally spread? → RL shines when it can 'find'
> something" · "Scrambling makes transmitted sequence look statistically random"

Read together this is one argument: if the payload is iid-uniform over the constellation — **and real
systems scramble precisely to guarantee that** — then there is no payload structure to discover, the
minimum-energy attack is closed form, and **a learner can at best rediscover it.** That would leave
"learned jammer" unmotivated for the single link.

- [ ] **Enumerate the structure that survives scrambling**, and pick which one the learner exploits:
  - [ ] **protocol-deterministic** structure — preamble, pilots, guard/DC nulls, control signalling
        (scrambling does not randomize these; this is exactly the "protocol-aware attack" the Intro already
        claims — and an argument for keeping *pilots* in M0);
  - [ ] **the detector's decision surface** — signature-shaping searches the defender's model, not the payload;
  - [ ] **channel / geometry** — per-link gains and phases (the M1 spatial step);
  - [ ] **coordination** — how N_J jammers split power and phase; no closed form, a genuine joint optimization.
- [ ] **Run the structure ablation** — sweep the amount of exploitable structure (iid-uniform → non-uniform
      symbol priors → correlated/unscrambled sequence → pilots present) and show the learned attacker's
      advantage over the closed-form boundary attack **appear as structure appears**. Answers his question
      with a curve instead of a paragraph.
- [ ] **Do this BEFORE more learner engineering** — it decides *where* the learning contribution lives.
- [ ] **Record the honest outcome either way:** if no advantage appears, the single-link case is solved by the
      closed form and the learning contribution lives entirely in coordination (M1) — which is also the
      destination he cares most about (§9).

---

## Priority for the time actually available

Rewritten 2026-09-01 after the 2026-08-21 meeting. The ordering principle changed: it is no longer "value ÷ hours over
the existing codebase" but **"what does the minimal model need"**. M0 runs on CPU, so cluster concurrency is
no longer the constraint — writing time is.

| When | Items |
|---|---|
| **Now** | §11 build M0 (+ the NP-optimal detector) · §3 the System & Threat Model write-up, now around M0 |
| **Next** | §8 the baseline envelope + §12 the dual-axis figure (**E1**) · §13 the noise ablation (**E2**) |
| **Then** | §13.1 the structure ablation — *does the learner have a job?* This gates everything downstream |
| **1–14 Sep (paper crunch)** | §12 pick the 2–3 main experiments and move material into Overleaf · §5 adaptation-cost rounds in M0 · writing |
| **After** | §11 M1 spatial (**E3**) · §7 desync · §13 scenario-size (needs multi-user first) |
| **Stretch** | §9 multi-agent + victim mobility · §10 hardware |

**Superseded:** the 2026-08-03 date-keyed table (13 Aug meeting → 19–20 Aug boundary attack → …). The meeting
happened; the boundary attack now lands inside M0 (§11) rather than inside `simulation08/frontier_channel.py`.
