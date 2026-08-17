# Supervisor TODO — A. Di Maio, email 2026-07-17

Exhaustive checklist of **every** actionable point in the feedback email, including the small ones.
Nothing here is paraphrased away — where the wording matters, his words are quoted.

Written 2026-08-03. Companion to the "Supervisor steer (2026-08-03)" section in `README.md`
(which has the *reasoning*); this file is the *checklist*.

**Status key:** `[ ]` open · `[x]` done · `[~]` decided, not yet executed · `[?]` needs his input at the meeting

---

## 1 · Meeting & communication — P0, blocks everything

- [x] **Reply to the email.** Sent 2026-08-17 (delayed by the first exam block — acknowledged in the
      reply). Proposed meeting week of 17–21 Aug, stated availability through submission, gave short
      answers to each point (MAPPO explanation, reward agreement, boundary-attack interest, adaptation-
      cost interest), and asked about his September availability for short/frequent feedback rounds.
- [x] **Hold the meeting — confirmed Fri 21 Aug** (time TBD). *(Supersedes the "Thu 13 Aug" placeholder
      below, which was this file's own draft proposal, not an agreed slot.)*
  > "I would need to understand better what you have done so far, e.g., during a meeting"
  - [ ] Prepare a short summary of work to date (the ladder sim00→sim08, what each falsified). See the
        "Meeting agenda (21 Aug)" section in `README.md` and the synced Artifact.
- [x] **Explain why the MAPPO jammer could not be trained against the CNN.** — answer drafted and sent:
  > "I did not fully get why the MAPPO jammer can't be trained against the CNN detector"
  - Answer given: reward barely varies across a training batch → normalized advantages are ≈noise → no
    consistent gradient direction. Confirmed structural (not tuning) by reproducing the same failure in
    a reduced 2D setup (sim06b). **Refined conclusion (2026-08-17 reply): the root cause is the *action
    parameterization* — raw IQ, very high-dimensional — not the reward formula itself.** This is the
    stronger, more precise framing to lead with at the meeting; it also directly motivates why the
    boundary attack / signature-shaping (low-dimensional action space) is the right next move, not a
    reward-engineering fix.
  - [ ] **Write it up as a result, not an excuse** — he explicitly accepts it:
    > "showing in what cases it is hard to beat is already a small result"
    - [ ] Characterize *in which cases* it is hard to beat (which detectors / regimes), not just "it failed".
- [ ] **[?] Who is the supervisor of record for the ETH registration?** Needed for the myStudies form —
      raise at the 21 Aug meeting; registration was targeted for the week of 17 Aug in the reply.
- [ ] **[?] Agree title, start date, end date, and task description** at the meeting → gates registration.
- [ ] **[?] His September availability** — asked directly in the 2026-08-17 reply (short, frequent
      feedback rounds vs. one large review); answer pending.

---

## 2 · Paper scope & structure

- [ ] **Triage every result into main paper / appendix / cut.**
  > "we will probably include a subset of those results in the final manuscript, but we can still
  > include those complementary results in the appendix"
  - [ ] Produce an explicit table: each result → main / appendix / dropped.
  - [~] Working assumption: **the characterization arc goes to the appendix** — Phase 0, Phase 0.5,
        the lossless recheck, sim08 m1, m2, matched-detectability.
  - [~] Main paper leads with the **adaptation-cost** contribution.
  - [ ] **[?] Confirm this split with him** — it is his call which subset he wants up front.
- [~] **Stop running further characterization sweeps.** They are appendix material and finished;
      more of them is the main way to waste the remaining hours.
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

- [ ] **Schedule risk (2026-08-17): the cluster is down for maintenance the week of 7 Sept**, cutting
      the compute window from Sep 1–8 to **Sep 1–6** and moving "core locked" two days earlier. R0/R1
      are largely already banked (see below); the genuinely new compute is the R2 round using the
      boundary-attack (§6) results as input, plus the boundary attack itself — schedule that
      immediately after the boundary attack and before the desync sweep, so it isn't the last thing
      attempted before the blackout. See `README.md` → "Cluster maintenance, week of 7 Sept".
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

## Priority for the time actually available

| When | Items |
|---|---|
| ~~Now (studying)~~ | ~~section 1 — the reply, nothing else~~ — done 2026-08-17 |
| **17–19 Aug** | section 2 research review (Related Works) — reprioritized ahead of §III, no cluster access this week |
| **19–21 Aug** | section 3 entire (System Model — Detector/Threat Model drafted 17 Aug, coordination assumption + goal paragraph open) |
| **Fri 21 Aug, meeting** | section 1 all · section 2 scope split · section 3 open questions · section 10 hardware question · September cadence answer |
| **1–6 Sep (cluster, 6 days only — down for maintenance from the 7th)** | section 4 write-up (no cluster needed, can slot anywhere) · section 6 boundary attack + SER (do first, cheap) · section 5 adaptation-cost R2 round (do second — highest-risk, must not be last) · section 8 baselines · section 7 desync (lowest priority, first to cut if the week runs short) — **core locked Sep 6** |
| **7–14 Sep (no cluster — maintenance week)** | writing · revision · figures · polish · buffer — no new experiments possible |
| **After Sep 15** | section 9 multi-agent + mobility · section 10 |
