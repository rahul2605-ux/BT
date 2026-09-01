# Related Work — publication-ready draft

> **What this is.** A complete `\section{Related Work}` for the paper, written to the
> *current* framing (effectiveness–detectability frontier + cost of adaptation), not the
> superseded "cooperative MARL jammer fools a CNN" framing that `related_works_draft.tex`
> and `paper/README.md` still carry. Prose below is meant to go into `main.tex` essentially
> as-is; citation numbers `[n]` map 1:1 to `Sources_And_Evaluation.md`, which also contains a
> ready-to-paste BibTeX block with the matching keys.
>
> Length: ~1,900 words + one `table*` ≈ 2.2 IEEE columns. See "Trimming to 1.5 columns"
> at the end for the exact cut order if the page budget bites.

---

## Related Work

Jamming research has moved, on both sides of the contest, from fixed heuristics to learned
policies. Recent surveys document this shift for attack and defense alike [1], [2], and a
parallel literature reviews the security of the learning components themselves [3]. The
shift changes the attacker's problem statement. Against a receiver whose only defense is a
power threshold, the attacker solves *maximize disruption subject to a power budget*; against
a receiver monitored by a trained classifier, the binding constraint is no longer power but
*detectability*, and the two are not interchangeable — a jammer can be quiet and conspicuous,
or loud and invisible, depending on what the classifier keys on. We organize prior work along
the axes this distinction opens up: classical and protocol-aware jamming with explicit energy
optimality; learning-based and cooperative jamming; learned jamming detection; adversarial
machine learning at the physical layer; stealth as a detection-theoretic constraint; and the
cost of adaptation in an attacker–defender arms race.

### Classical and protocol-aware jamming

The classical treatment of jamming pairs a disruption model with a detection statistic. Xu
*et al.* [4] established the canonical measurement side of this — packet-delivery-ratio,
signal-strength and location-consistency tests for deciding that jamming occurred — and
showed that naive detectors confuse jamming with congestion. On the attack side, OFDM's
deterministic structure invites surgical attacks that are far cheaper than barrage noise:
pilot jamming and pilot nulling reach a target error rate at a small fraction of barrage
power [6], attacks on timing synchronization and signal acquisition deny the link before
demodulation begins [7], and a tutorial treatment collects these alongside their
countermeasures [5]. The same logic extends to MIMO, where experimentally validated attacks
exploit channel-estimation and spatial-multiplexing structure rather than raw power [9].

Closest to our own attacker construction is the energy-optimal line of work. Amuru and
Buehrer [8] derive, in closed form, the jamming signal that maximizes error probability
against a given amplitude-phase constellation over AWGN, and show the non-obvious result that
matching the jammer's signal to the victim's is generally *not* optimal. That analysis is the
direct ancestor of the minimum-energy, decision-boundary-directed perturbation we evaluate.
The distinction is what the optimum is optimal *for*: [5]–[9] all price an attack in transmit
energy against a receiver, with detectability either absent from the model or reduced to a
radiometric threshold. None of them evaluates the attack against a learned detector, and none
reports the achievable BER as a function of a detection-probability budget — which is the
quantity that decides whether an efficient attack is also a survivable one.

### Learning-based and cooperative jamming

Learning entered the attacker first as online optimization over a discrete parameter set.
Amuru *et al.* [10] cast jammer configuration — signaling scheme, power level, pulse duration
— as a multi-armed bandit and prove convergence to the optimal arm in error rate and energy.
Erpek *et al.* [11] frame jamming as adversarial machine learning: the jammer trains a
classifier to predict which transmissions will succeed and jams only those, which is markedly
more effective than random or sensing-based jamming, and they pair it with a defense that
deliberately poisons the jammer's training data. Cooperative variants follow the same
template at team scale. Zhang and Wu [12] coordinate multiple jammers against a DRL-based
frequency-hopping link as a one-leader multi-follower Stackelberg game; Valianti *et al.* [13]
train cooperative agents that jointly optimize mobility and transmit power to jam rogue
drones; Qin *et al.* [14] study heterogeneous multi-agent adversarial games with per-agent
hardware asymmetry. Symmetric work exists on defense, from coordination-learning spectrum
access [15] to QMIX-based swarm resilience against reactive jammers [16] and decentralized
data-driven relocation out of jammed regions [17]. The multi-agent machinery these rest on —
centralized-training/decentralized-execution actor-critics [18], monotonic value
factorization [19], and multi-agent PPO [20] — is mature, as is the tooling for reproducible
multi-agent experiments [21], [22].

Two properties are near-universal in this group and separate it from our setting. First, the
action space is a discrete resource selection — which channel, which power level, where to
fly — so the learned policy optimizes a *proxy* for the real objective rather than the
waveform that causes the errors. Second, the reward contains throughput or error-rate terms
but rarely a detection term, and where a defender exists it is a threshold test or another
resource-selection agent, not a classifier trained on the received waveform. The
consequence is that the effectiveness–detectability trade-off, which is the entire subject of
this paper, does not appear in these formulations.

### Learned jamming detection

The defender we attack comes from a well-developed detection literature. Li *et al.* [23]
classify barrage, protocol-aware, single-tone and successive-pulse jamming in OFDM from
spectrogram images with a CNN, reporting 99.79% accuracy at 0.03% false alarm and
outperforming their own hand-crafted-feature model — the result we replicate as our frozen
defender. Zhang and Krunz [24] target *stealthy* Wi-Fi attacks (preamble, pilot,
interleaving) using continuous-wavelet features and a compact CNN. More recent work varies
the representation and the supervision: supervised CNNs against unsupervised convolutional
autoencoders on 5G spectrograms [25], transformer architectures on 5G UAV links [26], and
variational-autoencoder anomaly detection in MIMO-OFDM ISAC [27]. These learned detectors sit
on top of, rather than replace, the classical energy detector [28], and complement
model-based defenses that null the jammer's spatial subspace in MU-MIMO [29] or detect
deceptive jammers in low-probability-of-intercept links [30].

What this literature does not do is adversarial evaluation. Detectors are scored on held-out
samples drawn from the same jammer taxonomy used in training, and reported as a single
accuracy/false-alarm pair rather than as an operating point on a curve. Consequently three
quantities central to a security claim are unmeasured: how the detector behaves against
interference *optimized against it*, how much of its blind spot a trivial energy detector
already covers (and vice versa), and what extending coverage to a new jammer family costs in
false alarms on clean traffic. Our characterization results answer exactly these three
questions for a replicated state-of-the-art detector.

### Adversarial machine learning at the physical layer

The attack technique we use is adversarial evasion, adapted to a wireless channel. Sadeghi
and Larsson [31] showed that modulation classifiers fall to perturbations far weaker than
classical jamming, including universal, input-agnostic ones. Kim *et al.* [32] establish that
the channel must enter the perturbation design — attacks computed without it fail over the
air. Flowers *et al.* [33] taxonomize attacks by where the adversary sits relative to the
classifier input and argue for bit error rate, not perceptual similarity, as the wireless
success metric; Restuccia *et al.* [34] formalize the joint channel-and-waveform problem and
impose the constraint that the perturbed waveform must remain decodable. Two works are
conceptually closest to our stealth mechanism. Hameed *et al.* [35] perturb a transmitter's
own symbols to defeat an eavesdropping modulation classifier while preserving decodability at
the intended receiver — a dual objective in the same spirit as ours. DelVecchio *et al.* [36]
add a spectral-deception loss so that the adversarial signal *looks* spectrally legitimate,
which is the closest published analogue of shaping a jammer's spectrogram signature. Where
detector gradients are unavailable, surrogate-model transferability provides the standard
black-box route [37].

The inversion relative to our problem is precise and worth stating plainly. In [31]–[36] the
target is a classifier of the *legitimate* signal, success is a label flip, and the
perturbation is constrained to be small so the victim's own message survives — effectiveness
and evasion point the same way. In our setting the target is a detector's *jammed* label, and
the perturbation must destroy the victim's message while holding the label at *clean*:
effectiveness and evasion are in tension by construction, because the same energy that flips
bits is the energy the detector can see. A small perturbation is therefore not automatically
a successful attack, and the attack's value has to be read off a frontier rather than from an
attack-success rate.

### Stealth as a detection-theoretic constraint

The correct formal language for the stealth half is covert communication. Bash *et al.* [38]
established the square-root law, framing covertness as a constraint on a warden's
detection-error probability rather than a heuristic power cap — the discipline we adopt by
reporting detection probability against the defender's own false-alarm rate. Generative
methods have been applied to this objective: adversarial training over the air produces
spoofing signals statistically indistinguishable from legitimate transmissions [39], and
GAN-based designs allocate cooperative jammers' power to hide a covert transmission from a
monitor [40]. In radar, Ziemann and Metzler [41] learn waveforms that are simultaneously
useful for sensing and statistically indistinguishable from the RF background, which is the
strongest existing validation of the dual-objective generative paradigm we apply to
communications. In all of this work, however, the hidden signal is *benign*: its utility
(rate, sensing performance) does not require harming a third party, and the warden is
typically a radiometer with analytic statistics. Hiding a *disruptive* signal from a learned
detector couples the two objectives through the same physical quantity and admits no
analytical warden.

### The arms race and the cost of adaptation

Because a deployed detector has no ground-truth labels at run time, attacker and defender can
only adapt between rounds, offline — a GAN-like alternation rather than a continuous game.
The machine-learning literature has quantified what each round costs the defender: adversarial
training raises robustness at measurable expense [42], robust generalization demands
substantially more data than standard generalization [43], and robustness trades against
clean accuracy even in simple settings [44]. It has also shown that detection-based defenses
are the most fragile class [45] and that credible evaluation requires attacks adapted to the
specific defense [46]. On the wireless side, the closest cost measurement is continual
learning for jamming mitigation, where the defender must absorb shifting jammer patterns
without catastrophic forgetting [47]. What is missing is a price in *operational* physical-layer
units: false-alarm rate at a given \(E_b/N_0\), bit error rate the attacker recovers after the
defender retrains, and the samples and compute each side spends to get there.

### Research gap

Table I summarizes the positioning. Three gaps follow. (i) *Evaluation*: prior attacks are
scored at matched transmit power or matched configuration, and prior detectors at accuracy on
a fixed jammer taxonomy; neither yields the achievable BER at matched detection probability
against the full defender suite, with the stealth budget pinned to the defender's own clean
false-alarm rate. (ii) *Attack*: no prior work combines physical realizability through the
jammer's own fading channel, minimum-energy placement of the perturbation relative to the
victim's decision boundaries, and an explicit detectability constraint against a *learned*
detector — realizable, effective and stealthy at once, and degraded honestly by residual
carrier-frequency, timing and phase error. (iii) *Framing*: the arms race is reported as
win/loss rather than as a cost curve per adaptation round for both sides.

This paper addresses all three. We characterize a replicated state-of-the-art spectrogram
detector on a realistic fading channel against a detector suite rather than a single
classifier, construct the frontier of achievable BER versus suite detection probability, and
price one round of adaptation on each side. We also report two negative results that the
positive-result bias of this literature leaves unmeasured: black-box policy-gradient learning
over raw IQ actions is structurally untrainable from a scalar frame-level reward, and a genie
channel-aware subcarrier-selection attacker — the motivation for a learned channel-aware
jammer — yields no gain over blind selection once detectability, rather than power, is held
fixed.

---

**TABLE I — Positioning of representative prior work against the axes of this paper.**

| Ref. | Attacker action space | Detector in the loop | Stealth in objective | Realistic channel | Multi-agent | Adaptation cost |
|---|---|---|---|---|---|---|
| Amuru & Buehrer [8] | Jamming waveform (closed form) | None (receiver only) | Energy proxy | AWGN | No | No |
| Amuru *et al.* [10] | Discrete PHY parameter set | None | Energy in objective | AWGN | No | No |
| Erpek *et al.* [11] | Jam / do-not-jam timing | Transmit-decision classifier | Implicit (attack budget) | Yes | No | Partial (defense poisons attacker) |
| Zhang & Wu [12] | Frequency + power (discrete) | None (DRL victim) | No | Yes | Yes | No |
| Valianti *et al.* [13] | Mobility + power level | None | No | Path-loss | Yes | No |
| Abolhassani *et al.* [16] | Channel access (defense side) | Reactive energy threshold | n/a | Yes | Yes | No |
| Li *et al.* [23] | Fixed 4-class taxonomy | Spectrogram CNN | n/a | Over-the-air SDR | No | No |
| Viana *et al.* [26] | Fixed jammer set | Transformer | n/a | Yes | No | No |
| Sadeghi & Larsson [31] | Additive IQ perturbation | Modulation classifier | Perturbation-power cap | Partial | No | No |
| Kim *et al.* [32] | IQ perturbation with CSI | Modulation classifier | Perturbation-power cap | Yes | No | No |
| Hameed *et al.* [35] | Own-symbol perturbation | Modulation classifier | Yes (+ decodability) | Yes | No | No |
| DelVecchio *et al.* [36] | IQ perturbation + spectral loss | RFML classifier | Yes (spectral similarity) | Yes | No | No |
| Wen *et al.* [40] | Jammer power allocation | Analytic warden test | Yes (covertness) | Yes | Yes | No |
| Ziemann & Metzler [41] | Radar waveform (generative) | Learned critic | Yes (LPD) | Radar | No | Partial (GAN alternation) |
| **Proposed** | **Per-subcarrier complex OFDM interference (min-energy, boundary-directed)** | **Learned spectrogram CNN ∨ energy detector, frozen per round** | **Yes — BER at matched suite \(P(\text{det})\), budget = defender clean FAR** | **TDL fading + AWGN, 5–30 dB, with CFO/timing/phase error** | **Yes (CTDE team)** | **Yes — ΔFAR, Δaccuracy, samples/GPU-h per round** |

---

## Notes for the LaTeX version

**Section placement.** Replace all three overlapping blocks in `main.tex` — `\section{Related
Works}` (l. 72), `\section{Literature Review}` (l. 317) and `\section{Old Related Works}`
(l. 393) — with this one section. Keep the CTDE explanation (l. 340) and the
mobility-evasion discussion (l. 346) in a thesis-only appendix as `paper/README.md` planned;
both are cut from the paper.

**Table.** Table I is wide; render as `table*` at the top of a page with
`\footnotesize` and `\renewcommand{\arraystretch}{1.2}`. If it still overflows, drop the
*Multi-agent* column (least discriminating — six rows say "No") before dropping any other.

**Terminology consistency with §III.** The prose uses "detector suite", \(p_{\mathrm{suite}}\)
and "detectability budget \(\beta\)" exactly as defined in the System Model
(`main.tex` §III-D/E), so no glossary drift.

### Trimming to 1.5 columns

Cut in this order; each step is self-contained and costs no citation the gap statement
depends on:

1. Merge *Stealth as a detection-theoretic constraint* into the end of *Adversarial machine
   learning at the physical layer* — keep [38] and [41], drop [39], [40] to the table only.
   (−180 words)
2. Compress *Classical and protocol-aware jamming* to three sentences: OFDM structural
   attacks [5]–[7], [9] in one clause, then Amuru & Buehrer [8] in full — it is the one
   citation the boundary attack must situate itself against. (−150 words)
3. Reduce the MARL-machinery sentence to a single citation group [18]–[20] and drop the
   tooling cites [21], [22] to the Methodology section, where PettingZoo/BenchMARL actually
   belong. (−60 words)
4. Drop [27] and [30] from *Learned jamming detection*; [23]–[26] and [28], [29] already
   establish the representation-and-supervision spread. (−60 words)

Do **not** cut, in any version: [8] (energy-optimal ancestor of the boundary attack), [23]
(the replicated defender), [31], [32], [35], [36] (the evasion-attack lineage the method
belongs to), [38] (the stealth formalism), [42]–[44], [47] (the adaptation-cost framing that
is now the headline).
