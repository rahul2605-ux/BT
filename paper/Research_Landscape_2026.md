# Where the research stands — August 2026

Extension to `Literature_Review.md` and `Sources_And_Evaluation.md`, covering work that appeared
**after** the core literature those two documents are built on. Read this one for *currency and
threat assessment*: what changed in the last ~12 months, which of the thesis's claims it
strengthens, and which of its assumptions it puts under pressure.

**Numbering** uses `[R1]…[R24]` so it never collides with `[1]–[49]` in the other two files.

**Method and health warning.** Searched August 2026. Every entry below was resolved against the
arXiv API (`export.arxiv.org`) or Crossref for exact title, authors and date — no citation from
memory. Because journal review lag is 12–24 months, **2026 is preprint-dominated**: of the 24
entries here, 7 are peer-reviewed and 17 are preprints, each labelled. Treat preprint results as
directional evidence, not as settled numbers. Two consequences for the thesis: (i) cite preprints
sparingly and always marked as such; (ii) the *published* 2026 items are the ones safe to lean on
in a claim.

---

## 1. Executive summary — eight shifts, and what each does to this thesis

| # | Shift | Evidence | What it does to us |
|---|---|---|---|
| 1 | **Detectors became self-adaptive at run time** | SAJD [R1] | ⚠ **Pressures our central modelling premise** ("no ground-truth labels at execution time ⇒ frozen detector"). Survivable, but §III must now argue it rather than assert it. |
| 2 | **Detectors became explicitly adversarially aware** | CITADEL [R2], Wi-Fi CSI robustness study [R16] | ⚠ "Learned detectors are soft targets" is no longer free. Our claim must be scoped to a spectrogram-CNN + energy suite, not learned detectors in general. |
| 3 | **Open-set / zero-day detection is the new frontier** | CITADEL [R2] | ↔ Reframes our residual blind spot: it *is* the closed-set failure mode the field is now attacking. Good for relevance, bad for "nobody has closed this". |
| 4 | **Detector architecture zoo exploded** | Tsetlin machine [R6], transformer [R22], CAE/VAE, federated [R7], multimodal [R20], application-layer transfer learning [R8] | ✔ Validates the *suite* methodology we already use, and supplies cheap external-validity targets. |
| 5 | **Benchmarks and open datasets finally arrived** | OFDMA benchmark [R5], JamShield OTA [R19] | ✔ **A zero-GPU way to add external validity during the Sep 7–14 cluster blackout.** |
| 6 | **Generative/diffusion methods moved to the defence** | Diffusion receivers [R11], JMD [R12] | ⚠ A new defender class that can partly *undo* in-band interference. Name as a limitation before a reviewer does. |
| 7 | **RL-trained models disrupt gradient-based attacks** | Zou *et al.* [R4] | ⚠ Directly relevant to our surrogate-gradient plan: if the defender is RL-trained, transferred gradients degrade. |
| 8 | **Attacker-side detectability is now studied with ROC discipline** | Hazra & Harshan [R3] | ✔ **The strongest external validation of our matched-detectability framing** — an independent group argues undetectability via ROC-at-low-P_FA, exactly as we do. |

**One-line verdict:** the thesis's *question* got more relevant in 2026, its *headline*
(adaptation cost) got stronger, and two of its *assumptions* (frozen detector; learned detectors
are easy to evade) now need explicit defence rather than assertion.

---

## 2. The landscape by theme

### A. Detectors that adapt themselves — the assumption under pressure

**[R1] SAJD — Self-Adaptive Jamming Attack Detection in AI/ML-Integrated 5G O-RAN.**
M. H. Rahman, M. S. Hossen, N. H. Stephenson, V. K. Shah, A. Da Silva. arXiv:2511.17519;
demonstration paper arXiv:2510.09706, IEEE MILCOM 2025. *[preprint + demo]*
A closed-loop O-RAN system: an ML **xApp** does near-real-time jamming inference, while **rApps**
continuously monitor and retrain it. The key piece is a *labeler rApp* that uses live KPI
telemetry to detect model drift, triggers **unsupervised data labelling**, retrains via ClearML,
and hot-swaps the deployed model without service interruption.
- **For us — the important one.** Our System Model justifies a frozen detector with "there are no
  ground-truth labels at execution time" (the supervisor's own words). SAJD is the counter-example
  being built right now. **But read it carefully and the premise survives:** SAJD never obtains
  ground truth — it manufactures *pseudo-labels* from drift heuristics, and the adaptation it
  performs is a *retraining round*, orchestrated by a pipeline, not a within-frame update.
- **Action:** add ~4 sentences to §III-D. Concede that operational systems now automate the
  retraining loop; argue that this makes the round-based model *more* realistic, not less, and
  that our adaptation-cost metrics (samples, GPU-hours, ΔFAR) are precisely the quantities such a
  pipeline consumes. SAJD turns our headline from a thought experiment into a costed engineering
  question — that is a gift, if we cite it correctly.

### B. Detectors that resist evasion — the claim that needs scoping

**[R2] CITADEL — CSI-Based Jamming Detection and Open-Set Classification for IIoT.**
A. Bouferroum, I. Alla, V. Loscri, A. Benslimane, V. Lenders. arXiv:2606.22939, Jun 2026.
*[preprint]*
Two-stage hierarchical pipeline over commodity-device CSI. Claims to be the first to jointly
achieve closed-set classification of known attacks, **open-set detection of zero-day attacks**,
and **resistance to adversarial evasion**. Reported: 6 known attack types, 15 zero-day scenarios,
100% known-attack detection, 97.1% zero-day detection at 0.4% end-to-end false-positive rate;
gradient-based evasion held **below 2%** across tested perturbation budgets, and the strongest
published CSI attack generator achieves **<5%** average evasion; 14.2 ms / 95.9 mJ on an edge GPU.
- **For us.** This is the single most threatening 2026 paper to a naive reading of our stealth
  result — but the threat is narrower than it looks. CITADEL operates on **CSI** from commodity
  802.11-style IIoT devices, not on the composite pre-equalization spectrogram of an OFDM frame;
  its adversarial evaluation concerns **CSI-domain perturbations**, which are a different (and
  more constrained) realizability class than transmitting interference through a fading channel.
- **Action, two parts.** (i) **Scope the claim** in the abstract and conclusion: our stealth
  result is against a *spectrogram-CNN + energy* suite on a fading channel, not against
  detectors in general — say it plainly. (ii) **Use it as motivation**: CITADEL demonstrates that
  the defender-side answer to our attack (open-set + adversarial training) exists and is being
  built, which is exactly why the interesting question is now *what that costs*, not *who wins*.
  It supports the pivot the supervisor asked for.

**[R16] Towards Trustworthy Wi-Fi CSI-based Sensing: Systematic Evaluation of Adversarial
Robustness.** S. K. Gopalakrishnan, S. Hailes. arXiv:2511.20456, Nov 2025. *[preprint]*
Systematic robustness evaluation of CSI models under white-box, black-box/transfer and universal
perturbations, quantifying how model scale, training regime and physical constraints affect
robustness.
- **For us.** Methodological template for the ablation table we owe: report attacks at all three
  access levels rather than at one. Also evidence that *physical constraints*, not model
  capacity, dominate attack success — consistent with our own finding that the power↔detectability
  tension is partly physical.

### C. Attackers whose own detectability is measured — our methodological siblings

**[R3] Cooperative Mitigation against Learning-Based Reactive Jammers: Analysis and SDR
Validation.** S. Hazra, J. Harshan. arXiv:2606.01197, May 2026. *[preprint]*
Introduces a reactive jammer that *also monitors* energy statistics to detect whether the victim
has deployed countermeasures, using **generalized energy detectors** — statistical detectors on
instantaneous and distributional energy metrics *plus* data-driven ML classifiers. The victim's
countermeasure is then evaluated for **its own detectability**, on an SDR testbed, and the
argument is made with **ROC curves at low false-alarm probability**: the classifiers' ROC shows
no pronounced bulge away from the random classifier, so the countermeasure "cannot be detected
with high probability".
- **For us — the best news in this file.** An independent group, on hardware, argues
  undetectability exactly the way we do: not at a convention like \(P(\text{det}) \le 0.5\), but
  by showing the detector cannot reach useful \(P_D\) at low \(P_{FA}\). That is our
  FAR-matched stealth budget under a different name, and it retroactively vindicates the honesty
  correction we made in July. **Cite it in Performance Metrics (§III-E)** as independent
  precedent for the metric — it converts a self-imposed methodological choice into a community
  norm.
- **Differs**: roles are mirrored (their *defender* is the one hiding, from a *jammer's*
  detector); energy-domain statistics, no spectrogram CNN; no BER-vs-detectability frontier.

**[R9] Communication-constrained black-box adversarial attack against OFDM signal detection
network.** Y. Xu, Q. Tang *et al.* *Physical Communication*, vol. 78, p. 103235, Aug 2026.
DOI `10.1016/j.phycom.2026.103235`. *[published]*
Black-box adversarial attack against a deep OFDM **signal-detection** network, with an explicit
communication constraint on the attacker.
- **For us.** The closest *published 2026* work to our threat model: black-box, OFDM,
  constrained. **Differs**: the target is the receiver's signal-detection network (a
  demodulation-chain DNN), so success is decoding failure at a model the victim uses to decode —
  not evasion of a *jamming detector*. Our target's label is "is there a jammer", theirs is "what
  was the symbol". Worth one contrastive sentence in Related Work; it shows the community is
  converging on constrained black-box attacks in OFDM.

### D. The detector zoo — architectures beyond the spectrogram CNN

- **[R6] Convolutional Tsetlin Machine for 5G jamming detection.** V. Halenka, M. Amini,
  P.-A. Andersen, O.-C. Granmo, B. Kantarci. arXiv:2603.07336, Mar 2026. *[preprint]* —
  explainable, hardware-efficient, **non-differentiable by construction**.
  *For us:* a detector with no useful gradient is immune to surrogate-gradient transfer by
  construction. One sentence in Limitations; also an interesting future-work target.
- **[R7] Federated vs centralized RF jamming detection** on SSB IQ samples. S. Kuili, M. Amini,
  B. Kantarci. arXiv:2605.01705, May 2026. *[preprint]* — privacy-driven decentralization of the
  detector.
- **[R8] Jamming Detection at the Application Layer Using Deep Learning: A Transfer Learning
  Approach.** Y. Terraf, Y. Iraqi, A. Al-Dweik, A. Pandey, J.-P. Giacalone. *IEEE Open Journal of
  the Communications Society*, vol. 7, pp. 7302–7318, 2026. DOI `10.1109/OJCOMS.2026.3707737`.
  *[published]* — detection moved **up the stack**, away from the PHY signal entirely.
  *For us:* a cross-layer defender our PHY-only threat model does not cover; name it in the
  threat model as an out-of-scope detection modality, next to localization.
- **[R20] FedJam: Multimodal Federated Learning Framework for Jamming Detection.** I. Panitsas,
  I. Ofeidis, L. Tassiulas. arXiv:2508.09369, Aug 2025. *[preprint]*
- **[R22] PCA-Featured Transformer for Jamming Detection in 5G UAV Networks.** J. Viana *et al.*
  *IEEE OJ-COMS*, vol. 6, pp. 9287–9303, 2025. *[published]* (= `[26]` in the main list) —
  ~88% LoS / ~85% NLoS accuracy, i.e. **realistic-channel detector accuracy is far below the
  99.79% of the idealized benchmark**, which independently corroborates our channel-realism cost
  finding.

### E. Benchmarks and reproducibility — the cheap opportunity

**[R5] Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset.**
A. Schösser, M. Salehi, S. Ma, P. Schulz, G. Fettweis (TU Dresden, Vodafone Chair).
arXiv:2606.02102, Jun 2026; submitted to IEEE OJ-COMS. *[preprint]*
Open-source simulation framework plus a benchmark dataset of spectrograms from distributed
sensing units, five jammer types, industrial-factory environment, with supervised **and
unsupervised** baselines. Explicitly motivated by the observation that only a small fraction of
jamming-detection papers release data, blocking cross-study comparison.
- **For us — act on this.** It is an OFDMA spectrogram dataset with published baselines and it
  needs **no cluster time**. Evaluating our channel-valid detector (or just our energy detector)
  on it during the Sep 7–14 blackout would give the paper an external-validity paragraph that
  costs a laptop afternoon. It also pre-empts "your results are simulation-only" — partially, at
  least, since their data is also simulated but independently generated.

**[R19] JamShield: A Machine Learning Detection System for Over-the-Air Jamming Attacks.**
I. Panitsas, Y. Yigit, L. Tassiulas, L. Maglaras, B. Canberk. IEEE ICC 2025. *[published]* ⚠ an
extended version also appears on IEEE Xplore — check which to cite. Over-the-air dataset, hybrid
feature selection, auto-classification module that switches algorithm by network condition.

### F. Generative and model-based defences — the new limitation

- **[R11] Brownian Bridge Diffusion-Based Joint Channel Estimation and Data Detection for
  Jamming-Resilient Receivers.** H. She, Y. Cheng, T. Sun, P. Wang, S. Huang, K. Yang.
  arXiv:2606.28778, Jun 2026. *[preprint]* — diffusion receivers that project jammed observations
  back onto the learned clean-signal manifold.
  *For us:* **this is the sharpest new limitation.** Our threat model assumes the receiver
  equalizes and demaps conventionally; a diffusion receiver could partially *remove* the very
  in-band interference our stealthy jammer relies on, decoupling "undetected" from "effective".
  One paragraph in Limitations/Future Work.
- **[R12] Joint Jammer Mitigation and Data Detection (JMD).** G. Marti, C. Studer (ETH).
  arXiv:2510.02021, Oct 2025. *[preprint]* — removes the dedicated jammer-training phase by
  estimating and removing the jammer subspace jointly with data detection, handling smart and
  dynamic multi-antenna jammers. Successor to their TSP 2023 work (`[29]` in the main list).
  *For us:* reinforces the multi-antenna scope limitation. Also politically relevant: this is ETH
  work, so a reader at ETH will know it.
- **[R10] Deep Learning-Based Anti-Jamming Beamforming Designs Against Adversarial Jamming
  Attacks.** O. Kwon, H. Lee, M. Debbah, I. Lee. *IEEE Transactions on Wireless Communications*,
  vol. 25, pp. 20900–20912, 2026. DOI `10.1109/TWC.2026.3713089`. *[published]* — learned
  beamforming defence explicitly against *adversarial* jamming. One of the few peer-reviewed 2026
  papers that names an adversarial jammer as the threat model.

### G. Method-level results with direct bearing on our approach

**[R4] Reinforcement Learning Disrupts Gradient-Based Adversarial Optimization.**
X. Zou, C. Zhao, A. Aghabagherloo, D. Singelée, R. Degraeve, B. Preneel. arXiv:2606.12251,
Jun 2026. *[preprint]*
Classifiers trained with policy-gradient objectives + ε-greedy exploration substantially degrade
gradient-based attacks; RL acts as an implicit regularizer producing unstable gradient directions
and smaller gradient magnitudes. RL + adversarial training gives the best robustness across
PGD/AutoAttack, transfer-based and query-based attacks. (Vision: CIFAR-10/100, ImageNet-100.)
- **For us — read this before building milestone 3.** Our plan replaces black-box PPO with
  surrogate-gradient/direct-gradient optimization precisely because gradients work and scalar RL
  rewards do not. [R4] says a defender can deliberately destroy that gradient structure at
  training time, cheaply. It does not invalidate the plan (our detector is conventionally
  trained), but it is the obvious defender response to our attack and belongs in the
  adaptation-cost discussion as **a defender adaptation whose cost is a training-recipe change
  rather than more data** — a qualitatively different point on the cost curve, and a good
  future-work sentence.

**[R17] How to Combat Reactive and Dynamic Jamming Attacks with Reinforcement Learning.**
Y. E. Sagduyu, T. Erpek, K. Davaslioglu, S. Kompella. arXiv:2510.02265, Oct 2025. *[preprint]* —
the same group as `[11]`/`[47]`, continuing the defender-adaptation line.

**[R18] Coordinated Anti-Jamming Resilience in Swarm Networks via MARL.** B. Abolhassani *et al.*
arXiv:2512.16813, Dec 2025. *[preprint]* (= `[16]` in the main list).

### H. Covert communications, ISAC, and surveys

- **[R14] Covert Communication with Spatially Heterogeneous User Cooperation Against a
  Geometry-Aware Warden.** H. Yeom, J. Lee. arXiv:2608.10446, **11 Aug 2026** — the most recent
  item in this file. *[preprint]* Wardens are now modelled with spatial/geometric awareness rather
  than as a single radiometer.
  *For us:* the covert-comms field is moving toward richer warden models, the same direction we
  push (learned warden). Supports the framing; not a competitor.
- **[R13] Deep Learning-Driven Friendly Jamming for Secure Multicarrier ISAC Under Channel
  Uncertainty.** B. M. Tuan, V.-D. Nguyen, D. N. Nguyen, N. L. Trung, N. V. Huynh, D. T. Hoang,
  M. Krunz, E. Dutkiewicz. arXiv:2603.05062, Mar 2026. *[preprint]* — learned *friendly* jamming
  under imperfect CSI; a useful contrast for "jamming as a designed waveform" without the
  adversarial-detector element.
- **[R15] Secure Communications, Sensing, and Computing Towards Next-Generation Networks.**
  R. Liu, B. Zheng, J. Lee, S.-H. Lee, G. Kaddoum, O. Günlü. arXiv:2602.19942, Feb 2026.
  *[preprint]* — the 2026 roadmap survey; useful single citation for "where PHY security is
  heading".
- **[R21] Agent-Based Anti-Jamming Techniques for UAV Communications in Adversarial Environments:
  A Comprehensive Survey.** J. Yang, M. Cui, H. Zhang, F. Ji, Z. Lai, Y. Wang. arXiv:2508.11687,
  Aug 2025. *[preprint]* — the current survey of agent-based (RL/MARL) anti-jamming.

**Also published in 2026, peripheral to our axes** (metadata partially verified — look up authors
before citing): *DeepSpect: An RF spectrogram-based deep learning approach for near-real-time
attack detection in FANETs*, **Ad Hoc Networks**, vol. 185, p. 104178, 2026; *Heterogeneous
Federated Deep Reinforcement Learning-Empowered Dual-Threat Jamming Detection in Space–Air–Ground
ISAC Networks*, **IEEE Systems Journal**, 2026 (early access).

---

## 3. The thesis's three claims, re-checked against August 2026

| Claim | Status in Aug 2026 | What to do |
|---|---|---|
| **C1. The SOTA spectrogram CNN is largely an out-of-band-emission detector; closing the in-band blind spot costs FAR/accuracy** | **Unchallenged.** No 2026 work characterizes what a jamming-detection CNN actually keys on, and [R22]'s ~85–88% realistic-channel accuracy independently corroborates the channel-realism cost. | Keep as the solid core. Add [R22] as external corroboration. |
| **C2. A stealthy-and-effective region survives the CNN + energy suite on a realistic channel** | **Narrowed, not refuted.** [R2] shows an adversarially-hardened, open-set detector holding gradient-based evasion below 2% — but in the CSI modality, with CSI-domain perturbations. | **Scope the claim explicitly** to the spectrogram-CNN + energy suite; cite [R2] as the defender-side response our result motivates. Do not claim generality over learned detectors. |
| **C3. Adaptation is expensive, and we price one round on each side** | **Strengthened.** [R1] shows the retraining loop being *productized* (drift detection → pseudo-labels → retrain → hot-swap), and [R4] adds a second, cheaper defender adaptation axis (training recipe, not data). Nobody prices any of it. | Lead with it, as planned. [R1] is the "this is a real operational cost, not an academic one" citation. |

**Novelty re-check (Aug 2026).** Targeted searches for a *learned jammer* evaluated against a
*learned jamming detector* on a **BER-vs-detection-probability frontier**, with **per-round
adaptation cost**, returned nothing matching. The nearest 2026 neighbours are [R3] (ROC-based
undetectability, roles mirrored, energy domain) and [R9] (black-box constrained attack on an OFDM
detection network, not a jamming detector). The gap statement in `Literature_Review.md` stands as
written — but it is an absence-of-evidence result and this check should be re-run in the week
before submission.

---

## 4. Concrete changes this implies

Ordered by value ÷ hours, consistent with the current schedule.

1. **Threat Model paragraph on self-adaptive detection (≈30 min, no compute, do before the meeting).**
   §III-D currently asserts the no-online-adaptation constraint. Add: SAJD [R1] automates the
   retraining loop with drift-triggered *pseudo*-labels; ground truth is still absent; adaptation
   is still round-based; therefore our model holds and our cost metrics are exactly what such a
   pipeline spends. This turns the sharpest available objection into a supporting citation.
2. **Scope C2 in the abstract and conclusion (≈15 min).** One clause — "against a spectrogram-CNN
   and energy-detector suite" — pre-empts the CITADEL objection [R2] at zero cost.
3. **Add [R3] to Performance Metrics (≈15 min).** Independent, hardware-validated precedent for
   ROC-at-low-\(P_{FA}\) undetectability. It makes our honesty correction look like community
   practice rather than a retreat from an earlier claim.
4. **Limitations paragraph (≈30 min):** diffusion-based jamming-resilient receivers [R11];
   multi-antenna JMD [R12]; non-differentiable detectors [R6]; application-layer detection [R8].
   Four sentences, one citation each.
5. **Currency citations for Related Work (≈30 min).** Minimum set to look current: [R1], [R2],
   [R3], [R5], [R9], [R10]. Insert [R1]/[R2] in the detection paragraph, [R3]/[R9] in the
   adversarial-ML paragraph, [R10] in the learning-based-jamming paragraph, [R5] in Experiment
   Setup.
6. **Blackout-week experiment: evaluate on the public OFDMA benchmark [R5] (≈4–6 h, laptop-only,
   Sep 7–14).** The only external-validity result obtainable with no cluster access. Even a
   negative or partial result ("our channel-valid detector transfers at X% to an independently
   generated dataset") is worth a paragraph.
7. **Future-work sentence on RL-trained detectors [R4] (≈10 min).** Names the cheapest defender
   counter-move to our surrogate-gradient attack and shows we know it exists.

---

## 5. 2026 works against the thesis axes

| Ref | Side | Action space / modality | Detector modality | Detectability in objective | Adaptation | Status |
|---|---|---|---|---|---|---|
| SAJD [R1] | Defence | O-RAN KPI telemetry | ML xApp + retraining rApps | n/a | **Yes — automated, pseudo-labelled** | preprint |
| CITADEL [R2] | Defence | CSI features | 2-stage closed+open-set | n/a (resists evasion) | Open-set, no retrain loop | preprint |
| Hazra & Harshan [R3] | Both | Energy statistics | Statistical + ML energy detectors | **Yes — ROC at low \(P_{FA}\)** | No | preprint |
| Zou *et al.* [R4] | Defence | Model training recipe | Image classifier (vision) | n/a | **Yes — training-regime change** | preprint |
| Schösser *et al.* [R5] | Benchmark | OFDMA spectrograms | supervised + unsupervised baselines | n/a | n/a | preprint |
| Xu *et al.* [R9] | Attack | OFDM waveform, black-box | OFDM signal-detection DNN | Communication constraint | No | **published** |
| Kwon *et al.* [R10] | Defence | Beamforming | — | n/a | No | **published** |
| Terraf *et al.* [R8] | Defence | Application-layer features | Transfer-learned DNN | n/a | Transfer learning | **published** |
| Diffusion receivers [R11] | Defence | Received signal | — (mitigation, not detection) | n/a | No | preprint |
| JMD [R12] | Defence | Multi-antenna subspace | — (mitigation) | n/a | No | preprint |
| **This thesis** | **Attack + defence** | **Per-subcarrier OFDM interference** | **Spectrogram CNN ∨ energy, frozen per round** | **Yes — BER at matched \(p_{\text{suite}}\), budget = clean FAR** | **Yes — priced per round, both sides** | — |

---

## 6. Reference list

*All titles, authors and dates verified via the arXiv API or Crossref in August 2026.*

| # | Reference | Status |
|---|---|---|
| R1 | M. H. Rahman, M. S. Hossen, N. H. Stephenson, V. K. Shah, A. Da Silva, "SAJD: Self-Adaptive Jamming Attack Detection in AI/ML Integrated 5G O-RAN Networks," arXiv:2511.17519, 2025; demo: arXiv:2510.09706, IEEE MILCOM 2025 | preprint + demo |
| R2 | A. Bouferroum, I. Alla, V. Loscri, A. Benslimane, V. Lenders, "CITADEL: CSI-Based Jamming Detection and Open-Set Classification for IIoT Networks," arXiv:2606.22939, 22 Jun 2026 | preprint |
| R3 | S. Hazra, J. Harshan, "Cooperative Mitigation against Learning-Based Reactive Jammers: Analysis and SDR Validation," arXiv:2606.01197, 31 May 2026 | preprint |
| R4 | X. Zou, C. Zhao, A. Aghabagherloo, D. Singelée, R. Degraeve, B. Preneel, "Reinforcement Learning Disrupts Gradient-Based Adversarial Optimization," arXiv:2606.12251, 10 Jun 2026 | preprint |
| R5 | A. Schösser, M. Salehi, S. Ma, P. Schulz, G. Fettweis, "Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset," arXiv:2606.02102, 1 Jun 2026 (submitted to IEEE OJ-COMS) | preprint |
| R6 | V. Halenka, M. Amini, P.-A. Andersen, O.-C. Granmo, B. Kantarci, "Explainable and Hardware-Efficient Jamming Detection for 5G Networks Using the Convolutional Tsetlin Machine," arXiv:2603.07336, 7 Mar 2026 | preprint |
| R7 | S. Kuili, M. Amini, B. Kantarci, "Toward Resilient 5G Networks: Comparative Analysis of Federated and Centralized Learning for RF Jamming Detection," arXiv:2605.01705, 3 May 2026 | preprint |
| R8 | Y. Terraf, Y. Iraqi, A. Al-Dweik, A. Pandey, J.-P. Giacalone, "Jamming Detection at the Application Layer Using Deep Learning: A Transfer Learning Approach," *IEEE OJ-COMS*, vol. 7, pp. 7302–7318, 2026, DOI 10.1109/OJCOMS.2026.3707737 | **published** |
| R9 | Y. Xu, Q. Tang *et al.*, "Communication-constrained black-box adversarial attack against OFDM signal detection network," *Physical Communication*, vol. 78, p. 103235, Aug 2026, DOI 10.1016/j.phycom.2026.103235 | **published** |
| R10 | O. Kwon, H. Lee, M. Debbah, I. Lee, "Deep Learning-Based Anti-Jamming Beamforming Designs Against Adversarial Jamming Attacks," *IEEE Trans. Wireless Commun.*, vol. 25, pp. 20900–20912, 2026, DOI 10.1109/TWC.2026.3713089 | **published** |
| R11 | H. She, Y. Cheng, T. Sun, P. Wang, S. Huang, K. Yang, "Brownian Bridge Diffusion-Based Joint Channel Estimation and Data Detection for Jamming-Resilient Receivers," arXiv:2606.28778, 27 Jun 2026 | preprint |
| R12 | G. Marti, C. Studer, "Joint Jammer Mitigation and Data Detection," arXiv:2510.02021, 2 Oct 2025 | preprint |
| R13 | B. M. Tuan, V.-D. Nguyen, D. N. Nguyen, N. L. Trung, N. V. Huynh, D. T. Hoang, M. Krunz, E. Dutkiewicz, "Deep Learning-Driven Friendly Jamming for Secure Multicarrier ISAC Under Channel Uncertainty," arXiv:2603.05062, 5 Mar 2026 | preprint |
| R14 | H. Yeom, J. Lee, "Covert Communication with Spatially Heterogeneous User Cooperation Against a Geometry-Aware Warden," arXiv:2608.10446, 11 Aug 2026 | preprint |
| R15 | R. Liu, B. Zheng, J. Lee, S.-H. Lee, G. Kaddoum, O. Günlü, "Secure Communications, Sensing, and Computing Towards Next-Generation Networks," arXiv:2602.19942, 23 Feb 2026 | preprint |
| R16 | S. K. Gopalakrishnan, S. Hailes, "Towards Trustworthy Wi-Fi CSI-based Sensing: Systematic Evaluation of Adversarial Robustness," arXiv:2511.20456, 25 Nov 2025 | preprint |
| R17 | Y. E. Sagduyu, T. Erpek, K. Davaslioglu, S. Kompella, "How to Combat Reactive and Dynamic Jamming Attacks with Reinforcement Learning," arXiv:2510.02265, 2 Oct 2025 | preprint |
| R18 | B. Abolhassani, T. Erpek, K. Davaslioglu, Y. E. Sagduyu, S. Kompella, "Coordinated Anti-Jamming Resilience in Swarm Networks via Multi-Agent Reinforcement Learning," arXiv:2512.16813, Dec 2025 | preprint |
| R19 | I. Panitsas, Y. Yigit, L. Tassiulas, L. Maglaras, B. Canberk, "JamShield: A Machine Learning Detection System for Over-the-Air Jamming Attacks," IEEE ICC 2025 | **published** ⚠ |
| R20 | I. Panitsas, I. Ofeidis, L. Tassiulas, "FedJam: Multimodal Federated Learning Framework for Jamming Detection," arXiv:2508.09369, Aug 2025 | preprint |
| R21 | J. Yang, M. Cui, H. Zhang, F. Ji, Z. Lai, Y. Wang, "Agent-Based Anti-Jamming Techniques for UAV Communications in Adversarial Environments: A Comprehensive Survey," arXiv:2508.11687, Aug 2025 | preprint |
| R22 | J. Viana *et al.*, "PCA-Featured Transformer for Jamming Detection in 5G UAV Networks," *IEEE OJ-COMS*, vol. 6, pp. 9287–9303, 2025 | **published** |
| R23 | "DeepSpect: An RF spectrogram-based deep learning approach for near-real-time attack detection in FANETs," *Ad Hoc Networks*, vol. 185, p. 104178, 2026 | **published** ⚠ authors unverified |
| R24 | "Heterogeneous Federated Deep Reinforcement Learning-Empowered Dual-Threat Jamming Detection in Space–Air–Ground ISAC Networks," *IEEE Systems Journal*, 2026 (early access) | **published** ⚠ authors unverified |

### BibTeX for the six currency citations worth adding now

```bibtex
@misc{rahman2025sajd,
  title={{SAJD}: Self-Adaptive Jamming Attack Detection in {AI/ML} Integrated 5G {O-RAN} Networks},
  author={Rahman, Md Habibur and Hossen, Md Sharif and Stephenson, Nathan H. and Shah, Vijay K. and Da Silva, Aloizio},
  year={2025}, eprint={2511.17519}, archivePrefix={arXiv}, primaryClass={cs.NI}, note={preprint}}

@misc{bouferroum2026citadel,
  title={{CITADEL}: {CSI}-Based Jamming Detection and Open-Set Classification for {IIoT} Networks},
  author={Bouferroum, Aymen and Alla, Ildi and Loscri, Valeria and Benslimane, Abderrahim and Lenders, Vincent},
  year={2026}, eprint={2606.22939}, archivePrefix={arXiv}, primaryClass={cs.CR}, note={preprint}}

@misc{hazra2026cooperative,
  title={Cooperative Mitigation against Learning-Based Reactive Jammers: Analysis and {SDR} Validation},
  author={Hazra, Soumita and Harshan, J.},
  year={2026}, eprint={2606.01197}, archivePrefix={arXiv}, primaryClass={cs.IT}, note={preprint}}

@misc{schosser2026spectrum,
  title={Spectrum Anomaly Detection in {OFDMA} Systems: Simulation Framework and Benchmark Dataset},
  author={Sch{\"o}sser, Anton and Salehi, Mohammadhadi and Ma, Sinuo and Schulz, Philipp and Fettweis, Gerhard},
  year={2026}, eprint={2606.02102}, archivePrefix={arXiv}, primaryClass={eess.SP}, note={preprint}}

@article{xu2026communication,
  title={Communication-constrained black-box adversarial attack against {OFDM} signal detection network},
  author={Xu, Yamei and Tang, Qichao and others},
  journal={Physical Communication}, volume={78}, pages={103235}, year={2026},
  doi={10.1016/j.phycom.2026.103235}}

@article{kwon2026deep,
  title={Deep Learning-Based Anti-Jamming Beamforming Designs Against Adversarial Jamming Attacks},
  author={Kwon, Ohseung and Lee, Hoon and Debbah, Merouane and Lee, Inkyu},
  journal={IEEE Transactions on Wireless Communications}, volume={25}, pages={20900--20912}, year={2026},
  doi={10.1109/TWC.2026.3713089}}
```

---

## 7. Caveats

1. **Preprint dominance.** 17 of 24 entries are unrefereed. Their headline numbers (CITADEL's
   "<2% evasion", SAJD's "outperforms existing xApps") have not been through review — quote them
   as *claims*, attributed, never as established facts.
2. **Search coverage.** This is a keyword-driven sweep of arXiv/Crossref/web, not a systematic
   review. It will have missed venue-only 2026 papers not yet indexed, and anything behind
   paywalls without an indexed abstract. Notably, IEEE TIFS/TWC 2026 issues are only partially
   crawled — a manual pass over the 2026 TIFS and TWC tables of contents before submission is
   worth an hour.
3. **The novelty check is absence-of-evidence.** Re-run it in the submission week; a single
   September preprint could change the gap statement.
4. **Two entries ([R23], [R24]) have unverified author lists** — resolve before citing.
5. **[R19] JamShield** appears both as IEEE ICC 2025 and as a separate IEEE Xplore record;
   confirm which version to cite.
