# Sources, evaluation, and comparison plan

Companion to `Literature_Review.md`. Citation numbers match that document exactly.

**How each entry was verified.** Every reference below was resolved against Crossref
(`api.crossref.org`) or the Semantic Scholar graph API for title, venue, volume, issue, page
range and DOI; the two preprints and the workshop/conference-proceedings entries were resolved
against the publisher's own proceedings listing. Nothing here is cited from memory. The
handful of items where a field could not be machine-verified are tagged **⚠** with the exact
uncertainty, and the "Residual risks" section at the end lists them together.

**Venue policy.** Per supervisor feedback, MDPI journals are excluded — none of the 49
references below is MDPI (the one MDPI entry currently in `refs.bib`,
`electronics14163307`, should be deleted; see §6). Two entries are arXiv preprints, both
flagged and both defensible: one is the *only* source for a tool we use (Sionna), the other is
recent enough that no venue version exists yet. Everything else is IEEE
Transactions/Journals/Surveys, ACM proceedings, or a top-tier ML venue (NeurIPS / ICML / ICLR
/ JMLR).

**Legend for each entry**
- **Relates** — why it belongs in this paper's related work.
- **Differs** — the specific thing it does *not* do that we do. This is the sentence that
  survives into the rebuttal if a reviewer says "this has been done".
- **Use** — where applicable: how to use it concretely (baseline to run, detector to add,
  metric to adopt).

---

## 1. Surveys and reviews (framing)

**[1] H. Pirayesh and H. Zeng**, "Jamming attacks and anti-jamming strategies in wireless
networks: A comprehensive survey," *IEEE Communications Surveys & Tutorials*, vol. 24, no. 2,
pp. 767–809, 2022. DOI: `10.1109/COMST.2022.3159185`
- Venue: IEEE COMST — the highest-impact survey venue in the field.
- **Relates**: the standard opening citation; gives the jammer taxonomy (barrage, tone,
  protocol-aware, reactive) our classical baselines instantiate, and the anti-jamming
  taxonomy our defender sits in.
- **Differs**: a survey, so no method — but note it frames detection as a defense *step*, not
  as an adversarial target. The idea that the detector itself is what the attacker optimizes
  against is outside its scope.
- **⚠ Correction to `refs.bib`**: the existing entry `jamming_survey_2024` says `year={2024}`.
  The paper is **2022**. Fix the year and add the DOI.

**[2] P. Lohan, B. Kantarci, M. A. Ferrag, N. Tihanyi, and Y. Shi**, "From 5G to 6G networks:
A survey on AI-based jamming and interference detection and mitigation," *IEEE Open Journal of
the Communications Society*, vol. 5, pp. 3920–3974, 2024. DOI: `10.1109/OJCOMS.2024.3416808`
- Venue: IEEE OJ-COMS (open access, fully peer-reviewed IEEE ComSoc journal).
- **Relates**: the current state-of-the-art map of *AI-based* jamming detection — establishes
  that learned detectors are the assumed defender in 5G/6G, which is what makes attacking one
  a relevant threat model rather than a strawman.
- **Differs**: surveys detection and mitigation only; the attacker is always a member of a
  fixed taxonomy. No treatment of a detector-aware attacker or of retraining cost.

**[3] D. Adesina, C.-C. Hsieh, Y. E. Sagduyu, and L. Qian**, "Adversarial machine learning in
wireless communications using RF data: A review," *IEEE Communications Surveys & Tutorials*,
vol. 25, no. 1, pp. 77–100, 2023. DOI: `10.1109/COMST.2022.3205184`
- Venue: IEEE COMST.
- **Relates**: situates the paper in adversarial ML rather than only in jamming; gives the
  standard attack taxonomy (evasion / poisoning / inference) and confirms our attack is an
  *evasion* attack with a physical-realizability constraint.
- **Differs**: reviews attacks on classifiers of the legitimate signal (modulation, RF
  fingerprint, spectrum occupancy). Attacks whose success requires *also* degrading a link are
  not covered — that combination is our contribution.

---

## 2. Classical and protocol-aware jamming; energy-optimal jamming

**[4] W. Xu, W. Trappe, Y. Zhang, and T. Wood**, "The feasibility of launching and detecting
jamming attacks in wireless networks," in *Proc. ACM MobiHoc*, 2005, pp. 46–57. DOI:
`10.1145/1062689.1062697`
- Venue: ACM MobiHoc — top-tier, and the canonical citation for jamming *detection* metrics.
- **Relates**: the origin of the "detect jamming from observable statistics" problem, and of
  the false-alarm problem (congestion looks like jamming) that our FAR-matched stealth budget
  formalizes.
- **Differs**: detection statistics are packet-level (PDR, RSSI, location consistency), not
  waveform-level, and the attacker is fixed. Twenty years on, it is the historical anchor, not
  a competitor.

**[5] C. Shahriar, M. La Pan, M. Lichtman, T. C. Clancy, R. McGwier, R. Tandon, S. Sodagari,
and J. H. Reed**, "PHY-layer resiliency in OFDM communications: A tutorial," *IEEE
Communications Surveys & Tutorials*, vol. 17, no. 1, pp. 292–314, 2015. DOI:
`10.1109/COMST.2014.2349883`
- Venue: IEEE COMST.
- **Relates**: the reference tutorial for OFDM-specific attack surfaces (pilots, preamble,
  cyclic prefix, equalization) and their countermeasures. Directly supports our System Model's
  choice to expose pilot/guard/data subcarrier sets explicitly.
- **Differs**: analytical/heuristic attacks and countermeasures; no learned detector, no
  learned attacker, no detectability-constrained optimization.

**[6] T. C. Clancy**, "Efficient OFDM denial: Pilot jamming and pilot nulling," in *Proc. IEEE
ICC*, 2011, pp. 1–5. DOI: `10.1109/ICC.2011.5962467`
- Venue: IEEE ICC — flagship ComSoc conference.
- **Relates**: the strongest *classical* efficiency result in our exact setting: pilot jamming
  is ≈2 dB and pilot nulling ≈7.5 dB more efficient than barrage for a target BER on QPSK
  OFDM. It is the natural "smart classical attacker" upper baseline.
- **Differs**: efficiency is measured purely in signal-to-jamming ratio at the receiver.
  Because pilot attacks concentrate energy on a sparse, *known* subcarrier set, they are
  plausibly very visible to a spectrogram CNN — which is exactly the untested question our
  frontier answers.
- **Use**: **add as an attacker baseline.** High value per hour: our grid already exposes
  `\mathcal{K}_p`, so pilot jamming/nulling is a few lines in `build_jam()`, and it gives the
  paper a protocol-aware classical comparator that a reviewer will otherwise ask for.

**[7] M. J. La Pan, T. C. Clancy, and R. W. McGwier**, "Jamming attacks against OFDM timing
synchronization and signal acquisition," in *Proc. IEEE MILCOM*, 2012, pp. 1–7. DOI:
`10.1109/MILCOM.2012.6415749`
- Venue: IEEE MILCOM.
- **Relates**: establishes synchronization/preamble as an attack surface, and (usefully for
  us) that the *attacker* also depends on preamble timing — the technical basis for the
  desynchronization realism axis the supervisor asked for.
- **Differs**: denies acquisition rather than injecting in-band interference into a
  synchronized frame; detectability is not modeled.

**[8] S. Amuru and R. M. Buehrer**, "Optimal jamming against digital modulation," *IEEE
Transactions on Information Forensics and Security*, vol. 10, no. 10, pp. 2212–2224, Oct.
2015. DOI: `10.1109/TIFS.2015.2451081`
- Venue: IEEE TIFS.
- **Relates**: **the single most important reference for the minimum-energy boundary attack.**
  Derives in closed form the jamming signal maximizing symbol error probability against
  amplitude-phase constellations over AWGN, and shows matching the victim's signal is
  generally suboptimal. This is the published, peer-reviewed ancestor of the supervisor's
  IQ-plot observation ("points should cluster at the decision boundary, that is the
  minimal-energy alteration").
- **Differs**: AWGN only, no fading, no channel pre-compensation through the jammer's own
  link, and — decisively — no detector: energy is minimized for its own sake, not to buy
  stealth. Our contribution over [8] is to place the same minimum-energy logic on a
  frequency-selective channel *and* to evaluate whether the energy saving actually converts
  into reduced detectability against a learned detector (it need not, if the CNN keys on
  structure rather than magnitude).
- **Use**: **cite as the theoretical justification for the boundary attack, and run its
  closed-form solution as the AWGN reference point.** Framing the boundary attack as "[8]
  extended to a fading channel under a detectability constraint" is much stronger than
  presenting it as novel from scratch.

**[9] L. Zhang, F. Restuccia, T. Melodia, and S. M. Pudlewski**, "Jam sessions: Analysis and
experimental evaluation of advanced jamming attacks in MIMO networks," in *Proc. ACM MobiHoc*,
2019, pp. 61–70. DOI: `10.1145/3323679.3326504`
- Venue: ACM MobiHoc.
- **Relates**: shows that structure-exploiting jamming beats power-based jamming in practice,
  with SDR validation — the empirical precedent for "smarter, not louder".
- **Differs**: MIMO spatial structure, not stealth; no detector in the loop. Also useful as
  the citation for why our single-antenna scope needs stating (multi-antenna defense is [29]).

---

## 3. Learning-based and cooperative jamming

**[10] S. Amuru, C. Tekin, M. van der Schaar, and R. M. Buehrer**, "Jamming bandits—A novel
learning method for optimal jamming," *IEEE Transactions on Wireless Communications*, vol. 15,
no. 4, pp. 2792–2808, Apr. 2016. DOI: `10.1109/TWC.2015.2510643`
- Venue: IEEE TWC.
- **Relates**: the cleanest "learned attacker with convergence guarantees" in the jamming
  literature; jointly optimizes error inflicted and energy used, which is a two-objective
  attacker like ours.
- **Differs**: the action space is a *finite* set of PHY configurations (scheme, power, pulse
  duration), i.e. exactly the "proxy problem" the supervisor flagged; the objective pairs
  error rate with energy, never with a detection probability.
- **Use**: optional attacker baseline if a *learned-but-discrete* comparator is wanted; the
  cheaper substitute is our existing blind/channel-aware sweep, which already upper-bounds
  what discrete selection can achieve.

**[11] T. Erpek, Y. E. Sagduyu, and Y. Shi**, "Deep learning for launching and mitigating
wireless jamming attacks," *IEEE Transactions on Cognitive Communications and Networking*,
vol. 5, no. 1, pp. 2–14, Mar. 2019. DOI: `10.1109/TCCN.2018.2884910`
- Venue: IEEE TCCN.
- **Relates**: the standard citation for "jamming as adversarial ML", and the closest prior
  work to a *co-adaptive* loop: the jammer learns a classifier of transmission success, and the
  defense responds by poisoning the jammer's training data. That defense-response is a
  round-1 adaptation in our sense.
- **Differs**: the attacked model is the *transmitter's* channel-status classifier and the
  action is binary (jam this slot or not) — no waveform, no in-band shaping, no jamming
  detector. The adaptation is demonstrated but never priced (no ΔFAR, no sample/compute cost),
  which is precisely the gap our headline fills.

**[12] J. Zhang and X. Wu**, "Cooperative jamming over DRL-based frequency hopping wireless
communications: A one-leader multi-follower Stackelberg game approach," *IEEE Transactions on
Information Forensics and Security*, vol. 20, pp. 9220–9234, 2025. DOI:
`10.1109/TIFS.2025.3604229`
- Venue: IEEE TIFS.
- **Relates**: the most current cooperative-multi-jammer paper against a *learning* victim;
  the closest match to our "cooperative team" axis and a game-theoretic formalization of the
  two-sided interaction.
- **Differs**: coordination happens in the frequency/power domain over discrete choices; the
  victim is an anti-jamming DRL agent, not a detector; stealth is absent from the objective.
- **⚠ `refs.bib` fix**: the existing entry (key `article`) has `volume={PP}, pages={1-1}`
  (early access). Update to vol. 20, pp. 9220–9234, and rename the key to something readable
  (`zhang2025cooperative`) — `article` as a citation key is a bug waiting to happen.

**[13] P. Valianti, K. Malialis, P. Kolios, and G. Ellinas**, "Cooperative multi-agent jamming
of multiple rogue drones using reinforcement learning," *IEEE Transactions on Mobile
Computing*, vol. 23, no. 12, pp. 12345–12359, Dec. 2024. DOI: `10.1109/TMC.2024.3409050`
- Venue: IEEE TMC.
- **Relates**: genuine cooperative MARL *jamming* (not anti-jamming), with a joint
  mobility-and-power action and an explicit interference constraint on teammates — the
  structural analogue of our multi-agent power-splitting question.
- **Differs**: the jamming metric is received jamming power at the target, not BER through a
  modeled receiver chain; there is no detector and no waveform synthesis. The page range
  (12345–12359) looks like a placeholder but is the genuine range — verified via DOI.

**[14] Y. Qin, J. Tang, F. Tang, M. Zhao, and N. Kato**, "Multi-agent reinforcement learning
in adversarial game environments: Personalized anti-interference strategies for heterogeneous
UAV communication," *IEEE Transactions on Mobile Computing*, vol. 24, no. 9, pp. 8886–8898,
Sep. 2025. DOI: `10.1109/TMC.2025.3559123`
- Venue: IEEE TMC.
- **Relates**: current MARL-versus-adversary formulation with per-agent heterogeneity, and an
  equilibrium argument — useful for the zero-sum framing the supervisor asked for.
- **Differs**: defense side, channel-and-power action space, no detector, no stealth term.
- **⚠ `refs.bib` fix**: existing entry `qin2025multi` has no volume/pages; add them.

**[15] Y. Xu, Y. Xu, X. Dong, G. Ren, J. Chen, X. Wang, L. Jia, and L. Ruan**, "Convert harm
into benefit: A coordination-learning based dynamic spectrum anti-jamming approach," *IEEE
Transactions on Vehicular Technology*, vol. 69, no. 11, pp. 13018–13032, Nov. 2020. DOI:
`10.1109/TVT.2020.3018121`
- Venue: IEEE TVT.
- **Relates**: representative cooperative *defense* learning; establishes the symmetric side
  of the arms race we price.
- **Differs**: spectrum-access defense against a sweeping jammer; no learned detector, no
  waveform-level attack. Keep as one citation in a group, not a discussed paper.

**[16] B. Abolhassani, T. Erpek, K. Davaslioglu, Y. E. Sagduyu, and S. Kompella**,
"Coordinated anti-jamming resilience in swarm networks via multi-agent reinforcement
learning," arXiv:2512.16813, 2025. **⚠ preprint — no peer-reviewed version located.**
- **Relates**: QMIX-based cooperative defense against a *reactive* jammer with Markovian
  threshold dynamics, compared against a genie-aided bound — methodologically close to how we
  bound our attacker with a genie tier.
- **Differs**: defense side, discrete channel access, threshold-based adversary.
- **Recommendation**: keep, but cite it as a preprint and demote it to a group citation. If a
  peer-reviewed version appears before submission, swap it in; if the reference list needs
  trimming, this is the first jamming-side citation to cut, since [15] and [17] already cover
  cooperative defense from published venues.

**[17] S. Leuenberger, A. Di Maio, and T. Braun**, "Proactive decentralized multi-agent
data-driven node relocation for jammed mobile networks," in *Proc. IEEE Wireless Days (WD)*,
2025, pp. 1–8. DOI: `10.1109/WD67713.2025.11302544`
- Venue: IEEE Wireless Days.
- **Relates**: decentralized multi-agent, data-driven response to jamming, and the mobility
  dimension the supervisor named as the eventual destination ("mobile victims").
- **Differs**: spatial relocation as the defense, not signal-level detection or attack.
- **Note**: this is the supervisor's own group's work and the closest paper in the reference
  list to the intended thesis trajectory. Worth citing in the future-work paragraph as well as
  in related work — the multi-agent/mobility extension is a natural continuation of it.

**[18] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch**, "Multi-agent
actor-critic for mixed cooperative-competitive environments," in *Advances in Neural
Information Processing Systems (NeurIPS)*, vol. 30, 2017.
- Venue: NeurIPS. **Relates**: the CTDE paradigm our coordination assumption tier (c) is
  defined by. **Differs**: general MARL method, no wireless content. Already in `refs.bib` as
  `NIPS2017_68a97503`.

**[19] T. Rashid, M. Samvelyan, C. Schroeder de Witt, G. Farquhar, J. Foerster, and S.
Whiteson**, "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement
learning," in *Proc. ICML*, PMLR vol. 80, 2018, pp. 4295–4304.
- Venue: ICML. **Relates**: the algorithm behind the QMIX anti-jamming defenses [16] we
  contrast with. **Differs**: general MARL method.

**[20] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu**, "The surprising
effectiveness of PPO in cooperative multi-agent games," in *Advances in Neural Information
Processing Systems (NeurIPS)*, vol. 35, 2022, pp. 24611–24624.
- Venue: NeurIPS (Datasets & Benchmarks track).
- **Relates**: MAPPO is the algorithm our negative result concerns; citing it is what makes
  the negative result legible as *"MAPPO, tuned per the reference recipe, still fails here"*
  rather than *"our PPO implementation failed"*.
- **Differs**: benchmark domains have low-dimensional discrete or modest continuous actions.
  Our failure mode — a scalar frame-level reward over a ~1500-dimensional raw-IQ action — is
  outside the regime they validate, which is the honest way to state the result.

**[21] J. K. Terry, B. Black, N. Grammel, M. Jayakumar, A. Hari, R. Sullivan, L. S. Santos,
C. Dieffendahl, C. Horsch, R. Perez-Vicente, N. Williams, Y. Lokesh, and P. Ravi**,
"PettingZoo: Gym for multi-agent reinforcement learning," in *Advances in Neural Information
Processing Systems (NeurIPS)*, vol. 34, 2021.
- Venue: NeurIPS. **Relates**: the environment API the supervisor asked us to adopt for the
  multi-agent phase. **Differs**: tooling. Belongs in Methodology if space is tight.

**[22] M. Bettini, A. Prorok, and V. Moens**, "BenchMARL: Benchmarking multi-agent
reinforcement learning," *Journal of Machine Learning Research*, vol. 25, no. 217, pp. 1–10,
2024.
- Venue: JMLR. **Relates**: the MARL benchmarking library named as the alternative to RLlib.
  **Differs**: tooling.

---

## 4. Learned jamming detection — the defender

**[23] Y. Li, J. Pawlak, J. Price, K. Al Shamaileh, Q. Niyaz, S. Paheding, and V.
Devabhaktuni**, "Jamming detection and classification in OFDM-based UAVs via feature- and
spectrogram-tailored machine learning," *IEEE Access*, vol. 10, pp. 16859–16870, 2022. DOI:
`10.1109/ACCESS.2022.3150020`
- Venue: IEEE Access (peer-reviewed IEEE journal; lower selectivity than Transactions but a
  legitimate IEEE venue — and unavoidable here, since it *is* the replicated system).
- **Relates**: **the defender we replicate.** Four jamming classes (barrage, protocol-aware,
  single-tone, successive-pulse), spectrogram CNN at 99.79% accuracy / 0.03% FAR versus 92.20%
  / 1.35% for their feature-based model. Our detector reproduces this number, and their
  feature-based model's reliance on received power is the justification for including an
  energy detector in the suite.
- **Differs**: (i) evaluated only on the four training-time jammer families, so accuracy is a
  closed-world number; (ii) reported at a single operating point, not as an ROC; (iii) no
  adversary optimizing against the classifier; (iv) trained on real SDR captures from scratch,
  where we use a simulated channel — a faithfulness caveat to state explicitly, since their
  spectrogram generation is undocumented.
- **Use**: the primary detector in the suite, and the source of the headline
  characterization claim (a spectrogram CNN trained this way is largely an
  out-of-band-emission detector).

**[24] Z. Zhang and M. Krunz**, "Detection and classification of smart jamming in Wi-Fi
networks using machine learning," in *Proc. IEEE MILCOM*, 2023, pp. 919–924. DOI:
`10.1109/MILCOM58377.2023.10356126`
- Venue: IEEE MILCOM.
- **Relates**: the closest *stealth-aware* detector: it targets preamble, pilot and
  interleaving jamming — i.e. exactly the low-power protocol-aware attacks a power meter
  misses — using continuous-wavelet features and a compact CNN.
- **Differs**: attacks are drawn from a fixed set of hand-specified stealthy jammers, none
  optimized against the detector; features are hand-designed (wavelet), which the supervisor
  correctly flagged as a weakness.
- **Use**: **the best candidate for a second learned detector in the suite.** A CWT-based
  detector fails differently from an STFT-based one, so adding it converts "our jammer evades
  *a* CNN" into "our jammer evades two learned detectors with different time-frequency
  front-ends" — a substantially harder claim to dismiss.

**[25] M. Varotto, S. Valentin, and S. Tomasin**, "Detecting 5G signal jammers using
spectrograms with supervised and unsupervised learning," in *Proc. IEEE ICC Workshops*, 2024,
pp. 767–772. DOI: `10.1109/ICCWorkshops59551.2024.10615325`
- Venue: IEEE ICC Workshops.
- **Relates**: directly comparable setup (spectrograms of IQ samples from a 5G signal) and,
  crucially, compares a *supervised* CNN against an *unsupervised* convolutional autoencoder.
- **Differs**: fixed broadband jammer set; no adversarial evaluation.
- **Use**: **the strongest robustness check available to us, and cheap.** An unsupervised
  anomaly detector trained only on clean faded frames cannot be evaded by pushing across a
  learned clean/jammed boundary, so if our stealthy jammer survives it too, the stealth claim
  stops looking like an artifact of one classifier's decision surface. Recommended as the
  third suite member (after the CNN and the energy detector) if any compute remains.

**[26] J. Viana, H. Farkhari, P. Sebastião, V. P. Gil Jiménez, *et al.***, "PCA-featured
transformer for jamming detection in 5G UAV networks," *IEEE Open Journal of the
Communications Society*, vol. 6, pp. 9287–9303, 2025. DOI: `10.1109/OJCOMS.2025.3619817`
- Venue: IEEE OJ-COMS. **⚠** full author list truncated above; expand before citing.
- **Relates**: the most current learned-detector architecture (transformer encoder-decoder) in
  this exact application, reporting ~88% LoS / ~85% NLoS accuracy. Useful evidence that
  detector accuracy on *realistic* channels is well below the 99.79% of [23] — which
  independently corroborates our finding that channel realism costs the detector.
- **Differs**: architecture study, fixed jammer set, no adversary.
- **Use**: cite as "current SOTA detector architecture"; optionally use as a *transfer* target
  (attack tuned on our CNN, evaluated on a transformer) if the adaptive-attack discipline of
  [46] is to be honored fully.

**[27] L. Arcangeloni, E. Testi, and A. Giorgetti**, "Jamming detection in MIMO-OFDM ISAC
systems using variational autoencoders," in *Proc. IEEE International Symposium on Systems
Engineering (ISSE)*, 2024, pp. 1–7. DOI: `10.1109/ISSE63315.2024.10741110`
- Venue: IEEE ISSE (smaller conference).
- **Relates**: generative/anomaly-detection alternative to discriminative jamming detection.
- **Differs**: ISAC echo signals, not a communication link's composite frame. Weakest of the
  detector citations — first to cut for space (see the trim list in `Literature_Review.md`).

**[28] H. Urkowitz**, "Energy detection of unknown deterministic signals," *Proceedings of the
IEEE*, vol. 55, no. 4, pp. 523–531, 1967. DOI: `10.1109/PROC.1967.5573`
- Venue: Proceedings of the IEEE.
- **Relates**: the classical foundation of the energy detector in our suite, and the correct
  citation for calibrating a threshold to a target false-alarm rate.
- **Differs**: single-statistic radiometry with analytic performance — no learning. Its role in
  the paper is as the trivial-but-strong baseline that (per our lossless recheck) subsumed the
  CNN on an idealized channel and (per milestone 2) becomes redundant on a faded one. That
  role-flip is a result, and it needs this citation to be stated crisply.

**[29] G. Marti and C. Studer**, "Mitigating smart jammers in multi-user MIMO," *IEEE
Transactions on Signal Processing*, vol. 71, pp. 756–771, 2023. DOI: `10.1109/TSP.2023.3246226`
- Venue: IEEE TSP.
- **Relates**: state-of-the-art *model-based* defense — joint jammer estimation, channel
  estimation and data detection, effective against smart jammers without prior knowledge of
  the attack type. The strongest existing counter-argument to any "jammer wins" claim, and
  (usefully) from ETH.
- **Differs**: requires a multi-antenna receiver and exploits the jammer's spatial subspace;
  our single-antenna setting cannot invoke it. **This must be named as an explicit scope
  limitation**: our stealth result holds against a receiver-side *detector*, not against a
  multi-antenna *mitigator*. Stating this pre-empts the sharpest available reviewer objection.

**[30] H. Bouzabia, T. N. Do, and G. Kaddoum**, "Deep learning-enabled deceptive jammer
detection for low probability of intercept communications," *IEEE Systems Journal*, vol. 17,
no. 2, pp. 2166–2177, Jun. 2023. DOI: `10.1109/JSYST.2022.3180481`
- Venue: IEEE Systems Journal.
- **Relates**: detection of jammers that are themselves designed to be hard to intercept —
  the defender-side counterpart of our stealthy attacker.
- **Differs**: chirp/PN deceptive jamming in a radar-communication context; not OFDM in-band
  interference. Optional; cut before [24]–[26].

---

## 5. Adversarial machine learning at the physical layer

**[31] M. Sadeghi and E. G. Larsson**, "Adversarial attacks on deep-learning based radio signal
classification," *IEEE Wireless Communications Letters*, vol. 8, no. 1, pp. 213–216, Feb.
2019. DOI: `10.1109/LWC.2018.2867459`
- Venue: IEEE WCL.
- **Relates**: the foundational wireless-evasion result, and the origin of the claim we
  inherit — adversarial perturbations are far more power-efficient than classical jamming.
  Also introduces *universal* (input-agnostic) perturbations, which map onto our
  blind-tier attacker.
- **Differs**: the victim is a modulation classifier and success is misclassification; the
  legitimate link's BER is not the objective. Our attack must move two quantities at once.
- **Use**: **run a direct analogue as an ablation baseline** — a perturbation optimized *only*
  to minimize the detector's jammed-probability, with no BER term. It should be stealthy and
  useless, which quantifies how much of the frontier comes from the joint objective rather
  than from evasion alone.

**[32] B. Kim, Y. E. Sagduyu, K. Davaslioglu, T. Erpek, and S. Ulukus**, "Channel-aware
adversarial attacks against deep learning-based wireless signal classifiers," *IEEE
Transactions on Wireless Communications*, vol. 21, no. 6, pp. 3868–3880, Jun. 2022. DOI:
`10.1109/TWC.2021.3124855`
- Venue: IEEE TWC.
- **Relates**: proves that adversarial perturbations designed without the channel *fail* over
  the air — the published justification for our channel pre-compensation
  `S_j = d \cdot H_0 / G_j` and for the three-tier CSI threat model.
- **Differs**: again a modulation classifier; the perturbation must stay small to remain
  imperceptible, not large enough to break the link. Also: their channel-awareness improves
  attack success, whereas our matched-detectability analysis shows channel-aware *subcarrier
  selection* buys nothing once detectability is held fixed — a genuinely new and slightly
  counter-intuitive negative result worth contrasting with [32] explicitly.

**[33] B. Flowers, R. M. Buehrer, and W. C. Headley**, "Evaluating adversarial evasion attacks
in the context of wireless communications," *IEEE Transactions on Information Forensics and
Security*, vol. 15, pp. 1102–1113, 2020. DOI: `10.1109/TIFS.2019.2934069`
- Venue: IEEE TIFS.
- **Relates**: the methodological reference for evaluating wireless evasion attacks: attacks
  are taxonomized by *where* the adversary sits relative to the classifier input, and BER
  (not perceptual similarity) is argued to be the right metric. Both choices are ours.
- **Differs**: the attack still targets a signal classifier; BER enters as a *constraint* on
  the attacker's own communication, not as the objective to maximize.
- **Use**: cite when justifying the "detection happens at the RX on the composite
  pre-equalization frame" decision in §III — this is the paper that makes attack-location a
  first-class modeling choice.

**[34] F. Restuccia, S. D'Oro, A. Al-Shawabka, B. Costa Rendon, K. Chowdhury, S. Ioannidis, and
T. Melodia**, "Generalized wireless adversarial deep learning," *Computer Networks*, vol. 216,
p. 109264, Oct. 2022. DOI: `10.1016/j.comnet.2022.109264`
- Venue: Elsevier Computer Networks (Q1, respectable; the only non-IEEE/ACM/ML-venue journal
  in this list).
- **Relates**: formalizes the joint channel-and-waveform adversarial problem and enforces
  *physical realizability* — the perturbed waveform must remain a legal, decodable
  transmission. Our "realizable" requirement is the same constraint class.
- **Differs**: the goal is to fool a fingerprint/modulation classifier while preserving the
  attacker's own link; there is no victim BER and no detector of *interference*.

**[35] M. Z. Hameed, A. György, and D. Gündüz**, "The best defense is a good offense:
Adversarial attacks to avoid modulation detection," *IEEE Transactions on Information Forensics
and Security*, vol. 16, pp. 1074–1087, 2021. DOI: `10.1109/TIFS.2020.3025441`
- Venue: IEEE TIFS.
- **Relates**: **resolves the open triage question in `paper/README.md` — include it.** It is
  the cleanest published instance of the *dual* objective we adopt: perturb transmitted
  symbols so a learned classifier fails while the intended receiver still decodes. Structurally
  identical to "maximize BER subject to a detectability budget", with the two objectives
  swapped in sign.
- **Differs**: covert communication, not disruption — the perturbation protects a friendly
  link rather than destroying a hostile one, and the detector being evaded is an eavesdropper's
  classifier of the *legitimate* waveform. The mechanism transfers; the threat model does not.

**[36] M. DelVecchio, V. Arndorfer, and W. C. Headley**, "Investigating a spectral deception
loss metric for training machine learning-based evasion attacks," in *Proc. 2nd ACM Workshop on
Wireless Security and Machine Learning (WiseML @ WiSec)*, 2020, pp. 43–48. DOI:
`10.1145/3395352.3402624`
- Venue: ACM WiseML @ WiSec (workshop; small but the right community).
- **Relates**: **the closest published analogue of signature-shaping.** Adds an explicit loss
  term forcing the adversarial signal's *spectral shape* to match a legitimate signal, i.e.
  optimizing the perturbation to look benign in the frequency domain rather than merely to be
  small. This is the published precedent for "modulate per-OFDM-symbol so the spectrogram
  looks data-like".
- **Differs**: applied to preserving an attacker's own communications intent against an RFML
  classifier; no victim BER, no jamming detector, single-agent, and no notion of matched
  detectability. Our milestone-3 objective is this loss idea pointed at a jamming detector with
  a BER term attached.
- **Use**: **cite as prior art for the signature-shaping loss, and adopt its structure**
  (task loss + spectral-similarity term) rather than inventing a new formulation — it makes
  the method look situated instead of ad hoc.

**[37] N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami**, "Practical
black-box attacks against machine learning," in *Proc. ACM AsiaCCS*, 2017, pp. 506–519. DOI:
`10.1145/3052973.3053009`
- Venue: ACM AsiaCCS.
- **Relates**: the surrogate/transferability method that makes our black-box tier feasible —
  train a local substitute for the frozen detector, take gradients through it. Also the correct
  citation for classifying our threat model in standard adversarial-ML terms.
- **Differs**: image classifiers, no physical channel, no dual objective.
- **⚠** first author's initial is **N.** (Nicolas Papernot ➜ commonly cited as N. Papernot);
  verify the exact author-initial format your `IEEEtran` style expects.

---

## 6. Stealth as a detection-theoretic constraint (covert / LPD)

**[38] B. A. Bash, D. Goeckel, and D. Towsley**, "Limits of reliable communication with low
probability of detection on AWGN channels," *IEEE Journal on Selected Areas in Communications*,
vol. 31, no. 9, pp. 1921–1930, 2013. DOI: `10.1109/JSAC.2013.130923`
- Venue: IEEE JSAC.
- **Relates**: the square-root law — the canonical formalization of "undetectable" as a
  constraint on the *warden's detection error probability*, not on transmit power. This is the
  literature that justifies our insistence that stealth be measured against the defender's own
  false-alarm rate rather than at a convention like \(P(\text{det}) \le 0.5\), and it gives the
  correction we already made a principled name.
- **Differs**: information-theoretic, AWGN, analytic radiometric warden, and the hidden signal
  is a benign message whose value is rate. We hide a disruptive signal from a learned detector,
  so no analytic warden exists and the objective is BER, not rate — no square-root-law analogue
  is available, which is itself worth one sentence.

**[39] Y. Shi, K. Davaslioglu, and Y. E. Sagduyu**, "Generative adversarial network in the air:
Deep adversarial learning for wireless signal spoofing," *IEEE Transactions on Cognitive
Communications and Networking*, vol. 7, no. 1, pp. 294–303, Mar. 2021. DOI:
`10.1109/TCCN.2020.3010330`
- Venue: IEEE TCCN.
- **Relates**: a GAN minimax game played *over the air* to synthesize signals statistically
  indistinguishable from legitimate transmissions — the closest existing precedent for the
  GAN-like attacker/defender alternation the supervisor asked us to examine, and evidence that
  the alternation is trainable in a wireless setting.
- **Differs**: the goal is spoofing (be accepted as legitimate), not disruption; no victim BER;
  the discriminator is the adversary's own, not a deployed defender.
- **Use**: this is the paper to cite when answering the supervisor's "look at the GAN
  literature and see whether it transfers" item — it transfers as *framing* (round-based
  minimax) but not as *method*, because our defender cannot be differentiated through at
  execution time and has no labels at run time.

**[40] Y. Wen, Y. Huo, J. Li, J. Qian, and K. Wang**, "Generative adversarial network-aided
covert communication for cooperative jammers in CCRNs," *IEEE Transactions on Information
Forensics and Security*, vol. 20, pp. 1278–1289, 2025. DOI: `10.1109/TIFS.2025.3526058`
- Venue: IEEE TIFS.
- **Relates**: cooperative jammers + GAN + an explicit covertness constraint against a
  monitor — on paper the nearest neighbour to our axis combination.
- **Differs**: the jammers are *friendly* (they protect a primary transmission) and the covert
  payload is their own data; the optimization variable is power allocation/beamforming, not a
  waveform, and the warden is an analytic detection test. Effectiveness and stealth are not in
  tension there the way they are here.

**[41] M. R. Ziemann and C. A. Metzler**, "Adaptive LPD radar waveform design with generative
deep learning," *IEEE Transactions on Radar Systems*, vol. 3, pp. 417–429, 2025. DOI:
`10.1109/TRS.2025.3542283`
- Venue: IEEE Transactions on Radar Systems (new but fully peer-reviewed IEEE Transactions;
  part of this work appeared at Asilomar 2023).
- **Relates**: **resolves the second open triage question in `paper/README.md` — include it.**
  It is the strongest existing validation of the exact paradigm we apply: a generative model
  producing waveforms that are simultaneously *effective* (for sensing) and *statistically
  indistinguishable from the background* (undetectable), trained against a critic. Note that
  it has since been published in an IEEE Transactions, so the earlier "arXiv-only,
  cross-domain, might distract" objection no longer applies.
- **Differs**: radar sensing, not communications; the objective is detection performance
  against a target, not BER inflicted on a victim; and the "background" it must match is
  ambient RF rather than a legitimate OFDM frame. Cite as cross-domain corroboration in one
  sentence — do not over-claim similarity.

---

## 7. Arms races and the cost of adaptation (the headline framing)

**[42] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu**, "Towards deep learning
models resistant to adversarial attacks," in *Proc. ICLR*, 2018.
- Venue: ICLR. **Relates**: adversarial training is *the* defender adaptation mechanism our R1
  retraining round instantiates; the robust-optimization framing is the formal statement of our
  round-based min-max. **Differs**: vision, no physical constraints, no false-alarm notion.

**[43] L. Schmidt, S. Santurkar, D. Tsipras, K. Talwar, and A. Madry**, "Adversarially robust
generalization requires more data," in *Advances in Neural Information Processing Systems
(NeurIPS)*, vol. 31, 2018.
- Venue: NeurIPS. **Relates**: the theoretical basis for measuring defender adaptation cost in
  *samples* — robustness has strictly higher sample complexity than standard accuracy, and the
  gap is information-theoretic rather than an artifact of the training algorithm. This is what
  licenses "training samples required" as a legitimate cost axis in our round protocol.
  **Differs**: vision benchmarks; no operational metric like false-alarm rate at a given SNR.

**[44] D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, and A. Madry**, "Robustness may be at
odds with accuracy," in *Proc. ICLR*, 2019.
- Venue: ICLR. **Relates**: **the single best framing citation for our Phase-0.5 and
  milestone-2 results.** It proves a robustness/clean-accuracy tension in a simple setting;
  our detector exhibits the wireless instance of it — closing the in-band blind spot cost
  accuracy 99.8% → 90.5% and FAR 0% → 3.8%, and the channel-valid retrain pays ~20% FAR at
  5 dB. Citing this converts our numbers from "our retrain was mediocre" into "we measured the
  predicted trade-off in physical-layer units". **Differs**: classification-only, no detection
  operating point, no attacker cost.

**[45] N. Carlini and D. Wagner**, "Adversarial examples are not easily detected: Bypassing ten
detection methods," in *Proc. 10th ACM Workshop on Artificial Intelligence and Security
(AISec @ CCS)*, 2017. **⚠ page range not machine-verified** (commonly cited as pp. 3–14).
- Venue: ACM AISec. **Relates**: the reason a *detector* is a fragile defense class — ten
  proposed detection defenses fell to attacks adapted to them. Direct support for expecting a
  residual stealthy region to exist against any single learned detector, and for the suite
  (OR of complementary detectors) rather than one classifier. **Differs**: vision; detectors of
  adversarial examples, not of interference.

**[46] F. Tramèr, N. Carlini, W. Brendel, and A. Madry**, "On adaptive attacks to adversarial
example defenses," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33,
2020.
- Venue: NeurIPS. **Relates**: the evaluation discipline we must visibly follow — a defense is
  only meaningfully evaluated against an attack *adapted to it*, and thirteen published
  defenses failed this test. Our round protocol (R1 detector retrained on R0 attacks, R2
  attacker re-optimized against R1) is exactly this discipline applied symmetrically, and
  citing it pre-empts "your attacker never adapted to the retrained detector".
  **Differs**: vision; no cost accounting, which is our addition.

**[47] K. Davaslioglu, S. Kompella, T. Erpek, and Y. E. Sagduyu**, "Continual deep
reinforcement learning to prevent catastrophic forgetting in jamming mitigation," in *Proc.
IEEE MILCOM*, 2024, pp. 740–745. DOI: `10.1109/MILCOM61039.2024.10773861`
- Venue: IEEE MILCOM.
- **Relates**: **the closest existing measurement of adaptation cost in our own domain.** The
  defender must absorb new jammer patterns over time and pays for it in forgetting; PackNet-style
  continual learning is proposed to reduce that price. It is the strongest evidence that
  "adaptation is expensive" is a recognized, publishable problem in wireless — exactly the
  supervisor's headline.
- **Differs**: the adapting agent is an anti-jamming *policy* (channel selection), not a
  detector; the cost is reported as performance retention across tasks, not as ΔFAR/Δaccuracy
  on a detection operating point, and the attacker's re-optimization cost is not measured at
  all. Our contribution is to price both sides in operational PHY units.

---

## 8. Tooling and method components

**[48] J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vasudevan, M. Nimier-David, N. Binder,
G. Marcus, and A. Keller**, "Sionna: An open-source library for next-generation physical layer
research," arXiv:2203.11854, 2022. **⚠ preprint** — no peer-reviewed venue version; it is
nonetheless the canonical citation for the library (NVIDIA-maintained, widely cited).
- **Relates**: the simulation backend for the OFDM chain, TDL channel models and demapper.
  Belongs in Experiment Setup, not Related Work.
- **Differs**: tooling.

**[49] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios**, "Neural spline flows," in
*Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019, pp. 7511–7522.
- Venue: NeurIPS. **Relates**: the expressive policy class used in the sim03/sim03b/sim04
  generative-jammer experiments (exact log-likelihood, so it can serve as a policy head).
  **Differs**: density-estimation method, no wireless content. Cite only if the NSF ablation
  trail stays in the paper; under the current framing it is appendix material.

**Also relevant to the flow-policy claim (replaces the unverifiable `ward2019nf_rl`):**
- **P. N. Ward, A. Smofsky, and A. J. Bose**, "Improving exploration in soft-actor-critic with
  normalizing flows policies," *ICML 2019 Workshop on Invertible Neural Networks and
  Normalizing Flows*, arXiv:1906.02771. **⚠ workshop paper, not peer-reviewed proceedings; and
  the third author is Bose, not "Bhatt" as `paper/README.md` records.**
- **Peer-reviewed alternative, recommended instead**: **B. Mazoure, T. Doan, A. Durand,
  J. Pineau, and R. D. Hjelm**, "Leveraging exploration in off-policy algorithms via
  normalizing flows," in *Proc. Conference on Robot Learning (CoRL)*, PMLR vol. 100, 2020.
  Same claim (flow policies beat factored Gaussians for exploration in continuous control),
  citable venue.

---

## 9. What to compare against — concrete SOTA plan

The supervisor's "(which should also be ran for comparison)" is not optional. Split the
comparison into attacker baselines, detector baselines, and the protocol that makes the
numbers mean something.

### 9.1 Attacker baselines, in priority order

| # | Baseline | Source | Status | Cost | Why it earns its row |
|---|---|---|---|---|---|
| 1 | Barrage / broadband noise | [1] | implemented | — | The floor. Every jamming paper has it. |
| 2 | Sparse in-band, blind subcarrier choice | [1], [5] | implemented | — | Our main "dumb but tuned" comparator; already swept densely. |
| 3 | Genie channel-aware sparse | — | implemented | — | Upper bound on selection-based attacks; carries the negative result (≈0 gain at matched detectability). |
| 4 | **Pilot jamming / pilot nulling** | **[6]** | **missing** | ~2 h | The classical *protocol-aware* attacker, 2–7.5 dB more efficient than barrage. Its spectrogram signature is sparse and structured — likely very visible — which makes it a genuinely informative comparison, not a formality. |
| 5 | **Minimum-energy boundary attack (ours)** | derived from **[8]** | planned | ~10 h | The headline attacker. Frame it as [8] extended to fading + a detectability constraint. |
| 6 | **Evasion-only perturbation ablation** | **[31]**, **[37]** | **missing** | ~3 h | Optimize *only* to minimize \(p_{\text{cnn}}\), no BER term. Should be stealthy and harmless — isolates how much of the frontier is due to the joint objective. |
| 7 | **Signature-shaping (spectral-deception style)** | **[36]** | planned (m3) | ~8 h | The learned attacker. Adopt [36]'s loss structure (task term + spectral-similarity term) rather than a bespoke formulation. |
| 8 | Omniscient `jam = −2·tx` | — | implemented | — | Degenerate ceiling; keep as a sanity row only. |
| 9 | Learned discrete-config attacker | [10] | optional | ~6 h | Only if a reviewer demands a *learned* comparator in the proxy action space; #3 already bounds it. |

### 9.2 Detector baselines (the defender suite)

| # | Detector | Source | Status | Why |
|---|---|---|---|---|
| 1 | Energy detector, per-SNR FAR-calibrated | [28] | implemented | Classical; carries the role-flip result (dominant on lossless, redundant on faded). |
| 2 | Spectrogram CNN (EfficientNet-B0, complex STFT) | [23] | implemented (lossless + channel-valid) | The replicated SOTA and the primary target. |
| 3 | Kurtosis / higher-order statistics | — | implemented (sim02–04) | Cheap third classical member; already has an evasion history in our own ladder. |
| 4 | **Unsupervised autoencoder on clean faded frames** | **[25]** | **missing — highest-value addition** | Cannot be evaded by crossing a learned clean/jammed boundary, so it tests whether our stealth is real or an artifact of one decision surface. Trains on clean frames only; cheap. |
| 5 | **CWT-based compact CNN** | **[24]** | missing | Different time-frequency front-end ⇒ different blind spots. Upgrades the claim from "evades a CNN" to "evades two learned detectors". |
| 6 | Transformer detector | [26] | optional | Transferability check per [46]: does an attack tuned on our CNN transfer to a different architecture? |
| 7 | Multi-antenna jammer mitigation | [29] | **out of scope — declare it** | Requires MU-MIMO. Name explicitly as a limitation; do not let a reviewer find it first. |

### 9.3 Metrics and protocol

1. **BER *and* SER** at every operating point (supervisor asked for SER explicitly).
2. **Achievable frontier** \(\mathrm{BER}^*(\beta) = \max\{\mathrm{BER} : p_{\text{suite}} \le \beta\}\)
   reported as a *curve*; the headline number is read at \(\beta\) = the suite's own clean
   false-alarm rate, never at the \(\beta = 0.5\) convention. Formal justification: [38].
3. **Clean FAR alongside every detection rate**, per SNR — a jammer is only stealthy relative
   to a detector that is not already noisy (§III-E).
4. **ROC / AUC of the suite per attacker family**, so "the detector is blind to X" is a curve,
   not a single threshold. Motivated by [23]'s single-operating-point reporting being the gap.
5. **Adaptation cost per round** — defender: Δaccuracy, ΔFAR@\(E_b/N_0\), retraining samples,
   GPU-hours [43], [44], [47]; attacker: re-optimization compute and BER recovered.
6. **Cross-detector transfer**: attack optimized on detector A evaluated on detector B [37],
   and adaptive-attack discipline per [46].
7. **Report the negatives**: raw-IQ black-box policy gradient is untrainable from a scalar
   frame reward (contextualize with [20]); genie channel-aware selection gains ≈0 at matched
   detectability (contrast with [32], which found channel-awareness essential for a *different*
   objective).

---

## 10. `refs.bib` — corrections, deletions, and a ready-to-paste block

### 10.1 Fix these existing entries

| Key | Problem | Fix |
|---|---|---|
| `jamming_survey_2024` | `year={2024}` is wrong | → 2022, vol. 24, no. 2, pp. 767–809, DOI `10.1109/COMST.2022.3159185`. Consider renaming the key to `pirayesh2022jamming`. |
| `article` | Key is the literal word "article"; `volume={PP}, pages={1-1}` (early access) | → `zhang2025cooperative`, vol. 20, pp. 9220–9234, 2025 |
| `qin2025multi` | No volume/pages | → vol. 24, no. 9, pp. 8886–8898, Sep. 2025, DOI `10.1109/TMC.2025.3559123` |
| `abolhassani2025coordinated` | Cited as if published | Keep as `@misc`/`@article` with `note={preprint}`; arXiv:2512.16813 |
| `valianti2024cooperative` | Looks like placeholder pages — it is not | Verified correct: vol. 23, no. 12, pp. 12345–12359, DOI `10.1109/TMC.2024.3409050`. Add the DOI. |
| `wen2025generative` | No volume/pages | → vol. 20, pp. 1278–1289, 2025, DOI `10.1109/TIFS.2025.3526058` |
| `zhang2023detection` | Fine | Add DOI `10.1109/MILCOM58377.2023.10356126` |

### 10.2 Delete

- **`electronics14163307`** (MDPI *Electronics*) — supervisor explicitly flagged MDPI as a weak
  venue. Its content (VDN-based anti-jamming spectrum access for LEO) is covered by [15], [16].
- **`tong2025wirelessagent`** (LLM agents for wireless) — supervisor said cut the LLM content
  as tangential.
- **`djuhera2025r`** (R-SFLLM) — supervisor said cut as application-specific/out of scope.
- **`Nguyen2025_MARL_UAVRelay`** — arXiv preprint, UAV relay defense; adds nothing the
  published [13]–[17] do not cover.
- **`Li_Wu_Cui_Dong_Fang_Russell_2019`** (M3DDPG) — robust MARL, only relevant if the paper
  argues about MARL robustness; it does not.
- **`strasser2009novel`** (UFH thesis) — the draft's own note already says "not relevant to
  proposed method"; anti-jamming coordination, pre-learning era.
- **PyJama** — already decided: dropped from Related Work; mention in Experiment Setup beside
  Sionna [48] if at all.

### 10.3 New entries to add

```bibtex
@article{lohan2024survey,
  title={From 5G to 6G Networks: A Survey on AI-Based Jamming and Interference Detection and Mitigation},
  author={Lohan, Poonam and Kantarci, Burak and Ferrag, Mohamed Amine and Tihanyi, Norbert and Shi, Yi},
  journal={IEEE Open Journal of the Communications Society}, volume={5}, pages={3920--3974}, year={2024},
  doi={10.1109/OJCOMS.2024.3416808}}

@article{adesina2023adversarial,
  title={Adversarial Machine Learning in Wireless Communications Using RF Data: A Review},
  author={Adesina, Damilola and Hsieh, Chung-Chu and Sagduyu, Yalin E. and Qian, Lijun},
  journal={IEEE Communications Surveys \& Tutorials}, volume={25}, number={1}, pages={77--100}, year={2023},
  doi={10.1109/COMST.2022.3205184}}

@inproceedings{xu2005feasibility,
  title={The Feasibility of Launching and Detecting Jamming Attacks in Wireless Networks},
  author={Xu, Wenyuan and Trappe, Wade and Zhang, Yanyong and Wood, Timothy},
  booktitle={Proc. ACM MobiHoc}, pages={46--57}, year={2005}, doi={10.1145/1062689.1062697}}

@article{shahriar2015phy,
  title={PHY-Layer Resiliency in OFDM Communications: A Tutorial},
  author={Shahriar, Chowdhury and La Pan, Matthew and Lichtman, Marc and Clancy, T. Charles and
          McGwier, Robert and Tandon, Ravi and Sodagari, Shabnam and Reed, Jeffrey H.},
  journal={IEEE Communications Surveys \& Tutorials}, volume={17}, number={1}, pages={292--314}, year={2015},
  doi={10.1109/COMST.2014.2349883}}

@inproceedings{clancy2011efficient,
  title={Efficient OFDM Denial: Pilot Jamming and Pilot Nulling},
  author={Clancy, T. Charles},
  booktitle={Proc. IEEE International Conference on Communications (ICC)}, pages={1--5}, year={2011},
  doi={10.1109/ICC.2011.5962467}}

@inproceedings{lapan2012jamming,
  title={Jamming Attacks Against OFDM Timing Synchronization and Signal Acquisition},
  author={La Pan, Matthew J. and Clancy, T. Charles and McGwier, Robert W.},
  booktitle={Proc. IEEE MILCOM}, pages={1--7}, year={2012}, doi={10.1109/MILCOM.2012.6415749}}

@article{amuru2015optimal,
  title={Optimal Jamming Against Digital Modulation},
  author={Amuru, SaiDhiraj and Buehrer, R. Michael},
  journal={IEEE Transactions on Information Forensics and Security}, volume={10}, number={10},
  pages={2212--2224}, year={2015}, doi={10.1109/TIFS.2015.2451081}}

@inproceedings{zhang2019jam,
  title={Jam Sessions: Analysis and Experimental Evaluation of Advanced Jamming Attacks in MIMO Networks},
  author={Zhang, Liyang and Restuccia, Francesco and Melodia, Tommaso and Pudlewski, Scott M.},
  booktitle={Proc. ACM MobiHoc}, pages={61--70}, year={2019}, doi={10.1145/3323679.3326504}}

@article{amuru2016bandits,
  title={Jamming Bandits---A Novel Learning Method for Optimal Jamming},
  author={Amuru, SaiDhiraj and Tekin, Cem and van der Schaar, Mihaela and Buehrer, R. Michael},
  journal={IEEE Transactions on Wireless Communications}, volume={15}, number={4}, pages={2792--2808},
  year={2016}, doi={10.1109/TWC.2015.2510643}}

@article{erpek2019deep,
  title={Deep Learning for Launching and Mitigating Wireless Jamming Attacks},
  author={Erpek, Tugba and Sagduyu, Yalin E. and Shi, Yi},
  journal={IEEE Transactions on Cognitive Communications and Networking}, volume={5}, number={1},
  pages={2--14}, year={2019}, doi={10.1109/TCCN.2018.2884910}}

@inproceedings{rashid2018qmix,
  title={QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning},
  author={Rashid, Tabish and Samvelyan, Mikayel and Schroeder de Witt, Christian and Farquhar, Gregory and
          Foerster, Jakob and Whiteson, Shimon},
  booktitle={Proc. International Conference on Machine Learning (ICML)}, series={PMLR}, volume={80},
  pages={4295--4304}, year={2018}}

@inproceedings{yu2022mappo,
  title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={35},
  pages={24611--24624}, year={2022}}

@inproceedings{terry2021pettingzoo,
  title={PettingZoo: Gym for Multi-Agent Reinforcement Learning},
  author={Terry, J. K. and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and
          Sullivan, Ryan and Santos, Luis S. and Dieffendahl, Clemens and Horsch, Caroline and
          Perez-Vicente, Rodrigo and Williams, Niall and Lokesh, Yashas and Ravi, Praveen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={34}, year={2021}}

@article{bettini2024benchmarl,
  title={BenchMARL: Benchmarking Multi-Agent Reinforcement Learning},
  author={Bettini, Matteo and Prorok, Amanda and Moens, Vincent},
  journal={Journal of Machine Learning Research}, volume={25}, number={217}, pages={1--10}, year={2024}}

@article{li2022jamming_ofdm,
  title={Jamming Detection and Classification in OFDM-Based UAVs via Feature- and Spectrogram-Tailored Machine Learning},
  author={Li, Yuchen and Pawlak, Joshua and Price, Jeffrey and Al Shamaileh, Khair and Niyaz, Quamar and
          Paheding, Sidike and Devabhaktuni, Vijay},
  journal={IEEE Access}, volume={10}, pages={16859--16870}, year={2022},
  doi={10.1109/ACCESS.2022.3150020}}

@inproceedings{varotto2024detecting,
  title={Detecting 5G Signal Jammers Using Spectrograms with Supervised and Unsupervised Learning},
  author={Varotto, Matteo and Valentin, Stefan and Tomasin, Stefano},
  booktitle={Proc. IEEE ICC Workshops}, pages={767--772}, year={2024},
  doi={10.1109/ICCWorkshops59551.2024.10615325}}

@article{viana2025pca,
  title={PCA-Featured Transformer for Jamming Detection in 5G UAV Networks},
  author={Viana, Joseanne and Farkhari, Hamed and Sebasti{\~a}o, Pedro and Gil Jim{\'e}nez, V{\'i}ctor P. and others},
  journal={IEEE Open Journal of the Communications Society}, volume={6}, pages={9287--9303}, year={2025},
  doi={10.1109/OJCOMS.2025.3619817}}

@inproceedings{arcangeloni2024jamming,
  title={Jamming Detection in MIMO-OFDM ISAC Systems Using Variational Autoencoders},
  author={Arcangeloni, Luca and Testi, Enrico and Giorgetti, Andrea},
  booktitle={Proc. IEEE International Symposium on Systems Engineering (ISSE)}, pages={1--7}, year={2024},
  doi={10.1109/ISSE63315.2024.10741110}}

@article{urkowitz1967energy,
  title={Energy Detection of Unknown Deterministic Signals},
  author={Urkowitz, Harry},
  journal={Proceedings of the IEEE}, volume={55}, number={4}, pages={523--531}, year={1967},
  doi={10.1109/PROC.1967.5573}}

@article{marti2023mitigating,
  title={Mitigating Smart Jammers in Multi-User MIMO},
  author={Marti, Gian and Studer, Christoph},
  journal={IEEE Transactions on Signal Processing}, volume={71}, pages={756--771}, year={2023},
  doi={10.1109/TSP.2023.3246226}}

@article{bouzabia2023deep,
  title={Deep Learning-Enabled Deceptive Jammer Detection for Low Probability of Intercept Communications},
  author={Bouzabia, Hamda and Do, Tri Nhu and Kaddoum, Georges},
  journal={IEEE Systems Journal}, volume={17}, number={2}, pages={2166--2177}, year={2023},
  doi={10.1109/JSYST.2022.3180481}}

@article{sadeghi2019adversarial,
  title={Adversarial Attacks on Deep-Learning Based Radio Signal Classification},
  author={Sadeghi, Meysam and Larsson, Erik G.},
  journal={IEEE Wireless Communications Letters}, volume={8}, number={1}, pages={213--216}, year={2019},
  doi={10.1109/LWC.2018.2867459}}

@article{kim2022channel,
  title={Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers},
  author={Kim, Brian and Sagduyu, Yalin E. and Davaslioglu, Kemal and Erpek, Tugba and Ulukus, Sennur},
  journal={IEEE Transactions on Wireless Communications}, volume={21}, number={6}, pages={3868--3880},
  year={2022}, doi={10.1109/TWC.2021.3124855}}

@article{flowers2020evaluating,
  title={Evaluating Adversarial Evasion Attacks in the Context of Wireless Communications},
  author={Flowers, Bryse and Buehrer, R. Michael and Headley, William C.},
  journal={IEEE Transactions on Information Forensics and Security}, volume={15}, pages={1102--1113},
  year={2020}, doi={10.1109/TIFS.2019.2934069}}

@article{restuccia2022generalized,
  title={Generalized Wireless Adversarial Deep Learning},
  author={Restuccia, Francesco and D'Oro, Salvatore and Al-Shawabka, Amani and Costa Rendon, Bruno and
          Chowdhury, Kaushik and Ioannidis, Stratis and Melodia, Tommaso},
  journal={Computer Networks}, volume={216}, pages={109264}, year={2022},
  doi={10.1016/j.comnet.2022.109264}}

@article{hameed2021offense,
  title={The Best Defense Is a Good Offense: Adversarial Attacks to Avoid Modulation Detection},
  author={Hameed, Muhammad Zaid and Gy{\"o}rgy, Andr{\'a}s and G{\"u}nd{\"u}z, Deniz},
  journal={IEEE Transactions on Information Forensics and Security}, volume={16}, pages={1074--1087},
  year={2021}, doi={10.1109/TIFS.2020.3025441}}

@inproceedings{delvecchio2020spectral,
  title={Investigating a Spectral Deception Loss Metric for Training Machine Learning-Based Evasion Attacks},
  author={DelVecchio, Matthew and Arndorfer, Vanessa and Headley, William C.},
  booktitle={Proc. 2nd ACM Workshop on Wireless Security and Machine Learning (WiseML)},
  pages={43--48}, year={2020}, doi={10.1145/3395352.3402624}}

@inproceedings{papernot2017practical,
  title={Practical Black-Box Attacks Against Machine Learning},
  author={Papernot, Nicolas and McDaniel, Patrick and Goodfellow, Ian and Jha, Somesh and
          Celik, Z. Berkay and Swami, Ananthram},
  booktitle={Proc. ACM Asia Conference on Computer and Communications Security (AsiaCCS)},
  pages={506--519}, year={2017}, doi={10.1145/3052973.3053009}}

@article{bash2013limits,
  title={Limits of Reliable Communication with Low Probability of Detection on AWGN Channels},
  author={Bash, Boulat A. and Goeckel, Dennis and Towsley, Don},
  journal={IEEE Journal on Selected Areas in Communications}, volume={31}, number={9},
  pages={1921--1930}, year={2013}, doi={10.1109/JSAC.2013.130923}}

@article{sagduyu2021gan,
  title={Generative Adversarial Network in the Air: Deep Adversarial Learning for Wireless Signal Spoofing},
  author={Shi, Yi and Davaslioglu, Kemal and Sagduyu, Yalin E.},
  journal={IEEE Transactions on Cognitive Communications and Networking}, volume={7}, number={1},
  pages={294--303}, year={2021}, doi={10.1109/TCCN.2020.3010330}}

@article{ziemann2025lpd,
  title={Adaptive LPD Radar Waveform Design With Generative Deep Learning},
  author={Ziemann, Matthew R. and Metzler, Christopher A.},
  journal={IEEE Transactions on Radar Systems}, volume={3}, pages={417--429}, year={2025},
  doi={10.1109/TRS.2025.3542283}}

@inproceedings{madry2018towards,
  title={Towards Deep Learning Models Resistant to Adversarial Attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  booktitle={Proc. International Conference on Learning Representations (ICLR)}, year={2018}}

@inproceedings{schmidt2018adversarially,
  title={Adversarially Robust Generalization Requires More Data},
  author={Schmidt, Ludwig and Santurkar, Shibani and Tsipras, Dimitris and Talwar, Kunal and Madry, Aleksander},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={31}, year={2018}}

@inproceedings{tsipras2019robustness,
  title={Robustness May Be at Odds with Accuracy},
  author={Tsipras, Dimitris and Santurkar, Shibani and Engstrom, Logan and Turner, Alexander and Madry, Aleksander},
  booktitle={Proc. International Conference on Learning Representations (ICLR)}, year={2019}}

@inproceedings{carlini2017detected,
  title={Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods},
  author={Carlini, Nicholas and Wagner, David},
  booktitle={Proc. 10th ACM Workshop on Artificial Intelligence and Security (AISec)}, year={2017},
  doi={10.1145/3128572.3140444}}

@inproceedings{tramer2020adaptive,
  title={On Adaptive Attacks to Adversarial Example Defenses},
  author={Tram{\`e}r, Florian and Carlini, Nicholas and Brendel, Wieland and Madry, Aleksander},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={33}, year={2020}}

@inproceedings{davaslioglu2024continual,
  title={Continual Deep Reinforcement Learning to Prevent Catastrophic Forgetting in Jamming Mitigation},
  author={Davaslioglu, Kemal and Kompella, Sastry and Erpek, Tugba and Sagduyu, Yalin E.},
  booktitle={Proc. IEEE MILCOM}, pages={740--745}, year={2024},
  doi={10.1109/MILCOM61039.2024.10773861}}

@misc{hoydis2022sionna,
  title={Sionna: An Open-Source Library for Next-Generation Physical Layer Research},
  author={Hoydis, Jakob and Cammerer, Sebastian and Ait Aoudia, Fay{\c{c}}al and Vasudevan, Avinash and
          Nimier-David, Merlin and Binder, Nikolaus and Marcus, Guillermo and Keller, Alexander},
  year={2022}, eprint={2203.11854}, archivePrefix={arXiv}, primaryClass={cs.IT}, note={preprint}}

@inproceedings{durkan2019nsf,
  title={Neural Spline Flows},
  author={Durkan, Conor and Bekasov, Artur and Murray, Iain and Papamakarios, George},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={32},
  pages={7511--7522}, year={2019}}

@inproceedings{mazoure2020leveraging,
  title={Leveraging Exploration in Off-Policy Algorithms via Normalizing Flows},
  author={Mazoure, Bogdan and Doan, Thang and Durand, Audrey and Pineau, Joelle and Hjelm, R. Devon},
  booktitle={Proc. Conference on Robot Learning (CoRL)}, series={PMLR}, volume={100}, year={2020}}
```

---

## 11. Residual risks and unverified fields

Short list, so nothing is silently assumed:

1. **[45] Carlini & Wagner (AISec 2017)** — venue, authors, title and DOI verified; the page
   range `3--14` is the commonly cited one but was not machine-confirmed. The BibTeX entry
   above omits pages, which `IEEEtran` handles fine.
2. **[26] Viana *et al.*** — verified via Crossref, but the author list is longer than four; the
   entry uses `and others`. Expand it from IEEE Xplore before submission.
3. **[16] Abolhassani *et al.*** — arXiv only as of this review. Re-check for a venue version.
4. **[48] Sionna** — arXiv only by design; the library has no journal paper.
5. **Ward *et al.* flow-policy workshop paper** — workshop, not proceedings, and
   `paper/README.md` has the third author's name wrong (Bose, not Bhatt). Prefer
   `mazoure2020leveraging`.
6. **NeurIPS/ICLR entries ([18], [20], [21], [42]–[44], [46], [49])** — venue and year verified;
   page ranges for NeurIPS volumes come from the proceedings listings rather than Crossref
   (Crossref coverage of NeurIPS is patchy). ICLR has no page numbers by convention.
7. **Novelty check** — targeted searches for prior work combining a *learned jammer*, a
   *learned jamming detector*, and an effectiveness-vs-detectability frontier returned nothing
   matching. The nearest neighbours are [35] (dual objective, transmitter-side, covert),
   [36] (spectral-shaped evasion, no victim BER) and [41] (dual objective, radar). That is a
   defensible gap statement, but it is an absence-of-evidence result: re-run the check shortly
   before submission.
