# Run log

Every training run gets one row here. Update this file in the same commit/session as the run.

Convention for files in `artifacts/simXX/`:
- `runNNN.png` — training curves
- `runNNN_iq.png` — IQ scatter (if applicable)
- `runNNN_model.zip` — saved SB3 model (`model.save(...)`), if the run is worth reusing
  (`.pt` for non-SB3 / pure PyTorch models, e.g. sim03b's encoder+flow `state_dict`s)
- `runNNN_slurm_<jobid>.out/.err` — cluster logs (optional, only for notable runs)
- `tb/runNNN/` — TensorBoard event files (if the script logs with `SummaryWriter` / `tensorboard_log=`).
  View via `tensorboard --logdir artifacts/simXX/tb --port 6006` + SSH/VSCode port forwarding.

| Sim | Run | Policy | Key hyperparams | Steps | Result | Model saved? | Notes |
|---|---|---|---|---|---|---|---|
| 00 | - | n/a | N_LEGIT=2, JAMMER_POWER=50 | n/a | BER 0→0.5 when jammer on | no | observational baseline |
| 01 | 001 | MlpPolicy (Gaussian) | N=4, BETA=5 | 200k | BER→0.07, det→1%, power→1.5 | no | β too high, jammer too passive |
| 01 | 002 | MlpPolicy (Gaussian) | N=4, BETA=0.5 | 200k | BER→0.14, det→10%, power→1.9 | no | action space too small |
| 01 | 003 | MlpPolicy (Gaussian) | N=16, BETA=0.5 | 200k | BER→0.19, det→8-10%, power→2.2 | no | sweet spot |
| 01 | 004 | MlpPolicy (Gaussian) | N=16, BETA=0.5 | 200k | same as 003 | no | repeat, IQ=Gaussian cloud |
| 01 | 005 | MlpPolicy (Gaussian) | N=512, BETA=0.5 | 200k | BER flat ~0.237, det flat ~50% | no | action space too large to learn |
| 01 | 006 | MlpPolicy (Gaussian) | N=512, BETA=3 | 200k | same failure as 005 | no | higher beta didn't help |
| 01 | 007 | MlpPolicy (Gaussian) | N=512, BETA=3 | 1M | BER 0.24→0.27, det 40%→100%, power→3.7 | no | abandoned stealth entirely |
| 02 | 001 | MlpPolicy (Gaussian) | N=128, BETA=0.5, KURT_THRESH=-1.0, binary | 200k | det flat 100%, kurtosis stuck ~-0.25 | no | Gaussian policy walled off by kurtosis detector |
| 02 | 002 | MlpPolicy (Gaussian) | same as 001 | 200k | same, jam power 1.0→2.5 | no | agent gave up on stealth |
| 03 | 001 | FlowPolicy (NSF, zuko) | N=16, BETA=0.5, binary | 50k | det flat ~85%, kurtosis ~-0.25, KL→0.46 | no | binary penalty, no gradient |
| 03 | 002 | FlowPolicy (NSF, zuko) | N=16, BETA=2.0, GAMMA=0.02, binary | 50k | det flat ~85%, BER flat 0.25 | no | same failure |
| 03 | 003 | FlowPolicy (NSF, zuko) | N=16, BETA=2.0, GAMMA=0.02, continuous | 50k | det ~85%, kurtosis ~-0.25 | no | N=16 too noisy for kurtosis estimate |
| 03 | 004 | FlowPolicy (NSF, zuko) | N=128, BETA=2.0, GAMMA=0.02, continuous | 50k | det flat 100%, kurtosis ~-0.25 | no | NSF still Gaussian-shaped IQ; FlowPolicy ~1fps on CPU |
| 03b | - | direct gradient (no RL) | - | - | job 93449: empty output, timed out (1hr limit) | no | root cause: submit.sh requested `--gpus=1` but set `CUDA_VISIBLE_DEVICES=""`, job likely queued for a GPU node it never used and was killed before python started. Fixed: removed GPU request, added per-step timing + checkpointing every 500 steps, outputs now go to artifacts/sim03b |
| 03b | 001 | direct gradient, NSF (passes=2) | N=128, LAMBDA=2.0, GAMMA=0.02, KURT_THRESH=-1.0, no=1e-10 | 5000 (job 98432, ~7.25 steps/s, ~12min) | jam_power collapsed to ~0 by step 1250, BER→0, loss stuck at ~20 (clamp ceiling) for remaining 3750 steps | yes (run001_model.pt) | **Found bug:** demapper `no=1e-10` makes LLRs saturate `clamp(-20,20)` for any nonzero rx-tx deviation → BCE gradient ~0 → optimizer stuck once jam_power shrinks (kurtosis+power terms reward jam→0, ber term has no gradient to resist). Fixed: `no=1.0`, clamp tightened to (-10,10). Re-run needed. |

| 03c | 001 | GMMPolicy (K=8, PPO) | N=128, BETA=2.0, GAMMA=0.02, default init | 50k (job 98764) | unstable: approx_kl 0.1-1.8, clip_fraction~0.85, kurt~+0.4-0.6, det=1.0 | yes | wide log_std clamp (-5,2) allowed degenerate sharp components |
| 03c | 002 | GMMPolicy (K=8, PPO) | + log_std clamp (-2,0.5), target_kl=0.02 | 50k (job 98767) | PPO stable (approx_kl 0.04-0.16) but kurt~0-0.3, det=1.0, BER/power flat — unchanged outcome | yes | PPO mechanics fixed, RL outcome unchanged |
| 03c | 003 | GMMPolicy (K=8, PPO) | + ent_coef=0.01 | 50k (job 98768, 51,200 steps) | entropy_loss frozen at -390, outcome same as 002 | yes | entropy bonus negligible vs other loss terms at std≈1 |
| 03c | 004 | GMMPolicy (K=8, PPO) | bias-init log_std→-1.5 (jam starts near-silent) | 50k (job 98769) | det 1.0→0.3-0.4, BER 0.25→0.05, kurt pinned ~-1.0 (cliff edge), power 2.6→0.375 — big jump | yes | policy starts on same side of kurtosis-penalty cliff as optimum |
| 03c | 005 | GMMPolicy (K=8, PPO) | + lr 1e-4→3e-5 | 50k (job 98772) | approx_kl 0.08-0.14 (still 5-7x over target_kl), outcome identical to 004 | yes | lower LR helped approx_kl but not outcome |
| 03c | 006 | GMMPolicy (K=8, PPO) | + target_kl 0.02→0.15 | 30k (job 98775) | approx_kl pinned ~0.15, still ~2 updates/iteration, outcome identical | yes | target_kl=0.02 unrealistic for 128-symbol summed log_prob |
| 03c | 007 | GMMPolicy (K=8, PPO) | + ent_coef 0.01→0.1 | 30k (job 98776) | entropy_loss still frozen at -105, outcome identical | yes | ent_coef has no effect at this point |
| 03c | 008 | GMMPolicy (K=8, PPO) | target_kl=None | 30k (job 98777) | n_updates 10x higher (110-140), approx_kl~1.5, clip_fraction~0.87 — massive param movement, zero behavioral change | yes | **GMM permutation symmetry diagnosed — sim03c closed, see simulation03c/README.md** |

| frontier | - | none (pure inference sweep) | B=128, n_active×power×structure grid, frozen sim06 detector | n/a (job 101860, ~40s GPU) | **Detector is an out-of-band-emission detector.** In-band broadband n=52 pwr=8: P(det)=0.011 BER=0.36 vs out-of-band (all 64 bins) same power: P(det)=0.999. Blind in-band reaches BER=0.25@P(det)≤0.05, 0.42@≤0.5; omniscient BER=1.0@P(det)=0.002. Overturns sim06/07 "stealth impossible" wall (was OOB-leakage artifact). | outputs only (`artifacts/frontier/`) | Phase 0. See README "Frontier analysis" section. |

| frontier | 0.5 | EfficientNet-B0 retrain | clean + 4 classical + in-band jammers, 50 epochs, SGD | 1400 samples (job 101866, 4min) | Blind spot closes at a cost: in-band broadband P(det) 0.011→0.995, but acc 99.8→90.5%, FAR 0→3.8%, in-band DR only 68%. Re-swept frontier: stealthy BER (P(det)≤0.5) 0.42→0.02; genuine effectiveness–detectability tradeoff. | yes (`artifacts/frontier/detector/run001_best.pt`) | Phase 0.5. See README "Phase 0.5" section. Naive retrain — BER≈0 in-band samples labeled jammed inflates FAR. |

| 08 | m1 | none (channel-aware sweep) | TDL fading (model C, 100ns, 5.2GHz), EbN0 5-30dB, blind vs channel-aware, B=128 | n/a (job 101870, 54s) | Channel physics correct (clean floor 0.088→0.0003). Sparse in-band jammer = SNR-independent BER floor (~0.05-0.07 @ n=8). Channel-aware > blind up to +70% BER @ equal power (largest at low power). Lossless-trained detector invalid on faded channel (clean FAR≈13%) → milestone 2 retrain. | outputs only (`artifacts/sim08/frontier/`) | sim08 milestone 1. See README "Simulation 08" section. |

| frontier | recheck | complex-STFT CNN (run003) + energy detector | corrected spectrogram (complex, not real-part), suite eval, B=128 | detector retrain 100ep + sweep (job 102115) | **CORRECTION.** Out-of-band finding SURVIVES complex STFT (in-band P_cnn=0.086 vs OOB=1.000 @ same BER). BUT energy detector demolishes stealth: suite max BER @P≤0.5 = 0.005 (was 0.42 CNN-only). "CNN blind→jammer wins" dead on lossless; real question = sim08+suite under noise floor. | yes (`run003_best.pt`) | See README recheck section. Caveat: lossless = razor-sharp energy threshold. |

| 08 | m1b | + energy detector (per-SNR calibrated) | TDL fading, energy detector 1% FAR on faded clean, B=128 | job 102305, 56s | **Stealth REVIVED on realistic channel.** Jammer stays stealthy (P_energy≤0.5) at BER≈0.20–0.24 across 5–30dB (lossless gave 0.005). Noise floor gives cover the lossless channel didn't. Full-suite needs channel-valid CNN (m2). | outputs (`artifacts/sim08/frontier/stealth_vs_energy.png`) | This is the paper's positive result: stealthy-effective region exists only on realistic channels. |

| 08 | m2-det | channel-valid EfficientNet-B0 | faded clean+classical+in-band, complex-STFT, Eb/N0∈[5,30]dB per-batch, 60ep SGD | 2400 samples (job 102316, ~5min GPU) | **Channel-valid detector** (fixes m1's invalid-detector caveat). Held-out faded mix: acc 94.3% DR 91.2% FAR 2.3% F1 0.94 (vs lossless detector's ~13% FAR / no separation on faded). Fresh per-SNR: clean FAR 10%@5dB→1%@30dB, in-band DR 88%@5dB→76%@30dB (misses ~24% of in-band at high SNR = residual blind spot). | yes (`artifacts/sim08/detector/run001_best.pt`) | sim08 m2. `retrain_detector_channel.py`. In-band jammers routed through h_jam; classical added at RX. |

| 08 | m2-suite | full suite = channel-valid CNN OR energy (per-sample) | `frontier_channel.py` +p_suite, TDL fading, B=128 | job 102319, 27s | **Suite ≡ CNN on the realistic channel: energy detector catches 0 configs the CNN misses.** CNN shrinks stealthy region (P(det)≤0.5) from BER 0.22 (energy-only) → 0.065–0.11, but does NOT close it: residual sparse n≈16 pwr=1 jammer stays stealthy-effective (p_cnn≈0.4–0.47, p_en≈0.02). At 30dB floor 0.0004 → stealthy BER 0.065 (~160×; interference not noise). CNN clean FAR high@low SNR (~20%@5dB). | uses `run001_best.pt` | sim08 m2. Motivates learned jammer (m3) to widen the residual region. Plot: `artifacts/sim08/frontier/stealth_suite_vs_snr.png`. |

| 08 | m3-prep | matched-detectability (dense full-suite re-sweep) | `frontier_channel.py` --powers 0.25-6 (9) --n-active 1-52 (11), B=512, per-SNR checkpoint; `matched_detectability.py` post-proc | job 102390, ~12min GPU | **REFUTES channel-aware > blind.** At matched suite P(det)≤0.5 the CA gain collapses to ≈0 (-4.6/+1.5/-1.8/+6.2/+0.0% over 5-30dB; coarse grid's ±50% swings were noise) — m1 '+70%' was a matched-CONFIG artifact. Blind & CA achievable frontiers coincide. Residual stealthy region confirmed, slightly larger than m2 (max stealthy SUITE BER 0.145@5dB→0.094@30dB). Suite≡CNN softens @30dB (energy catches 1 config). | outputs (`artifacts/sim08/frontier_dense/matched_detectability.png`) | Channel-aware SC selection is a spent lever → m3 pivots to signature-shaping / multi-agent, not 'learn the genie'. See README matched-detectability section. |

**Open:** going forward, every `train_ppo.py` / `train.py` should write its `.png` outputs and (when worth keeping) `model.save(...)` checkpoint into `artifacts/simXX/`, and a new row should be added here.
