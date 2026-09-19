"""
D2a -- the learned control tier: a shaped-noise jammer, optimised black-box by
CMA-ES against one detector at a time (README §2.10, §3.4).

Why it exists. A stealth-trained GAN (D2) that beats *fixed* classical jammers
mixes up two effects: having a stealth objective at all, and generating raw IQ.
This tier gets the same objective and the same detectors with a structured
48-parameter family (attacks.shaped: 32 log-PSD gains + a 16-symbol periodic
envelope), so GAN vs shaped isolates the hypothesis class.

    sbatch --array=0-9 submit_train_shaped.sh        # one task per row of TASKS
    sbatch submit_train_shaped.sh --smoke            # 2 tasks x 1 JSR x 20 generations

Per (target detector, beta, JSR):
  * the attacker is black-box and score-based: it sees the detector's per-frame
    scores, never its gradients or internals. Candidates are ranked by
        E[BER] - beta * P_det,
    README §2.8's reward and nothing else. P_det is the hard fraction of frames
    the target flags at its calibrated alpha = 0.05 threshold. E[BER] is exact
    over the AWGN (attacks.expected_ber), so it is the BER the frames would give
    averaged over the receiver noise, not a surrogate;
  * TIES. With hard P_det alone the ranking is exactly flat wherever every
    candidate is flagged in every frame and E[BER] underflows (a fixed jammer at
    SNR 30 dB leaves Q(~76) per bit): smoke job 2267047 stopped at generation 1
    at -20 dB on the CNN -- README §A.5's flat-reward wall. Ties are broken by
    1e-6 x a monotone map of the soft miss rate, mean sigmoid(-(stat - threshold)
    / clean std), taken in the log domain so it never saturates; 1e-6 is below the reward's
    resolution (one frame of P_det is beta/256 >= 4e-5; the confirm set resolves
    BER to ~1e-6), so it only orders candidates the reward cannot. beta = 0 is
    ranked by log E[BER], the same order without underflow;
  * power is NOT in the fitness: channel.receive scales every frame to exactly
    the JSR (equality form). With a <= budget a learner could blast ~0.7 % of
    frames at full power, inside the 2-sigma confirmation tolerance;
  * CMA-ES (pycma) from theta = 0 (= the barrage jammer), sigma0 = 1 in
    log-power units, common random numbers per generation (every candidate sees
    the same bits, AWGN, noise draw, offsets: lk.setup(seed) resets Sionna's and
    torch's generators). The returned theta is the distribution mean
    (xfavorite), which a lucky draw does not bias as it does xbest;
  * the returned jammer is re-measured on n_confirm fresh frames against ALL
    five detectors (baselines.measure), with theta = 0 on the same frames as the
    reference -- fitness(theta) >= fitness(0) must hold, since white noise is in
    the family.
Then, per task, the confirmed-stealthy pick per detector (baselines.confirm).

TASKS: 0 = beta 0 (effectiveness only, detector irrelevant); 1-8 = target x
beta in {0.01, 1}; 9 = the seed check -- kurtosis and CNN at beta 1 re-run with
another seed at three JSRs. beta >= 1 already means "stealth first" (BER stays
below ~0.05 over this JSR range, so any detection reduction outweighs it);
beta 0.01 trades 1 % of detection for 1e-4 of BER.

Outputs: artifacts/cgan/learned/run001/task{T}.json.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import: no OptiX on the cluster (scene.py)

import argparse
import json
import math
import os
import time

import cma
import numpy as np
import torch

import attacks
import baselines
import detectors
import link as lk
import scene

RUN = "run001"
OUT = os.path.join(scene.ART, "..", "learned", RUN)
TARGETS = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn"]
BETAS = [0.01, 1.0]
JSR_GRID = [float(v) for v in np.arange(-40.0, 0.5, 4.0)]
SEED_CHECK = dict(targets=["kurtosis", "spec_cnn"], beta=1.0, jsr=[-32.0, -20.0, -8.0])
TASKS = ([dict(target=None, beta=0.0)]
         + [dict(target=t, beta=b) for t in TARGETS for b in BETAS]
         + [dict(seed_check=True)])
ALPHA = detectors.HEADLINE_ALPHA
SIGMA0, BOUND = 1.0, 10.0          # log-power units; +-10 = +-43 dB, never reached in practice
TIE = 1e-6                         # tie-break weight (module docstring)


def shaped_spec(theta, tag="shaped"):
    return dict(name="shaped", theta=[float(v) for v in theta], tag=tag)


def clean_scale(L, dfd, target, n=2048, seed=1):
    """Std of the target's statistic on clean frames: the soft miss rate's width."""
    lk.setup(L.device, seed=seed)
    out = attacks.frames(L, n, None, None, lk.SNR_DB)
    s, _ = dfd.statistics(out["r"], out["z"], dets=[target])
    return float(s[target].std())


def fitness(L, dfd, theta, target, beta, jsr, n_frames, seed, scale):
    """
    (rank key, E[BER], P_det) on the frames of `seed` (common random numbers). The
    rank key is E[BER] - beta * P_det + TIE * soft miss rate (module docstring),
    or log E[BER] when target is None (beta = 0).
    """
    lk.setup(L.device, seed=seed)
    out = attacks.frames(L, n_frames, shaped_spec(theta), [jsr], lk.SNR_DB, noiseless=True)
    n0 = L.noise_var(lk.SNR_DB)
    ber = attacks.expected_ber(out, n0)
    if target is None:
        return attacks.expected_ber(out, n0, log=True), ber, None
    s, _ = dfd.statistics(out["r"], out["z"], dets=[target])
    thr = dfd.threshold(target, ALPHA)
    pdet = detectors.p_detect(s[target], thr)
    x = (s[target].double() - thr) / scale
    log_miss = float(torch.logsumexp(torch.nn.functional.logsigmoid(-x), 0) - math.log(x.numel()))
    # 1 / (1 - log miss) maps (-inf, 0] onto (0, 1], monotone: a tie-break needs only the order
    return ber - beta * pdet + TIE / (1.0 - log_miss), ber, pdet


def optimise(L, dfd, target, beta, jsr, gens, n_frames, n_confirm, seed):
    """One CMA-ES run; returns theta*, its history, and both confirm measurements."""
    t0 = time.time()
    # tolfun/tolfunhist off: a plateau must not end the run (module docstring, TIES)
    es = cma.CMAEvolutionStrategy(np.zeros(attacks.SHAPED_DIM), SIGMA0,
                                  {"seed": seed, "maxiter": gens, "bounds": [-BOUND, BOUND],
                                   "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10, "verbose": -9})
    scale = clean_scale(L, dfd, target) if target else None
    hist = []
    g = 0
    while not es.stop():
        cands = es.ask()
        res = [fitness(L, dfd, c, target, beta, jsr, n_frames, seed * 100_000 + g, scale) for c in cands]
        es.tell(cands, [-r[0] for r in res])               # cma minimises
        f = np.array([r[0] for r in res])
        best = int(f.argmax())
        hist.append(dict(best=float(f[best]), mean=float(f.mean()), ber_best=res[best][1],
                         pdet_best=res[best][2], sigma=float(es.sigma)))
        g += 1
    theta = es.result.xfavorite
    # re-measure theta* and theta = 0 on the same fresh frames, every detector
    conf_seed = seed * 100_000 + 99_999
    lk.setup(L.device, seed=conf_seed)
    m = baselines.measure(L, dfd, shaped_spec(theta), [jsr], n_confirm)
    lk.setup(L.device, seed=conf_seed)
    m0 = baselines.measure(L, dfd, shaped_spec(np.zeros(attacks.SHAPED_DIM)), [jsr], n_confirm)
    fit = (lambda p: p["ber"] - beta * p["pdet"][target][str(ALPHA)]) if target else (lambda p: p["ber"])
    return dict(target=target, beta=beta, jsr_db=jsr, seed=seed, theta=[float(v) for v in theta],
                generations=g, evaluations=int(es.result.evaluations),
                stop={k: str(v) for k, v in es.stop().items()},
                history=hist, confirm=m, confirm_theta0=m0,
                fitness_confirm=fit(m), fitness_confirm_theta0=fit(m0), runtime_s=time.time() - t0)


def run_task(L, dfd, task, gens, n_frames, n_confirm, jsr_grid, seed):
    if task.get("seed_check"):
        runs = [optimise(L, dfd, t, SEED_CHECK["beta"], j, gens, n_frames, n_confirm, seed + 7)
                for t in SEED_CHECK["targets"] for j in SEED_CHECK["jsr"]]
        return dict(task="seed_check", runs=runs)
    runs = []
    for jsr in jsr_grid:
        r = optimise(L, dfd, task["target"], task["beta"], jsr, gens, n_frames, n_confirm, seed)
        c, c0 = r["confirm"], r["confirm_theta0"]
        print(f"  {task['target']} beta {task['beta']:g} JSR {jsr:+5.0f} dB: {r['generations']} gens, "
              f"fitness {r['fitness_confirm']:.4g} (theta=0: {r['fitness_confirm_theta0']:.4g}), "
              f"BER {c['ber']:.2e} (theta=0 {c0['ber']:.2e}), P(det)@{ALPHA} " +
              " ".join(f"{d} {c['pdet'][d][str(ALPHA)]:.3f}" for d in c["pdet"]) +
              f"  ({r['runtime_s']:.0f}s)", flush=True)
        runs.append(r)
    picks = baselines.confirm(L, dfd, lambda i: shaped_spec(runs[i]["theta"]), [r["confirm"] for r in runs],
                              lambda i: [runs[i]["jsr_db"]], n_confirm, baselines.DETS, detectors.ALPHAS)
    return dict(task=task, runs=runs, confirmed=picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--frames", type=int, default=256, help="frames per fitness evaluation")
    ap.add_argument("--confirm", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true", help="tasks 0 and the CNN task, 1 JSR, 20 generations")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lk.setup(seed=2000 + args.task)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = baselines.Defender(L)
    t0 = time.time()
    if args.smoke:
        res = {}
        for t in (0, TASKS.index(dict(target="spec_cnn", beta=1.0))):
            print(f"smoke: task {t} = {TASKS[t]}", flush=True)
            res[t] = run_task(L, dfd, TASKS[t], 20, 64, 256, [-20.0], seed=2000 + t)
        path = os.path.join(OUT, "smoke.json")
    else:
        task = TASKS[args.task]
        print(f"task {args.task} = {task}", flush=True)
        res = run_task(L, dfd, task, args.gens, args.frames, args.confirm, JSR_GRID, seed=2000 + args.task)
        path = os.path.join(OUT, f"task{args.task}.json")
    with open(path, "w") as f:
        json.dump(dict(run=RUN, gens=args.gens, frames=args.frames, n_confirm=args.confirm, alpha=ALPHA,
                       dim=attacks.SHAPED_DIM, sigma0=SIGMA0, result=res, runtime_s=time.time() - t0), f)
    print(f"wrote {os.path.relpath(path)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
