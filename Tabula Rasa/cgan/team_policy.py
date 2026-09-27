"""
D4b -- the LEARNED coordination policy (README §4.2 Q12). D4a (§3.3i) measured the
delay-decay curve by hand: a leader holds the victim's frame timing and followers
get it with error sigma, equal received split, no geometry. This learns the two
levers Q12 leaves to the policy, with minimal inductive bias:

  * u_k in [0, 1]  -- the fraction of a PER-DRONE cap P_J/K the drone transmits, so
                      a drone may go quiet (a weak drone that adds detectable power
                      but little BER should switch off). The per-drone cap is the
                      hard constraint (sigmoid); power never enters the loss (§2.8).
  * delta_k        -- a transmit ADVANCE [symbols] that must compensate the drone's
                      own propagation delay tau_k = d_kR/c. The policy is told only
                      its position relative to R, not that delay grows with distance;
                      learning delta_k ~ tau_k from geometry is the point (Q12 (a)).

OBSERVATION (own-geometry arm, the user's minimal-bias choice): each drone sees only
its 3-D position relative to R (normalised by the box) and its own sync level sigma_k
-- 0 for the leader, sigma for a follower, which is how a shared-parameter MLP tells
the two apart. Nothing about teammates (that is the second arm, Q12, not built here).
CTDE: one MLP, trained centrally by DIRECT GRADIENT over random drops, run per drone.

WHY DIRECT GRADIENT AND NOT RL. delta feeds a differentiable fractional delay (an FFT
phase ramp on the padded jammer stream -- channel.py's sinc taps reach only 8 samples,
Q12); u feeds a sqrt() amplitude. Both reach the frozen CNN-targeted D2 waveform (§3.3f,
task14 = spec_cnn_b10, fresh z per drone) and the log E[BER] / soft-P(det) surrogates of
D2 (attacks.log_expected_ber, detectors.soft_pdet). So the gradient of the D2 reward
    loss = -( log E[BER] - beta * P_det_soft )
reaches (u, delta) with no policy-gradient RL over IQ -- the A.5 wall does not apply
(§2.8 CGAN-track exception is not even needed: the actions are two scalars, not IQ).

    sbatch --array=0-3 submit_team_policy.sh                # train: (SNR 15,30) x (beta 0,10)
    sbatch --array=0-3 submit_team_policy.sh --mode eval    # eval the four arms on the 50 test drops

Arms compared at matched detectability (README §2.7), pooled over the 50 fixed test
drops like D4a: LEARNED (this policy) | HEURISTIC (delta_k = tau_k, full power) |
UNCOORDINATED (no shared clock: uniform offset, full power) | CEILING (one jammer at
the team's received total -- the containment bound, §3.3h). Output mirrors
team_fading.py so team_figures.py --policy reads it.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import attacks
import baselines
import calibrate_snr
import channel
import detectors
import link as lk
import models
import scene

C_LIGHT = 299_792_458.0
K_TEAM = 4                       # the minimal first cut is K = 4 = K_MAX (user, 2026-09-26)
LEADER = 0                       # drone 0 holds the victim's frame timing (sigma_k = 0)
DELTA_MAX = 6.0                  # symbols; delta = DELTA_MAX*sigmoid, bounding it to the physical
                                 # propagation range (box diagonal 4.74 symbols) -- unbounded delta
                                 # escaped to a generator-phase confound (diag: the shallow delta=tau
                                 # well is ~0.6 loss units, the escape sat lower)
SIGMA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]   # D4a's grid (Q12)
SHIFT_SYM = 128                  # uncoordinated offset period [symbols] = one frame (scene.N_SYM)
BUDGET_BAND = (-30.0, -10.0)     # training T_full band [dB]: straddles the team's BER onset
EVAL_BUDGET_GRID = [float(v) for v in np.arange(-38.0, -6.0, 1.0)]     # T_full sweep [dB]
EVAL_SIGMAS = [0.0, 1.0, 4.0, 16.0]                                    # the knee and past it
ALPHA = detectors.HEADLINE_ALPHA

ART = os.path.join(scene.ART, "..", "team_policy")
D2_GEN = os.path.join(scene.ART, "..", "gan", "run001", "task14_G.pt")  # spec_cnn_b10 (§3.3f)
TASKS = [(snr, beta) for snr in (15.0, 30.0) for beta in (0.0, 10.0)]


# ---------------------------------------------------------------- differentiable delay
def frac_delay(x, d_samples):
    """
    Delay each row of x [F, n] (complex) by d_samples[F] (real, may be fractional or
    negative) via an FFT phase ramp X(f) * exp(-2j pi f d). Circular over the padded
    stream, which is why the jammer enters PADDED and is cropped after. Differentiable
    in d_samples -- the only continuous path to delta_k the sinc-tap channel lacks.
    """
    n = x.shape[-1]
    f = torch.fft.fftfreq(n, device=x.device)                       # cycles / sample
    ramp = torch.exp(-2j * math.pi * f[None, :] * d_samples[:, None].to(torch.float64))
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * ramp.to(torch.complex64), dim=-1)


# ---------------------------------------------------------------- the policy
class Policy(nn.Module):
    """obs (pos rel R [3], sigma_k feature [1]) -> (u in [0,1], delta [symbols]). Shared."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        out = self.net(obs)
        return torch.sigmoid(out[..., 0]), DELTA_MAX * torch.sigmoid(out[..., 1])


# ---------------------------------------------------------------- geometry (analytic LOS)
def geom(positions):
    """
    positions [F, 2+K, 3] (rows T, R, J1..JK) -> per-drone (tau [F,K] symbols,
    g [F,K] free-space gain up to the shared lambda/4pi constant, which cancels in
    the split ratio). LOS free space; verify.test_scene checks RT == free_space_gain,
    so the analytic values match the cached RT drops used at eval to <1%.
    """
    R = positions[:, 1:2, :]
    J = positions[:, 2:, :]
    d = (J - R).norm(dim=-1).clamp_min(scene.MIN_SEPARATION)         # [F, K] metres
    tau = d / C_LIGHT * scene.SYMBOL_RATE                            # [F, K] symbols
    g = 1.0 / d.pow(2)
    return tau, g


def obs_of(positions, sig_k):
    """[F, K, 4]: (J - R) / box, then log1p(sigma_k). Own geometry only (Q12)."""
    rel = (positions[:, 2:, :] - positions[:, 1:2, :]) / torch.tensor(
        scene.BOX, device=positions.device, dtype=positions.dtype)
    return torch.cat([rel, torch.log1p(sig_k).unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------- the team jammer at R
def team_jammer(L, G, scale, positions, T_full_db, sigma, arm, pol=None, grad=False, delta_offset=0.0):
    """
    The summed received jammer of K drones on the victim's grid, [F, waveform_length].

    T_full_db: the FULL-POWER nominal total received JSR [dB] (the x-axis). Split by
    gain (equal transmit power -> received JSR_k^full = T_full * g_k / sum_j g_j), so
    at u = 1 the nominal total is exactly T_full and K = 1 reduces to the D-series
    single jammer. Each drone: fresh D2 waveform, delayed by async + (tau_k - delta_k
    + eps_k), scaled to u_k * JSR_k^full. Only delta_k (via frac_delay) and u_k (via
    the sqrt amplitude) carry gradient; geometry, eps and async are detached.

    arm: 'learned' (pol), 'heuristic' (delta=tau, u=1), 'uncoordinated' (no shared
    clock: uniform offset over the burst period, u=1).
    """
    dev, F, N = L.device, positions.shape[0], scene.N_SYM
    K = positions.shape[1] - 2
    tau, g = geom(positions)                                          # [F, K]
    jsr_full = 10.0 ** (float(T_full_db) / 10.0) * g / g.sum(-1, keepdim=True)   # [F, K] linear

    sig_k = torch.full((F, K), float(sigma), device=dev)
    sig_k[:, LEADER] = 0.0                                            # the leader is synced
    if arm == "learned":
        u, delta = pol(obs_of(positions, sig_k))                     # [F, K] each
    elif arm == "heuristic":
        u = torch.ones(F, K, device=dev)                             # compensate exactly, full power
        delta = tau.clone()                                          # delta_offset (diag): perturb delta = tau
        if isinstance(delta_offset, tuple):                          # (drone, offset): one follower only
            delta[:, delta_offset[0]] += float(delta_offset[1])
        else:
            delta = delta + float(delta_offset)
    elif arm == "uncoordinated":
        u, delta = torch.ones(F, K, device=dev), torch.zeros(F, K, device=dev)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    if arm == "uncoordinated":
        eps = torch.randint(0, SHIFT_SYM, (F, K), device=dev).to(torch.float64)   # no clock
    else:
        eps = torch.randn(F, K, device=dev, dtype=torch.float64) * sig_k.double()
        eps[:, LEADER] = 0.0

    length, a = L.waveform_length(N), L.active(N)
    total = None
    for k in range(K):
        x = attacks.gan_tx(L, F, N, G, scale, grad=False)            # [F, length+2*PAD] detached
        async_off = torch.rand(F, device=dev, dtype=torch.float64) * L.sps
        phase = torch.exp(2j * math.pi * torch.rand(F, device=dev)).to(torch.complex64)
        dly = async_off + (tau[:, k] - delta[:, k] + eps[:, k]) * L.sps       # [F] samples
        xk = frac_delay(x, dly) * phase[:, None]
        xk = xk[:, channel.PAD:channel.PAD + length]                 # crop to the victim frame
        p = xk[:, a].abs().pow(2).mean(-1, keepdim=True).detach()    # unit-waveform power
        amp = torch.sqrt((u[:, k] * jsr_full[:, k] * L.p_s).clamp_min(0.0)[:, None] / (p + 1e-30))
        jk = xk * amp.to(xk.dtype)
        total = jk if total is None else total + jk
    return total


# ---------------------------------------------------------------- training (direct gradient)
def clean_scale(L, dfd, snr_db, n=4096, seed=1):
    """Clean-frame std of the CNN statistic at this noise level -- soft_pdet's width."""
    lk.setup(L.device, seed=seed)
    out = attacks.frames(L, n, None, None, snr_db)
    s, _ = dfd.statistics(out["r"], out["z"], dets=["spec_cnn"])
    return float(s["spec_cnn"].std())


ACCUM = 8                        # micro-batches per update: the CNN P(det) noise at F=32 (~0.7 loss
                                 # units) buried the shallow delta=tau well (~0.6); F*ACCUM=256 sinks it


def train(L, dfd, G, scale, beta, snr_db, steps, F, seed):
    """
    One policy for (beta, snr_db). Each update sums ACCUM micro-batches at ONE sampled
    (sigma, T_full) -- effective batch F*ACCUM -- so the delta gradient clears the CNN
    noise. Returns the policy and a per-update history.
    """
    dev = L.device
    pol = Policy().to(dev)
    n0 = L.noise_var(snr_db)
    thr = dfd.threshold("spec_cnn", ALPHA)
    sc = clean_scale(L, dfd, snr_db)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    print(f"  CNN threshold {thr:.6g}, clean std {sc:.6g}; budget band {BUDGET_BAND} dB; "
          f"eff batch {F * ACCUM}", flush=True)
    hist, t0 = [], time.time()
    for step in range(steps):
        sigma = float(SIGMA_GRID[rng.integers(len(SIGMA_GRID))])
        T_full = float(rng.uniform(*BUDGET_BAND))
        opt.zero_grad(set_to_none=True)
        lb_acc = soft_acc = 0.0
        for m in range(ACCUM):
            lk.setup(dev, seed=seed * 1_000_000 + step * ACCUM + m)
            positions = torch.tensor(
                np.stack([scene.draw_positions(int(rng.integers(1 << 31)), n_nodes=2 + K_TEAM)
                          for _ in range(F)]), dtype=torch.float32, device=dev)
            bits, sym, x = L.modulate(F, scene.N_SYM)
            j = team_jammer(L, G, scale, positions, T_full, sigma, "learned", pol, grad=True)
            r = L.awgn(x, n0) + j
            log_ber = attacks.log_expected_ber(dict(bits=bits, z_nf=L.matched_filter(x + j, scene.N_SYM)), n0)
            s, _ = dfd.statistics(r, L.matched_filter(r, scene.N_SYM), dets=["spec_cnn"], grad=True)
            soft = detectors.soft_pdet(s["spec_cnn"], thr, sc)
            (-(log_ber - beta * soft) / ACCUM).backward()
            lb_acc += float(log_ber.detach()) / ACCUM
            soft_acc += float(soft.detach()) / ACCUM
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            hist.append(dict(step=step, sigma=sigma, T_full_db=T_full, loss=-(lb_acc - beta * soft_acc),
                             log_ber=lb_acc, soft_pdet=soft_acc))
            print(f"  step {step:4d}  sig {sigma:5.2f}  T {T_full:+6.1f}  loss {-(lb_acc - beta * soft_acc):.3f}  "
                  f"logE[BER] {lb_acc:.2f}  P_soft {soft_acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pol, hist


# ---------------------------------------------------------------- evaluation (pooled)
def make_rx(L, G, scale, drops_pos, arm, sigma, pol, seed=0):
    """A team receive model for baselines.measure: pool F random test drops per batch."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rx(link, sym, jsr_db_k):
        F = sym.shape[0]
        idx = torch.randint(0, drops_pos.shape[0], (F,), generator=gen)
        positions = drops_pos[idx].to(link.device)
        with torch.no_grad():
            return team_jammer(link, G, scale, positions, float(jsr_db_k[0]), sigma, arm, pol)

    return rx


def evaluate(L, dfd, G, scale, pol, snr_db, beta, frames, out_path):
    drops = scene.load_test_drops()
    pos = torch.tensor(drops["positions"], dtype=torch.float32)
    clean = baselines.measure(L, dfd, None, None, 4 * frames)
    print(f"clean: BER {clean['ber']:.3e}, FAR "
          f"{json.dumps({d: clean['pdet'][d]['0.05'] for d in clean['pdet']})}", flush=True)
    meta = dict(snr_db=snr_db, beta=beta, jsr_db=EVAL_BUDGET_GRID, jsr_axis="full-power total received JSR",
                n_frames=frames, n_sym=scene.N_SYM, K=K_TEAM, sigmas=EVAL_SIGMAS, n_drops=int(pos.shape[0]),
                dets=[d for d in dfd.dets if d != "lrt_noise"], leader=LEADER)
    res = dict(meta=meta, clean=clean, series={})
    t0 = time.time()

    def sweep(spec, key):
        pts = [baselines.measure(L, dfd, spec, [T], frames) for T in EVAL_BUDGET_GRID]
        res["series"][key] = pts
        with open(out_path, "w") as f:
            json.dump(res, f)                                         # checkpoint per series
        print(f"  {key:<24} max BER {max(p['ber'] for p in pts):.3e}  "
              f"CNN P(det) at max T {pts[-1]['pdet']['spec_cnn']['0.05']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    # CEILING: one jammer at the team's received total (D-series single jammer, aligned)
    sweep(dict(name="gan", G=G, scale=scale), "ceiling")
    # UNCOORDINATED: no shared clock (sigma-independent)
    sweep(dict(name="team", rx=make_rx(L, G, scale, pos, "uncoordinated", 0.0, None, seed=1)),
          "uncoordinated")
    # HEURISTIC and LEARNED: per sigma
    for sigma in EVAL_SIGMAS:
        sweep(dict(name="team", rx=make_rx(L, G, scale, pos, "heuristic", sigma, None, seed=2)),
              f"heuristic|sigma{sigma:g}")
        sweep(dict(name="team", rx=make_rx(L, G, scale, pos, "learned", sigma, pol, seed=3)),
              f"learned|sigma{sigma:g}")

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {os.path.relpath(out_path)} in {res['runtime_s']:.0f}s")


# ---------------------------------------------------------------- delta-landscape diagnostic
def diag(L, dfd, G, scale, beta, snr_db, T_full_db=-20.0, sigma=0.0, frames=4096, seed=7):
    """
    Map the training loss vs a COMMON offset Delta added to the heuristic's delta = tau,
    at fixed sigma and T_full, over random drops. If the minimum sits at Delta = 0 and is
    convex nearby, delta = tau is learnable by gradient; a flat or multi-modal curve is
    why the policy's delta wanders (Q12's periodic-delay risk). Prints the loss the
    trainer sees (log E[BER], soft P(det)) and the hard CNN P(det).
    """
    dev = L.device
    n0 = L.noise_var(snr_db)
    thr, sc = dfd.threshold("spec_cnn", ALPHA), clean_scale(L, dfd, snr_db)
    rng = np.random.default_rng(seed)
    print(f"delta landscape: SNR {snr_db:g} dB, beta {beta:g}, sigma {sigma:g}, T_full {T_full_db:g} dB, "
          f"{frames} drops (tau mean ~1.7, max ~4.1 symbols)", flush=True)
    print("  offsetting ALL drones (common shift off the victim grid) then FOLLOWER 1 only "
          "(relative misalignment, others exact):")
    print(f"  {'mode':>8}{'Delta[sym]':>11}{'logE[BER]':>12}{'soft_pdet':>11}{'loss':>10}{'hardP(det)_CNN':>16}")
    offsets = ([("all", D) for D in [-2, -1, -0.5, 0.0, 0.5, 1, 2, 4]]
               + [("foll1", D) for D in [-4, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2, 4]])
    for mode, D in offsets:
        lk.setup(dev, seed=seed * 977 + int(D * 100) + (0 if mode == "all" else 500))
        positions = torch.tensor(np.stack([scene.draw_positions(int(rng.integers(1 << 31)), n_nodes=2 + K_TEAM)
                                           for _ in range(frames)]), dtype=torch.float32, device=dev)
        bits, sym, x = L.modulate(frames, scene.N_SYM)
        off = D if mode == "all" else (1, D)
        with torch.no_grad():
            j = team_jammer(L, G, scale, positions, T_full_db, sigma, "heuristic", delta_offset=off)
            r = L.awgn(x, n0) + j
            lb = float(attacks.log_expected_ber(dict(bits=bits, z_nf=L.matched_filter(x + j, scene.N_SYM)), n0))
            s, _ = dfd.statistics(r, L.matched_filter(r, scene.N_SYM), dets=["spec_cnn"])
            soft = float(detectors.soft_pdet(s["spec_cnn"], thr, sc))
            hard = float(detectors.p_detect(s["spec_cnn"], thr))
        print(f"  {mode:>8}{D:>11.2f}{lb:>12.3f}{soft:>11.4f}{-(lb - beta * soft):>10.3f}{hard:>16.3f}",
              flush=True)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                    help=f"0..{len(TASKS) - 1}: (SNR 15,30) x (beta 0,10)")
    ap.add_argument("--mode", choices=["train", "eval", "diag"], default="train")
    ap.add_argument("--run", default="run001")
    ap.add_argument("--steps", type=int, default=600)      # 600 updates x ACCUM=8 micro-batches
    ap.add_argument("--train-frames", type=int, default=32)
    ap.add_argument("--eval-frames", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    snr_db, beta = TASKS[args.task]
    out_dir = os.path.join(ART, args.run)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"snr{snr_db:g}_b{beta:g}"
    device = lk.setup(seed=6000 + args.task)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = calibrate_snr.defender_at(L, snr_db)
    G, scale, _ = models.load_generator(D2_GEN, device)
    print(f"task {args.task} | {args.mode} | SNR {snr_db:g} dB | beta {beta:g} | device {device}", flush=True)

    if args.smoke:
        args.steps, args.eval_frames = 20, 64
        global EVAL_BUDGET_GRID, EVAL_SIGMAS
        EVAL_BUDGET_GRID, EVAL_SIGMAS = [-30.0, -18.0, -10.0], [0.0, 4.0]

    pol_path = os.path.join(out_dir, f"{tag}{'_smoke' if args.smoke else ''}_policy.pt")
    if args.mode == "diag":
        diag(L, dfd, G, scale, beta, snr_db)
        return
    if args.mode == "train":
        pol, hist = train(L, dfd, G, scale, beta, snr_db, args.steps, args.train_frames, seed=6000 + args.task)
        torch.save({"state_dict": pol.state_dict(), "snr_db": snr_db, "beta": beta,
                    "delta_max": DELTA_MAX, "history": hist}, pol_path)
        print(f"wrote {os.path.relpath(pol_path)}")
    else:
        pol = Policy().to(device)
        pol.load_state_dict(torch.load(pol_path, map_location=device)["state_dict"])
        pol.eval()
        out_path = os.path.join(out_dir, f"{tag}{'_smoke' if args.smoke else ''}.json")
        evaluate(L, dfd, G, scale, pol, snr_db, beta, args.eval_frames, out_path)


if __name__ == "__main__":
    main()
