"""
E3 -- does a TEAM of jammers recover, in a fading channel, what a single jammer
loses there? The cheap pre-check before any multi-agent training: generators are
EVALUATED, not retrained (transfer, like E2 Tier 1).

WHY THIS AND NOT A STEALTH-MARL RUN
-----------------------------------
The detector sits at the victim receiver, so it sees exactly the jammer sum the
victim decodes. Anything K jammers deliver at R, one jammer with control of its
received signal could deliver too, so no team beats the best single jammer on
the BER-P(det) plane: the lossless single jammer (D1/D2 as measured, README
§3.3f) is the CEILING. A realistic channel without jammer CSI pulls a single
jammer below it -- its received power becomes random per frame. K jammers with
UNCORRELATED signals average their fades out (diversity order K); the same
signal would not, since the composite channel sum_k h_k fades like one link.
So this measures:

  * the gap      single jammer, faded vs lossless;
  * recovery     K = 2, 4 jammers with independent content (fresh z / symbols
                 each), an equal split of a fixed total power, independent fades;
  * timing       generators only: bursts ALIGNED on the victim's frame grid (the
                 delay to R is computable from positions) vs a random integer-
                 symbol SHIFT per jammer over the 128-symbol segment (no timing
                 coordination). Noise and pulsed jammers are stationary, so
                 timing is moot for them.

CHANNEL. On each jammer->R link only; T->R stays unfaded, which keeps the
detector-blunting effect of legitimate-link fading out of this check. Block-flat
per frame (a 128-symbol frame at 1 MBd is 0.13 ms, far below a 2.4 GHz drone
link's coherence time), independent across frames and jammers, E|h|^2 = 1.
Rician K = 10 dB is a LOS-dominated drone link; Rayleigh is the worst case.
Amplitude only: the async carrier phase (channel.async_draw) is already uniform.

POWER. The x-axis is the nominal TOTAL JSR: mean received jammer power over
signal power. Each of K jammers is projected to JSR - 10 log10 K exactly as
everywhere else, THEN faded, so the transmit side is fixed and only the channel
varies. TRAP: the lossless K = 1 reference holds its JSR exactly per frame,
while a faded jammer's received power varies from frame to frame -- free random
pulsing, which Amuru shows can help at low JSR. "Fading helps a single jammer"
is therefore a possible outcome, and it would be an artifact of the per-frame
reference, not a benefit of the channel.

    sbatch submit_team_fading.sh --smoke             # pipeline check (task 5)
    sbatch --array=0-5 submit_team_fading.sh         # (SNR 15, 30) x (lossless, rician10, rayleigh)

D4a (README §4.2 Q12) reuses this sweep for the DELAY-DECAY CURVE: lossless, the
generators only, their K > 1 timing swept over the inter-jammer link's error sigma
(leader exact, followers N(0, sigma^2) symbols; shift_start), 'shifted' as the floor:

    sbatch --array=0,3 submit_team_fading.sh --run run001 --art ../artifacts/cgan/team_timing \
        --jammers spec_cnn_b10,plain_run001,kurtosis_b1 --ks 1,2,4 \
        --sigmas 0,0.25,0.5,1,2,4,8,16,32,64

Detectors are the E2 per-SNR calibrations (calibrate_snr cache): CNN weights
frozen at 30 dB, colour scale and thresholds re-fitted per level.

Output: artifacts/cgan/team_fading/<run>/<snr tag>_<channel>.json.
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import argparse
import json
import math
import os
import time

import torch

import attacks
import baselines
import calibrate_snr
import channel
import link as lk
import models
import scene
import snr_ablation

# linear Rician K-factor per jammer->R link; None = no fading (the D-series channel)
CHANNELS = {"lossless": None, "rician10": 10.0, "rayleigh": 0.0}
TASKS = [(snr, ch) for snr in (15.0, 30.0) for ch in CHANNELS]
# noise, Amuru pulsed, D1 plain GAN, the CNN-targeted D2 of the headline (README
# §3.3f/§3.3g) and the kurtosis-targeted D2 (the one detector the GAN beats everywhere)
JAMMERS = ["noise", "pulsed_p0.1", "plain_run001", "spec_cnn_b10", "kurtosis_b1"]
CLASSICAL = {"noise": dict(name="noise"), "pulsed_p0.1": dict(name="pulsed", p=0.1)}
KS = (1, 2, 4)
SHIFT_SYM = 128             # generator segment period [symbols] (models.SEG_LEN / sps)

ART = os.path.join(scene.ART, "..", "team_fading")


def fading_amp(n, k_lin, device):
    """|h| per frame with E|h|^2 = 1: Rician, linear K-factor k_lin (0 = Rayleigh). None = no fading."""
    if k_lin is None:
        return None
    los = math.sqrt(k_lin / (k_lin + 1.0))
    s = math.sqrt(0.5 / (k_lin + 1.0))                    # per real axis
    return torch.complex(los + s * torch.randn(n, device=device),
                         s * torch.randn(n, device=device)).abs()


def _crop(x_long, n_out, start):
    """Per-frame window x_long[f, start_f : start_f + n_out]."""
    idx = start[:, None] + torch.arange(n_out, device=x_long.device)
    return torch.gather(x_long, 1, idx)


def shift_start(timing, F, link):
    """
    Per-frame crop start [samples] into a stream SHIFT_SYM symbols longer than the
    frame. 'shifted': a uniform whole-symbol shift over the burst period (no timing
    coordination). A float sigma (D4a, README §4.2 Q12): the inter-jammer link's
    timing error, N(0, sigma^2) symbols at sample resolution, taken mod the period --
    the generator's bursts repeat every SHIFT_SYM symbols, so the offset acts mod it.
    """
    period = SHIFT_SYM * link.sps
    if timing == "shifted":
        return torch.randint(0, SHIFT_SYM, (F,), device=link.device) * link.sps
    eps = torch.randn(F, device=link.device) * float(timing) * link.sps
    return torch.round(eps).long() % period


def team_rx(spec, K, k_lin, timing="aligned"):
    """
    A receive model for attacks.jammer_at_rx (spec['rx']): K jammers of one type at
    a total nominal JSR jsr_db_k[0], each projected to JSR - 10 log10 K, then faded.
    K = 1, no fading, aligned reproduces attacks.jammer_at_rx draw for draw
    (verify.py §16a), so the lossless K = 1 series is the D-series jammer itself.
    timing: 'aligned', 'shifted', or a float sigma [symbols] (shift_start). Under
    sigma the first jammer is the LEADER, which holds the victim's frame timing as
    the single aligned jammer does; the followers get it over the inter-jammer link
    with error sigma. sigma = 0 is 'aligned' draw for draw (verify.py §17a).
    """
    base = spec["name"]

    def rx(link, sym, jsr_db_k):
        F, N = sym.shape
        jsr = torch.full((F,), float(jsr_db_k[0]) - 10.0 * math.log10(K), device=link.device)
        total = None
        for k in range(K):
            if base == "noise":
                j = channel.receive(link, attacks.noise_tx(link, F, N), jsr)
            elif base == "pulsed":
                delay, phase = channel.async_draw(link, F)
                j = channel.receive(link, attacks.pulsed_tx(link, F, N, spec["p"]), jsr, delay, phase)
            elif base == "gan":
                delay, phase = channel.async_draw(link, F)
                leader = timing != "shifted" and k == 0
                if timing == "aligned" or leader or (timing != "shifted" and float(timing) == 0):
                    x = attacks.gan_tx(link, F, N, spec["G"], spec["scale"])
                else:
                    # a longer stream on the same grid (gan_tx's alignment does not depend
                    # on n_sym), cropped per frame; a whole-symbol start stays on the grid
                    x_long = attacks.gan_tx(link, F, N + SHIFT_SYM, spec["G"], spec["scale"])
                    start = shift_start(timing, F, link)
                    x = _crop(x_long, x_long.shape[1] - SHIFT_SYM * link.sps, start)
                j = channel.receive(link, x, jsr, delay, phase)
            else:
                raise ValueError(f"team_rx: unsupported jammer {base!r}")
            a = fading_amp(F, k_lin, link.device)
            if a is not None:
                j = j * a[:, None]
            total = j if total is None else total + j
        return total

    return rx


def team_spec(spec, K, chan, timing="aligned"):
    return dict(name="team", rx=team_rx(spec, K, CHANNELS[chan], timing),
                tag=f"{attacks.spec_name(spec)}|K{K}|{timing_tag(timing)}")


def plan(jammers, ks, sigmas=None):
    """(jammer, K, timing) in run order. sigmas (D4a): the generators' K > 1 timings
    become the leader/follower errors in `sigmas`, then 'shifted' as the floor."""
    gen_timings = ("aligned", "shifted") if sigmas is None else (*sigmas, "shifted")
    for jam in jammers:
        timings = ("aligned",) if jam in CLASSICAL else gen_timings
        for K in ks:
            for t in (timings if K > 1 else ("aligned",)):
                yield jam, K, t


def timing_tag(t):
    return t if isinstance(t, str) else f"sigma{t:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help=f"0..{len(TASKS) - 1}")
    ap.add_argument("--run", default="run001")
    ap.add_argument("--frames", type=int, default=512, help="frames per sweep point")
    ap.add_argument("--jammers", default=",".join(JAMMERS))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--sigmas", default=None,
                    help="D4a: comma list of inter-jammer timing errors [symbols], e.g. 0,1,4,16")
    ap.add_argument("--art", default=ART, help="output root (D4a: ../artifacts/cgan/team_timing)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sigmas = None if args.sigmas is None else [float(s) for s in args.sigmas.split(",")]

    snr, chan = TASKS[args.task]
    tag = f"{calibrate_snr.level_tag(snr)}_{chan}"
    out_dir = os.path.join(args.art, args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}{'_smoke' if args.smoke else ''}.json")

    device = lk.setup(seed=args.seed + 100 * args.task)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    jsr_grid = [-30.0, -10.0, 0.0] if args.smoke else baselines.JSR_GRID_K1
    frames = 64 if args.smoke else args.frames
    ks = [int(k) for k in args.ks.split(",")]
    jammers = [j for j in args.jammers.split(",") if j]
    gen_path = dict(snr_ablation.generators())

    print(f"task {args.task} | SNR {snr:g} dB | channel {chan} (K-factor {CHANNELS[chan]}) | "
          f"device {device}", flush=True)
    dfd = calibrate_snr.defender_at(L, snr)
    clean = baselines.measure(L, dfd, None, None, 4 * frames)
    print(f"clean: BER {clean['ber']:.3e}, FAR "
          f"{json.dumps({d: clean['pdet'][d]['0.05'] for d in clean['pdet']})}", flush=True)

    meta = dict(run=args.run, task=args.task, snr_db=snr, channel=chan, k_factor=CHANNELS[chan],
                jsr_db=jsr_grid, jsr_axis="nominal total (mean received) JSR", split="equal",
                n_frames=frames, n_sym=scene.N_SYM, seed=args.seed + 100 * args.task,
                shift_sym=SHIFT_SYM, sigmas_sym=sigmas, tier=1,
                dets=[d for d in dfd.dets if d != "lrt_noise"],
                cnn="frozen 30 dB weights, re-calibrated scale + threshold")
    res = dict(meta=meta, clean=clean, series={})
    t0 = time.time()
    G_cache = {}
    for jam, K, timing in plan(jammers, ks, sigmas):
        if jam in CLASSICAL:
            spec = CLASSICAL[jam]
        else:
            if jam not in G_cache:
                G, scale, _ = models.load_generator(gen_path[jam], device)
                G_cache[jam] = dict(name="gan", G=G, scale=scale, tag=jam)
            spec = G_cache[jam]
        tspec = team_spec(spec, K, chan, timing)
        pts = [baselines.measure(L, dfd, tspec, [jsr], frames) for jsr in jsr_grid]
        res["series"][f"{jam}|K{K}|{timing_tag(timing)}"] = pts
        with open(out_path, "w") as f:
            json.dump(res, f)                                 # checkpoint per series
        print(f"  {jam:<14} K={K} {timing_tag(timing):<10} max BER {max(p['ber'] for p in pts):.3e}  "
              f"CNN P(det) at max JSR {pts[-1]['pdet']['spec_cnn']['0.05']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    res["runtime_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"wrote {os.path.relpath(out_path)} in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
