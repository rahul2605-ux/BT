"""
Baselines -- the sweep: every classical attack against every detector, K = 1..4
jammers (README §2.10, §3.4).

    sbatch submit_baselines.sh --k 1                  # first: universal K = 1 curve + test drops
    sbatch --array=2-4 submit_baselines.sh            # then K = 2, 3, 4 over the 50 test drops

K = 1 needs no geometry. With T power-controlled and every jammer asynchronous, a
single jammer's BER and each detector's P(det) depend on its received JSR alone
(scene.py; the two assumptions behind that are checked in verify.py §10). So K = 1
is ONE sweep over received JSR, -50..+15 dB in 1 dB steps, for all seven attacks,
and a drop's K = 1 result in dBm is that curve read at the drop's JSR.

K >= 2 is simulated over the 50 fixed test drops for the pulsed family only (equal
split of the budget P_J = -30..+50 dBm, 2 dB steps). Everything else follows from
the K = 1 curve and is assembled in baselines_figures.py, not simulated again:
  * K white-noise jammers are one white-noise jammer at the total JSR (verify §10);
  * the omniscient team is one genie at JSR P_J * sum_k g_kR / P_T,rx (scene.py);
  * the best-single-jammer-of-K reference is the K = 1 curve at the strongest JSR.

Stored per point: BER, SER (with error and bit counts), the requested per-jammer
JSR, and P(det) of each detector at each alpha, plus the mean of each statistic.
Picks at matched detectability (max BER with P(det) <= alpha) are re-measured on
fresh frames, because a max over noisy P(det) estimates favours lucky draws (§3.3c).

Outputs: artifacts/cgan/baselines/run001/sweep_K{K}.json, examples.pt (K = 1).
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import: no OptiX on the cluster (scene.py)

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import attacks
import detectors
import link as lk
import scene

RUN = "run001"
OUT = os.path.join(scene.ART, RUN)
JSR_GRID_K1 = [float(v) for v in np.arange(-50.0, 15.5, 1.0)]
BUDGET_GRID = [float(v) for v in np.arange(-30.0, 50.5, 2.0)]
DETS = ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn", "lrt_noise"]


# ---------------------------------------------------------------- the defender, loaded once
class Defender:
    """
    The calibrated detector suite at one noise level.

    `snr_db` / `thr_dir` exist for the E2 noise ablation (README §3.3g). Every
    threshold is a quantile of CLEAN frames, so it is a function of the noise level;
    calibrate_snr.calibrate_at writes one thresholds.json + clean_lrt_parts.pt per
    level and `thr_dir` points at it. The CNN's WEIGHTS are always the deployed 30 dB
    ones -- only its colour scale and threshold are re-fitted -- so an off-30 dB
    defender is a detector trained at 30 dB and deployed off-design, which is what
    the ablation reports. Both default to the 30 dB deployed set, so every caller
    that predates the ablation is unchanged.
    """

    def __init__(self, L, snr_db=lk.SNR_DB, thr_dir=None):
        thr_dir = scene.ART if thr_dir is None else thr_dir
        with open(os.path.join(thr_dir, "thresholds.json")) as f:
            self.thr = json.load(f)
        self.net, scale, _ = detectors.load_cnn(os.path.join(scene.ART, "detector_spec.pt"), L.device)
        # thresholds.json and the checkpoint carry the same scale at 30 dB (both written
        # from one SpecScale by train_spectrogram_cnn.py); off-design only the former moves.
        self.scale = detectors.SpecScale(**self.thr["spec_scale"]) if "spec_scale" in self.thr else scale
        parts = torch.load(os.path.join(thr_dir, "clean_lrt_parts.pt"), weights_only=False)
        self.clean_lrt = (parts["z"].to(L.device).to(torch.complex128),
                          parts["e_perp"].to(L.device).double(), parts["D"])
        self.snr_db = snr_db
        self.n0, self.p_s = L.noise_var(snr_db), L.p_s
        # the noise LRT is log(s1/s0) with s0 = n0: it has no finite form at the
        # noiseless anchor, so drop the row there rather than fake it (calibrate_snr).
        self.dets = [d for d in DETS if d != "lrt_noise" or self.thr.get("lrt_usable", True)]
        self._lrt_thr = {}

    def lrt_threshold(self, jsr_lin, alpha):
        key = (round(10 * math.log10(jsr_lin), 6), alpha)
        if key not in self._lrt_thr:
            s = detectors.lrt_noise(self.clean_lrt, self.n0, jsr_lin, self.p_s)
            self._lrt_thr[key] = detectors.calibrate(s, alpha)
        return self._lrt_thr[key]

    def statistics(self, r, z, noise_jsr_lin=None, dets=None, grad=False):
        """
        Per-frame statistics, each already in larger-is-more-suspicious form. `dets`
        restricts them to a subset (train_shaped.py scores one detector at a time;
        the CNN is the only expensive one). `grad` only reaches the CNN: power,
        kurtosis and the LRT are differentiable as written, the CNN needs the
        straight-through image path (detectors.cnn_statistic) and one un-batched
        forward pass, so it is off unless D2 is training against it.
        """
        want = set(self.dets if dets is None else dets)
        s, raw = {}, {}
        if want & {"power_one_sided", "power_two_sided"}:
            p = raw["power"] = detectors.power(r)
            s["power_one_sided"] = p
            s["power_two_sided"] = detectors.two_sided(p, self.thr["power_clean_mean"])
        if "kurtosis" in want:
            raw["kurtosis"] = detectors.kurtosis(r)
            s["kurtosis"] = detectors.two_sided(raw["kurtosis"], self.thr["kurtosis_clean_mean"])
        if "spec_cnn" in want:
            s["spec_cnn"] = raw["cnn"] = detectors.cnn_statistic(self.net, r, self.scale, grad=grad)
        if noise_jsr_lin is not None and "lrt_noise" in want:
            s["lrt_noise"] = detectors.lrt_noise(detectors.lrt_parts(r, z), self.n0, noise_jsr_lin, self.p_s)
        return s, raw

    def threshold(self, det, alpha, noise_jsr_lin=None):
        key = {"power_one_sided": "power_one_sided", "power_two_sided": "power_two_sided",
               "kurtosis": "kurtosis", "spec_cnn": "cnn"}
        if det == "lrt_noise":
            return self.lrt_threshold(noise_jsr_lin, alpha)
        return self.thr[key[det]][str(alpha)]


def measure(L, dfd, spec, jsr_db_k, n_frames, batch=512, keep_stats=False):
    """
    One sweep point. spec None = clean. jsr_db_k: list of K per-jammer JSRs [dB]
    (a float for the omniscient aggregate). Returns a JSON-able dict.
    """
    noise_jsr = None
    if spec is not None and spec["name"] in ("noise", "shaped"):    # shaped: mismatched, still a valid test
        noise_jsr = float(sum(10 ** (v / 10) for v in jsr_db_k))
    counts = np.zeros(4, dtype=np.int64)
    stats, raws = {}, {}
    for i in range(0, n_frames, batch):
        n = min(batch, n_frames - i)
        out = attacks.frames(L, n, spec, jsr_db_k, dfd.snr_db)
        counts += np.array(attacks.error_counts(out))
        s, raw = dfd.statistics(out["r"], out["z"], noise_jsr)
        for k, v in s.items():
            stats.setdefault(k, []).append(v)
        for k, v in raw.items():
            raws.setdefault(k, []).append(v)
    stats = {k: torch.cat(v) for k, v in stats.items()}
    pdet = {}
    for det, s in stats.items():
        pdet[det] = {str(a): detectors.p_detect(s, dfd.threshold(det, a, noise_jsr)) for a in detectors.ALPHAS}
    e, b, se, sb = (int(c) for c in counts)
    res = dict(ber=e / b, ser=se / sb, errors=e, bits=b, sym_errors=se, symbols=sb, frames=n_frames,
               pdet=pdet, mean_stat={k: float(torch.cat(v).mean()) for k, v in raws.items()})
    if keep_stats:
        res["stat_q"] = {det: detectors.stat_quantiles(s) for det, s in stats.items()}
    return res


def binom_sigma(alpha, n):
    return math.sqrt(alpha * (1 - alpha) / n)


def confirm(L, dfd, spec, points, jsr_of, n_confirm, dets, alphas):
    """
    For each detector and alpha, the max-BER point with P(det) <= alpha + 2 sigma,
    re-measured on fresh frames. `points`: list of measured dicts; `jsr_of(i)` the
    JSR argument that produced point i. `spec` is one attack spec, or a function
    i -> spec when every point is its own jammer (train_shaped.py).
    """
    spec_of = spec if callable(spec) else (lambda i: spec)
    picks = {}
    for det in dets:
        if det not in points[0]["pdet"]:
            continue
        for a in alphas:
            tol = 2 * binom_sigma(a, points[0]["frames"])
            ok = [i for i, p in enumerate(points) if p["pdet"][det][str(a)] <= a + tol]
            if not ok:
                picks.setdefault(det, {})[str(a)] = None
                continue
            i = max(ok, key=lambda j: points[j]["ber"])
            m = measure(L, dfd, spec_of(i), jsr_of(i), n_confirm)
            passed = m["pdet"][det][str(a)] <= a + 2 * binom_sigma(a, n_confirm)
            picks.setdefault(det, {})[str(a)] = dict(index=i, sweep_ber=points[i]["ber"],
                                                     sweep_pdet=points[i]["pdet"][det][str(a)],
                                                     confirmed_ber=m["ber"], confirmed_ser=m["ser"],
                                                     confirmed_pdet=m["pdet"][det][str(a)], passed=passed)
    return picks


# ---------------------------------------------------------------- examples for the figures
@torch.no_grad()
def examples(L, dfd, jsr_list=(-10.0, 0.0)):
    """Frames, matched-filter clouds, spectrograms and statistics per attack at two JSRs."""
    ex = {}
    specs = [None] + attacks.all_specs()
    for jsr in jsr_list:
        for spec in specs:
            name = "clean" if spec is None else attacks.spec_name(spec)
            if spec is None and jsr != jsr_list[0]:
                continue
            arg = None if spec is None else (jsr if spec["name"] == "omniscient" else [jsr])
            out = attacks.frames(L, 64, spec, arg, dfd.snr_db)
            s, raw = dfd.statistics(out["r"], out["z"])
            ex[f"{name}@{jsr:+.0f}"] = dict(
                r=out["r"][:2].cpu(), z=out["z"].flatten()[:8192].cpu(),
                spec_db=detectors.spectrogram_db(out["r"][:1]).cpu(),
                image=detectors.spectrogram_image(out["r"][:1], dfd.scale, normalise=False).cpu(),
                power=raw["power"].cpu(), kurtosis=raw["kurtosis"].cpu(), cnn=raw["cnn"].cpu(),
                ber=attacks.error_counts(out)[0] / attacks.error_counts(out)[1])
    return ex


# ---------------------------------------------------------------- runs
def run_k1(L, dfd, n_frames, n_confirm):
    t0 = time.time()
    drops = scene.load_test_drops()                 # builds and caches the test drops (Sionna RT)
    print(f"test drops: {len(drops['g'])}, realised SNR {drops['snr_realised_db'].min():.3f}.."
          f"{drops['snr_realised_db'].max():.3f} dB, T power {scene.w_to_dbm(drops['p_t_w']).min():.1f}.."
          f"{scene.w_to_dbm(drops['p_t_w']).max():.1f} dBm", flush=True)
    clean = measure(L, dfd, None, None, 4 * n_frames)
    print(f"clean: BER {clean['ber']:.2e}, FAR {json.dumps(clean['pdet'])}", flush=True)
    result = dict(run=RUN, K=1, jsr_db=JSR_GRID_K1, n_frames=n_frames, n_confirm=n_confirm,
                  alphas=detectors.ALPHAS, clean=clean, attacks={}, confirmed={})
    for spec in attacks.all_specs():
        name = attacks.spec_name(spec)
        omni = spec["name"] == "omniscient"
        jsr_of = (lambda i: JSR_GRID_K1[i]) if omni else (lambda i: [JSR_GRID_K1[i]])
        pts = []
        for i, jsr in enumerate(JSR_GRID_K1):
            pts.append(measure(L, dfd, spec, jsr_of(i), n_frames))
            p = pts[-1]
            print(f"  {name:<16} JSR {jsr:+5.0f} dB  BER {p['ber']:.2e}  P(det)@0.05 " +
                  " ".join(f"{d} {p['pdet'][d]['0.05']:.3f}" for d in p["pdet"]) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        result["attacks"][name] = pts
        result["confirmed"][name] = confirm(L, dfd, spec, pts, jsr_of, n_confirm, dfd.dets, detectors.ALPHAS)
        print(f"  {name} confirmed picks: {json.dumps(result['confirmed'][name])}", flush=True)
    torch.save(examples(L, dfd), os.path.join(OUT, "examples.pt"))
    result["runtime_s"] = time.time() - t0
    return result


def run_kn(L, dfd, K, n_frames, n_confirm, n_drops=None, which=("pulsed_p1",)):
    t0 = time.time()
    drops = scene.load_test_drops()
    g, p_t_rx = drops["g"][:, 1:], drops["p_t_rx_w"]
    D = g.shape[0] if n_drops is None else min(n_drops, g.shape[0])
    g = g[:D]
    specs = [s for s in attacks.all_specs() if attacks.spec_name(s) in which]
    result = dict(run=RUN, K=K, budget_dbm=BUDGET_GRID, n_drops=D, n_frames=n_frames, n_confirm=n_confirm,
                  alphas=detectors.ALPHAS, split="equal",
                  jsr_db=[[(10 * np.log10(scene.received_jsr(g[d], b, K, p_t_rx))).tolist()
                           for b in BUDGET_GRID] for d in range(D)],
                  attacks={}, confirmed={})
    for spec in specs:
        name = attacks.spec_name(spec)
        per_drop, conf = [], []
        for d in range(D):
            jsr_of = (lambda i, d=d: result["jsr_db"][d][i])
            pts = [measure(L, dfd, spec, jsr_of(i), n_frames) for i in range(len(BUDGET_GRID))]
            per_drop.append(pts)
            conf.append(confirm(L, dfd, spec, pts, jsr_of, n_confirm,
                                ["power_one_sided", "power_two_sided", "kurtosis", "spec_cnn"],
                                [detectors.HEADLINE_ALPHA]))
            print(f"  K={K} {name:<14} drop {d:2d}: max BER {max(p['ber'] for p in pts):.3f}, "
                  f"CNN picks {json.dumps(conf[-1].get('spec_cnn'))}  ({time.time() - t0:.0f}s)", flush=True)
        result["attacks"][name] = per_drop
        result["confirmed"][name] = conf
    result["runtime_s"] = time.time() - t0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)))
    ap.add_argument("--frames", type=int, default=None, help="frames per point (K=1: 2048, K>1: 256)")
    ap.add_argument("--confirm", type=int, default=4096)
    ap.add_argument("--drops", type=int, default=None, help="K>1: cap the number of test drops")
    ap.add_argument("--kn-attacks", default="pulsed_p1",
                    help="K>1: comma list of attacks (default the matched-QPSK jammer only)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = lk.setup(seed=1000 + args.k)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    dfd = Defender(L)
    global JSR_GRID_K1, BUDGET_GRID
    if args.smoke:
        JSR_GRID_K1 = [-30.0, -10.0, 0.0]
        BUDGET_GRID = [10.0, 30.0]
    if args.k == 1:
        res = run_k1(L, dfd, args.frames or (64 if args.smoke else 512), 64 if args.smoke else args.confirm)
    else:
        res = run_kn(L, dfd, args.k, args.frames or (32 if args.smoke else 256),
                     64 if args.smoke else 1024, n_drops=2 if args.smoke else args.drops,
                     which=tuple(args.kn_attacks.split(",")))
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"sweep_K{args.k}{tag}.json"), "w") as f:
        json.dump(res, f)
    print(f"wrote sweep_K{args.k}{tag}.json in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
