# copied from cgan/baselines.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment -- the defender of one environment, and one measured sweep point.

`Defender` and `measure` are cgan/baselines.py's, pruned to what the final experiment
reports: the detectors `energy`, `energy_2s`, `kurtosis`, `spec_cnn`, plus `energy_csi`
in the fading environments. `power_one_sided` (full band) is readable only so that
regress.py can load S4's cgan-format thresholds.json. The noise LRT, the K > 1 sweeps
and the confirmation pass are gone: no pick at alpha is reported (README §3.4).

`calibrate_env` is step 5 of cgan/train_spectrogram_cnn.py with the environment's
frames: every threshold is the (1 - alpha) quantile of the statistic over n_cal clean
frames drawn from the SAME environment (noise-level factor and fading included), so the
false-alarm rate stays alpha and the threshold rises instead -- the CFAR rule of §2.7.
"""

import json
import math
import os

import numpy as np
import torch

import attacks
import detectors

DETS = ["energy", "energy_2s", "kurtosis", "spec_cnn"]
CSI = "energy_csi"                                    # added in the fading environments
GATE_DETS = ["power_one_sided", "kurtosis", "spec_cnn"]
N_CALIBRATION = 20_000
_KEY = {"spec_cnn": "cnn"}                             # cgan-format thresholds.json names


def env_dets(faded):
    return DETS + ([CSI] if faded else [])


class Defender:
    """The calibrated detectors of one environment (weights, colour scale, thresholds)."""

    def __init__(self, L, snr_db, thr, net, scale, dets):
        self.L, self.snr_db, self.thr = L, snr_db, thr
        self.net, self.scale, self.dets = net, scale, list(dets)
        self.n0, self.p_s, self.c0 = L.noise_var(snr_db), L.p_s, L.c0

    @classmethod
    def load(cls, L, snr_db, thr_dir, cnn_path=None, dets=None):
        """thr_dir holds thresholds.json (final/ or cgan format); the CNN defaults to thr_dir's."""
        with open(os.path.join(thr_dir, "thresholds.json")) as f:
            thr = json.load(f)
        cnn_path = os.path.join(thr_dir, "detector_spec.pt") if cnn_path is None else cnn_path
        net, scale, _ = detectors.load_cnn(cnn_path, L.device)
        if "spec_scale" in thr:
            scale = detectors.SpecScale(**thr["spec_scale"])
        return cls(L, snr_db, thr, net, scale, dets if dets is not None else thr["dets"])

    def statistics(self, r, z, dets=None, grad=False, gain=None):
        """
        Per-frame statistics {det: [F]}, each larger-is-more-suspicious. `grad` only
        reaches the CNN (the others are differentiable as written). `gain` is
        attacks.frames' out.get("gain"), needed only by energy_csi.
        """
        want = set(self.dets if dets is None else dets)
        s = {}
        if want & {"energy", "energy_2s"}:
            e = detectors.energy(z)
            s["energy"] = e
            s["energy_2s"] = detectors.two_sided(e, self.thr["energy_clean_mean"])
        if CSI in want:
            if gain is None and self.L.rician_k_db is not None:
                raise ValueError("energy_csi on a faded link needs the per-frame gain")
            s[CSI] = detectors.energy_csi(z, gain, self.c0)
        if "power_one_sided" in want:
            s["power_one_sided"] = detectors.power(r)
        if "kurtosis" in want:
            s["kurtosis"] = detectors.two_sided(detectors.kurtosis(r), self.thr["kurtosis_clean_mean"])
        if "spec_cnn" in want:
            s["spec_cnn"] = detectors.cnn_statistic(self.net, r, self.scale, grad=grad)
        return {k: v for k, v in s.items() if k in want}

    def threshold(self, det, alpha):
        key = det if det in self.thr else _KEY.get(det, det)
        return self.thr[key][str(alpha)]


def measure(L, dfd, spec, jsr_db_k, n_frames, batch=512):
    """
    One sweep point. spec None = clean. Returns a JSON-able dict: BER, SER, PER with
    their counts, P(det) of each detector at each alpha, the mean of each statistic and,
    for a jammer, the mean and std over frames of its DC share.
    """
    counts = np.zeros(4, dtype=np.int64)
    frame_errors = 0
    stats, dcs = {}, []
    for i in range(0, n_frames, batch):
        n = min(batch, n_frames - i)
        out = attacks.frames(L, n, spec, jsr_db_k, dfd.snr_db)
        counts += np.array(attacks.error_counts(out))
        frame_errors += int((out["bits"] != out["bits_hat"]).any(dim=-1).sum())
        for k, v in dfd.statistics(out["r"], out["z"], gain=out.get("gain")).items():
            stats.setdefault(k, []).append(v)
        if "dc_share" in out:
            dcs.append(out["dc_share"])
    stats = {k: torch.cat(v) for k, v in stats.items()}
    pdet = {det: {str(a): detectors.p_detect(s, dfd.threshold(det, a)) for a in detectors.ALPHAS}
            for det, s in stats.items()}
    e, b, se, sb = (int(c) for c in counts)
    res = dict(ber=e / b, ser=se / sb, per=frame_errors / n_frames, errors=e, bits=b, sym_errors=se,
               symbols=sb, frame_errors=frame_errors, frames=n_frames, pdet=pdet,
               mean_stat={k: float(v.mean()) for k, v in stats.items()})
    if dcs:
        dc = torch.cat(dcs)
        res["dc_share"], res["dc_share_std"] = float(dc.mean()), float(dc.std())
    return res


def binom_sigma(p, n):
    return math.sqrt(max(p, 1.0 / n) * (1 - min(p, 1 - 1.0 / n)) / n)


# ---------------------------------------------------------------- CFAR calibration
@torch.no_grad()
def calibrate_env(L, net, scale, snr_db, dets, n_cal=N_CALIBRATION, batch=1024, verbose=True):
    """
    Every detector's CFAR threshold on n_cal clean frames of the environment L carries
    (L.noise_unc_db, L.rician_k_db already set). Also the clean statistics' quantiles and
    the clean damage floor (for excess damage). Returns the thresholds dict.
    """
    raw = dict(energy=[], kurtosis=[], spec_cnn=[], energy_csi=[])
    counts = np.zeros(4, dtype=np.int64)
    frame_errors = 0
    for i in range(0, n_cal, batch):
        out = attacks.frames(L, min(batch, n_cal - i), None, None, snr_db)
        counts += np.array(attacks.error_counts(out))
        frame_errors += int((out["bits"] != out["bits_hat"]).any(dim=-1).sum())
        raw["energy"].append(detectors.energy(out["z"]))
        raw["kurtosis"].append(detectors.kurtosis(out["r"]))
        raw["spec_cnn"].append(detectors.cnn_statistic(net, out["r"], scale))
        raw["energy_csi"].append(detectors.energy_csi(out["z"], out.get("gain"), L.c0))
    raw = {k: torch.cat(v) for k, v in raw.items()}
    em, km = float(raw["energy"].mean()), float(raw["kurtosis"].mean())
    stat = dict(energy=raw["energy"], energy_2s=detectors.two_sided(raw["energy"], em),
                kurtosis=detectors.two_sided(raw["kurtosis"], km), spec_cnn=raw["spec_cnn"],
                energy_csi=raw["energy_csi"])
    e, b, se, sb = (int(c) for c in counts)
    thr = dict(n_clean=n_cal, alphas=detectors.ALPHAS, snr_db=snr_db, dets=list(dets),
               noise_unc_db=L.noise_unc_db, rician_k_db=L.rician_k_db,
               energy_clean_mean=em, kurtosis_clean_mean=km,
               energy_clean_std=float(raw["energy"].std()), noise_var=L.noise_var(snr_db), p_s=L.p_s,
               c0=L.c0, spec_scale=scale.to_dict(),
               clean=dict(ber=e / b, ser=se / sb, per=frame_errors / n_cal, errors=e, bits=b,
                          frame_errors=frame_errors),
               clean_stat_q={d: detectors.stat_quantiles(stat[d]) for d in dets})
    for d in dets:
        thr[d] = {str(a): detectors.calibrate(stat[d], a) for a in detectors.ALPHAS}
    if verbose:
        print(f"  calibrated on {n_cal} clean frames: energy mean {em:.5f} (c0^2 + N0 = "
              f"{L.c0 ** 2 + L.noise_var(snr_db):.5f}), thresholds@0.05 "
              + " ".join(f"{d} {thr[d]['0.05']:.5g}" for d in dets)
              + f", clean BER {e / b:.2e} PER {frame_errors / n_cal:.4f}", flush=True)
    return thr
