"""
Per-SNR detector calibration for the E2 noise ablation (README §3.3g).

Every threshold in this track is an empirical (1-alpha) quantile of CLEAN frames
(`detectors.calibrate`), so it is a function of the noise level: move the SNR and
every detector must be re-calibrated or its false-alarm rate stops being alpha,
which is the whole reason the BER-P(det) comparison is matched. The spectrogram
CNN is doubly noise-dependent -- its fixed dB colour scale was fitted on clean
30 dB frames AND its weights were trained there.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It re-runs steps 1 and 5 of train_spectrogram_cnn.py with `snr_db` as an argument:
re-fit the colour scale on clean frames at that SNR, then re-measure every clean
statistic and take its quantile. It does NOT retrain the CNN (user decision,
2026-09-23): the weights stay the deployed 30 dB ones, so the ablation reports
"a detector trained at 30 dB, deployed off-design" -- the same framing §3.3c #6
used for the frozen sim08 CNN outside its training range. That is a stated
limitation (README §4.4), not an oversight.

train_spectrogram_cnn.py is deliberately NOT refactored to call this: it produced
the deployed detector and its 30 dB artifacts must not move. verify.py §15a
instead asserts that this path reproduces the stored thresholds.json at 30 dB.

    from calibrate_snr import calibrate_at, level_tag
    thr, parts_path = calibrate_at(L, net, snr_db)      # cached under ART/snr/<tag>/
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import json
import os
import time

import torch

import attacks
import detectors
import link as lk
import scene

N_CALIBRATION = 20_000      # same as train_spectrogram_cnn.py, so the quantiles are as tight
N_SCALE_FIT = 2048          # frames the colour scale is fitted on (its step 1)
BATCH = 1024

# The noiseless anchor. verify.py's idiom: an SNR so high the AWGN term is
# numerically zero. Two things degenerate there and both are guarded below.
NO_NOISE_DB = 300.0

# Guard 1 -- the colour scale. With no noise the STFT of a pure QPSK waveform has
# nulls that clamp at -200 dB, so a 1% quantile vmin would flatten every real
# feature into the top of the scale. Cap the dynamic range instead. At 30 dB the
# fitted range is 83.5 dB, so this is INACTIVE there and the 30 dB check still
# reproduces the stored scale exactly.
MAX_RANGE_DB = 120.0

# Guard 2 -- the noise LRT. detectors.lrt_noise forms log(s1/s0) with s0 = n0, so
# n0 -> 0 is an infinity. Below this the LRT row is dropped rather than faked;
# `thresholds["lrt_usable"]` records it and Defender omits the detector.
MIN_N0 = 1e-12


def level_tag(snr_db):
    """Filesystem tag for one noise level. Mirrors sim08_ablation.level_tag."""
    return "noiseless" if snr_db is None or snr_db >= NO_NOISE_DB else f"snr_{snr_db:g}"


def cal_dir(snr_db):
    return os.path.join(scene.ART, "snr", level_tag(snr_db))


def is_baseline(snr_db):
    """True for the 30 dB level, whose calibration already exists as the deployed one."""
    return snr_db is not None and abs(snr_db - lk.SNR_DB) < 1e-9


def _clean_frames(L, n, snr_db, batch=BATCH):
    """n clean received frames at this SNR, batched."""
    return torch.cat([attacks.frames(L, min(batch, n - i), None, None, snr_db)["r"]
                      for i in range(0, n, batch)])


def fit_scale(L, snr_db, n=N_SCALE_FIT):
    """The fixed dB colour scale at this noise level, with the dynamic-range guard."""
    s = detectors.SpecScale.fit(_clean_frames(L, n, snr_db))
    if s.vmax - s.vmin > MAX_RANGE_DB:
        s = detectors.SpecScale(s.vmax - MAX_RANGE_DB, s.vmax)
    return s


def defender_at(L, snr_db, n_cal=N_CALIBRATION, verbose=True):
    """
    A baselines.Defender calibrated at `snr_db`, calibrating first if the cache is
    cold. At 30 dB it returns the deployed defender untouched, so the ablation's
    30 dB row IS the §3.3f configuration and serves as the regression check.
    """
    import baselines
    if is_baseline(snr_db):
        return baselines.Defender(L)
    net, _, _ = detectors.load_cnn(os.path.join(scene.ART, "detector_spec.pt"), L.device)
    calibrate_at(L, net, snr_db, n_cal=n_cal, verbose=verbose)
    return baselines.Defender(L, snr_db=(NO_NOISE_DB if snr_db is None else snr_db),
                              thr_dir=cal_dir(snr_db))


def calibrate_at(L, net, snr_db, n_cal=N_CALIBRATION, force=False, verbose=True):
    """
    Re-calibrate every detector at `snr_db`. Returns (thresholds dict, lrt parts path).

    Cached under artifacts/cgan/baselines/snr/<tag>/ in the same schema as the 30 dB
    thresholds.json + clean_lrt_parts.pt, so Defender can load either one.
    """
    out = cal_dir(snr_db)
    thr_path = os.path.join(out, "thresholds.json")
    parts_path = os.path.join(out, "clean_lrt_parts.pt")
    if not force and os.path.exists(thr_path) and os.path.exists(parts_path):
        with open(thr_path) as f:
            cached = json.load(f)
        # A cache built from FEWER clean frames than asked for is not reusable: its
        # quantiles are looser. verify.py §15 deliberately re-calibrates at 4096 and
        # 1024 frames, and without this the sweep would silently inherit those.
        if cached.get("n_clean", 0) >= n_cal:
            return cached, parts_path
        if verbose:
            print(f"  [{level_tag(snr_db)}] cached calibration used "
                  f"{cached.get('n_clean', 0)} clean frames < {n_cal} requested; redoing",
                  flush=True)

    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    snr = NO_NOISE_DB if snr_db is None else snr_db
    n0 = L.noise_var(snr)

    scale = fit_scale(L, snr)
    if verbose:
        print(f"  [{level_tag(snr_db)}] scale vmin {scale.vmin:.2f} vmax {scale.vmax:.2f} dB, "
              f"n0 {n0:.3e}  ({time.time() - t0:.0f}s)", flush=True)

    stat = dict(power=[], kurtosis=[], cnn=[], z=[], e_perp=[])
    for i in range(0, n_cal, BATCH):
        o = attacks.frames(L, min(BATCH, n_cal - i), None, None, snr)
        s = detectors.statistics(o["r"], o["z"], net, scale)
        stat["power"].append(s["power"])
        stat["kurtosis"].append(s["kurtosis"])
        stat["cnn"].append(s["cnn"])
        stat["z"].append(s["lrt"][0].to(torch.complex64).cpu())
        stat["e_perp"].append(s["lrt"][1].cpu())
    D = s["lrt"][2]
    stat = {k: torch.cat(v) for k, v in stat.items()}
    pm, km = float(stat["power"].mean()), float(stat["kurtosis"].mean())
    # the clean side of the ROC: with the jammed quantiles a sweep point stores,
    # this gives P(det) at any budget, not only at alpha (detectors.stat_quantiles).
    clean_q = dict(power_one_sided=detectors.stat_quantiles(stat["power"]),
                   power_two_sided=detectors.stat_quantiles(detectors.two_sided(stat["power"], pm)),
                   kurtosis=detectors.stat_quantiles(detectors.two_sided(stat["kurtosis"], km)),
                   spec_cnn=detectors.stat_quantiles(stat["cnn"]))

    thresholds = dict(
        snr_db=(None if snr_db is None else float(snr_db)), tag=level_tag(snr_db),
        n_clean=n_cal, alphas=detectors.ALPHAS, power_clean_mean=pm, kurtosis_clean_mean=km,
        power_one_sided={str(a): detectors.calibrate(stat["power"], a) for a in detectors.ALPHAS},
        power_two_sided={str(a): detectors.calibrate(detectors.two_sided(stat["power"], pm), a)
                         for a in detectors.ALPHAS},
        kurtosis={str(a): detectors.calibrate(detectors.two_sided(stat["kurtosis"], km), a)
                  for a in detectors.ALPHAS},
        cnn={str(a): detectors.calibrate(stat["cnn"], a) for a in detectors.ALPHAS},
        lrt_d=D, noise_var=n0, p_s=L.p_s, spec_scale=scale.to_dict(),
        lrt_usable=bool(n0 > MIN_N0), clean_ber=float(L.ber_clean(snr)),
        clean_stat_q=clean_q, calibration_s=time.time() - t0,
    )
    torch.save(dict(z=stat["z"], e_perp=stat["e_perp"], D=D), parts_path)
    with open(thr_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    if verbose:
        print(f"  [{level_tag(snr_db)}] clean BER {thresholds['clean_ber']:.3e}, "
              f"power mean {pm:.5f}, kurtosis mean {km:.4f}, "
              f"lrt {'on' if thresholds['lrt_usable'] else 'OFF (n0 underflow)'}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return thresholds, parts_path
