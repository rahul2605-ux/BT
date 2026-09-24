"""
E2 example frames -- what the spectrogram CNN actually sees at each noise level
(README §3.3g). One small job; writes an .npz that snr_figures.py draws, so the
figure can be restyled on the login node without a GPU (the baselines.examples /
baselines_figures split, again).

For each level: a clean frame, then D1 (plain GAN) and D2 (CNN-targeted, beta 10)
each at the JSR where it reaches matched excess BER 3e-4 AT THAT LEVEL -- the
operating point of the §3.3g headline table. Every image is rendered exactly as the
CNN receives it: that level's re-fitted colour scale, viridis, Li et al.'s size.
Beside each: the CNN's P(det) over N_PDET fresh frames at the same point, so the
picture and the number can be read together.

Two things this makes visible that the curves only state. At 15 dB the D2 jammer
sits at the same JSR as at 30 dB, but 15 dB more noise surrounds it, so the image
barely changes and the CNN mostly misses it -- the mechanism of §3.3g finding 2.
And the colour scale itself moves with SNR (vmin -14.7 dB at 0 dB, -80.7 at the
noiseless anchor), which is what "a detector trained at 30 dB, deployed off-design"
looks like.

    sbatch submit_snr_examples.sh
"""

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # before any Sionna import (scene.py)

import json
import os
import time

import numpy as np
import torch

import attacks
import calibrate_snr
import detectors
import link as lk
import models
import snr_ablation
import snr_figures as F                     # numpy/matplotlib only: the matched-JSR lookup

LEVELS = [0.0, 15.0, 30.0, None]            # None = the noiseless anchor
ROWS = ["clean", "plain_run001", "spec_cnn_b10"]
BER_REF = 3e-4
N_PDET = 512
OUT = os.path.join(snr_ablation.ART, "run001", "examples.npz")


def main():
    device = lk.setup(seed=4242)
    L = lk.Link(**{k: lk.LINK[k] for k in ("sps", "pulse")})
    paths = dict(snr_ablation.generators())
    old_ad = F.AD
    F.AD = os.path.join(snr_ablation.ART, "run001")
    sweeps = {(d["snr"] if d["snr"] is not None else None): d for d in F.load()}
    F.AD = old_ad
    t0 = time.time()

    imgs, raw_db, jsr, pdet, scales, thr = [], [], [], [], [], []
    for snr in LEVELS:
        dfd = calibrate_snr.defender_at(L, snr)
        d = sweeps[snr]
        t_cnn = dfd.threshold("spec_cnn", detectors.HEADLINE_ALPHA)
        row_img, row_db, row_j, row_p = [], [], [], []
        for tag in ROWS:
            if tag == "clean":
                spec, arg, j = None, None, float("nan")
            else:
                _, j = F.at_matched_ber(d, tag, "spec_cnn", BER_REF)
                G, scale, _ = models.load_generator(paths[tag], device)
                spec, arg = dict(name="gan", G=G, scale=scale, tag=tag), [j]
            with torch.no_grad():
                out = attacks.frames(L, N_PDET, spec, arg, dfd.snr_db)
                stat = detectors.cnn_statistic(dfd.net, out["r"], dfd.scale)
                img = detectors.spectrogram_image(out["r"][:1], dfd.scale, normalise=False)
                db = detectors.spectrogram_db(out["r"][:1])
            row_img.append((img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            row_db.append(db[0].float().cpu().numpy())
            row_j.append(j)
            row_p.append(detectors.p_detect(stat, t_cnn))
            print(f"  {calibrate_snr.level_tag(snr):>10} {tag:<14} JSR {j:+6.1f} dB  "
                  f"P(det)_CNN {row_p[-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        imgs.append(row_img); raw_db.append(row_db); jsr.append(row_j); pdet.append(row_p)
        scales.append([dfd.scale.vmin, dfd.scale.vmax]); thr.append(t_cnn)

    np.savez_compressed(
        OUT, images=np.array(imgs), raw_db=np.array(raw_db), jsr=np.array(jsr, float),
        pdet=np.array(pdet, float), scales=np.array(scales, float), thr=np.array(thr, float),
        levels=np.array([np.nan if s is None else s for s in LEVELS], float),
        rows=np.array(ROWS), ber_ref=BER_REF, n_pdet=N_PDET)
    print(f"wrote {os.path.relpath(OUT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
