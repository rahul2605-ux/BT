"""
Final experiment -- the six environments, and where everything is written (README §3.4
"Final experiment"). New file; no Sionna, so figures.py can import it on the login node.

Every environment sits at our SNR 15 dB. `cgan/`'s SNR is mean power per sample over the
8x simulated band (sps 8), so Es/N0 = SNR + 10 log10(8) = SNR + 9.03 dB: 15 dB is Es/N0
24 dB (Eb/N0 21 dB). The paper states Es/N0.

    env          sigma_N    victim-link fading     role
    base         0          none                   the ideal link at the operating point
    noise        0.5 dB     none                   noise-level uncertainty, calibrated receiver
    noise_1db    1 dB       none                   the same with in-band interference
    fading       0          Rician K 12 dB         L-band CNPC fading (Matolak & Sun)
    fading_k28   0          Rician K 28 dB         C-band CNPC fading
    both         0.5 dB     Rician K 12 dB         the combination: candidate system model

sigma_N is the std [dB] of a per-frame log-normal factor on the noise variance, unit mean,
that the defender does not know (S5, README §3.3n). The fading is a per-frame Rician
AMPLITUDE on the victim's own link, unit mean power (link.Link.fading_gain).
"""

import math
import os

SNR_DB = 15.0
ESN0_OFFSET_DB = 10.0 * math.log10(8)          # sps 8 -> Es/N0 = SNR + 9.03 dB
N_SYM = 128                                    # symbols per frame = one detector observation
SYMBOL_RATE = 1e6                              # [Bd]; channel.py's sinc taps need a rate
ALPHA = 0.05                                   # every detector's false-alarm rate

ENVS = {
    "base":       dict(noise_unc_db=0.0, rician_k_db=None),
    "noise":      dict(noise_unc_db=0.5, rician_k_db=None),
    "noise_1db":  dict(noise_unc_db=1.0, rician_k_db=None),
    "fading":     dict(noise_unc_db=0.0, rician_k_db=12.0),
    "fading_k28": dict(noise_unc_db=0.0, rician_k_db=28.0),
    "both":       dict(noise_unc_db=0.5, rician_k_db=12.0),
}
LABEL = {"base": "base", "noise": "σN 0.5 dB", "noise_1db": "σN 1 dB", "fading": "Rician K 12 dB",
         "fading_k28": "Rician K 28 dB", "both": "σN 0.5 + K 12"}

HERE = os.path.dirname(os.path.abspath(__file__))
ART_ROOT = os.path.join(HERE, "..", "artifacts", "final")
CGAN_ART = os.path.join(HERE, "..", "artifacts", "cgan")
SCRATCH_ROOT = "/itet-stor/rrahman/net_scratch/final"       # checkpoints: home quota (README §C.4)
# FINAL_SMOKE=1 (submit_env.sh --smoke): a tiny end-to-end run of the chain into _smoke/,
# so a real environment's artifacts are never touched by a pipeline test.
SMOKE = os.environ.get("FINAL_SMOKE") == "1"
if SMOKE:
    ART_ROOT = os.path.join(ART_ROOT, "_smoke")
    SCRATCH_ROOT = os.path.join(SCRATCH_ROOT, "_smoke")


def faded(env):
    return ENVS[env]["rician_k_db"] is not None


def apply(L, env):
    """Put environment `env` on a link.Link: the per-frame noise factor and the fading."""
    cfg = ENVS[env]
    L.noise_unc_db = cfg["noise_unc_db"]
    L.rician_k_db = cfg["rician_k_db"]
    return L


def art(env, *parts):
    return os.path.join(ART_ROOT, env, *parts)


def save_big(obj, env, rel):
    """torch.save to net_scratch and symlink it from artifacts/final/<env>/<rel>."""
    import torch
    target = os.path.join(SCRATCH_ROOT, env, rel)
    link = art(env, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    tmp = target + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, target)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)
    return link
