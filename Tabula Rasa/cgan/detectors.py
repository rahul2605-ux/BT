"""
Baselines -- the defender: four SEPARATE detectors at the victim receiver R.

Every detector observes one received frame r (the victim's burst of N symbols,
pulse tails included, before any equalisation) and nothing else. Each turns it
into a scalar statistic built so that LARGER = MORE SUSPICIOUS, and each gets
its own threshold at false-alarm rate alpha from clean frames (`calibrate`, the
m0/detectors.py convention). They are never OR-combined into a suite.

    power       mean |r|^2. One-sided (the classical detector, blind to anything
                that lowers power) and two-sided |. - clean mean|, reported as
                two variants of the same detector.
    kurtosis    E|r|^4 / (E|r|^2)^2, two-sided |. - clean mean|. Scale-free, so
                it sees the SHAPE of the amplitude distribution (pulsing,
                Gaussianisation) that power cannot.
    spec_cnn    Li et al., IEEE Access 2022: EfficientNet-B0 on a spectrogram
                image, retrained on this link (train_spectrogram_cnn.py). Their
                waterfall images have a FIXED colour scale (Fig. 7: clean cyan,
                barrage orange); the frozen sim05/06 code stretched each image to
                its own 2-98 % range, which erases any jammer that raises the
                whole spectrum evenly. Here the dB range is fixed from clean data.
    lrt_noise   The Neyman-Pearson test for a white Gaussian jammer of known
                received power -- the optimal-detector ceiling, for noise rows
                only (below).

THE NOISE-JAMMER LRT (exact up to RRC truncation)
-------------------------------------------------
The victim's pulses p_m are orthonormal (unit-energy root-Nyquist, all fully
inside the frame), so the matched-filter samples z_m = <r, p_m> and the energy
outside their span, E_perp = ||r||^2 - sum_m |z_m|^2, are independent given the
symbols. With all white Gaussian terms of per-sample variance s2 (s2 = N0 clean,
N0 + JSR * P_s jammed):
    z_m ~ (1/4) sum_{s in QPSK} CN(s, s2),     E_perp ~ s2 * Gamma(D),  D = len(r) - N,
so the frame log-likelihood ratio of s2_1 against s2_0 is
    sum_m [log mix(z_m; s2_1) - log mix(z_m; s2_0)] - D log(s2_1/s2_0) - E_perp (1/s2_1 - 1/s2_0).
"""

import math

import torch

SQRT2_INV = 1.0 / math.sqrt(2.0)
QPSK = torch.tensor([complex(a * SQRT2_INV, b * SQRT2_INV) for a in (1, -1) for b in (1, -1)],
                    dtype=torch.complex128)
ALPHAS = [0.01, 0.05]
HEADLINE_ALPHA = 0.05

# spectrogram image (README §2.10 table: Li et al. vs ours)
N_FFT, HOP = 64, 16
IMG_H, IMG_W = 248, 422             # Li et al.'s 422 x 248 (W x H) after downscaling
HEADROOM_DB = 20.0                  # above the clean maximum before the colour scale saturates
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- thresholds
def calibrate(stat_clean, alpha):
    """Empirical (1-alpha) quantile of clean-frame statistics (m0/detectors.py)."""
    return float(torch.quantile(stat_clean.double().flatten(), 1.0 - alpha))


def p_detect(stat, threshold):
    """Fraction of frames flagged; on clean frames this is the realised FAR."""
    return float((stat.double() > threshold).double().mean())


# ---------------------------------------------------------------- power, kurtosis
def power(r):
    return r.abs().pow(2).double().mean(dim=-1)


def kurtosis(r):
    m2 = r.abs().pow(2).double().mean(dim=-1)
    m4 = r.abs().pow(4).double().mean(dim=-1)
    return m4 / m2.pow(2)


def two_sided(stat, clean_mean):
    return (stat - clean_mean).abs()


# ---------------------------------------------------------------- noise LRT
def lrt_parts(r, z):
    """Sufficient statistics of the noise LRT: (z [F, N] float64, E_perp [F], D)."""
    z = z.to(torch.complex128)
    e_perp = r.abs().pow(2).double().sum(-1) - z.abs().pow(2).sum(-1)
    return z, e_perp, r.shape[-1] - z.shape[-1]


def _log_mix(z, s2):
    q = QPSK.to(z.device).reshape(-1, 1, 1)
    d2 = (z.unsqueeze(0) - q).abs().pow(2)
    return torch.logsumexp(-d2 / s2, dim=0) - math.log(4.0 * math.pi * s2)


def lrt_noise(parts, n0, jsr_lin, p_s):
    """Frame LLR for a white Gaussian jammer at received JSR jsr_lin (scalar)."""
    z, e_perp, D = parts
    s0, s1 = float(n0), float(n0 + jsr_lin * p_s)
    llr_z = (_log_mix(z, s1) - _log_mix(z, s0)).sum(-1)
    return llr_z - D * math.log(s1 / s0) - e_perp * (1.0 / s1 - 1.0 / s0)


# ---------------------------------------------------------------- spectrogram CNN
def spectrogram_db(r):
    """Complex two-sided STFT power [dB], zero frequency centred: [F, time, freq]."""
    win = torch.hann_window(N_FFT, device=r.device)
    S = torch.stft(r if r.dtype == torch.complex128 else r.to(torch.complex64),
                   n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=win, center=False,
                   onesided=False, return_complex=True)
    P = torch.fft.fftshift(S.abs().pow(2), dim=1)
    return (10.0 * torch.log10(P.clamp_min(1e-20))).transpose(1, 2)


class SpecScale:
    """The fixed dB colour range, fitted once on clean frames."""

    def __init__(self, vmin, vmax):
        self.vmin, self.vmax = float(vmin), float(vmax)

    @classmethod
    def fit(cls, r_clean):
        db = spectrogram_db(r_clean).flatten()
        idx = torch.randperm(db.numel(), device=db.device)[:4_000_000]
        return cls(torch.quantile(db[idx].double(), 0.01),
                   torch.quantile(db[idx].double(), 0.999) + HEADROOM_DB)

    def to_dict(self):
        return dict(vmin=self.vmin, vmax=self.vmax)


_LUT = {}


def _viridis(device):
    if device not in _LUT:
        from matplotlib import colormaps
        cmap = colormaps["viridis"]
        _LUT[device] = torch.tensor([cmap(i / 255.0)[:3] for i in range(256)],
                                    dtype=torch.float32, device=device)
    return _LUT[device]


def spectrogram_image(r, scale, normalise=True):
    """[F, 3, IMG_H, IMG_W]: fixed-scale dB -> viridis -> Li et al.'s image size."""
    db = spectrogram_db(r)
    x = ((db - scale.vmin) / (scale.vmax - scale.vmin)).clamp(0, 1)
    rgb = _viridis(r.device)[(x * 255).long()].permute(0, 3, 1, 2)       # [F, 3, time, freq]
    img = torch.nn.functional.interpolate(rgb, size=(IMG_H, IMG_W), mode="bilinear",
                                          align_corners=False)
    if normalise:
        m = torch.tensor(IMAGENET_MEAN, device=r.device).reshape(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD, device=r.device).reshape(1, 3, 1, 1)
        img = (img - m) / s
    return img


def build_cnn(pretrained=True):
    """EfficientNet-B0 (torchvision), two-class head; ImageNet init (Li et al. do not say)."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 2)
    return net


@torch.no_grad()
def cnn_statistic(net, r, scale, batch=256):
    """logit(jammed) - logit(clean) per frame; larger = more suspicious."""
    net.eval()
    out = []
    for i in range(0, r.shape[0], batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=r.is_cuda):
            logits = net(spectrogram_image(r[i:i + batch], scale)).float()
        out.append((logits[:, 1] - logits[:, 0]).double())
    return torch.cat(out)


def load_cnn(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build_cnn(pretrained=False).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, SpecScale(**ckpt["scale"]), ckpt


# ---------------------------------------------------------------- all statistics of a batch
def statistics(r, z, net=None, scale=None):
    """Raw per-frame statistics every detector needs: power, kurtosis, cnn, LRT parts."""
    s = dict(power=power(r), kurtosis=kurtosis(r), lrt=lrt_parts(r, z))
    if net is not None:
        s["cnn"] = cnn_statistic(net, r, scale)
    return s
