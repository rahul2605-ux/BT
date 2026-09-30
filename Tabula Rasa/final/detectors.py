# copied from cgan/detectors.py at 68c97a4 (2026-09-29), pruned
"""
Final experiment -- the detectors at the victim receiver R.

Every detector observes one received frame and nothing else (energy_csi is also handed
the frame's true channel gain). Each turns it into a scalar statistic built so that
LARGER = MORE SUSPICIOUS, and each gets its own threshold at false-alarm rate alpha from
clean frames of the SAME environment (`calibrate`, the CFAR rule of README §2.7).

    energy      mean_k |z_k|^2 after the matched filter: the received symbol energy,
                clean mean c0^2 + N0 (README §3.4 decision 3). One-sided, the detector
                the energy-targeted GAN trains against. The matched filter keeps only
                the noise in its noise-equivalent bandwidth (the symbol rate, 1/8 of the
                simulated band), so noise-level uncertainty moves it far less than the
                full-band power.
    energy_2s   |energy - clean mean|: the same statistic, two-sided. Evaluated, not
                trained against. The genie push LOWERS the received energy; only a
                two-sided test sees it (§3.3d).
    energy_csi  energy - (g^2 - 1) c0^2: an energy detector handed the true per-frame
                fading gain g (as cgan's power_csi, §3.3k). A real receiver estimates g,
                so it lies between `energy` (calibrated on faded frames) and this.
    kurtosis    E|r|^4 / (E|r|^2)^2 on the received frame, two-sided |. - clean mean|.
                Scale-free, so it sees the SHAPE of the amplitude distribution.
    spec_cnn    Li et al., IEEE Access 2022: EfficientNet-B0 on a fixed-scale spectrogram
                image of the full band, retrained per environment (train_cnn.py).
    power       mean |r|^2 over the full band. KEPT ONLY FOR THE REGRESSION GATE against
                S4 (regress.py); no environment reports it.

Pruned from cgan/detectors.py: the noise LRT (dropped by the final experiment) and
power_csi. Added: energy, energy_csi.
"""

import torch

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


N_Q = 257


def stat_quantiles(stat, n=N_Q):
    """The statistic's empirical CDF on a fixed 257-point grid (P(det) at any budget)."""
    q = torch.linspace(0.0, 1.0, n, dtype=torch.float64, device=stat.device)
    return torch.quantile(stat.double().flatten(), q).cpu().tolist()


def soft_pdet(stat, threshold, scale):
    """
    Differentiable stand-in for P(det): mean sigmoid((stat - threshold)/scale), with
    `scale` the clean-frame std of the statistic. Used only in training; p_detect is
    what is reported.
    """
    return torch.sigmoid((stat.double() - threshold) / scale).mean()


# ---------------------------------------------------------------- energy, power, kurtosis
def energy(z):
    """mean_k |z_k|^2: the received symbol energy after the matched filter."""
    return z.abs().pow(2).double().mean(dim=-1)


def energy_csi(z, gain, c0):
    """energy minus what the fading added to the known signal energy: (g^2 - 1) c0^2.
    Equal to energy(z) for gain None (no fading)."""
    e = energy(z)
    if gain is None:
        return e
    return e - (gain.reshape(-1).double() ** 2 - 1.0) * c0 ** 2


def power(r):
    """Full-band mean |r|^2 -- the regression gate only."""
    return r.abs().pow(2).double().mean(dim=-1)


def kurtosis(r):
    m2 = r.abs().pow(2).double().mean(dim=-1)
    m4 = r.abs().pow(4).double().mean(dim=-1)
    return m4 / m2.pow(2)


def two_sided(stat, clean_mean):
    return (stat - clean_mean).abs()


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


def _viridis_apply(x, grad=False):
    """
    x in [0, 1] -> viridis RGB [..., 3]. The deployed detector quantises x onto the
    256-entry LUT (zero gradient). grad=True keeps that EXACT forward value and
    substitutes the colormap's local slope in the backward pass (straight-through), so
    the attacker trains against the actual weights, white-box.
    """
    lut = _viridis(x.device)
    xs = x * 255.0
    hard = lut[xs.long().clamp(0, 255)]
    if not grad:
        return hard
    i0 = xs.detach().floor().clamp(0, 254)
    w = (xs - i0).unsqueeze(-1).to(lut.dtype)
    soft = torch.lerp(lut[i0.long()], lut[i0.long() + 1], w)
    return soft + (hard - soft).detach()


def spectrogram_image(r, scale, normalise=True, grad=False):
    """[F, 3, IMG_H, IMG_W]: fixed-scale dB -> viridis -> Li et al.'s image size."""
    db = spectrogram_db(r)
    x = ((db - scale.vmin) / (scale.vmax - scale.vmin)).clamp(0, 1)
    rgb = _viridis_apply(x, grad).permute(0, 3, 1, 2)                    # [F, 3, time, freq]
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


def cnn_statistic(net, r, scale, batch=256, grad=False):
    """
    logit(jammed) - logit(clean) per frame; larger = more suspicious.
    grad=True (training only): one batch, fp32, no autocast, straight-through LUT. The
    reported statistic is always the grad=False path (autocast fp16 on CUDA).
    """
    net.eval()
    if grad:
        logits = net(spectrogram_image(r, scale, grad=True))
        return (logits[:, 1] - logits[:, 0]).double()
    out = []
    with torch.no_grad():
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
    for p in net.parameters():
        p.requires_grad_(False)      # frozen detector: the gradient goes to the jammer
    return net, SpecScale(**ckpt["scale"]), ckpt
