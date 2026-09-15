"""
CGAN track -- the loss terms of Zhou et al. 2025, in plain PyTorch (Decision 0).

The paper's loss is a composite and its equations do not fully line up with its
prose (README §4.2 Q7): the adversarial term is written as BCE (eqs. 1-6) yet the
discriminator carries a "gradient penalty" and an InstanceNorm critic, which are
Wasserstein-GP ingredients. Both readings are runnable (`--adv`); the default is
`wgan-gp` (below). Eqs. 1-6 are textbook background from Zhou's ref. [4].

Generator loss = adversarial + feature matching + time-frequency (STFT) + I/Q
distribution distance. Discriminator loss = adversarial + gradient penalty +
auxiliary classification. BER never appears -- the jammer is effective only by
imitating the target modulation.

Where Zhou is silent, the terms follow the papers Zhou cites (README §4.2 Q7):
  * adversarial + GP  -> WGAN-GP, the loss of [7] Saarinen & Koivunen 2020
    (conditional WGAN-GP) and the reason for Zhou's InstanceNorm critic;
    `nsgan` (non-saturating BCE + GP, Zhou's eqs. 4-5) is kept as the ablation.
  * feature matching and time-frequency loss -> [5] Fre-GAN: L1 distances, the
    spectrogram compressed as log(clamp(|S|, 1e-5)). Fre-GAN's weights
    (feat 2, spectrogram 45) are train_cgan.py's defaults.

Each term is a small, self-contained function so verify.py can check its zero
points, and step 2 can add or reweight a stealth term without touching the rest.
"""

import torch
import torch.nn.functional as F

GP_LAMBDA = 10.0


# ------------------------------------------------------------------ adversarial
def d_adv_loss(d_real_score, d_fake_score, mode="nsgan"):
    """Discriminator adversarial loss. Real should score high, fake low."""
    if mode == "nsgan":
        return (F.binary_cross_entropy_with_logits(d_real_score, torch.ones_like(d_real_score))
                + F.binary_cross_entropy_with_logits(d_fake_score, torch.zeros_like(d_fake_score)))
    if mode == "wgan-gp":
        return d_fake_score.mean() - d_real_score.mean()
    raise ValueError(mode)


def g_adv_loss(d_fake_score, mode="nsgan"):
    """Generator adversarial loss (non-saturating: maximise D's fake-is-real belief)."""
    if mode == "nsgan":
        return F.binary_cross_entropy_with_logits(d_fake_score, torch.ones_like(d_fake_score))
    if mode == "wgan-gp":
        return -d_fake_score.mean()
    raise ValueError(mode)


def gradient_penalty(D, real, fake):
    """Two-sided WGAN-GP penalty on the score head at interpolates real/fake."""
    b = real.size(0)
    eps = torch.rand(b, *([1] * (real.dim() - 1)), device=real.device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = D(inter)[0]
    grad = torch.autograd.grad(score.sum(), inter, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


# ------------------------------------------------------------------ auxiliary classifier
def classification_loss(class_logits, labels):
    """Cross-entropy of D's modulation head. Exactly 0 by construction when n_classes == 1."""
    if class_logits.size(1) == 1:
        return class_logits.sum() * 0.0
    return F.cross_entropy(class_logits, labels)


# ------------------------------------------------------------------ feature matching
def feature_matching_loss(f_real, f_fake):
    """
    L1 between batch-mean discriminator features. L1 as in [5] Fre-GAN; batch
    means (Salimans et al. 2016) because G(z) has no paired real sample. Only the
    final features: they are the one feature output Zhou's Fig. 3 exposes.
    """
    return F.l1_loss(f_fake.mean(0), f_real.mean(0))


# ------------------------------------------------------------------ time-frequency (STFT)
STFT_CLAMP = 1e-5       # HiFi-GAN / Fre-GAN dynamic-range compression floor


def _stft_logmag(x, n_fft=256, hop=64):
    """log(clamp(batch-mean |STFT|, 1e-5)) of the complex signal x[:, 0] + i x[:, 1]."""
    z = torch.complex(x[:, 0], x[:, 1])
    win = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(z, n_fft=n_fft, hop_length=hop, window=win,
                      return_complex=True, center=True)
    return torch.log(torch.clamp(spec.abs().mean(0), min=STFT_CLAMP))


def stft_loss(real, fake, n_fft=256, hop=64):
    """
    L1 between the log-magnitude STFTs of real and generated batches ([5]).
    Unpaired: z is random, so it compares batch-mean spectra, not paired samples.
    The log (not log1p of power) makes an out-of-band floor cost as much as an
    in-band error of the same ratio. Zero on identical batches.
    """
    return F.l1_loss(_stft_logmag(fake, n_fft, hop), _stft_logmag(real, n_fft, hop))


# ------------------------------------------------------------------ I/Q distribution distance
def iq_distribution_loss(real, fake):
    """
    Sorted-sample 1-D Wasserstein-1 distance on the pooled I, Q and |x| marginals.
    W1 between two equal-size empirical 1-D distributions is the mean |gap| of their
    order statistics. Zero on identical batches; differentiable in `fake`.
    """
    ri, rq = real[:, 0].reshape(-1), real[:, 1].reshape(-1)
    fi, fq = fake[:, 0].reshape(-1), fake[:, 1].reshape(-1)
    rmag = torch.sqrt(ri ** 2 + rq ** 2)
    fmag = torch.sqrt(fi ** 2 + fq ** 2)
    total = 0.0
    for r, f in ((ri, fi), (rq, fq), (rmag, fmag)):
        total = total + (torch.sort(f)[0] - torch.sort(r)[0].detach()).abs().mean()
    return total / 3.0


# ------------------------------------------------------------------ aggregates
def generator_loss(D, real, fake, labels, weights, mode="nsgan"):
    """G's total loss and its components. `weights` keys: adv, feat, stft, iq, cls."""
    fake_score, fake_class, f_fake = D(fake)
    _, _, f_real = D(real)
    terms = {
        "adv": g_adv_loss(fake_score, mode),
        "feat": feature_matching_loss(f_real.detach(), f_fake),
        "stft": stft_loss(real, fake),
        "iq": iq_distribution_loss(real, fake),
        "cls": classification_loss(fake_class, labels),
    }
    total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}


def discriminator_loss(D, real, fake, labels, weights, mode="nsgan"):
    """
    D's total loss and its components. Call with `fake` detached. The returned
    dict also carries `score_gap` = E D(real) - E D(fake) (logged, not optimised):
    the critic's Wasserstein estimate under wgan-gp, the logit margin under nsgan.
    It shows a discriminator that has won outright, whatever the loss scale.
    """
    real_score, real_class, _ = D(real)
    fake_score, _, _ = D(fake)
    terms = {
        "adv": d_adv_loss(real_score, fake_score, mode),
        "gp": gradient_penalty(D, real, fake),
        "cls": classification_loss(real_class, labels),
    }
    w = {"adv": 1.0, "gp": GP_LAMBDA, "cls": 1.0, **weights}
    total = sum(w.get(k, 1.0) * v for k, v in terms.items())
    stats = {k: float(v.detach()) for k, v in terms.items()}
    stats["score_gap"] = float((real_score.mean() - fake_score.mean()).detach())
    return total, stats
