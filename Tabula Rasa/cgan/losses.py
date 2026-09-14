"""
CGAN track -- the loss terms of Zhou et al. 2025, in plain PyTorch (Decision 0).

The paper's loss is a composite and its equations do not fully line up with its
prose (README §4.2 Q7): the adversarial term is written as BCE (eqs. 1-6) yet the
discriminator carries a "gradient penalty" and an InstanceNorm critic, which are
Wasserstein-GP ingredients. We reconcile that as the NON-SATURATING GAN loss with a
gradient penalty, and keep a pure `wgan-gp` mode as an ablation (`--adv`), so both
readings are runnable.

Generator loss = adversarial + feature matching + time-frequency (STFT) + I/Q
distribution distance. Discriminator loss = adversarial + gradient penalty +
auxiliary classification. All weights default to 1 and every change is logged
(README §4.2 Q7); BER never appears -- the jammer is effective only by imitating
the target modulation.

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
    """Salimans et al. 2016: match the batch-mean discriminator features."""
    return F.mse_loss(f_fake.mean(0), f_real.mean(0))


# ------------------------------------------------------------------ time-frequency (STFT)
def _stft_logpower(x, n_fft=256, hop=64):
    """Mean over the batch of log(1 + |STFT|^2) of the complex signal x[:, 0] + i x[:, 1]."""
    z = torch.complex(x[:, 0], x[:, 1])
    win = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(z, n_fft=n_fft, hop_length=hop, window=win,
                      return_complex=True, center=True)
    return torch.log1p(spec.abs().pow(2)).mean(0)


def stft_loss(real, fake, n_fft=256, hop=64):
    """
    L1 between the batch-mean log-power STFTs of real and generated batches.
    Unpaired: z is random, so it compares distributions, not paired samples.
    Zero on identical batches.
    """
    return F.l1_loss(_stft_logpower(fake, n_fft, hop), _stft_logpower(real, n_fft, hop))


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
    """D's total loss and its components. Call with `fake` detached."""
    real_score, real_class, _ = D(real)
    fake_score, _, _ = D(fake)
    terms = {
        "adv": d_adv_loss(real_score, fake_score, mode),
        "gp": gradient_penalty(D, real, fake),
        "cls": classification_loss(real_class, labels),
    }
    w = {"adv": 1.0, "gp": GP_LAMBDA, "cls": 1.0, **weights}
    total = sum(w.get(k, 1.0) * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}
