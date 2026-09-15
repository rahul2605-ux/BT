"""
CGAN track -- generator and discriminator, Zhou et al. 2025 Figs. 2-3, in plain
PyTorch (Decision 0, README §2.10). No Sionna here: these are tensor-in/tensor-out
modules, so the shape and loss checks in verify.py need no compute node.

Every dimension the paper states is reproduced; every one it omits is a named
constant here and tabulated in README §4.2 Q7. The conditioning path (a label
embedding into G, an auxiliary classifier out of D) is kept even though QPSK is
one class -- step 2 (stealth conditioning) reuses it, so n_classes is a parameter,
not a literal 1.

GENERATOR (Fig. 2)
    z ~ N(0, I) in R^400  (+)  Embedding(label) in R^16   -> concat -> (B, 1, 416)
    3x [Conv1d -> BatchNorm1d -> LeakyReLU -> Dropout -> MaxPool1d]
    AdaptiveAvgPool1d(POOL_TO)  -> flatten -> Linear -> Tanh
    -> (B, 2, SEG_LEN): the I and Q sample streams, in (-1, 1)

DISCRIMINATOR (Fig. 3)
    (B, 2, SEG_LEN) -> (B, 2, 1, SEG_LEN)      2 channels = I, Q; a 1-row "image"
    3x [Conv2d(1xk) -> (InstanceNorm2d) -> LeakyReLU -> MaxPool2d(1x2)]   (no IN in block 1)
    AdaptiveAvgPool2d((1, FEAT_LEN)) -> features f  (B, D_CH, 1, FEAT_LEN)
    three heads sharing f:
        f_flat  = features flattened               -> used by feature-matching loss
        score   = Linear -> LeakyReLU -> Linear     -> (B, 1)  real/fake logit
        class   = Linear                            -> (B, n_classes)  modulation logit
"""

import torch
import torch.nn as nn

# Zhou, stated.
Z_DIM = 400
LABEL_EMB = 16
SEG_LEN = 1024
D_CH = 256          # channels out of D's last conv block
FEAT_LEN = 128      # D feature length after adaptive pool (Fig. 3; text's "28" is a typo)

# Unstated -> our choices (README §4.2 Q7).
G_CH = (64, 128, 256)
D_CH_STAGES = (64, 128, D_CH)
KERNEL = 5
POOL = 2
DROPOUT = 0.3
LRELU = 0.2
G_POOL_TO = 16      # AdaptiveAvgPool1d target length before G's dense layer
HEADROOM = 0.95     # largest normalised real sample ([5] Fre-GAN / HiFi-GAN loader)


class Generator(nn.Module):
    def __init__(self, n_classes=1, z_dim=Z_DIM, seg_len=SEG_LEN):
        super().__init__()
        self.z_dim, self.seg_len = z_dim, seg_len
        self.embed = nn.Embedding(n_classes, LABEL_EMB)

        blocks, c_in = [], 1
        for c_out in G_CH:
            blocks += [nn.Conv1d(c_in, c_out, KERNEL, padding=KERNEL // 2),
                       nn.BatchNorm1d(c_out), nn.LeakyReLU(LRELU),
                       nn.Dropout(DROPOUT), nn.MaxPool1d(POOL)]
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(G_POOL_TO)
        self.head = nn.Sequential(nn.Linear(G_CH[-1] * G_POOL_TO, 2 * seg_len), nn.Tanh())

    def forward(self, z, labels):
        """z: (B, z_dim); labels: (B,) long. -> (B, 2, seg_len) in (-1, 1)."""
        x = torch.cat([z, self.embed(labels)], dim=1).unsqueeze(1)   # (B, 1, z_dim+16)
        x = self.pool(self.conv(x)).flatten(1)
        return self.head(x).reshape(-1, 2, self.seg_len)


class PeakScaler:
    """
    Zhou's normalisation: real training segments are divided by a global peak so
    they fall in (-1, 1) to match G's Tanh output; G's samples are multiplied back
    by that same scale ("inverse normalisation") before they hit the link. The
    scale is one saved number, fit once on a sample of clean waveforms. The round
    trip is exact by construction (verify.py checks it).

    HEADROOM 0.95 follows Zhou's normalisation reference [5] (Fre-GAN, via the
    HiFi-GAN loader: audio = normalize(audio) * 0.95): the largest real sample maps
    to 0.95, not to 1.0, which Tanh only reaches asymptotically.
    """

    def __init__(self, scale=1.0):
        self.scale = float(scale)

    @classmethod
    def fit(cls, waveforms, headroom=HEADROOM):
        """waveforms: (..., 2, seg_len). The largest |sample| normalises to `headroom`."""
        return cls(waveforms.abs().max().item() / headroom)

    def normalize(self, x):
        return x / self.scale

    def denormalize(self, x):
        return x * self.scale


class Discriminator(nn.Module):
    def __init__(self, n_classes=1, seg_len=SEG_LEN):
        super().__init__()
        blocks, c_in = [], 2
        for i, c_out in enumerate(D_CH_STAGES):
            layer = [nn.Conv2d(c_in, c_out, (1, KERNEL), padding=(0, KERNEL // 2))]
            if i > 0:                                    # no InstanceNorm in the first block
                layer.append(nn.InstanceNorm2d(c_out))
            layer += [nn.LeakyReLU(LRELU), nn.MaxPool2d((1, POOL))]
            blocks += layer
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, FEAT_LEN))
        feat = D_CH * FEAT_LEN
        self.score = nn.Sequential(nn.Linear(feat, 256), nn.LeakyReLU(LRELU), nn.Linear(256, 1))
        self.classify = nn.Linear(feat, n_classes)

    def features(self, x):
        """x: (B, 2, seg_len) -> flattened feature vector (B, D_CH*FEAT_LEN)."""
        return self.pool(self.conv(x.unsqueeze(2))).flatten(1)

    def forward(self, x):
        """-> (score logit (B, 1), class logits (B, n_classes), features (B, D_CH*FEAT_LEN))."""
        f = self.features(x)
        return self.score(f), self.classify(f), f
