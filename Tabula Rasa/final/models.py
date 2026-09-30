# copied from cgan/models.py at 68c97a4 (2026-09-29), pruned
"""
CGAN track -- the generator of Zhou et al. 2025 Fig. 2, in plain PyTorch (Decision 0,
README §2.10). No Sionna here. Pruned for final/: the discriminator and PeakScaler
(Zhou's imitation stage) are gone -- every final/ generator is trained from random
init, and run001_G.pt (Zhou's reproduced CGAN) is only loaded.

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
"""

import torch
import torch.nn as nn

# Zhou, stated.
Z_DIM = 400
LABEL_EMB = 16
SEG_LEN = 1024

# Unstated -> our choices (README §4.2 Q7).
G_CH = (64, 128, 256)
KERNEL = 5
POOL = 2
DROPOUT = 0.3
LRELU = 0.2
G_POOL_TO = 16      # AdaptiveAvgPool1d target length before G's dense layer


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


def load_generator(path, device):
    """
    Load a trained Generator and its peak scale from a checkpoint written by
    train_cgan.py / train_gan.py (run001 predates the `recipe` field, so read
    everything with defaults). Returns (G in eval mode, scale, checkpoint dict).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G = Generator(n_classes=ckpt.get("n_classes", 1),
                  seg_len=ckpt.get("seg_len", SEG_LEN)).to(device)
    G.load_state_dict(ckpt["state_dict"])
    G.eval()
    return G, float(ckpt.get("scale", 1.0)), ckpt
