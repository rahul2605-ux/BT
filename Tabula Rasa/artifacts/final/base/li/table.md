# env base: the CNN scored Li et al.'s way (per sample = one 128-symbol frame)

Clean FAR: argmax 0.0078, α 0.05 0.0530. Reference: Li et al., EfficientNet-B0: DR 100 % two-class, 99.79 % five-class, weighted FAR 0.03 % (five-class).
Li metrics on a 762 clean : 204 jammed test set (Li's balance) from the measured rates. Mean ± std over seeds where a jammer has three.

## Decision: argmax (Li's decision)

### A. Li's window: every jammer at the same JSRs, uniform over [-20, +10] dB (matched configuration)

| jammer | recall | mean excess PER | DR (eq. 2a) % | precision | F-score |
|---|---|---|---|---|---|
| genie flip (upper bound) | 0.007 | 0.839 | 78.4 | 0.204 | 0.014 |
| Li: single tone (constant vector) | 0.898 | 0.450 | 97.2 | 0.969 | 0.932 |
| Li: successive pulse | 0.944 | 0.405 | 98.2 | 0.970 | 0.957 |
| Li: barrage | 0.939 | 0.339 | 98.1 | 0.970 | 0.954 |
| Li: protocol-aware (Amuru p 0.25) | 0.674 | 0.642 | 92.5 | 0.959 | 0.792 |
| cGAN at Li's damage, white-box | 0.412 ± 0.005 | 0.515 ± 0.049 | 87.0 ± 0.1 | 0.934 ± 0.001 | 0.571 ± 0.005 |
| cGAN vs CNN, white-box | 0.783 ± 0.013 | 0.818 ± 0.007 | 94.8 ± 0.3 | 0.964 ± 0.001 | 0.864 ± 0.008 |
| cGAN at Li's damage, grey-box | 0.812 ± 0.181 | 0.527 ± 0.054 | 95.4 ± 3.8 | 0.964 ± 0.009 | 0.874 ± 0.117 |
| cGAN vs CNN, grey-box | 0.785 ± 0.026 | 0.815 ± 0.009 | 94.9 ± 0.5 | 0.964 ± 0.001 | 0.865 ± 0.016 |
| **Li's four pooled** (Li's two-class setting, 762 : 816) | 0.864 | — | 92.6 | 0.992 | 0.923 |

### B. Matched damage: each jammer where it breaks 10 / 50 / 90 % of frames (PER 0.9 = Li's damage)

| jammer | JSR @ PER 0.1 | recall | JSR @ PER 0.5 | recall | JSR @ PER 0.9 | recall | at PER 0.9: DR % / precision / F |
|---|---|---|---|---|---|---|---|
| genie flip (upper bound) | -15.0 | 0.010 | -15.0 | 0.010 | -15.0 | 0.010 | 78.5 / 0.251 / 0.019 |
| Li: single tone (constant vector) | -4.3 | 1.000 | -3.5 | 1.000 | -2.1 | 1.000 | 99.4 / 0.972 / 0.986 |
| Li: successive pulse | -4.2 | 1.000 | -2.2 | 1.000 | 0.6 | 1.000 | 99.4 / 0.972 / 0.986 |
| Li: barrage | -1.6 | 1.000 | 0.1 | 1.000 | 1.7 | 1.000 | 99.4 / 0.972 / 0.986 |
| Li: protocol-aware (Amuru p 0.25) | -11.0 | 0.373 | -9.3 | 0.521 | -7.7 | 0.668 | 92.4 / 0.958 / 0.787 |
| cGAN at Li's damage, white-box | -9.1 ± 1.8 | 0.181 ± 0.270 | -6.6 ± 1.4 | 0.010 ± 0.017 | 1.0 ± 2.3 | 0.169 ± 0.186 | 81.8 ± 3.9 / 0.584 ± 0.508 / 0.255 ± 0.264 |
| cGAN vs CNN, white-box | -21.7 ± 0.6 | 0.023 ± 0.006 | -18.2 ± 0.5 | 0.068 ± 0.022 | -4.3 ± 0.7 | 1.000 ± 0.000 | 99.4 ± 0.0 / 0.972 ± 0.000 / 0.986 ± 0.000 |
| cGAN at Li's damage, grey-box | -9.5 ± 2.2 | 0.925 ± 0.118 | -7.0 ± 1.7 | 0.798 ± 0.310 | -0.1 ± 0.8 | 0.839 ± 0.274 | 96.0 ± 5.8 / 0.963 ± 0.014 / 0.881 ± 0.180 |
| cGAN vs CNN, grey-box | -21.6 ± 0.7 | 0.026 ± 0.013 | -17.8 ± 1.1 | 0.090 ± 0.042 | -4.6 ± 0.6 | 1.000 ± 0.000 | 99.4 ± 0.0 / 0.972 ± 0.000 / 0.986 ± 0.000 |

### C. Most damage while flagged on at most 0.5 of samples (§2.7 standing read-out; grid points)

| jammer | max excess PER | at JSR [dB] |
|---|---|---|
| genie flip (upper bound) | 1.000 | 15.0 |
| Li: single tone (constant vector) | 0.000 | -19.0 |
| Li: successive pulse | 0.000 | -19.0 |
| Li: barrage | 0.000 | -19.0 |
| Li: protocol-aware (Amuru p 0.25) | 0.318 | -10.0 |
| cGAN at Li's damage, white-box | 0.911 ± 0.030 | 1.3 ± 0.6 |
| cGAN vs CNN, white-box | 0.721 ± 0.007 | -13.7 ± 0.6 |
| cGAN at Li's damage, grey-box | 0.295 ± 0.511 | -13.3 ± 9.0 |
| cGAN vs CNN, grey-box | 0.676 ± 0.049 | -14.0 ± 1.0 |

## Decision: α 0.05 (the paper's operating point)

### A. Li's window: every jammer at the same JSRs, uniform over [-20, +10] dB (matched configuration)

| jammer | recall | mean excess PER | DR (eq. 2a) % | precision | F-score |
|---|---|---|---|---|---|
| genie flip (upper bound) | 0.048 | 0.839 | 75.7 | 0.195 | 0.077 |
| Li: single tone (constant vector) | 0.961 | 0.450 | 95.0 | 0.829 | 0.890 |
| Li: successive pulse | 0.982 | 0.405 | 95.4 | 0.832 | 0.901 |
| Li: barrage | 0.981 | 0.339 | 95.4 | 0.832 | 0.901 |
| Li: protocol-aware (Amuru p 0.25) | 0.773 | 0.642 | 91.0 | 0.796 | 0.784 |
| cGAN at Li's damage, white-box | 0.624 ± 0.008 | 0.515 ± 0.049 | 87.9 ± 0.2 | 0.759 ± 0.002 | 0.685 ± 0.006 |
| cGAN vs CNN, white-box | 0.852 ± 0.016 | 0.818 ± 0.007 | 92.7 ± 0.3 | 0.811 ± 0.003 | 0.831 ± 0.009 |
| cGAN at Li's damage, grey-box | 0.959 ± 0.028 | 0.527 ± 0.054 | 95.0 ± 0.6 | 0.829 ± 0.004 | 0.889 ± 0.015 |
| cGAN vs CNN, grey-box | 0.855 ± 0.027 | 0.815 ± 0.009 | 92.8 ± 0.6 | 0.812 ± 0.005 | 0.833 ± 0.015 |
| **Li's four pooled** (Li's two-class setting, 762 : 816) | 0.924 | — | 93.5 | 0.949 | 0.937 |

### B. Matched damage: each jammer where it breaks 10 / 50 / 90 % of frames (PER 0.9 = Li's damage)

| jammer | JSR @ PER 0.1 | recall | JSR @ PER 0.5 | recall | JSR @ PER 0.9 | recall | at PER 0.9: DR % / precision / F |
|---|---|---|---|---|---|---|---|
| genie flip (upper bound) | -15.0 | 0.053 | -15.0 | 0.053 | -15.0 | 0.053 | 75.8 / 0.210 / 0.084 |
| Li: single tone (constant vector) | -4.3 | 1.000 | -3.5 | 1.000 | -2.1 | 1.000 | 95.8 / 0.835 / 0.910 |
| Li: successive pulse | -4.2 | 1.000 | -2.2 | 1.000 | 0.6 | 1.000 | 95.8 / 0.835 / 0.910 |
| Li: barrage | -1.6 | 1.000 | 0.1 | 1.000 | 1.7 | 1.000 | 95.8 / 0.835 / 0.910 |
| Li: protocol-aware (Amuru p 0.25) | -11.0 | 0.622 | -9.3 | 0.758 | -7.7 | 0.863 | 92.9 / 0.814 / 0.838 |
| cGAN at Li's damage, white-box | -9.1 ± 1.8 | 0.527 ± 0.372 | -6.6 ± 1.4 | 0.121 ± 0.160 | 1.0 ± 2.3 | 0.549 ± 0.487 | 86.3 ± 10.3 / 0.540 ± 0.457 / 0.542 ± 0.470 |
| cGAN vs CNN, white-box | -21.7 ± 0.6 | 0.107 ± 0.018 | -18.2 ± 0.5 | 0.233 ± 0.058 | -4.3 ± 0.7 | 1.000 ± 0.000 | 95.8 ± 0.0 / 0.835 ± 0.000 / 0.910 ± 0.000 |
| cGAN at Li's damage, grey-box | -9.5 ± 2.2 | 0.997 ± 0.006 | -7.0 ± 1.7 | 0.993 ± 0.012 | -0.1 ± 0.8 | 0.999 ± 0.002 | 95.8 ± 0.0 / 0.835 ± 0.000 / 0.910 ± 0.001 |
| cGAN vs CNN, grey-box | -21.6 ± 0.7 | 0.117 ± 0.028 | -17.8 ± 1.1 | 0.293 ± 0.116 | -4.6 ± 0.6 | 1.000 ± 0.000 | 95.8 ± 0.0 / 0.835 ± 0.000 / 0.910 ± 0.000 |

### C. Most damage while flagged on at most 0.5 of samples (§2.7 standing read-out; grid points)

| jammer | max excess PER | at JSR [dB] |
|---|---|---|
| genie flip (upper bound) | 1.000 | 15.0 |
| Li: single tone (constant vector) | 0.000 | -22.0 |
| Li: successive pulse | 0.000 | -22.0 |
| Li: barrage | 0.000 | -22.0 |
| Li: protocol-aware (Amuru p 0.25) | 0.002 | -13.0 |
| cGAN at Li's damage, white-box | 0.904 ± 0.019 | 1.0 ± 0.0 |
| cGAN vs CNN, white-box | 0.636 ± 0.057 | -15.7 ± 0.6 |
| cGAN at Li's damage, grey-box | 0.000 ± 0.000 | -20.7 ± 1.5 |
| cGAN vs CNN, grey-box | 0.582 ± 0.087 | -16.0 ± 1.0 |
