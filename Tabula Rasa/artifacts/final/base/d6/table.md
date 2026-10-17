# env base: the CNN before / after one retrain with cnnGAN1's frames (README §3.3s)

Same frames for both defenders (evaluate.py, paired by position). Retrained on Li's four + cnnGAN1 grey-box r0, r1 (204 frames, JSR U[−22, +10] dB); r2 held out.

| training (validation split) | deployed `cnn/` | `cnn_d6/` |
|---|---|---|
| accuracy | 0.922 | 0.920 |
| FAR (argmax) | 0.000 | 0.033 |
| AUC | 0.977 | 0.968 |
| correct on clean | 1.000 | 0.967 |
| correct on barrage | 0.984 | 1.000 |
| correct on tone | 0.867 | 0.897 |
| correct on pulse_comb | 0.893 | 0.963 |
| correct on protocol_aware | 0.649 | 0.719 |
| correct on cnn_b10_grey_r0 | — | 0.821 |
| correct on cnn_b10_grey_r1 | — | 0.821 |

## Decision: argmax (Li's decision)

Clean FAR 0.0078 → 0.0203. **Li's pooled DR (clean vs Li's four, eq. 2a): 92.6 % → 93.8 %** (recall 0.864 → 0.899).

| jammer | Li's window [−20, +10] dB: before | after | Δ | at PER 0.5: before | after | Δ |
|---|---|---|---|---|---|---|
| clean frames (1 − FAR) | 0.992 | 0.980 | -0.012 | 0.992 | 0.980 | -0.012 |
| Li: barrage | 0.939 | 0.985 | +0.045 | 1.000 | 1.000 | +0.000 |
| Li: single tone | 0.898 | 0.914 | +0.016 | 1.000 | 1.000 | +0.000 |
| Li: successive pulse | 0.944 | 0.983 | +0.038 | 1.000 | 1.000 | +0.000 |
| Li: protocol-aware (Amuru p 0.25) | 0.674 | 0.716 | +0.041 | 0.521 | 0.639 | +0.118 |
| Amuru pulsed p 0.05 | 0.617 | 0.663 | +0.046 | 0.080 | 0.154 | +0.074 |
| Amuru pulsed p 0.02 | 0.558 | 0.593 | +0.035 | 0.049 | 0.088 | +0.039 |
| genie flip (upper bound) | 0.007 | 0.018 | +0.010 | 0.010 | 0.012 | +0.002 |
| cnnGAN1 grey-box, seen (r0, r1) | 0.778 | 0.853 | +0.075 | 0.076 | 0.265 | +0.189 |
| cnnGAN1 grey-box, held out (r2) | 0.800 | 0.877 | +0.077 | 0.119 | 0.396 | +0.277 |

Every attacker evaluated (one row per tag; recall, window A and matched PER 0.5):

| tag | window: before | after | PER 0.5: before | after |
|---|---|---|---|---|
| omniscient_e1 | 0.007 | 0.018 | 0.010 | 0.012 |
| omniscient_e0.1 | 0.003 | 0.007 | 0.006 | 0.020 |
| noise | 0.939 | 0.985 | 1.000 | 1.000 |
| pulsed_p1 | 0.719 | 0.753 | 0.960 | 0.985 |
| pulsed_p0.5 | 0.701 | 0.739 | 0.794 | 0.867 |
| pulsed_p0.25 | 0.674 | 0.716 | 0.521 | 0.639 |
| pulsed_p0.1 | 0.640 | 0.687 | 0.183 | 0.291 |
| pulsed_p0.05 | 0.617 | 0.663 | 0.080 | 0.154 |
| pulsed_p0.02 | 0.558 | 0.593 | 0.049 | 0.088 |
| pulsed_p0.01 | 0.432 | 0.457 | 0.057 | 0.121 |
| tone | 0.898 | 0.914 | 1.000 | 1.000 |
| pulse_comb | 0.944 | 0.983 | 1.000 | 1.000 |
| zhou_cgan | 0.859 | 0.884 | 0.999 | 1.000 |
| control_r0 | 0.744 | 0.795 | 0.071 | 0.147 |
| control_r1 | 0.817 | 0.870 | 0.232 | 0.405 |
| control_r2 | 0.675 | 0.738 | 0.027 | 0.070 |
| cnn_b10_r0 | 0.769 | 0.841 | 0.050 | 0.147 |
| cnn_b10_r1 | 0.792 | 0.868 | 0.062 | 0.216 |
| cnn_b10_r2 | 0.789 | 0.867 | 0.092 | 0.288 |
| energy_b10_r0 | 0.683 | 0.736 | 0.031 | 0.078 |
| kurtosis_b10_r0 | 0.898 | 0.941 | 0.960 | 1.000 |
| random_push_e0.1 | 0.579 | 0.634 | 0.010 | 0.035 |
| cnn_li_r0 | 0.409 | 0.629 | 0.029 | 0.352 |
| cnn_li_r1 | 0.418 | 0.630 | 0.000 | 0.041 |
| cnn_li_r2 | 0.408 | 0.649 | 0.000 | 0.013 |
| cnn_b10_grey_r0 | 0.755 | 0.829 | 0.043 | 0.148 |
| cnn_b10_grey_r1 | 0.801 | 0.877 | 0.109 | 0.381 |
| cnn_b10_grey_r2 | 0.800 | 0.877 | 0.119 | 0.396 |
| cnn_li_grey_r0 | 0.603 | 0.874 | 0.441 | 0.931 |
| cnn_li_grey_r1 | 0.904 | 0.974 | 0.952 | 0.993 |
| cnn_li_grey_r2 | 0.929 | 0.984 | 1.000 | 1.000 |

## Decision: α 0.05 (the paper's operating point)

Clean FAR 0.0530 → 0.0518. **Li's pooled DR (clean vs Li's four, eq. 2a): 93.5 % → 93.6 %** (recall 0.924 → 0.925).

| jammer | Li's window [−20, +10] dB: before | after | Δ | at PER 0.5: before | after | Δ |
|---|---|---|---|---|---|---|
| clean frames (1 − FAR) | 0.947 | 0.948 | +0.001 | 0.947 | 0.948 | +0.001 |
| Li: barrage | 0.981 | 0.994 | +0.013 | 1.000 | 1.000 | +0.000 |
| Li: single tone | 0.961 | 0.946 | -0.016 | 1.000 | 1.000 | +0.000 |
| Li: successive pulse | 0.982 | 0.993 | +0.011 | 1.000 | 1.000 | +0.000 |
| Li: protocol-aware (Amuru p 0.25) | 0.773 | 0.769 | -0.004 | 0.758 | 0.775 | +0.017 |
| Amuru pulsed p 0.05 | 0.705 | 0.713 | +0.008 | 0.244 | 0.252 | +0.008 |
| Amuru pulsed p 0.02 | 0.665 | 0.660 | -0.005 | 0.143 | 0.145 | +0.002 |
| genie flip (upper bound) | 0.048 | 0.050 | +0.002 | 0.053 | 0.051 | -0.002 |
| cnnGAN1 grey-box, seen (r0, r1) | 0.847 | 0.885 | +0.038 | 0.256 | 0.386 | +0.130 |
| cnnGAN1 grey-box, held out (r2) | 0.871 | 0.908 | +0.038 | 0.367 | 0.560 | +0.193 |

Every attacker evaluated (one row per tag; recall, window A and matched PER 0.5):

| tag | window: before | after | PER 0.5: before | after |
|---|---|---|---|---|
| omniscient_e1 | 0.048 | 0.050 | 0.053 | 0.051 |
| omniscient_e0.1 | 0.022 | 0.018 | 0.053 | 0.047 |
| noise | 0.981 | 0.994 | 1.000 | 1.000 |
| pulsed_p1 | 0.817 | 0.802 | 0.991 | 0.991 |
| pulsed_p0.5 | 0.798 | 0.791 | 0.926 | 0.945 |
| pulsed_p0.25 | 0.773 | 0.769 | 0.758 | 0.775 |
| pulsed_p0.1 | 0.734 | 0.740 | 0.402 | 0.435 |
| pulsed_p0.05 | 0.705 | 0.713 | 0.244 | 0.252 |
| pulsed_p0.02 | 0.665 | 0.660 | 0.143 | 0.145 |
| pulsed_p0.01 | 0.550 | 0.537 | 0.182 | 0.186 |
| tone | 0.961 | 0.946 | 1.000 | 1.000 |
| pulse_comb | 0.982 | 0.993 | 1.000 | 1.000 |
| zhou_cgan | 0.924 | 0.920 | 1.000 | 1.000 |
| control_r0 | 0.826 | 0.838 | 0.273 | 0.264 |
| control_r1 | 0.890 | 0.903 | 0.499 | 0.549 |
| control_r2 | 0.754 | 0.781 | 0.124 | 0.139 |
| cnn_b10_r0 | 0.834 | 0.874 | 0.168 | 0.262 |
| cnn_b10_r1 | 0.864 | 0.903 | 0.249 | 0.395 |
| cnn_b10_r2 | 0.858 | 0.899 | 0.282 | 0.454 |
| energy_b10_r0 | 0.760 | 0.781 | 0.142 | 0.151 |
| kurtosis_b10_r0 | 0.950 | 0.962 | 0.998 | 1.000 |
| random_push_e0.1 | 0.661 | 0.680 | 0.082 | 0.078 |
| cnn_li_r0 | 0.622 | 0.719 | 0.306 | 0.690 |
| cnn_li_r1 | 0.618 | 0.721 | 0.033 | 0.139 |
| cnn_li_r2 | 0.633 | 0.733 | 0.023 | 0.130 |
| cnn_b10_grey_r0 | 0.823 | 0.863 | 0.160 | 0.241 |
| cnn_b10_grey_r1 | 0.870 | 0.907 | 0.353 | 0.532 |
| cnn_b10_grey_r2 | 0.871 | 0.908 | 0.367 | 0.560 |
| cnn_li_grey_r0 | 0.927 | 0.963 | 0.980 | 0.999 |
| cnn_li_grey_r1 | 0.974 | 0.992 | 1.000 | 1.000 |
| cnn_li_grey_r2 | 0.978 | 0.993 | 1.000 | 1.000 |
