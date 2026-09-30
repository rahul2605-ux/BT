"""
Explanatory figures for the final experiment's operating point (README §3.4, §4.2 Q13).
Session A, 2026-09-30. numpy / scipy / matplotlib only -- no Sionna, no GPU: runs on the
login node. Kept next to its outputs because final/ belongs to session B.

    python literature_figures.py          # -> fig_*.png / fig_*.pdf here, numbers to stdout

Every number drawn is either a closed form of our link (cgan/link.py conventions:
Es/N0 = SNR + 10 log10(sps), sps 8) or a value read from a source, named where it is used.
"""

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special, stats

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- link constants (final/env.py, cgan/link.py) --------------------------------------------
SPS = 8
SNR_DB = 15.0                                   # our code SNR: power per sample over the 8x band
OFFSET_ES = 10 * math.log10(SPS)                # 9.03 dB
ES_N0_DB = SNR_DB + OFFSET_ES                   # 24.03 dB
EB_N0_DB = ES_N0_DB - 10 * math.log10(2)        # 21.02 dB, QPSK: 2 bits/symbol
N_SYM = 128                                     # symbols per frame
ALPHA = 0.05

# ---- literature values (checked 2026-09-30, full text; README §4.2 Q13) ----------------------
KAKAR_REQ_DB = 14.0         # Kakar 2015, Table 4.3: required RX SNR = 6 + 2 + 6 dB, excess margin 0
LTE_QPSK_MAX_DB = (7 - 4.6176) / 0.5223   # Wang & Abdelhadi 2015 Eq. (1) + Table I: CQI <= 6 is QPSK
K_MAIN_DB, K_EXTRA_DB = 12.0, 28.0         # Matolak & Sun 2017 Parts I-III (L-band / C-band)
SHADOW_S1_DB = (1.0, 3.0)                  # S1's largest two shadowing stds (README §3.3k)
SIGMA_N_DB = (0.5, 1.0)                    # decided main / extra noise-level uncertainty
TS_X_DB = 1.0                              # Tandra & Sahai x = 1 dB; Shellhammer & Tandra +-1 dB

# ---- style (dataviz reference palette, light mode, first three slots) ------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#9a9994"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png/.pdf")


def qfunc(x):
    return 0.5 * special.erfc(x / math.sqrt(2))


# ---- closed forms used by several figures ------------------------------------------------------
def energy_clean_margin(es_n0_db=ES_N0_DB, n=N_SYM, alpha=ALPHA):
    """
    Clean CFAR margin of `energy` = mean_k |z_k|^2 after the matched filter, as a fraction of its
    clean mean. z_k = s_k + n_k, |s_k| = 1, n_k ~ CN(0, N0) iid (Nyquist sampling), so
    (2/N0) sum_k |z_k|^2 ~ noncentral chi^2(2n, 2n/N0): exact, no Gaussian approximation.
    """
    n0 = 10 ** (-es_n0_db / 10)
    tau = stats.ncx2.ppf(1 - alpha, df=2 * n, nc=2 * n / n0) * n0 / (2 * n)
    return tau / (1 + n0) - 1


def rician_power_db_pdf(x_db, k_db):
    """pdf of 10 log10 |h|^2, h Rician with E|h|^2 = 1 (final/link.fading_gain)."""
    k = 10 ** (k_db / 10)
    p = 10 ** (x_db / 10)
    z = 2 * np.sqrt(k * (k + 1) * p)
    f_p = (k + 1) * special.i0e(z) * np.exp(z - k - (k + 1) * p)      # noncentral chi^2, 2 dof
    return f_p * p * math.log(10) / 10


def lognormal_db_pdf(x_db, sigma_db):
    """pdf in dB of a unit-MEAN log-normal factor with dB std sigma (cgan/link.py convention)."""
    mu = -(sigma_db ** 2) * math.log(10) / 20
    return stats.norm.pdf(x_db, mu, sigma_db)


def rician_power_db_quantile(k_db, q, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    k = 10 ** (k_db / 10)
    s = math.sqrt(1 / (2 * (k + 1)))
    h2 = (math.sqrt(k / (k + 1)) + s * rng.standard_normal(n)) ** 2 + (s * rng.standard_normal(n)) ** 2
    x = 10 * np.log10(h2)
    return np.quantile(x, q), x.std(), h2.std()


# ---- (i) SNR ladder --------------------------------------------------------------------------
MARKS = [  # (Es/N0, colour, label, label alignment)
    (KAKAR_REQ_DB, ORANGE, "range edge\n(Kakar, Tab. 4.3)", "center"),
    (ES_N0_DB, BLUE, "our operating\npoint", "center"),
    (30 + OFFSET_ES, NEUTRAL, "earlier\nruns", "center"),
]
MARK_X = [m[0] for m in MARKS]


def fig_snr_ladder():
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    lo, hi = -4, 42                                        # Es/N0 window [dB]
    rails = [  # (y, label, offset: x = value + offset)
        (2.0, "our code SNR\n(per sample, 8× band)", OFFSET_ES),
        (1.0, "$E_s/N_0$\n(stated in the paper)", 0.0),
        (0.0, "$E_b/N_0$\n(QPSK, 2 bit/symbol)", 10 * math.log10(2)),
    ]
    for y, lab, off in rails:
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.0, zorder=1)
        first = math.ceil((lo - off) / 5) * 5
        for v in range(first, int(hi - off) + 1, 5):
            ax.plot([v + off] * 2, [y - 0.06, y + 0.06], color=INK2, lw=1.0)
            if min(abs(v + off - m) for m in MARK_X) > 1.6:      # keep clear of the marked lines
                ax.text(v + off, y - 0.14, f"{v}", ha="center", va="top", fontsize=8, color=INK2)
        ax.text(lo - 0.8, y, lab, ha="right", va="center", fontsize=8.5, color=INK)

    # QPSK regime under LTE link adaptation (the dropped argument)
    ax.axvspan(lo, LTE_QPSK_MAX_DB, color=NEUTRAL, alpha=0.13, lw=0, zorder=0)
    ax.text((lo + LTE_QPSK_MAX_DB) / 2, -0.35, "adaptive LTE\nwould use\nQPSK (CQI 1–6)",
            ha="center", va="top", fontsize=7.6, color=INK2)

    for x, c, lab, ha in MARKS:
        ax.plot([x, x], [-0.25, 2.25], color=c, lw=2.0, zorder=3, solid_capstyle="round")
        for y, _, off in rails:
            ax.plot(x, y, "o", ms=6.5, color=c, mec=SURFACE, mew=1.6, zorder=4)
            ax.text(x + 0.35, y + 0.09, f"{x - off:.1f}".rstrip("0").rstrip("."), fontsize=7.8,
                    color=INK, ha="left", va="bottom", zorder=5)
        dx = {"right": -0.6, "left": 0.6, "center": 0.0}[ha]
        ax.text(x + dx, 2.62, lab, ha=ha, va="bottom", fontsize=7.8, color=INK)

    # the +10 dB margin
    ax.annotate("", xy=(ES_N0_DB, -0.52), xytext=(KAKAR_REQ_DB, -0.52),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0))
    ax.text((ES_N0_DB + KAKAR_REQ_DB) / 2, -0.62,
            "+10 dB = 1/√10 ≈ 1/3 of the range in free space",
            ha="center", va="top", fontsize=7.8, color=INK2)
    ax.set_xlim(lo - 11.5, hi + 0.5)
    ax.set_ylim(-1.0, 3.35)
    ax.set_title("One operating point, three units: our 15 dB is $E_s/N_0$ 24 dB")
    save(fig, "fig_snr_ladder")


# ---- (ii) per-frame received power under fading vs S1 shadowing ------------------------------
def fig_frame_power(margin):
    margin_db = 10 * math.log10(1 + margin)
    x = np.linspace(-7, 4, 4001)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for s, ls in zip(SHADOW_S1_DB, ("--", ":")):
        ax.plot(x, lognormal_db_pdf(x, s), color=NEUTRAL, lw=1.4, ls=ls,
                label=f"S1 log-normal shadowing, σ = {s:g} dB")
    for k, c, role in ((K_MAIN_DB, BLUE, "main, L-band"), (K_EXTRA_DB, ORANGE, "extra, C-band")):
        ax.plot(x, rician_power_db_pdf(x, k), color=c, lw=2.0,
                label=f"Rician K = {k:g} dB ({role})")
        q95, _, _ = rician_power_db_quantile(k, 0.95)
        y95 = rician_power_db_pdf(np.array([q95]), k)[0]
        ax.plot([q95, q95], [0, y95], color=c, lw=1.0, ls=(0, (2, 2)))
        ax.text(q95 + 0.08, y95 + 0.03, f"95th pct\n+{q95:.2f} dB", fontsize=7.6, color=INK2,
                va="bottom")
    ax.axvline(margin_db, color=INK, lw=1.0, zorder=0)
    ax.annotate(f"energy detector's clean margin at our 15 dB:\n+{100 * margin:.2f} % of frame power"
                f" = +{margin_db:.3f} dB (α = 0.05)",
                xy=(margin_db, 1.45), xytext=(-6.8, 1.45), fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), va="center")
    ax.set_xlim(-7, 4)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("per-frame received signal power relative to its mean [dB]")
    ax.set_ylabel("density [1/dB]")
    ax.legend(loc="upper center", fontsize=7.8, bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Per-frame gain spread dwarfs the energy detector's clean margin")
    save(fig, "fig_frame_power")


# ---- (iii) per-frame noise level vs Tandra & Sahai's +-1 dB -------------------------------------
def fig_noise_level():
    x = np.linspace(-4, 4, 2001)
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.axvspan(-TS_X_DB, TS_X_DB, color=NEUTRAL, alpha=0.14, lw=0, zorder=0)
    ax.plot([-TS_X_DB, TS_X_DB], [1 / (2 * TS_X_DB)] * 2, color=NEUTRAL, lw=1.4, ls="--",
            label="uniform on ±1 dB (Shellhammer & Tandra's Bayesian model)")
    for s, c, role in zip(SIGMA_N_DB, (BLUE, ORANGE), ("main", "extra")):
        out = 2 * stats.norm.sf(TS_X_DB / s)
        ax.plot(x, lognormal_db_pdf(x, s), color=c, lw=2.0,
                label=f"$\\sigma_N$ = {s:g} dB ({role}): {100 * out:.1f} % of frames outside ±1 dB")
    ax.text(0, 0.9, "±1 dB: Tandra & Sahai's x = 1 dB;\nShellhammer & Tandra: ≥ 0.7 dB,"
            " rounded up to 1 dB", ha="center", va="top", fontsize=7.8, color=INK2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("per-frame noise variance relative to its nominal value [dB]")
    ax.set_ylabel("density [1/dB]")
    ax.legend(loc="upper center", fontsize=7.8, bbox_to_anchor=(0.5, -0.16), ncol=1)
    ax.set_title("$\\sigma_N$ = 0.5 dB keeps 95 % of frames inside the ±1 dB bound; 1 dB does not")
    save(fig, "fig_noise_level")


# ---- (iv) uncoded QPSK BER -----------------------------------------------------------------------
def fig_qpsk_ber():
    es = np.linspace(0, 26, 521)
    ber = qfunc(np.sqrt(10 ** (es / 10)))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.semilogy(es, ber, color=BLUE, lw=2.0)
    ax.axhline(3e-4, color=NEUTRAL, lw=1.2, ls="--")
    ax.text(0.3, 3e-4 * 1.8, "excess BER 3·10⁻⁴: the damage level of the result table",
            fontsize=7.8, color=INK2)
    b14 = qfunc(math.sqrt(10 ** (KAKAR_REQ_DB / 10)))
    ax.plot(KAKAR_REQ_DB, b14, "o", ms=7, color=ORANGE, mec=SURFACE, mew=1.6, zorder=4)
    ax.text(KAKAR_REQ_DB + 0.4, b14, f"range edge, 14 dB:\nBER 2.7·10⁻⁷", fontsize=7.8,
            va="center")
    b24 = qfunc(math.sqrt(10 ** (ES_N0_DB / 10)))
    ax.annotate(f"our operating point, 24 dB:\nclean BER ≈ 3·10⁻⁵⁷ (off the axis)",
                xy=(ES_N0_DB, 1e-15), xytext=(17.6, 3e-13), fontsize=7.8,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xlim(0, 26)
    ax.set_ylim(1e-15, 1)
    ax.set_xlabel("$E_s/N_0$ [dB]")
    ax.set_ylabel("BER")
    ax.set_title("Uncoded QPSK, BER = $Q(\\sqrt{E_s/N_0})$: no clean errors at 24 dB")
    save(fig, "fig_qpsk_ber")


# ---- numbers quoted in the report --------------------------------------------------------------
def report_numbers(margin):
    print(f"Es/N0 = {ES_N0_DB:.2f} dB, Eb/N0 = {EB_N0_DB:.2f} dB, offset {OFFSET_ES:.2f} dB")
    print(f"LTE QPSK below SNR {LTE_QPSK_MAX_DB:.2f} dB; 16QAM below {(10 - 4.6176) / 0.5223:.2f} dB")
    print(f"clean BER at 14 dB {qfunc(math.sqrt(10 ** 1.4)):.2e}, at 24.03 dB "
          f"{qfunc(math.sqrt(10 ** (ES_N0_DB / 10))):.1e}")
    print(f"free-space range ratio for +10 dB: {10 ** (-(ES_N0_DB - KAKAR_REQ_DB) / 20):.3f}")
    print(f"energy clean margin (post-MF, exact): {100 * margin:.3f} % = "
          f"{10 * math.log10(1 + margin):.4f} dB")
    for k in (K_MAIN_DB, K_EXTRA_DB):
        kk = 10 ** (k / 10)
        q05, _, _ = rician_power_db_quantile(k, 0.05)
        q95, sd_db, sd_lin = rician_power_db_quantile(k, 0.95)
        print(f"K {k:g} dB: power std {sd_lin:.4f} lin (closed form {math.sqrt(1 + 2 * kk) / (kk + 1):.4f}),"
              f" {sd_db:.3f} dB; 5/95 pct {q05:+.2f} / {q95:+.2f} dB;"
              f" diffuse share {100 / (kk + 1):.2f} %")
    for s in SIGMA_N_DB:
        print(f"sigma_N {s:g} dB: outside +-1 dB {100 * 2 * stats.norm.sf(1 / s):.1f} %, "
              f"outside +-2 dB {100 * 2 * stats.norm.sf(2 / s):.2f} %, "
              f"95th pct {stats.norm.ppf(0.95, -(s ** 2) * math.log(10) / 20, s):+.2f} dB")
    print("equivalent log-normal std of a +-1 dB bound:")
    print(f"  as +-2 sigma: 0.50 dB; uniform on +-1 dB: {1 / math.sqrt(3):.3f} dB;"
          f" one-sided tail alpha = 0.05 (S&F rule): {1 / stats.norm.isf(0.05):.3f} dB;"
          f" tail 1e-3 (S&F example): {1 / stats.norm.isf(1e-3):.3f} dB")
    print(f"  RSS of Shellhammer & Tandra's components 0.28/0.20/0.22 dB: "
          f"{math.sqrt(0.28 ** 2 + 0.2 ** 2 + 0.22 ** 2):.3f} dB (their linear sum: 0.70 dB)")
    print("coherence time vs frame (Doppler f_D = v f_c / c; T_c = 0.423/f_D and 9/(16 pi f_D)):")
    for band, fc in (("L", 970e6), ("C", 5060e6)):
        for v in (11.1, 33.3, 90.0):
            fd = v * fc / 3e8
            tc_hi, tc_lo = 0.423 / fd, 9 / (16 * math.pi * fd)
            print(f"  {band}-band v {v:5.1f} m/s: f_D {fd:7.1f} Hz, T_c {1e3 * tc_lo:.2f}-{1e3 * tc_hi:.2f} ms"
                  f" = {tc_lo / 128e-6:.1f}-{tc_hi / 128e-6:.1f} frames at 1 MBd,"
                  f" {tc_lo / (128 / 103.5e3):.2f}-{tc_hi / (128 / 103.5e3):.2f} at 103.5 kBd")


if __name__ == "__main__":
    m = energy_clean_margin()
    report_numbers(m)
    fig_snr_ladder()
    fig_frame_power(m)
    fig_noise_level()
    fig_qpsk_ber()
