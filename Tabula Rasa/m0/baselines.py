"""
Attack study, step 1 (README §3.1): naive jammers against QPSK, SER vs N0, at equal power.

    y = s + c + w,   w ~ CN(0, N0),   Es = 1,   h = 1 (M0's link)

FAIRNESS: every jammer gets the same average received power per symbol, P_J = E|c|^2, quoted as
SJR = Es / P_J (Xie's convention). It is a hard budget (README §2.8): each jammer spends
exactly P_J on average.

  barrage        c ~ CN(0, P_J) on every symbol: Gaussian noise, Xie's "noise jamming" and Li's
                 barrage. At one sample per symbol it is extra AWGN, so SER = SER_noise(N0 + P_J).
                 (M0's attacks.py `barrage` is constant-modulus random phase, Xie's "fixed power
                 jamming" -- a different jammer, not used here.)
  random symbol  c = sqrt(P_J) * u, u a QPSK point drawn uniformly and independently of s (the
                 supervisor's baseline, 2026-10-07). Symbol-synchronous and phase-aligned (h = 1).
  NWSE           Xie et al. 2022 (source_papers/NSWE.pdf), Alg. 1. The minimum-norm push that moves s
                 across one decision boundary toward an adjacent symbol (north / south / west / east;
                 on QPSK two of the four point at a neighbour, and one is drawn at random). It needs s:
                 the jammer knows the symbol, as the paper's AE does (it is computed on the labelled
                 waveform x). We do not run the AE search (PGD on a trained demodulator): we use the
                 solution the paper reports it converges to (Fig. 1(c): the push ends ON the decision
                 boundary, "with only a difference in numerical precision"), which for QPSK's sign
                 decisions is known in closed form. Minimum push
                 rho_min = (1 + ETA) / sqrt(2), ETA standing for the paper's "numerical precision".
                 - P_J >= rho_min^2: every symbol is pushed, with the whole budget, rho = sqrt(P_J)
                   (Alg. 1 line 5, "increase the jamming waveform perturbation");
                 - P_J <  rho_min^2: a random fraction t = P_J / rho_min^2 of the symbols is pushed by
                   rho_min (line 7, "time intermittent jamming").
                 The paper writes t = P_J / P_S; on QPSK that spends only P_J / 2, so t uses rho_min^2,
                 which meets the "target average jamming power" its text asks for.
  NWSE blind     the same push size and power rule, but the direction is drawn uniformly from all four
                 (N / S / W / E) without knowing s (M0's `boundary_blind`). On QPSK two of the four
                 cross a boundary and two push the symbol outward, so a pushed symbol errs half as often.

One column per SJR; SER vs N0, linear on top and log below. Each SER is checked against its
closed form (printed); the plot shows the simulation only. A second figure shows the received IQ
constellation y of every jammer at every SJR, at low receiver noise, coloured by decision.

    python baselines.py      # CPU, ~2 min; the login node is fine (M0)
    -> ../artifacts/m0/baselines.{png,json}, baselines_iq.png
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import link

SJR_DB = [0, 5, 10, 15]             # one column each, Xie's 0-15 dB range in 5 dB steps
N0_DB = list(range(-20, 31))        # Es = 1, so Es/N0 runs from +20 to -30 dB
N_SYM = 500_000                     # log floor ~2e-6
IQ_N0_DB = -20                      # IQ figure: low noise, so each jammer's shape shows
IQ_N_SYM = 3000
ETA = 0.01                          # NWSE overshoot past the boundary
SEED = 0
OUT = os.path.join("..", "artifacts", "m0")
R = link.INV_SQRT2                  # per-axis amplitude of a QPSK point

JAMMERS = ["none", "barrage", "random symbol", "NWSE", "NWSE blind"]
COLORS = {"none": "#8a8983", "barrage": "#2a78d6",          # dataviz reference palette,
          "random symbol": "#eb6834", "NWSE": "#1baf7a",    # slots 1-3 + a neutral;
          "NWSE blind": "#1baf7a"}                          # blind = NWSE's hue, dashed
LABELS = {"none": "noise only", "NWSE": "NWSE, knows symbol", "NWSE blind": "NWSE, blind"}
STYLES = {"none": ":", "NWSE blind": "--"}
INK, MUTED = "#1f1f1e", "#6b6a64"


def jam(name, sym, p, gen):
    """The jammer's contribution c at the receiver, E|c|^2 = p."""
    shape = sym.shape
    if name == "none" or p == 0:
        return torch.zeros_like(sym)
    if name == "barrage":
        return link.awgn(shape, math.sqrt(p / 2.0), generator=gen)
    if name == "random symbol":
        u = link.bits_to_symbols(link.random_bits(*shape, generator=gen))
        return math.sqrt(p) * u
    if name in ("NWSE", "NWSE blind"):
        rho_min = (1.0 + ETA) * R
        if p >= rho_min ** 2:
            rho, mask = math.sqrt(p), torch.ones(shape)
        else:
            rho = rho_min
            mask = (torch.rand(shape, generator=gen) < p / rho_min ** 2).float()
        use_i = (torch.rand(shape, generator=gen) < 0.5).float()
        if name == "NWSE":       # toward the boundary: against the sign of the chosen coordinate
            sgn_i, sgn_q = -torch.sign(sym.real), -torch.sign(sym.imag)
        else:                    # blind: a random sign, i.e. N / S / W / E uniformly
            sgn_i = sgn_q = torch.where(torch.rand(shape, generator=gen) < 0.5, 1.0, -1.0)
        dr = rho * sgn_i * use_i
        di = rho * sgn_q * (1.0 - use_i)
        return torch.complex(dr * mask, di * mask)
    raise ValueError(name)


def q(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ser_closed(name, n0, p):
    """Closed-form SER, used only as a check on the simulation."""
    s = math.sqrt(n0 / 2.0)                       # noise std per axis
    clean = q(R / s)                              # per-axis error, no jammer
    if name == "none" or p == 0:
        return 1.0 - (1.0 - clean) ** 2
    if name == "barrage":
        a = q(R / math.sqrt((n0 + p) / 2.0))
        return 1.0 - (1.0 - a) ** 2
    if name == "random symbol":
        a = math.sqrt(p) * R                      # per-axis jammer amplitude, sign random
        e = 0.5 * (q((R + a) / s) + q((R - a) / s))
        return 1.0 - (1.0 - e) ** 2
    if name in ("NWSE", "NWSE blind"):
        rho_min = (1.0 + ETA) * R
        rho, t = (math.sqrt(p), 1.0) if p >= rho_min ** 2 else (rho_min, p / rho_min ** 2)
        pushed = 1.0 - (1.0 - q((R - rho) / s)) * (1.0 - clean)
        if name == "NWSE blind":  # half the pushes go outward, away from every boundary
            pushed = 0.5 * pushed + 0.5 * (1.0 - (1.0 - q((R + rho) / s)) * (1.0 - clean))
        return t * pushed + (1.0 - t) * (1.0 - (1.0 - clean) ** 2)
    raise ValueError(name)


def measure():
    gen = torch.Generator().manual_seed(SEED)
    out = {}
    for sjr_db in SJR_DB:
        p = 10.0 ** (-sjr_db / 10.0)
        out[sjr_db] = {}
        for name in JAMMERS:
            rows, worst, spent = [], 0.0, []
            for n0_db in N0_DB:
                n0 = 10.0 ** (n0_db / 10.0)
                bits = link.random_bits(1, N_SYM, generator=gen)
                sym = link.bits_to_symbols(bits)
                c = jam(name, sym, p, gen)
                power = (c.abs() ** 2).mean().item()
                _, yhat = link.receive(sym, c, math.sqrt(n0 / 2.0), generator=gen)
                ber, ser = link.ber_ser(bits, link.symbols_to_bits(yhat))
                closed = ser_closed(name, n0, p)
                worst = max(worst, abs(ser - closed))
                if name != "none":
                    spent.append(power / p)
                rows.append(dict(n0_db=n0_db, ser=ser, ber=ber, ser_closed=closed,
                                 jammer_power=power, budget=p))
            out[sjr_db][name] = rows
            ratio = f"E|c|^2 / P_J in [{min(spent):.3f}, {max(spent):.3f}]" if spent else ""
            print(f"SJR {sjr_db:2d} dB  {name:13s}  max |SER - closed form| = {worst:.4f}  {ratio}")
    return out


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot(out, path):
    fig, axes = plt.subplots(2, len(SJR_DB), figsize=(3.6 * len(SJR_DB), 7.0), dpi=150,
                             sharex=True, sharey="row")
    for col, sjr_db in enumerate(SJR_DB):
        for row, log in enumerate((False, True)):
            ax = axes[row, col]
            for name in JAMMERS:
                rows = out[sjr_db][name]
                y = [r["ser"] if (r["ser"] > 0 or not log) else float("nan") for r in rows]
                ax.plot([r["n0_db"] for r in rows], y, lw=2, color=COLORS[name],
                        label=LABELS.get(name, name), ls=STYLES.get(name, "-"))
            if log:
                ax.set_yscale("log")
                ax.set_ylim(1e-6, 1.5)
                ax.set_xlabel(r"$N_0$ (dB; $E_s = 1$)", color=INK)
            else:
                ax.set_ylim(0, 1)
                p = 10.0 ** (-sjr_db / 10.0)
                ax.set_title(f"SJR {sjr_db} dB  ($P_J$ = {p:.3g})", color=INK, fontsize=10,
                             loc="left")
            style(ax)
    axes[0, 0].set_ylabel("SER (linear)", color=INK)
    axes[1, 0].set_ylabel("SER (log)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(JAMMERS), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"QPSK, jammers at equal average power ({N_SYM:,} symbols per point)",
                 color=INK, fontsize=11, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_iq(path):
    """Received constellation y = s + c + w per jammer (rows) and SJR (columns), low noise."""
    gen = torch.Generator().manual_seed(SEED + 1)
    sigma = math.sqrt(10.0 ** (IQ_N0_DB / 10.0) / 2.0)
    names = [n for n in JAMMERS if n != "none"]
    fig, axes = plt.subplots(len(names), len(SJR_DB), figsize=(2.9 * len(SJR_DB), 2.9 * len(names)),
                             dpi=150, sharex=True, sharey=True)
    lim = 2.2
    for i, name in enumerate(names):
        for j, sjr_db in enumerate(SJR_DB):
            p = 10.0 ** (-sjr_db / 10.0)
            bits = link.random_bits(1, IQ_N_SYM, generator=gen)
            sym = link.bits_to_symbols(bits)
            _, y = link.receive(sym, jam(name, sym, p, gen), sigma, generator=gen)
            wrong = (link.symbols_to_bits(y) != bits).any(dim=-1)[0]
            y = y[0]
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.axvline(0, color=MUTED, lw=0.8)
            for mask, color, label in ((~wrong, "#2a78d6", "decided correctly"),
                                       (wrong, "#eb6834", "symbol error")):
                ax.scatter(y.real[mask], y.imag[mask], s=3, color=color, alpha=0.5, lw=0,
                           label=label, rasterized=True)
            ax.scatter([R, R, -R, -R], [R, -R, R, -R], s=40, marker="+", color=INK, lw=1.2,
                       zorder=3, label="QPSK points")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
            ax.text(0.03, 0.97, f"SER {wrong.float().mean().item():.2f}", transform=ax.transAxes,
                    va="top", fontsize=8, color=INK)
            if i == 0:
                ax.set_title(f"SJR {sjr_db} dB", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{LABELS.get(name, name)}\nQ", color=INK)
            if i == len(names) - 1:
                ax.set_xlabel("I", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               markerscale=3, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Received IQ, {IQ_N_SYM:,} symbols per panel, $N_0$ = {IQ_N0_DB} dB "
                 f"($E_s/N_0$ = {-IQ_N0_DB} dB); axes = decision boundaries",
                 color=INK, fontsize=11, x=0.01, ha="left", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = measure()
    with open(os.path.join(OUT, "baselines.json"), "w") as f:
        json.dump(dict(n_sym=N_SYM, seed=SEED, eta=ETA, es=1.0, sjr_db=SJR_DB,
                       results={str(k): v for k, v in out.items()}), f, indent=1)
    plot(out, os.path.join(OUT, "baselines.png"))
    plot_iq(os.path.join(OUT, "baselines_iq.png"))
    print("->", os.path.join(OUT, "baselines.{png,json}"), "+ baselines_iq.png")


if __name__ == "__main__":
    main()
