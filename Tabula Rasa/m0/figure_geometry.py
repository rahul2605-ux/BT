"""
M0 -- explanatory figure: why the minimum-energy attack is the bar.

Draws the QPSK constellation, the decision boundaries, and what each attack
does to the received points. This is the picture the whole problem lives in.
Produces artifacts/m0/geometry.png
"""
import os, math, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import link, attacks

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "m0")
os.makedirs(OUT, exist_ok=True)

g = torch.Generator().manual_seed(7)
NF, NS, SIGMA, POWER = 1, 900, 0.12, 0.5

bits = link.random_bits(NF, NS, generator=g)
s = link.bits_to_symbols(bits)

panels = [
    ("none",     1.0, "No attacker\n(clean link)"),
    ("barrage",  1.0, f"Barrage, P={POWER}\n(random direction)"),
    ("boundary", 1.0, f"Boundary, P={POWER}\n(straight at nearest boundary)"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
for ax, (name, duty, title) in zip(axes, panels):
    d, _ = attacks.build_attack(name, s, power=POWER, duty=duty, generator=g)
    _, yhat = link.receive(s, d=d, sigma=SIGMA, generator=g)
    rx = link.symbols_to_bits(yhat)
    err = (rx != bits).any(-1)[0]
    ber, ser = link.ber_ser(bits, rx)

    yy = yhat[0]
    ax.scatter(yy.real[~err], yy.imag[~err], s=7, c="#3b7dd8", alpha=.55,
               label="decoded correctly", linewidths=0)
    ax.scatter(yy.real[err], yy.imag[err], s=9, c="#d1495b", alpha=.75,
               label="symbol error", linewidths=0)

    # decision boundaries = the two axes
    ax.axhline(0, color="k", lw=1.4, zorder=3)
    ax.axvline(0, color="k", lw=1.4, zorder=3)
    # the four transmitted symbols
    c = link.INV_SQRT2
    for sx, sy, lab in [(c, c, "00"), (-c, c, "10"), (c, -c, "01"), (-c, -c, "11")]:
        ax.plot(sx, sy, marker="*", ms=15, c="#f0b429", mec="k", mew=.7, zorder=5)
        ax.annotate(lab, (sx, sy), textcoords="offset points", xytext=(9, 7),
                    fontsize=9, fontweight="bold", zorder=6)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1); ax.set_aspect("equal")
    ax.set_title(f"{title}\nBER={ber:.3f}   SER={ser:.3f}", fontsize=10.5)
    ax.set_xlabel("I  (in-phase)"); ax.grid(alpha=.18)
axes[0].set_ylabel("Q  (quadrature)")
axes[0].legend(loc="upper left", fontsize=8, framealpha=.9)

# annotate the geometry on the last panel
axes[2].annotate("", xy=(0.02, 0.707), xytext=(0.707, 0.707),
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#2a9d4a"))
axes[2].text(0.30, 0.85, "push 0.707\nenergy 0.5", fontsize=8.5,
             color="#2a9d4a", fontweight="bold", ha="center")

fig.suptitle("M0 geometry: an error happens when a point is pushed ACROSS an axis. "
             "Same power, very different effect.", fontsize=11.5, y=1.0)
fig.tight_layout()
p = os.path.join(OUT, "geometry.png")
fig.savefig(p, dpi=145, bbox_inches="tight")
print("wrote", os.path.normpath(p))
