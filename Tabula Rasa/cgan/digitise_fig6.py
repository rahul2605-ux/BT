"""
C0 -- digitise Zhou, Tan & Xu 2025 (ISSET), Fig. 6: BER vs JSR for QPSK.

The paper publishes no data, only this figure, and the calibration (C2) and the
success bar (README §2.10) are both read against it. So the digitisation is a
script, not a transcription: it renders the figure from the PDF, calibrates both
axes on the plot's own gridlines, and locates every marker by colour.

    python digitise_fig6.py        # -> paper_fig6.json + ../artifacts/cgan/c0_fig6_digitised.png

Needs only pdftoppm, Pillow and numpy (no torch, no Sionna) -- it is the one
script in cgan/ that may run on the login node: it is image bookkeeping, not
compute.

WHAT THE FIGURE ACTUALLY SHOWS (found while building this, not stated in text)
------------------------------------------------------------------------------
* The markers sit at JSR = -10:2:10 dB (11 points). The axis TICKS are at
  -10:2.5:10, which is what a low-resolution look suggests the grid is.
* The y axis is log BER with labelled gridlines at 1e-1, 1e-3, ..., 1e-9; the
  dashed black line is the paper's "FEC threshold" at 1e-3 and is used here only
  as an independent check of the y calibration.
* At JSR >= 2 dB the three curves overlap and GAN (drawn last) occludes the
  others; points whose marker is partly hidden are flagged, not dropped.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "source_papers", "L.Zhou 2025.pdf")
OUT_JSON = os.path.join(HERE, "paper_fig6.json")
OUT_PNG = os.path.join(HERE, "..", "artifacts", "cgan", "c0_fig6_digitised.png")

# Render window: PDF page 4 (proceedings p. 376), right column, Fig. 6, 400 dpi.
PAGE, DPI = 4, 400
CROP = dict(x=2000, y=2440, W=1380, H=980)

JSR_TICKS = np.arange(-10.0, 10.01, 2.5)     # vertical gridlines
JSR_MARKERS = np.arange(-10.0, 10.01, 2.0)   # where the data points are
LOG_BER_GRID = np.array([-1.0, -5.0, -7.0, -9.0])   # grey gridlines (1e-3 is the black FEC line)

# Stated protocol: 1e4 symbols x 100 trials x 2 bits = 2e6 bits. Below ~1e-6 a
# BER estimate from that many bits is a handful of errors or none at all.
BITS_STATED = 1e4 * 100 * 2
UNMEASURABLE_BELOW = 1e-6


def render():
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "fig6")
        subprocess.run(["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", str(DPI),
                        "-x", str(CROP["x"]), "-y", str(CROP["y"]),
                        "-W", str(CROP["W"]), "-H", str(CROP["H"]),
                        "-png", PDF, stem], check=True)
        png = [f for f in os.listdir(tmp) if f.endswith(".png")][0]
        return np.asarray(Image.open(os.path.join(tmp, png)).convert("RGB")).astype(int)


def plot_box(img):
    """The frame: rows/columns that are dark over most of their length.

    The frame is dark grey (~RGB 97), not black. The FEC dashed line is also
    caught as a dark row, but it lies strictly inside the frame, so min/max
    are unaffected."""
    dark = img.sum(axis=2) < 400
    rows = np.where(dark.mean(axis=1) > 0.5)[0]
    cols = np.where(dark.mean(axis=0) > 0.5)[0]
    return rows.min(), rows.max(), cols.min(), cols.max()


def gridlines(img, axis, box, n_expected, skip_near=None):
    """
    Centres of dashed light-grey gridlines inside the plot box.

    Rows are scored only right of the legend (whose grey border would otherwise
    read as two short gridlines), and rows within 8 px of `skip_near` -- the
    black FEC line, whose anti-aliased edges are grey -- are dropped. A gridline
    must cover each third of the plot on its own, which rejects the grey "FEC
    Threshold" label text.
    """
    top, bot, left, right = box
    m = 8                                          # skip the frame's anti-aliased edge
    c0 = left + (300 if axis == 1 else m)
    sub = img[top + m:bot - m, c0:right - m]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    grey = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (r > 150) & (r < 235)
    thirds = np.array_split(grey, 3, axis=axis)    # axis=1 -> per row, axis=0 -> per column
    frac = np.min([t.mean(axis=axis) for t in thirds], axis=0)
    offset = (top + m) if axis == 1 else c0
    idx = np.where(frac > 0.2)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    centres = [g.mean() + offset for g in groups if len(g)]
    if skip_near is not None:
        centres = [c for c in centres if abs(c - skip_near) > 8]
    if len(centres) != n_expected:
        raise RuntimeError(f"expected {n_expected} gridlines on axis {axis}, found "
                           f"{len(centres)}: {np.round(centres, 1)}")
    return np.array(centres)


def dark_dashed_row(img, box):
    """Row of the black dashed FEC line (independent y-calibration check)."""
    top, bot, left, right = box
    sub = img[top + 3:bot - 2, left + 3:right - 2].sum(axis=2) < 150
    frac = sub.mean(axis=1)
    return int(np.argmax(frac)) + top + 3


def legend_colours(img, box):
    """Series colours, sampled from the legend's line samples (top-left of the box)."""
    top, _, left, _ = box
    region = img[top:top + 140, left:left + 260]
    colours = {}
    # The three legend entries are stacked; within each band take the most
    # saturated pixels, which are the line/marker, not the text.
    for k, name in enumerate(["optimal", "noise", "gan"]):
        band = region[15 + 35 * k:15 + 35 * k + 35, 10:80].reshape(-1, 3)
        sat = band.max(axis=1) - band.min(axis=1)
        colours[name] = np.median(band[sat > 0.8 * sat.max()], axis=0)
    return colours


def find_marker(img, colour, x, box, legend_bottom, half_w=13, min_width=7):
    """
    Marker centre in pixel rows at column x, for one series colour.

    Rows count as "marker" when the series colour spans >= min_width pixels in a
    window of +-half_w columns. A steep connecting line is only ~4 px wide
    horizontally, so it is rejected; a shallow line is wide but only ~5 rows tall
    and passes through the marker anyway. The longest such run of rows is the
    marker; its bbox centre is the data point (matplotlib centres markers on
    their bounding box).
    """
    top, bot, _, _ = box
    y0 = legend_bottom if x < box[2] + 290 else top + 3
    win = img[y0:bot - 2, int(round(x)) - half_w:int(round(x)) + half_w + 1]
    dist = np.sqrt(((win - colour) ** 2).sum(axis=2))
    width = (dist < 70).sum(axis=1)
    rows = np.where(width >= min_width)[0]
    if len(rows) == 0:
        return None, 0
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    run = max(runs, key=len)
    return 0.5 * (run[0] + run[-1]) + y0, len(run)


def main():
    img = render()
    box = plot_box(img)
    top, bot, left, right = box

    fec_row = dark_dashed_row(img, box)
    xs = gridlines(img, axis=0, box=box, n_expected=len(JSR_TICKS))
    ys = gridlines(img, axis=1, box=box, n_expected=len(LOG_BER_GRID), skip_near=fec_row)
    ax_x, bx_x = np.polyfit(JSR_TICKS, xs, 1)            # px = ax*JSR + bx
    ay, by = np.polyfit(LOG_BER_GRID, ys, 1)              # px = ay*log10(BER) + by
    res_x = xs - (ax_x * JSR_TICKS + bx_x)
    res_y = ys - (ay * LOG_BER_GRID + by)
    fec_log = (fec_row - by) / ay                         # should be -3

    colours = legend_colours(img, box)
    legend_bottom = top + 125

    curves, flags, sizes = {}, {}, {}
    for name, colour in colours.items():
        vals, fl, sz = [], [], []
        for jsr in JSR_MARKERS:
            x = ax_x * jsr + bx_x
            yc, h = find_marker(img, colour, x, box, legend_bottom)
            if yc is None:
                # Fully hidden under a curve drawn later (GAN is drawn last).
                vals.append(None); fl.append("not_found_occluded"); sz.append(0)
                continue
            vals.append(float(10 ** ((yc - by) / ay)))
            sz.append(int(h))
            fl.append("")
        # A fully visible marker is as tall as the typical one for its series;
        # a markedly shorter run means it is partly hidden under another curve.
        typical = np.median([s for s in sz if s > 0])
        for i, s in enumerate(sz):
            if s and s < 0.7 * typical:
                fl[i] = "occluded"
            if vals[i] is not None and vals[i] < UNMEASURABLE_BELOW:
                fl[i] = (fl[i] + "," if fl[i] else "") + "unmeasurable_by_stated_protocol"
        curves[name], flags[name], sizes[name] = vals, fl, sz

    decade_px = abs(ay)
    out = {
        "source": "Zhou, Tan & Xu, 'Communication Jamming Waveform Generation Technology "
                  "Based on CGANs', ISSET 2025, Fig. 6 (QPSK), proceedings p. 376 / PDF page 4",
        "method": "pixel-digitised by cgan/digitise_fig6.py from a 400-dpi pdftoppm render; "
                  "axes least-squares-calibrated on the plot's own gridlines; markers located "
                  "by legend colour",
        "snr_db": 30.0,
        "jsr_db": JSR_MARKERS.tolist(),
        "ber": curves,
        "flags": flags,
        "calibration": {
            "px_per_db": float(ax_x),
            "px_per_decade": float(decade_px),
            "gridline_residual_px_x": np.round(res_x, 2).tolist(),
            "gridline_residual_px_y": np.round(res_y, 2).tolist(),
            "fec_line_log10_ber": round(float(fec_log), 3),
        },
        "uncertainty": "about +-3 px on a marker centre, i.e. about +-%.2f decade in BER and "
                       "+-%.2f dB in JSR; larger where flagged occluded"
                       % (3.0 / decade_px, 3.0 / ax_x),
        "stated_test_bits": BITS_STATED,
        "unmeasurable_below": UNMEASURABLE_BELOW,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Overlay: detected points drawn back onto the render, for eyeballing.
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    pil = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for name, vals in curves.items():
        for jsr, v in zip(JSR_MARKERS, vals):
            if v is None:
                continue
            x, y = ax_x * jsr + bx_x, ay * np.log10(v) + by
            d.line([x - 16, y, x + 16, y], fill=(0, 0, 0), width=2)
            d.line([x, y - 16, x, y + 16], fill=(0, 0, 0), width=2)
    d.line([left, fec_row, right, fec_row], fill=(0, 200, 0), width=1)
    pil.save(OUT_PNG)

    print(f"plot box rows {top}-{bot}, cols {left}-{right}")
    print(f"x: {ax_x:.2f} px/dB, gridline residuals {np.round(res_x, 2)}")
    print(f"y: {decade_px:.2f} px/decade, gridline residuals {np.round(res_y, 2)}")
    print(f"FEC line reads log10(BER) = {fec_log:.3f} (expected -3)")
    print("legend colours:", {k: v.astype(int).tolist() for k, v in colours.items()})
    for name in curves:
        print(f"\n{name}")
        for jsr, v, fl, s in zip(JSR_MARKERS, curves[name], flags[name], sizes[name]):
            ber = "--" if v is None else f"{v:.2e}"
            print(f"  JSR {jsr:+5.1f} dB  BER {ber:>9}  run {s:3d}px  {fl}")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
