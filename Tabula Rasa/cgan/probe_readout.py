"""
Track 2's system-model probes, read out (README §3.3l S2, §3.3m S4, §3.3n S5). Login
node, no Sionna: JSON in, tables out. The metrics are S1's (shadow_figures.py): P(det)
at matched excess BER 3e-4 and at matched excess PER, frames broken per extra alarm
(best over JSR, extra >= 2 sigma), max damage with P(det) <= 0.5.

    python probe_readout.py s2      # sync vs async generators, per seed and mean +- std
    python probe_readout.py s4      # run003 at 15 dB vs generators trained there, + band check
    python probe_readout.py s5      # naive vs honest-CFAR defender per (SNR, noise uncertainty)
"""

import argparse
import copy
import glob
import json
import os

import numpy as np

import shadow_figures as S
from snr_figures import _cross_log

DETS = ["power_one_sided", "kurtosis", "spec_cnn"]


def load(run, name):
    return json.load(open(os.path.join(S.AD, run, name)))


def pts(d, key):
    return d["attacks"].get(key) or d["generators"].get(key)


def matched_per(d, p, det, ref):
    """(P(det), JSR) at the first JSR reaching excess PER `ref`."""
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = np.array([q["per"] for q in p]) - d["clean"].get("per", 0.0)
    x = _cross_log(jsr, np.where(exc > 0, exc, 0.0), ref)
    return (np.nan, None) if x is None else (float(np.interp(x, jsr, [S.pdet(q, det) for q in p])), x)


# ---------------------------------------------------------------- literature measures (2026-09-29)
# The user rejected "frames per extra alarm" (our own E2b unit; its best-over-JSR value is
# mostly the saturated 1/(1 - FAR)). What the literature reports instead, all built here:
#   * the damage-detection trade-off and P(det) at matched damage (stealthy-jamming papers);
#   * the warden's minimum detection error probability xi = min_t (P_FA + P_MD) -- the
#     covert-communication criterion (Bash et al., JSAC 2013) -- threshold-free, from the
#     stored 257-point statistic quantiles (only for sweeps that kept them);
#   * detection delay (sequential detection, Wald's SPRT): frames a defender watching the
#     per-frame alarms needs to decide "jammed" with both error rates 1 %.
SPRT_NUM = 0.99 * np.log(0.99 / 0.01) + 0.01 * np.log(0.01 / 0.99)    # Wald, alpha = beta = 0.01


def xi_auc(q0, q1):
    """(xi, AUC) of a larger-is-suspicious statistic from clean / jammed quantile lists:
    xi = 1 - max_t (F0(t) - F1(t)), AUC = P(jammed stat > clean stat)."""
    probs = np.linspace(0.0, 1.0, len(q0))
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    t = np.union1d(q0, q1)
    f0 = np.interp(t, q0, probs, left=0.0, right=1.0)
    f1 = np.interp(t, q1, probs, left=0.0, right=1.0)
    return 1.0 - max(0.0, float(np.max(f0 - f1))), float(np.mean(np.interp(q1, q0, probs, left=0.0, right=1.0)))


def sprt_frames(p, f):
    """Expected frames a 1 %/1 % SPRT on the alarm bits needs under 'jammed' (inf if p <= f)."""
    if not (0.0 < f < 1.0) or np.isnan(p) or p <= f:
        return np.inf
    p = min(p, 1.0 - 1e-12)
    kl = p * np.log(p / f) + (1.0 - p) * np.log((1.0 - p) / (1.0 - f))
    return SPRT_NUM / kl


def matched_xi(d, p, det, ref):
    """xi at the first JSR reaching excess PER `ref`, or nan if the sweep kept no quantiles."""
    c = d["clean"].get("stat_q", {}).get(det)
    if c is None or any("stat_q" not in q or det not in q["stat_q"] for q in p):
        return np.nan
    jsr = np.array(d["meta"]["jsr_db"], float)
    exc = np.array([q["per"] for q in p]) - d["clean"].get("per", 0.0)
    x = _cross_log(jsr, np.where(exc > 0, exc, 0.0), ref)
    return np.nan if x is None else float(np.interp(x, jsr, [xi_auc(c, q["stat_q"][det])[0] for q in p]))


def row(d, key):
    """All metrics of one jammer in one result file, as a flat dict."""
    p = pts(d, key)
    out = {}
    for det in DETS:
        m, x = S.matched(d, p, det)
        out[f"ber_{det}"], out["jsr_ber"] = m, x
        m, x = matched_per(d, p, det, 0.5)
        out[f"per_{det}"], out["jsr_per"] = m, x
        m, x = matched_per(d, p, det, 0.1)
        out[f"per01_{det}"], out["jsr_per01"] = m, x
        far = S.pdet(d["clean"], det)
        out[f"sprt01_{det}"] = 0.1 * sprt_frames(out[f"per01_{det}"], far)    # frames broken first
        out[f"sprt05_{det}"] = 0.5 * sprt_frames(out[f"per_{det}"], far)
        out[f"xi01_{det}"], out[f"xi05_{det}"] = matched_xi(d, p, det, 0.1), matched_xi(d, p, det, 0.5)
        out[f"fpa_{det}"] = S.frames_per_alarm(d, p, det)
        out[f"budget_{det}"] = S.at_budget(p, det, "per")
    return out


def fmt(v, f="{:.3f}"):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def table(title, entries):
    """entries: [(label, row dict)]."""
    print(f"\n{title}")
    print(f"  {'':<30}{'JSR@BER':>8}  P(det)@BER3e-4 pow/kurt/cnn   P(det)@PER0.5 pow/kurt/cnn"
          f"   frames/extra alarm pow/kurt/cnn   max PER @P(det)<=0.5 pow/kurt/cnn")
    for lab, r in entries:
        print(f"  {lab:<30}{fmt(r['jsr_ber'], '{:+.1f}'):>8}  "
              + " ".join(f"{fmt(r[f'ber_{d}']):>6}" for d in DETS) + "        "
              + " ".join(f"{fmt(r[f'per_{d}']):>6}" for d in DETS) + "       "
              + " ".join(f"{fmt(r[f'fpa_{d}'], '{:.2f}'):>6}" for d in DETS) + "         "
              + " ".join(f"{fmt(r[f'budget_{d}']):>6}" for d in DETS))


def mean_std(entries, keys=("ber_power_one_sided", "ber_spec_cnn", "per_spec_cnn", "fpa_power_one_sided",
                            "fpa_spec_cnn", "jsr_ber")):
    return {k: (np.nanmean([r[k] for r in entries]), np.nanstd([r[k] for r in entries], ddof=1))
            for k in keys}


# ---------------------------------------------------------------- S2
def s2():
    sync = [("r0", load("sync_run003", "shadow_0.json"))] + \
           [(f"r{r}", load(f"sync_run003_r{r}", "shadow_0.json")) for r in (1, 2, 3)
            if os.path.exists(os.path.join(S.AD, f"sync_run003_r{r}", "shadow_0.json"))]
    # async seeds: run003 is r0; beta 10 (task 14) r1-r3 are cold002_4k_r*, beta 1 (task 13)
    # r1-r3 were trained for S2 as async003_r* (same recipe; cold002_4k never ran task 13)
    asyn = [("r0 (run003)", load("shadow_run003", "shadow_0.json"))] + \
           [(f"r{r} ({g}_r{r})", load(f"{e}_r{r}", "shadow_0.json"))
            for e, g in (("async_run003", "cold002_4k"), ("async_b1_run003", "async003")) for r in (1, 2, 3)
            if os.path.exists(os.path.join(S.AD, f"{e}_r{r}", "shadow_0.json"))]
    base = asyn[0][1]
    table("S2 classical + untargeted, 30 dB, r0 only: async (shadow_run003) vs sync (sync_run003)",
          [(f"{k} {m}", row(d, k)) for k in ("pulsed_p1", "pulsed_p0.1", "eff", "power_one_sided_b10")
           for m, d in (("async", base), ("sync", sync[0][1]))])
    for tag in ("spec_cnn_b1", "spec_cnn_b10"):
        ea = [(f"async {s}", row(d, tag)) for s, d in asyn if pts(d, tag)]
        es = [(f"sync {s}", row(d, tag)) for s, d in sync if pts(d, tag)]
        table(f"S2 {tag}, per seed", ea + es)
        for lab, e in (("async", ea), ("sync", es)):
            if len(e) > 1:
                ms = mean_std([r for _, r in e])
                print(f"  {lab} mean +- std over {len(e)} seeds: " + "  ".join(
                    f"{k} {m:.3f} +- {s:.3f}" for k, (m, s) in ms.items()))


# ---------------------------------------------------------------- S4
def s4():
    base = load("snr15_run003_base", "snr_snr_15.json")
    tags = ["eff", "power_one_sided_b10", "spec_cnn_b1", "spec_cnn_b10"]
    print("S4 band check against the 15 dB defender (first JSR where P(det) >= 0.1 / 0.5 / 0.95):")
    for key in ("noise", "pulsed_p1", "eff", "spec_cnn_b10"):
        jsr = base["meta"]["jsr_db"]
        cells = []
        for det in ("power_one_sided", "spec_cnn"):
            p = [S.pdet(q, det) for q in pts(base, key)]
            cells.append(f"{det}: " + " / ".join(fmt(next((j for j, v in zip(jsr, p) if v >= t), None), "{:+.0f}")
                                                  for t in (0.1, 0.5, 0.95)))
        print(f"  {key:<22}" + "   ".join(cells))
    f = os.path.join(S.AD, "snr15_run003", "snr_snr_15.json")
    trained = json.load(open(f)) if os.path.exists(f) else None
    entries = [(k, row(base, k)) for k in ("noise", "pulsed_p1", "pulsed_p0.1")]
    for t in tags:
        entries.append((f"{t} trained 30 dB", row(base, t)))
        if trained and pts(trained, t):
            entries.append((f"{t} trained 15 dB", row(trained, t)))
    far = {d: S.pdet(base["clean"], d) for d in DETS}
    table(f"S4 at 15 dB, CNN retrained at 15 dB | FAR " + " ".join(f"{S.SHORT[d]} {v:.3f}" for d, v in far.items()),
          entries)


# ---------------------------------------------------------------- S5
def view(d, which):
    """The result file as seen by one defender: 'naive' (pdet) or 'cfar' (pdet_cfar)."""
    if which == "naive":
        return d
    v = copy.deepcopy(d)
    for p in [v["clean"]] + [q for grp in ("attacks", "generators") for ps in v[grp].values() for q in ps]:
        p["pdet"] = p["pdet_cfar"]
        if "stat_q_cfar" in p:
            p["stat_q"] = p["stat_q_cfar"]
    return v


def s5():
    fs = sorted(glob.glob(os.path.join(S.AD, "noise_unc_run003", "snr_*_unc_*.json")))
    fs = [f for f in fs if not f.endswith("_smoke.json")]
    levels = sorted((json.load(open(f)) for f in fs),
                    key=lambda d: (-d["meta"]["snr_db"], d["meta"]["noise_unc_db"]))
    print("S5 realised FAR at alpha 0.05 (clean frames), naive / cfar, and the one-sided power "
          "threshold's margin over the clean mean:")
    for d in levels:
        m = d["meta"]
        c = d["clean"]
        marg = lambda t: 100 * (t["power_one_sided"]["0.05"] / t["power_clean_mean"] - 1)
        print(f"  SNR {m['snr_db']:>4g} dB, unc {m['noise_unc_db']:<3g} dB: "
              + "  ".join(f"{S.SHORT[k]} {c['pdet'][k]['0.05']:.3f}/{c['pdet_cfar'][k]['0.05']:.3f}"
                          for k in c["pdet"])
              + f"   margin naive {marg(m['thresholds']):+.2f} %  cfar {marg(m['thresholds_cfar']):+.2f} %")
    keys = [("noise", "noise"), ("pulsed_p1", "matched QPSK"), ("pulsed_p0.1", "Amuru p0.1"),
            ("eff", "GAN beta0"), ("power_one_sided_b10", "GAN pow1s b10"), ("spec_cnn_b10", "GAN cnn b10")]
    for d in levels:
        m = d["meta"]
        for which in ("naive", "cfar"):
            v = view(d, which)
            table(f"S5 SNR {m['snr_db']:g} dB, noise uncertainty {m['noise_unc_db']:g} dB -- {which} defender",
                  [(lab, row(v, k)) for k, lab in keys if pts(v, k)])
    # one line per (detector, jammer, defender) across the levels -- the table the README quotes
    done = [d for d in levels if all(pts(d, k) for k, _ in keys)]
    print("\nS5 across levels, cell = P(det)@BER3e-4 / frames per extra alarm; columns "
          + "  ".join(f"{d['meta']['snr_db']:g}dB,{d['meta']['noise_unc_db']:g}" for d in done))
    for det in ("power_one_sided", "spec_cnn"):
        for k, lab in keys:
            for which in ("naive", "cfar"):
                cells = [f"{fmt(r[f'ber_{det}'])}/{fmt(r[f'fpa_{det}'], '{:.2f}')}"
                         for r in (row(view(d, which), k) for d in done)]
                print(f"  {S.SHORT[det]:<4}{lab:<14}{which:<6}" + "  ".join(f"{c:>11}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["s2", "s4", "s5"])
    {"s2": s2, "s4": s4, "s5": s5}[ap.parse_args().probe]()


if __name__ == "__main__":
    main()
