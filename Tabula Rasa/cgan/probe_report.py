"""
Data for the Track 2 report page (README §3.3k-n): every curve and number the page plots,
pulled from the sweep JSONs into one file. Login node, no Sionna.

    python probe_report.py        # -> artifacts/cgan/probes/report_data.json

Measures (probe_readout.py): the damage-detection trade-off at alpha = 0.05 (PER vs
per-frame P(det), one point per JSR), P(det) at matched PER 0.1 / 0.5, the warden's
minimum detection error probability xi where quantiles were kept, and the frames a
1 %/1 % SPRT on the alarms needs. "Frames per extra alarm" appears only in the one
figure that shows why it was dropped.
"""

import glob
import json
import os

import numpy as np

import probe_readout as P

S = P.S
OUT = os.path.join(S.AD, "..", "probes")
JSR_LO, JSR_HI = -45.0, 15.0            # the part of the -50..+15 dB grid where anything moves


def num(v, nd=4):
    """JSON-safe float: None for nan, 'inf' kept as a large sentinel the page caps."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if np.isinf(v):
        return 1e9
    return round(float(v), nd)


def curve(d, key, dets=("power_one_sided", "spec_cnn")):
    """JSR-indexed PER / BER / P(det) of one jammer, trimmed to the moving part of the grid."""
    jsr = d["meta"]["jsr_db"]
    keep = [i for i, j in enumerate(jsr) if JSR_LO <= j <= JSR_HI]
    p = P.pts(d, key)
    out = dict(jsr=[jsr[i] for i in keep], per=[num(p[i]["per"]) for i in keep],
               ber=[num(p[i]["ber"], 7) for i in keep])
    for det in dets:
        out[det] = [num(S.pdet(p[i], det)) for i in keep]
    return out


def summary(d, key):
    r = P.row(d, key)
    keys = [k for k in r if k.split("_")[0] in ("per01", "per", "sprt01", "sprt05", "xi01", "xi05", "ber")
            or k in ("jsr_per01", "jsr_per", "jsr_ber")]
    return {k: num(r[k]) for k in keys}


def far(d, det):
    return num(S.pdet(d["clean"], det))


# ---------------------------------------------------------------- per probe
def s1():
    P.DETS = ["power_one_sided", "power_csi", "spec_cnn"]
    fs = [f for f in glob.glob(os.path.join(S.AD, "shadow_run003", "shadow_*.json")) if "smoke" not in f]
    levels = sorted((json.load(open(f)) for f in fs), key=lambda d: d["meta"]["shadow_db"])
    rows = {}
    for key in ("pulsed_p0.1", "eff", "spec_cnn_b10"):
        rows[key] = {det: [num(P.row(d, key)[f"per01_{det}"]) for d in levels] for det in P.DETS}
    P.DETS = ["power_one_sided", "kurtosis", "spec_cnn"]
    return dict(sigma=[d["meta"]["shadow_db"] for d in levels], per01=rows)


def s2():
    base = P.load("shadow_run003", "shadow_0.json")
    sync = [P.load("sync_run003", "shadow_0.json")] + [P.load(f"sync_run003_r{r}", "shadow_0.json") for r in (1, 2, 3)]
    asyn = {"spec_cnn_b10": [base] + [P.load(f"async_run003_r{r}", "shadow_0.json") for r in (1, 2, 3)],
            "spec_cnn_b1": [base] + [P.load(f"async_b1_run003_r{r}", "shadow_0.json") for r in (1, 2, 3)]}
    out = dict(far=dict(power=far(base, "power_one_sided"), cnn=far(base, "spec_cnn")), arms={})
    for key in ("spec_cnn_b1", "spec_cnn_b10"):
        out["arms"][key] = dict(
            async_=[dict(curve=curve(d, key), s=summary(d, key)) for d in asyn[key]],
            sync=[dict(curve=curve(d, key), s=summary(d, key)) for d in sync])
    out["qpsk"] = dict(async_=curve(base, "pulsed_p1"), sync=curve(sync[0], "pulsed_p1"),
                       s_async=summary(base, "pulsed_p1"), s_sync=summary(sync[0], "pulsed_p1"))
    return out


def s4():
    base = P.load("snr15_run003_base", "snr_snr_15.json")
    trained = P.load("snr15_run003", "snr_snr_15.json")
    d30 = P.load("shadow_run003", "shadow_0.json")                 # deployed CNN at 30 dB
    out = dict(far=dict(power=far(base, "power_one_sided"), cnn=far(base, "spec_cnn")),
               band=dict(cnn30=curve(d30, "noise"), cnn15=curve(base, "noise"),
                         bands=dict(cnn30=[-48, -16], cnn15=[-39, -7], pow30=[-32, 0], pow15=[-24, 0])),
               classical={k: dict(curve=curve(base, k), s=summary(base, k)) for k in ("noise", "pulsed_p1", "pulsed_p0.1")},
               arms={})
    for key in ("eff", "power_one_sided_b10", "spec_cnn_b1", "spec_cnn_b10"):
        out["arms"][key] = dict(t30=dict(curve=curve(base, key), s=summary(base, key)),
                                t15=dict(curve=curve(trained, key), s=summary(trained, key)))
    # the metric demonstration: frames per extra alarm vs JSR for the untargeted generator
    demo = {}
    for lab, d in (("t30", base), ("t15", trained)):
        c = curve(d, "eff")
        f = S.pdet(d["clean"], "power_one_sided")
        demo[lab] = dict(jsr=c["jsr"], fpa=[num(pr / (pd - f)) if (pd is not None and pd - f >= 2 * np.sqrt(f * (1 - f) / 512)) else None
                                            for pr, pd in zip(c["per"], c["power_one_sided"])],
                         per=c["per"], pdet=c["power_one_sided"], best=num(S.frames_per_alarm(d, P.pts(d, "eff"), "power_one_sided")))
    out["demo"] = demo
    return out


def s5():
    fs = sorted(f for f in glob.glob(os.path.join(S.AD, "noise_unc_run003", "snr_*_unc_*.json")) if "smoke" not in f)
    levels = sorted((json.load(open(f)) for f in fs), key=lambda d: (-d["meta"]["snr_db"], d["meta"]["noise_unc_db"]))
    out = dict(unc=sorted({d["meta"]["noise_unc_db"] for d in levels}), snr={})
    dets = ("power_one_sided", "power_two_sided", "kurtosis", "spec_cnn")
    jams = ("noise", "pulsed_p1", "pulsed_p0.1", "eff", "power_one_sided_b10", "spec_cnn_b10")
    for snr in (30.0, 15.0):
        L = [d for d in levels if d["meta"]["snr_db"] == snr]
        mg = lambda t: 100 * (t["power_one_sided"]["0.05"] / t["power_clean_mean"] - 1)
        o = dict(far={det: dict(naive=[num(d["clean"]["pdet"][det]["0.05"]) for d in L],
                                cfar=[num(d["clean"]["pdet_cfar"][det]["0.05"]) for d in L]) for det in dets},
                 margin=dict(naive=[num(mg(d["meta"]["thresholds"])) for d in L],
                             cfar=[num(mg(d["meta"]["thresholds_cfar"])) for d in L]),
                 jams={})
        for key in jams:
            o["jams"][key] = {w: [dict(curve=curve(P.view(d, w), key), s=summary(P.view(d, w), key)) for d in L]
                              for w in ("naive", "cfar")}
        out["snr"][f"{snr:g}"] = o
    return out


def headline():
    """Frames broken before a 1 %/1 % SPRT on the energy detector's alarms decides, for the
    CNN-targeted generator (run003 beta 10) at PER 0.1, across the conditions the probes set up."""
    rows = []
    d = P.load("shadow_run003", "shadow_0.json")
    rows.append(("30 dB, energy detector knows gain and noise", P.row(d, "spec_cnn_b10")))
    for s in (0.1, 1.0):
        d = json.load(open(os.path.join(S.AD, "shadow_run003", f"shadow_{s:g}.json")))
        rows.append((f"30 dB, shadowing {s:g} dB, naive energy (S1)", P.row(d, "spec_cnn_b10")))
    d = P.load("snr15_run003_base", "snr_snr_15.json")
    rows.append(("15 dB, energy detector knows gain and noise (S4)", P.row(d, "spec_cnn_b10")))
    for u in ("1", "2"):
        d = P.load("noise_unc_run003", f"snr_15_unc_{u}.json")
        rows.append((f"15 dB, noise ±{u} dB, honest CFAR energy (S5)", P.row(P.view(d, "cfar"), "spec_cnn_b10")))
        rows.append((f"15 dB, noise ±{u} dB, naive energy (S5)", P.row(d, "spec_cnn_b10")))
    return [dict(label=lab, pdet=num(r["per01_power_one_sided"]), sprt=num(r["sprt01_power_one_sided"]),
                 xi=num(r["xi01_power_one_sided"])) for lab, r in rows]


def main():
    os.makedirs(OUT, exist_ok=True)
    data = dict(s1=s1(), s2=s2(), s4=s4(), s5=s5(), headline=headline())
    path = os.path.join(OUT, "report_data.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path) / 1024:.0f} KB)")
    for h in data["headline"]:
        print(f"  {h['label']:<52} P(det) {h['pdet']}  SPRT frames broken {h['sprt']}  xi {h['xi']}")


if __name__ == "__main__":
    main()
