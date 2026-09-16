"""
sim08 ablations -- noise level, jammer power, number of jammers (README §3.3c).

Asked for with the supervisor (2026-09-16): on the realistic sim08 channel, which
jammer setting is strongest at matched detectability, and where is the CNN evaded?
Axes are his two named ablations (README §B.2): noise level on a log grid, and
scenario size as the number of jammers, with jammer power on a log grid.

The frozen stack is IMPORTED, never edited (CLAUDE.md): the channel is
simulation08/channel.py, the jammer builder, energy detector and chunked CNN
inference are simulation08/frontier_channel.py, the OFDM chain and the frozen
channel-valid CNN are simulation06/. What is new here is only what the frozen
evaluate() cannot express: N_J > 1 jammers (it applies jammer 0 only), a noiseless
anchor, and the received JSR.

    sbatch submit_sweep.sh                  # array 0-17: one noise level per task
    python -u ablation.py --task 5          # inside a job: Eb/N0 = 12.5 dB
    python -u ablation.py --task 17         # inside a job: noiseless anchor

Per noise level (one task):
  1. calibrate the energy detector on faded clean frames (1% FAR, as sim08), and
     measure the clean reference -- BER floor and each detector's clean FAR, which
     is the stealth budget (CLAUDE.md) -- on 8 x B frames;
  2. sweep N_J x power x n_active, B frames each;
  3. confirm: per N_J and detector (CNN alone, CNN-or-energy suite), take the
     frontier config at budgets {own clean FAR, 0.1, 0.5}, re-run it on fresh
     8 x B frames next to a fresh clean reference, and flag it if its confirmed
     P(detect) exceeds the budget by more than 2 sigma. Picking the max-BER config
     under a noisy P(detect) favours lucky draws; reported numbers come from here.

Jammers: N_J independent, uncoordinated, blind sparse jammers (sim08's
sparse_blind), each through its own TDL-C channel at 0 dB path gain, each on its
own random n_active in-band subcarriers. EQUAL TOTAL POWER: each transmits
power / N_J per active subcarrier, so n_active * power per OFDM symbol for every
N_J. Channel-aware selection is left out: it bought ~0 at matched detectability
(README A.7).
"""

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulation08"))
sys.path.insert(0, os.path.join(HERE, "..", "simulation06"))
import sionna.phy as sn                                                    # noqa: E402
import ofdm                                                                # noqa: E402
from detector import load_detector                                         # noqa: E402
from channel import MultiLinkChannel                                       # noqa: E402
from frontier_channel import (build_jam, frame_power, calibrate_energy,    # noqa: E402
                              detect_chunked)

ART = os.path.join(HERE, "..", "artifacts", "sim08_ablation")
DETECTOR = os.path.join(HERE, "..", "artifacts", "sim08", "detector", "run001_best.pt")

EBNO_GRID = [2.5 * i for i in range(17)]            # 0 .. 40 dB; task 17 = noiseless
N_TASKS = len(EBNO_GRID) + 1
POWERS = [round(10 ** (e / 4), 6) for e in range(-8, 5)]    # 10^-2 .. 10^1, 4 per decade
N_ACTIVE = [1, 2, 4, 8, 16, 32, 52]
N_JAMMERS = [1, 2, 4]
DETECTORS = ("cnn", "suite")
BUDGETS = ("far", 0.1, 0.5)
CNN_THRESHOLD = 0.5                                 # as sim08: the CNN is not FAR-calibrated
ENERGY_FAR = 0.01
# What the frozen CNN was trained on (retrain_detector_channel.py): outside it, OOD.
CNN_TRAIN = dict(ebno_db=(5.0, 30.0), power=(10 ** -0.5, 10 ** 0.9))
# Noiseless anchor: the channel adds no noise; the demapper still needs no > 0. For
# Gray QPSK the APP LLR of each bit is -4a*y/no, so its sign does not depend on no.
NOISELESS_DEMAP_NO = 1e-3
CHANNEL = dict(model="C", delay_spread=100e-9, carrier_freq=5.2e9)     # sim08's


def task_level(task):
    """Array index -> Eb/N0 in dB, or None for the noiseless anchor."""
    return EBNO_GRID[task] if task < len(EBNO_GRID) else None


def level_tag(ebno_db):
    return "noiseless" if ebno_db is None else f"ebno_{ebno_db:g}"


def noise_variances(ebno_db):
    """(channel noise variance, demapper noise variance)."""
    if ebno_db is None:
        return 0.0, NOISELESS_DEMAP_NO
    no = float(sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=1.0,
                                  resource_grid=ofdm.resource_grid))
    return no, no


def setup(seed, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ofdm.init_device(device)
    sn.config.seed = seed
    torch.manual_seed(seed)
    detector = load_detector(DETECTOR, device=device)
    for p in detector.parameters():
        p.requires_grad = False
    eff = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(eff.cpu().numpy() if hasattr(eff, "cpu") else eff,
                               dtype=torch.long, device=device)
    chan = MultiLinkChannel(ofdm.resource_grid, n_jammers=max(N_JAMMERS), device=device,
                            **CHANNEL)
    return device, detector, eff_full, chan


def build_jams(n_jammers, n_active, power, tx_grid, h_tx, h_jam, eff_full, device):
    """N_J independent blind jammers at equal total power -> list of [B, N_OFDM, FFT]."""
    if n_jammers == 0 or power <= 0:
        return []
    return [build_jam("sparse_blind", n_active, power / n_jammers, tx_grid, h_tx, [h_jam[k]],
                      eff_full, device)
            for k in range(n_jammers)]


@torch.no_grad()
def evaluate(chan, detector, eff_full, e_thresh, ebno_db, n_jammers, n_active, power,
             B, n_batches, device):
    """One config on n_batches x B fresh frames. Mirrors frontier_channel.evaluate."""
    no, no_demap = noise_variances(ebno_db)
    acc = dict(ber=0.0, ber2=0.0, cnn=0.0, energy=0.0, suite=0.0, score=0.0, pj=0.0, ps=0.0)
    for _ in range(n_batches):
        tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
        h_tx, h_jam = chan.sample(B)
        jams = build_jams(n_jammers, n_active, power, tx_grid, h_tx, h_jam, eff_full, device)
        h_used = h_jam[:len(jams)]
        rx_grid = chan.apply(tx_grid, jams, h_tx, h_used, no)

        rx_time = ofdm.modulator(rx_grid)[:, 0, 0]
        score = detect_chunked(detector, rx_time)
        cnn_hit = score > CNN_THRESHOLD
        en_hit = frame_power(rx_time) > e_thresh
        ber = ofdm.compute_ber(ofdm.remove_nulled(chan.equalize_zf(rx_grid, h_tx)),
                               tx_bits, no=no_demap).double()

        acc["ber"] += float(ber.sum()); acc["ber2"] += float((ber ** 2).sum())
        acc["cnn"] += float(cnn_hit.sum()); acc["energy"] += float(en_hit.sum())
        acc["suite"] += float((cnn_hit | en_hit).sum()); acc["score"] += float(score.sum())
        acc["ps"] += float((h_tx * tx_grid[:, 0, 0])[..., eff_full].abs().pow(2).mean())
        if jams:
            j_rx = sum(h * j for h, j in zip(h_used, jams))[..., eff_full]
            acc["pj"] += float(j_rx.abs().pow(2).mean())

    n = B * n_batches
    ber_mean = acc["ber"] / n
    ber_std = math.sqrt(max(acc["ber2"] / n - ber_mean ** 2, 0.0))
    return dict(n_jammers=int(n_jammers), n_active=int(n_active), power=float(power),
                total_power=float(n_active * power) if n_jammers else 0.0, frames=n,
                ber_mean=ber_mean, ber_std=ber_std, ber_sem=ber_std / math.sqrt(n),
                p_cnn=acc["cnn"] / n, p_energy=acc["energy"] / n, p_suite=acc["suite"] / n,
                cnn_score_mean=acc["score"] / n,
                jsr_rx_db=(10 * math.log10(acc["pj"] / acc["ps"]) if acc["pj"] > 0 else None))


def frontier_pick(rows, detector, budget):
    """Max-BER config with P(detect) <= budget, or None."""
    ok = [r for r in rows if r[f"p_{detector}"] <= budget + 1e-12]
    return max(ok, key=lambda r: r["ber_mean"]) if ok else None


def excess_sigma(p, n_p, budget, far_n=None):
    """(p - budget) in standard errors; the budget's own error counts when it is a FAR."""
    var = p * (1 - p) / n_p + (budget * (1 - budget) / far_n if far_n else 0.0)
    return (p - budget) / math.sqrt(var) if var > 0 else (math.inf if p > budget else 0.0)


def confirm(rows, clean, run_eval, confirm_batches):
    """Re-run every frontier pick on fresh frames; see module docstring, step 3."""
    clean_c = run_eval(0, 0, 0.0, confirm_batches)
    picks = {}
    for nj in N_JAMMERS:
        sub = [r for r in rows if r["n_jammers"] == nj]
        for det in DETECTORS:
            for b in BUDGETS:
                beta = clean[f"p_{det}"] if b == "far" else b
                r = frontier_pick(sub, det, beta)
                picks[(nj, det, str(b))] = (beta, r)
    cache, out = {}, []
    for (nj, det, b), (beta, r) in picks.items():
        entry = dict(n_jammers=nj, detector=det, budget=b, budget_sweep=beta, pick=None)
        if r is not None:
            key = (r["n_jammers"], r["n_active"], r["power"])
            if key not in cache:
                cache[key] = run_eval(*key, confirm_batches)
            c = cache[key]
            p = c[f"p_{det}"]
            if b == "far":
                beta_c, far_n = clean_c[f"p_{det}"], clean_c["frames"]
            else:
                beta_c, far_n = float(b), None
            sig = excess_sigma(p, c["frames"], beta_c, far_n)
            entry.update(pick=dict(n_active=r["n_active"], power=r["power"],
                                   total_power=r["total_power"]),
                         sweep=dict(ber=r["ber_mean"], p_det=r[f"p_{det}"]),
                         confirmed=c, budget_confirmed=beta_c, excess_sigma=sig,
                         flagged=bool(sig > 2.0))
        out.append(entry)
    return clean_c, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help=f"0..{N_TASKS - 1}")
    ap.add_argument("--run", default="run001")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--clean-batches", type=int, default=8)
    ap.add_argument("--confirm-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ebno = task_level(args.task)
    tag = level_tag(ebno)
    seed = args.seed + args.task
    out_dir = os.path.join(ART, args.run)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"sweep_{tag}.json")

    device, detector, eff_full, chan = setup(seed)
    B = args.batch
    no, no_demap = noise_variances(ebno)
    e_thresh = calibrate_energy(chan, no, B, device, far=ENERGY_FAR, n_batches=args.clean_batches)

    def run_eval(nj, na, p, n_batches):
        return evaluate(chan, detector, eff_full, e_thresh, ebno, nj, na, p, B, n_batches, device)

    t0 = time.time()
    clean = run_eval(0, 0, 0.0, args.clean_batches)
    print(f"{tag}: no={no:.3g} e_thresh={e_thresh:.4f} | clean BER {clean['ber_mean']:.5f} "
          f"FAR cnn {clean['p_cnn']:.4f} energy {clean['p_energy']:.4f} "
          f"suite {clean['p_suite']:.4f}", flush=True)

    meta = dict(run=args.run, task=args.task, ebno_db=ebno, noiseless=ebno is None, tag=tag,
                no=no, no_demap=no_demap, e_thresh=e_thresh, energy_far_target=ENERGY_FAR,
                cnn_threshold=CNN_THRESHOLD, batch=B, clean_batches=args.clean_batches,
                confirm_batches=args.confirm_batches, seed=seed, powers=POWERS,
                n_active=N_ACTIVE, n_jammers=N_JAMMERS, channel=CHANNEL,
                detector=os.path.relpath(DETECTOR, os.path.join(HERE, "..")),
                cnn_train=CNN_TRAIN, jammer="sparse_blind x N_J, equal total power")

    def dump(rows, **extra):
        with open(out, "w") as f:
            json.dump(dict(meta=meta, clean=clean, rows=rows, **extra), f, indent=1)

    rows = []
    for nj in N_JAMMERS:
        for p in POWERS:
            for na in N_ACTIVE:
                r = run_eval(nj, na, p, 1)
                rows.append(r)
                print(f"  N_J={nj} n={na:>2} p={p:<9g} | BER {r['ber_mean']:.4f} "
                      f"cnn {r['p_cnn']:.3f} en {r['p_energy']:.3f} suite {r['p_suite']:.3f} "
                      f"JSR {r['jsr_rx_db']:+.1f} dB", flush=True)
        dump(rows)                                     # checkpoint per N_J
        print(f"  [checkpoint] N_J={nj} done, {len(rows)} rows ({time.time() - t0:.0f}s)",
              flush=True)

    clean_c, confirmed = confirm(rows, clean, run_eval, args.confirm_batches)
    dump(rows, clean_confirmed=clean_c, confirm=confirmed)
    print(f"\n{tag}: confirmed frontier picks (fresh {clean_c['frames']} frames each)")
    for e in confirmed:
        if e["pick"] is None:
            print(f"  N_J={e['n_jammers']} {e['detector']:<5} budget {e['budget']:<5} -> none")
            continue
        c = e["confirmed"]
        print(f"  N_J={e['n_jammers']} {e['detector']:<5} budget {e['budget']:<5} "
              f"(={e['budget_confirmed']:.4f}) -> n={e['pick']['n_active']} "
              f"p={e['pick']['power']:g} | BER {c['ber_mean']:.4f} "
              f"P_det {c['p_' + e['detector']]:.4f} ({e['excess_sigma']:+.1f} sigma)"
              f"{'  FLAGGED' if e['flagged'] else ''}", flush=True)
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
