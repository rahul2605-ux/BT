"""
sim08 milestone 1 — channel-aware effectiveness–detectability frontier.

Ports the Phase-0 frontier onto a realistic frequency-selective fading channel
with finite SNR (Sionna TDL). Adds the decisive new axis: CHANNEL-AWARE vs BLIND
subcarrier selection. On a fading channel the effective interference on subcarrier
n (after the RX equalizes the TX link) is (h_jam[n]/h_tx[n])·jam[n], so a jammer
that concentrates energy where it is strong relative to the TX does far more damage
than one that jams random subcarriers — this is the gap a learned, channel-aware
jammer can exploit, and why the lossless-channel result alone is not enough.

Strategies (single interfering source, n_jammers=1):
  clean              — jammer off (BER floor + detector false-alarm reference)
  sparse_blind       — n_active random in-band SCs (blind baseline)
  sparse_channelaware- n_active in-band SCs with the largest |h_jam/h_tx| (genie
                       upper bound the learner approaches)
  broadband_inband   — all 52 in-band SCs

Detector note: pass --detector-model the CHANNEL-VALID CNN from milestone 2
(retrain_detector_channel.py → artifacts/sim08/detector/run001_best.pt). With that
detector the CNN P(det) is meaningful, and this script folds it into a per-sample
detector SUITE (CNN hit OR energy hit) so the effectiveness–detectability frontier
is evaluated against BOTH detectors together — the paper's core figure. Passing the
lossless-trained sim06 detector instead gives indicative CNN numbers only.
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import sionna.phy as sn
import ofdm
from detector import load_detector, detect
from channel import MultiLinkChannel


@torch.no_grad()
def detect_chunked(detector, rx_time_flat, chunk=64):
    out = []
    for i in range(0, rx_time_flat.shape[0], chunk):
        out.append(detect(detector, rx_time_flat[i:i + chunk]))
    return torch.cat(out, dim=0)


@torch.no_grad()
def frame_power(rx_time_flat):
    return rx_time_flat.abs().pow(2).mean(dim=-1)


@torch.no_grad()
def calibrate_energy(chan, no, B, device, far=0.01, n_batches=8):
    """Energy-detector threshold on FADED clean frames at this SNR (target FAR).

    On a fading+noisy channel the clean received power fluctuates, so the
    threshold is looser than on the lossless channel — this is the room a
    low-power jammer might hide in.
    """
    powers = []
    for _ in range(n_batches):
        _, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
        h_tx, h_jam = chan.sample(B)
        zero = torch.zeros(B, ofdm.N_OFDM_SYMBOLS, ofdm.FFT_SIZE,
                           dtype=torch.complex64, device=device)
        rx = chan.apply(tx_grid, [zero], h_tx, h_jam, no)
        powers.append(frame_power(ofdm.modulator(rx)[:, 0, 0]))
    return torch.quantile(torch.cat(powers), 1.0 - far).item()


def _held(mask, power, n_ofdm):
    """mask [B,FFT] real → held complex jam [B,N_OFDM,FFT] (random phase)."""
    B, fft = mask.shape
    phase = torch.rand(B, fft, device=mask.device) * (2 * math.pi)
    vals = mask * (power ** 0.5) * torch.exp(1j * phase.to(torch.complex64))
    return vals[:, None, :].expand(B, n_ofdm, fft).contiguous()


def build_jam(strategy, n_active, power, tx_grid, h_tx, h_jam, eff_full, device):
    B = tx_grid.shape[0]
    fft = ofdm.FFT_SIZE
    n_ofdm = ofdm.N_OFDM_SYMBOLS
    n_eff = eff_full.shape[0]

    if strategy == "clean":
        return torch.zeros(B, n_ofdm, fft, dtype=torch.complex64, device=device)

    if strategy == "broadband_inband":
        mask = torch.zeros(B, fft, device=device)
        mask[:, eff_full] = 1.0
        return _held(mask, power, n_ofdm)

    n_active = min(n_active, n_eff)
    if strategy == "sparse_blind":
        sel = torch.rand(B, n_eff, device=device).argsort(dim=1)[:, :n_active]
    elif strategy == "sparse_channelaware":
        score = h_jam[0].abs().mean(1) / (h_tx.abs().mean(1) + 1e-6)   # [B, FFT]
        sel = score[:, eff_full].argsort(dim=1, descending=True)[:, :n_active]
    else:
        raise ValueError(strategy)
    full_idx = eff_full[sel]
    mask = torch.zeros(B, fft, device=device)
    mask.scatter_(1, full_idx, 1.0)
    return _held(mask, power, n_ofdm)


@torch.no_grad()
def evaluate(strategy, n_active, power, ebno_db, chan, detector, eff_full,
             e_thresh, B, device):
    tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    h_tx, h_jam = chan.sample(B)
    no = float(sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=2,
                                  coderate=1.0, resource_grid=ofdm.resource_grid))
    jam = build_jam(strategy, n_active, power, tx_grid, h_tx, h_jam, eff_full, device)
    rx_grid = chan.apply(tx_grid, [jam], h_tx, h_jam, no)

    rx_time = ofdm.modulator(rx_grid)
    p_jam = detect_chunked(detector, rx_time[:, 0, 0])   # CNN score [B]
    pw = frame_power(rx_time[:, 0, 0])                   # mean power [B]
    cnn_hit = (p_jam > 0.5).float()
    en_hit = (pw > e_thresh).float()
    suite_hit = ((p_jam > 0.5) | (pw > e_thresh)).float()  # per-sample OR = the SUITE

    rx_eq = chan.equalize_zf(rx_grid, h_tx)
    rx_eff = ofdm.remove_nulled(rx_eq)
    ber = ofdm.compute_ber(rx_eff, tx_bits, no=no)

    return {"strategy": strategy, "n_active": int(n_active), "power": float(power),
            "ebno_db": float(ebno_db),
            "p_jam_mean": float(p_jam.mean()), "p_cnn": float(cnn_hit.mean()),
            "p_energy_mean": float(en_hit.mean()), "p_suite": float(suite_hit.mean()),
            "ber_mean": float(ber.mean()), "ber_std": float(ber.std())}


def save_plots(rows, out_dir):
    ebnos = sorted({r["ebno_db"] for r in rows})

    # Fig 1: BER vs EbN0 per strategy (at a representative n_active/power)
    fig, ax = plt.subplots(figsize=(8, 5))
    def series(strat, n_active=None, power=None):
        pts = [r for r in rows if r["strategy"] == strat
               and (n_active is None or r["n_active"] == n_active)
               and (power is None or abs(r["power"] - power) < 1e-9)]
        pts = sorted(pts, key=lambda d: d["ebno_db"])
        return [r["ebno_db"] for r in pts], [r["ber_mean"] for r in pts]
    hi_p = max((r["power"] for r in rows if r["strategy"] != "clean"), default=4.0)
    for strat, lbl, col, na in [
        ("clean", "clean (BER floor)", "black", None),
        ("sparse_blind", "sparse blind (n=8)", "steelblue", 8),
        ("sparse_channelaware", "sparse channel-aware (n=8)", "crimson", 8),
        ("broadband_inband", "broadband in-band", "seagreen", None)]:
        x, y = series(strat, na, None if strat in ("clean",) else hi_p)
        if x:
            ax.semilogy(x, np.clip(y, 1e-5, 1), "-o", color=col, label=lbl)
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("BER")
    ax.set_title(f"sim08 — jamming effectiveness vs SNR (power={hi_p:g})\n"
                 "frequency-selective TDL fading, ZF equalization")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ber_vs_snr.png"), dpi=150)
    plt.close(fig)

    # Fig 2: channel-aware vs blind BER vs n_active (mid SNR)
    mid = ebnos[len(ebnos) // 2]
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, lbl, col in [("sparse_blind", "blind", "steelblue"),
                            ("sparse_channelaware", "channel-aware", "crimson")]:
        pts = sorted([r for r in rows if r["strategy"] == strat
                      and abs(r["ebno_db"] - mid) < 1e-9
                      and abs(r["power"] - hi_p) < 1e-9],
                     key=lambda d: d["n_active"])
        if pts:
            ax.plot([r["n_active"] for r in pts], [r["ber_mean"] for r in pts],
                    "-o", color=col, label=lbl)
    cf = [r for r in rows if r["strategy"] == "clean" and abs(r["ebno_db"] - mid) < 1e-9]
    if cf:
        ax.axhline(cf[0]["ber_mean"], color="black", ls="--", label="clean floor")
    ax.set_xlabel("# active subcarriers"); ax.set_ylabel("BER")
    ax.set_title(f"sim08 — channel-aware vs blind subcarrier selection "
                 f"(Eb/N0={mid:g} dB, power={hi_p:g})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "channelaware_vs_blind.png"), dpi=150)
    plt.close(fig)

    # Fig 3: BER vs energy-detection at mid SNR — can jamming hide under the noise floor?
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, lbl, col in [("sparse_blind", "blind", "steelblue"),
                            ("sparse_channelaware", "channel-aware", "crimson"),
                            ("broadband_inband", "broadband", "seagreen")]:
        pts = [r for r in rows if r["strategy"] == strat and abs(r["ebno_db"] - mid) < 1e-9]
        if pts:
            ax.scatter([r["p_energy_mean"] for r in pts], [r["ber_mean"] for r in pts],
                       c=col, s=42, edgecolor="k", linewidth=0.3, label=lbl)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("P(detect) — energy detector"); ax.set_ylabel("BER")
    ax.set_title(f"sim08 — can jamming hide under the noise floor?\n"
                 f"(energy detector, Eb/N0={mid:g} dB; upper-LEFT = stealthy AND effective)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stealth_vs_energy.png"), dpi=150)
    plt.close(fig)

    # Fig 4: full-suite complementarity — max stealthy BER (P(det)<=0.5) vs SNR,
    # for each detector alone and the CNN+energy suite. If the suite curve drops to
    # the clean floor, the detectors together close the stealthy-effective region.
    if all(k in rows[0] for k in ("p_cnn", "p_suite")):
        fig, ax = plt.subplots(figsize=(8, 5))

        def max_stealthy(ebno, key):
            c = [r for r in rows if r["strategy"] != "clean"
                 and abs(r["ebno_db"] - ebno) < 1e-9 and r[key] <= 0.5]
            return max((r["ber_mean"] for r in c), default=float("nan"))

        for key, lbl, col in [("p_energy_mean", "energy only", "seagreen"),
                              ("p_cnn", "CNN only", "steelblue"),
                              ("p_suite", "CNN + energy (suite)", "crimson")]:
            ax.plot(ebnos, [max_stealthy(e, key) for e in ebnos], "-o", color=col, label=lbl)
        floors = []
        for e in ebnos:
            cf = [r for r in rows if r["strategy"] == "clean" and abs(r["ebno_db"] - e) < 1e-9]
            floors.append(cf[0]["ber_mean"] if cf else float("nan"))
        ax.plot(ebnos, floors, "k--", lw=1, label="clean floor")
        ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("max stealthy BER (P(det)<=0.5)")
        ax.set_title("sim08 m2 — stealthy-effective region vs the detector suite\n"
                     "(realistic channel; suite→floor means the region closes)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stealth_suite_vs_snr.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-model",
                    default="../artifacts/sim08/detector/run001_best.pt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="../artifacts/sim08/frontier")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)
    out_dir = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(out_dir, exist_ok=True)

    ofdm.init_device(device)
    detector = load_detector(os.path.join(os.path.dirname(__file__), args.detector_model),
                             device=device)
    for p in detector.parameters():
        p.requires_grad = False

    eff_arr = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(eff_arr.cpu().numpy() if hasattr(eff_arr, "cpu") else eff_arr,
                               dtype=torch.long, device=device)

    # Single interfering source at 0 dB average path gain relative to TX.
    chan = MultiLinkChannel(ofdm.resource_grid, n_jammers=1, device=device,
                            model="C", delay_spread=100e-9, carrier_freq=5.2e9)

    if args.smoke:
        ebno_list = [10, 30]
        n_active_list = [4, 16]
        power_list = [4.0]
        B = min(args.batch, 32)
    else:
        ebno_list = [5, 10, 15, 20, 30]
        n_active_list = [1, 2, 4, 8, 16, 52]
        power_list = [1.0, 4.0]
        B = args.batch

    rows = []
    t0 = time.time()
    for ebno in ebno_list:
        no = float(sn.utils.ebnodb2no(ebno, num_bits_per_symbol=2,
                                      coderate=1.0, resource_grid=ofdm.resource_grid))
        e_thresh = calibrate_energy(chan, no, B, device, far=0.01)
        rc = evaluate("clean", 0, 0.0, ebno, chan, detector, eff_full, e_thresh, B, device)
        rows.append(rc)
        print(f"clean            EbN0={ebno:>3} | BER floor={rc['ber_mean']:.4f} "
              f"P_energy={rc['p_energy_mean']:.3f} (e_thresh={e_thresh:.3f})", flush=True)
        for power in power_list:
            for strat in ("sparse_blind", "sparse_channelaware"):
                for n_active in n_active_list:
                    r = evaluate(strat, n_active, power, ebno, chan, detector,
                                 eff_full, e_thresh, B, device)
                    rows.append(r)
                    print(f"{strat:<19} EbN0={ebno:>3} n={n_active:>3} pwr={power:g} | "
                          f"BER={r['ber_mean']:.4f} P_energy={r['p_energy_mean']:.3f}", flush=True)
            rb = evaluate("broadband_inband", 52, power, ebno, chan, detector,
                          eff_full, e_thresh, B, device)
            rows.append(rb)
            print(f"broadband_inband  EbN0={ebno:>3}       pwr={power:g} | "
                  f"BER={rb['ber_mean']:.4f} P_energy={rb['p_energy_mean']:.3f}", flush=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    save_plots(rows, out_dir)

    # Headline: channel-aware gain over blind at each SNR (n=8, high power)
    hi_p = max(power_list)
    print("\n=== HEADLINE: channel-aware vs blind BER (n=8, power=%g) ===" % hi_p, flush=True)
    for ebno in ebno_list:
        def g(strat):
            m = [r for r in rows if r["strategy"] == strat and r["n_active"] == 8
                 and abs(r["ebno_db"] - ebno) < 1e-9 and abs(r["power"] - hi_p) < 1e-9]
            return m[0]["ber_mean"] if m else float("nan")
        cf = [r for r in rows if r["strategy"] == "clean" and abs(r["ebno_db"] - ebno) < 1e-9]
        floor = cf[0]["ber_mean"] if cf else float("nan")
        print(f"  EbN0={ebno:>3} dB | clean floor={floor:.4f} | "
              f"blind={g('sparse_blind'):.4f} | channel-aware={g('sparse_channelaware'):.4f}",
              flush=True)

    # Headline: does low-power jamming hide under the noise floor from the energy detector?
    print("\n=== ENERGY DETECTOR: max BER achievable while staying stealthy (P_energy<=0.5) ===",
          flush=True)
    for ebno in ebno_list:
        def mb(strat):
            c = [r for r in rows if r["strategy"] == strat
                 and abs(r["ebno_db"] - ebno) < 1e-9 and r["p_energy_mean"] <= 0.5]
            return max((r["ber_mean"] for r in c), default=float("nan"))
        cf = [r for r in rows if r["strategy"] == "clean" and abs(r["ebno_db"] - ebno) < 1e-9]
        floor = cf[0]["ber_mean"] if cf else float("nan")
        print(f"  EbN0={ebno:>3} dB | clean floor={floor:.4f} | stealthy blind={mb('sparse_blind'):.4f} "
              f"| stealthy channel-aware={mb('sparse_channelaware'):.4f}", flush=True)

    # Headline: FULL SUITE (channel-valid CNN + energy). The energy detector misses
    # broadband-thin in-band jamming (spread power = no per-frame power spike); the
    # channel-valid CNN should catch exactly that. Does the region a jammer can be
    # BOTH stealthy AND effective survive when BOTH detectors are active?
    print("\n=== FULL SUITE vs single detectors: max stealthy BER (P(det)<=0.5) over all jammers ===",
          flush=True)
    for ebno in ebno_list:
        def mb_all(key):
            c = [r for r in rows if r["strategy"] != "clean"
                 and abs(r["ebno_db"] - ebno) < 1e-9 and r[key] <= 0.5]
            return max((r["ber_mean"] for r in c), default=float("nan"))
        cf = [r for r in rows if r["strategy"] == "clean" and abs(r["ebno_db"] - ebno) < 1e-9]
        floor = cf[0]["ber_mean"] if cf else float("nan")
        print(f"  EbN0={ebno:>3} dB | floor={floor:.4f} | energy-only={mb_all('p_energy_mean'):.4f} "
              f"| CNN-only={mb_all('p_cnn'):.4f} | SUITE={mb_all('p_suite'):.4f}", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
