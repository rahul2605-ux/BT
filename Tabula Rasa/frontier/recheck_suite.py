"""
Phase 0 recheck — corrected spectrogram (complex STFT) + detector SUITE.

Addresses two faithfulness concerns raised about the original Phase 0 result:
  1. Spectrogram now uses the full complex baseband (two-sided STFT), not just
     the real part — the standard RF representation (detector.py fixed + retrained).
  2. Adds an ENERGY detector (mean received power vs a clean-calibrated threshold)
     alongside the CNN, so the effectiveness–detectability frontier is evaluated
     against the DETECTOR SUITE, not the CNN alone. A loud in-band jammer can fool
     the spectrogram CNN yet be trivially caught by a power meter — the honest
     evaluation must reflect that.

Answers two questions:
  (A) Does the out-of-band finding survive the corrected complex spectrogram?
      (compare broadband in-band vs out-of-band P_cnn)
  (B) Does the stealthy-high-BER region survive adding the energy detector?
      (CNN-only frontier vs suite frontier)

Requires a detector retrained on the complex spectrogram (see submit_recheck.sh).
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation06"))
import ofdm
from detector import load_detector
from frontier_sweep import build_jam, detect_chunked


@torch.no_grad()
def frame_power(rx_time_flat):
    return rx_time_flat.abs().pow(2).mean(dim=-1)


@torch.no_grad()
def evaluate(strategy, n_active, power, detector, eff_full, e_thresh, B, device):
    tx_bits, _, tx_grid, _ = ofdm.generate_ofdm_frame(B)
    if strategy == "clean":
        jam = torch.zeros(B, ofdm.N_OFDM_SYMBOLS, ofdm.FFT_SIZE,
                          dtype=torch.complex64, device=device)
    else:
        jam = build_jam(strategy, n_active, power, tx_grid, eff_full, device)
    rx = tx_grid.clone()
    rx[:, 0, 0] = rx[:, 0, 0] + jam
    rx_time = ofdm.modulator(rx)

    pj = detect_chunked(detector, rx_time[:, 0, 0])   # CNN score in [0,1], [B]
    pw = frame_power(rx_time[:, 0, 0])                # mean power, [B]
    _, rx_eff = ofdm.demodulate_frame(rx_time)
    ber = ofdm.compute_ber(rx_eff, tx_bits)

    cnn_hit = (pj > 0.5).float()
    en_hit = (pw > e_thresh).float()
    suite_hit = ((pj > 0.5) | (pw > e_thresh)).float()
    return {"strategy": strategy, "n_active": int(n_active), "power": float(power),
            "p_cnn": float(cnn_hit.mean()), "p_cnn_score": float(pj.mean()),
            "p_energy": float(en_hit.mean()), "p_suite": float(suite_hit.mean()),
            "ber": float(ber.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-model", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.01, help="energy-detector target FAR")
    ap.add_argument("--out", default="../artifacts/frontier_recheck")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(out_dir, exist_ok=True)
    ofdm.init_device(device)

    det_path = (args.detector_model if os.path.isabs(args.detector_model)
                else os.path.join(os.path.dirname(__file__), args.detector_model))
    det = load_detector(det_path, device=device)
    for p in det.parameters():
        p.requires_grad = False

    eff = ofdm.resource_grid.effective_subcarrier_ind
    eff_full = torch.as_tensor(eff.cpu().numpy() if hasattr(eff, "cpu") else eff,
                               dtype=torch.long, device=device)
    B = args.batch

    # Calibrate energy detector on clean frames -> target FAR.
    with torch.no_grad():
        pw = [frame_power(ofdm.modulator(ofdm.generate_ofdm_frame(B)[2])[:, 0, 0])
              for _ in range(16)]
        clean_pow = torch.cat(pw)
        e_thresh = torch.quantile(clean_pow, 1.0 - args.far).item()
    print(f"energy threshold (clean FAR={args.far}): {e_thresh:.4f}", flush=True)

    rows = []
    r0 = evaluate("clean", 0, 0.0, det, eff_full, e_thresh, B, device)
    rows.append(r0)
    print(f"CLEAN | P_cnn={r0['p_cnn']:.3f} (score {r0['p_cnn_score']:.3f}) "
          f"P_energy={r0['p_energy']:.3f} BER={r0['ber']:.4f}", flush=True)

    n_list = [1, 2, 4, 8, 16, 52]
    p_list = [0.3, 1.0, 2.0, 4.0, 8.0]
    for strat in ["sparse_blind", "broadband", "broadband_oob"]:
        ns = n_list if strat == "sparse_blind" else [52 if strat == "broadband" else 64]
        for n in ns:
            for power in p_list:
                r = evaluate(strat, n, power, det, eff_full, e_thresh, B, device)
                rows.append(r)
                print(f"{strat:<14} n={n:>3} pw={power:>4g} | "
                      f"P_cnn={r['p_cnn']:.3f} P_energy={r['p_energy']:.3f} "
                      f"P_suite={r['p_suite']:.3f} BER={r['ber']:.3f}", flush=True)
    for n in n_list:
        rows.append(evaluate("sparse_matched", n, 0.0, det, eff_full, e_thresh, B, device))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"far": args.far, "e_thresh": e_thresh, "rows": rows}, f, indent=2)

    # ── Plot: CNN-only vs suite frontier ────────────────────────────────
    pts = [r for r in rows if r["strategy"] in ("sparse_blind", "broadband")]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([r["p_cnn"] for r in pts], [r["ber"] for r in pts],
               c="#0E7C88", s=45, edgecolor="k", linewidth=0.3, label="CNN only")
    ax.scatter([r["p_suite"] for r in pts], [r["ber"] for r in pts],
               c="#A8443B", marker="X", s=55, label="CNN + energy (suite)")
    for r in pts:  # connect each config's CNN point to its suite point
        ax.plot([r["p_cnn"], r["p_suite"]], [r["ber"], r["ber"]],
                color="#999", lw=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel("P(detect)"); ax.set_ylabel("BER")
    ax.set_title("Effectiveness vs detectability: CNN alone vs detector suite\n"
                 "(corrected complex-STFT detector; energy detector adds power sensitivity)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "suite_frontier.png"), dpi=150)
    plt.close(fig)

    # ── Headlines ───────────────────────────────────────────────────────
    def maxber(key, tau):
        c = [r for r in pts if r[key] <= tau]
        return max((r["ber"] for r in c), default=float("nan"))

    print("\n=== HEADLINE: max stealthy BER (corrected detector) ===", flush=True)
    for tau in (0.05, 0.10, 0.50):
        print(f"  P(det) <= {tau:.2f} | CNN-only max BER={maxber('p_cnn', tau):.3f}  "
              f"|  SUITE max BER={maxber('p_suite', tau):.3f}", flush=True)

    def pick(strat, power):
        m = [r for r in rows if r["strategy"] == strat and abs(r["power"] - power) < 1e-9]
        return m[0] if m else None
    ib, ob = pick("broadband", 8.0), pick("broadband_oob", 8.0)
    if ib and ob:
        print("\n=== out-of-band check @ power=8 (does the finding survive complex STFT?) ===",
              flush=True)
        print(f"  in-band      P_cnn={ib['p_cnn']:.3f}  BER={ib['ber']:.3f}", flush=True)
        print(f"  out-of-band  P_cnn={ob['p_cnn']:.3f}  BER={ob['ber']:.3f}", flush=True)

    print(f"\nDone. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
