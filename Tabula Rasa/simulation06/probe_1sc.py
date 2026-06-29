"""
Diagnostic probe: what does the CNN detector say when we jam only 1 subcarrier?

Tests the hypothesis: is P(jam)≈0.999 because of action-space dimensionality
(128 dims of random noise) or because ANY jamming triggers the CNN?

Generates clean OFDM frames, injects jam = scale * (-2*tx) on a single data
subcarrier at various power levels, and reports P(jammed) from the frozen CNN.

No training. Runs in ~10 seconds.
"""

import torch
import ofdm
from detector import load_detector, detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ofdm.init_device(device)

detector = load_detector("../artifacts/sim06/detector/run002_best.pt", device=device)
for p in detector.parameters():
    p.requires_grad = False

B = 256
data_mask = ofdm.get_data_mask()  # [N_OFDM, N_EFF_SC] bool

# Pick a data subcarrier: find an FFT index that's a data subcarrier
# Effective SCs exclude guard + DC. FFT indices for effective SCs:
#   left guard = 6, DC = 1, right guard = 5 → effective = indices 6..37 (left) + 39..58 (right)
# But we need an FFT index. Let's pick index 20 (well inside the data region).
TARGET_SC = 20

# Also identify which effective-SC index this maps to (for BER)
# effective SCs are the 52 non-null subcarriers
rg = ofdm.resource_grid
eff_arr = rg.effective_subcarrier_ind
eff_indices = eff_arr.cpu().numpy() if hasattr(eff_arr, 'cpu') else eff_arr
target_eff_idx = list(eff_indices).index(TARGET_SC) if TARGET_SC in eff_indices else None
print(f"Target FFT SC: {TARGET_SC}, effective SC idx: {target_eff_idx}")

# Test various jam strategies at various power scales
scales = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"\n{'Strategy':<25} {'Scale':>6} {'P(jam)':>8} {'BER':>8} {'Jam Power':>10}")
print("-" * 65)

for strategy_name, get_jam in [
    ("1SC: jam=-2*tx", lambda tx_sc, s: -2.0 * s * tx_sc),
    ("1SC: random noise", lambda tx_sc, s: s * torch.randn_like(tx_sc)),
    ("ALL SC: random noise", None),  # special case
]:
    for scale in scales:
        with torch.no_grad():
            tx_bits, tx_syms, tx_grid, tx_time = ofdm.generate_ofdm_frame(B)

            if get_jam is not None:
                # Single-subcarrier jam
                rx_grid = tx_grid.clone()
                for t in range(ofdm.N_OFDM_SYMBOLS):
                    tx_sc = tx_grid[:, 0, 0, t, TARGET_SC]  # [B] complex
                    jam_sc = get_jam(tx_sc, scale)
                    rx_grid[:, 0, 0, t, TARGET_SC] = tx_sc + jam_sc
            else:
                # All-subcarrier random noise
                rx_grid = tx_grid.clone()
                noise = scale * torch.randn_like(tx_grid.real) + 1j * scale * torch.randn_like(tx_grid.real)
                rx_grid = tx_grid + noise

            rx_time = ofdm.modulator(rx_grid)
            rx_time_flat = rx_time[:, 0, 0]

            p_jammed = detect(detector, rx_time_flat)  # [B]

            _, rx_eff = ofdm.demodulate_frame(rx_time)
            ber = ofdm.compute_ber(rx_eff, tx_bits, no=1.0)

            jam_power = 0.0
            if get_jam is not None and scale > 0:
                tx_sc_val = tx_grid[:, 0, 0, :, TARGET_SC]
                jam_vals = get_jam(tx_sc_val, scale)
                jam_power = jam_vals.abs().pow(2).mean().item()
            elif get_jam is None and scale > 0:
                jam_power = (scale**2)  # E[|noise|²] per dim

            label = strategy_name if scale == scales[0] else ""
            print(f"{label:<25} {scale:>6.1f} {p_jammed.mean().item():>8.4f} "
                  f"{ber.mean().item():>8.4f} {jam_power:>10.4f}")
    print()
