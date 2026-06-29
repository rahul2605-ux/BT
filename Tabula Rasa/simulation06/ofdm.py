"""
OFDM module — Sionna-based OFDM chain on GPU.

Uses sionna.phy.ofdm for modulation/demodulation with 802.11a-like parameters.
Sionna modules are created once at import; call init_device() before use to
move everything to the target device via sn.config.device.

Signal chain:
  BinarySource → Mapper → ResourceGridMapper → OFDMModulator → time domain
  time domain → OFDMDemodulator → RemoveNulledSubcarriers → Demapper → bits
"""

import torch
import sionna.phy as sn

# ── OFDM parameters (802.11a-like) ──────────────────────────────────────────
FFT_SIZE = 64
CP_LENGTH = 16
SYMBOL_LENGTH = FFT_SIZE + CP_LENGTH  # 80 samples per OFDM symbol
N_OFDM_SYMBOLS = 14
N_GUARD_LEFT = 6
N_GUARD_RIGHT = 5
SUBCARRIER_SPACING = 312.5e3
BITS_PER_SYMBOL = 2  # QPSK

# Kronecker pilots: OFDM symbols 2 and 11 are all-pilot
PILOT_OFDM_INDICES = [2, 11]

# ── Module-level state (initialized by init_device) ─────────────────────────
_initialized = False
resource_grid = None
source = None
constellation = None
mapper = None
demapper = None
rg_mapper = None
modulator = None
demodulator = None
remove_nulled = None


def init_device(device='cpu'):
    """Initialize all Sionna modules on the given device. Call once at startup."""
    global _initialized, resource_grid, source, constellation, mapper, demapper
    global rg_mapper, modulator, demodulator, remove_nulled

    device_str = str(device)
    if device_str == "cuda":
        device_str = "cuda:0"
    sn.config.device = device_str

    resource_grid = sn.ofdm.ResourceGrid(
        num_ofdm_symbols=N_OFDM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=CP_LENGTH,
        num_guard_carriers=(N_GUARD_LEFT, N_GUARD_RIGHT),
        dc_null=True,
        pilot_pattern='kronecker',
        pilot_ofdm_symbol_indices=PILOT_OFDM_INDICES,
    )

    source = sn.mapping.BinarySource()
    constellation = sn.mapping.Constellation("qam", num_bits_per_symbol=BITS_PER_SYMBOL)
    mapper = sn.mapping.Mapper(constellation=constellation)
    demapper = sn.mapping.Demapper("app", constellation=constellation)
    rg_mapper = sn.ofdm.ResourceGridMapper(resource_grid)
    modulator = sn.ofdm.OFDMModulator(cyclic_prefix_length=CP_LENGTH)
    demodulator = sn.ofdm.OFDMDemodulator(
        fft_size=FFT_SIZE, l_min=0, cyclic_prefix_length=CP_LENGTH
    )
    remove_nulled = sn.ofdm.RemoveNulledSubcarriers(resource_grid)

    _initialized = True
    print(f"OFDM initialized on {device}: "
          f"{resource_grid.num_data_symbols} data syms, "
          f"{resource_grid.num_pilot_symbols} pilot syms, "
          f"{resource_grid.num_effective_subcarriers} effective SC",
          flush=True)


def _check_init():
    if not _initialized:
        raise RuntimeError("Call ofdm.init_device(device) before using OFDM functions")


# ── Signal generation ───────────────────────────────────────────────────────

def generate_ofdm_frame(batch_size):
    """Generate a complete clean OFDM frame.

    Returns:
        tx_bits:     [B, 1, 1, num_data_symbols * bits_per_symbol] float
        tx_syms:     [B, 1, 1, num_data_symbols] complex
        tx_grid:     [B, 1, 1, N_OFDM, FFT_SIZE] complex resource grid
        time_signal: [B, 1, 1, N_OFDM * SYMBOL_LENGTH] complex time-domain
    """
    _check_init()
    n_data = resource_grid.num_data_symbols
    bits = source([batch_size, 1, 1, n_data * BITS_PER_SYMBOL])
    syms = mapper(bits)
    grid = rg_mapper(syms)
    time_sig = modulator(grid)
    return bits, syms, grid, time_sig


def demodulate_frame(time_signal):
    """Demodulate time-domain signal back to frequency grid.

    Args:
        time_signal: [B, 1, 1, T] complex

    Returns:
        rx_grid:      [B, 1, 1, N_OFDM, FFT_SIZE] complex full grid
        rx_effective: [B, 1, 1, N_OFDM, N_EFF_SC] complex (guard/DC removed)
    """
    _check_init()
    rx_grid = demodulator(time_signal)
    rx_effective = remove_nulled(rx_grid)
    return rx_grid, rx_effective


def compute_ber(rx_effective, tx_bits, no=1.0):
    """Compute BER from received effective subcarriers and transmitted bits.

    Args:
        rx_effective: [B, 1, 1, N_OFDM, N_EFF_SC] complex
        tx_bits:      [B, 1, 1, num_data_symbols * bits_per_symbol] float
        no:           noise variance for demapper

    Returns:
        ber: [B] float — per-batch BER
    """
    _check_init()
    pp = resource_grid.pilot_pattern
    data_mask = (pp.mask[0, 0] == 0)  # [N_OFDM, N_EFF_SC]

    # Extract data symbols only (skip pilot OFDM symbols)
    rx_data = rx_effective[:, 0, 0][..., data_mask]  # [B, num_data_symbols]
    rx_data = rx_data.reshape(rx_effective.shape[0], 1, 1, -1)

    llr = demapper(rx_data, no)
    rx_bits = sn.utils.hard_decisions(llr)
    errors = (tx_bits != rx_bits).float()
    return errors.mean(dim=(1, 2, 3))  # [B]


def soft_ber_loss(rx_effective, tx_bits, no=1.0):
    """Differentiable soft-BER loss (BCE with flipped labels).

    Used in direct-gradient training. For MAPPO, use compute_ber instead.

    Args:
        rx_effective: [B, 1, 1, N_OFDM, N_EFF_SC] complex
        tx_bits:      [B, 1, 1, num_data_symbols * bits_per_symbol] float
        no:           noise variance

    Returns:
        loss: scalar — binary cross-entropy with wrong labels
    """
    _check_init()
    pp = resource_grid.pilot_pattern
    data_mask = (pp.mask[0, 0] == 0)

    rx_data = rx_effective[:, 0, 0][..., data_mask]
    rx_data = rx_data.reshape(rx_effective.shape[0], 1, 1, -1)

    llr = demapper(rx_data, no).clamp(-10, 10)
    wrong_labels = 1.0 - tx_bits
    return torch.nn.functional.binary_cross_entropy_with_logits(llr, wrong_labels)


# ── Convenience accessors ───────────────────────────────────────────────────

def get_data_mask():
    """Boolean mask [N_OFDM, N_EFF_SC] — True for data positions."""
    _check_init()
    return resource_grid.pilot_pattern.mask[0, 0] == 0


def get_num_data_symbols():
    _check_init()
    return resource_grid.num_data_symbols


def get_num_effective_subcarriers():
    _check_init()
    return resource_grid.num_effective_subcarriers


def get_frame_time_samples():
    """Number of time-domain samples per frame."""
    return N_OFDM_SYMBOLS * SYMBOL_LENGTH  # 14 * 80 = 1120
