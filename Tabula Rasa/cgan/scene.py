"""
Baselines -- the 3-D scene (README §2.10, system model decided 2026-09-17).

Positions -> path gains (Sionna RT, empty scene, line of sight only) -> the
transmitter's power control (sionna.sys) -> each jammer's received JSR. That is
ALL the geometry the baselines use. Why so little is left is a set of stated
assumptions, each checked in verify.py:

  * the victim receiver R is synchronised to its own transmitter T, so T's
    delay and carrier phase are R's time and phase reference;
  * every realisable jammer is ASYNCHRONOUS: its clock offset is uniform over a
    symbol and its carrier phase uniform, per frame. A geometric delay or phase
    added on top of a uniform one changes nothing in law, so neither is applied
    (channel.py);
  * T sets its power from the path loss it knows (full path-loss compensation),
    so T's received power, and hence the whole clean received signal, is the
    same in every drop.

So a jammer is fully described at R by its received jamming-to-signal ratio

    JSR_k = P_k * g_kR / P_T,rx,        P_T,rx = SNR * N,

with P_k its transmit power (a hard budget: an equal split of the team's total
P_J), g_kR its path gain to R, and N the receiver's noise power over the
simulated band. SNR and JSR keep link.py's definitions (mean power per sample
over the full simulated band, sps x symbol rate).

DROPS are nested: row order is [T, R, J1, J2, J3, J4], and a K-jammer scenario
uses J1..JK of the same drop, so K = 1..4 are compared on the same placements.
The fixed test drops are built once and cached (test_drops.json) so every
baseline and every later GAN is evaluated on exactly the same 50 placements.

Sionna RT needs Mitsuba. The cluster's driver ships no OptiX (libnvoptix), so
Mitsuba's CUDA variant cannot start (job 2266119); the LLVM CPU variant is used.
Path solving is line-of-sight only in an empty scene, which is cheap on CPU.
"""

import json
import math
import os

import numpy as np
import torch

import link as lk

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "cgan", "baselines")

F_C = 2.4e9                         # carrier [Hz] (ISM band, UAV links)
SYMBOL_RATE = 1e6                   # [Bd]
SPS = lk.LINK["sps"]
BANDWIDTH = SPS * SYMBOL_RATE       # simulated band = sampling rate [Hz]
TEMPERATURE = 290.0                 # [K]
NOISE_FIGURE_DB = 7.0
SNR_DB = lk.SNR_DB                  # 30 dB at R, held by T's power control
TX_MAX_DBM = 60.0                   # power-control cap, never binding in this box (checked)
BOX = (1000.0, 1000.0, 150.0)       # x, y, z extent [m]
MIN_SEPARATION = 50.0               # [m], keeps every node out of the others' near field
K_MAX = 4
N_SYM = 128                         # symbols per frame; one frame is one detector observation
TEST_SEED = 100_000
N_TEST_DROPS = 50
ROWS = ["T", "R"] + [f"J{k + 1}" for k in range(K_MAX)]


def dbm_to_w(dbm):
    return 10.0 ** ((np.asarray(dbm, dtype=float) - 30.0) / 10.0)


def w_to_dbm(w):
    return 10.0 * np.log10(np.asarray(w, dtype=float)) + 30.0


# ---------------------------------------------------------------- positions
def draw_positions(seed, n_nodes=2 + K_MAX):
    """[n_nodes, 3] i.i.d. uniform in BOX, pairwise >= MIN_SEPARATION (rejection)."""
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        p = rng.uniform(0.0, 1.0, size=(n_nodes, 3)) * np.array(BOX)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        if d[np.triu_indices(n_nodes, 1)].min() >= MIN_SEPARATION:
            return p
    raise RuntimeError("could not place nodes with the minimum separation")


# ---------------------------------------------------------------- Sionna RT
def _rt():
    """Import Sionna RT on Mitsuba's LLVM variant (no OptiX on the cluster)."""
    import mitsuba as mi
    if mi.variant() != "llvm_ad_mono_polarized":
        raise RuntimeError(
            f"Mitsuba variant is {mi.variant()!r}: `import sionna` picks the CUDA variant, which needs "
            "OptiX and this cluster has none. Put `import mitsuba as mi; "
            "mi.set_variant('llvm_ad_mono_polarized')` at the top of the entry point, before any "
            "Sionna import (switching afterwards leaves Dr.Jit types mixed).")
    import sionna.rt as rt
    return mi, rt


def rt_scene():
    """An empty RT scene at the link's carrier, band and temperature, isotropic antennas."""
    _, rt = _rt()
    scene = rt.load_scene()
    scene.frequency = F_C
    scene.bandwidth = BANDWIDTH
    scene.temperature = TEMPERATURE
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    return scene


def path_gains(positions, return_delays=False):
    """
    Line-of-sight path power gains |a|^2 from every transmitter of a drop to its
    receiver, computed by Sionna RT.

    positions: [D, 2 + K, 3] rows [T, R, J1..JK]. All drops go into ONE empty
    scene (D receivers, D*(1+K) transmitters) and are solved together; only the
    pairs within a drop are read back. Returns g [D, 1 + K] (T first), and
    optionally the delays tau [D, 1 + K] in seconds.
    """
    mi, rt = _rt()
    positions = np.asarray(positions, dtype=float)
    D, n = positions.shape[:2]
    scene = rt_scene()
    tx_rows = [0] + list(range(2, n))
    for d in range(D):
        for i in tx_rows:
            scene.add(rt.Transmitter(name=f"t{d}_{i}",
                                      position=mi.Point3f(*map(float, positions[d, i]))))
        scene.add(rt.Receiver(name=f"r{d}", position=mi.Point3f(*map(float, positions[d, 1]))))
    paths = rt.PathSolver()(scene, max_depth=0, los=True, specular_reflection=False,
                            refraction=False)
    a, tau = paths.cir(normalize_delays=False, out_type="torch")
    # a: [num_rx, rx_ant, num_tx, tx_ant, paths, time]; tau: [num_rx, rx_ant, num_tx, tx_ant, paths]
    # or [num_rx, num_tx, paths] (synthetic arrays). Transmitters were added in drop order.
    g = np.zeros((D, len(tx_rows)))
    t = np.zeros((D, len(tx_rows)))
    tau = tau.reshape(D, -1, D * len(tx_rows), tau.shape[-1])[:, 0]
    for d in range(D):
        cols = torch.as_tensor(d * len(tx_rows) + np.arange(len(tx_rows)), device=a.device)
        g[d] = a[d, 0, cols, 0, 0, 0].abs().pow(2).double().cpu().numpy()
        t[d] = tau[d, cols, 0].double().cpu().numpy()
    return (g, t) if return_delays else g


def noise_power_w():
    """Receiver noise over the simulated band: Sionna RT's k*T*B, times the noise figure."""
    return float(rt_scene().thermal_noise_power[0]) * 10.0 ** (NOISE_FIGURE_DB / 10.0)


def free_space_gain(d):
    """(lambda / 4 pi d)^2 -- the closed form RT is checked against."""
    lam = 299_792_458.0 / F_C
    return (lam / (4.0 * math.pi * np.asarray(d, dtype=float))) ** 2


# ---------------------------------------------------------------- power control
def tx_power_w(g_tr, noise_w):
    """
    T's transmit power [W]: Sionna's open-loop power control with full path-loss
    compensation (alpha = 1), targeting received power SNR * N at R. One
    "subcarrier", so the 3GPP per-PRB term is 10 log10(1) = 0.
    """
    import sionna.sys as ss
    p_rx_dbm = float(w_to_dbm(noise_w * 10.0 ** (SNR_DB / 10.0)))
    g = torch.as_tensor(np.asarray(g_tr, dtype=float), dtype=torch.float64)
    p = ss.open_loop_uplink_power_control(1.0 / g, torch.ones_like(g), alpha=1.0,
                                          p0_dbm=p_rx_dbm, ut_max_power_dbm=TX_MAX_DBM)
    return p.double().cpu().numpy()


def received_jsr(g_jr, budget_dbm, K, p_t_rx_w):
    """
    Received JSR (linear) of jammers J1..JK when they split a total transmit budget
    P_J [dBm] equally: JSR_k = (P_J/K) g_kR / P_T,rx. g_jr: [..., >= K].
    """
    g = np.asarray(g_jr, dtype=float)[..., :K]
    return dbm_to_w(budget_dbm) / K * g / p_t_rx_w


def omniscient_jsr(g_jr, budget_dbm, K, p_t_rx_w):
    """
    The genie team's coherent optimum: with the budget split P_k ∝ g_kR the
    amplitudes add, (sum_k sqrt(P_k g_kR))^2 = P_J sum_k g_kR. Linear JSR.
    """
    return dbm_to_w(budget_dbm) * np.asarray(g_jr, dtype=float)[..., :K].sum(-1) / p_t_rx_w


# ---------------------------------------------------------------- the fixed test drops
def build_test_drops(n=N_TEST_DROPS, seed=TEST_SEED):
    """Positions, RT path gains, noise, T's power and realised SNR for the test drops."""
    positions = np.stack([draw_positions(seed + i) for i in range(n)])
    g, tau = path_gains(positions, return_delays=True)
    noise = noise_power_w()
    p_t = tx_power_w(g[:, 0], noise)
    p_t_rx = noise * 10.0 ** (SNR_DB / 10.0)
    return dict(
        seed=seed, rows=ROWS, f_c=F_C, bandwidth=BANDWIDTH, temperature=TEMPERATURE,
        noise_figure_db=NOISE_FIGURE_DB, snr_db=SNR_DB, box=BOX, min_separation=MIN_SEPARATION,
        noise_w=noise, p_t_rx_w=p_t_rx,
        positions=positions.tolist(),
        g=g.tolist(),                       # [D, 1 + K_MAX], T first
        tau_s=tau.tolist(),
        p_t_w=p_t.tolist(),
        snr_realised_db=(10 * np.log10(p_t * g[:, 0] / noise)).tolist(),
    )


def load_test_drops(path=None, rebuild=False):
    """Cached test drops (built with Sionna RT on first use)."""
    path = path or os.path.join(ART, "test_drops.json")
    if rebuild or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        drops = build_test_drops()
        with open(path, "w") as f:
            json.dump(drops, f, indent=1)
    with open(path) as f:
        drops = json.load(f)
    for k in ("positions", "g", "tau_s", "p_t_w", "snr_realised_db"):
        drops[k] = np.asarray(drops[k])
    return drops
