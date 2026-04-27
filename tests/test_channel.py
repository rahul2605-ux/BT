import numpy as np
import pytest
from channel.base import MockChannel

N_SUB = 16
NOISE = -90.0


def _pos(x, y, z):
    return np.array([[x, y, z]], dtype=np.float32)


def test_output_shape():
    ch = MockChannel()
    sinr = ch.compute_sinr(
        tx_positions=_pos(0, 0, 10),
        rx_positions=_pos(50, 0, 10),
        jam_positions=_pos(90, 0, 10),
        tx_power_dbm=np.array([20.0]),
        jam_power_dbm=np.array([[30.0] * N_SUB]),
        noise_power_dbm=NOISE,
        n_subcarriers=N_SUB,
    )
    assert sinr.shape == (1, N_SUB)
    assert sinr.dtype == np.float32


def test_all_values_finite():
    ch = MockChannel()
    sinr = ch.compute_sinr(
        tx_positions=_pos(0, 0, 0),
        rx_positions=_pos(10, 0, 0),
        jam_positions=_pos(5, 0, 0),
        tx_power_dbm=np.array([20.0]),
        jam_power_dbm=np.array([[30.0] * N_SUB]),
        noise_power_dbm=NOISE,
        n_subcarriers=N_SUB,
    )
    assert np.all(np.isfinite(sinr))


def test_no_jammer_gives_higher_sinr():
    ch = MockChannel(seed=0)
    tx = _pos(0, 0, 10)
    rx = _pos(20, 0, 10)
    tx_pwr = np.array([20.0])

    sinr_no_jam = ch.compute_sinr(
        tx_positions=tx,
        rx_positions=rx,
        jam_positions=np.empty((0, 3)),
        tx_power_dbm=tx_pwr,
        jam_power_dbm=np.empty((0, N_SUB)),
        noise_power_dbm=NOISE,
        n_subcarriers=N_SUB,
    )
    sinr_with_jam = ch.compute_sinr(
        tx_positions=tx,
        rx_positions=rx,
        jam_positions=_pos(20, 0, 10),   # jammer right on top of RX
        tx_power_dbm=tx_pwr,
        jam_power_dbm=np.array([[30.0] * N_SUB]),
        noise_power_dbm=NOISE,
        n_subcarriers=N_SUB,
    )
    assert np.all(sinr_no_jam > sinr_with_jam)


def test_closer_jammer_lowers_sinr():
    ch = MockChannel(seed=0)
    tx = _pos(0, 0, 0)
    rx = _pos(50, 0, 0)
    tx_pwr = np.array([20.0])
    jam_pwr = np.array([[30.0] * N_SUB])

    sinr_far = ch.compute_sinr(
        tx_positions=tx, rx_positions=rx,
        jam_positions=_pos(90, 0, 0),   # jammer far from RX
        tx_power_dbm=tx_pwr, jam_power_dbm=jam_pwr,
        noise_power_dbm=NOISE, n_subcarriers=N_SUB,
    )
    sinr_close = ch.compute_sinr(
        tx_positions=tx, rx_positions=rx,
        jam_positions=_pos(51, 0, 0),   # jammer right next to RX
        tx_power_dbm=tx_pwr, jam_power_dbm=jam_pwr,
        noise_power_dbm=NOISE, n_subcarriers=N_SUB,
    )
    assert np.all(sinr_close < sinr_far)


def test_multi_rx_shape():
    ch = MockChannel()
    rx = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float32)
    sinr = ch.compute_sinr(
        tx_positions=_pos(0, 0, 0),
        rx_positions=rx,
        jam_positions=_pos(50, 0, 0),
        tx_power_dbm=np.array([20.0]),
        jam_power_dbm=np.array([[25.0] * N_SUB]),
        noise_power_dbm=NOISE,
        n_subcarriers=N_SUB,
    )
    assert sinr.shape == (3, N_SUB)
