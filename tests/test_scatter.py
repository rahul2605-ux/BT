import numpy as np
import pytest
from core.scatter import RandomScatter, SymmetricScatter, CustomScatter, make_scatter

BOUNDS = (100.0, 80.0, 50.0)
RNG = np.random.default_rng(0)


def test_random_shape():
    pos = RandomScatter().place(5, BOUNDS, RNG)
    assert pos.shape == (5, 3)
    assert pos.dtype == np.float32


def test_random_within_bounds():
    pos = RandomScatter().place(20, BOUNDS, RNG)
    x_max, y_max, z_max = BOUNDS
    assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] <= x_max)
    assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] <= y_max)
    assert np.all(pos[:, 2] >= 0) and np.all(pos[:, 2] <= z_max)


def test_symmetric_left_shape():
    pos = SymmetricScatter(side="left").place(4, BOUNDS, RNG)
    assert pos.shape == (4, 3)


def test_symmetric_left_x_range():
    pos = SymmetricScatter(side="left").place(8, BOUNDS, RNG)
    assert np.all(pos[:, 0] < BOUNDS[0] / 2 + 1e-6)


def test_symmetric_right_x_range():
    pos = SymmetricScatter(side="right").place(8, BOUNDS, RNG)
    assert np.all(pos[:, 0] >= BOUNDS[0] / 2 - 1e-6)


def test_symmetric_altitude():
    frac = 0.3
    pos = SymmetricScatter(side="left", altitude_fraction=frac).place(4, BOUNDS, RNG)
    expected_z = BOUNDS[2] * frac
    assert np.allclose(pos[:, 2], expected_z)


def test_custom_returns_positions():
    fixed = np.array([[10, 20, 5], [30, 40, 10]], dtype=np.float32)
    pos = CustomScatter(fixed).place(2, BOUNDS, RNG)
    np.testing.assert_array_equal(pos, fixed)


def test_custom_truncates_to_count():
    fixed = np.ones((5, 3), dtype=np.float32)
    pos = CustomScatter(fixed).place(3, BOUNDS, RNG)
    assert pos.shape == (3, 3)


def test_custom_raises_if_too_few():
    fixed = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        CustomScatter(fixed).place(5, BOUNDS, RNG)


def test_make_scatter_factory():
    for mode in ("random", "symmetric", "custom"):
        kwargs = {"positions": np.ones((4, 3))} if mode == "custom" else {}
        scatter = make_scatter(mode, **kwargs)
        pos = scatter.place(4, BOUNDS, RNG)
        assert pos.shape == (4, 3)


def test_make_scatter_unknown_mode():
    with pytest.raises(ValueError):
        make_scatter("nonexistent")
