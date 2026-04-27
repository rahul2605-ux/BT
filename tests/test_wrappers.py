import numpy as np
import pytest
from core.config import EnvironmentConfig
from envs.wireless_env import WirelessEnv
from wrappers.sinr_normalize import SINRNormalizeObservation
from wrappers.rescale_action import PowerRescaleAction

N_SUB = 76


@pytest.fixture
def base_env():
    e = WirelessEnv(EnvironmentConfig())
    yield e
    e.close()


# ---------------------------------------------------------------------------
# SINRNormalizeObservation
# ---------------------------------------------------------------------------

class TestSINRNormalize:
    def test_obs_in_unit_range(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB)
        obs, _ = env.reset()
        assert np.all(obs[:N_SUB] >= 0.0)
        assert np.all(obs[:N_SUB] <= 1.0)

    def test_obs_shape_unchanged(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB)
        obs, _ = env.reset()
        assert obs.shape == base_env.observation_space.shape

    def test_non_sinr_part_unchanged(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB)
        obs, _ = env.reset()
        # position + step_frac are already in [0,1] and must not be rescaled
        assert np.all(obs[N_SUB:] >= 0.0)
        assert np.all(obs[N_SUB:] <= 1.0 + 1e-6)

    def test_observation_space_updated(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB)
        assert np.all(env.observation_space.low[:N_SUB]  == 0.0)
        assert np.all(env.observation_space.high[:N_SUB] == 1.0)

    def test_clip_floor(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB,
                                       sinr_min_db=100.0, sinr_max_db=200.0)
        # All real SINR values will be below 100 dB → should clip to 0
        obs, _ = env.reset()
        assert np.all(obs[:N_SUB] == 0.0)

    def test_clip_ceiling(self, base_env):
        env = SINRNormalizeObservation(base_env, n_subcarriers=N_SUB,
                                       sinr_min_db=-200.0, sinr_max_db=-100.0)
        # All real SINR values will be above -100 dB → should clip to 1
        obs, _ = env.reset()
        assert np.all(obs[:N_SUB] == 1.0)


# ---------------------------------------------------------------------------
# PowerRescaleAction
# ---------------------------------------------------------------------------

class TestPowerRescaleAction:
    def test_action_space_is_minus1_to_1(self, base_env):
        env = PowerRescaleAction(base_env)
        assert np.all(env.action_space.low  == -1.0)
        assert np.all(env.action_space.high ==  1.0)

    def test_action_maps_minus1_to_low(self, base_env):
        env = PowerRescaleAction(base_env)
        mapped = env.action(np.full(N_SUB, -1.0, dtype=np.float32))
        np.testing.assert_allclose(mapped, base_env.action_space.low, atol=1e-6)

    def test_action_maps_plus1_to_high(self, base_env):
        env = PowerRescaleAction(base_env)
        mapped = env.action(np.full(N_SUB, 1.0, dtype=np.float32))
        np.testing.assert_allclose(mapped, base_env.action_space.high, atol=1e-6)

    def test_action_maps_zero_to_midpoint(self, base_env):
        env = PowerRescaleAction(base_env)
        mid = 0.5 * (base_env.action_space.low + base_env.action_space.high)
        mapped = env.action(np.zeros(N_SUB, dtype=np.float32))
        np.testing.assert_allclose(mapped, mid, atol=1e-6)

    def test_reverse_action_is_inverse(self, base_env):
        env = PowerRescaleAction(base_env)
        original = env.action_space.sample()
        roundtrip = env.reverse_action(env.action(original))
        np.testing.assert_allclose(roundtrip, original, atol=1e-5)

    def test_step_accepts_rescaled_action(self, base_env):
        env = PowerRescaleAction(base_env)
        env.reset()
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
