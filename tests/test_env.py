import numpy as np
import pytest
from core.config import EnvironmentConfig
from envs.wireless_env import WirelessEnv


@pytest.fixture
def env():
    e = WirelessEnv(EnvironmentConfig())
    yield e
    e.close()


def test_reset_obs_shape(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape


def test_reset_obs_dtype(env):
    obs, _ = env.reset()
    assert obs.dtype == np.float32


def test_step_shapes(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_space_bounds(env):
    assert np.all(env.action_space.low  == 0.0)
    assert np.all(env.action_space.high == env.config.jammers.max_power_dbm)


def test_reward_is_negative_mean_sinr(env):
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)
    assert np.isclose(reward, -info["mean_sinr_db"], atol=1e-5)


def test_truncation_at_max_steps(env):
    env.reset()
    action = env.action_space.sample()
    truncated = False
    for _ in range(env.config.max_steps):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert truncated


def test_info_keys(env):
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ("step", "leg_positions", "jam_positions", "mean_sinr_db"):
        assert key in info


def test_reset_clears_step_count(env):
    env.reset()
    for _ in range(5):
        env.step(env.action_space.sample())
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["step"] == 1


def test_seeded_reset_reproducible():
    e1 = WirelessEnv(EnvironmentConfig(seed=7))
    e2 = WirelessEnv(EnvironmentConfig(seed=7))
    obs1, _ = e1.reset()
    obs2, _ = e2.reset()
    np.testing.assert_array_equal(obs1, obs2)
    e1.close(); e2.close()
