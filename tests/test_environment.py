"""Automated tests for RoboImitate Gymnasium environment."""

import numpy as np
import pytest
from roboimitate.env.manipulation_env import RoboImitatePickAndPlaceEnv
from roboimitate.env.wrappers import ImageObservationWrapper


def test_environment_initialization():
    """Test environment creation, observation keys, shapes, and action bounds."""
    env = RoboImitatePickAndPlaceEnv()
    assert env.action_space.shape == (4,)
    assert "image" in env.observation_space.spaces
    assert "proprio" in env.observation_space.spaces
    assert env.observation_space["image"].shape == (84, 84, 3)
    assert env.observation_space["proprio"].shape == (4,)
    env.close()


def test_environment_reset_and_determinism():
    """Test deterministic reset behavior with random seeds."""
    env = RoboImitatePickAndPlaceEnv()
    obs1, info1 = env.reset(seed=42)
    p1 = env.get_privileged_state()["object_pos"]

    obs2, info2 = env.reset(seed=42)
    p2 = env.get_privileged_state()["object_pos"]

    np.testing.assert_allclose(p1, p2, atol=1e-5)
    np.testing.assert_allclose(obs1["proprio"], obs2["proprio"], atol=1e-5)
    env.close()


def test_environment_stepping_and_clipping():
    """Test stepping with out-of-bounds actions to verify safety clipping."""
    env = RoboImitatePickAndPlaceEnv()
    env.reset(seed=100)

    # Action exceeding bounds
    extreme_action = np.array([10.0, -10.0, 5.0, 2.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(extreme_action)

    assert next_obs["image"].shape == (84, 84, 3)
    assert next_obs["proprio"].shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in info
    env.close()


def test_image_observation_wrapper():
    """Test ImageObservationWrapper channel transposition and normalization."""
    env = RoboImitatePickAndPlaceEnv()
    wrapped_env = ImageObservationWrapper(env, normalize=True, channel_first=True)

    obs, _ = wrapped_env.reset(seed=42)
    assert obs["image"].shape == (3, 84, 84)
    assert obs["image"].dtype == np.float32
    assert obs["image"].min() >= 0.0 and obs["image"].max() <= 1.0

    act = np.zeros(4, dtype=np.float32)
    next_obs, _, _, _, _ = wrapped_env.step(act)
    assert next_obs["image"].shape == (3, 84, 84)
    wrapped_env.close()

