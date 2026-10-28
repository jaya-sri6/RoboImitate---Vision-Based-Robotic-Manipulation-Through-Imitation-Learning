"""Automated tests for dataset loading, normalization, and validation."""

from pathlib import Path
import numpy as np
import pytest
import torch

from roboimitate.data.dataset import RoboImitateDataset, compute_norm_stats
from roboimitate.data.validate import validate_dataset


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a temporary valid synthetic demonstration dataset."""
    num_episodes = 5
    ep_len = 10
    total_steps = num_episodes * ep_len

    images = np.random.randint(0, 256, size=(total_steps, 84, 84, 3), dtype=np.uint8)
    proprio = np.random.uniform(0.1, 0.9, size=(total_steps, 4)).astype(np.float32)
    actions = np.random.uniform(-0.03, 0.03, size=(total_steps, 4)).astype(np.float32)
    actions[:, 3] = np.random.choice([-1.0, 1.0], size=total_steps) # gripper
    rewards = np.zeros(total_steps, dtype=np.float32)
    dones = np.zeros(total_steps, dtype=bool)
    episode_ends = np.arange(ep_len, total_steps + 1, ep_len, dtype=np.int64)
    episode_lengths = np.full(num_episodes, ep_len, dtype=np.int64)
    episode_successes = np.ones(num_episodes, dtype=bool)

    npz_path = tmp_path / "test_demos.npz"
    np.savez_compressed(
        npz_path,
        images=images,
        proprio=proprio,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_ends=episode_ends,
        episode_lengths=episode_lengths,
        episode_successes=episode_successes,
    )
    return npz_path


def test_dataset_bc_loading(synthetic_dataset: Path):
    """Test dataset loading in Behavior Cloning mode."""
    dataset = RoboImitateDataset(
        dataset_path=synthetic_dataset,
        split="train",
        train_val_split=0.8,
        mode="bc",
    )
    assert len(dataset) > 0
    sample = dataset[0]

    assert "image" in sample
    assert "proprio" in sample
    assert "action" in sample
    assert sample["image"].shape == (3, 84, 84)
    assert sample["proprio"].shape == (4,)
    assert sample["action"].shape == (4,)
    assert isinstance(sample["image"], torch.Tensor)


def test_dataset_act_chunking(synthetic_dataset: Path):
    """Test dataset loading in ACT mode with action chunking and masks."""
    chunk_size = 6
    dataset = RoboImitateDataset(
        dataset_path=synthetic_dataset,
        split="train",
        mode="act",
        chunk_size=chunk_size,
    )
    sample = dataset[0]

    assert sample["action"].shape == (chunk_size, 4)
    assert sample["action_mask"].shape == (chunk_size,)
    assert sample["action_mask"].sum() > 0


def test_compute_norm_stats():
    """Test normalization statistics calculation."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    stats = compute_norm_stats(data)

    np.testing.assert_allclose(stats["mean"], [3.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(stats["std"], [1.632993, 1.632993], atol=1e-5)


def test_dataset_validation_valid(synthetic_dataset: Path):
    """Test validation passes on properly formatted dataset."""
    is_valid, report = validate_dataset(synthetic_dataset)
    assert is_valid is True
    assert len(report["issues"]) == 0
    assert report["total_transitions"] == 50


def test_dataset_validation_corrupted(tmp_path: Path):
    """Test validation correctly flags NaN and corrupted fields."""
    corrupted_path = tmp_path / "corrupted.npz"
    # Action array containing NaNs
    images = np.zeros((10, 84, 84, 3), dtype=np.uint8)
    proprio = np.zeros((10, 4), dtype=np.float32)
    actions = np.full((10, 4), np.nan, dtype=np.float32)
    episode_ends = np.array([10], dtype=np.int64)

    np.savez(corrupted_path, images=images, proprio=proprio, actions=actions, episode_ends=episode_ends)

    is_valid, report = validate_dataset(corrupted_path)
    assert is_valid is False
    assert any("NaN" in iss for iss in report["issues"])

