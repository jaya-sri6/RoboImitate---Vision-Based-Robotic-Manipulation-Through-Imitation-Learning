"""Automated tests for model checkpoint saving, restoring, and device mapping."""

from pathlib import Path
import numpy as np
import pytest
import torch

from roboimitate.policies.behavior_cloning import BCPolicy
from roboimitate.utils.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load(tmp_path: Path):
    """Test saving a policy checkpoint and restoring state."""
    model = BCPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = {
        "epoch": 5,
        "model_type": "bc",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": 0.045,
        "norm_stats": {
            "proprio": {"mean": np.zeros(4, dtype=np.float32), "std": np.ones(4, dtype=np.float32)},
        },
    }

    chk_path = save_checkpoint(state, checkpoint_dir=tmp_path, filename="model_test.pt", is_best=True)
    assert chk_path.exists()
    assert (tmp_path / "best_model.pt").exists()

    loaded_state = load_checkpoint(chk_path, device="cpu")
    assert loaded_state["epoch"] == 5
    assert loaded_state["model_type"] == "bc"
    assert "model_state_dict" in loaded_state
    assert "norm_stats" in loaded_state


def test_checkpoint_missing_file():
    """Test FileNotFoundError on non-existent checkpoint path."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_checkpoint_file.pt")

