"""Dataset validation module for RoboImitate demonstration datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import h5py


def validate_dataset(
    dataset_path: Union[str, Path],
    action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    expected_image_shape: Tuple[int, int, int] = (84, 84, 3),
    expected_proprio_dim: int = 4,
    expected_action_dim: int = 4,
    strict: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate demonstration dataset integrity, shapes, and value ranges.

    Args:
        dataset_path: Path to .npz or .h5 file.
        action_bounds: Optional tuple of (low, high) bounds for action clipping validation.
        expected_image_shape: Expected shape of single camera frame (H, W, C).
        expected_proprio_dim: Expected dimensionality of proprioceptive state.
        expected_action_dim: Expected dimensionality of continuous action.
        strict: If True, raises ValueError upon failure.

    Returns:
        Tuple of (is_valid: bool, summary_report: dict).
    """
    path = Path(dataset_path)
    issues: List[str] = []
    warnings: List[str] = []

    if not path.is_file():
        msg = f"Dataset file does not exist at: {path}"
        if strict:
            raise FileNotFoundError(msg)
        return False, {"valid": False, "issues": [msg]}

    # Default action bounds for RoboImitatePickAndPlace: delta xyz [-0.04, 0.04], gripper [-1.0, 1.0]
    if action_bounds is None:
        act_low = np.array([-0.0401, -0.0401, -0.0401, -1.001], dtype=np.float32)
        act_high = np.array([0.0401, 0.0401, 0.0401, 1.001], dtype=np.float32)
    else:
        act_low, act_high = action_bounds

    # Load arrays
    try:
        if path.suffix == ".npz":
            with np.load(path) as data:
                required_keys = ["images", "proprio", "actions", "episode_ends"]
                missing = [k for k in required_keys if k not in data]
                if missing:
                    issues.append(f"Missing required fields: {missing}")
                    if strict:
                        raise ValueError("; ".join(issues))
                    return False, {"valid": False, "issues": issues}

                images = data["images"]
                proprio = data["proprio"]
                actions = data["actions"]
                episode_ends = data["episode_ends"]
                episode_lengths = data["episode_lengths"] if "episode_lengths" in data else None
                episode_successes = data["episode_successes"] if "episode_successes" in data else None
        elif path.suffix in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as hf:
                required_keys = ["images", "proprio", "actions", "episode_ends"]
                missing = [k for k in required_keys if k not in hf]
                if missing:
                    issues.append(f"Missing required HDF5 fields: {missing}")
                    if strict:
                        raise ValueError("; ".join(issues))
                    return False, {"valid": False, "issues": issues}

                images = hf["images"][:]
                proprio = hf["proprio"][:]
                actions = hf["actions"][:]
                episode_ends = hf["episode_ends"][:]
                episode_lengths = hf["episode_lengths"][:] if "episode_lengths" in hf else None
                episode_successes = hf["episode_successes"][:] if "episode_successes" in hf else None
        else:
            issues.append(f"Unsupported dataset extension: {path.suffix}")
            if strict:
                raise ValueError("; ".join(issues))
            return False, {"valid": False, "issues": issues}
    except Exception as e:
        issues.append(f"Corrupted dataset format or read failure: {e}")
        if strict:
            raise ValueError("; ".join(issues))
        return False, {"valid": False, "issues": issues}

    # 1. Total transitions consistency
    num_images = images.shape[0]
    num_proprio = proprio.shape[0]
    num_actions = actions.shape[0]

    if not (num_images == num_proprio == num_actions):
        issues.append(
            f"Mismatched transition counts: images ({num_images}), proprio ({num_proprio}), actions ({num_actions})"
        )

    if num_images == 0:
        issues.append("Dataset is empty (0 transitions).")

    # 2. Shape validation
    if images.shape[1:] != expected_image_shape:
        issues.append(f"Unexpected image shape: {images.shape[1:]}, expected {expected_image_shape}")

    if proprio.shape[1] != expected_proprio_dim:
        issues.append(f"Unexpected proprio dimension: {proprio.shape[1]}, expected {expected_proprio_dim}")

    if actions.shape[1] != expected_action_dim:
        issues.append(f"Unexpected action dimension: {actions.shape[1]}, expected {expected_action_dim}")

    # 3. Finite value checks (no NaN, no Inf)
    if not np.all(np.isfinite(proprio)):
        issues.append("Proprioception array contains NaN or Inf values.")

    if not np.all(np.isfinite(actions)):
        issues.append("Action array contains NaN or Inf values.")

    # 4. Action range validation
    min_act = np.min(actions, axis=0)
    max_act = np.max(actions, axis=0)

    if np.any(min_act < act_low) or np.any(max_act > act_high):
        issues.append(
            f"Actions exceed environment bounds. Min observed: {min_act.tolist()}, Max observed: {max_act.tolist()}"
        )

    # 5. Episode boundary validation
    if len(episode_ends) == 0:
        issues.append("Episode ends array is empty.")
    else:
        if not np.all(np.diff(episode_ends) > 0):
            issues.append("Episode ends must be strictly monotonically increasing.")
        if episode_ends[-1] != num_actions:
            issues.append(f"Final episode end ({episode_ends[-1]}) does not equal total actions ({num_actions}).")

    if episode_lengths is not None:
        computed_lengths = np.diff(np.concatenate([[0], episode_ends]))
        if not np.array_equal(episode_lengths, computed_lengths):
            issues.append("Stored episode lengths do not match differences of episode_ends.")

    # Success rate statistics
    success_rate = None
    if episode_successes is not None and len(episode_successes) > 0:
        success_rate = float(np.mean(episode_successes))
        if success_rate < 0.5:
            warnings.append(f"Low expert demonstration success rate: {success_rate:.1%}")

    is_valid = len(issues) == 0

    report: Dict[str, Any] = {
        "valid": is_valid,
        "dataset_path": str(path),
        "total_transitions": int(num_actions),
        "num_episodes": len(episode_ends),
        "avg_episode_len": float(num_actions / max(1, len(episode_ends))),
        "image_shape": list(images.shape),
        "proprio_shape": list(proprio.shape),
        "action_shape": list(actions.shape),
        "action_min": min_act.tolist() if len(actions) > 0 else [],
        "action_max": max_act.tolist() if len(actions) > 0 else [],
        "success_rate": success_rate,
        "issues": issues,
        "warnings": warnings,
    }

    if strict and not is_valid:
        raise ValueError(f"Dataset validation failed with {len(issues)} issues: {'; '.join(issues)}")

    return is_valid, report

