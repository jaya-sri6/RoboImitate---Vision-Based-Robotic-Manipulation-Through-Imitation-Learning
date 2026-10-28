"""PyTorch Dataset implementation for Behavior Cloning and ACT policies."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


def compute_norm_stats(data: np.ndarray, eps: float = 1e-5) -> Dict[str, np.ndarray]:
    """Compute mean and standard deviation for normalization.

    Args:
        data: Array of shape (N, D).
        eps: Small epsilon to prevent division by zero for zero-variance dimensions.

    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (D,).
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std < eps, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class RoboImitateDataset(Dataset):
    """PyTorch Dataset supporting both single-step BC and chunked ACT policies."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        split: str = "train",
        train_val_split: float = 0.8,
        mode: str = "bc",
        chunk_size: int = 10,
        norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        normalize_proprio: bool = True,
        normalize_actions: bool = True,
    ):
        """Initialize the RoboImitate dataset.

        Args:
            dataset_path: Path to .npz or .h5 demonstration dataset.
            split: 'train' or 'val'.
            train_val_split: Fraction of episodes allocated to training split.
            mode: 'bc' (single action) or 'act' (action chunking).
            chunk_size: Action horizon length K for ACT mode.
            norm_stats: Optional pre-computed normalization statistics.
            normalize_proprio: Whether to z-score normalize proprioceptive state.
            normalize_actions: Whether to z-score normalize action outputs.
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.split = split.lower()
        self.train_val_split = train_val_split
        self.mode = mode.lower()
        self.chunk_size = chunk_size
        self.normalize_proprio = normalize_proprio
        self.normalize_actions = normalize_actions

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # Load dataset arrays
        if self.dataset_path.suffix == ".npz":
            with np.load(self.dataset_path) as data:
                raw_images = data["images"]
                raw_proprio = data["proprio"]
                raw_actions = data["actions"]
                raw_episode_ends = data["episode_ends"]
        elif self.dataset_path.suffix in [".h5", ".hdf5"]:
            with h5py.File(self.dataset_path, "r") as hf:
                raw_images = hf["images"][:]
                raw_proprio = hf["proprio"][:]
                raw_actions = hf["actions"][:]
                raw_episode_ends = hf["episode_ends"][:]
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_path.suffix}")

        num_episodes = len(raw_episode_ends)
        num_train_episodes = max(1, int(num_episodes * self.train_val_split))

        # Episode-level train/validation split to prevent temporal leakage
        episode_starts = np.concatenate([[0], raw_episode_ends[:-1]])
        if self.split == "train":
            selected_episodes = range(0, num_train_episodes)
        elif self.split in ["val", "validation"]:
            selected_episodes = range(num_train_episodes, num_episodes)
            if len(selected_episodes) == 0:
                # Fallback for very small datasets during smoke tests
                selected_episodes = range(0, num_episodes)
        else:
            selected_episodes = range(0, num_episodes)

        # Slice data for selected episodes
        valid_indices: List[int] = []
        self.episode_boundaries: List[Tuple[int, int]] = []

        for ep_idx in selected_episodes:
            start_idx = episode_starts[ep_idx]
            end_idx = raw_episode_ends[ep_idx]
            self.episode_boundaries.append((start_idx, end_idx))
            valid_indices.extend(range(start_idx, end_idx))

        self.valid_indices = valid_indices
        self.images = raw_images
        self.proprio = raw_proprio
        self.actions = raw_actions
        self.raw_episode_ends = raw_episode_ends

        # Precompute per-timestep episode end lookup for chunk padding
        self.timestep_to_ep_end = np.zeros(len(self.images), dtype=np.int64)
        for s_idx, e_idx in zip(episode_starts, raw_episode_ends):
            self.timestep_to_ep_end[s_idx:e_idx] = e_idx

        # Compute or apply normalization statistics
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            # Compute statistics exclusively from training indices
            train_proprio = self.proprio[valid_indices]
            train_actions = self.actions[valid_indices]
            self.norm_stats = {
                "proprio": compute_norm_stats(train_proprio),
                "actions": compute_norm_stats(train_actions),
            }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        t_idx = self.valid_indices[index]

        # 1. Image observation: (H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]
        img_np = self.images[t_idx].astype(np.float32) / 255.0
        img_ch_first = np.transpose(img_np, (2, 0, 1))
        image_tensor = torch.from_numpy(img_ch_first).float()

        # 2. Proprioception: (4,) float32
        proprio_np = self.proprio[t_idx].copy().astype(np.float32)
        if self.normalize_proprio:
            p_mean = self.norm_stats["proprio"]["mean"]
            p_std = self.norm_stats["proprio"]["std"]
            proprio_np = (proprio_np - p_mean) / p_std
        proprio_tensor = torch.from_numpy(proprio_np).float()

        if self.mode == "bc":
            # Single-step action target: (4,) float32
            act_np = self.actions[t_idx].copy().astype(np.float32)
            if self.normalize_actions:
                a_mean = self.norm_stats["actions"]["mean"]
                a_std = self.norm_stats["actions"]["std"]
                act_np = (act_np - a_mean) / a_std
            action_tensor = torch.from_numpy(act_np).float()

            return {
                "image": image_tensor,
                "proprio": proprio_tensor,
                "action": action_tensor,
            }

        elif self.mode == "act":
            # Action Chunking: Sequence of actions [a_t, ..., a_{t+K-1}]
            # With episode boundary handling and padding
            ep_end = self.timestep_to_ep_end[t_idx]
            available_steps = ep_end - t_idx
            chunk_len = min(self.chunk_size, available_steps)

            action_dim = self.actions.shape[1]
            action_chunk = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
            action_mask = np.zeros((self.chunk_size,), dtype=np.float32)

            raw_chunk = self.actions[t_idx : t_idx + chunk_len].copy().astype(np.float32)
            if self.normalize_actions:
                a_mean = self.norm_stats["actions"]["mean"]
                a_std = self.norm_stats["actions"]["std"]
                raw_chunk = (raw_chunk - a_mean) / a_std

            action_chunk[:chunk_len] = raw_chunk
            # Repeat last valid action for padding to prevent sudden jumps
            if chunk_len < self.chunk_size and chunk_len > 0:
                action_chunk[chunk_len:] = raw_chunk[-1]
            action_mask[:chunk_len] = 1.0

            return {
                "image": image_tensor,
                "proprio": proprio_tensor,
                "action": torch.from_numpy(action_chunk).float(), # (chunk_size, action_dim)
                "action_mask": torch.from_numpy(action_mask).float(), # (chunk_size,)
            }

        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

