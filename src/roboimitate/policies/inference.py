"""Inference wrapper handling observation preprocessing, action unnormalization, and action chunking queues."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class PolicyInferenceWrapper:
    """Standardized inference interface for evaluation and deployment of BC and ACT policies."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "bc",
        norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        device: Union[str, torch.device] = "cpu",
        chunk_size: int = 10,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.model = model
        self.model_type = model_type.lower()
        self.norm_stats = norm_stats or {}
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        if action_bounds is None:
            self.act_low = np.array([-0.04, -0.04, -0.04, -1.0], dtype=np.float32)
            self.act_high = np.array([0.04, 0.04, 0.04, 1.0], dtype=np.float32)
        else:
            self.act_low, self.act_high = action_bounds

        self.model.to(self.device)
        self.model.eval()

        # Action chunk queue for ACT policy execution
        self.action_queue: List[np.ndarray] = []
        self.last_latency_seconds: float = 0.0

    def reset(self) -> None:
        """Reset internal action buffer for a new evaluation episode."""
        self.action_queue = []
        self.last_latency_seconds = 0.0

    def get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the next continuous environment action from observation.

        Args:
            observation: Dictionary with 'image' (84, 84, 3) uint8 and 'proprio' (4,) float32.

        Returns:
            np.ndarray of shape (4,) within action bounds.
        """
        # If executing an ACT policy and actions remain in queue, pop next action
        if self.model_type == "act" and len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        # Preprocess observations
        t_start = time.perf_counter()

        img_raw = observation["image"].astype(np.float32) / 255.0
        # (H, W, C) -> (1, C, H, W)
        img_ch_first = np.transpose(img_raw, (2, 0, 1))
        img_tensor = torch.from_numpy(img_ch_first).unsqueeze(0).to(self.device)

        proprio_raw = observation["proprio"].copy().astype(np.float32)
        if "proprio" in self.norm_stats:
            p_mean = self.norm_stats["proprio"]["mean"]
            p_std = self.norm_stats["proprio"]["std"]
            proprio_raw = (proprio_raw - p_mean) / p_std
        proprio_tensor = torch.from_numpy(proprio_raw).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            if self.model_type == "bc":
                pred = self.model(img_tensor, proprio_tensor) # (1, action_dim)
                pred_np = pred.squeeze(0).cpu().numpy()
            elif self.model_type == "act":
                pred_chunk, _, _ = self.model(img_tensor, proprio_tensor, actions=None) # (1, K, action_dim)
                pred_np = pred_chunk.squeeze(0).cpu().numpy() # (K, action_dim)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        self.last_latency_seconds = time.perf_counter() - t_start

        # Unnormalize actions
        if "actions" in self.norm_stats:
            a_mean = self.norm_stats["actions"]["mean"]
            a_std = self.norm_stats["actions"]["std"]
            if self.model_type == "bc":
                pred_np = pred_np * a_std + a_mean
            else:
                pred_np = pred_np * a_std[None, :] + a_mean[None, :]

        # Clip actions to safety bounds
        if self.model_type == "bc":
            action = np.clip(pred_np, self.act_low, self.act_high)
            return action.astype(np.float32)
        else:
            # ACT: Populate queue with unnormalized chunk
            clipped_chunk = np.clip(pred_np, self.act_low[None, :], self.act_high[None, :]).astype(np.float32)
            self.action_queue = [clipped_chunk[i] for i in range(1, len(clipped_chunk))]
            return clipped_chunk[0]

