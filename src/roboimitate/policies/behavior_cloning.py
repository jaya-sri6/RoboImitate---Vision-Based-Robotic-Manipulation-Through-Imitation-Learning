"""Supervised Behavior Cloning (BC) policy with visual and proprioceptive encoders."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """Standard Nature CNN vision backbone for robotic image observations (84x84)."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten size with dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            out = self.conv(dummy)
            flatten_dim = out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting visual features.

        Args:
            x: Image tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        features = self.conv(x)
        return self.fc(features)


class BCPolicy(nn.Module):
    """Behavior Cloning policy mapping RGB observations and proprioception to 4D actions."""

    def __init__(
        self,
        image_channels: int = 3,
        proprio_dim: int = 4,
        action_dim: int = 4,
        vision_feature_dim: int = 256,
        proprio_feature_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.vision_encoder = NatureCNN(in_channels=image_channels, feature_dim=vision_feature_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, proprio_feature_dim),
            nn.ReLU(),
        )

        in_dim = vision_feature_dim + proprio_feature_dim
        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h

        layers.append(nn.Linear(curr_dim, action_dim))
        self.action_head = nn.Sequential(*layers)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Predict continuous action from image and proprioception.

        Args:
            image: Image tensor of shape (B, C, H, W).
            proprio: Proprioception tensor of shape (B, proprio_dim).

        Returns:
            Predicted action tensor of shape (B, action_dim).
        """
        v_feat = self.vision_encoder(image)
        p_feat = self.proprio_encoder(proprio)
        fused = torch.cat([v_feat, p_feat], dim=-1)
        pred_action = self.action_head(fused)
        return pred_action

    def compute_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        loss_type: str = "l1",
    ) -> Dict[str, torch.Tensor]:
        """Compute supervised action-prediction loss.

        Args:
            pred_action: Predicted actions of shape (B, action_dim).
            target_action: Expert actions of shape (B, action_dim).
            loss_type: 'l1' (MAE) or 'mse'.

        Returns:
            Dictionary with 'loss' scalar.
        """
        if loss_type.lower() == "mse":
            loss = F.mse_loss(pred_action, target_action)
        else:
            loss = F.l1_loss(pred_action, target_action)

        return {"loss": loss}

