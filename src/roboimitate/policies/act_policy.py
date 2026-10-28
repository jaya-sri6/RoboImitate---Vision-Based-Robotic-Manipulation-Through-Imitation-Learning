"""Action Chunking with Transformers (ACT) policy implementation with CVAE style latent prior.

Reference:
    Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (RSS 2023).

Architecture Overview:
    - Vision Backbone (NatureCNN): Encodes (B, 3, 84, 84) to vision tokens.
    - Proprioception Encoder: Linear projection to token embedding dimension.
    - CVAE Encoder (Training): Encodes future action chunk (B, K, A) + conditioning state into
      latent style distribution z ~ N(mu, sigma).
    - Transformer Decoder: Synthesizes sequence of K actions using learned position queries,
      cross-attending to memory tokens [latent_z, vision_token, proprio_token].
    - Loss: L1 reconstruction loss on valid action mask + beta * KL divergence.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from roboimitate.policies.behavior_cloning import NatureCNN


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding for transformer sequence queries."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to sequence tensor of shape (B, T, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class ACTPolicy(nn.Module):
    """Action Chunking with Transformers (ACT) with Conditional VAE latent style modeling."""

    def __init__(
        self,
        image_channels: int = 3,
        proprio_dim: int = 4,
        action_dim: int = 4,
        chunk_size: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        use_cvae: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.use_cvae = use_cvae

        # 1. Observation Encoders
        self.vision_encoder = NatureCNN(in_channels=image_channels, feature_dim=d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        # 2. CVAE Style Latent Encoder (used during training to model multimodality)
        if self.use_cvae:
            self.action_proj = nn.Linear(action_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.latent_proj_mu = nn.Linear(d_model, latent_dim)
            self.latent_proj_logvar = nn.Linear(d_model, latent_dim)
            self.latent_to_memory = nn.Linear(latent_dim, d_model)
        else:
            self.cvae_encoder = None

        # 3. Transformer Decoder
        self.pos_emb = SinusoidalPositionEmbedding(d_model=d_model, max_len=chunk_size + 10)
        self.query_embed = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 4. Action Prediction Head
        self.action_head = nn.Linear(d_model, action_dim)

    def _encode_cvae(
        self,
        action_seq: torch.Tensor,
        state_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode action chunk and state into latent style distribution z ~ N(mu, sigma).

        Args:
            action_seq: Target action chunk (B, chunk_size, action_dim).
            state_tokens: Observation tokens (B, num_state_tokens, d_model).

        Returns:
            Tuple of (sampled_z, mu, logvar).
        """
        B = action_seq.size(0)
        act_tokens = self.action_proj(action_seq) # (B, chunk_size, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)

        # Concatenate [CLS, State, Actions]
        seq = torch.cat([cls_tokens, state_tokens, act_tokens], dim=1)
        seq = self.pos_emb(seq)
        encoded = self.cvae_encoder(seq)

        # Use CLS token output for latent prediction
        cls_out = encoded[:, 0, :]
        mu = self.latent_proj_mu(cls_out)
        logvar = self.latent_proj_logvar(cls_out)
        logvar = torch.clamp(logvar, -10.0, 10.0)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass predicting action chunk (B, chunk_size, action_dim).

        Args:
            image: Image tensor (B, C, H, W).
            proprio: Proprioception tensor (B, proprio_dim).
            actions: Optional target actions for CVAE encoding during training (B, chunk_size, action_dim).

        Returns:
            Tuple of (pred_actions, mu, logvar).
        """
        B = image.size(0)

        # Encode observations
        v_token = self.vision_encoder(image).unsqueeze(1) # (B, 1, d_model)
        p_token = self.proprio_proj(proprio).unsqueeze(1) # (B, 1, d_model)
        obs_memory = torch.cat([v_token, p_token], dim=1) # (B, 2, d_model)

        # Style latent prior handling
        mu, logvar = None, None
        if self.use_cvae:
            if actions is not None:
                # Training mode with target action conditioning
                z, mu, logvar = self._encode_cvae(actions, obs_memory)
            else:
                # Inference mode: zero latent or sampled standard normal
                z = torch.zeros(B, self.latent_dim, device=image.device)
            z_token = self.latent_to_memory(z).unsqueeze(1) # (B, 1, d_model)
            memory = torch.cat([z_token, obs_memory], dim=1) # (B, 3, d_model)
        else:
            memory = obs_memory

        # Sequence queries for action horizon
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) # (B, chunk_size, d_model)
        queries = self.pos_emb(queries)

        # Transformer cross-attention decoding
        dec_out = self.decoder(tgt=queries, memory=memory) # (B, chunk_size, d_model)
        pred_actions = self.action_head(dec_out) # (B, chunk_size, action_dim)

        return pred_actions, mu, logvar

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        loss_type: str = "l1",
    ) -> Dict[str, torch.Tensor]:
        """Compute ACT training loss (Reconstruction loss + KL divergence).

        Args:
            pred_actions: (B, chunk_size, action_dim)
            target_actions: (B, chunk_size, action_dim)
            action_mask: (B, chunk_size) binary mask of valid steps before episode termination.
            mu: CVAE mean (B, latent_dim).
            logvar: CVAE log-variance (B, latent_dim).
            loss_type: 'l1' or 'mse'.

        Returns:
            Dictionary with 'loss', 'recon_loss', and 'kl_loss'.
        """
        if loss_type.lower() == "mse":
            error = (pred_actions - target_actions) ** 2
        else:
            error = torch.abs(pred_actions - target_actions)

        if action_mask is not None:
            mask = action_mask.unsqueeze(-1) # (B, chunk_size, 1)
            recon_loss = (error * mask).sum() / (mask.sum() * self.action_dim + 1e-6)
        else:
            recon_loss = error.mean()

        total_loss = recon_loss
        kl_loss = torch.tensor(0.0, device=pred_actions.device)

        if self.use_cvae and mu is not None and logvar is not None:
            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_div = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl_div.mean()
            total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

