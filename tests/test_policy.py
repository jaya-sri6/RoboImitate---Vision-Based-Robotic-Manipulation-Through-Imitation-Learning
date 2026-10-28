"""Automated tests for BC and ACT policy forward passes, losses, and inference."""

import numpy as np
import pytest
import torch

from roboimitate.policies.behavior_cloning import BCPolicy
from roboimitate.policies.act_policy import ACTPolicy
from roboimitate.policies.inference import PolicyInferenceWrapper


def test_bc_policy_forward_and_loss():
    """Test Behavior Cloning policy forward pass, output shape, and loss backward."""
    B = 4
    policy = BCPolicy(image_channels=3, proprio_dim=4, action_dim=4)

    dummy_img = torch.randn(B, 3, 84, 84)
    dummy_proprio = torch.randn(B, 4)
    dummy_target = torch.randn(B, 4)

    pred_act = policy(dummy_img, dummy_proprio)
    assert pred_act.shape == (B, 4)

    loss_dict = policy.compute_loss(pred_act, dummy_target, loss_type="l1")
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() >= 0.0

    loss_dict["loss"].backward()
    # Check gradients computed
    for param in policy.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_act_policy_forward_and_cvae():
    """Test Action Chunking with Transformers forward pass, sequence shape, and KL loss."""
    B = 2
    chunk_size = 8
    policy = ACTPolicy(
        image_channels=3,
        proprio_dim=4,
        action_dim=4,
        chunk_size=chunk_size,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        use_cvae=True,
    )

    dummy_img = torch.randn(B, 3, 84, 84)
    dummy_proprio = torch.randn(B, 4)
    dummy_target_actions = torch.randn(B, chunk_size, 4)
    dummy_mask = torch.ones(B, chunk_size)

    # Training forward pass (conditioned on target actions)
    pred_actions, mu, logvar = policy(dummy_img, dummy_proprio, actions=dummy_target_actions)
    assert pred_actions.shape == (B, chunk_size, 4)
    assert mu is not None and logvar is not None

    loss_dict = policy.compute_loss(pred_actions, dummy_target_actions, action_mask=dummy_mask, mu=mu, logvar=logvar)
    assert "loss" in loss_dict
    assert "recon_loss" in loss_dict
    assert "kl_loss" in loss_dict

    loss_dict["loss"].backward()

    # Inference forward pass (without target actions)
    policy.eval()
    with torch.no_grad():
        inf_preds, _, _ = policy(dummy_img, dummy_proprio, actions=None)
        assert inf_preds.shape == (B, chunk_size, 4)


def test_policy_inference_wrapper():
    """Test PolicyInferenceWrapper action queue popping and safety clipping."""
    policy = BCPolicy(image_channels=3, proprio_dim=4, action_dim=4)
    wrapper = PolicyInferenceWrapper(model=policy, model_type="bc")

    obs = {
        "image": np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8),
        "proprio": np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float32),
    }

    action = wrapper.get_action(obs)
    assert action.shape == (4,)
    assert np.all(action >= np.array([-0.04, -0.04, -0.04, -1.0]))
    assert np.all(action <= np.array([0.04, 0.04, 0.04, 1.0]))
    assert wrapper.last_latency_seconds > 0.0

