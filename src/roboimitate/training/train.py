"""Policy training pipeline for Behavior Cloning (BC) and Action Chunking with Transformers (ACT)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from roboimitate.data.dataset import RoboImitateDataset
from roboimitate.policies.behavior_cloning import BCPolicy
from roboimitate.policies.act_policy import ACTPolicy
from roboimitate.utils.checkpoint import save_checkpoint
from roboimitate.utils.config import load_config, merge_configs
from roboimitate.utils.reproducibility import set_seed
from roboimitate.visualization.plots import plot_loss_curves

logger = logging.getLogger(__name__)


def train_policy(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    model_type: str = "bc",
    dataset_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    device: str = "auto",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Train a Behavior Cloning or ACT policy on demonstrated trajectories.

    Args:
        config: Path to YAML training configuration or config dict.
        model_type: 'bc' or 'act'.
        dataset_path: Path to .npz or .h5 demonstration dataset.
        output_dir: Directory where checkpoints and plots are written.
        epochs: Number of training epochs (overrides config).
        batch_size: Mini-batch size (overrides config).
        lr: Learning rate (overrides config).
        seed: Random seed for deterministic reproducibility.
        device: 'auto', 'cpu', or 'cuda'.
        smoke_test: If True, runs a rapid 2-epoch CPU validation run.

    Returns:
        Dictionary summarizing training results and checkpoint paths.
    """
    # 1. Load and merge configuration
    if isinstance(config, (str, Path)):
        base_cfg = load_config(config)
    elif isinstance(config, dict):
        base_cfg = config
    else:
        cfg_file = Path("configs/training.yaml")
        base_cfg = load_config(cfg_file) if cfg_file.exists() else {}

    model_type = model_type.lower()
    if model_type not in ["bc", "act"]:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'bc' or 'act'.")

    model_cfg = base_cfg.get(model_type, {})
    dataset_cfg = base_cfg.get("dataset", {})
    logging_cfg = base_cfg.get("logging", {})

    # Seed
    exp_seed = seed if seed is not None else base_cfg.get("seed", 42)
    set_seed(exp_seed)

    # Device selection
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    print(f"Training on device: {target_device}")

    # Dataset path resolution
    if dataset_path is None:
        data_dir = Path(dataset_cfg.get("data_dir", "data/demos"))
        filename = dataset_cfg.get("filename", "expert_demonstrations.npz")
        resolved_dataset_path = data_dir / filename
    else:
        resolved_dataset_path = Path(dataset_path)

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Demonstration dataset not found at: {resolved_dataset_path}")

    # Output directories
    out_dir = Path(output_dir) if output_dir is not None else Path(logging_cfg.get("output_dir", "outputs"))
    chk_dir = out_dir / "checkpoints" / model_type
    chk_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots" / model_type
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    train_val_split = float(dataset_cfg.get("train_val_split", 0.8))
    num_epochs = epochs if epochs is not None else (
        model_cfg.get("smoke_test_epochs", 2) if smoke_test else model_cfg.get("epochs", 30)
    )
    b_size = batch_size if batch_size is not None else model_cfg.get("batch_size", 32)
    learning_rate = lr if lr is not None else float(model_cfg.get("learning_rate", 3e-4))
    chunk_size = int(model_cfg.get("chunk_size", 10))

    # 2. Datasets & DataLoaders
    print(f"Loading dataset from {resolved_dataset_path}...")
    train_dataset = RoboImitateDataset(
        dataset_path=resolved_dataset_path,
        split="train",
        train_val_split=train_val_split,
        mode=model_type,
        chunk_size=chunk_size,
    )

    val_dataset = RoboImitateDataset(
        dataset_path=resolved_dataset_path,
        split="val",
        train_val_split=train_val_split,
        mode=model_type,
        chunk_size=chunk_size,
        norm_stats=train_dataset.norm_stats,
    )

    actual_batch_size = min(b_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > actual_batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(actual_batch_size, len(val_dataset)),
        shuffle=False,
    )

    print(f"Dataset loaded: {len(train_dataset)} training steps, {len(val_dataset)} validation steps.")

    # 3. Model Instantiation
    if model_type == "bc":
        model = BCPolicy(
            image_channels=model_cfg.get("image_channels", 3),
            proprio_dim=model_cfg.get("proprio_dim", 4),
            action_dim=model_cfg.get("action_dim", 4),
            hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else: # act
        model = ACTPolicy(
            image_channels=model_cfg.get("image_channels", 3),
            proprio_dim=model_cfg.get("proprio_dim", 4),
            action_dim=model_cfg.get("action_dim", 4),
            chunk_size=chunk_size,
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 2),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            kl_weight=float(model_cfg.get("kl_weight", 10.0)),
            use_cvae=model_cfg.get("use_cvae", True),
        )

    model.to(target_device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 4. Training Loop
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }
    if model_type == "act":
        history["recon_loss"] = []
        history["kl_loss"] = []

    best_val_loss = float("inf")
    patience = logging_cfg.get("early_stopping", {}).get("patience", 10)
    patience_counter = 0

    print(f"\n--- Starting {model_type.upper()} Training ({num_epochs} Epochs) ---")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss_accum = 0.0
        train_recon_accum = 0.0
        train_kl_accum = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            img = batch["image"].to(target_device)
            proprio = batch["proprio"].to(target_device)
            target_act = batch["action"].to(target_device)

            if model_type == "bc":
                pred_act = model(img, proprio)
                loss_dict = model.compute_loss(pred_act, target_act, loss_type=model_cfg.get("loss_type", "l1"))
                loss = loss_dict["loss"]
            else:
                act_mask = batch["action_mask"].to(target_device)
                pred_act, mu, logvar = model(img, proprio, actions=target_act)
                loss_dict = model.compute_loss(
                    pred_act,
                    target_act,
                    action_mask=act_mask,
                    mu=mu,
                    logvar=logvar,
                    loss_type=model_cfg.get("loss_type", "l1"),
                )
                loss = loss_dict["loss"]
                train_recon_accum += float(loss_dict["recon_loss"].item())
                train_kl_accum += float(loss_dict["kl_loss"].item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += float(loss.item())
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, train_batches)
        history["train_loss"].append(avg_train_loss)

        if model_type == "act":
            history["recon_loss"].append(train_recon_accum / max(1, train_batches))
            history["kl_loss"].append(train_kl_accum / max(1, train_batches))

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(target_device)
                proprio = batch["proprio"].to(target_device)
                target_act = batch["action"].to(target_device)

                if model_type == "bc":
                    pred_act = model(img, proprio)
                    val_loss_dict = model.compute_loss(pred_act, target_act, loss_type=model_cfg.get("loss_type", "l1"))
                else:
                    act_mask = batch["action_mask"].to(target_device)
                    # For validation in ACT, evaluate conditional generation on held-out actions
                    pred_act, mu, logvar = model(img, proprio, actions=target_act)
                    val_loss_dict = model.compute_loss(
                        pred_act,
                        target_act,
                        action_mask=act_mask,
                        mu=mu,
                        logvar=logvar,
                        loss_type=model_cfg.get("loss_type", "l1"),
                    )

                val_loss_accum += float(val_loss_dict["loss"].item())
                val_batches += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)
        history["val_loss"].append(avg_val_loss)

        # Print progress
        if epoch % max(1, num_epochs // 10) == 0 or epoch == 1 or epoch == num_epochs:
            act_extra = ""
            if model_type == "act":
                act_extra = f" | Recon: {history['recon_loss'][-1]:.4f} | KL: {history['kl_loss'][-1]:.4f}"
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{act_extra}")

        # Checkpoint state dictionary
        state = {
            "epoch": epoch,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "norm_stats": train_dataset.norm_stats,
            "config": base_cfg,
            "model_cfg": model_cfg,
            "seed": exp_seed,
        }

        # Save last checkpoint
        save_checkpoint(state, chk_dir, filename="checkpoint_last.pt", is_best=False)

        # Check best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(state, chk_dir, filename="checkpoint_best.pt", is_best=True)
        else:
            patience_counter += 1

        # Early stopping
        if logging_cfg.get("early_stopping", {}).get("enabled", True) and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
            break

    # 5. Plot Loss Curves & Save Summary
    loss_plot_path = plots_dir / f"{model_type}_loss_curve.png"
    plot_loss_curves(history, loss_plot_path, title=f"{model_type.upper()} Policy Loss Curves")
    print(f"Saved loss curve plot to {loss_plot_path}")

    # Save training metrics history JSON
    history_path = chk_dir / f"{model_type}_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_chk_path = chk_dir / "checkpoint_best.pt"
    print(f"Training complete. Best checkpoint saved to {best_chk_path} (Val Loss: {best_val_loss:.4f})")

    return {
        "model_type": model_type,
        "epochs_trained": epoch,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "checkpoint_path": str(best_chk_path),
        "loss_plot_path": str(loss_plot_path),
        "history_path": str(history_path),
    }

