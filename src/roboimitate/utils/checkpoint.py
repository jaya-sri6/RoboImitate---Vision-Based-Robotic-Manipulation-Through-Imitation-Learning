"""Model checkpointing utilities for saving and restoring training state."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "checkpoint_best.pt",
    is_best: bool = False,
) -> Path:
    """Save training checkpoint and metadata to disk.

    Args:
        state: Dictionary containing model_state_dict, optimizer_state_dict,
               epoch, loss, norm_stats, config, etc.
        checkpoint_dir: Directory where checkpoints should be stored.
        filename: Checkpoint filename.
        is_best: Whether this checkpoint achieved the best validation loss so far.

    Returns:
        Path to the saved checkpoint.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    target_path = chk_dir / filename

    torch.save(state, target_path)

    if is_best:
        best_path = chk_dir / "best_model.pt"
        torch.save(state, best_path)

    return target_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load checkpoint dictionary from disk with device mapping.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target torch device ('cpu', 'cuda', etc.). If None, maps to CPU.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint file is corrupted or cannot be parsed.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    return checkpoint
