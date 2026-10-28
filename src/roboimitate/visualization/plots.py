"""Plotting and visualization utilities for RoboImitate."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Training and Validation Loss",
) -> Path:
    """Plot and save training and validation loss curves.

    Args:
        history: Dict containing 'train_loss' and 'val_loss' lists.
        output_path: Filepath where the plot will be saved.
        title: Plot title.

    Returns:
        Path to the saved figure.
    """
    save_file = Path(output_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.0)
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.0, linestyle="--")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(save_file, dpi=150)
    plt.close()
    return save_file


def plot_policy_vs_baseline(
    comparison_data: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Plot comparison of success rates between policies and baseline.

    Args:
        comparison_data: Mapping of method name to dict containing 'success_rate' and 'avg_steps'.
        output_path: Save destination.

    Returns:
        Path to saved figure.
    """
    save_file = Path(output_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    names = list(comparison_data.keys())
    success_rates = [comparison_data[n].get("success_rate", 0.0) * 100 for n in names]
    avg_steps = [comparison_data[n].get("avg_steps", 0.0) for n in names]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(names))
    width = 0.35

    color_sr = "#2ca02c"
    rects1 = ax1.bar(x - width / 2, success_rates, width, label="Success Rate (%)", color=color_sr, alpha=0.85)
    ax1.set_ylabel("Success Rate (%)", color=color_sr, fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor=color_sr)

    ax2 = ax1.twinx()
    color_steps = "#1f77b4"
    rects2 = ax2.bar(x + width / 2, avg_steps, width, label="Avg Steps", color=color_steps, alpha=0.85)
    ax2.set_ylabel("Avg Completion Steps", color=color_steps, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_steps)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, fontweight="bold")
    plt.title("Method Comparison: Success Rate vs Efficiency", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_file, dpi=150)
    plt.close()
    return save_file


def plot_generalization_comparison(
    position_results: Dict[str, float],
    output_path: Union[str, Path],
    title: str = "Generalization Across Held-Out Positions",
) -> Path:
    """Plot bar chart comparing success rates across spatial positions.

    Args:
        position_results: Dict mapping position name to success rate (0.0 to 1.0).
        output_path: Save destination.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    save_file = Path(output_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    names = list(position_results.keys())
    rates = [position_results[n] * 100 for n in names]

    plt.figure(figsize=(10, 5))
    colors = ["#2ca02c" if "train" in n.lower() else "#d62728" for n in names]
    bars = plt.bar(names, rates, color=colors, alpha=0.85, width=0.5)

    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{rate:.1f}%", ha="center", fontsize=10)

    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_file, dpi=150)
    plt.close()
    return save_file


def plot_failure_distribution(
    failure_counts: Dict[str, int],
    output_path: Union[str, Path],
) -> Path:
    """Plot failure category breakdown.

    Args:
        failure_counts: Dict mapping category name to count.
        output_path: Destination path.

    Returns:
        Path to saved figure.
    """
    save_file = Path(output_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    categories = [k for k, v in failure_counts.items() if v > 0]
    counts = [failure_counts[k] for k in categories]

    if not categories:
        # No failures to plot
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "100% Success — No Failures Observed", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.savefig(save_file, dpi=150)
        plt.close()
        return save_file

    plt.figure(figsize=(7, 5))
    plt.pie(counts, labels=categories, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Failure Mode Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_file, dpi=150)
    plt.close()
    return save_file

