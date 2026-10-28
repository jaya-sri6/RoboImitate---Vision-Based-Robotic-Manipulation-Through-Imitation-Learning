#!/usr/bin/env python3
"""CLI script to train Behavior Cloning or ACT policies."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roboimitate.training.train import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoboImitate policies (BC or ACT).")
    parser.add_argument("--model", type=str, default="bc", choices=["bc", "act"], help="Policy architecture.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to demonstration dataset.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for checkpoints.")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 2-epoch CPU smoke test.")

    args = parser.parse_args()

    results = train_policy(
        config=args.config,
        model_type=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        smoke_test=args.smoke_test,
    )

    print("\nTraining Run Finished:")
    print(f"  Model: {results['model_type'].upper()}")
    print(f"  Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"  Checkpoint: {results['checkpoint_path']}")
    print(f"  Loss Plot: {results['loss_plot_path']}")


if __name__ == "__main__":
    main()

