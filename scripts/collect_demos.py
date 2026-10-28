#!/usr/bin/env python3
"""CLI script to collect expert demonstration datasets."""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roboimitate.data.collect import collect_demonstrations


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for RoboImitate.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of demonstration episodes to collect.")
    parser.add_argument("--save-dir", type=str, default="data/demos", help="Directory where dataset will be saved.")
    parser.add_argument("--filename", type=str, default="expert_demonstrations.npz", help="Dataset filename.")
    parser.add_argument("--config", type=str, default="configs/environment.yaml", help="Path to environment config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-h5", action="store_true", help="Disable HDF5 export.")

    args = parser.parse_args()

    print(f"Collecting {args.episodes} demonstrations into {args.save_dir}/{args.filename}...")
    summary = collect_demonstrations(
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        filename=args.filename,
        env_config=args.config,
        seed=args.seed,
        save_h5=not args.no_h5,
    )
    print("\nCollection Summary:")
    print(f"  Total Transitions: {summary['total_transitions']}")
    print(f"  Success Rate: {summary['success_rate']:.1%} ({summary['successful_episodes']}/{summary['num_episodes']})")
    print(f"  Average Episode Length: {summary['avg_episode_length']:.1f} steps")


if __name__ == "__main__":
    main()

