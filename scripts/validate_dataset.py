#!/usr/bin/env python3
"""CLI script to validate demonstration dataset integrity."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roboimitate.data.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RoboImitate demonstration dataset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/demos/expert_demonstrations.npz",
        help="Path to .npz or .h5 dataset.",
    )
    parser.add_argument("--strict", action="store_true", help="Raise error on validation failure.")

    args = parser.parse_args()

    print(f"Validating dataset at: {args.dataset_path}")
    is_valid, report = validate_dataset(args.dataset_path, strict=args.strict)

    print("\n" + "=" * 60)
    print(" RoboImitate: Dataset Integrity Validation Report")
    print("=" * 60)
    print(f" Status: {'VALID [OK]' if is_valid else 'INVALID [FAIL]'}")
    print(f" Total Transitions: {report.get('total_transitions', 0)}")
    print(f" Total Episodes: {report.get('num_episodes', 0)}")
    print(f" Average Length: {report.get('avg_episode_len', 0.0):.1f} steps")
    print(f" Image Shape: {report.get('image_shape', [])}")
    print(f" Proprio Shape: {report.get('proprio_shape', [])}")
    print(f" Action Shape: {report.get('action_shape', [])}")
    if report.get("success_rate") is not None:
        print(f" Demonstration Success Rate: {report['success_rate']:.1%}")

    if report.get("issues"):
        print("\n [ISSUES DETECTED]:")
        for iss in report["issues"]:
            print(f"  - {iss}")

    if report.get("warnings"):
        print("\n [WARNINGS]:")
        for w in report["warnings"]:
            print(f"  - {w}")

    print("=" * 60)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

