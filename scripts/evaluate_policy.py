#!/usr/bin/env python3
"""CLI script to evaluate policies and baselines across standard and held-out configurations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from roboimitate.evaluation.evaluate import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboImitate policies or scripted baseline.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file.")
    parser.add_argument("--model", type=str, default=None, choices=["bc", "act", "baseline"], help="Model type.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of in-distribution episodes.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for evaluation results.")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Evaluation config path.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    parser.add_argument("--render-video", action="store_true", help="Record MP4 videos of evaluation rollouts.")
    parser.add_argument("--no-held-out", action="store_true", help="Skip 5 held-out generalization positions.")

    args = parser.parse_args()

    controller_type = "baseline" if args.model == "baseline" else "policy"

    if controller_type == "policy" and args.checkpoint is None:
        # Default checkpoint resolution if available
        default_chk = Path("outputs/checkpoints") / (args.model or "bc") / "checkpoint_best.pt"
        if default_chk.exists():
            args.checkpoint = str(default_chk)
        else:
            parser.error("--checkpoint is required when evaluating a policy.")

    results = evaluate_policy(
        checkpoint_path=args.checkpoint,
        controller_type=controller_type,
        model_type=args.model if args.model != "baseline" else None,
        num_episodes=args.episodes,
        eval_config=args.config,
        output_dir=args.output_dir,
        device=args.device,
        render_video=args.render_video,
        test_held_out=not args.no_held_out,
    )

    print("\nSummary Results:")
    print(f"  Overall Success Rate: {results['success_rate']:.1%}")
    print(f"  Inference Latency Mean: {results['latency']['inference_mean_ms']:.2f} ms")
    print(f"  Report: {results['json_path']}")


if __name__ == "__main__":
    main()

