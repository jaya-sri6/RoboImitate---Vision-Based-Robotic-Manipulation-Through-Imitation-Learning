#!/usr/bin/env python3
"""End-to-end pipeline runner for RoboImitate: Setup -> Demos -> Validation -> Train -> Eval -> Compare."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roboimitate.data.collect import collect_demonstrations
from roboimitate.data.validate import validate_dataset
from roboimitate.evaluation.evaluate import evaluate_policy
from roboimitate.training.train import train_policy
from roboimitate.visualization.plots import plot_policy_vs_baseline
from setup_check import run_setup_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete RoboImitate end-to-end pipeline.")
    parser.add_argument("--episodes", type=int, default=15, help="Number of demo episodes to collect.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per policy.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Outputs directory.")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--skip-act", action="store_true", help="Skip ACT training and only train BC.")
    parser.add_argument("--render-video", action="store_true", help="Render MP4 videos during evaluation.")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" RoboImitate: Complete End-to-End Pipeline Execution")
    print("=" * 70)

    # 1. Setup Check
    print("\n>>> STEP 1/6: Verifying Environment & Dependencies...")
    if not run_setup_check():
        print("[FAIL] Setup check failed. Aborting pipeline.")
        sys.exit(1)

    # 2. Demonstration Collection
    print(f"\n>>> STEP 2/6: Collecting {args.episodes} Expert Demonstrations...")
    data_dir = Path("data/demos")
    demo_file = "expert_demonstrations.npz"
    collect_summary = collect_demonstrations(
        num_episodes=args.episodes,
        save_dir=data_dir,
        filename=demo_file,
        seed=42,
    )
    dataset_path = data_dir / demo_file

    # 3. Dataset Validation
    print("\n>>> STEP 3/6: Validating Demonstration Dataset Integrity...")
    is_valid, val_report = validate_dataset(dataset_path, strict=True)
    print(f"  Dataset Valid: {is_valid} ({val_report['total_transitions']} transitions)")

    # 4. Train Policies (BC and ACT)
    print(f"\n>>> STEP 4/6: Training Policies (Epochs: {args.epochs})...")
    print("\n  [4A] Training Behavior Cloning (BC) Baseline...")
    bc_train_results = train_policy(
        model_type="bc",
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )

    act_train_results = None
    if not args.skip_act:
        print("\n  [4B] Training Action Chunking with Transformers (ACT)...")
        act_train_results = train_policy(
            model_type="act",
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            device=args.device,
        )

    # 5. Evaluate Methods
    print("\n>>> STEP 5/6: Evaluating Scripted Baseline and Trained Policies...")
    results_dir = Path(args.results_dir)

    print("\n  [5A] Evaluating Scripted Baseline...")
    baseline_eval = evaluate_policy(
        controller_type="baseline",
        num_episodes=args.eval_episodes,
        output_dir=results_dir,
        render_video=args.render_video,
    )

    print("\n  [5B] Evaluating Trained BC Policy...")
    bc_eval = evaluate_policy(
        checkpoint_path=bc_train_results["checkpoint_path"],
        controller_type="policy",
        model_type="bc",
        num_episodes=args.eval_episodes,
        output_dir=results_dir,
        render_video=args.render_video,
    )

    act_eval = None
    if act_train_results is not None:
        print("\n  [5C] Evaluating Trained ACT Policy...")
        act_eval = evaluate_policy(
            checkpoint_path=act_train_results["checkpoint_path"],
            controller_type="policy",
            model_type="act",
            num_episodes=args.eval_episodes,
            output_dir=results_dir,
            render_video=args.render_video,
        )

    # 6. Comparison Summary & Plot
    print("\n>>> STEP 6/6: Synthesizing Benchmark Comparison...")
    comparison_data = {
        "Scripted Baseline": {
            "success_rate": baseline_eval["success_rate"],
            "avg_steps": baseline_eval["avg_steps"],
        },
        "BC Policy": {
            "success_rate": bc_eval["success_rate"],
            "avg_steps": bc_eval["avg_steps"],
        },
    }
    if act_eval is not None:
        comparison_data["ACT Policy"] = {
            "success_rate": act_eval["success_rate"],
            "avg_steps": act_eval["avg_steps"],
        }

    comp_plot_path = results_dir / "comparison_summary.png"
    plot_policy_vs_baseline(comparison_data, comp_plot_path)

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    for method, data in comparison_data.items():
        print(f"  {method:20s} | Success Rate: {data['success_rate']:.1%} | Avg Steps: {data['avg_steps']:.1f}")
    print(f"\nSaved Comparison Chart to: {comp_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
