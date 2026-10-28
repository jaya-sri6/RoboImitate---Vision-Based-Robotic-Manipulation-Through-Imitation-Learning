"""Evaluation runner for Scripted baseline and learned imitation policies."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from roboimitate.env.manipulation_env import RoboImitatePickAndPlaceEnv
from roboimitate.env.wrappers import VideoRecorderWrapper
from roboimitate.evaluation.baseline import ScriptedPickAndPlaceExpert
from roboimitate.evaluation.metrics import MetricsTracker
from roboimitate.policies.behavior_cloning import BCPolicy
from roboimitate.policies.act_policy import ACTPolicy
from roboimitate.policies.inference import PolicyInferenceWrapper
from roboimitate.utils.checkpoint import load_checkpoint
from roboimitate.utils.config import load_config
from roboimitate.utils.reproducibility import set_seed
from roboimitate.visualization.plots import plot_generalization_comparison, plot_failure_distribution


def evaluate_policy(
    checkpoint_path: Optional[Union[str, Path]] = None,
    controller_type: str = "policy",
    model_type: Optional[str] = None,
    num_episodes: int = 20,
    eval_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    output_dir: Union[str, Path] = "results",
    seed: int = 1234,
    device: str = "cpu",
    render_video: bool = False,
    video_limit: int = 3,
    test_held_out: bool = True,
) -> Dict[str, Any]:
    """Run standardized evaluation across in-distribution and held-out configurations.

    Args:
        checkpoint_path: Path to model checkpoint .pt (required if controller_type='policy').
        controller_type: 'policy' or 'baseline'.
        model_type: 'bc' or 'act' (inferred from checkpoint if None).
        num_episodes: Number of in-distribution evaluation episodes.
        eval_config: Path or dictionary for evaluation settings.
        env_config: Path or dictionary for environment settings.
        output_dir: Results directory for reports, plots, and videos.
        seed: Evaluation random seed.
        device: 'cpu' or 'cuda'.
        render_video: Whether to record MP4 videos.
        video_limit: Max number of videos to record.
        test_held_out: Whether to run generalization tests on 5 held-out positions.

    Returns:
        Summary evaluation results dictionary.
    """
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    # Load configurations
    if isinstance(env_config, (str, Path)):
        env_cfg = load_config(env_config)
    elif isinstance(env_config, dict):
        env_cfg = env_config
    else:
        env_cfg_path = Path("configs/environment.yaml")
        env_cfg = load_config(env_cfg_path) if env_cfg_path.exists() else {}

    if isinstance(eval_config, (str, Path)):
        eval_cfg = load_config(eval_config)
    elif isinstance(eval_config, dict):
        eval_cfg = eval_config
    else:
        eval_cfg_path = Path("configs/evaluation.yaml")
        eval_cfg = load_config(eval_cfg_path) if eval_cfg_path.exists() else {}

    # Initialize environment
    base_env = RoboImitatePickAndPlaceEnv(config=env_cfg)
    if render_video:
        env = VideoRecorderWrapper(base_env, video_dir=video_dir, fps=20, enabled=True)
    else:
        env = base_env

    # Setup controller
    policy_wrapper = None
    expert_controller = None

    if controller_type.lower() == "baseline":
        method_name = "Scripted_Baseline"
        expert_controller = ScriptedPickAndPlaceExpert()
    else:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when controller_type is 'policy'.")

        chk = load_checkpoint(checkpoint_path, device=device)
        m_type = (model_type or chk.get("model_type", "bc")).lower()
        method_name = f"{m_type.upper()}_Policy"

        norm_stats = chk.get("norm_stats", {})
        model_cfg = chk.get("model_cfg", {})

        if m_type == "bc":
            model = BCPolicy(
                image_channels=model_cfg.get("image_channels", 3),
                proprio_dim=model_cfg.get("proprio_dim", 4),
                action_dim=model_cfg.get("action_dim", 4),
                hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
                dropout=model_cfg.get("dropout", 0.1),
            )
        else:
            model = ACTPolicy(
                image_channels=model_cfg.get("image_channels", 3),
                proprio_dim=model_cfg.get("proprio_dim", 4),
                action_dim=model_cfg.get("action_dim", 4),
                chunk_size=int(model_cfg.get("chunk_size", 10)),
                d_model=int(model_cfg.get("d_model", 128)),
                nhead=int(model_cfg.get("nhead", 4)),
                num_encoder_layers=int(model_cfg.get("num_encoder_layers", 2)),
                num_decoder_layers=int(model_cfg.get("num_decoder_layers", 2)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
                dropout=0.0,
                kl_weight=float(model_cfg.get("kl_weight", 10.0)),
                use_cvae=model_cfg.get("use_cvae", True),
            )

        model.load_state_dict(chk["model_state_dict"])
        policy_wrapper = PolicyInferenceWrapper(
            model=model,
            model_type=m_type,
            norm_stats=norm_stats,
            device=device,
            chunk_size=int(model_cfg.get("chunk_size", 10)),
        )

    tracker = MetricsTracker()
    recorded_videos = 0

    print(f"\n=======================================================")
    print(f" Starting Evaluation: {method_name}")
    print(f" In-distribution Episodes: {num_episodes} | Device: {device}")
    print(f"=======================================================")

    # 1. In-Distribution Evaluation
    for ep in range(num_episodes):
        ep_seed = seed + ep * 137
        obs, info = env.reset(seed=ep_seed)
        if policy_wrapper is not None:
            policy_wrapper.reset()
        if expert_controller is not None:
            expert_controller.reset()

        step = 0
        total_reward = 0.0
        t_ep_start = time.perf_counter()
        ep_success = False

        while True:
            step += 1
            t_step_start = time.perf_counter()

            if expert_controller is not None:
                t_inf_start = time.perf_counter()
                priv_state = env.unwrapped.get_privileged_state() if hasattr(env, "unwrapped") else env.get_privileged_state()
                action = expert_controller.get_action(priv_state)
                inf_latency = time.perf_counter() - t_inf_start
            else:
                action = policy_wrapper.get_action(obs)
                inf_latency = policy_wrapper.last_latency_seconds

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            step_duration = time.perf_counter() - t_step_start

            tracker.record_latency(inf_latency, step_duration)
            total_reward += float(reward)
            obs = next_obs

            if step_info.get("success", False):
                ep_success = True

            if terminated or truncated:
                break

        ep_duration = time.perf_counter() - t_ep_start
        priv = env.unwrapped.get_privileged_state() if hasattr(env, "unwrapped") else env.get_privileged_state()

        outcome = tracker.record_episode(
            episode_idx=ep + 1,
            seed=ep_seed,
            success=ep_success,
            steps=step,
            reward=total_reward,
            duration_seconds=ep_duration,
            min_dist_to_obj=step_info.get("min_dist_to_obj", 1.0),
            final_dist_obj_target=step_info.get("dist_obj_target", 1.0),
            has_grasped=step_info.get("has_grasped", False),
            has_lifted=step_info.get("has_lifted", False),
            position_name="training_distribution",
            object_pos=priv["object_pos"],
            target_pos=priv["target_pos"],
        )

        if render_video and isinstance(env, VideoRecorderWrapper) and recorded_videos < video_limit:
            env.save_video(prefix=f"{method_name.lower()}_in_dist", success=ep_success)
            recorded_videos += 1

        if (ep + 1) % max(1, num_episodes // 5) == 0 or (ep + 1) == num_episodes:
            curr_summary = tracker.compute_summary()
            print(f"  [In-Dist] Episode {ep + 1:02d}/{num_episodes:02d} | Success Rate: {curr_summary['success_rate']:.1%} | Outcome: {outcome}")

    # 2. Generalization Evaluation on 5 Held-Out Positions
    held_out_list = eval_cfg.get("generalization", {}).get("positions", [])
    if test_held_out and held_out_list:
        print(f"\n--- Testing Generalization on {len(held_out_list)} Held-Out Positions ---")
        ep_per_pos = max(2, num_episodes // len(held_out_list))

        for pos_cfg in held_out_list:
            pos_name = pos_cfg["name"]
            fixed_pos = np.array(pos_cfg["pos"], dtype=np.float32)

            for pos_ep in range(ep_per_pos):
                pos_seed = seed + 5000 + pos_ep * 31
                obs, info = env.reset(seed=pos_seed, options={"object_pos": fixed_pos})
                if policy_wrapper is not None:
                    policy_wrapper.reset()
                if expert_controller is not None:
                    expert_controller.reset()

                step = 0
                total_reward = 0.0
                t_ep_start = time.perf_counter()
                ep_success = False

                while True:
                    step += 1
                    t_step_start = time.perf_counter()

                    if expert_controller is not None:
                        t_inf_start = time.perf_counter()
                        priv_state = env.unwrapped.get_privileged_state() if hasattr(env, "unwrapped") else env.get_privileged_state()
                        action = expert_controller.get_action(priv_state)
                        inf_latency = time.perf_counter() - t_inf_start
                    else:
                        action = policy_wrapper.get_action(obs)
                        inf_latency = policy_wrapper.last_latency_seconds

                    next_obs, reward, terminated, truncated, step_info = env.step(action)
                    step_duration = time.perf_counter() - t_step_start

                    tracker.record_latency(inf_latency, step_duration)
                    total_reward += float(reward)
                    obs = next_obs

                    if step_info.get("success", False):
                        ep_success = True

                    if terminated or truncated:
                        break

                ep_duration = time.perf_counter() - t_ep_start
                priv = env.unwrapped.get_privileged_state() if hasattr(env, "unwrapped") else env.get_privileged_state()

                tracker.record_episode(
                    episode_idx=len(tracker.episodes) + 1,
                    seed=pos_seed,
                    success=ep_success,
                    steps=step,
                    reward=total_reward,
                    duration_seconds=ep_duration,
                    min_dist_to_obj=step_info.get("min_dist_to_obj", 1.0),
                    final_dist_obj_target=step_info.get("dist_obj_target", 1.0),
                    has_grasped=step_info.get("has_grasped", False),
                    has_lifted=step_info.get("has_lifted", False),
                    position_name=pos_name,
                    object_pos=priv["object_pos"],
                    target_pos=priv["target_pos"],
                )

                if render_video and isinstance(env, VideoRecorderWrapper) and recorded_videos < (video_limit * 2):
                    env.save_video(prefix=f"{method_name.lower()}_{pos_name}", success=ep_success)
                    recorded_videos += 1

            pos_stats = tracker.compute_summary()["position_breakdown"].get(pos_name, {})
            print(f"  [{pos_name}] Success Rate: {pos_stats.get('success_rate', 0.0):.1%} ({pos_stats.get('successful', 0)}/{pos_stats.get('total', 0)})")

    env.close()

    # 3. Save Summary, CSV & Figures
    prefix = method_name.lower()
    json_path, csv_path = tracker.save_reports(out_dir, prefix=prefix)
    summary = tracker.compute_summary()

    # Plots
    pos_rates = {k: v["success_rate"] for k, v in summary["position_breakdown"].items()}
    gen_plot_path = out_dir / f"{prefix}_generalization.png"
    plot_generalization_comparison(pos_rates, gen_plot_path, title=f"{method_name} Spatial Generalization")

    fail_plot_path = out_dir / f"{prefix}_failures.png"
    plot_failure_distribution(summary["failure_counts"], fail_plot_path)

    print(f"\n=======================================================")
    print(f" Evaluation Complete: {method_name}")
    print(f" Overall Success Rate: {summary['success_rate']:.1%} ({summary['successful_episodes']}/{summary['total_episodes']})")
    print(f" Mean Inference Latency: {summary['latency']['inference_mean_ms']:.2f} ms")
    print(f" Reports saved:")
    print(f"   Summary JSON: {json_path}")
    print(f"   Episodes CSV: {csv_path}")
    print(f"   Generalization Plot: {gen_plot_path}")
    print(f"   Failure Distribution: {fail_plot_path}")
    print(f"=======================================================\n")

    summary["json_path"] = str(json_path)
    summary["csv_path"] = str(csv_path)
    summary["gen_plot_path"] = str(gen_plot_path)
    summary["fail_plot_path"] = str(fail_plot_path)

    return summary
