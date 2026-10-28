"""Evaluation metrics and failure classification taxonomy for RoboImitate."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class MetricsTracker:
    """Tracks per-episode evaluation metrics, timing statistics, and failure categories."""

    def __init__(self, failure_categories: Optional[List[str]] = None):
        if failure_categories is None:
            self.failure_categories = [
                "reach_failure",
                "grasp_failure",
                "drop_failure",
                "placement_failure",
                "timeout",
                "out_of_bounds",
            ]
        else:
            self.failure_categories = failure_categories

        self.episodes: List[Dict[str, Any]] = []
        self.inference_latencies: List[float] = []
        self.step_durations: List[float] = []

    def record_episode(
        self,
        episode_idx: int,
        seed: int,
        success: bool,
        steps: int,
        reward: float,
        duration_seconds: float,
        min_dist_to_obj: float,
        final_dist_obj_target: float,
        has_grasped: bool,
        has_lifted: bool,
        position_name: str = "training_distribution",
        object_pos: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None,
        out_of_bounds: bool = False,
    ) -> str:
        """Record outcome of an evaluation episode and classify failure mode if unsuccessful.

        Returns:
            The assigned outcome string ('success' or failure category).
        """
        if success:
            outcome = "success"
        elif out_of_bounds:
            outcome = "out_of_bounds"
        elif min_dist_to_obj > 0.055:
            outcome = "reach_failure"
        elif not has_lifted:
            outcome = "grasp_failure"
        elif final_dist_obj_target > 0.045:
            outcome = "drop_failure"
        elif has_lifted and final_dist_obj_target <= 0.045:
            outcome = "placement_failure"
        else:
            outcome = "timeout"

        ep_record = {
            "episode": episode_idx,
            "seed": seed,
            "position_name": position_name,
            "success": bool(success),
            "outcome": outcome,
            "steps": int(steps),
            "reward": float(reward),
            "duration_seconds": float(duration_seconds),
            "min_dist_to_obj": float(min_dist_to_obj),
            "final_dist_obj_target": float(final_dist_obj_target),
            "has_grasped": bool(has_grasped),
            "has_lifted": bool(has_lifted),
            "object_pos": object_pos.tolist() if object_pos is not None else None,
            "target_pos": target_pos.tolist() if target_pos is not None else None,
        }
        self.episodes.append(ep_record)
        return outcome

    def record_latency(self, inference_time_sec: float, step_time_sec: float) -> None:
        """Record model inference latency and total simulation step time."""
        self.inference_latencies.append(inference_time_sec)
        self.step_durations.append(step_time_sec)

    def to_dataframe(self) -> pd.DataFrame:
        """Export per-episode records as a pandas DataFrame."""
        return pd.DataFrame(self.episodes)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate benchmark metrics across all recorded episodes."""
        total_episodes = len(self.episodes)
        if total_episodes == 0:
            return {"total_episodes": 0, "success_rate": 0.0}

        df = self.to_dataframe()
        successful_episodes = int(df["success"].sum())
        success_rate = float(successful_episodes / total_episodes)

        # Failure distribution
        failure_counts: Dict[str, int] = {}
        for cat in self.failure_categories:
            failure_counts[cat] = int((df["outcome"] == cat).sum())

        # Timing breakdown
        latencies_ms = np.array(self.inference_latencies) * 1000.0 if self.inference_latencies else np.array([0.0])
        steps_ms = np.array(self.step_durations) * 1000.0 if self.step_durations else np.array([0.0])

        # Per-position summary
        position_breakdown: Dict[str, Dict[str, Any]] = {}
        for pos_name, group in df.groupby("position_name"):
            pos_total = len(group)
            pos_succ = int(group["success"].sum())
            position_breakdown[str(pos_name)] = {
                "total": pos_total,
                "successful": pos_succ,
                "success_rate": float(pos_succ / max(1, pos_total)),
                "avg_steps": float(group["steps"].mean()),
            }

        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": total_episodes - successful_episodes,
            "success_rate": success_rate,
            "avg_steps": float(df["steps"].mean()),
            "std_steps": float(df["steps"].std()) if total_episodes > 1 else 0.0,
            "avg_reward": float(df["reward"].mean()),
            "avg_duration_sec": float(df["duration_seconds"].mean()),
            "latency": {
                "inference_mean_ms": float(np.mean(latencies_ms)),
                "inference_median_ms": float(np.median(latencies_ms)),
                "inference_p95_ms": float(np.percentile(latencies_ms, 95)),
                "step_mean_ms": float(np.mean(steps_ms)),
            },
            "failure_counts": failure_counts,
            "position_breakdown": position_breakdown,
        }

    def save_reports(self, output_dir: Union[str, Path], prefix: str = "eval") -> Tuple[Path, Path]:
        """Save JSON summary report and CSV per-episode log.

        Returns:
            Tuple of (json_path, csv_path).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = self.compute_summary()
        json_path = out_dir / f"{prefix}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        csv_path = out_dir / f"{prefix}_episodes.csv"
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)

        return json_path, csv_path
