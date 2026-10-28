"""Automated tests for evaluation metrics and failure classification."""

from pathlib import Path
import numpy as np
import pytest

from roboimitate.evaluation.metrics import MetricsTracker


def test_metrics_tracker_success():
    """Test recording a successful episode."""
    tracker = MetricsTracker()
    outcome = tracker.record_episode(
        episode_idx=1,
        seed=42,
        success=True,
        steps=35,
        reward=10.5,
        duration_seconds=1.2,
        min_dist_to_obj=0.01,
        final_dist_obj_target=0.02,
        has_grasped=True,
        has_lifted=True,
    )
    assert outcome == "success"

    summary = tracker.compute_summary()
    assert summary["total_episodes"] == 1
    assert summary["successful_episodes"] == 1
    assert summary["success_rate"] == 1.0


def test_failure_classification():
    """Test failure taxonomy categorization on different failure modes."""
    tracker = MetricsTracker()

    # Reach failure: end effector never approached object
    out1 = tracker.record_episode(
        episode_idx=1, seed=1, success=False, steps=60, reward=0.0,
        duration_seconds=1.0, min_dist_to_obj=0.15, final_dist_obj_target=0.3,
        has_grasped=False, has_lifted=False,
    )
    assert out1 == "reach_failure"

    # Grasp failure: reached object but never lifted
    out2 = tracker.record_episode(
        episode_idx=2, seed=2, success=False, steps=60, reward=0.0,
        duration_seconds=1.0, min_dist_to_obj=0.02, final_dist_obj_target=0.3,
        has_grasped=False, has_lifted=False,
    )
    assert out2 == "grasp_failure"

    # Drop failure: lifted object but dropped it before target
    out3 = tracker.record_episode(
        episode_idx=3, seed=3, success=False, steps=60, reward=0.0,
        duration_seconds=1.0, min_dist_to_obj=0.02, final_dist_obj_target=0.2,
        has_grasped=True, has_lifted=True,
    )
    assert out3 == "drop_failure"

    summary = tracker.compute_summary()
    assert summary["total_episodes"] == 3
    assert summary["successful_episodes"] == 0
    assert summary["failure_counts"]["reach_failure"] == 1
    assert summary["failure_counts"]["grasp_failure"] == 1
    assert summary["failure_counts"]["drop_failure"] == 1


def test_metrics_report_saving(tmp_path: Path):
    """Test saving summary JSON and CSV logs."""
    tracker = MetricsTracker()
    tracker.record_episode(
        episode_idx=1, seed=10, success=True, steps=40, reward=9.0,
        duration_seconds=0.8, min_dist_to_obj=0.02, final_dist_obj_target=0.01,
        has_grasped=True, has_lifted=True,
    )
    tracker.record_latency(0.005, 0.02)

    json_path, csv_path = tracker.save_reports(tmp_path, prefix="test")
    assert json_path.exists()
    assert csv_path.exists()

