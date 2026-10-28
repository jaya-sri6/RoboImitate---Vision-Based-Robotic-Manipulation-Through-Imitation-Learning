#!/usr/bin/env python3
"""RoboImitate Setup Check Script.

Validates Python runtime, core dependencies, MuJoCo physics simulation,
Gymnasium environment stepping, camera rendering, and device availability.
"""

import sys
from pathlib import Path
import numpy as np

def run_setup_check() -> bool:
    print("=" * 60)
    print(" RoboImitate: System & Environment Verification Check")
    print("=" * 60)

    # 1. Check Python version
    print(f"\n[1/6] Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  [FAIL] Python 3.10+ required.")
        return False
    print("  [OK] Python version supported.")

    # 2. Check PyTorch
    try:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[2/6] PyTorch: v{torch.__version__} (Device: {device_str})")
        x = torch.randn(2, 3)
        _ = x + 1.0
        print("  [OK] PyTorch tensor operations functional.")
    except Exception as e:
        print(f"  [FAIL] PyTorch check failed: {e}")
        return False

    # 3. Check MuJoCo & Physics
    try:
        import mujoco
        print(f"\n[3/6] MuJoCo: v{mujoco.__version__}")
        model_xml = """<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(model_xml)
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        print("  [OK] MuJoCo physics engine functional.")
    except Exception as e:
        print(f"  [FAIL] MuJoCo check failed: {e}")
        return False

    # 4. Check RoboImitate Environment
    try:
        from roboimitate.utils.config import load_config
        from roboimitate.env.manipulation_env import RoboImitatePickAndPlaceEnv

        cfg_path = Path("configs/environment.yaml")
        cfg = load_config(cfg_path) if cfg_path.exists() else {}
        env = RoboImitatePickAndPlaceEnv(config=cfg)
        obs, info = env.reset(seed=42)

        print(f"\n[4/6] RoboImitatePickAndPlaceEnv:")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Image observation shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
        print(f"  Proprio observation shape: {obs['proprio'].shape}, dtype: {obs['proprio'].dtype}")
        print(f"  Action space: {env.action_space}")

        # Step environment
        act = np.array([0.01, 0.0, -0.01, -1.0], dtype=np.float32)
        next_obs, rew, term, trunc, info = env.step(act)
        print(f"  Step test passed. Reward: {rew:.3f}, Terminated: {term}, Truncated: {trunc}")
        env.close()
        print("  [OK] Environment stepping and reset functional.")
    except Exception as e:
        print(f"  [FAIL] RoboImitate environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Check Scientific Stack
    try:
        import cv2
        import matplotlib
        import pandas as pd
        import yaml
        import h5py
        print(f"\n[5/6] Supporting Libraries:")
        print(f"  OpenCV: v{cv2.__version__}")
        print(f"  Matplotlib: v{matplotlib.__version__}")
        print(f"  Pandas: v{pd.__version__}")
        print(f"  PyYAML: v{yaml.__version__}")
        print(f"  H5Py: v{h5py.__version__}")
        print("  [OK] All supporting libraries available.")
    except Exception as e:
        print(f"  [FAIL] Supporting libraries check failed: {e}")
        return False

    # 6. Overall Readiness
    print("\n[6/6] Overall Status: READY")
    print("=" * 60)
    print(" System is fully verified and ready for demonstration collection,")
    print(" imitation learning policy training (BC & ACT), and evaluation.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_setup_check()
    sys.exit(0 if success else 1)
