# RoboImitate Troubleshooting Guide

Common issues, environment diagnostic steps, and resolutions for RoboImitate.

---

## 1. MuJoCo Rendering & Display Issues

### Issue: `GLContext` or `GLFW` initialization error on headless Linux / Windows
- **Symptom**: `RuntimeError: Failed to initialize MuJoCo offscreen OpenGL context.`
- **Resolution**: RoboImitate includes an automatic headless synthetic visual fallback (`_render_synthetic_camera`). If OpenGL fails or is unavailable, the environment automatically renders top-down RGB observations using NumPy projection without crashing.
- **For hardware OpenGL on Linux**: Set `export MUJOCO_GL=egl` or `export MUJOCO_GL=osmesa`.

---

## 2. Python Environment & Dependency Conflicts

### Issue: `No module named 'roboimitate'`
- **Cause**: Package not installed in editable mode.
- **Resolution**:
  ```bash
  pip install -e .
  ```
  Or ensure repository root is in `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src  # Linux/macOS
  $env:PYTHONPATH = "$PWD/src;$env:PYTHONPATH" # Windows PowerShell
  ```

### Issue: PyTorch CPU vs CUDA
- **Resolution**: RoboImitate automatically detects CUDA availability. To force CPU execution:
  ```bash
  python scripts/train_policy.py --device cpu
  python scripts/evaluate_policy.py --device cpu
  ```

---

## 3. Training & Convergence Diagnostics

### Issue: Diverging or stagnant loss during Behavior Cloning
- **Check**: Ensure input normalization statistics were computed properly. Verify `norm_stats` exist in checkpoint.
- **Action Clipping**: Check `scripts/validate_dataset.py` to confirm expert actions do not exceed $[-0.04, 0.04]$ and $[-1.0, 1.0]$.

### Issue: ACT CVAE KL divergence collapse
- **Check**: The hyperparameter `kl_weight` balances reconstruction with Gaussian prior divergence. If KL drops to 0 or explodes, tune `kl_weight` in `configs/training.yaml` between `1.0` and `10.0`.

