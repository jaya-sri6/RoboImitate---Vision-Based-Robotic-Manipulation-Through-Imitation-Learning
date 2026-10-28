# RoboImitate Experiment Report Template

Use this template to record systematic robotics learning experiments, ablation studies, and generalization benchmarks.

---

## 1. Experiment Overview
- **Experiment Title / ID**: `[e.g. EXP-001-BC-vs-ACT-SampleEfficiency]`
- **Date / Timestamp**: `[YYYY-MM-DD]`
- **Author / Engineer**: `[Name]`
- **Core Hypothesis**: `[State precise hypothesis, e.g., Action Chunking with Transformers (chunk=10) reduces covariate shift compared to single-step BC, improving success rate on held-out positions by at least 25%.]`

---

## 2. Experimental Setup
- **Environment**: `RoboImitate-PickAndPlace-v0`
- **Control Rate**: `40 Hz` (timestep 0.005s, frame skip 5)
- **Observation Space**: Overhead camera (84x84x3 RGB) + 4D Proprioception `[ee_x, ee_y, ee_z, gripper]`
- **Action Space**: 4D continuous end-effector delta `[dx, dy, dz, gripper]`
- **Random Seed**: `[e.g. 42]`
- **Compute Hardware**: `[e.g. NVIDIA RTX 4090 / Intel Core i7 / Apple M-series]`

---

## 3. Dataset & Demonstration Profile
- **Dataset File**: `data/demos/expert_demonstrations.npz`
- **Demonstration Collector**: `ScriptedPickAndPlaceExpert` (8-phase FSM)
- **Total Demonstrated Episodes**: `[e.g. 100]`
- **Total Transitions**: `[e.g. 5,420]`
- **Demonstration Success Rate**: `[e.g. 96.0%]`
- **Train / Validation Trajectory Split**: `80% / 20%`

---

## 4. Policy Configuration & Hyperparameters

| Hyperparameter | Behavior Cloning (BC) | Action Chunking with Transformers (ACT) |
| :--- | :---: | :---: |
| **Vision Backbone** | NatureCNN (256-d) | NatureCNN (128-d token projection) |
| **Proprioception Dim** | 64-d projection | 128-d token projection |
| **Action Chunk Horizon ($K$)** | 1 (single-step) | 10 timesteps |
| **Transformer Layers** | N/A | 2 Encoder / 2 Decoder |
| **Attention Heads** | N/A | 4 heads |
| **CVAE Latent Dim ($z$)** | N/A | 32 dimensions ($\beta_{KL} = 10.0$) |
| **Batch Size** | 32 | 16 |
| **Learning Rate** | $3 \times 10^{-4}$ (Cosine) | $2 \times 10^{-4}$ (Cosine) |
| **Epochs** | 30 | 30 |

---

## 5. Empirical Results

### Quantitative Benchmark Comparison

| Method | In-Distribution Success Rate ($N=100$) | Held-Out Generalization Success Rate ($N=50$) | Mean Steps to Success | Mean Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Scripted Baseline** | `%` | `%` | steps | ms |
| **BC Policy** | `%` | `%` | steps | ms |
| **ACT Policy** | `%` | `%` | steps | ms |

### Generalization Across 5 Held-Out Positions

| Position Key | Target Coordinates | Scripted Baseline | BC Policy | ACT Policy |
| :--- | :---: | :---: | :---: | :---: |
| `held_out_far_left` | `[-0.22, 0.50, 0.42]` | `%` | `%` | `%` |
| `held_out_center_near`| `[0.00, 0.40, 0.42]` | `%` | `%` | `%` |
| `held_out_far_back` | `[-0.10, 0.65, 0.42]` | `%` | `%` | `%` |
| `held_out_near_left` | `[-0.20, 0.38, 0.42]` | `%` | `%` | `%` |
| `held_out_center_back`| `[0.00, 0.65, 0.42]` | `%` | `%` | `%` |

---

## 6. Failure Mode Classification Taxonomy

| Failure Mode | Description | BC Count (%) | ACT Count (%) |
| :--- | :--- | :---: | :---: |
| **Reach Failure** | End-effector failed to reach within 0.05m of object | | |
| **Grasp Failure** | Reached object but failed to establish friction grasp | | |
| **Drop Failure** | Object lifted off surface but dropped prematurely | | |
| **Placement Failure** | Object reached receptacle perimeter but missed zone | | |
| **Timeout** | Exceeded 60 episode steps without completion | | |
| **Out-of-Bounds** | Robot drifted outside physical safety workspace | | |

---

## 7. Analysis & Discussion
- **Hypothesis Validation**: `[Did the empirical results confirm or refute the hypothesis?]`
- **Covariate Shift / Compounding Error Observations**: `[Observations on drift over time]`
- **Limitations**: `[Hardware constraints, synthetic lighting, dataset size]`
- **Next Proposed Iteration**: `[e.g. Data augmentation, increasing dataset to 200 demos, ResNet18 backbone]`

