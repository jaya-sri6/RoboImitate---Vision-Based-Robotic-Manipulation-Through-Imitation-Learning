# RoboImitate Methodology & Algorithmic Formulation

This document details the mathematical and algorithmic methodology employed in **RoboImitate** for expert demonstration generation, policy learning, and evaluation.

---

## 1. Task Formulation

The tabletop manipulation task is framed as a discrete-time Markov Decision Process (MDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

- **State Space $\mathcal{S}$**: The privileged state $s_t = \langle p_{ee}, q_{ee}, p_{obj}, p_{tgt} \rangle \in \mathbb{R}^{10}$ encompasses the 3D end-effector coordinate $p_{ee} \in \mathbb{R}^3$, the gripper scalar command $q_{ee} \in \mathbb{R}$, 3D object center of mass $p_{obj} \in \mathbb{R}^3$, and 3D target receptacle coordinates $p_{tgt} \in \mathbb{R}^3$.
- **Policy Observation Space $\mathcal{O}$**: The policy operates strictly on non-privileged, partial observations $o_t = \langle I_t, q_t \rangle$:
  - $I_t \in \mathbb{R}^{84 \times 84 \times 3}$: Overhead RGB camera image.
  - $q_t \in \mathbb{R}^4$: Proprioceptive end-effector feedback $[p_{ee, x}, p_{ee, y}, p_{ee, z}, q_{ee}]$.
- **Action Space $\mathcal{A}$**: 4D continuous end-effector delta displacement and gripper command:
  $$a_t = [\Delta x, \Delta y, \Delta z, g] \in [-0.04, 0.04]^3 \times [-1.0, 1.0]$$
- **Control Rate**: Simulation integration timestep $\delta t = 0.005\text{s}$ with frame skip $k = 5$, yielding an effective control frequency of $40\text{ Hz}$ ($\Delta t = 0.025\text{s}$).

---

## 2. Privileged Expert Controller

Demonstrations are synthesized using a deterministic 8-phase finite-state machine (FSM):
1. **APPROACH**: End-effector translates toward a standoff waypoint $p_{obj} + [0, 0, h_{hover}]^T$ ($h_{hover} = 0.12\text{m}$).
2. **DESCEND**: Gripper descends vertically until straddling the box ($z \le 0.460\text{m}$).
3. **GRASP**: Actuator command $g = +1.0$ is maintained for 4 timesteps to establish bilateral contact friction.
4. **LIFT**: End-effector lifts vertically to transit clearance $z_{lift} = 0.540\text{m}$.
5. **TRANSIT**: Transports the object horizontally toward the target receptacle $(x_{tgt}, y_{tgt}, z_{lift})$.
6. **DESCEND_PLACE**: Lowers object to receptacle height $z_{place} = 0.465\text{m}$.
7. **RELEASE**: Gripper opens ($g = -1.0$) for 4 timesteps.
8. **RETRACT**: Ascends to transit clearance, terminating the trajectory.

---

## 3. Supervised Policy Learning

### A. Behavior Cloning (BC)

Supervised regression minimizes empirical risk over demonstrated state-action pairs:
$$\mathcal{L}_{BC}(\theta) = \frac{1}{N} \sum_{i=1}^N \| \pi_\theta(I_i, q_i) - a_i^* \|_1$$
where $\pi_\theta$ consists of a convolutional vision backbone (NatureCNN) and proprioceptive linear projections.

### B. Action Chunking with Transformers (ACT)

To overcome compounding errors and temporal inconsistency in single-step policies, ACT predicts an action chunk $A_t = [a_t, a_{t+1}, \dots, a_{t+K-1}]$ of length $K = 10$.

#### CVAE Formulation
A Conditional Variational Autoencoder (CVAE) models the distribution of future action sequences conditioned on current observations:
- **Recognition Encoder** $q_\phi(z \mid A_t, o_t)$: Maps ground-truth action chunks and current state into latent style distribution $\mathcal{N}(\mu, \sigma)$.
- **Prior** $p(z) = \mathcal{N}(0, I)$: Standard Gaussian prior over stylistic variations.
- **Decoder** $p_\theta(A_t \mid z, o_t)$: Transformer decoder predicting the action sequence from $K$ query positions cross-attending to $[z, \text{visual\_tokens}, \text{proprio\_tokens}]$.

#### Objective Function
$$\mathcal{L}_{ACT}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi}\left[ \sum_{k=0}^{K-1} m_{t+k} \| \hat{a}_{t+k} - a_{t+k}^* \|_1 \right] + \beta_{KL} D_{KL}(q_\phi(z \mid A_t, o_t) \parallel p(z))$$
where $m_{t+k} \in \{0, 1\}$ masks padding beyond episode termination.

---

## 4. Evaluation Protocol & Generalization

1. **In-Distribution Evaluation**: Evaluates 20–100 episodes with initial object coordinates sampled from the training bounding box $x \in [-0.18, -0.06]$, $y \in [0.44, 0.56]$.
2. **Out-of-Distribution (Held-Out) Evaluation**: Evaluates robustness against 5 distinct spatial coordinates outside the training distribution.
3. **Failure Categorization**:
   - $\text{Reach Failure}$: $\min_t \|p_{ee} - p_{obj}\| > 0.05\text{m}$.
   - $\text{Grasp Failure}$: $\text{has\_lifted} = \text{False}$.
   - $\text{Drop Failure}$: $\text{has\_lifted} = \text{True}$ but dropped before target zone.
   - $\text{Placement Failure}$: Reached target perimeter but released outside acceptance tolerance.
   - $\text{Timeout}$: $\text{step\_count} \ge \text{max\_episode\_steps}$.

