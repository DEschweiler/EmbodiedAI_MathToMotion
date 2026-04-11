---
id: 011
title: From Simulation to Reality — and the Live Test
summary: Close the sim-to-real gap, estimate state, and run the full stack.
tags: [sim-to-real, robustness, state-estimation]
learning_goals:
  - Understand the sources of the sim-to-real gap.
  - Explain domain randomization and privileged learning as bridging strategies.
  - Describe the Kalman filter for sensor fusion and state estimation.
  - Map the complete pipeline from voice command to physical movement.
  - Critically assess current limitations and open research questions.
---

The previous ten sessions built every component of the system: from data representations through neural networks, foundation models, language understanding, multimodal perception, agent planning, and reinforcement learning. This final session closes the loop — from simulation to physical hardware — and critically reflects on what works, what doesn't, and where the field is heading.

## The Sim-to-Real Gap

Training a locomotion policy in simulation offers major advantages: simulation runs thousands of times faster than real time, is perfectly safe (no hardware damage from falls), and can generate arbitrary amounts of data. But simulation is a simplified model of reality, and that simplification creates problems when the policy is deployed on real hardware.

**Sources of the sim-to-real gap:**

| Simulation Assumption | Reality |
|---|---|
| Fixed friction coefficient | Varies by surface (tile, carpet, gravel) |
| Perfect, noise-free sensors | IMU noise, encoder quantization, drift |
| Instantaneous actuator response | Motor delays of 5–20 ms |
| Known, static world geometry | Obstacles appear and move unpredictably |
| Uniform lighting conditions | Specular reflections, shadows, direct sunlight |
| Simple contact models | Complex compliant contacts, slip |

A policy that achieves perfect scores in simulation often fails on its first real-world test — it has overfit to the simulator's assumptions. Closing this gap is one of the central challenges of robotics.

## Domain Randomization

The most effective single technique: make the simulation intentionally messy. During training, randomize the physical parameters at the start of each episode:

- **Friction:** sample $\mu \sim \mathcal{U}(0.4, 1.2)$ per episode
- **Motor delay:** sample $\tau_\text{delay} \sim \mathcal{U}(0, 20\text{ ms})$
- **Sensor noise:** add $\epsilon \sim \mathcal{N}(0, \sigma^2)$ to all sensor readings
- **Body mass:** perturb mass of each link by $\pm 20\%$
- **Terrain:** randomly generated slopes, steps, and deformable surfaces

If the policy can walk robustly under all these variations, real-world conditions become "just another sample" within the training distribution. Domain randomization is data augmentation applied to physics rather than pixels — and it connects directly to the augmentation techniques from Session 3.

**Push recovery:** explicitly randomize external disturbances — apply random force impulses to the robot's body during training. The policy learns to recover from perturbations without ever having been programmed with explicit balance rules.

## Privileged Learning and Teacher-Student Distillation

Domain randomization creates a difficulty: during training, the policy can access the true physical parameters (because the simulation knows them). But during deployment, the robot doesn't know the exact friction or motor delay — it must infer them from sensors.

**Privileged learning** (also called teacher-student distillation in this context) addresses this asymmetry:

1. **Teacher policy:** trained in simulation with access to privileged information (true friction, true motor delay, ground-truth contact forces). Learns near-optimal behavior.
2. **Student policy:** trained to *imitate* the teacher, but using only observations available at deployment (sensor readings, proprioceptive state). The student learns to implicitly estimate the hidden parameters from the pattern of sensor readings.

This two-stage approach consistently outperforms direct policy training without privileged information. The teacher defines "what good behavior looks like"; the student learns to achieve it from observable information only.

## State Estimation: The Kalman Filter

The robot never knows its exact state — sensors are noisy, delayed, and sometimes inconsistent. Multiple sensors (IMU, joint encoders, camera, depth) each provide partial, imperfect information about the true state. **Sensor fusion** combines these into a single, coherent state estimate.

The **Kalman filter** is the optimal linear estimator for Gaussian noise. It alternates two steps:

**Predict** (propagate state using the motion model):

$$\hat{\mathbf{x}}_{t|t-1} = F \hat{\mathbf{x}}_{t-1} + B \mathbf{u}_t, \quad P_{t|t-1} = F P_{t-1} F^\top + Q$$

**Update** (correct using the measurement $\mathbf{z}_t$):

$$K_t = P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1}$$

$$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_{t|t-1} + K_t(\mathbf{z}_t - H \hat{\mathbf{x}}_{t|t-1})$$

$$P_t = (I - K_t H) P_{t|t-1}$$

where $F$ is the state transition matrix, $H$ the observation matrix, $Q$ the process noise covariance, $R$ the measurement noise covariance, $P$ the error covariance, and $K_t$ the **Kalman gain** — a matrix that optimally balances trust in the prediction (from the model) against trust in the measurement (from the sensors).

The residual $\mathbf{z}_t - H\hat{\mathbf{x}}_{t|t-1}$ is called the **innovation**: the new information provided by the measurement. If the innovation is large, the model and the measurement disagree — potentially indicating a sensor failure, a slip event, or unexpected terrain.

**Extended Kalman Filter (EKF):** for nonlinear systems (quadruped kinematics are highly nonlinear), the EKF linearizes the transition and observation functions around the current estimate. More powerful, but requires computing Jacobians. Used in practice for full robot state estimation.

**LLM-assisted diagnostics:** the LLM planner monitors filter innovations and flags anomalies: if the velocity estimate and the visual odometry consistently disagree, the LLM can detect a potential sensor failure and alert the operator — an example of the cognitive layer supervising the physical layer in real time.

## Online Adaptation

Even with domain randomization and privileged learning, deployment conditions may fall outside the training distribution. **Online adaptation** adjusts the policy in real time based on observed behavior.

**Rapid Motor Adaptation (RMA; Kumar et al., 2021):** a secondary adaptation network is trained to estimate the hidden environment embedding from a short history of proprioceptive observations. This embedding is fed to the main policy, allowing it to adjust its gait for the current terrain without any explicit environment identification step.

**In-context adaptation:** the LLM planner can also adapt online — if the robot consistently fails to navigate a specific corridor segment, the planner updates its mental map and tries an alternative route, without retraining any weights.

## The Live Test

Selected teams demonstrate their full solution on the real Unitree Go2. The task:

> The system receives a natural language command: *"Go to the door at the end of the corridor, wait 5 seconds, then return."*
>
> It must execute autonomously — from voice input through LLM planning, VLM scene understanding, CLIP-based object localization, PPO locomotion, and Kalman-filtered state estimation — to physical movement and return.

Evaluated on: task completion, robustness to disturbances, recovery from failures, and communication back to the operator.

## Tracing the Full Stack

Every session has contributed a component to the final system:

| Component | Session | Role in the Live Test |
|---|---|---|
| Feature vectors and MLP | 1 | Proprioceptive state representation for all controllers |
| Loss functions, gradient descent | 2 | Training objective for all learned components |
| Regularization, validation | 3 | Generalization — ensures trained models work at test time |
| Self-supervised pretraining | 4 | Foundation models pretrained without task labels |
| Visual perception (CNN, ViT, DINO) | 5 | Scene analysis, obstacle detection |
| Language understanding (Transformer, GPT) | 6 | Processing voice commands into token sequences |
| Instruction following (RLHF, DPO) | 7 | Aligned LLM that follows commands reliably |
| Multimodal perception (CLIP, VLM) | 8 | Connecting visual scene to language commands |
| Agent architecture and tool use | 9 | High-level planning loop, tool call orchestration |
| Locomotion via RL (PPO) | 10 | Low-level gait control, motion skill library |
| Domain randomization, Kalman filter | 11 | Sim-to-real robustness, sensor fusion |

## What Works, What Doesn't

An honest assessment of the current state:

**Strengths:**
- Locomotion on varied terrain is now robust — commercial quadrupeds walk over stairs, gravel, and uneven ground reliably.
- Foundation models provide powerful zero-shot generalization to new scenes and commands.
- The modular architecture is maintainable — each component can be improved independently.

**Current failure modes:**
- **Hallucination:** LLMs generate plausible but false information. An LLM planner might confidently report "I see the exit on the left" when the camera shows a blank wall.
- **Distribution shift:** RL policies fail silently when conditions are far outside training. A policy trained on flat terrain with mild slopes may fail without warning on an unusually steep ramp.
- **Latency and compute:** running a full VLM + LLM planning loop at useful frequency requires significant onboard or edge compute — a real-world engineering constraint.
- **Long-horizon planning:** current LLM agents work well for tasks with <10 steps but struggle with multi-hour, multi-objective missions requiring consistent state tracking.

## Ethics and Open Questions

Deploying autonomous systems in shared human spaces raises genuine concerns:

- **Safety:** what happens when the robot makes a wrong decision near a person? Who is liable?
- **Privacy:** onboard cameras in public and private spaces — who owns the data? How is it retained?
- **Bias:** training data from specific environments may encode assumptions that fail for other populations or spaces.
- **Energy cost:** training a large foundation model consumes significant electricity. Deployment at scale compounds this.
- **Autonomy and control:** as systems become more capable, the question of meaningful human oversight becomes pressing. A robot that can override its operator's instructions for "safety reasons" is a policy choice, not a technical inevitability.

These are not problems to solve after the technology is built — they are design constraints from the beginning.

## Further Reading

- Kumar et al.: "RMA: Rapid Motor Adaptation for Legged Robots" (RSS 2021) — [arxiv.org/abs/2107.04034](https://arxiv.org/abs/2107.04034)
- Lee et al.: "Learning Quadrupedal Locomotion over Challenging Terrain" (*Science Robotics*, 2020)
- Tobin et al.: "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (IROS 2017)

---

Embodied AI — systems that perceive, reason, and act in the physical world — is one of the most active research frontiers. The methods in this series are the current foundations. The applications are just beginning: logistics, healthcare, infrastructure inspection, scientific exploration. The stack from math to motion is assembled. What you build with it is up to you.
