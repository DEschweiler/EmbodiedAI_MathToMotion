---
id: 011
title: From Simulation to Reality — and the Live Test
summary: Close the sim-to-real gap, estimate state, and run the full stack.
tags: [sim-to-real, robustness, state-estimation]
learning_goals:
  - Explain domain randomization for robustness.
  - Describe Kalman filter basics for noisy sensors.
  - Map the full pipeline from command to movement.
---

### The Sim-to-Real Gap

In simulation: uniform friction, perfect sensors, clean physics. In reality: varying friction, sensor noise, changing lighting, unforeseen obstacles. A model trained only in simulation has overfit to the simulator's assumptions and often fails catastrophically in the real world.

### Domain Randomization

Make the simulation intentionally messy. Randomize friction coefficients, add sensor noise, vary lighting, place random obstacles. If the model performs under all these variations, reality becomes "just another variation" within the training distribution. This is data augmentation applied to physics rather than pixels.

### State Estimation: The Kalman Filter

The robot never knows its exact state — sensors are noisy. The **Kalman filter** fuses predictions from a motion model with noisy measurements to produce an optimal state estimate. In two steps:

**Predict** (using the motion model):

$$\hat{\mathbf{x_{t|t-1}}} = F \hat{\mathbf{x_{t-1}}} + B \mathbf{u_t}, \quad P_{t|t-1} = F P_{t-1} F^\top + Q$$

**Update** (incorporating the measurement $\mathbf{z}_t$):

$$K_t = P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1}$$

$$\hat{\mathbf{x_t}} = \hat{\mathbf{x_{t|t-1}}} + K_t(\mathbf{z_t} - H \hat{\mathbf{x_{t|t-1}}})$$

$$P_t = (I - K_t H) P_{t|t-1}$$

where $F$ is the state transition matrix, $H$ the observation matrix, $Q$ the process noise covariance, $R$ the measurement noise covariance, and $K_t$ the **Kalman gain** — which optimally balances trust in the prediction vs. the measurement. This is classical Bayesian inference with Gaussians, connecting probability theory from the engineering mathematics curriculum directly to the ML stack.

The LLM can serve as a *system diagnostician*: monitoring filter residuals $(\mathbf{z_t} - H\hat{\mathbf{x_{t|t-1}}})$ and flagging anomalies — "Warning: large position estimate deviation. Possible cause: slippery surface."

### The Live Test

Selected teams demonstrate their full solution on the real Unitree Go2. The task: the system receives a natural language command (e.g., "go to the door at the end of the corridor") and executes it autonomously — from LLM planning through navigation to physical movement.

### Tracing the Full Stack

Every post has contributed a component:

| Component | Post |
|-----------|------|
| Data as feature vectors | 1 |
| MLP architecture | 1 |
| Loss functions and gradient descent | 2 |
| Training, regularization, diagnostics | 3 |
| Self-supervised pretraining | 4 |
| Visual perception (CNN, ViT, DINO) | 5 |
| Language understanding (Transformer, GPT) | 6 |
| Instruction following (RLHF, prompting) | 7 |
| Multimodal scene understanding (CLIP, VLM) | 8 |
| Agent architecture and planning | 9 |
| Locomotion via RL (PPO) | 10 |
| Sim-to-real robustness, state estimation | 11 |

### What Works, What Doesn't

An honest assessment: LLMs hallucinate (generating plausible but false information). Vision models can be fooled by adversarial inputs. RL policies fail in out-of-distribution situations. The system is impressive but far from infallible — understanding limitations is as important as understanding capabilities.

### Ethics and Open Questions

Safety of autonomous systems, bias in training data, privacy in data collection, environmental cost of large-scale training, and the fundamental question: how close are we to AGI — and is that even a meaningful goal?

Embodied AI — systems that perceive, reason, and act in the physical world — is one of the most active research frontiers. The methods in this series are the foundations. The applications are just beginning.
