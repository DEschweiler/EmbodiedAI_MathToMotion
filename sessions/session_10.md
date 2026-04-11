---
id: 010
title: Learning to Walk — Deep Reinforcement Learning
summary: Formulate locomotion as an MDP and train with PPO.
tags: [reinforcement-learning, locomotion, ppo]
learning_goals:
  - Define states, actions, rewards, and the MDP formulation for locomotion.
  - Understand value functions, the Bellman equation, and advantage estimation.
  - Explain PPO's clipped objective for stable policy updates.
  - Understand the actor-critic architecture.
  - Describe reward design challenges and LLM-guided reward engineering.
---

Sessions 1–9 gave the robot dog the ability to see, understand language, and plan. But we have been assuming it can *move*. Locomotion — walking, trotting, turning, recovering from stumbles — is a fundamentally different problem from supervised learning: there are no labeled examples of "correct walking." The robot must discover how to walk through millions of trials in simulation. This session introduces reinforcement learning and the algorithm that trains the locomotion controller.

## A Different Kind of Learning

In supervised learning, every training example has a ground-truth label. The loss is computed per example. Learning is stable and well-understood.

In reinforcement learning, no ground truth exists. The agent takes actions, receives a reward signal, and must infer which actions caused which rewards — often with a significant delay. If the robot falls after 3 seconds of walking, which of the 300 motor commands over those 3 seconds was the critical mistake?

This is the **credit assignment problem**: attributing reward to the right actions across time. Solving it efficiently is the central challenge of RL.

## The Markov Decision Process

RL is formalized as a **Markov Decision Process (MDP)**, defined by the tuple $(S, A, P, R, \gamma)$:

- **State space** $S$: everything the agent knows about its current situation.
  - For the Unitree Go2: 12 joint angles, 12 joint velocities, 12 motor torques, 3-axis linear acceleration, 3-axis angular velocity, 3D orientation (quaternion) ≈ 48 values.
  - Plus task-specific context: target velocity, elapsed time, contact forces (4 feet × 3 axes = 12 values).
- **Action space** $A$: the set of actions the agent can take.
  - For locomotion: 12 target joint positions (one per joint). The low-level PD controller translates these to motor currents at 100 Hz.
- **Transition function** $P(s' \mid s, a)$: the physics of the environment (gravity, friction, joint limits). Unknown to the agent — it must be estimated from experience.
- **Reward function** $R(s, a, s') \in \mathbb{R}$: a scalar feedback signal after each step.
- **Discount factor** $\gamma \in [0, 1)$: future rewards are worth less than immediate ones. Typical $\gamma = 0.99$.

**The Markov property:** the future depends only on the current state, not on the history. This is an approximation for physical systems (it holds exactly only if the state is fully observable), but it is computationally essential.

The **policy** $\pi_\theta(a \mid s)$ maps states to a distribution over actions. In continuous action spaces, the policy is typically a Gaussian: the neural network outputs mean $\mu_\theta(s)$ and (log) standard deviation $\sigma_\theta(s)$ for each action dimension, and actions are sampled:

$$a_t \sim \mathcal{N}(\mu_\theta(s_t), \text{diag}(\sigma_\theta(s_t)^2))$$

## Reward Design

The reward function encodes what "good walking" means. Designing it is one of the most important and difficult steps in RL. Too sparse (reward only when the robot reaches a goal) → the robot almost never receives feedback, and learning is extremely slow. Too dense (reward every tiny improvement) → the robot may find unexpected ways to optimize the reward without achieving the intended behavior.

A typical locomotion reward combines several terms:

$$R(s, a, s') = r_\text{velocity} + r_\text{stability} + r_\text{energy} + r_\text{contact} - r_\text{penalty}$$

- $r_\text{velocity} = \exp(-\|v_\text{xy} - v_\text{target}\|^2 / \sigma^2)$: reward for matching target forward velocity.
- $r_\text{stability} = -\|v_z\|^2 - \|\omega_{xy}\|^2$: penalize vertical velocity and rolling/pitching.
- $r_\text{energy} = -\|\tau \cdot \dot{q}\|$: penalize motor torque × joint velocity (energy consumption).
- $r_\text{contact}$: reward feet making contact in the correct pattern for the gait.
- $r_\text{penalty}$: hard penalty for falling (body height below threshold).

The relative weights of these terms determine what gait emerges. High energy penalty → slow, efficient crawl. High velocity reward → fast but potentially unstable trot. Finding the right balance requires iteration.

**Reward hacking:** a notorious RL failure mode where the agent finds an unexpected way to maximize the reward without achieving the intended behavior. Classic example: tell the robot to maximize forward velocity, and it learns to slide on its side — technically high forward velocity, not useful locomotion. Every reward term is a constraint that shapes the solution space; missing a term opens a loophole.

## Value Functions and the Bellman Equation

To learn efficiently, the agent needs to estimate the long-term consequences of its actions — not just the immediate reward.

The **state-value function** $V^\pi(s)$ measures expected cumulative discounted reward from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s\right]$$

The **action-value function** (Q-function) $Q^\pi(s, a)$ measures expected return for taking action $a$ in state $s$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s, a_0 = a\right]$$

Both satisfy recursive **Bellman equations**:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[R(s, a) + \gamma \mathbb{E}_{s' \sim P}[V^\pi(s')]\right]$$

$$Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim P, a' \sim \pi}[Q^\pi(s', a')]$$

These recursions are fundamental: the value of a state equals the immediate reward plus the discounted value of the expected successor state. All temporal-difference RL algorithms are, at their core, methods for estimating and using these equations.

The **advantage function** measures how much better action $a$ is than the average action in state $s$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

If $A > 0$, the action is better than average and should be made more likely. If $A < 0$, it's worse than average and should be suppressed. Advantage estimates are the learning signal in policy gradient methods.

## Actor-Critic Architecture

PPO uses an **actor-critic** architecture: two networks sharing parameters or running in parallel.

- **Actor** $\pi_\theta(a \mid s)$: the policy network. Outputs action distribution parameters. Trained to maximize expected advantage.
- **Critic** $V_\phi(s)$: the value network. Outputs a scalar estimate of $V^\pi(s)$. Trained to minimize the TD error.

Both are MLPs processing the same state vector. Shared layers extract common features; separate heads output the action distribution and the value estimate respectively.

### Generalized Advantage Estimation (GAE)

Rather than using the raw TD error as the advantage estimate (high variance) or the full return (high bias), **GAE** (Schulman et al., 2016) computes a weighted mixture:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error and $\lambda \in [0, 1]$ controls the bias-variance tradeoff. $\lambda = 0$ gives pure TD (low variance, high bias); $\lambda = 1$ gives full Monte Carlo returns (high variance, low bias). Typical $\lambda = 0.95$.

## PPO: Proximal Policy Optimization

**Policy gradient** methods update $\theta$ to increase the probability of actions that led to high advantage. The naive update is:

$$\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right]$$

The problem: large gradient steps can catastrophically change the policy, causing it to diverge. **PPO** (Schulman et al., 2017) constrains updates via a clipped surrogate objective:

$$\mathcal{L}_\text{PPO}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ measures how much the new policy differs from the old one. The clip prevents $r_t$ from moving outside $(1-\epsilon, 1+\epsilon)$, typically $\epsilon = 0.2$.

**Intuition:** If an action had positive advantage and the new policy assigns it more probability ($r_t > 1$), increase it — but only up to the clip threshold, after which no additional gradient flows. This prevents the policy from over-committing to any single update. The same mechanism appeared in Session 7's RLHF loss; the mathematical structure is identical.

**Parallel training:** hundreds of simulated robot instances collect experience simultaneously, all feeding into the same policy update. This parallelism is essential — locomotion training requires billions of simulation steps, and even a fast physics simulator takes hours with a single instance.

## The LLM as Reward Designer

Designing reward functions by hand is labor-intensive and error-prone. An exciting recent direction: describe the desired behavior in natural language and use an LLM to generate the reward function automatically.

> "Walk energy-efficiently at 0.5 m/s on flat terrain. Maintain a stable body posture. Prefer smooth leg trajectories over jerky ones."

The LLM translates this into Python code implementing $R(s, a, s')$, which is then executed in the simulation loop. Errors or unexpected behaviors can be described back to the LLM, which revises the reward. This closes the reward engineering loop and bridges the cognitive and motor layers: the LLM planner not only issues commands but can also shape the learning objectives of the motor controller.

## The Motion Skill Library

After training, the result is a library of robust, specialized policies:

| Skill | Description | Input |
|---|---|---|
| `walk_forward` | Stable quadruped gait | Target speed [m/s] |
| `trot` | Energy-efficient faster gait | Target speed [m/s] |
| `turn_in_place` | Rotate body about Z axis | Target angle [rad] |
| `side_step` | Lateral movement | Direction, speed |
| `stand_still` | Maintain stable stance | — |
| `recover` | Stand up from fallen position | — |

Each skill is a trained policy that handles millisecond-level motor coordination autonomously. The LLM planner from Session 9 calls them as tools via tool use — "walk_forward(speed=0.4)" — without needing to know anything about joint angles or motor torques.

## Further Reading

- Schulman et al.: "Proximal Policy Optimization Algorithms" (2017) — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman et al.: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (ICLR 2016) — [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)
- Kumar et al.: "Learning Agile Locomotion via Adversarial Training" — for adversarial reward shaping in locomotion
- Ma et al.: "EurekaHuman-Level Reward Design via Coding Large Language Models" (ICLR 2024) — [arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)

---

### → Next

We can train gait controllers in simulation. But simulation is clean, deterministic, and perfectly rendered — reality is none of those things. Real surfaces have unexpected friction, sensors are noisy, and motor delays vary. Session 11 covers how to close the sim-to-real gap and runs the full stack on actual hardware.
