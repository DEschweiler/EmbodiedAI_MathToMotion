---
id: 010
title: Learning to Walk — Deep Reinforcement Learning
summary: Formulate locomotion as an MDP and train with PPO.
tags: [reinforcement-learning, locomotion, ppo]
learning_goals:
  - Define states, actions, rewards, and the MDP for locomotion.
  - Explain PPO’s clipped objective for stable updates.
  - Describe building a motion skill library for the agent.
---

### A Different Kind of Learning

Supervised learning requires input-output pairs. For locomotion, there are no "correct" motor commands. Instead, the dog discovers how to walk through **trial and error**: trying joint movements, receiving reward signals, and gradually improving.

### The Markov Decision Process

RL is formalized as an MDP defined by the tuple $(S, A, P, R, \gamma)$:

- **States** $S$: joint angles, angular velocities, body orientation, contact forces.
- **Actions** $A$: motor torques for each joint.
- **Transition function** $P(s' \mid s, a)$: the physics of the environment.
- **Reward function** $R(s, a, s')$: a scalar signal (e.g., +1 for forward progress, −1 for falling).
- **Discount factor** $\gamma \in [0, 1)$: controls how much future rewards matter.

The **policy** $\pi_\theta(a \mid s)$ maps states to a probability distribution over actions. The goal is to find $\pi_\theta$ that maximizes expected cumulative discounted reward:

$$J(\theta) = \mathbb{E_{\pi_\theta}}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1})\right]$$

### The Bellman Equation and Value Functions

The **value function** $V^\pi(s)$ measures the expected return from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E_\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s\right]$$

It satisfies the **Bellman equation**:

$$V^\pi(s) = \mathbb{E_{a \sim \pi}}\left[R(s, a) + \gamma \mathbb{E_{s'} \sim P}[V^\pi(s')]\right]$$

This recursive structure is the mathematical foundation of all RL algorithms: the value of a state equals the immediate reward plus the discounted value of the expected next state.

### PPO: Proximal Policy Optimization

Policy gradient methods update $\theta$ to increase the probability of actions that led to high rewards. PPO constrains how much the policy changes per update step, preventing instability:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between new and old policy, $\hat{A}_t$ is the advantage estimate (how much better an action was than expected), and $\epsilon$ is a small clipping threshold. The clipping prevents destructive large updates — essential for stable locomotion training.

In practice, hundreds of robot instances train in parallel in simulation, each exploring different strategies and sharing gradients. This massively parallelized training is what makes RL feasible for complex continuous control.

### The LLM as Reward Designer

Designing reward functions by hand is notoriously difficult. Specify "move forward" and the robot might learn to slide on its belly. Add "stay upright" and it might stand still. LLM-guided reward engineering offers a compelling alternative: describe desired behavior in natural language and have the LLM generate the mathematical objective. The LLM becomes the bridge between human intent and optimization targets.

### The Motion Library

The result: a set of robust skills — "walk forward", "walk sideways", "trot", "turn in place" — that the LLM planner from Section 9 calls as tools. Each skill is a trained policy handling millisecond-level motor coordination autonomously.

### → Next

We can train gait controllers in simulation. But simulation is clean, deterministic, and forgiving. Reality is none of those things. How do we bridge the gap?
