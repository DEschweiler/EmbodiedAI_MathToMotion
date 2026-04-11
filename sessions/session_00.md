---
id: 000
title: The Big Picture
summary: A quick tour of the full embodied-AI stack and how math, ML, and hardware connect.
tags: [overview, roadmap]
learning_goals:
  - See the full stack from math to motion.
  - Understand the three-layer architecture: cognitive, motor, physical.
  - Know the phases and topics covered across the series.
---

A four-legged robot stands in a hallway. Someone says: *"Go to the door and wait."* The robot turns, avoids a chair, identifies the door, stops, and reports: *"I've arrived."*

Nothing about this is magic. It's a stack of well-understood methods — linear algebra, optimization, neural networks, language models, control theory — wired together into a system. This series walks through every layer of that stack, from the mathematical foundations to the final integration on real hardware. The running example throughout: a Unitree Go2 robot.

<img src="figures/dog_navigation_scheme.png" alt="Dog Navigation scheme" style="display:left;margin:16px auto;max-width:80%;height:auto;">

## What Is Embodied AI?

Most AI systems live entirely in the digital world: they process text, generate images, answer questions. *Embodied AI* is different — it refers to systems that perceive and act in the physical world through a body. A robot, a drone, an autonomous vehicle: all of these are embodied agents.

Embodied AI is harder than pure digital AI for a fundamental reason: the real world does not wait, does not repeat, and does not forgive errors cleanly. A language model that gives a wrong answer can be corrected in the next turn. A robot that misjudges a step falls. This series takes embodied AI seriously — the goal is not a demo that works once, but a system that works reliably.

The Unitree Go2 is a commercially available quadruped robot: four legs, onboard compute, a suite of sensors (cameras, IMU, joint encoders). It is complex enough to be interesting and well-documented enough to be a concrete reference. Every concept introduced in this series is grounded in what that robot actually needs to do.

## Why Machine Learning?

Consider writing a program that distinguishes cats from dogs in photographs. You might try hand-crafted rules — "cats have pointed ears" — but exceptions immediately break them. For tasks involving perception, language, or complex decision-making, explicit programming doesn't scale. Machine learning offers an alternative: instead of specifying rules, provide examples and let the system discover patterns.

A concrete proof of concept makes the need for learning approaches tangible. An AND gate from two binary inputs can be implemented as a weighted sum with a threshold:

$$y = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Setting $w_1 = w_2 = 1$, $b = -1.5$ gives a perfect AND gate — trivially hand-crafted. Now try the same for a 224×224 RGB image (over 150,000 input values). Hand-crafting weights becomes impossible. We need an automated way to find them, and that's what learning algorithms provide.

The same argument applies to locomotion. Writing explicit rules for how a four-legged robot should place its feet on uneven terrain is intractable. But a reinforcement learning agent, given a reward for not falling, can discover effective gait strategies through millions of simulated trials.

## The Hierarchy: AI, ML, Deep Learning

*Artificial Intelligence* is the broadest term — any computational system solving tasks that typically require human intelligence. *Machine Learning* narrows this to systems that learn from data rather than being explicitly programmed. *Deep Learning* narrows further to models based on layered function compositions.

```
Artificial Intelligence
└── Machine Learning
    └── Deep Learning
        ├── Convolutional Neural Networks (vision)
        ├── Transformers (language, vision, multimodal)
        └── Reinforcement Learning (locomotion, planning)
```

An important clarification: a neural network doesn't "know" what a cat is. It has learned to map certain numerical patterns to a label. Internal representations are statistical compressions of training data, not knowledge databases. This distinction matters enormously when discussing LLMs and their capabilities later.

## The Robot Dog as a System

The Unitree Go2 serves as a concrete reference system throughout. Its architecture maps onto three layers:

- **Cognitive layer:** An LLM-based planner that interprets natural language commands and decomposes them into subtasks. (→ Sessions 6–9)
- **Motor layer:** ML models that have learned movement strategies — walking, trotting, obstacle avoidance. (→ Sessions 1–3, 10)
- **Physical layer:** Control theory and mathematics that translate joint angles into motor currents. (→ Classical engineering, Session 11)

Every method introduced in this series plugs into one of these layers. Keeping the architecture in mind helps orient each new concept: *where does this go, and what does it interface with?*

## Prerequisites

This series assumes comfort with the following:

- **Linear algebra:** vectors, matrices, dot products, matrix multiplication. You should not be surprised by the notation $\mathbf{y} = W\mathbf{x} + \mathbf{b}$.
- **Calculus:** derivatives and the chain rule. Backpropagation is just the chain rule applied repeatedly.
- **Basic probability:** expectations, probability distributions, conditional probability.
- **Programming:** examples and pseudocode are Python-flavored. Familiarity with NumPy helps.

## Three Phases, Twelve Sessions

**Phase I – Foundations (Sessions 1–3):** Data representations, neural networks, optimization, and the practicalities of training.

**Phase II – Building Intelligence (Sessions 4–9):** Foundation models, CNNs, vision transformers, large language models, RLHF, multimodal systems, and agent architectures.

**Phase III – Integration & Reality (Sessions 10–11):** Reinforcement learning for locomotion, sim-to-real transfer, and the live test on real hardware.

| # | Title | Phase | Key Concepts |
|---|-------|-------|--------------|
| 0 | The Big Picture | — | Embodied AI, system architecture |
| 1 | Data, Representations, and the First Model | I | Feature spaces, perceptron, MLP |
| 2 | Learning as Optimization | I | Loss functions, gradient descent, backprop |
| 3 | Training in Practice | I | Overfitting, regularization, validation |
| 4 | The Data Problem: Annotation and Self-Supervision | II | Self-supervised learning, foundation models |
| 5 | Seeing: CNNs and Vision Transformers | II | Convolutions, receptive fields, ViT |
| 6 | Listening: From N-Grams to Language Models | II | Tokenization, transformers, GPT |
| 7 | Following Instructions: RLHF and Prompting | II | Instruction tuning, RLHF, prompt engineering |
| 8 | Seeing *and* Listening: Multimodal Models | II | CLIP, contrastive training, VLMs |
| 9 | The Dog Gets a Brain: Agents and Tool Use | II | Agent loop, tool calls, planning |
| 10 | Learning to Walk: Deep Reinforcement Learning | III | MDP, PPO, skill library |
| 11 | From Simulation to Reality and the Live Test | III | Domain randomization, sim-to-real, full stack |

Each session ends with a transition to the next — they are designed to be read in sequence.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "But what is a neural network?" — [youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk) — best visual intuition for the foundations
- Andrej Karpathy: "Neural Networks: Zero to Hero" — [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html) — builds everything from scratch in Python
- fast.ai: Practical Deep Learning for Coders — [course.fast.ai](https://course.fast.ai) — top-down, very hands-on

**Go deeper** *(technical references)*
- LeCun, Bengio & Hinton: "Deep Learning" (*Nature*, 2015) — accessible high-level overview by the field's founders
- Goodfellow, Bengio & Courville: *Deep Learning* (MIT Press, 2016) — [deeplearningbook.org](https://www.deeplearningbook.org) — the standard reference textbook

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. What distinguishes *embodied* AI from purely digital AI systems like a language model? Give a concrete example of each.
2. A robot dog receives the command "sit." In which of the three layers (cognitive, motor, physical) is each of the following handled: (a) parsing the word "sit," (b) computing the correct joint angles, (c) sending current to the motors?
3. Why is hand-crafting weights for a 224×224 RGB image classifier infeasible, even though hand-crafting them for a 2-input AND gate is trivial?
4. A neural network classifies cats vs. dogs with 97% accuracy. A colleague says: "The model now *understands* what a cat is." Do you agree? Justify your answer.
5. *(Preliminary)* Looking at the session table: which two sessions do you expect to be most mathematically demanding, and why?

---

### → Next

We've established why ML is necessary, how the terms relate, and how the robot dog's architecture is structured. But before any model can learn, raw data — pixels, sensor readings, text — must be converted into numbers. And once the data is numerical, we need a model that can do something with it. Session 1 covers both: how data becomes math, and the first neural network that operates on it.
