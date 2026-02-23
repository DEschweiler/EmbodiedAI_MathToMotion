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

<img src="figures/dog_navigation_scheme.png" alt="Dog Navigation scheme" style="display:block;margin:16px auto;max-width:60%;height:auto;">

## Why Machine Learning?

Consider writing a program that distinguishes cats from dogs in photographs. You might try hand-crafted rules — "cats have pointed ears" — but exceptions immediately break them. For tasks involving perception, language, or complex decision-making, explicit programming doesn't scale. Machine learning offers an alternative: instead of specifying rules, provide examples and let the system discover patterns.

A concrete proof of concept makes the need for learning approaches tangible. An AND gate from two binary inputs can be implemented as a weighted sum with a threshold:

$$y = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Setting $w_1 = w_2 = 1$, $b = -1.5$ gives a perfect AND gate — trivially hand-crafted. Now try the same for a 224×224 RGB image (over 150,000 input values). Hand-crafting weights becomes impossible. We need an automated way to find them, and that's what learning algorithms provide.

## The Hierarchy: AI, ML, Deep Learning

*Artificial Intelligence* is the broadest term — any computational system solving tasks that typically require human intelligence. *Machine Learning* narrows this to systems that learn from data rather than being explicitly programmed. *Deep Learning* narrows further to models based on layered function compositions. Most of what this series covers falls under deep learning, but the foundational ML ideas apply throughout.

An important clarification: a neural network doesn't "know" what a cat is. It has learned to map certain numerical patterns to a label. Internal representations are statistical compressions of training data, not knowledge databases. This distinction matters enormously when discussing LLMs and their capabilities later.

## The Robot Dog as a System

The Unitree Go2 serves as a concrete reference system throughout. Its architecture maps onto three layers:

- **Cognitive layer:** An LLM-based planner that interprets natural language commands and decomposes them into subtasks. (→ Sections 6–9)
- **Motor layer:** ML models that have learned movement strategies — walking, trotting, obstacle avoidance. (→ Sections 1–3, 10)
- **Physical layer:** Control theory and mathematics that translate joint angles into motor currents. (→ Classical engineering)

Every method introduced in this series plugs into one of these layers.

## Three Phases, Twelve Sections

**Phase I – Foundations (Section 1–3):** Data representations, neural networks, optimization, and the practicalities of training. The mathematical tools for everything that follows.

**Phase II – Building Intelligence (Section 4–9):** Foundation models, CNNs, vision transformers, large language models, RLHF, multimodal systems, and agent architectures. The building blocks that give the robot perception, language, and reasoning.

**Phase III – Integration & Reality (Section 10–11):** Reinforcement learning for locomotion, sim-to-real transfer, and the live test on real hardware.

| # | Title | Phase |
|---|-------|-------|
| 0 | The Big Picture | — |
| 1 | Data, Representations, and the First Model | I |
| 2 | Learning as Optimization | I |
| 3 | Training in Practice | I |
| 4 | The Data Problem: Annotation and Self-Supervision | II |
| 5 | Seeing: CNNs and Vision Transformers | II |
| 6 | Reading: From N-Grams to Language Models | II |
| 7 | Following Instructions: RLHF and Prompting | II |
| 8 | Seeing *and* Reading: Multimodal Models | II |
| 9 | The Dog Gets a Brain: Agents and Tool Use | II |
| 10 | Learning to Walk: Deep Reinforcement Learning | III |
| 11 | From Simulation to Reality and the Live Test | III |

Each section ends with a transition to the next — they are designed to be read in sequence.

### → Next

We've established why ML is necessary, how the terms relate, and how the robot dog's architecture is structured. But before any model can learn, raw data — pixels, sensor readings, text — must be converted into numbers. And once the data is numerical, we need a model that can do something with it. Section 1 covers both: how data becomes math, and the first neural network that operates on it.
