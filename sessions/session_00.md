---
id: 000
title: The Big Picture
summary: A full tour of the embodied-AI stack — what it is, why it is hard, and how math, machine learning, and hardware connect across the series.
tags: [overview, roadmap, embodied-ai]
duration: 90 min
learning_goals:
  - See the full stack from math to motion and the three-layer architecture (cognitive, motor, physical).
  - Understand the perception–action loop and why embodiment makes AI harder.
  - Place the major learning paradigms and the history of the field in context.
  - Trace a single natural-language command through the whole system.
  - Know the phases, topics, and prerequisites for the series.
---

A four-legged robot stands in a hallway. Someone says: *"Go to the door and wait."* The robot turns, avoids a chair, identifies the door, stops, and reports: *"I've arrived."*

Nothing about this is magic. It's a stack of well-understood methods — linear algebra, optimization, neural networks, language models, control theory — wired together into a system. This series walks through every layer of that stack, from the mathematical foundations to the final integration on real hardware. The running example throughout: a Unitree Go2 robot.

<img src="figures/dog_navigation_scheme.png" alt="Dog Navigation scheme" style="display:left;margin:16px auto;max-width:80%;height:auto;">

This first session is a map, not a derivation. By the end you should know *what* the major pieces are, *why* each is needed, and *where* in the series each is built. Nothing here is proven; everything here is promised.

---

## What Is Embodied AI?

Most AI systems live entirely in the digital world: they process text, generate images, answer questions. *Embodied AI* is different — it refers to systems that perceive and act in the physical world through a body. A robot, a drone, an autonomous vehicle, a robotic arm on an assembly line: all of these are embodied agents.

Embodied AI is harder than pure digital AI for a fundamental reason: the real world does not wait, does not repeat, and does not forgive errors cleanly. A language model that gives a wrong answer can be corrected in the next turn. A robot that misjudges a step falls — and a fall may break the hardware, not just the metric. This series takes embodied AI seriously: the goal is not a demo that works once, but a system that works reliably.

There is a deep and counterintuitive lesson here, known as **Moravec's paradox**: tasks that are hard for humans (chess, integral calculus, recalling facts) turned out to be relatively easy for computers, while tasks that are effortless for humans (walking over uneven ground, picking up a cup, recognizing a face) turned out to be extraordinarily hard. A toddler's motor control still exceeds our best robots in robustness. Embodiment is where AI meets its oldest, most stubborn problems.

The Unitree Go2 is a commercially available quadruped robot: four legs, onboard compute, and a suite of sensors (cameras, IMU, joint encoders). It is complex enough to be interesting and well-documented enough to be a concrete reference. Every concept introduced in this series is grounded in what that robot actually needs to do.

---

## The Perception–Action Loop

Every embodied agent runs the same closed loop, over and over, many times per second:

```
        ┌──────────── world ────────────┐
        │                               │
     sensors                         actuators
        │                               │
        ▼                               ▲
   perception  ──►  decision/plan  ──►  control
```

The agent **senses** the world, **perceives** structure in the raw signals (where is the door? is the floor flat?), **decides** what to do, and **acts** through its motors — which changes the world, producing new sensations. The loop never stops while the robot is on.

Two properties of this loop make embodied AI distinctive. First, it is *closed*: the agent's own actions change what it will sense next, so mistakes compound rather than reset. Second, it operates under **partial observability** — the sensors never reveal the full state of the world. A camera cannot see around a corner; an IMU drifts. Much of the engineering in later sessions exists to estimate what the sensors cannot directly measure.

---

## Why Machine Learning?

Consider writing a program that distinguishes cats from dogs in photographs. You might try hand-crafted rules — "cats have pointed ears" — but exceptions immediately break them. For tasks involving perception, language, or complex decision-making, explicit programming doesn't scale. Machine learning offers an alternative: instead of specifying rules, provide examples and let the system discover patterns.

A concrete proof of concept makes the need for learning tangible. An AND gate from two binary inputs can be implemented as a weighted sum with a threshold:

$$y = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Setting $w_1 = w_2 = 1$, $b = -1.5$ gives a perfect AND gate — trivially hand-crafted. Now try the same for a 224×224 RGB image (over 150,000 input values). Hand-crafting weights becomes impossible. We need an automated way to find them, and that's what learning algorithms provide.

The same argument applies to locomotion. Writing explicit rules for how a four-legged robot should place its feet on uneven terrain is intractable. But a reinforcement learning agent, given a reward for not falling, can discover effective gait strategies through millions of simulated trials.

### Why Now?

Machine learning is decades old, yet the current wave is recent. Three ingredients had to arrive together: **data** (the internet made billion-scale datasets available), **compute** (GPUs made training large networks affordable), and **algorithms** (backpropagation, attention, and stable training recipes). Remove any one and the others stall. This is why a field that simmered for fifty years boiled over in the last fifteen.

---

## A Short History: From Rules to Learning

The series builds modern methods, but it helps to know the arc that produced them.

- **Symbolic AI (1950s–1980s).** Intelligence as logic and search. Expert systems encoded human rules by hand. Powerful in narrow domains, brittle everywhere else — they could not handle the messiness of perception.
- **The connectionist revival (1980s–1990s).** Neural networks and backpropagation offered learning from data, but limited compute and small datasets held them back.
- **The deep learning revolution (2012–).** A deep network (AlexNet) crushed the ImageNet competition, and the field pivoted to learned representations. Vision, speech, and translation were transformed within a few years.
- **Foundation models (2018–).** Models trained once on enormous data, then adapted to many tasks — BERT, GPT, CLIP. Capability began to scale with size and data in predictable ways.
- **The embodied turn (now).** Those same foundation models are being connected to bodies: language models that plan, vision models that perceive, learned controllers that move. This series sits squarely in that turn.

---

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

---

## Three Ways to Learn

Almost every method in this series is one of three learning paradigms, distinguished by what kind of feedback the model receives:

- **Supervised learning.** The data comes with correct answers (labels). The model learns to reproduce them. *Example:* classify a camera image as "door" or "wall." Covered in Sessions 1–3, 5, 8.
- **Self-supervised learning.** Labels are manufactured from the data itself — hide part of the input and predict it. This is how foundation models and large language models are trained without armies of annotators. Covered in Sessions 4, 6.
- **Reinforcement learning.** No labels at all; the agent acts and receives a scalar *reward*, then adjusts to earn more reward over time. *Example:* learn to walk by being rewarded for forward progress and penalized for falling. Covered in Sessions 7, 10.

Keep this taxonomy in mind: whenever a new method appears, asking "which kind of feedback does it use?" immediately clarifies what it can and cannot do.

---

## The Robot Dog as a System

The Unitree Go2 serves as a concrete reference system throughout. Its architecture maps onto three layers:

- **Cognitive layer:** An LLM-based planner that interprets natural language commands and decomposes them into subtasks. (→ Sessions 6–9)
- **Motor layer:** ML models that have learned movement strategies — walking, trotting, obstacle avoidance. (→ Sessions 1–3, 10)
- **Physical layer:** Control theory and mathematics that translate joint angles into motor currents. (→ Classical engineering, Session 11)

The layers differ not just in function but in *timescale*. The cognitive layer thinks in seconds ("plan a route to the door"). The motor layer reacts in tens of milliseconds ("adjust the gait"). The physical layer runs in microseconds ("hold this current"). A robust system respects this hierarchy: slow deliberation on top, fast reflexes underneath. Every method introduced in this series plugs into one of these layers. Keeping the architecture in mind helps orient each new concept: *where does this go, and what does it interface with?*

---

## Walking the Stack: "Go to the door and wait"

To see how the pieces connect, trace the opening command through the whole system:

1. **Speech → text.** The spoken command is transcribed into a string. (Adjacent to Session 6.)
2. **Language understanding & planning.** A language-model agent parses the instruction and produces a plan: *locate door → navigate to it → stop → report.* (Sessions 6, 7, 9.)
3. **Perception.** The camera stream is processed to find the door and any obstacles, fusing vision with language to ground "the door" in the image. (Sessions 5, 8.)
4. **Motor control.** A learned locomotion policy turns "move toward that point" into joint targets, stepping over and around obstacles. (Sessions 1–3, 10.)
5. **Low-level control.** Joint targets become motor currents through classical control, with state estimation correcting for sensor drift. (Session 11.)
6. **Feedback.** New sensor readings flow back to every layer, closing the loop until the task is done.

Every later session is, in effect, a deep dive into one stage of this single sentence.

---

## Challenges Unique to Embodied AI

Why devote a whole series to this? Because embodiment adds problems that purely digital AI never faces:

- **Real-time constraints.** A controller that needs 200 ms to decide is useless if the robot falls in 100 ms.
- **Safety and irreversibility.** A wrong action can damage hardware or people. There is no "undo."
- **Partial observability.** The true state of the world is never fully visible; it must be estimated.
- **The sim-to-real gap.** Most learning happens in simulation because reality is slow and dangerous — but a policy trained in simulation rarely transfers perfectly to the real robot. (Session 11.)
- **Long horizons.** "Tidy the room" is thousands of coordinated actions, not one decision.
- **Data scarcity.** You cannot download a billion examples of *your* robot falling; physical data is expensive. This is why simulation and self-supervision matter so much.

These challenges are the through-line of the series; each session chips away at one or more of them.

---

## Prerequisites

This series assumes comfort with the following:

- **Linear algebra:** vectors, matrices, dot products, matrix multiplication. You should not be surprised by the notation $\mathbf{y} = W\mathbf{x} + \mathbf{b}$.
- **Calculus:** derivatives and the chain rule. Backpropagation (Session 2) is just the chain rule applied repeatedly.
- **Basic probability:** expectations, probability distributions, conditional probability.
- **Programming:** examples and pseudocode are Python-flavored. Familiarity with NumPy helps.

If any of these feel shaky, that is fine — the early sessions reintroduce what they need as they need it. The expectation is familiarity, not fluency.

---

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
- Moravec: *Mind Children* (1988) — the origin of Moravec's paradox, on why the easy things are hard

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. What distinguishes *embodied* AI from purely digital AI systems like a language model? Give a concrete example of each.
2. State Moravec's paradox in your own words and give one example of an "easy-for-humans, hard-for-robots" task on the Go2.
3. Draw the perception–action loop and label each stage. Why does *partial observability* make the loop harder?
4. A robot dog receives the command "sit." In which of the three layers (cognitive, motor, physical) is each handled: (a) parsing the word "sit," (b) computing the correct joint angles, (c) sending current to the motors?
5. Name the three learning paradigms and match each to a robot capability from this session.
6. Why is hand-crafting weights for a 224×224 RGB image classifier infeasible, even though hand-crafting them for a 2-input AND gate is trivial?
7. A neural network classifies cats vs. dogs with 97% accuracy. A colleague says: "The model now *understands* what a cat is." Do you agree? Justify your answer.
8. Pick two of the "challenges unique to embodied AI" and explain why a purely digital chatbot never has to deal with them.

---

### → Next

We've established why ML is necessary, how the terms relate, and how the robot dog's architecture is structured. But before any model can learn, raw data — pixels, sensor readings, text — must be converted into numbers. And once the data is numerical, we need a model that can do something with it. Session 1 covers both: how data becomes math, and the first neural network that operates on it.
