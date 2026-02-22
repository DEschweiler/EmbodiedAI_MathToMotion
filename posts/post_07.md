---
id: 007
title: Following Instructions — RLHF and Prompting
summary: Turn a base LLM into a helpful assistant with instruction tuning and RLHF.
tags: [rlhf, prompting, llm]
learning_goals:
  - Explain why base LLMs need alignment.
  - Describe instruction tuning and RLHF at a high level.
  - Show prompt engineering patterns for better control.
---

### From Text Generator to Assistant

A base GPT model generates plausible continuations. Ask "What is the capital of France?" and it might output "What is the capital of Germany? What is the capital of Spain?" — because in its training data, questions are often followed by more questions, not answers. The model has no concept of "being helpful."

### Instruction Tuning

First step: supervised fine-tuning on (instruction, desired response) pairs. "Name the capital of France." → "The capital of France is Paris." This teaches the model what helpful responses *look like* — a specific format, not just plausible text.

### RLHF — Reinforcement Learning from Human Feedback

Second step: human raters compare multiple model responses (which is more helpful, accurate, less harmful?). Their rankings train a **reward model** $R_\phi$ that predicts human preferences:

$$R_\phi(\text{prompt}, \text{response}) \rightarrow \text{scalar score}$$

The LLM is then fine-tuned to maximize this reward using PPO (Proximal Policy Optimization):

$$\mathcal{L_{\text{RLHF}}} = -\mathbb{E}\left[R_\phi(\text{prompt}, \text{response})\right] + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

The KL-divergence term prevents the model from deviating too far from the pretrained distribution — otherwise it might learn to "game" the reward model by producing adversarial outputs that score high but are nonsensical. PPO is the same algorithm that appears in Post 10 for training locomotion policies — the connection is not accidental.

### Prompt Engineering

Even after RLHF, output quality depends heavily on how the input is framed:

- **System prompt:** Defines the model's role and behavior constraints.
- **Few-shot examples:** Providing input-output examples in the prompt to demonstrate the desired format.
- **Chain-of-thought:** Explicitly asking the model to reason step by step before answering, which dramatically improves performance on complex tasks.

On the robot dog: the system prompt defines the LLM's role — *"You are a robot dog planner. You receive voice commands and decompose them into a sequence of movement commands. Respond in JSON format."*

### LLM-Guided Reward Engineering

Instead of hand-coding a reward function for gait training, describe the desired behavior in natural language — "walk energy-efficiently but maintain stability" — and have the LLM generate the mathematical objective function. The LLM acts as a "high-level compiler" translating human intent into optimization targets. This bridges the cognitive and motor layers of the robot architecture.

### → Next

The robot can now process language and analyze images — but separately. "Go to the red chair" requires understanding "red chair" linguistically *and* identifying it in the camera feed simultaneously. This calls for multimodal models that connect text and images in a shared representation space.
