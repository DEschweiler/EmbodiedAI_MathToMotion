---
id: 007
title: Following Instructions — RLHF and Prompting
summary: Turn a base LLM into a helpful assistant with instruction tuning and RLHF.
tags: [rlhf, prompting, llm]
learning_goals:
  - Explain why base LLMs need alignment and what "alignment" means precisely.
  - Describe the supervised fine-tuning step and what data it requires.
  - Explain RLHF end-to-end — reward modeling, PPO, and the KL constraint.
  - Understand DPO as a simpler alternative to RLHF.
  - Apply prompt engineering patterns: system prompts, few-shot, chain-of-thought.
---

Session 6 produced a base GPT model: a powerful text generator that predicts probable continuations. This session covers the critical gap between "probable continuation" and "helpful, honest, harmless response."

## The Alignment Problem

A base GPT model has one goal: predict the next token. It doesn't know that "being helpful" is desirable, that "refusing harmful requests" is expected, or that "answering in a structured format" is useful. Its outputs reflect the statistics of internet text — which includes questions followed by more questions, harmful content, and confident misinformation.

**Alignment** refers to adjusting model behavior to match human values and intentions:
1. **Behavioral alignment**: teach the model to follow instructions and answer questions directly.
2. **Value alignment**: teach the model to decline harmful requests and behave safely.

## Step 1: Supervised Fine-Tuning (SFT)

Fine-tune on curated (instruction, ideal response) pairs with standard cross-entropy loss. This teaches the model *what helpful responses look like* — a specific format, a helpful tone, a direct answer. SFT requires relatively little data (thousands of examples) because the model already has knowledge from pretraining; it is shifting style, not learning new knowledge.

## Step 2: Reward Modeling

Rather than specifying ideal responses, have human raters *compare* multiple responses. A **reward model** $R_\phi$ is trained on these pairwise preferences:

$$\mathcal{L}_\text{reward} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l)) \right]$$

where $y_w$ is the preferred response and $y_l$ the less preferred one. Comparison is cognitively easier than generation for raters — faster and more reliable than writing the perfect response from scratch.

## Step 3: RLHF — Reinforcement Learning from Human Feedback

Fine-tune the SFT model using the reward model as signal, via **PPO**:

$$\mathcal{L}_\text{RLHF}(\theta) = -\mathbb{E}\left[ R_\phi(x, y) - \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

The **KL divergence penalty** $\beta \cdot D_\text{KL}[\pi_\theta \| \pi_\text{ref}]$ is essential: without it, the policy would "reward hack" — finding adversarial inputs that score high on the reward model while producing nonsensical outputs. The reward model is only an approximation of human preferences; treating it as a perfect oracle causes the policy to degenerate.

PPO solves this with a clipped surrogate objective that limits how much the policy changes per update. The same algorithm appears in Session 10 for locomotion — the mathematical structure is identical.

## Alternative: Direct Preference Optimization (DPO)

RLHF is complex: a separate reward model, online PPO sampling, careful tuning. **DPO** (Rafailov et al., 2023) shows mathematically that the RLHF objective has an equivalent form optimizable directly on preference pairs:

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

No RL, no reward model, no online sampling — just a supervised loss on preference data. DPO is now widely adopted for open-source LLM alignment (Llama, Mistral, Zephyr).

## Prompt Engineering

Even after RLHF, output quality depends heavily on input framing.

### System Prompts

Define the model's role and constraints before the user turn:

```
You are a robot dog planning assistant. You receive natural language
navigation commands and decompose them into a JSON action sequence.
Supported actions: NAVIGATE(target), WAIT(seconds), TURN(degrees),
REPORT(message). Always respond in valid JSON.
```

### Few-Shot Prompting

Provide input-output examples within the prompt to demonstrate the desired format — no fine-tuning required.

### Chain-of-Thought (CoT)

Adding "Let's think step by step" consistently improves accuracy on multi-step reasoning tasks. The hypothesis: generating intermediate reasoning steps gives the model more computation before committing to an answer. For the robot dog, CoT helps decompose complex commands like "navigate to the east wing, pick up the package, and return avoiding the main corridor."

---

## Further Reading

**Start here** *(accessible introductions)*
- OpenAI: "ChatGPT — how it was trained" blog post — [openai.com/blog/chatgpt](https://openai.com/blog/chatgpt)
- Lilian Weng: "Prompt Engineering" — [lilianweng.github.io/posts/2023-03-15-prompt-engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering) — comprehensive, well-structured overview
- Hugging Face: "RLHF: Reinforcement Learning from Human Feedback" — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

**Go deeper** *(technical references)*
- Ouyang et al.: "Training language models to follow instructions" (InstructGPT, NeurIPS 2022) — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov et al.: "Direct Preference Optimization" (NeurIPS 2023) — [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- Bai et al.: "Constitutional AI" (Anthropic, 2022) — [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. Why does a base GPT model not follow instructions reliably, even though it has broad world knowledge?
2. What is the role of the KL divergence term in the RLHF objective? What goes wrong if you remove it?
3. What is the key implementation advantage of DPO over RLHF? What does it give up in return?
4. Write a system prompt (3–5 sentences) that would cause an LLM planner to always prefer the safest rather than the fastest path when navigating the robot dog.
5. Why does chain-of-thought prompting improve performance on multi-step reasoning tasks? What is the proposed mechanism?

---

### → Next

The robot can now process language and analyze images — but as separate systems. "Go to the red chair" requires understanding "red chair" linguistically *and* identifying it in the camera feed simultaneously. Session 8 covers multimodal models that connect text and images in a shared representation space.
