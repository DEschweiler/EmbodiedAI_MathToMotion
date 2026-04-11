---
id: 006
title: Listening — From N-Grams to Language Models
summary: Tokenization, N-grams, and transformers for next-token prediction.
tags: [language, transformer, llm]
learning_goals:
  - Explain tokenization including BPE and its design tradeoffs.
  - Describe N-gram models and their fundamental limitations.
  - Understand how the transformer is adapted for causal language modeling.
  - Explain token embeddings and positional encodings.
  - Connect next-token prediction to GPT-style foundation models.
---

The robot dog needs to understand language. When someone says "go to the door on your left and wait there," the system must parse this into a structured representation it can act on. This session covers how neural networks learn to process text — from basic statistical models to the transformer-based language models that power modern AI.

## From Characters to Tokens

Text could be encoded at the character level, but this is inefficient — assembling words from individual characters requires enormous context windows. At the opposite extreme, treating entire words as atomic units produces huge vocabularies. **Tokenization** strikes a middle ground: frequent character sequences are grouped into tokens (roughly morphemes or syllables). "Understanding" might become `["under", "stand", "ing"]`.

### Byte-Pair Encoding (BPE)

The dominant algorithm, used in GPT and Llama:

1. Initialize the vocabulary with all individual characters (bytes).
2. Count all adjacent token pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until the target vocabulary size is reached (typically 30,000–100,000 tokens).

BPE is built entirely from corpus statistics — no linguistic knowledge required. Common words become single tokens; rare words split into subword units. Any string is representable, including out-of-vocabulary words.

## N-Grams: The Simplest Language Statistics

Estimate the probability of the next token from relative counts:

$$P(x_t \mid x_{t-N+1}, \dots, x_{t-1}) = \frac{\text{count}(x_{t-N+1}, \dots, x_t)}{\text{count}(x_{t-N+1}, \dots, x_{t-1})}$$

With small $N$: locally plausible but globally incoherent text. With large $N$: mostly copies training data verbatim. The fundamental problem: possible N-grams grow as $|V|^N$. For $|V| = 50{,}000$ and $N = 5$: $\sim 3 \times 10^{23}$ possible sequences — the model cannot generalize to novel sentences. Long-range dependencies (subject and verb 20 tokens apart) are inaccessible to any fixed-window model.

## Neural Language Models

Instead of storing counts, train a neural network to estimate $P(x_t \mid x_{<t})$ directly. Because the function is parameterized and continuous, it generalizes to contexts never seen during training. The transformer enables this at scale.

## The Transformer for Language

The same building block from Session 5 — multi-head self-attention + feedforward layers — adapted for language modeling with two key modifications.

### Token Embeddings

Each token ID is mapped to a dense vector $\mathbf{e}_t \in \mathbb{R}^d$ via a learned embedding matrix $E \in \mathbb{R}^{|V| \times d}$. Tokens with similar meanings learn to have similar embedding vectors.

### Positional Encoding

Self-attention is permutation-invariant — it treats the token sequence as an unordered set. Language is not. A **positional encoding** is added to each token embedding to inject order information:

$$\mathbf{x}_t^\text{input} = \mathbf{e}_t + \text{PE}(t)$$

The original transformer used fixed sinusoidal functions. Modern LLMs use **RoPE** (Rotary Position Embedding), which encodes relative rather than absolute position and generalizes to longer sequences.

### Causal Masking

For language *generation*, the model must not see future tokens — otherwise it trivially copies them. **Causal masking** sets attention weights for all future positions to $-\infty$ before the softmax, zeroing them out:

$$\text{Attention}(Q,K,V)_{ij} = 0 \quad \text{if } j > i$$

Each token can only attend to itself and previous tokens — the autoregressive property is preserved.

### Language Modeling Head

The transformer's hidden state at position $t$ is projected to logits over the vocabulary:

$$P(x_{t+1} = v \mid x_{\leq t}) = \text{softmax}(W_\text{vocab} \mathbf{h}_t)_v$$

$W_\text{vocab}$ is often *tied* with the input embedding matrix $E^\top$, reducing parameters.

## GPT: The Principle

GPT is a decoder-only transformer trained on next-token prediction:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

No labels needed — the next token *is* the label. Trained on web-scale text (hundreds of billions to trillions of tokens), the model learns grammar, facts, reasoning patterns, and much more from this single objective. Scaling from GPT-1 (117M parameters) to GPT-3 (175B) produced emergent capabilities consistent with the scaling laws discussed in Session 4.

## Encoder vs. Decoder Architectures

| | **Encoder (BERT)** | **Decoder (GPT)** |
|---|---|---|
| Attention mask | Bidirectional | Causal (past only) |
| Pretraining objective | Masked language modeling | Next-token prediction |
| Use case | Classification, embeddings | Text generation, instruction following |

Most modern large language models (GPT-4, Llama, Mistral) are decoder-only.

## Connecting to the Robot Dog

The robot's LLM-based planner (Session 9) is a GPT-style decoder-only transformer. The same tokenizer that handles English instructions also handles task-specific tokens added during fine-tuning: `<NAVIGATE>`, `<WAIT>`, `<OBSTACLE_DETECTED>`.

---

## Further Reading

**Start here** *(accessible introductions)*
- The Illustrated GPT-2 — [jalammar.github.io/illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2/) — visual walkthrough of autoregressive generation
- Andrej Karpathy: "Let's build GPT from scratch" — [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) — builds a GPT in ~2 hours of video
- Hugging Face: "The Transformer model family" — [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)

**Go deeper** *(technical references)*
- Vaswani et al.: "Attention Is All You Need" (NeurIPS 2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Radford et al.: "Language Models are Unsupervised Multitask Learners" (GPT-2, OpenAI 2019)
- Brown et al.: "Language Models are Few-Shot Learners" (GPT-3, NeurIPS 2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. Why does the causal attention mask use $-\infty$ (negative infinity) rather than 0 before the softmax?
2. Explain BPE tokenization in three steps. What problem does it solve compared to character-level or word-level encoding?
3. What is the key architectural difference between BERT and GPT? For which tasks is each better suited, and why?
4. Why do N-gram models fail to generalize to novel sentences, even with large $N$?
5. Positional encodings are added to token embeddings before the transformer. What would happen if you removed them entirely?

---

### → Next

We now have an LLM that generates plausible text — but it is a pattern-completion engine, not an instruction-following assistant. Asked "What is the capital of France?", a raw GPT model might continue with more questions rather than an answer. Session 7 covers how to turn a text generator into a reliable, instruction-following system through RLHF.
