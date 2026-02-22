---
id: 006
title: Reading — From N-Grams to Language Models
summary: Tokenization, N-grams, and transformers for next-token prediction.
tags: [language, transformer, llm]
learning_goals:
  - Explain tokenization and N-gram limits.
  - Describe transformer language models and causal masking.
  - Connect next-token prediction to GPT-style LLMs.
---

### From Characters to Tokens

Text is a string of characters, but encoding individual characters is inefficient: the model would need to assemble words from letters, requiring enormous context windows. **Tokenization** groups frequent character sequences into tokens (roughly "syllables"). "Understanding" might become ["Under", "stand", "ing"]. Each token gets an integer ID, and the model works with integer sequences.

A tokenizer like BPE (Byte-Pair Encoding) is built by iteratively merging the most frequent character pairs in a corpus. The resulting vocabulary typically contains 30,000–100,000 tokens, balancing granularity with vocabulary size.

### N-Grams: The Simplest Language Statistics

Given a text corpus, count how often specific token sequences of length $N$ occur. To generate new text, compute:

$$P(x_t \mid x_{t-N+1}, \dots, x_{t-1}) = \frac{\text{count}(x_{t-N+1}, \dots, x_t)}{\text{count}(x_{t-N+1}, \dots, x_{t-1})}$$

and sample the next token proportionally. With small $N$, the model produces locally plausible but globally incoherent text. With large $N$, it mostly copies the training data verbatim. The fundamental problem: the number of possible N-grams grows as $|V|^N$ (where $|V|$ is vocabulary size), and most sequences never appear in training data. The model cannot generalize.

### The Jump to Neural Networks

Instead of storing counts, train a neural network to *estimate* $P(x_t \mid x_{<t})$. The network maps a context of previous tokens to a probability distribution over the vocabulary. Because the function is parameterized and continuous, it can generalize to contexts never seen during training — predicting meaningful next tokens even for novel sentences.

### The Transformer for Language

The same architecture from Post 5, now applied to text. Each token is embedded as a vector. The transformer processes the full sequence with self-attention, and the output at position $t$ predicts the distribution over the next token:

$$P(x_t \mid x_1, \dots, x_{t-1}) = \text{softmax}(W_{\text{vocab}} \cdot \mathbf{h}_t)$$

where $\mathbf{h}_t$ is the transformer's hidden state at position $t$ and $W_{\text{vocab}}$ projects back to vocabulary size. A critical detail: **causal masking** ensures that position $t$ can only attend to positions $\leq t$, preserving the autoregressive structure (the model can't peek at future tokens).

### Embeddings

Before entering the transformer, each token ID is mapped to a dense vector via a learned embedding matrix $E \in \mathbb{R}^{|V| \times d}$. These embeddings encode semantic similarity: tokens with similar meaning end up as nearby vectors. This is closely related to dimensionality reduction techniques like SVD — a mathematical tool that also applies to the robot dog's sensor data (projecting high-dimensional IMU readings into a compact latent space).

### GPT: The Principle

GPT is a decoder-only transformer trained on next-token prediction over massive text corpora. The training objective is maximum likelihood:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

No manual labels are needed — the next token *is* the label. This makes GPT a foundation model for language, trained self-supervised at scale. But it's "just" a text generator: given context, it produces the most probable continuation. It doesn't inherently answer questions or follow instructions. How to get from here to a useful assistant is the subject of Post 7.

### → Next

We have an LLM that generates plausible text — but it's a pattern-completion engine, not an instruction-following assistant. "What is the capital of France?" might be continued with more questions rather than an answer. How do you turn a text generator into a helpful system that follows instructions?
