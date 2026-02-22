---
id: 008
title: Seeing and Reading — Multimodal Models
summary: Bridge vision and language with shared embeddings and VLMs.
tags: [multimodal, clip, vlm]
learning_goals:
  - Define a shared representation space across modalities.
  - Describe CLIP-style contrastive training.
  - Explain how VLMs process image and text tokens together.
---

### The Gap Between Modalities

An LLM understands text, a vision model understands images — but neither can identify "the red chair on the left in this photo." Bridging modalities requires mapping both into a **shared representation space** where text and images can be directly compared.

### CLIP: Connecting Images and Text

CLIP (Contrastive Language-Image Pretraining) is trained on millions of (image, caption) pairs. Let $f_I$ be the image encoder and $f_T$ the text encoder. For a batch of $N$ matched pairs, the training objective maximizes similarity between matched pairs and minimizes it for mismatched ones — this is exactly the contrastive loss from Section 4, applied across modalities:

$$\mathcal{L_{\text{CLIP}}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(f_I(\mathbf{x_i}), f_T(\mathbf{t_i})) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(f_I(\mathbf{x_i}), f_T(\mathbf{t_j})) / \tau)}$$

After training, images and text live in the same vector space. This enables **zero-shot classification**: given an image, compare its embedding to candidate text embeddings ("a cat", "a dog") and pick the closest match — no task-specific training needed.

### Vision-Language Models

VLMs go further: they combine a vision encoder (converting images into tokens) with an LLM (processing image tokens alongside text tokens). The result: a model that "sees" an image and "speaks" about it.

On the robot dog: camera image → vision encoder produces image tokens → LLM processes these with context → output: "I see a student carrying a package in a long corridor. There's an open door on the right."

### The Shared Representation Space

Tying the threads together: IMU data can be projected into compact vectors via SVD. Camera images become vectors via vision encoders. Language commands become vectors via tokenization and embedding. All of these can be compared and combined in a shared space — the foundation for multimodal robot intelligence.

### → Next

We now have all perceptual and linguistic building blocks. The dog can see, understand language, and connect both. But who coordinates everything? Who decides when to use the camera, when to make a plan, when to start walking? That's the job of an *agent*.
