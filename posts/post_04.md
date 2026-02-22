---
id: 004
title: The Data Problem — Annotation and Self-Supervision
summary: Labels are scarce; self-supervision and foundation models change the game.
tags: [self-supervision, foundation-models, data]
learning_goals:
  - Explain the annotation bottleneck.
  - Define self-supervised learning and give examples.
  - Describe foundation models, fine-tuning, and linear probing.
---

### Label Bottleneck

Phase I assumed that every training example has a label. In practice, this is the biggest scaling bottleneck. Medical images require expert annotation (slow, expensive). Robot sensor data would need per-terrain labels. Scaling supervised learning to billions of examples requires an alternative.

**Annotation strategies** range across a spectrum: manual annotation (gold standard, doesn't scale), weak supervision (heuristics generate noisy labels automatically), and — an elegant example — CAPTCHAs. Every "select all images with traffic lights" click is human annotation for a vision model's training set. User selections on image patches accumulate into heatmaps — rudimentary segmentations generated entirely from human annotation.

### Self-Supervised Learning

The most scalable solution: derive labels from the data itself.

**Masked Autoencoder:** Mask random patches of an image. Train the model to reconstruct the missing parts. The label is the original image — freely available.

**Contrastive Learning:** Create augmented views of the same image (crop, rotate, color-jitter). Train the model to recognize them as related (high similarity) while pushing unrelated images apart (low similarity). Formally, for a positive pair $(v_i, v_i^+)$ and negative pairs $(v_i, v_j^-)$:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(v_i, v_i^+) / \tau)}{\sum_j \exp(\text{sim}(v_i, v_j^-) / \tau)}$$

where $\text{sim}$ is cosine similarity and $\tau$ is a temperature parameter. No labels — just data augmentations.

**Next-Token Prediction:** Given a text sequence, predict the next token. The label is the text itself. (→ Post 6)

### Foundation Models

A **foundation model** is the result of self-supervised pretraining on massive datasets. It learns general-purpose representations that transfer to many downstream tasks. Examples: GPT (text), DINO (images), CLIP (text + images).

On the robot dog: instead of training separate models from scratch for terrain classification, obstacle detection, and object recognition, take a single foundation model and adapt it.

### Adapting Them

Two strategies for adapting a pretrained model $f_\theta$ with frozen backbone parameters $\theta_{\text{pre}}$:

**Fine-tuning:** Initialize with $\theta_{\text{pre}}$, then continue training all parameters on the downstream task. Flexible, but requires more data and compute. Risk of "catastrophic forgetting" — losing useful pretrained representations.

**Linear probing:** Freeze $\theta_{\text{pre}}$ entirely. Add a single linear layer $g_\phi(\mathbf{h}) = W\mathbf{h} + \mathbf{b}$ on top and only train $\phi$. Fast and simple, but limited to what the pretrained features already capture.

The tradeoff is clear: fine-tuning adapts the full model to the new task, linear probing leverages the pretrained representation as-is.

### → Next

We now know how to learn general representations from massive unlabeled data. But which model architecture works best for images? MLPs treat every pixel independently. For images, we need architectures that exploit spatial structure: convolutional neural networks for local patterns, and vision transformers for global context.