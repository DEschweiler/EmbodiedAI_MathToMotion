---
id: 004
title: The Data Problem — Annotation and Self-Supervision
summary: Labels are scarce; self-supervision and foundation models change the game.
tags: [self-supervision, foundation-models, data]
learning_goals:
  - Explain the annotation bottleneck and its practical consequences.
  - Define self-supervised learning and describe three major paradigms.
  - Understand what foundation models are and how scaling affects capabilities.
  - Describe fine-tuning, linear probing, and parameter-efficient adaptation.
---

Phase I assumed that every training example has a label. In practice, obtaining labels is the biggest scaling bottleneck in machine learning — and overcoming it is one of the defining achievements of the last decade.

## The Annotation Bottleneck

Consider what labeling actually requires:

- **ImageNet** (1.2M images, 1000 classes): ~$1M and 2.5 years of crowdsourced annotation.
- **Medical imaging**: requires a specialist physician per annotation — hours of expert time per study.
- **Robot sensor data**: a locomotion controller needs terrain type, friction coefficient, obstacle locations — none of which come automatically from sensor readings.

Modern models need billions of training examples, making exhaustive manual annotation impossible. Annotation strategies fall along a spectrum from manual (gold standard, doesn't scale) to crowdsourcing to weak supervision to self-supervised learning.

An elegant example of inadvertent annotation: CAPTCHAs. Every "select all images containing traffic lights" click is a human-annotated training example. Google used reCAPTCHA data to train Street View detectors — the annotation cost was borne by users unknowingly.

## Self-Supervised Learning

**Labels can be derived from the data itself** by hiding part of the data and asking the model to predict it. No human annotators required.

### Paradigm 1: Masked Prediction

Hide a portion of the input and train the model to reconstruct it.

**Masked Language Modeling (BERT):** Replace 15% of tokens with `[MASK]`. Train to predict the original tokens. The training signal is the text itself.

**Masked Autoencoders (MAE):** Randomly mask 75% of patches in an image. Train a vision transformer to reconstruct the missing patches in pixel space. The high masking ratio forces the model to learn global structure.

### Paradigm 2: Contrastive Learning

Create multiple *views* of the same input and train the model to recognize them as related while pushing different inputs apart.

**SimCLR:** For each image, sample two augmented views $(v_i, v_i')$. The NT-Xent loss maximizes agreement between positive pairs while minimizing it for cross-pairs:

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i') / \tau)}{\sum_{j \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}$$

No labels — just augmentation choices. The intuition: two crops of the same image should look similar to a model that understands content.

### Paradigm 3: Autoregressive Prediction

Predict the next element in a sequence, using the sequence itself as supervision.

**Next-token prediction (GPT):** Given $(t_1, \ldots, t_{n-1})$, predict $t_n$ via cross-entropy. Every text document on the internet is training data — no annotation whatsoever. Details in Session 6.

## Foundation Models

A **foundation model** is the result of self-supervised pretraining at massive scale. It learns general-purpose representations that transfer to many downstream tasks with very little task-specific data.

Examples:
- **GPT-4**: text prediction on trillions of tokens → reasoning, coding, conversation.
- **DINOv2**: self-supervised visual features → segmentation, depth estimation, classification.
- **CLIP**: image-text contrastive training → zero-shot image classification.
- **SAM**: prompt-based image segmentation from 1B masks.

### Why Scale Matters

Performance follows *scaling laws* (Kaplan et al., 2020): loss decreases predictably as a power law of model size, dataset size, and compute. Larger models exhibit *emergent capabilities* — abilities absent in smaller models that appear sharply at a certain scale (e.g., multi-step reasoning, in-context learning).

For the robot dog: instead of training separate models from scratch for terrain classification, obstacle detection, and object recognition, take a single vision foundation model and adapt it to each task.

## Adapting Foundation Models

### Linear Probing

Freeze all pretrained parameters. Extract representations $\mathbf{h} = f_{\theta_\text{pre}}(\mathbf{x})$ and train only a linear head $\hat{y} = W\mathbf{h} + \mathbf{b}$. Fast, data-efficient, and a clean measure of representation quality.

### Fine-Tuning

Initialize with pretrained weights, then train all parameters on the downstream task. More flexible, requires more data, risks *catastrophic forgetting*.

### Parameter-Efficient Fine-Tuning (PEFT)

Fine-tuning billions of parameters is expensive. PEFT methods inject a small number of trainable parameters while freezing the backbone:

- **Adapters:** small bottleneck modules inserted between transformer layers (~1% of parameters).
- **LoRA:** decompose weight updates as $\Delta W = AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, rank $r \ll d$. Trainable parameters scale with $r$, not $d^2$.
- **Prompt tuning:** learn soft prompt vectors prepended to the input; the model is untouched.

PEFT is the standard approach for adapting large models in resource-constrained settings.

---

## Further Reading

**Start here** *(accessible introductions)*
- Yannic Kilcher: "BERT explained" — [youtube.com/@YannicKilcher](https://www.youtube.com/@YannicKilcher) — clear explanations of major SSL papers
- The Illustrated BERT — [jalammar.github.io/illustrated-bert](https://jalammar.github.io/illustrated-bert/) — visual walkthrough
- Hugging Face course: "The big picture" — [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course) — practical foundation model usage

**Go deeper** *(technical references)*
- Devlin et al.: "BERT" (NAACL 2019) — [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- Chen et al.: "SimCLR" (ICML 2020) — [arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
- He et al.: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022) — [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
- Hu et al.: "LoRA" (ICLR 2022) — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. Explain in one sentence why self-supervised learning scales better than supervised learning.
2. What is the "masked prediction" objective? How does it generate a training signal without any human annotation?
3. What is catastrophic forgetting, and under which adaptation strategy does it most likely occur?
4. LoRA decomposes a weight update as $\Delta W = AB$ with rank $r \ll d$. Why does this reduce the number of trainable parameters compared to full fine-tuning?
5. When would you prefer linear probing over fine-tuning for adapting a foundation model?

---

### → Next

We now know how to learn general representations from massive unlabeled data. But which architecture works best for images? MLPs treat every pixel independently — they discard spatial structure entirely. Session 5 introduces two architectures that exploit image geometry: convolutional neural networks and vision transformers.
