---
id: 005
title: Seeing — CNNs and Vision Transformers
summary: Why convolutions suit images, and how attention brings global context.
tags: [vision, cnn, transformer]
learning_goals:
  - Define convolution and CNN architecture basics including stride, padding, and pooling.
  - Explain receptive fields and how they grow with depth.
  - Understand residual connections and why they enabled deeper networks.
  - Derive the self-attention mechanism and explain ViT's patch-based approach.
  - Compare CNNs and ViTs and understand when each is appropriate.
---

The robot dog needs to see. Its depth camera generates 640×480 frames at 30 Hz. From this stream, it must identify floor surfaces, obstacles, doors, and the humans giving it commands. This session covers the two dominant architectures for visual perception: Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

## Why MLPs Fail on Images

An MLP applied to a 224×224 RGB image flattens the input into 150,528 values. Three problems emerge:

1. **No spatial inductive bias.** Pixel $(0,0)$ is connected identically to pixel $(100,100)$, even though nearby pixels are almost always more correlated than distant ones.
2. **Parameter explosion.** A single hidden layer with 1024 units requires ~154M parameters — just for one layer, one image size.
3. **No translation invariance.** A cat in the upper-left corner produces completely different activations from a cat in the lower-right.

Convolutions address all three problems simultaneously.

## Convolution

A **convolution** slides a small filter $K \in \mathbb{R}^{k \times k}$ across the image, computing a weighted sum at each position:

$$(I * K)(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m, n)$$

Three key inductive biases: **locality** (only neighboring pixels interact), **weight sharing** (the same filter is applied everywhere — a 3×3 filter has 9 parameters regardless of image size), and **translation equivariance** (a pattern activates the same filter regardless of position). The filters are *learned parameters*.

### Padding, Stride, and Multi-Channel Convolutions

**Stride** $s$ controls step size; stride 2 halves spatial dimensions. Output size: $W_\text{out} = \lfloor(W - k + 2p)/s\rfloor + 1$. **Padding** $p$ controls border handling. A layer with $C_\text{in}$ input and $C_\text{out}$ output channels has $C_\text{out} \times C_\text{in} \times k^2$ weight parameters — far fewer than an MLP equivalent.

## CNN Architecture

A typical CNN alternates convolutional layers (local feature extraction) → ReLU activations → pooling layers (spatial downsampling, e.g., max-pooling) → fully connected layers at the end. Spatial resolution decreases through the network while channel count increases — compressing spatial detail into semantic content.

### Residual Connections (ResNet)

Deep CNNs (50+ layers) suffer from *vanishing gradients*: gradients near zero by the time they reach early layers. **Residual connections** (He et al., 2016) add a shortcut from each block's input to its output:

$$\mathbf{h}_\text{out} = \mathcal{F}(\mathbf{h}_\text{in}, \theta_\text{block}) + \mathbf{h}_\text{in}$$

The gradient flows directly to earlier layers through the addition, enabling training of networks with hundreds of layers. ResNet is now standard in virtually all deep vision architectures.

### Receptive Fields

The **receptive field** of a neuron is the region of the input image that influences its activation. Stacking $L$ layers of 3×3 convolutions gives a receptive field of $(2L+1)^2$. Strided convolutions and pooling expand it faster. This is the CNN's fundamental limitation: intermediate layers cannot reason about relationships between distant parts of the image.

## Vision Transformers and Attention

The **Vision Transformer (ViT; Dosovitskiy et al., 2021)** abandons locality entirely.

### Patch Embedding

A 224×224 image is divided into non-overlapping 16×16 patches → 196 tokens. Each patch is flattened and linearly projected to a $d$-dimensional embedding. A learnable `[CLS]` token is prepended, and *positional embeddings* are added (since transformers have no intrinsic notion of position). Result: a sequence of 197 token embeddings.

### Self-Attention

For token embeddings $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$QK^\top \in \mathbb{R}^{n \times n}$ contains pairwise similarities between all tokens. The softmax normalizes each row into attention weights. Every token attends to every other token — genuine global context from layer 1. The $\sqrt{d_k}$ scaling prevents large dot products from saturating the softmax.

**Multi-head attention** runs the mechanism $h$ times in parallel with independent projections, then concatenates results. Different heads learn to attend to different relationships simultaneously.

## CNN vs. ViT: When to Use Which

| Property | CNN | ViT |
|---|---|---|
| Inductive biases | Locality, translation equivariance | None (learned from data) |
| Data efficiency | High — good with small datasets | Low — needs large datasets |
| Global context | Limited (grows slowly with depth) | Full from layer 1 |
| Best use case | Small-to-medium datasets, fine-tuning | Large-scale pretraining |

For the robot dog: with a small robot-specific dataset, a pretrained CNN (fine-tuned from ImageNet) likely outperforms a ViT. At larger scale or with foundation model pretraining, ViTs dominate.

## DINO and Segment Anything

**DINOv2** (Oquab et al., 2023): self-supervised ViT producing semantically meaningful features — similar objects cluster in feature space, enabling segmentation and classification with a simple linear probe.

**SAM** (Kirillov et al., 2023): a ViT trained on 1 billion masks. Given any prompt (point, box, or text), it isolates the corresponding object with high accuracy — no task-specific training needed. For the robot dog: DINOv2 as backbone for terrain classification, SAM for interactive object segmentation triggered by the LLM planner.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "But what is a convolution?" — [youtube.com/watch?v=KuXjwB4LzSA](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- CS231n Stanford: Convolutional Neural Networks — [cs231n.github.io](https://cs231n.github.io) — detailed notes with visualizations
- The Illustrated Transformer — [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/) — best visual walkthrough of attention

**Go deeper** *(technical references)*
- He et al.: "Deep Residual Learning for Image Recognition" (CVPR 2016) — [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Dosovitskiy et al.: "An Image is Worth 16×16 Words" (ICLR 2021) — [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- Oquab et al.: "DINOv2" (2023) — [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. List the three inductive biases encoded by convolution. Why is each one well-suited to image data?
2. A CNN has 5 layers of 3×3 convolutions with no padding and stride 1. What is the receptive field of a neuron in the final layer?
3. Why does self-attention in a ViT require positional encodings, whereas convolutions in a CNN do not?
4. Remove all residual connections from a 50-layer ResNet. What training problem do you expect, and why?
5. You have a new visual classification task with only 2,000 labeled images. Would you choose a CNN or a ViT as the backbone? Justify your answer.

---

### → Next

The robot can now see. But it also needs to understand language — "go to the table" must be parsed into a representation a model can act on. Session 6 covers the path from N-gram language models to the transformer-based systems that power modern LLMs.
