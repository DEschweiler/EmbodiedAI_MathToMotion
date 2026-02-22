---
id: 005
title: Seeing — CNNs and Vision Transformers
summary: Why convolutions suit images, and how attention brings global context.
tags: [vision, cnn, transformer]
learning_goals:
  - Define convolution and CNN architecture basics.
  - Explain receptive fields and pooling.
  - Introduce ViT and self-attention for global context.
---

### Why MLPs Fail on Images

An MLP on a 224×224 RGB image has 150,528 input dimensions, each fully connected to the first hidden layer. This ignores spatial structure entirely: pixel (0,0) is treated identically to pixel (100,100), even though neighboring pixels are far more correlated. The number of parameters explodes, and the model has no inductive bias toward local patterns.

### Convolution

A **convolution** slides a small filter $K \in \mathbb{R}^{k \times k}$ across the image $I$, computing a weighted sum at each position:

$$(I * K)(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m, n)$$

This has three key properties: (1) **locality** — only neighboring pixels interact, (2) **weight sharing** — the same filter is applied at every position, drastically reducing parameters, and (3) **translation equivariance** — a pattern is detected regardless of where it appears. In a CNN, the filters $K$ are not hand-designed — they are *learned parameters*, part of $\theta$.

### CNN Architecture

Multiple convolutional layers (with progressively more abstract features — first edges, then textures, then object parts), interleaved with **pooling** (spatial downsampling, typically max-pooling: $\text{pool}(x) = \max_{i,j \in \text{window}} x_{i,j}$), and fully connected layers at the end for classification. The spatial resolution decreases through the network while the number of feature channels increases — compressing spatial information into semantic information.

### The Limitation: No Global Context

A CNN's receptive field — the input region that influences a given output neuron — grows only gradually with depth. A deep network eventually "sees" the whole image, but intermediate layers are locally constrained. For the robot dog, a CNN can detect a traffic light in a patch but may not understand that the light is at the end of a long, clear corridor.

### Vision Transformers and Attention

The **Vision Transformer (ViT)** treats an image as a sequence of patches. A 224×224 image is divided into non-overlapping 16×16 patches, yielding a sequence of $14 \times 14 = 196$ tokens. Each patch is linearly projected into a vector (embedding), and a transformer processes all tokens simultaneously.

The core mechanism is **self-attention**. For a sequence of token embeddings $\mathbf{X} \in \mathbb{R}^{n \times d}$, three matrices are computed:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices. The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The term $QK^\top$ computes pairwise similarity between all tokens. The softmax normalizes these into attention weights. The division by $\sqrt{d_k}$ prevents the dot products from growing too large (which would push softmax into saturation). The result: every patch can attend to every other patch — genuine global context from the first layer.

**Multi-head attention** runs this process $h$ times in parallel with different projections, concatenating the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

This allows the model to capture different types of relationships simultaneously — spatial proximity, semantic similarity, color coherence, etc.

### DINO and Segment Anything

**DINO** is a self-supervised ViT trained via a teacher-student framework without any labels. Its features are semantically meaningful: similar objects cluster together in feature space, enabling segmentation without supervision. **Segment Anything** provides universal segmentation — given any prompt (point, box, text), it isolates the corresponding object. For the robot dog, these form the visual perception pipeline: segment the scene into floor, obstacles, doors, people.

### → Next

The robot can now see. But it also needs to understand language — "go to the table" must be processed into a representation a model can act on. This leads to the most influential AI systems of recent years: Large Language Models.
