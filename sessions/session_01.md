---
id: 001
title: Data, Representations, and the First Model
summary: From raw data to feature spaces, and building the first perceptron/MLP.
tags: [foundations, data, mlp]
learning_goals:
  - Define feature spaces for tabular, image, and text data.
  - Understand the formal learning setup (X, Y, dataset, hypothesis).
  - Explain perceptrons, XOR, and why we need MLPs.
  - Describe common activation functions and why nonlinearity matters.
---

Every machine learning model starts the same way: raw data arrives, and it must be turned into numbers. This session covers that conversion for the three main modalities the robot dog works with — tabular sensor data, images, and text — and then builds the first model capable of learning from those numbers.

## From Raw Data to Numbers

Computers operate on numbers, so every modality needs a *feature space*: a mathematical representation that captures relevant information while discarding irrelevant variation.

**Tabular data** is the simplest format — rows are samples, columns are features. Example: the dog's joint angles, motor currents, and IMU readings over time. If there are $d$ features per sample, mathematically each sample is a vector $\mathbf{x} \in \mathbb{R}^d$. The entire dataset is a matrix $X \in \mathbb{R}^{N \times d}$ where $N$ is the number of samples.

**Images as tensors.** A 224×224 RGB image is a tensor $\mathbf{X} \in \mathbb{R}^{224 \times 224 \times 3}$ — that's 150,528 values. This is high-dimensional but *spatially structured*: nearby pixels tend to be correlated, and the same object can appear at different locations. Flattening the image into a single vector discards this structure; Session 5 covers architectures that exploit it.

**Text as sequences.** Text is a string of discrete symbols that must be mapped to integers (*tokenization*). Details come in Session 6; the key idea is that text ultimately becomes a sequence of integer indices, which are then mapped to continuous vectors via an embedding table.

### Preprocessing and Normalization

Raw features often live on very different scales. Joint angles might range from $-\pi$ to $\pi$, while motor currents might range from 0 to 20 amperes. Training with unnormalized features causes optimization problems: the gradient of the loss with respect to large-scale features dominates, slowing convergence.

The standard fix is *standardization*: subtract the mean and divide by the standard deviation, so every feature has zero mean and unit variance:

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are estimated from the training set. Always compute normalization statistics on training data only — applying training statistics to validation and test data ensures no information leaks.

## Formal Learning Setup

The notation that carries through the entire series:

- **Input space** $\mathcal{X}$: the set of all possible inputs.
- **Output space** $\mathcal{Y}$: the set of all possible outputs (e.g., $\{0, 1\}$ for binary classification, or $\mathbb{R}$ for regression).
- **Dataset** $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$: a collection of $N$ input-output pairs assumed to be drawn i.i.d. from an unknown distribution $p(\mathbf{x}, y)$.
- **Hypothesis** $f_\theta : \mathcal{X} \rightarrow \mathcal{Y}$: a parameterized function. Learning consists of finding parameters $\theta$ such that $f_\theta(\mathbf{x})$ approximates $y$ well across $\mathcal{D}$ — and, crucially, on *new* inputs drawn from the same distribution.

The goal is *generalization*, not memorization. Session 3 addresses how to measure and improve generalization.

## The Perceptron

The simplest instantiation of $f_\theta$: a single artificial neuron. Given input $\mathbf{x} \in \mathbb{R}^d$, compute:

$$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

$$\hat{y} = \sigma(z)$$

where $\mathbf{w} \in \mathbb{R}^d$ are weights, $b \in \mathbb{R}$ is a bias, and $\sigma$ is an activation function. The parameters $\theta = \{\mathbf{w}, b\}$ define a hyperplane in $\mathbb{R}^d$.

The AND gate from Session 0 is exactly this: $w_1 = w_2 = 1$, $b = -1.5$, with a step activation $\sigma(z) = \mathbb{1}[z \geq 0]$.

The Perceptron Learning Rule (Rosenblatt, 1958) updates weights when the model makes an error:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (y_i - \hat{y}_i) \cdot \mathbf{x}_i$$

This converges to a correct solution *if and only if* the data are linearly separable.

## Why One Perceptron Isn't Enough: XOR

The XOR function outputs 1 when exactly one of two binary inputs is 1:

| $x_1$ | $x_2$ | XOR |
|--------|--------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot these in 2D: the two classes cannot be separated by a single line. No choice of $\mathbf{w}$ and $b$ solves this — XOR is *not linearly separable* (Minsky & Papert, 1969). The issue is not the activation function: even with a sigmoid or ReLU, the decision boundary of a single perceptron remains a hyperplane. We need to *transform* the input space so that XOR becomes linearly separable in the new representation. That transformation is learned by hidden layers.

## Multi-Layer Perceptrons

The solution: stack perceptrons into layers. An MLP with two hidden layers computes:

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$

$$\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$

$$\hat{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3$$

where $W_k \in \mathbb{R}^{d_{k} \times d_{k-1}}$ are weight matrices and $\sigma$ is applied element-wise. The hidden layers $\mathbf{h}_1, \mathbf{h}_2$ are *learned representations* — they progressively transform the input into a form that makes the output easy to compute.

### Architecture Choices

Practical rules of thumb:
- More layers allow more abstract representations but require more data and make optimization harder.
- For many problems, 2–4 hidden layers with 64–512 neurons each is a good starting point.
- The output layer size is fixed by the task: $k$ neurons for $k$-class classification, $k$ neurons for $k$-dimensional regression.

### The Universal Approximation Theorem

An MLP with a single hidden layer and a sufficient number of neurons can approximate any continuous function on a compact domain to arbitrary accuracy (Cybenko, 1989; Hornik, 1991). In practice, deeper networks generalize better and are easier to train than very wide shallow ones.

## Activation Functions

Without activation functions, stacking layers collapses into a single linear transformation.

$$\text{Step}(z) = \mathbb{1}[z \geq 0] \quad \text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

$$\text{ReLU}(z) = \max(0, z) \quad \text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The step function is non-differentiable — cannot be used with gradient descent. **ReLU** is the standard default for hidden layers. Sigmoid for binary output probability. Softmax for multi-class classification.

## Connecting to the Robot Dog

An MLP can serve as the simplest motor controller: input = 48 sensor values (joint angles, velocities, torques, IMU), output = 12 target joint positions. This is not hypothetical — early locomotion controllers for quadruped robots were exactly MLPs mapping proprioceptive sensor readings to joint commands, trained with reinforcement learning.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "But what *is* a neural network?" — [youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk)
- Andrej Karpathy: "The spelled-out intro to neural networks" — [youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0) — builds a micrograd engine step by step
- Goodfellow et al.: *Deep Learning*, Ch. 6 — [deeplearningbook.org](https://www.deeplearningbook.org) — formal treatment of feedforward networks

**Go deeper** *(technical references)*
- Minsky & Papert: *Perceptrons* (1969) — the original proof of XOR's linear inseparability
- Cybenko: "Approximation by superpositions of a sigmoidal function" (*Mathematics of Control*, 1989) — the universal approximation theorem
- LeCun, Bengio & Hinton: "Deep Learning" (*Nature*, 2015)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. A training dataset has features on very different scales (joint angles in radians ≈ ±3, motor currents in amperes ≈ 0–20). What pre-processing step is needed and why does it matter for training?
2. Why can't a single perceptron solve XOR? Sketch the argument geometrically in 2D.
3. What happens to the output of an MLP if you remove all activation functions? What does this mean for the model's representational power?
4. For a 4-class classification problem, what should the output layer of an MLP look like (how many neurons, which activation function)?
5. What is the difference between the hypothesis $f_\theta$ and the dataset $\mathcal{D}$ in the formal learning setup?

---

### → Next

We can define a model and compute its forward pass. But the weights are initialized randomly — the model knows nothing yet. How do we systematically find weights that make the model produce correct outputs? The answer is one of the most important algorithms in modern AI: gradient descent and backpropagation. Session 2 derives both from first principles.
