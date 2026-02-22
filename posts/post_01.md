---
id: 001
title: Data, Representations, and the First Model
summary: From raw data to feature spaces, and building the first perceptron/MLP.
tags: [foundations, data, mlp]
learning_goals:
  - Define feature spaces for tabular, image, and text data.
  - Understand the formal learning setup (X, Y, dataset, hypothesis).
  - Explain perceptrons, XOR, and why we need MLPs.
---

### From Raw Data to Numbers

Computers operate on numbers, so every modality needs a feature space.

**Tabular data** is the simplest format — rows are samples, columns are features. Example: the dog's joint angles, motor currents, and IMU readings over time. Mathematically, each sample is a vector $\mathbf{x} \in \mathbb{R}^d$.

**Images as grids.** A 224×224 RGB image is a tensor $\mathbf{X} \in \mathbb{R}^{224 \times 224 \times 3}$ — that's 150,528 values. This is high-dimensional but spatially structured. 

**Text as sequences.** Text is a string of discrete symbols that must be mapped to integers (tokenization). Details come in Section 6; the key idea is that text ultimately becomes a sequence of numbers.

### Formal Learning Setup

The notation that carries through the entire series:

- **Input space** $\mathcal{X}$: the set of all possible inputs (e.g., all possible 224×224 RGB images).
- **Output space** $\mathcal{Y}$: the set of all possible outputs (e.g., $\{0, 1\}$ for binary classification, or $\mathbb{R}$ for regression).
- **Dataset** $\mathcal{D} = \{(\mathbf{x}\_i, y\_i)\}\_{i=1}^{N}$: a collection of $N$ input-output pairs drawn from an unknown distribution.
- **Hypothesis** $f_\theta : \mathcal{X} \rightarrow \mathcal{Y}$: a parameterized function. Learning consists of finding parameters $\theta$ such that $f_\theta(\mathbf{x})$ approximates $y$ well across $\mathcal{D}$.

### The Perceptron

The simplest instantiation of $f_\theta$: a single neuron. Given input $\mathbf{x} \in \mathbb{R}^d$, compute:

$$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

$$y = \sigma(z)$$

where $\mathbf{w} \in \mathbb{R}^d$ are weights, $b \in \mathbb{R}$ is a bias, and $\sigma$ is an activation function. The parameters $\theta = \{\mathbf{w}, b\}$ define a hyperplane in $\mathbb{R}^d$ — the perceptron classifies inputs based on which side of this hyperplane they fall on.

The AND gate from Section 0 is exactly this: $w_1 = w_2 = 1$, $b = -1.5$, with a step activation $\sigma(z) = \mathbb{1}[z \geq 0]$.

### Why One Perceptron Isn't Enough: XOR

The XOR function outputs 1 when exactly one of two binary inputs is 1:

| $x_1$ | $x_2$ | XOR |
|--------|--------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot these in 2D: the two classes cannot be separated by a single line. No choice of $\mathbf{w}$ and $b$ solves this — XOR is *not linearly separable*. This was formally shown by Minsky & Papert (1969) and is a fundamental limitation of single-layer models.

A subtle but important note: the issue is not the activation function. Even with a sigmoid or ReLU, the decision boundary of a single perceptron remains a hyperplane. What we need is a way to *transform* the input space so that XOR *becomes* linearly separable in the new representation.

### Multi-Layer Perceptrons

The solution: stack perceptrons into layers. Each layer transforms the representation, creating a new feature space in which the problem may be easier. Mathematically, an MLP with two hidden layers computes:

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$

$$\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$

$$\hat{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3$$

where $W_k \in \mathbb{R}^{d_{k} \times d_{k-1}}$ are weight matrices and $\sigma$ is applied element-wise. The full parameter set is $\theta = \{W_1, \mathbf{b}_1, W_2, \mathbf{b}_2, W_3, \mathbf{b}_3\}$.

The hidden layers $\mathbf{h}_1, \mathbf{h}_2$ are *learned representations* — they progressively transform the input into a form that makes the output easy to compute. For XOR, a single hidden layer with two neurons suffices: the first hidden layer maps the four input points into a new 2D space where the classes are linearly separable.

### Activation Functions

Without activation functions, stacking layers collapses into a single linear transformation (a product of matrices is still a matrix). The nonlinearity is essential.

$$\text{ReLU}(z) = \max(0, z)$$

$$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

ReLU is the standard default in hidden layers — simple, effective, and well-behaved during optimization. Sigmoid is used in output layers for binary probabilities. Softmax generalizes sigmoid to multi-class classification, converting a vector of logits into a probability distribution.

### Connecting to the Robot Dog

An MLP can serve as the simplest motor controller: input = 16 sensor values (joint angles + IMU data), output = 12 motor commands (one per joint). The architecture is exactly the MLP above, with $d_0 = 16$ and $d_{\text{out}} = 12$. The open question: how do we find the right weight matrices? That requires an optimization algorithm.

### → Next

We can define a model and compute its forward pass. But the weights are random — the model knows nothing yet. How do we systematically find weights that make the model produce correct outputs? The answer is one of the most important algorithms in modern AI: gradient descent and backpropagation.
