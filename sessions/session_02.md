---
id: 002
title: Learning as Optimization
summary: Loss functions, gradient descent, backprop, and practical optimizers.
tags: [foundations, optimization, training]
learning_goals:
  - Define and choose loss functions (MSE, cross-entropy).
  - Derive the gradient descent update and role of learning rate.
  - Explain backpropagation via the chain rule.
  - Name key optimizers (SGD, Adam) and understand when to use each.
  - Understand learning rate schedules and weight initialization.
---

Session 1 introduced the model: a parameterized function $f_\theta$ that maps inputs to outputs. The parameters $\theta$ start random — the model knows nothing. This session answers the fundamental question: how do we find $\theta$ values that make the model useful?

## What Learning Means

Learning means adjusting $\theta$ so that predictions $f_\theta(\mathbf{x})$ match true labels $y$. "Match" is quantified by a **loss function** $\mathcal{L}(f_\theta(\mathbf{x}), y) \in \mathbb{R}_{\geq 0}$ — a differentiable scalar that measures how wrong the model is. Learning is then formally defined as:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

This is called *empirical risk minimization*. Two requirements on $\mathcal{L}$: it must be (1) **non-negative**, and (2) **differentiable** with respect to $\theta$ almost everywhere.

## Loss Functions

### Mean Squared Error (Regression)

$$\mathcal{L}_\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|f_\theta(\mathbf{x}_i) - y_i\|^2$$

MSE penalizes large errors quadratically. Appropriate when large errors are especially costly and the output is a continuous quantity. For the robot dog: predicting joint torques from sensor readings → MSE.

One subtlety: MSE is sensitive to outliers. *Huber loss* combines quadratic behavior for small errors with linear behavior for large ones.

### Binary Cross-Entropy (Binary Classification)

$$\mathcal{L}_\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Cross-entropy penalizes *confident but wrong* predictions especially hard: if the model predicts $\hat{y} = 0.99$ for a sample with true label $y = 0$, the loss is $-\log(0.01) \approx 4.6$.

### Categorical Cross-Entropy (Multi-class)

For $K$ classes with one-hot labels and softmax outputs:

$$\mathcal{L}_\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

**Rule of thumb:** use MSE for regression, cross-entropy for classification. Applying MSE to classification is a common beginner mistake.

## Gradient Descent

The loss $\mathcal{L}(\theta)$ defines a surface over the parameter space. The gradient $\nabla_\theta \mathcal{L}$ points in the direction of steepest *ascent*. To minimize, step in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta > 0$ is the **learning rate**. Too large → oscillation or divergence. Too small → extremely slow convergence. A common starting point: $\eta \approx 10^{-3}$.

### Mini-Batch Gradient Descent

Computing gradients over the full dataset is expensive. **Mini-batch SGD** approximates the gradient using $B$ randomly sampled examples per step:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla_\theta \mathcal{L}_i$$

The noise in the gradient estimate often helps escape sharp minima. Typical batch sizes: 32–512. One full pass over the training set is an **epoch**.

## Backpropagation

Computing $\nabla_\theta \mathcal{L}$ requires derivatives of the loss with respect to every weight. For an MLP, this is done efficiently via the **chain rule**.

Consider a two-layer network: $z_1 = W_1 \mathbf{x} + \mathbf{b}_1$, $\mathbf{h} = \sigma(z_1)$, $\hat{y} = \mathbf{w}_2^\top \mathbf{h} + b_2$, $\mathcal{L} = (\hat{y} - y)^2$.

**Backward pass (right to left):**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{h}$$

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{w}_2 \odot \sigma'(z_1), \quad \frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial z_1} \cdot \mathbf{x}^\top$$

A single forward pass followed by a single backward pass computes *all* gradients simultaneously, at roughly twice the cost of the forward pass alone. Modern frameworks (PyTorch, JAX) implement this automatically via *automatic differentiation*.

## Practical Optimizers

### SGD with Momentum

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla_\theta \mathcal{L}_t, \quad \theta_{t+1} = \theta_t - \eta \cdot \mathbf{v}_{t+1}$$

Typical $\beta = 0.9$. Builds up velocity in directions of consistent gradient, dampening oscillations.

### Adam

Maintains per-parameter running averages of the gradient ($m_t$) and its square ($v_t$):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L}_t)^2$$

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$. Adam adapts the effective learning rate per parameter and is the most widely used optimizer in practice.

## Learning Rate Schedules

A fixed learning rate is rarely optimal. Common strategies:
- **Step decay:** halve $\eta$ every $k$ epochs.
- **Cosine annealing:** smooth decay from $\eta_\text{max}$ to $\eta_\text{min}$ over training.
- **Warmup:** increase $\eta$ linearly for the first few hundred steps before decaying.

## Weight Initialization

If all weights are zero, all neurons compute identical gradients — symmetry is never broken. **He initialization** (for ReLU):

$$w \sim \mathcal{N}\left(0, \frac{2}{d_\text{in}}\right)$$

**Xavier initialization** (for tanh/sigmoid) uses $\sqrt{6/(d_\text{in}+d_\text{out})}$. Both are standard defaults in all major frameworks.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "Gradient descent, how neural networks learn" — [youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w) — outstanding visual explanation
- Andrej Karpathy: "Backpropagation, intuitively" — [youtube.com/watch?v=q8SA3rM6ckI](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- distill.pub: "Why Momentum Really Works" — [distill.pub/2017/momentum](https://distill.pub/2017/momentum) — interactive visual explanation

**Go deeper** *(technical references)*
- Rumelhart, Hinton & Williams: "Learning representations by back-propagating errors" (*Nature*, 1986) — the original backprop paper
- Kingma & Ba: "Adam: A Method for Stochastic Optimization" (ICLR 2015) — [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Goodfellow et al.: *Deep Learning*, Ch. 4 (numerical optimization) and Ch. 8 (optimization for training)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. You train a model with MSE loss on a binary classification task (labels 0 and 1). Why is this suboptimal compared to cross-entropy?
2. If the learning rate is too large, what do you expect to see in the training loss curve over epochs?
3. Explain in your own words why backpropagation is just the chain rule. Walk through one derivative step for a 2-layer network.
4. Adam maintains two running averages per parameter. What are they, and why are both useful?
5. A model trained with batch size 512 generalizes slightly worse than the same model trained with batch size 32, even with the same number of epochs. Propose an explanation.

---

### → Next

We can now define a model, choose a loss, and optimize weights via gradient descent and backpropagation. In practice, a host of problems await: the model might memorize training data rather than generalize, training might diverge, or the hyperparameters might be wrong. Session 3 covers the practical side of training — the diagnostics and techniques that make the difference between a model that works and one that doesn't.
