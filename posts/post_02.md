---
id: 002
title: Learning as Optimization
summary: Loss functions, gradient descent, backprop, and practical optimizers.
tags: [foundations, optimization, training]
learning_goals:
  - Define and choose loss functions (MSE, cross-entropy).
  - Derive the gradient descent update and role of learning rate.
  - Explain backpropagation and name key optimizers (SGD, Adam).
---

### What Learning Means

Learning means adjusting $\theta$ so that predictions $f_\theta(\mathbf{x})$ match true labels $y$. "Match" is quantified by a **loss function** $\mathcal{L}(f_\theta(\mathbf{x}), y)$ — a differentiable scalar that measures how wrong the model is. Learning = minimizing this scalar over the dataset:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

### Losses

**Mean Squared Error (regression):**

$$\mathcal{L_\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (f_\theta(\mathbf{x}_i) - y_i)^2$$

MSE penalizes large errors quadratically — a prediction that's off by 10 contributes 100× more than one that's off by 1.

**Binary Cross-Entropy (classification):**

$$\mathcal{L_{\text{BCE}}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

where $\hat{y_i} = f_\theta(\mathbf{x}_i) \in (0, 1)$ is the predicted probability. Cross-entropy penalizes *confident but wrong* predictions especially hard: if the model predicts $\hat{y} = 0.99$ but the true label is 0, the loss is $-\log(0.01) \approx 4.6$ — a strong corrective signal.

On the robot dog: predicting energy consumption from speed → MSE. Classifying terrain as safe vs. dangerous → cross-entropy.

### Gradient Descent

The loss $\mathcal{L}(\theta)$ defines a surface over the parameter space. The gradient $\nabla_\theta \mathcal{L}$ points in the direction of steepest *ascent*. To minimize, step in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta > 0$ is the **learning rate**. This is the fundamental update rule of gradient-based optimization.

The landscape metaphor: the loss is altitude, the gradient points uphill, and we walk downhill. The learning rate controls step size. Too large → oscillation or divergence (overshooting valleys). Too small → extremely slow convergence. Finding the right $\eta$ is one of the most important practical decisions in training.

### Backpropagation

Computing $\nabla_\theta \mathcal{L}$ requires derivatives of the loss with respect to every weight in the network. For an MLP with multiple layers, this is done efficiently via the **chain rule**.

Consider a simple two-layer network with scalar output. Let $z_1 = W_1 \mathbf{x} + \mathbf{b}_1$, $\mathbf{h} = \sigma(z_1)$, $\hat{y} = \mathbf{w}_2^\top \mathbf{h} + b_2$, and $\mathcal{L} = (\hat{y} - y)^2$. Then:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{w}_2} = 2(\hat{y} - y) \cdot \mathbf{h}$$

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

The error propagates *backward* through the network — from the output layer through each hidden layer to the input — hence the name. Each layer's gradient depends on the gradients of the layers above it, computed recursively via the chain rule. This is computationally efficient: a single forward pass followed by a single backward pass computes *all* gradients simultaneously.

### Practical Optimizers

**Stochastic Gradient Descent (SGD)** computes gradients on random mini-batches of size $B$ rather than the full dataset:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta \mathcal{L}_i$$

This is noisier but much faster per step, and the noise often helps escape local minima.

**Adam** (Adaptive Moment Estimation) maintains per-parameter running averages of the gradient and its square, effectively adapting the learning rate for each parameter individually. It's the most widely used optimizer in practice and often works well with minimal tuning.

### → Next

We can now define a model, choose a loss, and optimize weights via gradient descent with backpropagation. In theory, this is all we need. In practice, a host of problems await: the model might memorize training data rather than generalize, the training might diverge, or the hyperparameters might be wrong. Section 3 covers the practical side of training — the diagnostics and techniques that make the difference between a model that works and one that doesn't.
