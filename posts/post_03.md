---
id: 003
title: Training in Practice
summary: Overfitting, underfitting, validation, regularization, and reading curves.
tags: [foundations, training, regularization]
learning_goals:
  - Spot overfitting vs. underfitting via curves.
  - Use train/val/test splits correctly.
  - Apply dropout and weight decay; practice early stopping.
---

### Overfitting vs. Underfitting

A model with enough parameters can memorize any training set — driving the training loss to zero while learning nothing generalizable. That's **overfitting**: perfect on training data, catastrophic on new data. Conversely, a model that's too small or poorly configured can't even fit the training data — **underfitting**.

The goal is never the best training loss; it's the best performance on *unseen data*. This is the bias-variance tradeoff: too simple → high bias (overfitting), too complex → high variance (underfitting).

### Train / Val / Test

- **Training set**: the model learns from this.
- **Validation set**: evaluated after each epoch — used to tune hyperparameters and detect overfitting. The model never trains on this data.
- **Test set**: evaluated once at the very end — the unbiased final evaluation. Never used for any decision during development.

### Regularization

**Dropout** randomly sets a fraction $p$ of neuron activations to zero during training:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{with probability } 1-p \end{cases}$$

The rescaling by $1/(1-p)$ ensures the expected value of $\tilde{h}_i$ remains equal to $h_i$. Dropout forces the network to learn redundant representations — no single neuron can become a critical bottleneck, which improves generalization.

**Weight decay** (L2 regularization) adds a penalty on the magnitude of weights to the loss:

$$\mathcal{L_{\text{total}}} = \mathcal{L_{\text{data}}} + \lambda \|\theta\|^2 = \mathcal{L_{\text{data}}} + \lambda \sum_j \theta_j^2$$

This discourages large weights, effectively preferring simpler models. The hyperparameter $\lambda$ controls the strength of the penalty.

### Reading Curves

The most important diagnostic tool: training loss and validation loss plotted over epochs.

- Both falling together → training is progressing well.
- Training loss falling while validation loss rises → overfitting. Stop training (early stopping).
- Both barely falling → underfitting. The model may be too small, or the learning rate is wrong.

**Early stopping** monitors the validation loss and halts training when it stops improving (typically with a patience parameter — wait $k$ epochs before stopping to avoid premature termination).

### Hyperparameters vs. Learned Parameters

Learned parameters ($\theta$: weights, biases) are found by gradient descent. **Hyperparameters** — learning rate $\eta$, batch size $B$, network depth, dropout rate $p$, weight decay $\lambda$, number of epochs — are set *before* training and are not learned by the optimizer. Tuning them is part art, part systematic search (grid search, random search, Bayesian optimization).

### → Next

Phase I is complete. We have the full pipeline: data → representation → model → loss → optimization → training. We can train a neural network for a specific task — *if* we have enough labeled data.

But labels are expensive. Annotating millions of images by hand is impractical, and for many tasks labels simply don't exist. How do you train a model that needs billions of examples without manually labeling any of them? The answer is a paradigm shift: self-supervised learning and foundation models. Phase II begins here.
