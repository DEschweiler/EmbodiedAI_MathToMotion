---
id: 003
title: Training in Practice
summary: Overfitting, underfitting, validation, regularization, and reading curves.
tags: [foundations, training, regularization]
learning_goals:
  - Spot overfitting vs. underfitting via learning curves.
  - Use train/val/test splits correctly and understand k-fold cross-validation.
  - Apply dropout, weight decay, and batch normalization.
  - Understand data augmentation as a regularization strategy.
  - Read training diagnostics and apply early stopping.
---

Sessions 1 and 2 built the theoretical machinery: model, loss, optimizer. This session covers what actually happens when you run the training loop — and the many ways it can go wrong. The gap between "the algorithm is correct" and "the model works well" is filled by the techniques here.

## The Central Problem: Generalization

A model with enough parameters can memorize any training set — driving the training loss to exactly zero while learning nothing generalizable. That's **overfitting**: perfect on training data, poor on new data. The opposite failure is **underfitting**: the model is too constrained to capture the underlying pattern.

The goal is never the lowest training loss. The goal is the lowest loss on *unseen data from the same distribution*.

### The Bias-Variance Tradeoff

Expected generalization error decomposes into three terms:

$$\mathbb{E}[(f_\theta(\mathbf{x}) - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

- **Bias**: error from wrong assumptions — a model too simple for the task. Corresponds to underfitting.
- **Variance**: error from sensitivity to training data fluctuations — a model that memorizes noise. Corresponds to overfitting.
- **Noise**: irreducible error from inherent randomness in the data.

Increasing model capacity reduces bias but increases variance. Regularization allows high-capacity models while limiting variance.

## Train / Validation / Test Splits

- **Training set** (~70%): the model learns from this.
- **Validation set** (~15%): monitored after each epoch to detect overfitting and tune hyperparameters. The model *never* updates weights based on this set.
- **Test set** (~15%): evaluated *once* at the very end — the unbiased final estimate of generalization.

A critical rule: normalization statistics must be computed on the training set only, then applied to validation and test sets. Computing them on the full dataset constitutes *data leakage*.

### K-Fold Cross-Validation

When data is scarce, a single validation split introduces high variance. K-fold cross-validation divides the training data into $K$ parts, trains $K$ times (using $K-1$ for training, 1 for validation), and averages the scores. Typical $K = 5$ or $10$. The test set remains untouched.

## Regularization

Regularization reduces generalization error without reducing model capacity.

### Weight Decay (L2 Regularization)

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{data} + \lambda \|\theta\|^2$$

The gradient update becomes $\theta_{t+1} = \theta_t(1 - 2\lambda\eta) - \eta \nabla_\theta \mathcal{L}_\text{data}$, shrinking weights at every step. Typical $\lambda \in [10^{-5}, 10^{-2}]$.

**L1 regularization** uses $\|\theta\|_1$ instead, encouraging *sparsity* — many weights go to exactly zero.

### Dropout

Randomly sets a fraction $p$ of neuron activations to zero during training:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{with probability } 1-p \end{cases}$$

The rescaling ensures the expected value of $\tilde{h}_i$ equals $h_i$. Dropout forces the network to learn redundant representations and can be interpreted as training an ensemble of $2^n$ sub-networks. Typical rates: $p = 0.1$–$0.3$ for fully connected layers.

### Batch Normalization

Normalizes activations within each mini-batch:

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \quad y_i = \gamma \hat{z}_i + \beta$$

where $\gamma$, $\beta$ are learned. Benefits: faster training, reduced sensitivity to initialization, slight regularization effect. For transformers, *Layer Normalization* (normalizing across features rather than the batch) is preferred.

### Data Augmentation

Creates modified copies of training examples — random crops, flips, color jitter for images; Gaussian noise on sensor readings for the robot. Free regularization: no additional annotation, only compute.

## Reading Learning Curves

| Pattern | Diagnosis | Action |
|--------|-----------|--------|
| Both losses falling together | Good training | Continue |
| Training ↓, validation ↑ | Overfitting | Add regularization, reduce capacity, more data |
| Both barely moving | Underfitting | Increase capacity, adjust learning rate |
| Both oscillating | Learning rate too high | Reduce $\eta$ |
| Validation plateaus early | Need LR decay | Apply schedule |

**Early stopping:** monitor validation loss and halt when it stops improving, using a patience parameter of $k$ epochs. The checkpoint at the best validation epoch is kept.

## Hyperparameters vs. Learned Parameters

Learned parameters ($\theta$: weights, biases) are found by gradient descent. **Hyperparameters** — learning rate, batch size, network depth, dropout rate, weight decay — are set before training. Tuning strategies: grid search, random search, Bayesian optimization. The learning rate is the single most important hyperparameter; tune it first.

## Connecting to the Robot Dog

The locomotion controller must generalize from training conditions to real deployment. Regularization is directly relevant:
- **Dropout** reduces reliance on specific sensor channels — robustness to sensor failures.
- **Noise augmentation** on sensor inputs mimics real sensor noise — better sim-to-real transfer.
- **Weight decay** keeps controller weights small — smoother, more stable motor commands.

These connections reappear concretely in Session 11.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "What is overfitting?" — search on YouTube — short intuitive explanation
- fast.ai: Lesson 1–2 — [course.fast.ai](https://course.fast.ai) — hands-on treatment of training pitfalls
- StatQuest (Josh Starmer): "Regularization" playlist — [youtube.com/@statquest](https://www.youtube.com/@statquest) — excellent visual intuitions for bias-variance and regularization

**Go deeper** *(technical references)*
- Srivastava et al.: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (JMLR 2014)
- Ioffe & Szegedy: "Batch Normalization" (ICML 2015) — [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
- Goodfellow et al.: *Deep Learning*, Ch. 7 (regularization for deep learning)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. A model achieves 99% training accuracy but only 65% validation accuracy. What is the diagnosis, and what are three possible remedies?
2. Why must normalization statistics always be computed on the training set only, and never on the full dataset?
3. Dropout rescales activations by $1/(1-p)$ during training. Why is this rescaling necessary?
4. Your training loss falls steadily but your validation loss plateaued 20 epochs ago. What should you do, and why?
5. What is the difference between a *learned parameter* and a *hyperparameter*? Give two examples of each from a typical MLP training setup.

---

### → Next

Phase I is complete. We have the full pipeline: data → representation → model → loss → optimization → regularized training. We can train a neural network on a labeled dataset — *if* we have enough labels. But labels are expensive. Session 4 covers the paradigm shift that solved this: self-supervised learning and foundation models.
