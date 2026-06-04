---
id: 002
title: Learning as Optimization
summary: How a model goes from knowing nothing to being useful — loss functions, gradient descent, backpropagation, and the optimizers that make it all work.
tags: [foundations, optimization, training, gradient-descent, backpropagation]
duration: 90 min
learning_goals:
  - Define and choose loss functions (MSE, cross-entropy) and understand their probabilistic meaning.
  - Picture the loss landscape and derive the gradient descent update.
  - Explain backpropagation as the chain rule and trace it through a worked example.
  - Name the key optimizers (SGD, momentum, Adam) and know when to use each.
  - Recognize common training pathologies and the defenses against them.
---

## Opening: A Network That Knows Nothing

At the end of Session 1 we had a Go2 controller: an MLP with about 81,000 parameters mapping 48 sensor values to 12 target joint positions. But those 81,000 numbers were initialized *randomly*. If we put this network on the robot right now, the legs would jerk in meaningless directions — it has the right *shape* to be a controller but none of the right *values*.

This session is about the missing step: how do we move from 81,000 random numbers to 81,000 numbers that make the robot walk? The answer is **optimization** — and the same three ideas (a loss, a gradient, an update rule) train every model in this series, from this tiny controller to a billion-parameter language model.

---

## What Learning Means

Learning means adjusting the parameters $\theta$ so that predictions $f_\theta(\mathbf{x})$ match the true targets $y$. To make "match" precise we need a number that measures how wrong the model is: a **loss function** $\mathcal{L}(f_\theta(\mathbf{x}), y) \in \mathbb{R}_{\geq 0}$.

Learning is then formally the search for the parameters that make the average loss over the dataset as small as possible:

$$\theta^* = \arg\min_\theta \; \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

This objective is called **empirical risk minimization** — "empirical" because we minimize the loss on the data we actually have, as a stand-in for the *true* risk over the whole distribution $p(\mathbf{x}, y)$, which we can never see. The gap between the two is the subject of Session 3.

A loss function must satisfy two properties to be usable: it must be (1) **non-negative**, so that zero means "perfect," and (2) **differentiable** with respect to $\theta$ almost everywhere, so that we can compute gradients.

---

## Loss Functions

The choice of loss encodes what we mean by "wrong." Different tasks call for different losses.

### Mean Squared Error (Regression)

$$\mathcal{L}_\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|f_\theta(\mathbf{x}_i) - y_i\|^2$$

MSE penalizes large errors quadratically: an error of 2 costs four times as much as an error of 1. This is appropriate when the output is a continuous quantity and large mistakes are disproportionately bad — predicting joint torques for the Go2 is a natural fit.

MSE has a clean probabilistic justification: minimizing it is equivalent to **maximum likelihood estimation** under the assumption that the targets are corrupted by Gaussian noise. The quadratic penalty is the negative log of a Gaussian.

Its weakness is the same quadratic: a single large outlier dominates the sum. The **Huber loss** fixes this by behaving quadratically for small errors and linearly for large ones, making it robust to occasional bad labels.

### Cross-Entropy (Classification)

For binary classification with a model output $\hat{y} \in (0,1)$:

$$\mathcal{L}_\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

and for $K$ classes with one-hot labels and softmax outputs:

$$\mathcal{L}_\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

Cross-entropy is the negative log-likelihood of the correct class under the model's predicted distribution. It punishes *confident and wrong* predictions brutally: predicting $\hat{y} = 0.99$ when the true label is 0 costs $-\log(0.01) \approx 4.6$, whereas a hesitant wrong guess of $0.6$ costs only $-\log(0.4) \approx 0.9$. This pressure to be calibrated is exactly what we want from a classifier.

**Rule of thumb:** MSE for regression, cross-entropy for classification. Using MSE on a classification problem is a classic beginner mistake — it produces weak gradients precisely when the model is most wrong, so training stalls.

---

## The Loss Landscape

It helps to picture the loss as a *surface* sitting above the parameter space. Every possible setting of $\theta$ is a point on the ground; the height of the surface above it is the loss. Training is a walk downhill toward a valley.

For a single linear neuron with MSE, this surface is a smooth bowl — **convex**, with one global minimum. Gradient descent is guaranteed to reach it. For any network with hidden layers, the surface is **non-convex**: a rugged terrain of valleys, ridges, and saddle points, with many local minima.

This sounds alarming, but high-dimensional intuition saves us. In a space of 81,000 dimensions, true local minima that trap optimization are rare; far more common are **saddle points**, which gradient descent can usually escape because it only needs *one* downhill direction. In practice, most of the many minima found by gradient descent are roughly equally good. We are not looking for *the* perfect valley — just a low one.

---

## Gradient Descent

The **gradient** $\nabla_\theta \mathcal{L}$ is the vector of partial derivatives of the loss with respect to every parameter. It points in the direction of steepest *increase* of the loss. To go downhill, we step in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

The scalar $\eta > 0$ is the **learning rate** — the single most important hyperparameter in deep learning.

- Too large: the steps overshoot the valley; the loss oscillates or diverges to infinity.
- Too small: the steps are tiny; training takes forever and may stall on a plateau.
- A common starting point is $\eta \approx 10^{-3}$, then tune from there.

### A Worked Single Step

Take a trivial model $f(x) = w x$ with one parameter, one data point $(x, y) = (2, 6)$, and squared-error loss $\mathcal{L}(w) = (wx - y)^2$. Suppose $w = 1$.

The prediction is $f(2) = 2$, so the error is $2 - 6 = -4$. The derivative is

$$\frac{d\mathcal{L}}{dw} = 2(wx - y)\,x = 2(-4)(2) = -16.$$

With $\eta = 0.1$ the update is $w \leftarrow 1 - 0.1 \cdot (-16) = 2.6$. The new prediction $f(2) = 5.2$ is much closer to 6. One more step and we are nearly exact. That is gradient descent in miniature; everything else is the same idea with millions of parameters at once.

### Stochastic and Mini-Batch Gradient Descent

Computing the gradient over the *entire* dataset for every step is expensive — for a robot, the dataset may be millions of timesteps. **Mini-batch gradient descent** estimates the gradient from a small random subset $\mathcal{B}_t$ of $B$ examples:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla_\theta \mathcal{L}_i$$

The estimate is noisy, but cheap and — usefully — the noise helps the optimizer jump out of sharp, brittle minima toward flatter ones that tend to generalize better. Typical batch sizes are 32–512. One full pass over the training set is an **epoch**. The extreme of $B = 1$ is classic stochastic gradient descent (SGD); the other extreme of $B = N$ is full-batch.

---

## Backpropagation

Gradient descent needs $\nabla_\theta \mathcal{L}$: the derivative of the loss with respect to *every* weight in the network. **Backpropagation** computes all of them efficiently in a single backward sweep, and it is nothing more than the **chain rule** applied to the network's computational graph.

Consider a two-layer network with scalar output:

$$z_1 = W_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{h} = \sigma(z_1), \quad \hat{y} = \mathbf{w}_2^\top \mathbf{h} + b_2, \quad \mathcal{L} = (\hat{y} - y)^2$$

The backward pass works right to left, reusing each quantity as it goes:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y), \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}}\, \mathbf{h}$$

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}}\, \mathbf{w}_2 \odot \sigma'(z_1), \qquad \frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial z_1}\, \mathbf{x}^\top$$

### Why It Is Efficient

The key economy: the gradient with respect to an early weight reuses the gradient already computed for the later layers — the term $\partial \mathcal{L} / \partial z_1$ flows backward and multiplies in. A single forward pass followed by a single backward pass computes *all* gradients simultaneously, at roughly twice the cost of the forward pass alone. Computing each derivative independently would be hopelessly expensive; backprop makes deep learning tractable.

### A Tiny Numeric Example

Let $\hat{y} = 0.8$, $y = 1$, and a hidden activation $h = 0.5$ feeding the output weight $w_2$. Then $\partial \mathcal{L} / \partial \hat{y} = 2(0.8 - 1) = -0.4$, and $\partial \mathcal{L} / \partial w_2 = -0.4 \times 0.5 = -0.2$. The negative sign says: increasing $w_2$ would raise $\hat y$ toward the target, lowering the loss — so the descent step will indeed increase $w_2$. Every weight in a billion-parameter model is updated by exactly this logic.

---

## Automatic Differentiation

You will never write these derivatives by hand. Modern frameworks — PyTorch, JAX, TensorFlow — record every operation in the forward pass as a graph and then apply the chain rule automatically in reverse. This is **reverse-mode automatic differentiation**, and backpropagation is its special case for neural networks.

The practical consequence is enormous: you define only the *forward* computation, and the gradients come for free, exactly and automatically. This is why researchers can prototype novel architectures in an afternoon — the hardest part, the gradients, is handled by the framework.

---

## Practical Optimizers

Plain gradient descent works but can be slow and fragile. A few refinements dominate practice.

### SGD with Momentum

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla_\theta \mathcal{L}_t, \qquad \theta_{t+1} = \theta_t - \eta \, \mathbf{v}_{t+1}$$

Momentum (typically $\beta = 0.9$) accumulates a running velocity. In directions where the gradient consistently points the same way, speed builds up; in directions where it oscillates, the swings cancel. The effect is like a heavy ball rolling downhill — faster and steadier than a light one.

### Adam

Adam keeps two per-parameter running averages: the mean of the gradient ($m_t$) and the mean of its square ($v_t$):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta\mathcal{L}_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta\mathcal{L}_t)^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The bias-correction terms $\hat m_t, \hat v_t$ undo the fact that $m_t$ and $v_t$ start at zero and are therefore biased low early in training. Dividing the step by $\sqrt{\hat v_t}$ gives each parameter its *own* effective learning rate: parameters with large, noisy gradients take small steps, and parameters with small, steady gradients take larger ones. Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$ work across a remarkable range of problems, which is why Adam is the most widely used optimizer in deep learning. **AdamW**, a variant that decouples weight decay from the gradient step, is the standard for training large models today.

---

## Learning Rate Schedules

A single fixed learning rate is rarely optimal: you want big steps early, when far from a minimum, and small steps late, for fine-tuning. Common schedules:

- **Step decay:** multiply $\eta$ by a factor (e.g. halve it) every $k$ epochs.
- **Cosine annealing:** smoothly decay $\eta$ from $\eta_\text{max}$ to near zero following a cosine curve.
- **Warmup:** *increase* $\eta$ linearly for the first few hundred steps before decaying. Early gradients on randomly initialized weights are large and unreliable; warmup prevents an early divergence and is essential for training transformers.

---

## Weight Initialization

Where the walk *starts* matters. If all weights are initialized to zero, every neuron in a layer computes the same output and receives the same gradient — the symmetry is never broken and the layer behaves like a single neuron forever. Initialization must therefore be random, but with carefully chosen scale: too large and activations explode, too small and they vanish.

**He initialization** (for ReLU networks) draws weights from

$$w \sim \mathcal{N}\!\left(0, \frac{2}{d_\text{in}}\right)$$

which keeps the variance of activations roughly constant from layer to layer. **Xavier (Glorot) initialization**, tuned for tanh/sigmoid, uses a variance based on $d_\text{in} + d_\text{out}$. Both are the defaults in every major framework, so you rarely set them by hand — but knowing *why* they exist explains a whole class of training failures.

---

## Common Training Pathologies

When training misbehaves, the symptom usually points to one of a handful of causes:

- **Loss diverges to NaN/infinity:** learning rate too high, or no input normalization. Lower $\eta$, add warmup, check the data.
- **Loss is flat from the start:** learning rate too low, dead ReLUs, or vanishing gradients in a deep sigmoid/tanh stack.
- **Exploding gradients:** common in deep or recurrent networks; the fix is **gradient clipping**, which caps the gradient norm at a threshold.
- **Loss decreases then plateaus far from zero:** may need a schedule, more capacity, or it may simply be the limit of the data (Session 3).

A cheap sanity check before a long run: overfit a *single* mini-batch. A correct network and training loop should drive the loss on a handful of examples to nearly zero. If it cannot, the bug is in the model or the gradients, not the data.

---

## Connecting to the Robot Dog

Return to the random controller from the opening. To train it in a supervised way, we would collect a dataset of (sensor reading, expert joint command) pairs, use **MSE loss** because joint positions are continuous, compute gradients by **backpropagation**, and update with **Adam** at $\eta = 10^{-3}$ over many mini-batches until the loss flattens. The 81,000 random numbers gradually become a working policy.

In practice, locomotion has no "expert joint commands" to imitate — there is no labeled dataset of correct walking. That is why Session 10 replaces the supervised loss with a *reward* and learns by trial and error. But the optimization machinery is identical: a scalar objective, gradients via backprop, an Adam-style update. Master it here and it never changes for the rest of the series.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "Gradient descent, how neural networks learn" — [youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w) — outstanding visual explanation
- Andrej Karpathy: "The spelled-out intro to backpropagation" — [youtube.com/watch?v=q8SA3rM6ckI](https://www.youtube.com/watch?v=q8SA3rM6ckI) — derives backprop by building it
- distill.pub: "Why Momentum Really Works" — [distill.pub/2017/momentum](https://distill.pub/2017/momentum) — interactive visual explanation

**Go deeper** *(technical references)*
- Rumelhart, Hinton & Williams: "Learning representations by back-propagating errors" (*Nature*, 1986) — the original backprop paper
- Kingma & Ba: "Adam: A Method for Stochastic Optimization" (ICLR 2015) — [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Goodfellow et al.: *Deep Learning*, Ch. 4 (numerical optimization) and Ch. 8 (optimization for training)

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. Why is cross-entropy preferred over MSE for a binary classification task with labels 0 and 1? Think about the gradient when the model is confidently wrong.
2. What probabilistic assumption makes minimizing MSE equivalent to maximum likelihood estimation?
3. If the learning rate is far too large, what does the training-loss curve look like over epochs? What if it is far too small?
4. Walk through one backward step of the chain rule for the two-layer network in the text. Why does computing all gradients cost only about twice a forward pass?
5. Adam maintains two running averages per parameter. What are they, and what does each contribute to the update?
6. Why does initializing all weights to zero make a layer fail to learn?
7. A training run produces NaN loss after a few steps. List two likely causes and a fix for each.

---

### → Next

We can now define a model, choose a loss, and optimize the weights via gradient descent and backpropagation. But a model that minimizes training loss is not automatically a *good* model: it may have memorized the training set and fail on anything new, or it may diverge before it ever converges. Session 3 covers the practical side of training — overfitting, validation, regularization, and the diagnostics that separate a model that works from one that only looks like it does.
