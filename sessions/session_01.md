---
id: 001
title: Data, Representations, and the First Model
summary: From raw sensor readings to feature spaces, and building the first model capable of learning — the perceptron and the MLP.
tags: [foundations, data, mlp, perceptron, activation-functions]
duration: 90 min
learning_goals:
  - Define feature spaces for tabular, image, and text data.
  - Understand the formal learning setup (X, Y, dataset, hypothesis).
  - Explain perceptrons, XOR, and why we need MLPs.
  - Describe common activation functions and why nonlinearity matters.
---

## Opening: What Does a Robot Actually See?

The Unitree Go2 produces a continuous stream of sensor readings. At any given moment, a single snapshot looks roughly like this:

```
timestamp  | FL_hip  | FL_thigh | FL_calf | ... | acc_x  | acc_y  | acc_z
-----------+---------+----------+---------+-----+--------+--------+--------
0.000      |  0.021  |  0.873   | -1.520  | ... |  0.12  | -0.03  |  9.81
0.002      |  0.019  |  0.871   | -1.518  | ... |  0.14  | -0.02  |  9.80
0.004      |  0.022  |  0.875   | -1.521  | ... |  0.11  | -0.04  |  9.81
```

Twelve joints × (position + velocity + torque) = 36 values. Three IMU axes × (acceleration + angular velocity) = 6 values. Add foot contact flags, battery state, and timestamp: around 48 values per timestep, arriving at 500 Hz.

**The core problem:** a machine learning model cannot work with a CSV file directly. It needs numbers organized in a specific mathematical structure — a *feature vector*. And once those numbers exist, we need a model that can learn patterns from them.

This session answers two questions:
1. How do we turn raw data — sensor readings, images, text — into a form a model can process?
2. What is the simplest model capable of learning, and why does it fall short on its own?

---

## From Raw Data to Numbers

Computers operate on numbers, so every modality needs a *feature space*: a mathematical representation that captures relevant information while discarding irrelevant variation.

### Tabular Data

Tabular data is the simplest format — rows are samples, columns are features. Our robot's sensor stream is exactly this. If there are $d$ features per sample, each sample is a vector:

$$\mathbf{x} \in \mathbb{R}^d$$

The entire dataset of $N$ samples is a matrix:

$$X \in \mathbb{R}^{N \times d}$$

For the Go2 with 48 features and one second of data at 500 Hz: $X \in \mathbb{R}^{500 \times 48}$. Each row is one timestep; each column is one sensor channel.

### Images as Tensors

A 224×224 RGB image is not a flat list of numbers — it is a three-dimensional tensor:

$$\mathbf{X}_{\text{img}} \in \mathbb{R}^{224 \times 224 \times 3}$$

That's $224 \times 224 \times 3 = 150{,}528$ values. The three dimensions correspond to height, width, and color channel (Red, Green, Blue), where each value is typically normalized to $[0, 1]$ before training.

**The structure matters.** Nearby pixels are correlated — the sky tends to be blue in large connected regions, edges form continuous lines. Flattening the image into a single vector of length 150,528 would destroy this spatial structure. Session 5 covers architectures (CNNs, Vision Transformers) that explicitly exploit it. For now, understand that the tensor shape encodes the spatial organization of the data.

### Text as Sequences

Text is a string of discrete symbols — words, subwords, characters. Neural networks require continuous inputs, so text must be converted in two steps:

1. **Tokenization:** split the string into atomic units called *tokens* and map each token to an integer index. The token "robot" might map to index 14293 in a vocabulary of 50,000 entries.
2. **Embedding:** replace each integer index with a learned continuous vector $\mathbf{e} \in \mathbb{R}^{d_{\text{emb}}}$ from an embedding table $E \in \mathbb{R}^{|V| \times d_{\text{emb}}}$.

The full details of tokenization and embedding come in Session 6. The key point: text ultimately becomes a sequence of continuous vectors, one per token.

### Preprocessing and Normalization

Raw features often live on very different scales. Joint angles range from $-\pi$ to $\pi$ (roughly ±3), while motor currents range from 0 to 20 amperes. Training with unnormalized features causes optimization problems: gradients with respect to large-scale features dominate, and gradients with respect to small-scale features are tiny — the loss landscape becomes elongated and poorly conditioned. This slows convergence dramatically.

The standard fix is *standardization*: subtract the mean and divide by the standard deviation so every feature has zero mean and unit variance:

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$ across the training set.

**Critical rule:** always compute $\mu_j$ and $\sigma_j$ on the *training set only*, then apply those same statistics to validation and test data. Computing statistics on the full dataset would leak information about validation/test samples into training — a subtle but serious mistake.

---

## Visualization: Why We Standardize Features

<img src="media/session_01/standardization.png" alt="Raw vs standardized feature clouds with loss contours" style="max-width:100%;border-radius:10px;">

*Features on very different scales give elongated loss contours and slow, zig-zagging descent; standardizing to zero mean and unit variance makes the contours round, so gradient descent heads straight to the minimum.*

---

## The Formal Learning Setup

We now have numbers. The next question is: what are we trying to do with them? The following notation carries through the entire series — every model, every algorithm, every experiment is described in these terms.

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | **Input space** — the set of all possible inputs (e.g., $\mathbb{R}^{48}$ for sensor vectors) |
| $\mathcal{Y}$ | **Output space** — the set of all possible outputs (e.g., $\{0,1\}$ for binary classification, $\mathbb{R}$ for regression) |
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ | **Dataset** — $N$ input-output pairs |
| $p(\mathbf{x}, y)$ | **True data distribution** — unknown; we only observe samples from it |
| $f_\theta : \mathcal{X} \rightarrow \mathcal{Y}$ | **Hypothesis** — a parameterized function we want to learn |
| $\theta$ | **Parameters** — the numbers we adjust during training |

The dataset $\mathcal{D}$ is assumed to consist of samples drawn *independently and identically distributed* (i.i.d.) from $p(\mathbf{x}, y)$. This is an assumption that real-world data often violates — but it is the foundation for most theoretical analysis.

**The goal of learning** is to find parameters $\theta$ such that $f_\theta(\mathbf{x}) \approx y$ — not just on $\mathcal{D}$, but on *new* inputs drawn from the same distribution. This is the distinction between *memorization* and *generalization*. A model that memorizes training data but fails on new inputs is useless. Session 3 addresses how to measure and improve generalization.

For the Go2 motor controller: $\mathcal{X} = \mathbb{R}^{48}$ (sensor readings), $\mathcal{Y} = \mathbb{R}^{12}$ (target joint positions).

---

## The Perceptron

The simplest instantiation of $f_\theta$: a single artificial neuron — the *perceptron* (Rosenblatt, 1958). Given input $\mathbf{x} \in \mathbb{R}^d$:

$$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

$$\hat{y} = \sigma(z)$$

where $\mathbf{w} \in \mathbb{R}^d$ are the *weights*, $b \in \mathbb{R}$ is the *bias*, and $\sigma$ is an *activation function*. The parameters are $\theta = \{\mathbf{w}, b\}$.

Geometrically, $z = 0$ defines a **hyperplane** in $\mathbb{R}^d$: the set of points equidistant from the two classes. The weights $\mathbf{w}$ define the orientation of this hyperplane; the bias $b$ shifts it from the origin.

The AND gate from Session 0 is exactly this: $w_1 = w_2 = 1$, $b = -1.5$, with a step activation $\sigma(z) = \mathbb{1}[z \geq 0]$.

### The Perceptron Learning Rule

Rosenblatt's original algorithm: for each training example, if the prediction is wrong, nudge the weights toward the correct answer.

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (y_i - \hat{y}_i) \cdot \mathbf{x}_i$$
$$b \leftarrow b + \eta \cdot (y_i - \hat{y}_i)$$

where $\eta > 0$ is the *learning rate*. The update is intuitive: if we predicted too low, we increase $\mathbf{w}$ in the direction of $\mathbf{x}_i$, making future predictions larger for similar inputs.

**Convergence guarantee:** if the training data are *linearly separable*, the perceptron learning rule is guaranteed to converge to a correct solution in a finite number of steps. If the data are *not* linearly separable, the algorithm never converges — it oscillates indefinitely. This is not a flaw in the algorithm; it reveals a fundamental limitation of the model itself.

---

## Visualization: The Perceptron Learns Its Boundary

<video src="media/session_01/perceptron_boundary.mp4" controls loop muted playsinline style="max-width:100%;border-radius:10px;"></video>

*The learning rule nudges the decision line after each misclassified point until the two classes are separated. The red arrow is the weight vector $\mathbf{w}$, which always stays orthogonal to the boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$.*

---

## Why One Perceptron Isn't Enough: XOR

The XOR function outputs 1 when exactly one of two binary inputs is 1:

| $x_1$ | $x_2$ | XOR |
|--------|--------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot the four points in 2D with their labels (circle = 0, cross = 1):

```
x₂
1 |  ×  ○
  |
0 |  ○  ×
  +--------
     0  1   x₁
```

Draw a single straight line that separates the circles from the crosses. It is impossible. The two classes are not linearly separable — no choice of $\mathbf{w}$ and $b$ works. Minsky & Papert (1969) proved this formally. The decision boundary of a single perceptron is *always* a hyperplane, regardless of the activation function. A hyperplane cannot carve the XOR pattern out of $\mathbb{R}^2$.

**Why does this matter for the robot?** Real perception tasks — detecting a staircase from depth images, classifying terrain type, identifying a door handle — are far more complex than XOR, yet they share its essential character: the classes are not linearly separable in the raw input space. If the simplest non-trivial logical function already defeats a single perceptron, raw pixels certainly will. The lesson is not that perceptrons are useless, but that we need a way to *reshape* the input space until the problem becomes linearly separable. That reshaping is exactly what hidden layers do.

---

## Visualization: A Hidden Layer Warps the Space

<video src="media/session_01/mlp_warp.mp4" controls loop muted playsinline style="max-width:100%;border-radius:10px;"></video>

*One hidden layer $\mathbf{h} = \tanh(W\mathbf{x} + \mathbf{b})$ bends the input plane — a linear change of basis followed by the tanh nonlinearity — until the XOR classes become linearly separable in the latent space, where a single straight line works.*

---

## Multi-Layer Perceptrons

The solution to XOR is to stack perceptrons into *layers*. Instead of mapping the input straight to the output, we first map it to an intermediate **hidden representation**, then map that to the output.

### Solving XOR Explicitly

XOR can be written as the logical combination $(x_1 \text{ OR } x_2) \text{ AND } (x_1 \text{ NAND } x_2)$. Two hidden neurons compute the two parts, and one output neuron combines them — all with step activations:

| Neuron | Weights | Bias | Computes |
|---|---|---|---|
| $h_1$ | $[1, 1]$ | $-0.5$ | OR |
| $h_2$ | $[-1, -1]$ | $1.5$ | NAND |
| $\hat{y}$ | $[1, 1]$ | $-1.5$ | $h_1$ AND $h_2$ |

Tracing all four inputs confirms it reproduces XOR exactly. The crucial insight: in the *hidden* space $(h_1, h_2)$, the four original points have been repositioned so that the two classes *are* now linearly separable. The hidden layer learned a new representation in which the problem became easy. A single perceptron could not do this; two layers can.

### The General MLP

An MLP with two hidden layers computes:

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{\mathbf{y}} = W_3 \mathbf{h}_2 + \mathbf{b}_3$$

where $W_k \in \mathbb{R}^{d_k \times d_{k-1}}$ are weight matrices, $\mathbf{b}_k$ are bias vectors, and $\sigma$ is applied element-wise. The hidden vectors $\mathbf{h}_1, \mathbf{h}_2$ are *learned representations* — progressively transformed versions of the input that make the final output easy to compute.

### Dimension Bookkeeping and Parameter Counting

Getting the matrix shapes right is half the battle when implementing networks. For a Go2 controller with input dimension 48, two hidden layers of 256 units, and output dimension 12:

| Layer | Shape of $W$ | Weights | Biases | Total |
|---|---|---|---|---|
| Input → $h_1$ | $256 \times 48$ | 12,288 | 256 | 12,544 |
| $h_1 \to h_2$ | $256 \times 256$ | 65,536 | 256 | 65,792 |
| $h_2 \to$ output | $12 \times 256$ | 3,072 | 12 | 3,084 |
| **Total** | | | | **81,420** |

About 81,000 parameters — every one of them adjusted during training. This is tiny by modern standards (large language models have hundreds of billions), but it is already far beyond what any human could set by hand. That is precisely why we need *learning*.

<img src="media/session_01/mlp_architecture.png" alt="MLP architecture 48-256-256-12 with per-layer parameter counts" style="max-width:100%;border-radius:10px;">

*The Go2 motor controller as an MLP: 48 sensor inputs → two hidden layers of 256 units (ReLU) → 12 joint outputs, fully connected, 81,420 parameters in total.*

### How Deep, How Wide?

Practical rules of thumb:

- More layers allow more abstract representations but make optimization harder and require more data.
- More units per layer increase capacity but also the risk of overfitting (Session 3).
- A common starting point for tabular problems: 2–4 hidden layers of 64–512 units.
- The output layer is fixed by the task: $K$ neurons for $K$-class classification, $k$ neurons for $k$-dimensional regression.

### The Universal Approximation Theorem

A remarkable result justifies all of this. An MLP with a single hidden layer and enough neurons can approximate *any* continuous function on a compact domain to arbitrary accuracy (Cybenko, 1989; Hornik, 1991).

Two caveats keep this from being the end of the story. First, "enough neurons" can mean an astronomically wide layer. Second, the theorem guarantees that a good approximation *exists* — not that gradient descent will *find* it. In practice, deeper networks represent complex functions far more efficiently than very wide shallow ones, and they tend to be easier to train. Depth, not just width, is what makes deep learning work.

---

## Activation Functions

Activation functions are the source of a network's expressive power. Without them, stacking layers is pointless: a composition of linear maps is itself just a linear map, so a 100-layer linear network has exactly the representational power of a single perceptron. The nonlinearity $\sigma$ is what lets each layer bend the space.

$$\text{Step}(z) = \mathbb{1}[z \geq 0] \qquad \text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad \text{ReLU}(z) = \max(0, z)$$

| Function | Range | Derivative behavior | Typical use |
|---|---|---|---|
| Step | $\{0,1\}$ | zero everywhere (non-differentiable) | historical only |
| Sigmoid | $(0,1)$ | saturates → vanishing gradients | binary output probability |
| Tanh | $(-1,1)$ | zero-centered, still saturates | older hidden layers |
| ReLU | $[0,\infty)$ | 1 if $z>0$, else 0 | **default for hidden layers** |
| Leaky ReLU | $\mathbb{R}$ | small slope for $z<0$ | fixes "dead" neurons |
| GELU | $\mathbb{R}$ | smooth, used in transformers | modern default in LLMs |

**The vanishing gradient problem.** Sigmoid and tanh flatten out for large $|z|$: their derivative approaches zero. When many such layers are stacked, gradients shrink multiplicatively during backpropagation (Session 2) until the early layers barely learn. ReLU avoids this for positive inputs because its derivative is exactly 1 — one of the main reasons deep networks became trainable.

**ReLU's failure mode.** A neuron whose input is always negative outputs zero forever and receives zero gradient — a "dead" ReLU. Leaky ReLU, which leaks a small negative slope, is a common fix.

### Softmax for the Output

For multi-class classification, the output layer uses **softmax**, which turns a vector of scores into a probability distribution:

$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The outputs are positive and sum to 1, so they can be read as class probabilities. Softmax pairs naturally with the cross-entropy loss introduced in Session 2.

---

## Visualization: Activation Functions and Their Derivatives

<img src="media/session_01/activation_functions.png" alt="Activation functions and their derivatives" style="max-width:100%;border-radius:10px;">

*Solid = function, dashed = derivative. The sigmoid and tanh derivatives vanish at the extremes — the saturation that bends the plane in the hidden-layer visualization — while ReLU's derivative stays exactly 1 for positive inputs, which is why it resists vanishing gradients.*

---

## Representation Learning: The Real Payoff

Step back from the mechanics and notice what hidden layers actually *do*. They don't just add parameters — they learn a new coordinate system for the data. Geometrically, each layer **deforms the coordinate space itself**: a linear map (a change of basis) followed by a nonlinear activation that stretches and folds the plane, so that data which was hopelessly tangled in the input space becomes neatly separable in the transformed space — exactly the warping shown in the hidden-layer visualization above. In the XOR example, the hidden layer moved the points until a straight line could separate them. In a vision network, the first layers learn to detect edges, later layers combine edges into textures and shapes, and the final layers respond to whole objects (Session 5 makes this concrete).

This is the central idea of deep learning: **we do not hand-design features anymore; the network learns the features and the decision rule jointly, end to end.** Everything that follows in this series — CNNs, transformers, multimodal models — is a different architecture for learning useful representations of a different kind of data.

---

## Connecting to the Robot Dog

The MLP is not a toy here — it is a genuine motor controller. An early but effective locomotion policy for a quadruped is exactly an MLP:

- **Input** ($\mathbb{R}^{48}$): joint positions, joint velocities, IMU orientation and angular velocity, plus the previous action and a velocity command.
- **Hidden layers**: two layers of 256 units with ReLU (the ~81,000-parameter network counted above).
- **Output** ($\mathbb{R}^{12}$): target joint positions, sent to low-level motor controllers.

Run this network at a few hundred hertz and you have a feedback policy that maps "what the body is doing right now" to "what the joints should do next." The weights are not hand-tuned — they are discovered by reinforcement learning in simulation (Session 10). For now, the takeaway is that the simplest deep model from this session is already enough to make a robot walk; everything else in the series is about perceiving the world well enough to tell it *where* to walk.

---

## Further Reading

**Start here** *(accessible introductions)*
- 3Blue1Brown: "But what *is* a neural network?" — [youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk) — best visual intuition for the foundations
- Andrej Karpathy: "The spelled-out intro to neural networks" — [youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0) — builds a micrograd engine step by step
- Google: Machine Learning Crash Course — [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course) — hands-on with feature representations

**Go deeper** *(technical references)*
- Goodfellow, Bengio & Courville: *Deep Learning*, Ch. 6 — [deeplearningbook.org](https://www.deeplearningbook.org) — formal treatment of feedforward networks
- Minsky & Papert: *Perceptrons* (1969) — the original proof of XOR's linear inseparability
- Cybenko: "Approximation by superpositions of a sigmoidal function" (*Math. of Control*, 1989) — the universal approximation theorem

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. The Go2 sends joint angles (≈ ±3 rad) and motor currents (0–20 A) to a model. What preprocessing step is needed, and why does training suffer without it?
2. Why must normalization statistics ($\mu_j, \sigma_j$) be computed on the training set only? What goes wrong if you use the whole dataset?
3. A single perceptron cannot solve XOR. Sketch the four points in 2D and explain geometrically why no single line separates the classes.
4. Write down weights and biases for a two-hidden-unit MLP that computes XOR, and verify it on all four inputs.
5. An MLP has input dimension 48, two hidden layers of 128 units, and output dimension 12. How many parameters does it have?
6. What happens to an MLP's representational power if every activation function is removed? Why?
7. Why is ReLU usually preferred over sigmoid for hidden layers in deep networks?

---

### → Next

We can define a model and compute its forward pass — but the weights start random, so the model knows nothing. How do we systematically find weights that make the model produce correct outputs? The answer is one of the most important algorithms in modern AI: gradient descent and backpropagation. Session 2 derives both from first principles.
