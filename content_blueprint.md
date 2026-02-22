# Embodied AI – From Math to Motion

### A Blog-Post Series on Building Intelligent Robots with Modern Machine Learning

---

# Post 0: The Big Picture

A four-legged robot stands in a hallway. Someone says: *"Go to the door and wait."* The robot turns, avoids a chair, identifies the door, stops, and reports: *"I've arrived."*

Nothing about this is magic. It's a stack of well-understood methods — linear algebra, optimization, neural networks, language models, control theory — wired together into a system. This series walks through every layer of that stack, from the mathematical foundations to the final integration on real hardware. The running example throughout: a Unitree Go2 quadruped robot.

## Two Interleaved Threads

Every post addresses two questions simultaneously:

**The "what" — ML fundamentals.** What are feature spaces, loss functions, transformers, foundation models? This thread provides transferable knowledge that applies far beyond robotics.

**The "so what" — practical application.** How does each concept map onto the robot dog? Optimization becomes gait training. Vision models become the dog's eyes. Language models become its brain.

## Why Machine Learning?

Consider writing a program that distinguishes cats from dogs in photographs. You might try hand-crafted rules — "cats have pointed ears" — but exceptions immediately break them. For tasks involving perception, language, or complex decision-making, explicit programming doesn't scale. Machine learning offers an alternative: instead of specifying rules, provide examples and let the system discover patterns.

A concrete proof of concept makes this tangible. An AND gate from two binary inputs can be implemented as a weighted sum with a threshold:

$$y = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Setting $w_1 = w_2 = 1$, $b = -1.5$ gives a perfect AND gate — trivially hand-crafted. Now try the same for a 224×224 RGB image (over 150,000 input values). Hand-crafting weights becomes impossible. We need an automated way to find them, and that's what learning algorithms provide.

## The Hierarchy: AI, ML, Deep Learning

*Artificial Intelligence* is the broadest term — any computational system solving tasks that typically require human intelligence. *Machine Learning* narrows this to systems that learn from data rather than being explicitly programmed. *Deep Learning* narrows further to models based on layered function compositions. Most of what this series covers falls under deep learning, but the foundational ML ideas apply throughout.

An important clarification: a neural network doesn't "know" what a cat is. It has learned to map certain numerical patterns to a label. Internal representations are statistical compressions of training data, not knowledge databases. This distinction matters enormously when discussing LLMs and their capabilities later.

## The Robot Dog as a System

The Unitree Go2 serves as a concrete reference system throughout. Its architecture maps onto three layers:

- **Cognitive layer:** An LLM-based planner that interprets natural language commands and decomposes them into subtasks. (→ Posts 6–9)
- **Motor layer:** ML models that have learned movement strategies — walking, trotting, obstacle avoidance. (→ Posts 1–3, 10)
- **Physical layer:** Control theory and mathematics that translate joint angles into motor currents. (→ Classical engineering)

Every method introduced in this series plugs into one of these layers.

## Three Phases, Twelve Posts

**Phase I – Foundations (Posts 1–3):** Data representations, neural networks, optimization, and the practicalities of training. The mathematical tools for everything that follows.

**Phase II – Building Intelligence (Posts 4–9):** Foundation models, CNNs, vision transformers, large language models, RLHF, multimodal systems, and agent architectures. The building blocks that give the robot perception, language, and reasoning.

**Phase III – Integration & Reality (Posts 10–11):** Reinforcement learning for locomotion, sim-to-real transfer, and the live test on real hardware.

| # | Title | Phase |
|---|-------|-------|
| 0 | The Big Picture | — |
| 1 | Data, Representations, and the First Model | I |
| 2 | Learning as Optimization | I |
| 3 | Training in Practice | I |
| 4 | The Data Problem: Annotation and Self-Supervision | II |
| 5 | Seeing: CNNs and Vision Transformers | II |
| 6 | Reading: From N-Grams to Language Models | II |
| 7 | Following Instructions: RLHF and Prompting | II |
| 8 | Seeing *and* Reading: Multimodal Models | II |
| 9 | The Dog Gets a Brain: Agents and Tool Use | II |
| 10 | Learning to Walk: Deep Reinforcement Learning | III |
| 11 | From Simulation to Reality and the Live Test | III |

Each post ends with a transition to the next — they are designed to be read in sequence.

### → Next

We've established why ML is necessary, how the terms relate, and how the robot dog's architecture is structured. But before any model can learn, raw data — pixels, sensor readings, text — must be converted into numbers. And once the data is numerical, we need a model that can do something with it. Post 1 covers both: how data becomes math, and the first neural network that operates on it.

---

# Post 1: Data, Representations, and the First Model

**Phase I – Foundations**

### Learning Goals

- Explain what a feature space is and why ML requires one.
- Describe tabular data, images, and text as input formats and express them mathematically.
- State the formal learning setup: input space $\mathcal{X}$, output space $\mathcal{Y}$, dataset, hypothesis $f_\theta$.
- Describe a perceptron as a weighted sum with activation function.
- Explain the XOR limitation and how MLPs overcome it.
- Sketch an MLP architecture and name common activation functions.

### From Raw Data to Numbers

A computer doesn't see images or read text — it operates on numbers. The first step in any ML pipeline is converting raw data into a numerical representation: a *feature space*.

For the robot dog: the camera produces RGB pixels (a 3D tensor), the IMU delivers acceleration vectors, and a voice command must be converted into a sequence of numbers (tokens). These are all different feature spaces.

**Tabular data** is the simplest format — rows are samples, columns are features. Example: the dog's joint angles, motor currents, and IMU readings over time. Mathematically, each sample is a vector $\mathbf{x} \in \mathbb{R}^d$.

**Images as grids.** A 224×224 RGB image is a tensor $\mathbf{X} \in \mathbb{R}^{224 \times 224 \times 3}$ — that's 150,528 values. This is high-dimensional but spatially structured. CNNs (Post 5) exploit this structure; MLPs ignore it.

**Text as sequences.** Text is a string of discrete symbols that must be mapped to integers (tokenization). Details come in Post 6; the key idea is that text ultimately becomes a sequence of numbers.

### The Formal Learning Setup

The notation that carries through the entire series:

- **Input space** $\mathcal{X}$: the set of all possible inputs (e.g., all possible 224×224 RGB images).
- **Output space** $\mathcal{Y}$: the set of all possible outputs (e.g., $\{0, 1\}$ for binary classification, or $\mathbb{R}$ for regression).
- **Dataset** $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$: a collection of $N$ input-output pairs drawn from an unknown distribution.
- **Hypothesis** $f_\theta : \mathcal{X} \rightarrow \mathcal{Y}$: a parameterized function. Learning consists of finding parameters $\theta$ such that $f_\theta(\mathbf{x})$ approximates $y$ well across $\mathcal{D}$.

On the robot dog: $\mathcal{X}$ could be the space of all joint-state + IMU vectors, $\mathcal{Y}$ the space of motor commands, and $f_\theta$ the neural network mapping one to the other.

### The Perceptron

The simplest instantiation of $f_\theta$: a single neuron. Given input $\mathbf{x} \in \mathbb{R}^d$, compute:

$$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

$$y = \sigma(z)$$

where $\mathbf{w} \in \mathbb{R}^d$ are weights, $b \in \mathbb{R}$ is a bias, and $\sigma$ is an activation function. The parameters $\theta = \{\mathbf{w}, b\}$ define a hyperplane in $\mathbb{R}^d$ — the perceptron classifies inputs based on which side of this hyperplane they fall on.

The AND gate from Post 0 is exactly this: $w_1 = w_2 = 1$, $b = -1.5$, with a step activation $\sigma(z) = \mathbb{1}[z \geq 0]$.

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

---

# Post 2: Learning as Optimization

**Phase I – Foundations**

### Learning Goals

- Define a loss function as a differentiable measure of model quality.
- Derive the gradient descent update rule and explain the role of the learning rate.
- Describe backpropagation as the chain rule applied to compute gradients through an MLP.
- Name practical optimizers (SGD, Adam) and their key properties.

### What Does "Learning" Mean?

Learning means adjusting $\theta$ so that predictions $f_\theta(\mathbf{x})$ match true labels $y$. "Match" is quantified by a **loss function** $\mathcal{L}(f_\theta(\mathbf{x}), y)$ — a differentiable scalar that measures how wrong the model is. Learning = minimizing this scalar over the dataset:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

### Loss Functions

**Mean Squared Error (regression):**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (f_\theta(\mathbf{x}_i) - y_i)^2$$

MSE penalizes large errors quadratically — a prediction that's off by 10 contributes 100× more than one that's off by 1.

**Binary Cross-Entropy (classification):**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

where $\hat{y}_i = f_\theta(\mathbf{x}_i) \in (0, 1)$ is the predicted probability. Cross-entropy penalizes *confident but wrong* predictions especially hard: if the model predicts $\hat{y} = 0.99$ but the true label is 0, the loss is $-\log(0.01) \approx 4.6$ — a strong corrective signal.

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

We can now define a model, choose a loss, and optimize weights via gradient descent with backpropagation. In theory, this is all we need. In practice, a host of problems await: the model might memorize training data rather than generalize, the training might diverge, or the hyperparameters might be wrong. Post 3 covers the practical side of training — the diagnostics and techniques that make the difference between a model that works and one that doesn't.

---

# Post 3: Training in Practice

**Phase I – Foundations**

### Learning Goals

- Define overfitting and underfitting; recognize both in training curves.
- Explain train/validation/test splits and why all three are necessary.
- Name regularization techniques (dropout, weight decay) and describe their mathematical formulation.
- Distinguish hyperparameters from learnable parameters.
- Interpret a loss-vs-epochs plot and apply early stopping.

### Overfitting vs. Underfitting

A model with enough parameters can memorize any training set — driving the training loss to zero while learning nothing generalizable. That's **overfitting**: perfect on training data, catastrophic on new data. Conversely, a model that's too small or poorly configured can't even fit the training data — **underfitting**.

The goal is never the best training loss; it's the best performance on *unseen data*. This is the bias-variance tradeoff: too simple → high bias (underfitting), too complex → high variance (overfitting).

### Train / Validation / Test

The dataset is split three ways:

- **Training set**: the model learns from this.
- **Validation set**: evaluated after each epoch — used to tune hyperparameters and detect overfitting. The model never trains on this data.
- **Test set**: evaluated once at the very end — the unbiased final evaluation. Never used for any decision during development.

On the robot dog: train on concrete and grass, validate on carpet, test on sand. Does it generalize?

### Regularization

**Dropout** randomly sets a fraction $p$ of neuron activations to zero during training:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{with probability } 1-p \end{cases}$$

The rescaling by $1/(1-p)$ ensures the expected value of $\tilde{h}_i$ remains equal to $h_i$. Dropout forces the network to learn redundant representations — no single neuron can become a critical bottleneck, which improves generalization.

**Weight decay** (L2 regularization) adds a penalty on the magnitude of weights to the loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|\theta\|^2 = \mathcal{L}_{\text{data}} + \lambda \sum_j \theta_j^2$$

This discourages large weights, effectively preferring simpler models. The hyperparameter $\lambda$ controls the strength of the penalty.

### Reading Training Curves

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

---

# Post 4: The Data Problem — Annotation and Self-Supervision

**Phase II – Building Intelligence**

### Learning Goals

- Explain the annotation bottleneck and describe strategies to address it.
- Define self-supervised learning: deriving training objectives from the data itself.
- Explain what a foundation model is: a large model pretrained on massive data for general-purpose use.
- Describe fine-tuning and linear probing as adaptation strategies.

### The Label Bottleneck

Phase I assumed that every training example has a label. In practice, this is the biggest scaling bottleneck. Medical images require expert annotation (slow, expensive). Robot sensor data would need per-terrain labels. Scaling supervised learning to billions of examples requires an alternative.

**Annotation strategies** range across a spectrum: manual annotation (gold standard, doesn't scale), weak supervision (heuristics generate noisy labels automatically), and — an elegant example — CAPTCHAs. Every "select all images with traffic lights" click is human annotation for a vision model's training set. User selections on image patches accumulate into heatmaps — rudimentary segmentations generated entirely from human annotation.

### Self-Supervised Learning

The most scalable solution: derive labels from the data itself.

**Masked Autoencoder:** Mask random patches of an image. Train the model to reconstruct the missing parts. The label is the original image — freely available.

**Contrastive Learning:** Create augmented views of the same image (crop, rotate, color-jitter). Train the model to recognize them as related (high similarity) while pushing unrelated images apart (low similarity). Formally, for a positive pair $(v_i, v_i^+)$ and negative pairs $(v_i, v_j^-)$:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(v_i, v_i^+) / \tau)}{\sum_j \exp(\text{sim}(v_i, v_j^-) / \tau)}$$

where $\text{sim}$ is cosine similarity and $\tau$ is a temperature parameter. No labels — just data augmentations.

**Next-Token Prediction:** Given a text sequence, predict the next token. The label is the text itself. (→ Post 6)

The key insight: none of these require human labels. They only require raw data — and the internet provides that at virtually unlimited scale.

### Foundation Models

A **foundation model** is the result of self-supervised pretraining on massive datasets. It learns general-purpose representations that transfer to many downstream tasks. Examples: GPT (text), DINO (images), CLIP (text + images).

On the robot dog: instead of training separate models from scratch for terrain classification, obstacle detection, and object recognition, take a single foundation model and adapt it.

### Fine-Tuning vs. Linear Probing

Two strategies for adapting a pretrained model $f_\theta$ with frozen backbone parameters $\theta_{\text{pre}}$:

**Fine-tuning:** Initialize with $\theta_{\text{pre}}$, then continue training all parameters on the downstream task. Flexible, but requires more data and compute. Risk of "catastrophic forgetting" — losing useful pretrained representations.

**Linear probing:** Freeze $\theta_{\text{pre}}$ entirely. Add a single linear layer $g_\phi(\mathbf{h}) = W\mathbf{h} + \mathbf{b}$ on top and only train $\phi$. Fast and simple, but limited to what the pretrained features already capture.

The tradeoff is clear: fine-tuning adapts the full model to the new task, linear probing leverages the pretrained representation as-is.

### → Next

We now know how to learn general representations from massive unlabeled data. But which model architecture works best for images? MLPs treat every pixel independently. For images, we need architectures that exploit spatial structure: convolutional neural networks for local patterns, and vision transformers for global context.

---

# Post 5: Seeing — CNNs and Vision Transformers

**Phase II – Building Intelligence**

### Learning Goals

- Define convolution mathematically and explain why it suits image data.
- Describe the CNN architecture (convolutional layers, pooling, fully connected layers).
- Explain why CNNs lack global context and how Vision Transformers (ViT) address this via attention.
- Describe the attention mechanism mathematically: queries, keys, values.

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

---

# Post 6: Reading — From N-Grams to Language Models

**Phase II – Building Intelligence**

### Learning Goals

- Explain tokenization and why tokens (not characters) are the basic unit of LLMs.
- Describe the N-gram model and conditional probability $P(\text{next token} \mid \text{context})$.
- Explain why N-grams hit a ceiling and how transformers overcome it.
- State the core of an LLM: a transformer trained on next-token prediction over massive text.
- Relate context window size to text quality and model capability.

### From Characters to Tokens

Text is a string of characters, but encoding individual characters is inefficient: the model would need to assemble words from letters, requiring enormous context windows. **Tokenization** groups frequent character sequences into tokens (roughly "syllables"). "Understanding" might become ["Under", "stand", "ing"]. Each token gets an integer ID, and the model works with integer sequences.

A tokenizer like BPE (Byte-Pair Encoding) is built by iteratively merging the most frequent character pairs in a corpus. The resulting vocabulary typically contains 30,000–100,000 tokens, balancing granularity with vocabulary size.

### N-Grams: The Simplest Language Statistics

Given a text corpus, count how often specific token sequences of length $N$ occur. To generate new text, compute:

$$P(x_t \mid x_{t-N+1}, \dots, x_{t-1}) = \frac{\text{count}(x_{t-N+1}, \dots, x_t)}{\text{count}(x_{t-N+1}, \dots, x_{t-1})}$$

and sample the next token proportionally. With small $N$, the model produces locally plausible but globally incoherent text. With large $N$, it mostly copies the training data verbatim. The fundamental problem: the number of possible N-grams grows as $|V|^N$ (where $|V|$ is vocabulary size), and most sequences never appear in training data. The model cannot generalize.

### The Jump to Neural Networks

Instead of storing counts, train a neural network to *estimate* $P(x_t \mid x_{<t})$. The network maps a context of previous tokens to a probability distribution over the vocabulary. Because the function is parameterized and continuous, it can generalize to contexts never seen during training — predicting meaningful next tokens even for novel sentences.

### The Transformer for Language

The same architecture from Post 5, now applied to text. Each token is embedded as a vector. The transformer processes the full sequence with self-attention, and the output at position $t$ predicts the distribution over the next token:

$$P(x_t \mid x_1, \dots, x_{t-1}) = \text{softmax}(W_{\text{vocab}} \cdot \mathbf{h}_t)$$

where $\mathbf{h}_t$ is the transformer's hidden state at position $t$ and $W_{\text{vocab}}$ projects back to vocabulary size. A critical detail: **causal masking** ensures that position $t$ can only attend to positions $\leq t$, preserving the autoregressive structure (the model can't peek at future tokens).

### Embeddings

Before entering the transformer, each token ID is mapped to a dense vector via a learned embedding matrix $E \in \mathbb{R}^{|V| \times d}$. These embeddings encode semantic similarity: tokens with similar meaning end up as nearby vectors. This is closely related to dimensionality reduction techniques like SVD — a mathematical tool that also applies to the robot dog's sensor data (projecting high-dimensional IMU readings into a compact latent space).

### GPT: The Principle

GPT is a decoder-only transformer trained on next-token prediction over massive text corpora. The training objective is maximum likelihood:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

No manual labels are needed — the next token *is* the label. This makes GPT a foundation model for language, trained self-supervised at scale. But it's "just" a text generator: given context, it produces the most probable continuation. It doesn't inherently answer questions or follow instructions. How to get from here to a useful assistant is the subject of Post 7.

### → Next

We have an LLM that generates plausible text — but it's a pattern-completion engine, not an instruction-following assistant. "What is the capital of France?" might be continued with more questions rather than an answer. How do you turn a text generator into a helpful system that follows instructions?

---

# Post 7: Following Instructions — RLHF and Prompting

**Phase II – Building Intelligence**

### Learning Goals

- Explain why a raw GPT model isn't yet a useful assistant.
- Describe instruction tuning and RLHF at a high level, including the role of reward models.
- Understand prompt engineering as a method for steering LLM behavior.
- Connect this to robot control: how a natural language command becomes a structured plan.

### From Text Generator to Assistant

A base GPT model generates plausible continuations. Ask "What is the capital of France?" and it might output "What is the capital of Germany? What is the capital of Spain?" — because in its training data, questions are often followed by more questions, not answers. The model has no concept of "being helpful."

### Instruction Tuning

First step: supervised fine-tuning on (instruction, desired response) pairs. "Name the capital of France." → "The capital of France is Paris." This teaches the model what helpful responses *look like* — a specific format, not just plausible text.

### RLHF — Reinforcement Learning from Human Feedback

Second step: human raters compare multiple model responses (which is more helpful, accurate, less harmful?). Their rankings train a **reward model** $R_\phi$ that predicts human preferences:

$$R_\phi(\text{prompt}, \text{response}) \rightarrow \text{scalar score}$$

The LLM is then fine-tuned to maximize this reward using PPO (Proximal Policy Optimization):

$$\mathcal{L}_{\text{RLHF}} = -\mathbb{E}\left[R_\phi(\text{prompt}, \text{response})\right] + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

The KL-divergence term prevents the model from deviating too far from the pretrained distribution — otherwise it might learn to "game" the reward model by producing adversarial outputs that score high but are nonsensical. PPO is the same algorithm that appears in Post 10 for training locomotion policies — the connection is not accidental.

### Prompt Engineering

Even after RLHF, output quality depends heavily on how the input is framed:

- **System prompt:** Defines the model's role and behavior constraints.
- **Few-shot examples:** Providing input-output examples in the prompt to demonstrate the desired format.
- **Chain-of-thought:** Explicitly asking the model to reason step by step before answering, which dramatically improves performance on complex tasks.

On the robot dog: the system prompt defines the LLM's role — *"You are a robot dog planner. You receive voice commands and decompose them into a sequence of movement commands. Respond in JSON format."*

### LLM-Guided Reward Engineering

Instead of hand-coding a reward function for gait training, describe the desired behavior in natural language — "walk energy-efficiently but maintain stability" — and have the LLM generate the mathematical objective function. The LLM acts as a "high-level compiler" translating human intent into optimization targets. This bridges the cognitive and motor layers of the robot architecture.

### → Next

The robot can now process language and analyze images — but separately. "Go to the red chair" requires understanding "red chair" linguistically *and* identifying it in the camera feed simultaneously. This calls for multimodal models that connect text and images in a shared representation space.

---

# Post 8: Seeing *and* Reading — Multimodal Models

**Phase II – Building Intelligence**

### Learning Goals

- Explain what a multimodal representation space is and why it's useful.
- Describe CLIP: contrastive learning between text and images.
- Describe Vision-Language Models (VLMs) as systems that can "see" and "talk" about images.
- Apply these concepts to the robot dog: scene understanding in natural language.

### The Gap Between Modalities

An LLM understands text, a vision model understands images — but neither can identify "the red chair on the left in this photo." Bridging modalities requires mapping both into a **shared representation space** where text and images can be directly compared.

### CLIP: Connecting Images and Text

CLIP (Contrastive Language-Image Pretraining) is trained on millions of (image, caption) pairs. Let $f_I$ be the image encoder and $f_T$ the text encoder. For a batch of $N$ matched pairs, the training objective maximizes similarity between matched pairs and minimizes it for mismatched ones — this is exactly the contrastive loss from Post 4, applied across modalities:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(f_I(\mathbf{x}_i), f_T(\mathbf{t}_i)) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(f_I(\mathbf{x}_i), f_T(\mathbf{t}_j)) / \tau)}$$

After training, images and text live in the same vector space. This enables **zero-shot classification**: given an image, compare its embedding to candidate text embeddings ("a cat", "a dog") and pick the closest match — no task-specific training needed.

### Vision-Language Models

VLMs go further: they combine a vision encoder (converting images into tokens) with an LLM (processing image tokens alongside text tokens). The result: a model that "sees" an image and "speaks" about it.

On the robot dog: camera image → vision encoder produces image tokens → LLM processes these with context → output: "I see a student carrying a package in a long corridor. There's an open door on the right."

### The Shared Representation Space

Tying the threads together: IMU data can be projected into compact vectors via SVD. Camera images become vectors via vision encoders. Language commands become vectors via tokenization and embedding. All of these can be compared and combined in a shared space — the foundation for multimodal robot intelligence.

### → Next

We now have all perceptual and linguistic building blocks. The dog can see, understand language, and connect both. But who coordinates everything? Who decides when to use the camera, when to make a plan, when to start walking? That's the job of an *agent*.

---

# Post 9: The Dog Gets a Brain — Agents and Tool Use

**Phase II – Building Intelligence**

### Learning Goals

- Explain how a next-token predictor gains access to external tools.
- Describe the agent loop: perception → planning → action → feedback.
- Explain chain-of-thought reasoning as a strategy for task decomposition.
- Sketch the concrete system architecture: LLM as planner, sensors as input, motor skills as tools.

### From Text Generator to Agent

An LLM alone can't open a door or steer a robot. But it can *decide* which action to take next and call the appropriate tool.

The architecture: the LLM receives a system prompt listing available tools (e.g., `look()`, `move_forward(steps)`, `turn(direction)`). A user gives a natural language command. The LLM decomposes it into a sequence of tool calls, executes them one by one, receives feedback after each step, and adjusts its plan as needed.

### Chain-of-Thought Planning

For complex tasks ("find Prof. X and deliver document Y"), a single tool call isn't enough. Chain-of-thought prompting encourages the model to make planning steps explicit before acting: Where am I? → Plan route → Navigate → Verify arrival → Report. This dramatically improves plan quality compared to direct action generation.

### Reasoning

Modern LLMs can internally weigh options: "Should I go left or right around the chair? Left is tighter but shorter. Right is safer. → I'll go right." This internal deliberation is a key capability for embodied systems operating in unpredictable environments.

### The Full System Architecture

Bringing everything together:

    Voice command → LLM planner (cognitive layer)
        → Sequence of tool calls → Motor layer (ML skills: walk, turn, stop)
            → Physical execution (ROS2, motor currents)
                → Sensor feedback → back to LLM planner

Every post in this series has contributed a component. The LLM planner orchestrates them: data representations (Post 1), neural network training (Posts 2–3), foundation model adaptation (Post 4), vision (Post 5), language understanding (Posts 6–7), and multimodal perception (Post 8).

**ROS2 integration:** The LLM (running on a server) communicates with the robot (running its own hardware) via ROS2 — a messaging framework where components operate as nodes. The LLM is one node, motor control another, the camera a third.

### → Next

The system works on screen. But we've been hand-waving about one crucial component: how does the dog actually *walk*? We haven't trained a gait controller. That's fundamentally different from supervised learning — there are no "correct" walking data. The dog must discover how to walk through trial and error: reinforcement learning.

---

# Post 10: Learning to Walk — Deep Reinforcement Learning

**Phase III – Integration & Reality**

### Learning Goals

- Define the key RL concepts: states, actions, rewards, policy, value function.
- Formalize the problem as a Markov Decision Process (MDP).
- Describe PPO as a stable policy gradient method.
- Explain how the LLM acts as a reward designer.
- Understand the result: a universal motion library callable by the agent.

### A Different Kind of Learning

Supervised learning requires input-output pairs. For locomotion, there are no "correct" motor commands. Instead, the dog discovers how to walk through **trial and error**: trying joint movements, receiving reward signals, and gradually improving.

### The Markov Decision Process

RL is formalized as an MDP defined by the tuple $(S, A, P, R, \gamma)$:

- **States** $S$: joint angles, angular velocities, body orientation, contact forces.
- **Actions** $A$: motor torques for each joint.
- **Transition function** $P(s' \mid s, a)$: the physics of the environment.
- **Reward function** $R(s, a, s')$: a scalar signal (e.g., +1 for forward progress, −1 for falling).
- **Discount factor** $\gamma \in [0, 1)$: controls how much future rewards matter.

The **policy** $\pi_\theta(a \mid s)$ maps states to a probability distribution over actions. The goal is to find $\pi_\theta$ that maximizes expected cumulative discounted reward:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1})\right]$$

### The Bellman Equation and Value Functions

The **value function** $V^\pi(s)$ measures the expected return from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s\right]$$

It satisfies the **Bellman equation**:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[R(s, a) + \gamma \mathbb{E}_{s' \sim P}[V^\pi(s')]\right]$$

This recursive structure is the mathematical foundation of all RL algorithms: the value of a state equals the immediate reward plus the discounted value of the expected next state.

### PPO: Proximal Policy Optimization

Policy gradient methods update $\theta$ to increase the probability of actions that led to high rewards. PPO constrains how much the policy changes per update step, preventing instability:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between new and old policy, $\hat{A}_t$ is the advantage estimate (how much better an action was than expected), and $\epsilon$ is a small clipping threshold. The clipping prevents destructive large updates — essential for stable locomotion training.

In practice, hundreds of robot instances train in parallel in simulation, each exploring different strategies and sharing gradients. This massively parallelized training is what makes RL feasible for complex continuous control.

### The LLM as Reward Designer

Designing reward functions by hand is notoriously difficult. Specify "move forward" and the robot might learn to slide on its belly. Add "stay upright" and it might stand still. LLM-guided reward engineering offers a compelling alternative: describe desired behavior in natural language and have the LLM generate the mathematical objective. The LLM becomes the bridge between human intent and optimization targets.

### The Motion Library

The result: a set of robust skills — "walk forward", "walk sideways", "trot", "turn in place" — that the LLM planner from Post 9 calls as tools. Each skill is a trained policy handling millisecond-level motor coordination autonomously.

### → Next

We can train gait controllers in simulation. But simulation is clean, deterministic, and forgiving. Reality is none of those things. How do we bridge the gap?

---

# Post 11: From Simulation to Reality — and the Live Test

**Phase III – Integration & Reality**

### Learning Goals

- Explain the sim-to-real gap and why simulation-trained models fail on real hardware.
- Describe domain randomization as a robustness strategy.
- Understand state estimation under uncertainty (Kalman filter basics).
- Trace the complete pipeline from voice command to robot movement, mapping each component to a post.
- Critically evaluate strengths, limitations, and open questions.

### The Sim-to-Real Gap

In simulation: uniform friction, perfect sensors, clean physics. In reality: varying friction, sensor noise, changing lighting, unforeseen obstacles. A model trained only in simulation has overfit to the simulator's assumptions and often fails catastrophically in the real world.

### Domain Randomization

Make the simulation intentionally messy. Randomize friction coefficients, add sensor noise, vary lighting, place random obstacles. If the model performs under all these variations, reality becomes "just another variation" within the training distribution. This is data augmentation applied to physics rather than pixels.

### State Estimation: The Kalman Filter

The robot never knows its exact state — sensors are noisy. The **Kalman filter** fuses predictions from a motion model with noisy measurements to produce an optimal state estimate. In two steps:

**Predict** (using the motion model):

$$\hat{\mathbf{x}}_{t|t-1} = F \hat{\mathbf{x}}_{t-1} + B \mathbf{u}_t, \quad P_{t|t-1} = F P_{t-1} F^\top + Q$$

**Update** (incorporating the measurement $\mathbf{z}_t$):

$$K_t = P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1}$$

$$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_{t|t-1} + K_t(\mathbf{z}_t - H \hat{\mathbf{x}}_{t|t-1})$$

$$P_t = (I - K_t H) P_{t|t-1}$$

where $F$ is the state transition matrix, $H$ the observation matrix, $Q$ the process noise covariance, $R$ the measurement noise covariance, and $K_t$ the **Kalman gain** — which optimally balances trust in the prediction vs. the measurement. This is classical Bayesian inference with Gaussians, connecting probability theory from the engineering mathematics curriculum directly to the ML stack.

The LLM can serve as a *system diagnostician*: monitoring filter residuals $(\mathbf{z}_t - H\hat{\mathbf{x}}_{t|t-1})$ and flagging anomalies — "Warning: large position estimate deviation. Possible cause: slippery surface."

### The Live Test

Selected teams demonstrate their full solution on the real Unitree Go2. The task: the system receives a natural language command (e.g., "go to the door at the end of the corridor") and executes it autonomously — from LLM planning through navigation to physical movement.

### Tracing the Full Stack

Every post has contributed a component:

| Component | Post |
|-----------|------|
| Data as feature vectors | 1 |
| MLP architecture | 1 |
| Loss functions and gradient descent | 2 |
| Training, regularization, diagnostics | 3 |
| Self-supervised pretraining | 4 |
| Visual perception (CNN, ViT, DINO) | 5 |
| Language understanding (Transformer, GPT) | 6 |
| Instruction following (RLHF, prompting) | 7 |
| Multimodal scene understanding (CLIP, VLM) | 8 |
| Agent architecture and planning | 9 |
| Locomotion via RL (PPO) | 10 |
| Sim-to-real robustness, state estimation | 11 |

### What Works, What Doesn't

An honest assessment: LLMs hallucinate (generating plausible but false information). Vision models can be fooled by adversarial inputs. RL policies fail in out-of-distribution situations. The system is impressive but far from infallible — understanding limitations is as important as understanding capabilities.

### Ethics and Open Questions

Safety of autonomous systems, bias in training data, privacy in data collection, environmental cost of large-scale training, and the fundamental question: how close are we to AGI — and is that even a meaningful goal?

Embodied AI — systems that perceive, reason, and act in the physical world — is one of the most active research frontiers. The methods in this series are the foundations. The applications are just beginning.

---

# Appendix: Exercise Structure (Recommendation)

Exercises are organized as extended blocks rather than weekly sessions:

**Block A** (after Post 1): Python basics, data manipulation, MLP from scratch.

**Block B** (after Post 3): Training a small network, hyperparameter tuning, overfitting experiments.

**Block C** (after Post 6): N-gram language model, tokenization, CAPTCHA demo, foundation model usage.

**Block D** (after Post 8): CLIP experiments, VLM usage, prompt engineering for the robot dog.

**Block E** (after Post 11): Build an agent, sim-to-real experiment, preparation for the live test.

---

*This document serves as the baseline for all derived materials — lecture slides, exercise sheets, Jupyter notebooks, and assessment design. Each post can be expanded independently without losing the overall narrative.*
