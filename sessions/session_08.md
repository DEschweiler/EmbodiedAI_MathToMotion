---
id: 008
title: Seeing and Listening — Multimodal Models
summary: Bridge vision and language with shared embeddings and VLMs.
tags: [multimodal, clip, vlm]
learning_goals:
  - Explain the modality gap and why shared representation spaces are needed.
  - Describe CLIP's contrastive training objective and its zero-shot capabilities.
  - Understand how VLMs combine vision encoders with language models.
  - Explain visual grounding and its role in robot perception.
---

The robot dog's camera produces images; the user speaks in language. These two modalities are processed by entirely different architectures — vision encoders and language transformers — with no natural way to connect them. Telling the robot to "pick up the blue object near the window" requires jointly understanding what "blue object near the window" means *and* finding it in the camera feed. This session covers the architectures that unify vision and language.

## The Modality Gap

After training separately, a vision encoder and a language model live in different representation spaces. The image embedding of a cat and the text embedding of "a cat" are arbitrary vectors in different high-dimensional spaces — there is no reason they should be similar, and in practice they are not.

To bridge modalities, we need either:
1. A **shared embedding space** where matching image-text pairs map to nearby points (CLIP approach), or
2. A **unified sequence model** that processes image tokens and text tokens interchangeably (native multimodal approach).

Both approaches exist; the most practical systems combine them.

## CLIP: Connecting Images and Text

**CLIP** (Contrastive Language-Image Pretraining; Radford et al., 2021) is trained on 400 million (image, caption) pairs scraped from the web. It consists of two encoders:

- **Image encoder** $f_I$: a Vision Transformer (ViT-B/32 or larger) that maps images to unit-norm vectors in $\mathbb{R}^d$.
- **Text encoder** $f_T$: a transformer that maps captions to unit-norm vectors in the same $\mathbb{R}^d$.

### Contrastive Training Objective

For a batch of $N$ matched (image, text) pairs, CLIP maximizes similarity between matched pairs and minimizes it for all cross-pairs — exactly the contrastive loss from Session 4, applied across modalities:

$$\mathcal{L}_\text{CLIP} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(f_I(\mathbf{x}_i), f_T(\mathbf{t}_i)) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(f_I(\mathbf{x}_i), f_T(\mathbf{t}_j)) / \tau)} + \log \frac{\exp(\text{sim}(f_T(\mathbf{t}_i), f_I(\mathbf{x}_i)) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(f_T(\mathbf{t}_i), f_I(\mathbf{x}_j)) / \tau)} \right]$$

The loss is symmetric: each image should match its paired text, and each text should match its paired image. After training, the image and text encoders produce embeddings in a shared space where semantic similarity corresponds to geometric proximity.

### Zero-Shot Classification

The most striking CLIP capability: classify images without any task-specific training. Given $K$ candidate class labels ("a dog", "a cat", "a bird"), encode each as a text embedding. Then encode the image and find the nearest class embedding by cosine similarity:

$$\hat{y} = \arg\max_k \, \text{sim}\!\left(f_I(\mathbf{x}),\; f_T\!\left(\text{"a photo of a "} + \text{class}_k\right)\right)$$

On ImageNet 1000-class classification, zero-shot CLIP matches a ResNet-50 trained fully supervised on ImageNet — without ever seeing a single labeled ImageNet image. The world's text descriptions of images transfer remarkably well.

### CLIP for the Robot Dog

CLIP enables the robot to ground language in perception:
- "Is there a person in the camera feed?" → compute similarity between the image and the text "a person" vs. "no person".
- "Navigate to the door" → segment the image into regions, score each against "an open door", navigate to the highest-scoring region.
- **Open-vocabulary detection:** pair CLIP with a detection head (e.g., GLIP, OWL-ViT) to detect arbitrary objects described in natural language — no pre-defined class list.

## Vision-Language Models (VLMs)

CLIP aligns embeddings but doesn't *generate* language. **Vision-Language Models** combine a vision encoder with a language model to support rich image-grounded reasoning and generation:

> Input: [camera image] + "Describe what you see and suggest a navigation path to the exit."
> Output: "I see a wide corridor with fluorescent lighting. There are two people walking toward me on the left side. The exit sign is visible at the far end, approximately 15 meters ahead. Suggested path: stay right, reduce speed to 0.3 m/s while passing the people, then proceed directly to the exit."

### Architecture Patterns

**Projection-based (LLaVA, MiniGPT-4):** The simplest approach. The image encoder (e.g., CLIP ViT) produces $N$ patch embeddings. A lightweight projection layer (MLP or single linear layer) maps these to the LLM's token embedding space. The LLM then processes image tokens and text tokens in a single sequence.

```
[image patch tokens (projected)] + [text tokens] → LLM → [output tokens]
```

Only the projection layer and (optionally) the LLM are trained; the vision encoder is frozen. This is parameter-efficient and works surprisingly well.

**Cross-attention (Flamingo; Alayrac et al., 2022):** New cross-attention layers are inserted into the frozen LLM. These layers attend to vision encoder outputs at every transformer block. The LLM "reads" the image at every layer, not just the input. More expressive but more complex.

**Native multimodal (GPT-4o, Gemini):** The model is trained from scratch (or from a language model) on interleaved image-text data, with image patches tokenized directly alongside text tokens. No separate fusion step — vision and language are unified from the ground up.

### Visual Grounding

**Grounding** refers to connecting language expressions to specific regions of an image. "Pick up the object on the left" requires locating "the object on the left" as a bounding box or pixel mask in the current camera frame.

Grounded VLMs (e.g., Grounding DINO, Grounded SAM) output both natural language descriptions and spatial references (bounding boxes, segmentation masks). This is essential for manipulation tasks: the planner says "pick up the blue cube" → the grounded VLM produces the cube's 3D position → the motion controller executes the grasp.

## The Unified Representation Space

Tying the threads together: every sensor modality can be projected into a common vector space.

| Modality | Representation | Dimension |
|---|---|---|
| Camera image | ViT patch embeddings → CLIP projection | $d$ |
| Text command | Token embeddings → LLM hidden state | $d$ |
| IMU data | Linear projection of 12 sensor values | $d$ |
| Depth map | Specialized encoder | $d$ |

Once in a common space, these representations can be concatenated, compared, and processed by a unified transformer. This is the architecture of current state-of-the-art robot foundation models (RT-2, OpenVLA) — they take camera images, proprioceptive sensor readings, and natural language instructions as a single sequence and output robot actions.

---

## Further Reading

**Start here** *(accessible introductions)*
- Yannic Kilcher: "CLIP — Connecting Text and Images" — [youtube.com/watch?v=T9XSU0pKX2E](https://www.youtube.com/watch?v=T9XSU0pKX2E) — clear walkthrough of the CLIP paper and its implications
- Lilian Weng: "Generalized Visual Language Models" — [lilianweng.github.io/posts/2022-06-09-vlm](https://lilianweng.github.io/posts/2022-06-09-vlm) — comprehensive blog post covering CLIP, Flamingo, and related models

**Go deeper** *(technical references)*
- Radford et al.: "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, ICML 2021) — [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) — the original CLIP paper; well-written and accessible
- Liu et al.: "Visual Instruction Tuning" (LLaVA, NeurIPS 2023) — [arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485) — minimal projection-based VLM achieving strong results
- Alayrac et al.: "Flamingo: a Visual Language Model for Few-Shot Learning" (NeurIPS 2022) — [arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198) — cross-attention based VLM from DeepMind
- Brohan et al.: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023) — [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) — multimodal foundation models applied directly to robot action generation

---

## ✏️ Check Your Understanding *(Preliminary)*

> *These questions are placeholders to be refined. Use them to self-assess before moving on.*

1. Before CLIP, a vision classifier trained on ImageNet could only recognize the 1,000 classes in its training set. Explain why CLIP's zero-shot classification is fundamentally different, and what allows it to generalize to arbitrary class labels.

2. CLIP is trained with a contrastive loss on (image, caption) pairs. In a batch of $N = 256$ pairs, how many negative pairs does each image's loss term have to push apart? What does making the batch size larger do to the training signal?

3. The projection-based VLM approach (LLaVA style) freezes the vision encoder and only trains a small projection layer. What are the trade-offs compared to training the full vision encoder end-to-end?

4. Visual grounding connects text expressions to image regions. Give one concrete example of how grounding is necessary for a manipulation task on the Unitree Go2. What goes wrong if the robot has language understanding but no grounding?

5. *(Preliminary)* The session describes "robot foundation models" (RT-2, OpenVLA) that take images, sensor data, and language as a unified sequence and output actions. What architectural components from Sessions 5–8 would such a model need to incorporate?

---

### → Next

We now have all perceptual and linguistic building blocks. The robot can see, understand language, and connect both modalities. But who coordinates everything? Who decides when to look at the camera, when to make a plan, when to start walking, and when to ask for clarification? That's the job of an *agent*. Session 9 covers agent architectures and tool use.
