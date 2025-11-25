# Regularization

![Cover: Macbeth by John Martin, 1820](cover.jpg)

Regularization is a very important technique in machine learning. It helps both to prevent overfitting and to speed up convergence. In this project, we will explore all of the most important types of regularization techniques and then implement them.

- [Regularization](#regularization)
  - [Motivation](#motivation)
  - [Notation](#notation)
  - [Types](#types)
    - [Norm-based](#norm-based)
      - [L1](#l1)
      - [L2](#l2)
      - [Elastic Net](#elastic-net)
      - [Max-norm](#max-norm)
      - [Group Lasso](#group-lasso)
      - [Nuclear norm](#nuclear-norm)
    - [Noise](#noise)
      - [Dropout](#dropout)
      - [DropConnect](#dropconnect)
      - [Stochastic depth](#stochastic-depth)
      - [Noise injection](#noise-injection)
    - [Data](#data)
      - [Data augmentation](#data-augmentation)
      - [Mixup](#mixup)
      - [Cutout](#cutout)
      - [Adversarial training](#adversarial-training)
    - [Early Stopping](#early-stopping)
      - [Early stopping](#early-stopping-1)
      - [Learning rate schedules](#learning-rate-schedules)
      - [Adaptive optimizers](#adaptive-optimizers)
    - [Architectural](#architectural)
      - [Weight tying](#weight-tying)
      - [Sparse connectivity](#sparse-connectivity)
      - [Low-rank factorization](#low-rank-factorization)
      - [Pruning](#pruning)
    - [Output](#output)
      - [Label smoothing](#label-smoothing)
      - [Confidence Penalty](#confidence-penalty)
      - [Focal loss](#focal-loss)
    - [Gradient](#gradient)
      - [Jacobian](#jacobian)
      - [Orthogonality](#orthogonality)
      - [Spectral norm](#spectral-norm)
      - [Gradient penalties](#gradient-penalties)
    - [Representation](#representation)
      - [Contrastive](#contrastive)
      - [Center loss](#center-loss)
      - [Manifold](#manifold)
      - [Subspace regularization](#subspace-regularization)
    - [Bayesian](#bayesian)
      - [Bayesian](#bayesian-1)
      - [Variational](#variational)
      - [Spike-and-slab](#spike-and-slab)
    - [Domain](#domain)
      - [Graph Laplacian](#graph-laplacian)
      - [Temporal](#temporal)
      - [Spatial smoothness](#spatial-smoothness)
  - [Applications](#applications)
  - [Project Structure](#project-structure)
    - [**1. MNIST**](#1-mnist)
    - [**2. CIFAR-10**](#2-cifar-10)
    - [**3. Fashion-MNIST**](#3-fashion-mnist)
    - [**4. California Housing Dataset**](#4-california-housing-dataset)
    - [**5. UCI Wine datasets (Classification)**](#5-uci-wine-datasets-classification)
    - [Some Notes on the Implementation](#some-notes-on-the-implementation)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)

## Motivation

When training a complex (or even simple) model, there is a risk of overfitting the training data. Since the model is trying to aggressively reduce the loss, it will try to get as close to every point as possible. This might create a curve that's "too specific" to the particular data. This increases training time since, with every iteration, we get closer to the exact curve that minimizes the loss, and also creates overfitting.

By adding regularization, we try to *constrain* the training process. The function becomes less flexible, but this trade-off often makes it more accurate in the real world.

## Notation

Throughout this writeup, I will be writing about regularization techniques. Some of them have formulas, with common variables:

- $\lambda$: Regularization strength $\in \mathbb{R}^+$
- $w$: Model weights/parameters ($\text{vector/matrix}$)
- $i,j$: Common indices (for summation, etc.)
- $x$: Features
- $y$: Labels
- $p$: Predicted probabilities
- $\mathcal{L}$: Loss function

## Types

### Norm-based

Norm-based regularization is the simplest type of regularization. It penalizes large weight magnitudes to control model capacity and reduce overfitting. Usually seen in generic regression or neural network tasks.

Norm-based regularization directly modifies the loss calculation to encourage smaller-magnitude parameters and simpler functions. The intuition is straightforward: smaller weights mean less complex decision boundaries, leading to better generalization. You can also combine these techniques to balance sparsity and smoothness.

A norm-regularized function will probably be more robust and also faster to train.

#### L1
  
$$\mathcal{L}_{\text{reg}} = \lambda\,\lVert w \rVert_1 = \lambda\sum_{i}|w_i|$$

where $w_i$ are individual weight components.

#### L2
  
$$\mathcal{L}_{\text{reg}} = \tfrac{\lambda}{2}\,\lVert w \rVert_2^2 = \tfrac{\lambda}{2}\sum_i w_i^2$$

where the $1/2$ factor simplifies gradient computation.

#### Elastic Net
  
$$\mathcal{L}_{\text{reg}} = \lambda_1\,\lVert w \rVert_1 + \tfrac{\lambda_2}{2}\,\lVert w \rVert_2^2$$

where $\lambda_1$ controls L1 penalty, $\lambda_2$ controls L2 penalty.

#### Max-norm
  
$$\text{constraint: } \lVert w \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{reg}} = \lambda\,\max\{0,\, \lVert w \rVert_2 - c\}$$

where $c$ is the maximum allowed norm.

#### Group Lasso
  
$$\mathcal{L}_{\text{reg}} = \lambda\sum_{g} \lVert w_{g} \rVert_2$$

where $g$ indexes groups of weights $w_g$.

#### Nuclear norm
  
$$\mathcal{L}_{\text{reg}} = \lambda\,\lVert W \rVert_* = \lambda\sum_{i} \sigma_i(W)$$

where $W$ is a weight matrix, $\sigma_i(W)$ are its singular values.

### Noise

Noise-based regularization intentionally adds random noise to the data or model during training. The key idea is to enforce stability: the model should produce similar outputs even when the input is slightly perturbed. This forces the model to learn robust features that generalize well to unseen data.

#### Dropout

Dropout is a technique where, during training, we randomly "drop out" (set to zero) a fraction of the neurons in each layer. This prevents the network from relying too heavily on any single neuron and encourages redundancy. When we have this redundancy, any small number of neurons is more likely to accurately predict the output. Thus, they also 'balance' each other out, which forms a more robust model. At test time, we scale the weights by the dropout probability to maintain the expected output.

#### DropConnect

Similar to dropout, but instead of dropping neurons, we drop connections (weights) randomly. This can lead to sparser representations and better generalization by forcing the model to learn more robust features.

#### Stochastic depth

In deep networks, we randomly skip entire layers during training. This allows training very deep networks by preventing vanishing gradients and overfitting, making it easier to scale up model depth.

#### Noise injection

We add random noise to the inputs, activations, or even gradients during training. This forces the model to be robust to perturbations and can improve generalization by making it less sensitive to small input changes.

### Data

Data-based regularization changes the training data itself. We can add random 'noise features' that don't exist in real life and don't predict anything, create synthetic features that are combinations of real features, create adversarial permutations, or scale the data.

Oftentimes, this type of regularization happens implicitly during data preprocessing. However, it's not the same as data cleaning, feature engineering, or augmentation.

Although many types of models can benefit from this regularization, it's not very commonly used, especially not in simple applications.

#### Data augmentation

We artificially expand the training dataset by applying transformations like rotations, translations, flips, or color changes to the existing data. This helps the model learn invariant features and reduces overfitting without needing more real data.

#### Mixup

Mixup creates new training examples by taking convex combinations of pairs of examples and their labels. For example, mixing an image of a cat and a dog with weights 0.6 and 0.4 creates a blended image with label 0.6 cat + 0.4 dog. This encourages the model to behave linearly between training examples and improves generalization.

#### Cutout

Cutout randomly masks out one or more square regions in the input image during training. This forces the model to learn from the remaining parts and improves robustness to occlusions or missing data.

#### Adversarial training

We generate adversarial examples by adding small perturbations to inputs that fool the model, then train on these examples. This makes the model more robust to adversarial attacks and can improve generalization by learning more stable decision boundaries.

### Early Stopping

Early stopping treats overfitting as a temporal problem. Instead of changing the model or the data, it changes *when* we stop training.

Early stopping modifies the learning algorithm itself (whether it be gradient descent or something else) to stop when certain conditions are reached (i.e., the model is 'accurate enough', or validation performance stops improving).

Early stopping is one of the most widely used and effective regularizers in neural network training. It's simple to implement and requires minimal tuning compared to other methods.

#### Early stopping

Early stopping monitors the average loss during training and stops when we get close to convergence, after some pre-set hyperparameter of epochs is reached. This prevents overfitting by not letting the model train for too long. It just simply stops training early.

#### Learning rate schedules

Instead of using a constant learning rate, we decrease it over time according to a schedule, like halving it every few epochs or using exponential decay. This allows larger steps early for fast convergence and smaller steps later for fine-tuning, helping avoid overshooting the optimum.

#### Adaptive optimizers

Optimizers like Adam or RMSProp adapt the learning rate for each parameter based on the history of gradients. They use moving averages of squared gradients to scale the learning rates, making them more efficient for sparse or noisy gradients and often leading to faster convergence.

### Architectural

Architectural regularization changes the very structure of the model. Instead of tweaking the data or the loss, we redesign the model from the ground up. For example, we may force certain neurons to have the exact same weights as other neurons. We may force the data to take certain 'paths' when being processed. Many other techniques can be deployed.

This type of regularization requires a high degree of domain knowledge and should be deployed in accuracy-critical applications. There are no specific formulas; rather, we aim to create architectural simplicity in the model itself.

#### Weight tying

Weight tying shares parameters between different parts of the model, like tying encoder and decoder weights in autoencoders. This reduces the number of parameters to learn and can improve generalization by enforcing symmetry in the model. The model stops being hyper-specific to the details of the data and is forced to reflect real-world relations.

#### Sparse connectivity

Instead of fully connected layers, we use sparse connections, where each neuron connects to only a subset of neurons in the previous layer. This type of regularization comes "for free" with convolutional layers.

#### Low-rank factorization

We approximate weight matrices as products of lower-rank matrices. For example, a matrix $W \approx U V^\top$ where $U$ and $V$ have fewer columns. This reduces the number of parameters and can capture the essential structure while regularizing against overfitting.

#### Pruning

After training, we remove (set to zero) weights that are below a certain threshold. This creates a sparse model that's faster to run, uses less memory, and is less prone to overfitting due to reduced complexity.

### Output

The goal of output regularization is to smooth the target outputs and uncertainty estimates. For example, when a model is very confident but ends up being very wrong, it is penalized extra heavily. This forces it to be more cautious and adapt better to rare or difficult cases. It also gives it more room to be 'lenient' in ranges where we have a lot of data (prevents biasing towards current noise).

#### Label smoothing
  
$$\mathcal{L}_{\text{LS}} = -\sum_{k=1}^{K} \tilde{y}_k\,\log p_k,\quad \tilde{y}_k =
\begin{cases}
1-\varepsilon, & k=y \\
\dfrac{\varepsilon}{K-1}, & k\ne y
\end{cases}$$

where $\varepsilon$ is smoothing parameter, $K$ is number of classes, $\tilde{y}_k$ are smoothed labels, $p_k$ are predicted probabilities.

#### Confidence Penalty
  
$$\mathcal{L}_{\text{CP}} = \mathcal{L}_{\text{CE}} - \beta\,\mathcal{H}(p) = -\sum_k y_k\log p_k + \beta\sum_k p_k\log p_k$$

where $\beta$ controls entropy penalty, $\mathcal{H}(p)$ is prediction entropy.

#### Focal loss
  
$$\mathcal{L}_{\text{FL}} = -\alpha_t\,(1-p_t)^{\gamma}\,\log p_t$$

where $\alpha_t$ is class weight, $\gamma$ focuses on hard examples, $p_t$ is probability for true class.

### Gradient

Gradient regularization is a set of mathematical tricks that change the loss function itself with the intention of constraining the gradients. They add constraints on the gradients during training to stabilize the learning process. This prevents the model from changing too stochastically and converges to a more general solution.

#### Jacobian
  
$$\mathcal{L}_{\text{Jac}} = \lambda\,\mathbb{E}_{x}\big[\,\lVert J_{f}(x) \rVert_F^2\,\big] = \lambda\,\mathbb{E}_{x}\bigg[\sum_{i,j}\Big(\frac{\partial f_i(x)}{\partial x_j}\Big)^2\bigg]$$

where $J_f(x)$ is the Jacobian of function $f$ w.r.t. $x$.

#### Orthogonality
  
$$\mathcal{L}_{\text{Orth}} = \lambda\,\lVert W^\top W - I \rVert_F^2$$

where $W$ is weight matrix, $I$ is identity matrix.

#### Spectral norm
  
$$\text{constraint: } \lVert W \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{SN}} = \lambda\,\lVert W \rVert_2$$

where $c$ is maximum spectral norm.

#### Gradient penalties
  
$$\mathcal{L}_{\text{GP}} = \lambda\,\mathbb{E}_{\hat{x}}\big[(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1)^2\big]$$

where $\hat{x}$ are generated samples, $D$ is discriminator, $\nabla$ denotes gradient.

### Representation

Representation regularization works on the features the model learns inside. Its goal is to make those features more meaningful. Embedding or feature learning tasks take advantage of representation regularization.

A representation-regularized model is often not only more interpretable but also more versatile across tasks.

#### Contrastive
  
$$\mathcal{L}_{\text{Contr}} = y\,\lVert z_i - z_j \rVert_2^2 + (1-y)\,[\max\{0,\, m - \lVert z_i - z_j \rVert_2\}]^2$$

where $y$ is similarity label, $z_i, z_j$ are embeddings, $m$ is margin.

#### Center loss
  
$$\mathcal{L}_{\text{Center}} = \tfrac{1}{2}\sum_i \lVert x_i - c_{y_i} \rVert_2^2$$

where $x_i$ are features, $c_{y_i}$ is center for class $y_i$.

#### Manifold
  
$$\mathcal{L}_{\text{Manifold}} = \tfrac{1}{2}\sum_{i,j} S_{ij}\,\lVert z_i - z_j \rVert_2^2 = \mathrm{Tr}(Z^\top L Z)$$

where $S_{ij}$ is similarity matrix, $Z$ is embedding matrix, $L$ is graph Laplacian.

#### Subspace regularization
  
$$\mathcal{L}_{\text{Subspace}} = \lambda\,\lVert Z - UU^\top Z \rVert_F^2,\quad U^\top U=I$$

where $U$ is orthonormal subspace basis.

### Bayesian

Bayesian regularization uses a Bayesian probabilistic philosophy to train the model. Parameters are seen as uncertain, with the goal of reflecting prior knowledge acquired from learning. The optimization is directly modified to account for priors.

As a result, it creates models that are not just point estimates but full distributions. For example, if we want to predict an artist's album sales, we may use prior knowledge about the artist's previous sales.

#### Bayesian
  
$$\mathcal{L}_{\text{MAP}} = -\log p(\mathcal{D}\mid w) - \log p(w) \approx \mathcal{L}_{\text{CE}} + \tfrac{\lambda}{2}\,\lVert w \rVert_2^2$$

where the MAP approximation equals L2 regularization specifically under a Gaussian prior on weights. Different priors yield different regularization forms.

#### Variational

$$\mathcal{L}_{\text{VI}} = \mathbb{E}_{q(w)}\big[ -\log p(\mathcal{D}\mid w) \big] + \mathrm{KL}\big(q(w)\,\|\, p(w)\big)$$

where $q(w)$ is variational posterior, $\mathrm{KL}$ is Kullback-Leibler divergence.

#### Spike-and-slab
  
$$p(w_i) = \pi\,\delta(w_i) + (1-\pi)\,\mathcal{N}(w_i; 0, \sigma^2),\quad \mathcal{L}= -\log p(\mathcal{D}\mid w) - \sum_i \log p(w_i)$$

where $\pi$ is spike probability, $\delta$ is Dirac delta, $\mathcal{N}$ is normal distribution, $\sigma$ is standard deviation.*

### Domain

Domain regularization is when *you* create your own type of regularization depending on the domain of current work. It considers the relationships in the data and encodes those relationships into the loss. The goal is to respect natural patterns rather than learning generic constraints.

When working with advanced systems, creating domain-specific regularization techniques will probably be the most effective way to solve completely novel or unique problems. Perhaps you want to reuse one of the more commonly known ones, or design one tailored to your use case.

Some examples include:

#### Graph Laplacian
  
$$\mathcal{L}_{\text{Lap}} = \tfrac{1}{2}\sum_{i,j} A_{ij}\,\lVert f_i - f_j \rVert_2^2 \;=\; \mathrm{Tr}(F^\top L F)$$

where $A_{ij}$ is adjacency matrix, $f_i$ are node features, $F$ is feature matrix, $L$ is Laplacian matrix.*

#### Temporal
  
$$\mathcal{L}_{\text{Temp}} = \sum_{t} \lVert f_{t+1} - f_t \rVert_2^2$$

where $t$ indexes time steps, $f_t$ are features at time $t$.*

#### Spatial smoothness
  
$$\mathcal{L}_{\text{Spatial}} = \sum_{i}\sum_{n\in \mathcal{N}(i)} \lVert f_i - f_n \rVert_2^2 \;\;\text{or}\;\; \lambda\int \lVert \nabla f(x) \rVert_2^2\,dx$$

where $\mathcal{N}(i)$ are neighbors of node $i$, $\nabla$ denotes spatial gradient.*

## Applications

In all sorts of machine learning tasks, overfitting is a real risk. Anywhere it is a risk, regularization can and should be applied. There are many examples; some of the most relevant:

- All supervised learning tasks like neural/linear regression and classification
- Unsupervised learning tasks like clustering and dimensionality reduction
- Reinforcement learning tasks
- Time series forecasting
- Generative models

## Project Structure

This project is going to implement different PyTorch neural networks that train on some data. Each one will train without regularization, and then with different types of regularization. The results will be compared.

This is the plan right now:

---

### **1. MNIST**

- Contains a bunch of handwritten digits (0-9) in grayscale, 28x28 pixels.
- Will first train a simple fully-connected neural network (MLP) on MNIST without any regularization, and then with different types of regularization.

  - **Norm-based:** L1, L2, Elastic Net on fully-connected networks.
  - **Noise:** Dropout, DropConnect.
  - **Data:** Data augmentation (rotations, translations), Mixup, Cutout.
  - **Output:** Label smoothing.
  - **Architectural:** Weight tying in fully connected layers or embeddings.
  - **Gradient penalties:** Jacobian norm to smooth predictions.

---

### **2. CIFAR-10**

- Small, colour images, more complex than MNIST; overfits if the network is slightly big.
- Will train a small CNN on CIFAR-10 without regularization first, then apply different methods.

  - **Norm-based:** Max-norm constraints on CNNs.
  - **Noise:** Stochastic depth on ResNets.
  - **Data:** Augmentations (flip, crop, colour jitter), Mixup, Cutout.
  - **Architectural:** Sparse connectivity (pruning filters, channel sparsity).
  - **Output:** Confidence penalty.
  - **Representation:** Contrastive loss (self-supervised), center loss.
  - **Domain / Graph Laplacian:** Could illustrate label propagation in graph-like versions of CIFAR.

---

### **3. Fashion-MNIST**

- Same simplicity as MNIST but slightly harder; models still overfit easily.
- Will train a simple MLP or small CNN without regularization first, then with different regularization.

  - Same as MNIST, but differences in difficulty allow **highlighting where regularization becomes essential**.
  - **Noise injection:** Gaussian noise on inputs.
  - **Gradient-based:** Spectral norm constraints on convolutional layers.

---

### **4. California Housing Dataset**

- Tiny tabular dataset; can overfit with fully-connected MLP.
- Train a simple regression MLP without regularization, then experiment with penalties and priors.

  - **Norm-based:** L1/L2 regression penalties.
  - **Bayesian:** Variational or spike-and-slab priors.
  - **Early stopping / adaptive optimizers:** improves generalization in regression.
  - **Representation:** Subspace regularization (project weights to low-dimensional space).
  - **Gradient penalties:** Jacobian penalties for smooth predictions.

---

### **5. UCI Wine datasets (Classification)**

- Extremely small, easy to overfit; perfect for showing clear differences with/without regularization.
- Will train a tiny MLP or logistic regression baseline, then show improvements with regularization.

  - **Norm-based:** L1/L2, group Lasso on feature groups.
  - **Bayesian:** Small-scale spike-and-slab illustration.
  - **Early stopping / learning rate schedules**: immediate effect visible.
  - **Architectural:** Weight tying or low-rank factorization in small MLP.
  - **Output:** Label smoothing.

---

### Some Notes on the Implementation

- All datasets are available via `torchvision.datasets` (for images) or can be loaded via `sklearn.datasets` (tabular).
- Use **very small networks** (1–3 layers for tabular, small CNN for images) to exaggerate overfitting.
- Train **without regularization first**. Log validation/test loss and accuracy.
- Overfitting is almost guaranteed in MNIST/Fashion-MNIST/CIFAR-10 with small datasets and big networks; perfect for clear demonstration.

```md
├── MNIST/              # First Demo
├── CF10/               # Second Demo
├── FMNIST/             # Third Demo
├── BH/                 # Fourth Demo
├── UCIW/               # Fifth Demo
├── .gitignore          # Stuff git will leave out
├── cover.jpg           # Cover image
├── LICENSE             # License information
├── README.md           # This file
└── requirements.txt    # Python prerequisites
```

## Installation

### Prerequisites

- Python 3.8+

### Setup

Clone this repo:

```bash
git clone https://github.com/intelligent-username/Regularization
cd Regularization
```

Install the prerequisites:

```bash
# Create a virtual environment first (recommended)
python -m venv venv
# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

Run the demos in ipynb files or scripts. Or just read them.

---

This project is licensed under the [MIT License](LICENSE).
