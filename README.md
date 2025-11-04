# Regularization

![Macbeth by John Martin, 1820](cover.jpg)

Regularization is a very important technique in machine learning. It helps to both prevent overfitting and speed up convergence time. In this project, we will explore all of the most important types of regularization techniques and then implement them.

- [Regularization](#regularization)
  - [Motivation](#motivation)
  - [Notation](#notation)
  - [Types](#types)
    - [Norm-based](#norm-based)
    - [Noise](#noise)
    - [Data](#data)
    - [Early Stopping](#early-stopping)
    - [Architectural](#architectural)
    - [Output](#output)
    - [Gradient](#gradient)
    - [Representation](#representation)
    - [Bayesian](#bayesian)
    - [Domain](#domain)
  - [Applications](#applications)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [License](#license)

## Motivation

When training a complex (or even simple) model, there is a risk of overfitting the training data. Since the model is trying to aggressively reduce the loss, it will try to get as close to every point as possible. But this might create a curve that's "too specific" to the particular data. This increases training time since, with every iteration, we get closer to the exact curve that absolutely minimizes the loss, and also creates overfitting.

By adding regularization, we try to *constrain* the training process. The function becomes less flexible, but this trade-off often makes it more accurate in the real world.

## Notation

Common variables used in the formulas below:

- $\lambda$: Regularization strength (hyperparameter controlling penalty weight)
- $w$: Model weights/parameters (vector or matrix)
- $i,j$: Indices (e.g., for summation over dimensions or samples)
- $x$: Input data/features
- $y$: True labels or targets
- $p$: Predicted probabilities
- $\mathcal{L}$: Loss function

## Types

### Norm-based

Norm-based regularization is the simplest type of regularization. It penalizes large weight magnitudes to control model capacity and reduce overfitting. Usually seen in generic regression or neural network tasks.

Norm-based regularization directly modifies the loss calculation to encourage smaller-magnitude parameters and simpler functions. The intuition is straightforward: smaller weights mean less complex decision boundaries, leading to better generalization. You can also combine these techniques to balance sparsity and smoothness.

A norm-regularized function will probably be more robust and also faster to train.

- L1
  
  $$\mathcal{L}_{\text{reg}} = \lambda\,\lVert w \rVert_1 = \lambda\sum_{i}|w_i|$$

  *where $w_i$ are individual weight components.*

- L2
  
  $$\mathcal{L}_{\text{reg}} = \tfrac{\lambda}{2}\,\lVert w \rVert_2^2 = \tfrac{\lambda}{2}\sum_i w_i^2$$

  *where the 1/2 factor simplifies gradient computation.*

- Elastic Net
  
  $$\mathcal{L}_{\text{reg}} = \lambda_1\,\lVert w \rVert_1 + \tfrac{\lambda_2}{2}\,\lVert w \rVert_2^2$$

  *where $\lambda_1$ controls L1 penalty, $\lambda_2$ controls L2 penalty.*

- Max-norm
  
  $$\text{constraint: } \lVert w \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{reg}} = \lambda\,\max\{0,\, \lVert w \rVert_2 - c\}$$

  *where $c$ is the maximum allowed norm.*

- Group Lasso
  
  $$\mathcal{L}_{\text{reg}} = \lambda\sum_{g} \lVert w_{g} \rVert_2$$

  *where $g$ indexes groups of weights $w_g$.*

- Nuclear norm
  
  $$\mathcal{L}_{\text{reg}} = \lambda\,\lVert W \rVert_* = \lambda\sum_{i} \sigma_i(W)$$

  *where $W$ is a weight matrix, $\sigma_i(W)$ are its singular values.*

### Noise

Noise-based regularization intentionally adds random noise to the data or model during training. The key idea is to enforce stability: the model should produce similar outputs even when the input is slightly perturbed. This forces the model to learn robust features that generalize well to unseen data.

- Dropout
- DropConnect
- Stochastic depth
- Noise injection

### Data

Data-based regularization changes the training data itself. We can add random 'noise features' that don't exist in real life and don't predict anything, we can create synthetic features that are a combination of real features, create adverserial permutations, or scale the data.

Oftentimes, this type of regularization happens implicitly during data preprocessing. However, it's not the same as data cleaning, feature engineering, or augmentation.

Although many different types of models can benefit from this type of regularization, it's not very commonly used, especially not in simple applications.

- Data augmentation
- Mixup
- Cutout
- Adversarial training

### Early Stopping

Early stopping treats overfitting as a temporal problem. Instead of changing the model or the data, it changes *when* we stop training.

Early stopping modifies the learning algorithm itself (whether it be gradient descent or something else) to stop when certain conditions are reached (i.e. the model is 'accurate enough', or validation performance stops improving).

Early stopping is one of the most widely used and effective regularizers in neural network training. It's simple to implement and requires minimal tuning compared to other methods.

- Early stopping
- Learning rate schedules
- Adaptive optimizers

### Architectural

Architectural regularization changes the very structure of the model. Instead of tweaking the data or the loss, we redesign the model from the ground up. For example, we may force certain neurons to have the exact same weights as other neurons. We may force the data to take certain 'paths' when being processed. Many other techniques can be deployed.

This type of regularization requires a very high degree of domain knowledge and should be deployed in accuracy-critical applications. There are no specific formulas, rather we aim to create architectural simplicity in the model itself.

- Weight tying
- Sparse connectivity
- Low-rank factorization
- Pruning

### Output

Output regularization modifies the loss function to penalize overconfident predictions. It directly constrains the optimization to encourage less extreme predictions. The goal is to prevent the model from being too confident in its answers. Most often, it's used in classification tasks where soft labels are more realistic.

Output regularization focuses on smoothing the target outputs and uncertainty estimates. For example, when a model is very confident but ends up being very wrong, it is penalized extra heavily. This forces it to be more cautious and adapt better to rare or difficult cases.

- Label smoothing
  
  $$\mathcal{L}_{\text{LS}} = -\sum_{k=1}^{K} \tilde{y}_k\,\log p_k,\quad \tilde{y}_k =
  \begin{cases}
  1-\varepsilon, & k=y \\
  \dfrac{\varepsilon}{K-1}, & k\ne y
  \end{cases}$$

  *where $\varepsilon$ is smoothing parameter, $K$ is number of classes, $\tilde{y}_k$ are smoothed labels, $p_k$ are predicted probabilities.*

- Confidence penalty
  
  $$\mathcal{L}_{\text{CP}} = \mathcal{L}_{\text{CE}} - \beta\,\mathcal{H}(p)\;=\; -\sum_k y_k\log p_k\; +\; \beta\sum_k p_k\log p_k$$

  *where $\beta$ controls entropy penalty, $\mathcal{H}(p)$ is prediction entropy.*

- Focal loss
  
  $$\mathcal{L}_{\text{FL}} = -\alpha_t\,(1-p_t)^{\gamma}\,\log p_t$$

  *where $\alpha_t$ is class weight, $\gamma$ focuses on hard examples, $p_t$ is probability for true class.*

### Gradient

Gradient regularization looks at how the model learns. It adds constraints on the gradients during training to make the learning process smoother and more stable. This prevents the model from changing too drastically and helps it converge to a better, more general solution. It's like putting training wheels on a bike to keep it steady while learning to ride.

- Jacobian
  
  $$\mathcal{L}_{\text{Jac}} = \lambda\,\mathbb{E}_{x}\big[\,\lVert J_{f}(x) \rVert_F^2\,\big] = \lambda\,\mathbb{E}_{x}\bigg[\sum_{i,j}\Big(\frac{\partial f_i(x)}{\partial x_j}\Big)^2\bigg]$$

  *where $J_f(x)$ is the Jacobian of function $f$ w.r.t. $x$.*

- Orthogonality
  
  $$\mathcal{L}_{\text{Orth}} = \lambda\,\lVert W^\top W - I \rVert_F^2$$

  *where $W$ is weight matrix, $I$ is identity matrix.*

- Spectral norm
  
  $$\text{constraint: } \lVert W \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{SN}} = \lambda\,\lVert W \rVert_2$$

  *where $c$ is maximum spectral norm.*

- Gradient penalties
  
  $$\mathcal{L}_{\text{GP}} = \lambda\,\mathbb{E}_{\hat{x}}\big[\big(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1\big)^2\big]$$

  *where $\hat{x}$ are generated samples, $D$ is discriminator, $\nabla$ denotes gradient.*

### Representation

Representation regularization works on the features the model learns inside. Its goal is to make those features more meaningful. Embedding or feature learning tasks take advantage of representation regularization.

Representation regularization focuses on organizing the learned representations better. It directly encourages certain structures, like pulling similar items closer. The point is to create features that capture the essence of the data without overfitting to specifics. This leads to better transfer learning and generalization.

A representation-regularized model is often not only more interpretable but also more versatile across tasks.

- Contrastive
  
  $$\mathcal{L}_{\text{Contr}} = y\,\lVert z_i - z_j \rVert_2^2 + (1-y)\,[\max\{0,\, m - \lVert z_i - z_j \rVert_2\}]^2$$

  *where $y$ is similarity label, $z_i, z_j$ are embeddings, $m$ is margin.*

- Center loss
  
  $$\mathcal{L}_{\text{Center}} = \tfrac{1}{2}\sum_i \lVert x_i - c_{y_i} \rVert_2^2$$

  *where $x_i$ are features, $c_{y_i}$ is center for class $y_i$.*

- Manifold
  
  $$\mathcal{L}_{\text{Manifold}} = \tfrac{1}{2}\sum_{i,j} S_{ij}\,\lVert z_i - z_j \rVert_2^2 \;=\; \operatorname{Tr}(Z^\top L Z)$$

  *where $S_{ij}$ is similarity matrix, $Z$ is embedding matrix, $L$ is graph Laplacian.*

- Subspace regularization
  
  $$\mathcal{L}_{\text{Subspace}} = \lambda\,\lVert Z - UU^\top Z \rVert_F^2,\quad U^\top U=I$$

  *where $U$ is orthonormal subspace basis.*

### Bayesian

Bayesian regularization uses a Bayesian probabilistic philosophy to train the model. Basically, parameters are seen as uncertain, with the goal of reflecting prior knowledge acquired from learning. The optimization is directly modified to account for priors.

As a result, it creates models that are not just point estimates but full distributions. For example, if we want to predict an artist's album sales, we may use prior knowledge about the artist's previous sales.

- Bayesian
  
  $$\mathcal{L}_{\text{MAP}} = -\log p(\mathcal{D}\mid w) - \log p(w)\;\approx\; \mathcal{L}_{\text{CE}} + \tfrac{\lambda}{2}\,\lVert w \rVert_2^2$$

  *where the MAP approximation equals L2 regularization specifically under a Gaussian prior on weights. Note: different priors yield different regularization forms.*

- Variational
  
  $$\mathcal{L}_{\text{VI}} = \mathbb{E}_{q(w)}\big[ -\log p(\mathcal{D}\mid w) \big] + \operatorname{KL}\big(q(w)\,\|\, p(w)\big)$$

  *where $q(w)$ is variational posterior, $\operatorname{KL}$ is Kullback-Leibler divergence.*

- Spike-and-slab
  
  $$p(w_i) = \pi\,\delta(w_i) + (1-\pi)\,\mathcal{N}(w_i; 0, \sigma^2),\quad \mathcal{L}= -\log p(\mathcal{D}\mid w) - \sum_i \log p(w_i)$$

  *where $\pi$ is spike probability, $\delta$ is Dirac delta, $\mathcal{N}$ is normal distribution, $\sigma$ is standard deviation.*

### Domain

Domain regularization exploits the structure of your specific problem. It considers the relationships in your data—whether graphs, time series, or spatial structure—and encodes those relationships into the loss. The goal is to respect natural patterns rather than learning generic constraints.

Understanding domain regularization is key to applying regularization effectively in your own work. Perhaps you want to reuse one of the more commonly known ones, or perhaps you want to design one tailored to your domain.

Some examples include:

- Graph Laplacian
  
  $$\mathcal{L}_{\text{Lap}} = \tfrac{1}{2}\sum_{i,j} A_{ij}\,\lVert f_i - f_j \rVert_2^2 \;=\; \operatorname{Tr}(F^\top L F)$$

  *where $A_{ij}$ is adjacency matrix, $f_i$ are node features, $F$ is feature matrix, $L$ is Laplacian matrix.*

- Temporal
  
  $$\mathcal{L}_{\text{Temp}} = \sum_{t} \lVert f_{t+1} - f_t \rVert_2^2$$

  *where $t$ indexes time steps, $f_t$ are features at time $t$.*

- Spatial smoothness
  
  $$\mathcal{L}_{\text{Spatial}} = \sum_{i}\sum_{n\in \mathcal{N}(i)} \lVert f_i - f_n \rVert_2^2 \;\;\text{or}\;\; \lambda\int \lVert \nabla f(x) \rVert_2^2\,dx$$

  *where $\mathcal{N}(i)$ are neighbors of node $i$, $\nabla$ denotes spatial gradient.*

## Applications

In all sorts of machine learning, overfitting is a risk. As such, anywhere where it is a risk, regularization can and should be applied. A few examples include:

- Linear models
- Neural networks
- Reinforcement learning
- Ensemble methods
- Probabilistic models

## Project Structure

- Will imeplement a PyTorch neural network that trains on some x, y, or z data. During training, we will pass in different types of regularization functions depending on the context.

```md
|
|
|
|
|
|
```

## Installation

### Prerequisites

- a
- b
- c

### Setup

Clone this repo:

```bash
git clone https://github.com/intelligent-username/Regularization
cd Regularization
```

[More stuff once I actually implement :)]

### License

MIT
