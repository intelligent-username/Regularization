# Regularization

![The Orrery by Joseph Wright of Arbey, 1766](cover.jpg)

Regularization is a very important technique in machine learning. It helps to both prevent overfitting and speed up convergence time. In this project, we will explore all of the most important types of regularization techniques and then implement them.

## Motivation

When training a complex (or even simple) model, there is a risk of overfitting the training data. Since the model is trying to aggressively reduce the loss, it will try to get as close to every point as possible. But this might create a curve that's "too specific" to the particular data. This increases training time since, with every iteration, we get closer to the exact curve that absolutely minimizes the loss, and also creates overfitting.

By adding regularization, we try to *simplify* the training process. The function becomes less flexible, but, in the real world, this is often more accurate.

## Types

### Norm-based

Norm-based regularization is the simplest type of regularization. It's goal is to just make us see *less stuff*. Most often, it's used in generic regression or neural network tasks.

Norm-based regularization focuses on reducing the particularity of the parameters. It directly modifies either the loss calculation or the data processing. The point is to make smaller-magnitude parameters and simpler-degree functions. We get a sort of mini-[PCA](https://github.com/intelligent-username/PCA) for free by using these techniques. This makes it even easier to simplify our function even further later. Other ways of simplifying our function would be to include, for example, some sort of regularization term that penalizes numbers that are further away from being integers.

A norm-regularized function is often not only more robust but also faster.

- L1
  
  $$\mathcal{L}_{\text{reg}} = \lambda\,\lVert w \rVert_1 = \lambda\sum_{i}|w_i|$$

- L2
  
  $$\mathcal{L}_{\text{reg}} = \tfrac{\lambda}{2}\,\lVert w \rVert_2^2 = \tfrac{\lambda}{2}\sum_i w_i^2$$

- Elastic Net
  
  $$\mathcal{L}_{\text{reg}} = \lambda_1\,\lVert w \rVert_1 + \tfrac{\lambda_2}{2}\,\lVert w \rVert_2^2$$

- Max-norm
  
  $$\text{constraint: } \lVert w \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{reg}} = \lambda\,\max\{0,\, \lVert w \rVert_2 - c\}$$

- Group Lasso
  
  $$\mathcal{L}_{\text{reg}} = \lambda\sum_{g} \lVert w_{g} \rVert_2$$

- Nuclear norm
  
  $$\mathcal{L}_{\text{reg}} = \lambda\,\lVert W \rVert_* = \lambda\sum_{i} \sigma_i(W)$$

### Noise

Noise-based regularization intentionally adds random noise to the data during training. Since there is always a high chance of a strong amount of noise appearing *around* the data, the algorithm is forced to create a more general model that can handle both the noise which aberrs around the data and the data itself. The model is thus able to perform "equally well" on data that is slightly different from the current training data.

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

Early stopping modifies the learning algorithm itself (whether it be gradient descent or something else) to stop when certain conditions are reached (i.e. the model is 'accurate enough', or enough time has passed).

Early stopping is an example of a regularization technique that should probably be avoided, as it requires a lot of 'by-hand' tweaking and may unintentionally lead to the opposite problem: underfitting.

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

Output regularization is a technique that adjusts the model's predictions themselves. It directly modifies the loss function to encourage less extreme predictions. The goal is to prevent the model from being too confident in its answers. Most often, it's used in classification tasks where labels can be softened.

Output regularization focuses on smoothing the target outputs or penalizing high confidence. For example, when a model is very confident but ends up being very wrong, it is penalized extra heavily. This forces it to specifically adapt to rare or difficult cases.

- Label smoothing
  
  $$\mathcal{L}_{\text{LS}} = -\sum_{k=1}^{K} \tilde{y}_k\,\log p_k,\quad \tilde{y}_k =
  \begin{cases}
  1-\varepsilon, & k=y \\
  \dfrac{\varepsilon}{K-1}, & k\ne y
  \end{cases}$$

- Confidence penalty
  
  $$\mathcal{L}_{\text{CP}} = \mathcal{L}_{\text{CE}} - \beta\,\mathcal{H}(p)\;=\; -\sum_k y_k\log p_k\; +\; \beta\sum_k p_k\log p_k$$

- Focal loss
  
  $$\mathcal{L}_{\text{FL}} = -\alpha_t\,(1-p_t)^{\gamma}\,\log p_t$$

### Gradient

Gradient regularization looks at how the model learns. It adds constraints on the gradients during training to make the learning process smoother and more stable. This prevents the model from changing too drastically and helps it converge to a better, more general solution. It's like putting training wheels on a bike to keep it steady while learning to ride.

- Jacobian
  
  $$\mathcal{L}_{\text{Jac}} = \lambda\,\mathbb{E}_{x}\big[\,\lVert J_{f}(x) \rVert_F^2\,\big] = \lambda\,\mathbb{E}_{x}\bigg[\sum_{i,j}\Big(\frac{\partial f_i(x)}{\partial x_j}\Big)^2\bigg]$$

- Orthogonality
  
  $$\mathcal{L}_{\text{Orth}} = \lambda\,\lVert W^\top W - I \rVert_F^2$$

- Spectral norm
  
  $$\text{constraint: } \lVert W \rVert_2 \le c\quad\text{or penalty: }\; \mathcal{L}_{\text{SN}} = \lambda\,\lVert W \rVert_2$$

- Gradient penalties
  
  $$\mathcal{L}_{\text{GP}} = \lambda\,\mathbb{E}_{\hat{x}}\big[\big(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1\big)^2\big]$$

### Representation

Representation regularization works on the features the model learns inside. Its goal is to make those features more meaningful. Most often, it's used in embedding or feature learning tasks.

Representation regularization focuses on organizing the learned representations better. It directly encourages certain structures, like pulling similar items closer. The point is to create features that capture the essence of the data without overfitting to specifics. This leads to better transfer learning and generalization.

A representation-regularized model is often not only more interpretable but also more versatile across tasks.

- Contrastive
  
  $$\mathcal{L}_{\text{Contr}} = y\,\lVert z_i - z_j \rVert_2^2 + (1-y)\,[\max\{0,\, m - \lVert z_i - z_j \rVert_2\}]^2$$

- Center loss
  
  $$\mathcal{L}_{\text{Center}} = \tfrac{1}{2}\sum_i \lVert x_i - c_{y_i} \rVert_2^2$$

- Manifold
  
  $$\mathcal{L}_{\text{Manifold}} = \tfrac{1}{2}\sum_{i,j} S_{ij}\,\lVert z_i - z_j \rVert_2^2 \;=\; \operatorname{Tr}(Z^\top L Z)$$

- Subspace regularization
  
  $$\mathcal{L}_{\text{Subspace}} = \lambda\,\lVert Z - UU^\top Z \rVert_F^2,\quad U^\top U=I$$

### Bayesian

Bayesian regularization takes a Bayesian probabilistic philosophy to train the model. Parameters are treated as uncertain, with the goal of incorporating prior knowledge into the learning. Bayesian regularization focuses on using probability distributions for weights. It directly modifies the optimization to account for priors. The point is to create models that are not just point estimates but full distributions. For example, if we want to predict an artist's album sales, we may use prior knowledge about the artist's previous sales.

- Bayesian
  
  $$\mathcal{L}_{\text{MAP}} = -\log p(\mathcal{D}\mid w) - \log p(w)\;\approx\; \mathcal{L}_{\text{CE}} + \tfrac{\lambda}{2}\,\lVert w \rVert_2^2$$

- Variational
  
  $$\mathcal{L}_{\text{VI}} = \mathbb{E}_{q(w)}\big[ -\log p(\mathcal{D}\mid w) \big] + \operatorname{KL}\big(q(w)\,\|\, p(w)\big)$$

- Spike-and-slab
  
  $$p(w_i) = \pi\,\delta(w_i) + (1-\pi)\,\mathcal{N}(w_i; 0, \sigma^2),\quad \mathcal{L}= -\log p(\mathcal{D}\mid w) - \sum_i \log p(w_i)$$

### Domain

The most important takeaway from studying these regularization techniques is to be able to apply them to your own domain. Perhaps you want to reuse one of the more commonly known ones, or perhaps you want to make one yourself. Domain regularization considers the bigger picture of where the data comes from. Its goal is to respect the natural relationships in the data. Most often, it's used in tasks with structured data like graphs or time series.

Some examples include:

- Graph Laplacian
  
  $$\mathcal{L}_{\text{Lap}} = \tfrac{1}{2}\sum_{i,j} A_{ij}\,\lVert f_i - f_j \rVert_2^2 \;=\; \operatorname{Tr}(F^\top L F)$$

- Temporal
  
  $$\mathcal{L}_{\text{Temp}} = \sum_{t} \lVert f_{t+1} - f_t \rVert_2^2$$

- Spatial smoothness
  
  $$\mathcal{L}_{\text{Spatial}} = \sum_{i}\sum_{n\in \mathcal{N}(i)} \lVert f_i - f_n \rVert_2^2 \;\;\text{or}\;\; \lambda\int \lVert \nabla f(x) \rVert_2^2\,dx$$

## Applications

- Neural networks
- Linear models
- Ensemble methods
- Reinforcement learning
- Probabilistic models
- Basically everywhere

## Structure

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
cd regularization
```

[More stuff once I actually implement :)]

### License

MIT
