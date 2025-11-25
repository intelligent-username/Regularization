# CIFAR-10

CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is commonly used for training various image processing systems. In this part of the project, we'll use it to demonstrate the following regularization methods:

  - **Norm-based:** Max-norm constraints on CNNs.
  - **Noise:** Stochastic depth on ResNets.
  - **Data:** Augmentations (flip, crop, colour jitter), Mixup, Cutout.
  - **Architectural:** Sparse connectivity (pruning filters, channel sparsity).
  - **Output:** Confidence penalty.
  - **Representation:** Contrastive loss (self-supervised), center loss.
  - **Domain / Graph Laplacian:** Could illustrate label propagation in graph-like versions of CIFAR.

The database will be imported using PyTorch.

## Training

The model will be a small CNN that has convolutional layers, pooling, and dense activations. The final result is then passed to ReLU for classification.

## Structure

`demo.ipynb` will contain the code for importing the data, defining the model, training, and evaluating it. Different `.py` files will then implement the regularization techniques we're trying to demonstrate, and `regularized.ipynb` will contain the code to run experiments with different regularization methods. `Comparator.ipynb` will then put all of the results side-by-side for comparison.
