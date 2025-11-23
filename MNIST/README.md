# MNIST

MNIST stands for Modified National Institute of Standards and Technology database. It is a large database of handwritten digits that is commonly used for training various image processing systems. In this part of the project, we'll use it to demonstrate the following regularization methods:

  - **Norm-based:** L1, L2, Elastic Net on fully-connected networks.
  - **Noise:** Dropout, DropConnect.
  - **Data:** Data augmentation (rotations, translations), Mixup, Cutout.
  - **Output:** Label smoothing.
  - **Architectural:** Weight tying in fully connected layers or embeddings.
  - **Gradient penalties:** Jacobian norm to smooth predictions.

The database will be imported using PyTorch.

## Training

The model will be a simple CNN that has two `Conv2d`, followed by some pooling and two layers of dense activations. `Conv2d` is just some filter/kernel applied to the input with matrix values learned during training. The final result is then passed to ReLU for classification.

## Structure

`demo.ipynb` will contain the code for importing the data, defining the model, training, and evaluating it. Different `.py` files will then implement the regularization techniques we're trying to demonstrate, and `regularized.ipynb` will contain the code to run experiments with different regularization methods. `Comaprator.ipynb` will then put all of the results side-by-side for comparison.
