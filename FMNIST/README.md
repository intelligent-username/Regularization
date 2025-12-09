# Fashion-MNIST

Fashion-MNIST is a dataset of 70,000 grayscale images of fashion items in 10 classes, with 7,000 images per class. It is similar to MNIST but more challenging. In this part of the project, we'll use it to demonstrate the following regularization methods:

  - Same as MNIST, but differences in difficulty allow **highlighting where regularization becomes essential**.
  - **Noise injection:** Gaussian noise on inputs.
  - **Gradient-based:** Spectral norm constraints on convolutional layers.

The database will be imported using PyTorch.

## Training

The model will be a simple CNN or MLP that has convolutional or dense layers, followed by pooling and activations. The final result is then passed to ReLU for classification.

## Structure

`demo.ipynb` will contain the code for importing the data, defining the model, training, and evaluating it. Different `.py` files will then implement the regularization techniques we're trying to demonstrate, and `regularized.ipynb` will contain the code to run experiments with different regularization methods. `Comparator.ipynb` will then put all of the results side-by-side for comparison.

With this setup, we will test out:

- Noise Injection
- Gradient Penalty-based Regularization
- Architectural Regularization
- And others as needed.
