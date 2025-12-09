# California Housing Dataset

The California Housing dataset is a tabular dataset with features related to housing prices in California. It is commonly used for regression tasks. In this part of the project, we'll use it to demonstrate the following regularization methods:

  - **Norm-based:** L1/L2 regression penalties.
  - **Bayesian:** Variational or spike-and-slab priors.
  - **Early stopping / adaptive optimizers:** improves generalization in regression.
  - **Representation:** Subspace regularization (project weights to low-dimensional space).
  - **Gradient penalties:** Jacobian penalties for smooth predictions.

The dataset will be imported using scikit-learn.

## Training

The model will be a simple MLP with dense layers for regression. The final output will be a continuous value for housing prices.

## Structure

`demo.ipynb` will contain the code for importing the data, defining the model, training, and evaluating it. Different `.py` files will then implement the regularization techniques we're trying to demonstrate, and `regularized.ipynb` will contain the code to run experiments with different regularization methods. `Comparator.ipynb` will then put all of the results side-by-side for comparison.

With this setup, we will test out:

- Norm-based Regularization
- Bayesian Regularization
- Early Stopping / Adaptive Optimizers
- Representation-based Regularization
- Gradient Penalty-based Regularization
- And others as needed.
