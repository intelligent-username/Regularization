# UCI Wine Datasets (Classification)

The UCI Wine dataset is a small tabular dataset for classification of wine types based on chemical features. It is commonly used for demonstrating classification algorithms. In this part of the project, we'll use it to demonstrate the following regularization methods:

  - **Norm-based:** L1/L2, group Lasso on feature groups.
  - **Bayesian:** Small-scale spike-and-slab illustration.
  - **Early stopping / learning rate schedules:** immediate effect visible.
  - **Architectural:** Weight tying or low-rank factorization in small MLP.
  - **Output:** Label smoothing.

The dataset will be imported using scikit-learn.

## Training

The model will be a tiny MLP or logistic regression with dense layers for classification. The final output will be class probabilities.

## Structure

`demo.ipynb` will contain the code for importing the data, defining the model, training, and evaluating it. Different `.py` files will then implement the regularization techniques we're trying to demonstrate, and `regularized.ipynb` will contain the code to run experiments with different regularization methods. `Comparator.ipynb` will then put all of the results side-by-side for comparison.
