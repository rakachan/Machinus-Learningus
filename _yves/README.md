# Machine Learning - Project 1

This is the repository for the Machine Learning EPFL course first project.

This file explains the organisation and functions of the python scripts. For more information about the implementation, see the [PDF reports](https://github.com/rakachan/Machinus-Learningus/tree/master/Rapport) and the commented code.

First, you should place `train.csv` and `test.csv` in a `data` folder at the root of the project.

### ML_project_1.ipynb

Python notebook used to train the model of this project. The last part of the notebook contains code to create a submission, similar to `run.py`.

### costs.py

Contain 2 different cost functions like:
- **`calculate_mse`**: Mean square error
- **`compute_loss_neg_log_likelihood`**: Negative log likelihood

### cross_validation.py

Contain helper methods for cross validation.
- **`build_k_indices`** Builds k indices for k-fold cross validation
- **`cross_validation_visualization`** Creates a plot showing the accuracy given a lambda value
- **`best_model_cross_validation`** Find the best parameters for the cross validation

### helpers.py
Contain multiple methods for data processing and utilitary methods necessary to achieve the regression methods:
- ​
- **`sigmoid`** Apply the sigmoid function
- **`calculate_gradient_log`** Compute the gradient for log likelihood function
- **`batch_iter`** Generate a minibatch iterator for a dataset
- **`build_poly_matrix`** Add columns to the matrix which are powers of all columns raised to [2,…,degree + 1]
- **`standardize`** Standardize the original data set
- **`build_k_indices`** 
- **`accuracy`** Compute the accuracy of the prediction
- **`arrange_prediction`** Return the prediction of y in the original order 
- **`add_constant_columns`** Add front column made of 1s
- **`get_overall_predictions`** Compute the overall prediction using the weights of all 3 jets

### implementations.py
Contain the 6 regression methods needed for this project
- **`least_squares_GD`**: Linear regression using gradient descent
- **`least_squares_SGD`**: Linear regression using stochastic gradient descent
- **`least_squares`**: Least squares regression using normal equations
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: using stochastic gradient descent
- **`reg_logistic_regression`**: Penalized logistic regression
- **`reg_logistic_regression_SGD`** Penalized logistic regression using stochastic gradient descent

Helper function for logistic regression

* **`learning_by_gradient_descent`** One step of gradient descen using logistic regression

### proj1_helpers.py

Contain functions used to load the data and generate a CSV submission file.

### run.py
Script that generates the exact CSV file submitted on Kaggle.
