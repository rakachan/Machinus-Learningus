# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from implementations import *
from proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
def cross_validation(y, x, k_indices, k, regression_method, **args):
    """
    Completes k-fold cross-validation using the regression method
    passed as argument.
    """
    # get k'th subgroup in test, others in train
    k_test = k_indices[k]
    k_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[k_train]
    x_test = x[k_test]
    y_train = y[k_train]
    y_test = y[k_test]

    # compute weights using given method
    weights, loss = regression_method(y=y_train, tx=x_train, **args)

    # predict output for train and test data
    y_train_pred = predict_labels(weights, x_train)
    y_test_pred = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test, weights

def cross_validation_visualization(lambds, acc_train, acc_test):
    """visualization the curves of acc_train and acc_test."""
    plt.semilogx(lambds, acc_train, marker=".", color='b', label='train error')
    plt.semilogx(lambds, acc_test, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=10)
    plt.grid(True)
    plt.show()
