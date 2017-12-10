# -*- coding: utf-8 -*-
"""
Project 1 method implementations.
Authors: Thomas Garcia, Yves Lamonato, Pierre-Alexandre Lee
"""

import numpy as np
from helpers import *

def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # if initial_w is None, we initialize it
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters to store weight and loss
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = calculate_gradient(y, tx, w)
        loss = calculate_mse(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

    return w, loss
    
def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    """
    # if initial_w is None, we initialize it to a zeros vector
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters of the algorithm
    batch_size = 1
    batch_size = 32

    # Define parameters to store w and loss
    loss = 0
    w = initial_w

    for n_iter, [ybatch, xbatch] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # compute gradient and loss
        grad = calculate_gradient(ybatch, xbatch, w)
        loss = calculate_mse(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

    return w, loss

def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    x = tx @ w
    calculate_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    xtx = tx.T @ tx
    w = np.array(np.linalg.solve(xtx + 2*lambda_*N*np.eye(len(xtx)), tx.T @ y))
    return w, calculate_mse(y, tx, w)
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    # Map y to [|0, 1|]
    y = (1 + y) / 2
    losses = []
    threshold = 0.1

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        
        w, loss = learning_by_log_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized Logistic regression"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    # Map y to [|0, 1|]
    y = (1 + y) / 2
    losses = []
    threshold = 0.1

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_log_gradient_descent(y, tx, w, lambda_, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
