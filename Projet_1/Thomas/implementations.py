# -*- coding: utf-8 -*-
"""
Useful functions
"""

import numpy as np
from costs import compute_mse, calculate_log_likelihood
from helpers import *

#===============================================================================
#    LEAST SQUARES FUNCTIONS
#===============================================================================

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        
    loss = compute_mse(y, tx, w)
    return w, loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    
    for (batch_y, batch_tx) in batch_iter(y, tx, 1, max_iters):
        gradient = compute_gradient(batch_y, batch_tx, w)
        w = w - gamma*gradient

    loss = compute_mse(y, tx, w)
    return w, loss
    
def least_squares(y, tx):
    """calculate the least squares solution."""
    tx_t = tx.T
    w_star = np.linalg.solve(tx_t@tx, tx_t@y)
    mse = compute_mse(y, tx, w_star)
    return w_star, mse
    

#===============================================================================
#    RIDGE REGRESSION
#===============================================================================

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_prime = (lambda_)*(2*len(y))
    tx_t = tx.T
    
    try:
        w_star = np.linalg.solve(tx_t@tx + lambda_prime*np.identity(tx.shape[1]), tx_t@y)
    except np.linalg.LinAlgError:
        print("********** SINGULAR MATRIX, SKIPPING... **********")
        w_star = np.ones(tx.shape[1])
        
    mse = compute_mse(y, tx, w_star)
    
    return w_star, mse


#===============================================================================
#    LOGISTIC REGRESSION FUNCTIONS
#===============================================================================

def learning_by_gradient_descent(y, tx, w, lambda_, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_log_likelihood(y, tx, w, lambda_)
    grad = calculate_gradient_log(y, tx, w, lambda_)
    w = w - gamma*grad

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    just use 0 as lambda to not be penalized.
    """
    return reg_logistic_regression(y, tx, 0 , initial_w, max_iters, gamma)
    
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    penalized logistic regression.
    """
    w = initial_w
    threshold = 1e-8
    previous_loss = 0
    
    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, lambda_, gamma)
        # converge criterion
        if i > 0 and np.abs(previous_loss - loss) < threshold:
            break
        previous_loss = loss
    
    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, gamma, max_iters, batch_size):
    """
    penalized logistic regression using stochastic gradient descent.
    """
    w = initial_w
    threshold = 1e-8
    previous_loss = 0
    
    # start the logistic regression
    for i, (batch_y, batch_tx) in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(batch_y, batch_tx, w, lambda_, gamma)
        # converge criterion
        if i > 0 and np.abs(previous_loss - loss) < threshold:
            break
        previous_loss = loss
    
    return w, loss



