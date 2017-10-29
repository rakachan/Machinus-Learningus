# -*- coding: utf-8 -*-
"""
helper functions
"""

import numpy as np
from proj1_helpers import *
from costs import compute_mse, calculate_log_likelihood

def compute_gradient(y, tx, w):
    """Compute the gradient for MSE."""
    e = y - tx@w
    
    return (-1/len(y))*tx.transpose()@e

def sigmoid(t):
    """apply sigmoid function on t."""
    s = np.logaddexp(0, -t).flatten()
    result = np.exp(-s)
    
    return result

def calculate_gradient_log(y, tx, w, lambda_):
    """compute the gradient for log likelihood loss function."""
    
    s = sigmoid(tx@w)
    result = tx.T@(s - y) + (2*lambda_)*w

    return result

#===============================================================================
#    OTHER FUNCTIONS
#===============================================================================

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def build_poly_matrix(M, degree):
    """
    Add columns to the matrix which are powers of all columns
    Will add columns raised to [2, .., degree+1].
    """
    degrees = np.arange(2, degree+1)
    
    added_powers = [] 
    for col in M.T:
        tmp = np.power(col.reshape((-1, 1)), degrees)
        added_powers.append(tmp)

    #concatenate the matrix with its raised columns
    result = np.c_[M, np.column_stack(added_powers)]
    
    result = add_constant_columns(result)
    return result

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def accuracy(y, x=None, w=None, y_pred=None):
    """
    Compute the accuracy of the prediction.
    Can either be directly given the prediction
    or can just give the weight and data to 
    automatically make the prediction.
    """
    if y_pred is None:
        y_pred = predict_labels(w, x)

    return np.equal(y, y_pred).sum()/len(y)

def arrange_prediction(w_array, xx, jet_indices):
    """
    return the prediction of y in the original order
    from the all the data and weight subsets.
    """
    jet_2_3 = np.logical_or(jet_indices[2], jet_indices[3])
    
    total_length = 0
    for jet in range(3):
        total_length += len(xx[jet])
        
    y_jet = np.zeros(total_length)
    
    y_jet[jet_indices[0]] = predict_labels(w_array[0], xx[0])  
    y_jet[jet_indices[1]] = predict_labels(w_array[1], xx[1])    
    y_jet[jet_2_3] = predict_labels(w_array[2], xx[2]) 
    
    return y_jet

def add_constant_columns(tx):
    """
    Add front columns made of 1s.
    """
    
    tx = np.hstack((np.ones((tx.shape[0], 1)), tx))

    return tx

