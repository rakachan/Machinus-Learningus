# -*- coding: utf-8 -*-
"""
costs functions
"""

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE."""
    e = (y - tx@w)
    N = len(y)
    result = (1/(N << 1))*e.transpose()@e
    
    return result

def calculate_log_likelihood(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    t = tx@w
    s = np.logaddexp(0, t).flatten()
    
    return (s - y*t).sum() + lambda_*(w.T@w)