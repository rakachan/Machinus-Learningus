import numpy as np

from implementations import *
from helpers import *

def cross_validation(y, x, k_indices, k, lambda_):
    """return the accuracy of ridge regression for this k-fold."""
    # ***************************************************
    # get k'th subgroup in test, others in train
    # ***************************************************
    te_x = x[k_indices[k]]
    te_y = y[k_indices[k]]
    
    tr_x = np.delete(x, k_indices[k], axis=0)
    tr_y = np.delete(y, k_indices[k], axis=0)   
    
    # ***************************************************
    # regression
    # ***************************************************
    w_star, _ = ridge_regression(tr_y, tr_x, lambda_)
    
    # ***************************************************
    # calculate the accuracy for train and test data
    # ***************************************************
    accuracy_tr = accuracy(tr_y, tr_x, w_star)
    accuracy_te = accuracy(te_y, te_x, w_star)
    
    return accuracy_tr, accuracy_te, w_star
	
def best_model_cross_validation(x, y, seed, degrees_range, k_fold=4, lambdas=None):
    """
    This function will iterate over the given degrees and lambdas to
    find the ones which yield the best accuracy using cross validation.
    """
    degrees = range(degrees_range[0], degrees_range[1]+1)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
     
    #variables to store the best parameters and results    
    best1_accuracy = 0.0
    best1_lambda = 0
    best1_degree = 0
    best1_w = [];
    
    for i, degree in enumerate(degrees):
            
        best0_accuracy = 0.0
        best0_lambda = 0
        best0_w = [];
        
        #Add the columns with raised power.
        x_poly = build_poly_matrix(x, degree)
        
        for j, lambda_ in enumerate(lambdas):
            
            #prepare variables for the k-fold average
            tmp_te = 0
            tmp_w = 0

            for k in range(k_fold):
                _, accuracy_te, w_star = cross_validation(y, x_poly, k_indices, k, lambda_)
                    
                tmp_te += accuracy_te
                #We also average the weights
                tmp_w += w_star

            te_accuracy = tmp_te/k_fold

            if te_accuracy > best0_accuracy:
                best0_accuracy = te_accuracy
                best0_lambda = lambda_
                best0_w = tmp_w/k_fold
                
        if best0_accuracy > best1_accuracy:
            best1_accuracy = best0_accuracy
            best1_lambda = best0_lambda
            best1_degree = degree
            best1_w = best0_w
        
    return best1_accuracy, best1_w, best1_degree, best1_lambda