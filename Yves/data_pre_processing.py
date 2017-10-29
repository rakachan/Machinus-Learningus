# -*- coding: utf-8 -*-
"""
Useful functions for data pre processing
"""

import numpy as np
from helpers import standardize

def get_jet_indices(x):
    """
    Get the corresponding indices
    for each jet values.
    """
    
    jet_indices = []
    for i in range(4):
        jet_indices.append(x[:, 22] == i)
        
    return jet_indices

def preprocess_data(x, y, augment=True, clean=True):
    """
    return in an array at postion 0, 1, and 2 the rows of x
    where the jet value is 0, 1 and (2 or 3).
    """
    
    jet_indices = get_jet_indices(x)
    xx = []
    yy = []

    xx.append(x[jet_indices[0]])
    yy.append(y[jet_indices[0]])
    
    xx.append(x[jet_indices[1]])
    yy.append(y[jet_indices[1]])
    
    #We put jet values of 2 and 3 together since
    #they have the same columns to keep
    jet_2_3 = np.logical_or(jet_indices[2], jet_indices[3])
    xx.append(x[jet_2_3])
    yy.append(y[jet_2_3])
    
    #clean each dataset
    if clean:
        xx = clean_data(xx)
    
    #standardize each dataset
    for i in range(3):
        x_stand, _, _ = standardize(xx[i])
        xx[i] = x_stand
       
    #augment each dataset
    if augment:
        xx = augment_data(xx)
    
    
    return xx, yy, jet_indices

def clean_data(xx):
    """
    Remove the useless (all values -999) columns for each jet.
    Also replace the missing values left with 0.
    """
    
    to_remove = []
    #columns to remove from jet 0
    to_remove.append([22, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29])
    #columns to remove from jet 1
    to_remove.append([22, 4, 5, 6, 12, 26, 27, 28])
    #columns to remove from jet 2 and 3
    to_remove.append([22])


    for i in range(3):
        xx[i] = np.delete(xx[i], to_remove[i], axis=1)
        
    for i in range(3):
        xx[i][xx[i]==-999] = 0
    
    return xx

def augment_data(xx):
    """
    Add columns with various transformation of 
    the original columns for each jet.
    """
       
    for jet in range(3):
                    
        #add new columns formed by the
        #cos function applied to all columns.
        augment1 = []
        for i in range(xx[jet].shape[1]):
            tmp = np.cos(xx[jet][:, i])
            augment1.append(tmp)

        #add new columns formed by the
        #sin function applied to all columns.
        augment2 = []
        for i in range(xx[jet].shape[1]):
            tmp = np.sin(xx[jet][:, i])
            augment2.append(tmp)
          
        #add new columns formed by the
        #sqrt of all positive columns.
        augment3 = []
        for i in range(xx[jet].shape[1]):
            if np.all(xx[jet][:, i] >= 0): 
                tmp = np.sqrt(xx[jet][:, i])
                augment3.append(tmp)
                
        #add new columns formed by the
        #log of all positive columns.
        augment4 = []
        for i in range(xx[jet].shape[1]):
            if np.all(xx[jet][:, i] > 0): 
                tmp = np.log(xx[jet][:, i])
                augment4.append(tmp)

        #horizontaly stack the originial matrix with the augmented ones.
        xx[jet] = np.hstack((xx[jet], np.array(augment1).T, np.array(augment2).T, np.array(augment3).T, np.array(augment4).T))
    
    return xx
 
    