import numpy as np
from implementations import *
from scripts.proj1_helpers import *

def build_poly2(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #return np.array([np.vander(elem, degree+1, increasing=True).T.flatten() for elem in x])
    
    vander = np.concatenate(np.power(np.hsplit(x, x.shape[1]), degree), axis=1)
    combinations=[]
    for i in range(len(vander)):
        for j in range(i+degree, len(vander)):
            combinations.append(vander[i]*vander[j])
            
    return np.concatenate(vander, combinations)
    #return po.reshape(-1, po.shape[-1])
    #return np.vander(x, degree+1, increasing=True)
    
def build_poly(x, degree):
    """ Apply a polynomial basis to all the X features. """
    # First, we find the combinations of columns for which we have to
    # compute the product
    m, n = x.shape

    combinations = {}

    # Add combinations of same column power
    for i in range(n * degree):
        if i < n:
            combinations[i] = [i]
        else:
            col_number = i - n
            cpt = 2
            while col_number >= n:
                col_number -= n
                cpt += 1
            combinations[i] = [col_number] * cpt

    # Add combinations of products between columns
    cpt = i + 1

    for i in range(n):
        for j in range(i + 1, n):
            combinations[cpt] = [i, j]
            cpt = cpt + 1

    # Now we can fill a new matrix with the column combinations
    eval_poly = np.zeros(
        shape=(m, n + len(combinations))
    )

    for i, c in combinations.items():
        eval_poly[:, i] = x[:, c].prod(1)

    # Add square root
    for i in range(0, n):
        eval_poly[:, len(combinations) + i] = np.abs(x[:, i]) ** 0.5
        
    #eval_poly = np.append(eval_poly, np.cos(x), axis=1)
    #eval_poly = np.append(eval_poly, np.sin(x), axis=1)

    return eval_poly

def remove_correlated(m):
    s = np.corrcoef(m)
    mask_column = np.unique(np.array([np.where(row < 0.8)[0] for row in s]))
    uncorrelated_m = m[:, mask_column]
    if uncorrelated_m.shape[1] != m.shape[1]:
        print("uncorrelated")
    return uncorrelated_m
    

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis = 0).flatten()
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    
    x_tr = x[train_indices]
    y_tr = y[train_indices]
    x_te = x[test_indices]
    y_te = y[test_indices]
    y_pred_train = np.zeros(y_tr.shape)
    y_pred_test = np.zeros(y_te.shape)
    train_indices_nj = jets_indices(x_tr)
    test_indices_nj = jets_indices(x_te)
    for train_index, test_index in zip(train_indices_nj, test_indices_nj):
        x_tr_nj = x_tr[train_index[0]][:,train_index[1]]
        x_te_nj = x_te[test_index[0]][:,test_index[1]]
        y_tr_nj = y_tr[train_index[0]]
        
        m_tr = build_poly(x_tr_nj, degree)
        m_te = build_poly(x_te_nj, degree)

        w_tr, mse = ridge_regression(y_tr_nj, m_tr, lambda_)

        loss_tr = mse
        #loss_te = compute_mse(y_te, m_te, w_tr)
        y_pred_train[train_index] = predict_labels(w_tr, x_tr_nj)
        y_pred_test[test_index] = predict_labels(w_tr,x_te_nj)
    train_accuracy = accuracy(y_tr, y_pred_train)
    test_accuracy = accuracy(y_te, y_pred_test)
    
    return train_accuracy, test_accuracy

def cross_validation_generic(y, x, regression, k_indices, k, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis = 0).flatten()
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    x_tr = x[train_indices]
    y_tr = y[train_indices]
    x_te = x[test_indices]
    y_te = y[test_indices]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    degree = 7
    
    y_pred_train = np.zeros(y_tr.shape)
    y_pred_test = np.zeros(y_te.shape)
    train_indices_nj = jets_indices(x_tr)
    test_indices_nj = jets_indices(x_te)
    for train_index, test_index in zip(train_indices_nj, test_indices_nj):
        x_tr_nj = x_tr[train_index]
        x_te_nj = x_te[test_index]
        y_tr_nj = y_tr[train_index]
        
        m_tr = build_poly(x_tr_nj, degree)
        m_te = build_poly(x_te_nj, degree)

        w_tr, mse = regression(y_tr_nj, m_tr, initial_w, max_iters, gamma)

        loss_tr = mse
        #loss_te = compute_mse(y_te, m_te, w_tr)
        y_pred_train[train_index] = predict_labels(w_tr, x_tr_nj)
        y_pred_test[test_index] = predict_labels(w_tr,x_te_nj)
    train_accuracy = accuracy(y_tr, y_pred_train)
    test_accuracy = accuracy(y_te, y_pred_test)
    return w_tr, train_accuracy, test_accuracy

def cross_validation_logistic(y, x, k_indices, k, initial_w, max_iters, gamma, lambda_, poly, degree=1):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis = 0).flatten()

    x_tr = x[train_indices]
    y_tr = y[train_indices]
    x_te = x[test_indices]
    y_te = y[test_indices]
    
    y_pred_train = np.zeros(len(y_tr))
    y_pred_test = np.zeros(len(y_te))
    train_indices_nj = jets_indices(x_tr)
    test_indices_nj = jets_indices(x_te)
    w = initial_w
    
    for train_nj, test_nj in zip(train_indices_nj, test_indices_nj):
        if len(train_nj[0]) != 0:
            w = initial_w[train_nj[1]]
            x_tr_nj = x_tr[train_nj[0]][:,train_nj[1]]
            x_te_nj = x_te[test_nj[0]][:,test_nj[1]]
            y_tr_nj = y_tr[train_nj[0]]
            x_tr_nj = standardize(x_tr_nj)
            x_te_nj = standardize(x_te_nj)

            if poly:
                x_tr_nj = build_poly(x_tr_nj, degree)
                x_te_nj = build_poly(x_te_nj, degree)
                x_tr_nj = np.hstack((np.ones((x_tr_nj.shape[0], 1)), x_tr_nj))
                w = np.zeros(x_tr_nj.shape[1])

            w, mse = reg_logistic_regression(y_tr_nj, x_tr_nj, w, max_iters, gamma, lambda_)
            y_pred_train[train_nj[0]] = predict_labels(w, x_tr_nj)
    print(y_pred_train)
    print(y_tr)
    train_accuracy = accuracy(y_tr, y_pred_train)
    return y_pred_train, train_accuracy, 0#test_accuracy

def cross_validation_jets(y, x, k_indices, k, initial_w, max_iters, gamma, lambda_, degree=1):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis = 0).flatten()

    x_tr = x[train_indices]
    y_tr = y[train_indices]
    x_te = x[test_indices]
    y_te = y[test_indices]

    #x_tr = standardize(x_tr)
    #x_te = standardize(x_te)


    x_tr = build_poly(x_tr, degree)
    #x_te = build_poly(x_te, degree)
    x_tr = np.hstack((np.ones((x_tr.shape[0], 1)), x_tr))
    w = np.zeros(x_tr.shape[1])

    w, mse = reg_logistic_regression(y_tr, x_tr, w, max_iters, gamma, lambda_)
    y_pred = predict_labels(w, x_tr)
    train_accuracy = accuracy(y_tr, y_pred)
    return y_pred, train_accuracy, w

def find_param_jets(y,x):
    seed = 1
    k_fold = 2
    max_iters = 200
    
    b_gamma = []
    b_lambda = []
    b_degree = []
    
    x=x[:,1:]
    
    w = np.zeros(x.shape[1])
    best_acc = 0
    best_average_acc = 0
    
    y_pred = np.zeros(len(y))
    y_pred_average = np.zeros(len(y))
    jets_ind = jets_indices(x)
    x = standardize(x)
    
    gamma = np.arange(0.1, 1, 0.1)
    lambda_ = [0.001, 0.001, 0.01, 0.6, 0.1]
    
    for index in [jets_ind[2]]:
        x_tr = x[index[0]][:,index[1]]
        y_tr = y[index[0]]
        
        k_indices = build_k_indices(y_tr, k_fold, seed)
        
        best_degree = 0
        for degree in range(4, 8):
            best_gamma = 0
            
            for g in gamma:
                best_lambda = 0
                
                for l in lambda_:
                    ws = []
                    
                    for i in range(k_fold): 

                        y_pred, acc_train, w = cross_validation_jets(y_tr, x_tr, k_indices, i, w, max_iters, g, l, degree)
                        ws.append(w)
                        if acc_train < 0.7:
                            break
                        if acc_train > best_acc:
                            best_acc = acc_train
                            best_pred = y_pred
                            best_lambda = l
                            best_gamma = g
                            best_degree = degree

                            print('{} | d: {} | g: {} | l: {} | fold {}'.format(acc_train, degree, g, l, i))
                            
                    average_w = np.array(ws).mean(axis=0)
                    x_poly = build_poly(x_tr, degree)
                    x_poly = np.hstack((np.ones((x_poly.shape[0], 1)), x_poly))
                    average_pred = predict_labels(average_w, x_poly)

                    average_acc = accuracy(y_tr, average_pred)
                    
                    if average_acc > best_average_acc:
                        best_average_acc = average_acc
                        best_average_pred = average_pred
                        print("-> new average acc: {}".format(average_acc))
                        
        y_pred[index[0]] = best_pred
        y_pred_average[index[0]] = best_average_pred
        b_gamma.append(best_gamme)
        b_lambda.append(best_lambda)
        b_degree.append(best_degree)
                        
    total_acc = accuracy(y, y_pred)
    total_average_acc = accuracy(y, y_pred_average)
    
    return total_acc, total_average_acc, b_degree, b_gamma, b_lambda


