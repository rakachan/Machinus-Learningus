import numpy as np
from scripts.proj1_helpers import *
from logistic_helpers import *

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



def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    e = y - tx@w
    loss = compute_mse(y, tx, w)
    n = len(y)
    return -tx.T @ e / n, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient, loss = compute_gradient(y, tx, w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma*gradient
        # store w and loss
        """
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))"""
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        for batchy, batchx in batch_iter(y, tx, batch_size):
            gradient, loss = compute_gradient(batchy, batchx, w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma*gradient
        # store w and loss
        """
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))"""

    return w, loss

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.lstsq(tx, y)[0]
    return w, compute_mse(y, tx, w)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    l = lambda_ * 2 * tx.shape[0]
    w = np.linalg.solve(tx.T @ tx + l * np.identity(tx.shape[1]),tx.T @ y)
    return w, compute_mse(y, tx, w)

def learning_by_newton_method(y, tx, w, lambda_):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient and hessian: TODO
    # ***************************************************
    hessian = calculate_hessian(y, tx, w)
    #print('hessian done')
    gradient = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    #np.linalg.inv(hessian)
    w = w - lambda_* gradient
    return w, loss

def logistic_regression_newton_method(y, x, initial_w, max_iter, lambda_):
    # init parameters
    #max_iter = 100
    threshold = 1e-8
    #lambda_ = 0.1
    losses = []

    # build tx
    tx = x#tx = np.c_[np.ones((y.shape[0], 1)), x]
    #w = np.zeros((tx.shape[1], 1))
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, lambda_)
        # log info
        if iter % 20 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            y_pred = predict_labels(w, x)
            acc = accuracy(y, y_pred)
            print('Accuracy of the method: {}'.format(acc))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method")
    return w, loss

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_, i):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    y = normalize_y(y)
    hessian, s = calculate_hessian(y, tx, w)
    #hessian_inv, s = calculate_inverse_hessian(y, tx, w, xTx_inv)
    """
    sig = sigmoid(tx@w)
    s = sig*(1-sig)
    """
    gradient = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w, lambda_)
    """
    if i == 0:
        hessian_inverse = hk_1
    else :
        hessian_inverse = update_inverse_hessian(old_w, w, old_gradient, gradient, hk_1)
    """
    
    xw = tx@w
    z = xw + np.power(s,1) * (y - sigmoid(xw)) 
    w = np.linalg.pinv(hessian) @ tx.T * s @ z
    #w = w - gamma*gradient - lambda_/2*w.shape[0]
    return loss, w

def reg_logistic_regression(y, x, initial_w, max_iter, gamma, lambda_):
    threshold = 1e-8
    losses = []


    tx = x
    w = initial_w
    loss = 0
   

    for iter in range(max_iter):
        loss, ws = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, iter)
        # log info
        if iter % 50 == 0:
            y_partial = predict_labels(ws, tx)
            acc = accuracy(y, y_partial)
            print("Current iteration={i}, loss={l}, acc={a}, gamma={g}, lambda={b}".format(i=iter, l=loss, a=acc, g=gamma, b=lambda_))
            
        # converge criterion
        losses.append(loss)
        old_w = w
        w = ws
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss