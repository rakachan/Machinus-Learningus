import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    #return 1/(1+np.exp(-t))
    return np.exp(-np.logaddexp(0, -t))

def calculate_loss(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    """
    m = tx.shape[0] #number of training examples
    w = reshape(w,(len(w),1))

    #y = reshape(y,(len(y),1))
    
    J = (1./m) * (-transpose(y).dot(log(sigmoid(tx.dot(w)))) - transpose(1-y).dot(log(1-sigmoid(x.dot(w)))))
    
    grad = transpose((1./m)*transpose(sigmoid(x.dot(w)) - y).dot(x))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]
    """
    xTw = tx@w#[:,np.newaxis]
    return np.sum(np.logaddexp(0, xTw)-y*xTw) + lambda_/2.0*np.sum(np.linalg.norm(w))
    """
    s = tx.dot(w)
    l = np.log(1 + tx.dot(w))

    for i in range(len(y)):
        if np.isinf(l[i]):
            l[i] = s[i]

    for i in range(len(y)):
        if l[i] == 0:
            l[i] = np.log(2) + s[i] / 2

    sum = 0

    for i in range(len(y)):
        sum += l[i] - y[i]*(tx[i].dot(w))

    return sum
    """
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return tx.T @ (sigmoid(tx@w)-y)

def update_inverse_hessian(old_w, w, old_gradient, gradient, h):
    s = w - old_w
    y = old_gradient - gradient
    phi = np.power(y.T@s, -1)
    left = phi * (s @ y.T)
    right = phi * (y @ s.T)
    return (1-left) * h * (1-right) + phi*s@s.T

def calculate_inverse_hessian(y, tx, w, xTx_inv):
    sig = sigmoid(tx@w)
    s = sig*(1-sig)
    s_inv = np.power(s, -1)
    return xTx_inv * s_inv, s

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    sig = sigmoid(tx@w)
    s = sig*(1-sig)
 
    return tx.T * s.flatten() @ tx, s