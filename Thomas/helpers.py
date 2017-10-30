import numpy as np

def sigmoid(t):
    """ Apply sigmoid function on t. """
    return np.exp(-np.logaddexp(0, -t))
    
def build_column_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.vander(x, degree+1, increasing=True)

def build_poly(x, degree):
	data = build_column_poly(x[:,0], degree)
	m, n = x.shape
	for i in range(1, n):
		np.append(data, build_column_poly(x[:,i], degree), axis = 1)
	return data
	
def calculate_gradient(y, tx, w):
    """ Linear regression using gradient descent. """
    e = y - tx.dot(w)
    n = len(y)
    return -np.dot(tx.T, e) / n

def calculate_mse(y, tx, w):
	""" Computes the mse """
	e = y - tx.dot(w)
	return 1 / 2 * np.mean(e ** 2)

def calculate_log_loss(y, tx, w):
	""" Computes the cost by negative log likelihood."""
	predict = tx.dot(w)
	s = np.logaddexp(0, predict).flatten()
	return (s - y*predict).sum()
    #return np.sum(np.log(1 + np.exp(predict)) - y * predict)

def calculate_log_gradient(y, tx, w):
    """ Compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)
    
def learning_by_log_gradient_descent(y, tx, w, gamma):
    """
    Does one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_log_loss(y, tx, w)
    grad = calculate_log_gradient(y, tx, w)
    w = w - gamma * grad
    return w, loss
    
def learning_by_penalized_log_gradient_descent(y, tx, w, lambda_, gamma):
    """
    Does one step of gradient descen using regularized logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_log_loss(y, tx, w)# + lambda_*np.norm(w)**2/2.
    grad = calculate_log_gradient(y, tx, w) + lambda_*w
    w = w - gamma * grad
    return w, loss

def jet_mask(x):
    """
    Returns 3 data models corresponding to the rows of x with a jet value
    of 0, 1, 2 and 3 respectively.
    """
    jet0 = x[:,22] == 0
    jet1 = x[:,22] == 1
    jet2 = x[:,22] >  1
    return jet0, jet1, jet2
    
def split_models_and_process(x, y):
	
	# Indices of -999 for each jet
	#na_0 = []
	#na_1 = []
	#na_2 = []
	jet0, jet1, jet2 = jet_mask(x)
	
	
	na_0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28]
	na_1 = [4, 5, 6, 12, 22, 26, 27, 28]
	na_2 = [22]
	
	# Sometimes, x[0] == -999 !!!
	del0 = np.delete(x[jet0], na_0, axis=1)
	del1 = np.delete(x[jet1], na_1, axis=1)
	del2 = np.delete(x[jet2], na_2, axis=1)
	
	return process_data(del0, 3), process_data(del1, 3), process_data(del2, 3), y[jet0], y[jet1], y[jet2]

def build_product(data):
	m, n = data.shape
	for i in range(n):
		for j in range(n):
			if (i!=j):
				data = np.append(data, np.array([data[:, i] * data[:, j]]).T, axis = 1)
	return data
	
def process_data(x, degree):
	# Ignore first column as it sometimes has -999 in it.
	data = np.delete(x, [0], axis = 1)
	data, _, _ = standardize(data)
	data = np.hstack((np.ones((data.shape[0], 1)), data))
	"""data = np.append(build_poly(data, degree), build_product(data), axis = 1)
	data = np.append(data, np.cos(x), axis = 1)
	data = np.append(data, np.sin(x), axis = 1)"""
	
	return data
	
"""data = np.append(data, np.array([(data[:, 3] * data[:, 11])]).T, axis=1)
data = np.append(data, np.array([(data[:, 2] * data[:, 6])]).T, axis=1)
data = np.append(data, np.array([(data[:, 6] * data[:, 29])]).T, axis=1)
data = np.append(data, np.array([(data[:, 2] * data[:, 2])]).T, axis=1)
data = np.append(data, np.array([(data[:, 3] * data[:, 6])]).T, axis=1)
data = np.append(data, np.array([(data[:, 2] * data[:, 31])]).T, axis=1)
data = np.append(data, np.array([(data[:, 2] * data[:, 29])]).T, axis=1)
data = np.append(data, np.array([(data[:, 18] * data[:, 29])]).T, axis=1)"""
	

def compute_accuracy(y_pred, y):
    """Computes accuracy"""
    sum = 0
    for idx, y_val in enumerate(y):
        if y_val == y_pred[idx]:
            sum += 1
    return sum / len(y)
    
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

def compute_gradient(y, tx, w):
    """Compute the gradient for MSE."""
    e = y - tx@w
    return (-1/len(y))*tx.transpose()@e
    
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x
