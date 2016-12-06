""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid
from check_grad import check_grad


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    y = np.zeros((data.shape[0]))
    #separate weight and biases
    w = weights[0:weights.size-1]
    b = weights[-1]
    z = np.dot(data, w) + b
    y= sigmoid(z)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    N = y.shape[0]
    ce = np.sum(targets * - np.log(y))
    frac_correct = np.sum(1-np.round(np.abs(targets - y)))/N

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        f,df = compute_base_loss_and_derivative(weights, data, targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    w_alpha = hyperparameters['weight_decay']
    f, df = compute_base_loss_and_derivative(weights, data, targets)
    if w_alpha != 0:
        w = weights[0:weights.size-1]
        f += (w_alpha / 2.0) * (np.dot(w.T,w))
        df[0:weights.size-1] += w_alpha * w
        f = f[0]
        f = f[0]
    return f, df

def compute_base_loss_and_derivative(weights, data, targets):
    # Again, we want to separate weight into weights and bias
    w = weights[0:weights.size - 1]
    b = weights[-1]
    z = np.dot(data, w) + b

    f = z * (1 - targets) + np.log(1 + np.exp(-z))
    f = np.sum(f)

    m = (1 - targets - sigmoid(-z))

    df = np.zeros_like(weights)
    df[0:weights.size - 1, 0] = np.sum(data.T * m.T, axis=1)
    df[-1] = np.sum(m)

    return f, df

if __name__ == '__main__':
    test_weights = np.array([[1, 2, 3]]).T
    test_data = np.array([[1, 0], [0,1], [0,0], [1,1], [2,2], [1,2], [-1,-1]])
    test_targets = np.array([[1, 1, 1, 1, 1, 1, 0]]).T

    test_dict = {}
    test_dict['weight_regularization'] = False

    logistic_predict(test_weights, test_data)

    logistic(test_weights, test_data, test_targets, test_dict)

    print check_grad(logistic, test_weights, 0.001, test_data, test_targets, test_dict)
