import numpy as np


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        x_1 = np.c_[np.ones(x.shape[0]), x]
        if x.shape[1] == theta.shape[0]: # (_,n) (n, _)
            return x.dot(theta)
        return x_1.dot(theta)
    except Exception:
        return None