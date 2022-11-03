import numpy as np


def simple_predict(x, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    try:
        m = x.shape[0]
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if m == 0:
            return None
        if theta.ndim != 2 and theta.shape[1] != 1:
            return None
        if theta.shape[0] != x.shape[1] + 1:
            return None
        y = np.zeros((m,1))
        x_ = np.hstack((np.ones((m, 1)), x))
        for i in range(m):
            y[i] = np.dot(x_[i], theta)
        return y
    except Exception:
        return None