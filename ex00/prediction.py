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
    # x_1 = np.c_[np.ones(x.shape[0]), x]
    # return x_1.dot(theta)
    try:
        m = x.shape[0]
        y = np.zeros((m,1))
        for i,line in enumerate(x):
            val = theta[0][0]
            for teta, xi in zip(theta[1:],line):
                val += teta[0] * xi
            y[i][0] = val
        return y
    except Exception:
        return None