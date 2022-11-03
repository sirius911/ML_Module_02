import numpy as np




def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    try:
        m = x.shape[0]
        n = x.shape[1]

        if (y.shape[1] != 1) or (theta.shape[1] != 1) \
        or (m != y.shape[0]):
            return None
        x_1 = np.hstack((np.ones((m, 1)), x))
        x_t = x_1.T
        h = x @ theta
        diff = h - y
        grad = x_t @ diff
        return grad / m
    except Exception:
        return None