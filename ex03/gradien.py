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
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    # print(f"x.shape={x.shape}\ty.shape={y.shape}\ttheta.shape={theta.shape}")
    try:
        m = len(x)
        x_1 = np.c_[np.ones(x.shape[0]), x]
        x_t = x_1.T
        # print(f"x_t.shape = {x_t.shape}")
        if x.shape[1] == theta.shape[0]:
            h =  x.dot(theta)
        else:
            h = x_1.dot(theta)
        diff = h - y
        return x_t.dot(diff) / m
    except Exception:
        return None