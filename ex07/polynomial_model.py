import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    if len(x) == 0 or power <=0:
        return None
    try:
        if len(x.shape) > 1 and x.shape[1] != 1:
            return None
        ret = np.zeros((len(x), power), int)

        for j, n in enumerate(x):
            # n + n^i
            for i in range(power):
                ret[j][i] = n[0] ** (i + 1)
        return ret
    except Exception:
        return None

