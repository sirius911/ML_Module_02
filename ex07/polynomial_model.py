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
    try:
        # if not isinstance(x, np.ndarray) or not isinstance(power, int):
        #     print("Error in add_polynomial_features: not numpy array")
        #     return None
        # if not x.ndim in [1, 2]:
        #     print("Error in add_polynomial_features: x.ndim not in [1, 2]")
        #     return None
        # if power <0:
        #     print("Error in add_polynomial_features: powe < 0")
        #     return None
        # if x.ndim == 2 and x.shape[1] != 1:
        #     print("Error in add_polynomial_features: x.shape != 1")
        #     return None
        if power == 0:
            return np.ones((x.shape[0], 1))
        if power == 1:
            return np.array(x, copy=True).reshape(-1,1)
        return np.vander(x.reshape(-1,), N=power + 1, increasing=True)[:,1:]
    except Exception as e:
        print(e)
        return None

