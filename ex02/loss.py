import numpy as np

    
def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print("Not numpy.array")
        return None
    try:
        if y.shape[1] != 1 or y_hat.shape[0] != y.shape[0]:
            print("incompatible shape")
            return None
        loss = float((y_hat - y).T @ (y_hat - y) / (2.0 * y.shape[0]))
        return loss
    except Exception:
        return None
