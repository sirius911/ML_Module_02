from math import floor
import numpy as np
from re import I
import time

def ft_progress(lst):
    print("\x1b[?25l") # hide cursor
    i = 1
    start = time.time()
    while i <= len(lst):
        yield i
        pourc = int(i * 100 / len (lst))
        nb = int(i * 20 / len(lst))
        arrow = ">".rjust(nb, "=")
        top = time.time() - start
        eta = (len(lst) * top / i) - top
        if eta <= 100:
            etah = 0
            etam = 0
            etas = eta
        elif eta > 100 and eta < 3600:
            etah = 0
            etam = floor(eta / 60)
            etas = eta - (etam * 60)
        else:
            etah = floor(eta / (60 * 60))
            etam = floor((eta - (etah * 60 * 60)) / 60)
            etas = eta - (etah * 60 *60) - (etam * 60)
        if top <= 100:
            toph = 0
            topm = 0
            tops = top
        elif top > 100 and top < 3600:
            toph = 0
            topm = floor(top / 60)
            tops = top - (topm * 60)
        else:
            toph = floor(top / (60 * 60))
            topm = floor((top - (toph * 60 * 60)) / 60)
            tops = top - (toph * 60 * 60) - (topm * 60)
        label = f"ETA:"
        if etah > 0:
            label = f"{label} {etah}h"
        if etam > 0 or etah > 0:
            label = f"{label} {etam:02}mn"
        label = f"{label} {etas:05.2f}s [{pourc:3}%] [{arrow:<20}] {i}/{len(lst)} | elapsed time"
        if toph > 0:
            label = f"{label} {toph}h"
        if topm > 0 or toph > 0:
            label = f"{label} {topm}mn"
        label = f"{label} {tops:05.2f}s    "
        print(f"{label}", end='\r', flush=True)
        i += 1
    print("\x1b[?25h") #show cursor

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        x_1 = np.c_[np.ones(x.shape[0]), x]
        if x.shape[1] == theta.shape[0]: # (_,n) (n, _)
            return x.dot(theta)
        return x_1.dot(theta)
    except Exception:
        return None

def gradient_(x, y, theta):
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
        h = predict_(x, theta)
        # if x.shape[1] == theta.shape[0]:
        #     h =  x.dot(theta)
        # else:
        #     h = x_1.dot(theta)
        diff = h - y
        return x_t.dot(diff) / m
    except Exception:
        return None

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    
    new_theta = theta.copy()
    for i in ft_progress(range(max_iter)):
        gradien = gradient_(x,y,new_theta)
        # print(gradien)
        for n, g_n in enumerate(gradien):
            tn = new_theta[n][0]            
            tn -= (alpha * gradien[n][0])
            new_theta[n][0] = tn
    return(new_theta)
