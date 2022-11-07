import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) == 0 or len(x.shape) > 2:
        return None
    if len(x.shape) == 2 and x.shape[1] != 1:
        return None
    try:
        return ((x - np.mean(x)) / np.std(x))
    except Exception:
        return None

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)

from polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

if __name__ == "__main__":
    # Importation of the dataset + basic checking:
    try:
        data = pd.read_csv("space_avocado.csv", dtype=np.float64)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    target = data.target.values.reshape(-1, 1) #price
    Xs = data[['weight','prod_distance','time_delivery']].values # features
    weight = data.weight.values.reshape(-1, 1)
    prod_distance = data.prod_distance.values.reshape(-1, 1)
    time_delivery = data.time_delivery.values.reshape(-1, 1)
   
    figure, axis = plt.subplots(3, figsize=(20, 10))
    

    axis[0].scatter(weight, target, c='blue')
    axis[0].set_xlabel("weight in ton")
    axis[1].scatter(prod_distance, target, c='red')
    axis[1].set_xlabel("dist in Mkm")
    axis[2].scatter(time_delivery, target, c='orange')
    axis[2].set_xlabel("time in days")
    plt.show()

    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)
    
    # weight
    weight_train = x_train.T[0]
    weight_test = x_test.T[0]
    
    # normalisation
    y_train = zscore(y_train)
    y_test = zscore(y_test)
    weight_train_norm = zscore(weight_train)
    weight_test_norm = zscore(weight_test)
   
    try:
        mylr = pickle.load( open( "weight.pickle", "rb" ) )
        
    except FileNotFoundError:

        theta = np.random.rand(4, 1)
        mylr = MyLR(thetas=theta, alpha=1e-2, max_iter=1000000, progress_bar=True)
        mylr.fit_(add_polynomial_features(weight_train_norm, 3), y_train)
        # mylr.thetas = np.array([[ 1.20893361e-01], [ 9.47634630e-01], [-1.18295546e-01], [-6.23921560e-04], [-1.46523240e-03]])
        print(mylr.thetas)
        print("save mylr ...", end='')
        pickle.dump( mylr, open( "weight.pickle", "wb" ) )
        print(" ok")
        
    prediction = mylr.predict_(add_polynomial_features(weight_test_norm, 3))
    # print(prediction)
    mse = MyLR.mse_(y_test.T,prediction)
    print(f"MSE = {mse}")
    plt.scatter(weight_train, y_train)
    plt.scatter(weight_test, prediction)
    # plt.plot(weight_test, prediction)
    plt.show()

    #distance
    distance_train = x_train.T[1]
    distance_test = x_test.T[1]
    #normalisation
    distance_train_norm = zscore(distance_train)
    distance_test_norm = zscore(distance_test)

    try:
        mylr = pickle.load( open( "distance.pickle", "rb" ) )
    except FileNotFoundError:
        theta = np.random.rand(5, 1)
        mylr = MyLR(thetas=theta, alpha=1e-2, max_iter=1000000, progress_bar=True)
        # mylr.fit_(add_polynomial_features(distance_train_norm, 4), y_train)
        mylr.thetas = np.array([[-0.28295574], [-0.36618452], [ 0.32636224], [ 0.15421623], [-0.02273116]])
        print(mylr.thetas)
        print("save mylr ...", end='')
        pickle.dump( mylr, open( "distance.pickle", "wb" ) )
        print(" ok")
        
    prediction = mylr.predict_(add_polynomial_features(distance_test_norm,4))
    # print(prediction)
    mse = MyLR.mse_(y_test.T,prediction)
    print(f"MSE = {mse}")
    plt.scatter(distance_train, y_train)
    plt.scatter(distance_test, prediction)
    # x_sampling = np.linspace(1000, 3000, 4000)
    # plt.plot(x_sampling, mylr.predict_(add_polynomial_features(x_sampling,4)), c='r')
    plt.show()
    print(f"pour x=1500, prix = {mylr.predict_(add_polynomial_features(np.array([[1500]]), 4))}")

    #time
    time_train = x_train.T[2]
    time_test = x_test.T[2]
    #normalisation
    time_train_norm = zscore(time_train)
    time_test_norm = zscore(time_test)
    try:
        mylr = pickle.load( open( "time.pickle", "rb" ) )
    except FileNotFoundError:
        theta = np.random.rand(5, 1)
        mylr = MyLR(thetas=theta, alpha=1e-2, max_iter=1000000, progress_bar=True)
        mylr.fit_(add_polynomial_features(time_train_norm, 4), y_train)
        # mylr.thetas = np.array([[ 0.00348599], [ 0.02234647], [ 0.00741201], [-0.01350792], [-0.00617828]])
        print(mylr.thetas)
        print("save mylr ...", end='')
        pickle.dump( mylr, open( "time.pickle", "wb" ) )
        print(" ok")
    prediction = mylr.predict_(add_polynomial_features(time_test_norm,4))
    # print(prediction)
    mse = MyLR.mse_(y_test.T,prediction)
    print(f"MSE = {mse}")
    plt.scatter(time_train, y_train)
    plt.scatter(time_test, prediction)
    plt.show()