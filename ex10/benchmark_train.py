import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
   
    # figure, axis = plt.subplots(3, figsize=(20, 10))
    

    # axis[0].scatter(weight, target, c='blue')
    # axis[0].set_xlabel("weight in ton")
    # axis[1].scatter(prod_distance, target, c='red')
    # axis[1].set_xlabel("dist in Mkm")
    # axis[2].scatter(time_delivery, target, c='orange')
    # axis[2].set_xlabel("time in days")
    # plt.show()

    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)
    weight_train = x_train.T[0]
    # normalisation
    weight_train = zscore(weight_train)
    y_train = zscore(y_train)
    y_test = zscore(y_test)
    # x_train.T[0] == weight
    # x_train.T[1] == distance
    # x_train.T[2] == time
   
    feature = add_polynomial_features(weight,4)
    # print(feature.shape)
    feature_weight = add_polynomial_features(weight_train, 4)
    # print(feature_weight.shape)
    # print(target.shape)
    # print(y_train.shape)
    theta = np.random.rand(5, 1)
    mylr = MyLR(thetas=theta, alpha=1e-2, max_iter=1000000, progress_bar=True)
    # print(MyLR.gradien_(feature_weight, y_train, theta))
    mylr.fit_(feature_weight, y_train)
    prediction = mylr.predict_(feature_weight)
    # print(prediction)
    mse = MyLR.mse_(y_test.T,prediction)
    print(f"MSE = {mse}")
    plt.scatter(weight_train, y_train)
    plt.show()