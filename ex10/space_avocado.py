from cmath import inf
import os
import sys
import numpy as np
import pandas as pd
from ft_yaml import load_model
from polynomial_model import add_polynomial_features
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

class Normalizer():
    def __init__(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        pass
        
    def norme(self, X):
        X_tr = np.copy(X)
        X_tr -= self.mean_
        X_tr /= self.std_
        return X_tr

def main():
    # Importation of the dataset
    print("Loading models ...")
    try:
        data = pd.read_csv("space_avocado.csv", dtype=np.float64)
        #init models.yaml
        try:
            with open('models.yaml'): pass
        except IOError as e:
            print(e)
            sys.exit()
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    
    update_list = load_model()

    #best model
    best_model = None
    best_mse = inf
    for model in update_list:
        if float(model['mse']) is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
            print(f"Model {yellow}{model['name']}{reset} {red}not good{reset}")
        else:
            if float(model['mse']) < best_mse:
                best_mse = float(model['mse'])
                best_model = model
    # Importation of the dataset
    target = data.target.values.reshape(-1, 1) #price
    Xs = data[['weight','prod_distance','time_delivery']].values # features

    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)

    #normalisation
    scaler_x = Normalizer(x_train)
    scaler_y = Normalizer(y_train)

    x = scaler_x.norme(x_train)
    y = scaler_y.norme(y_train)

    x_test = scaler_x.norme(x_test)
    y_test = scaler_y.norme(y_test)
    
    name = best_model['name']
    hypo = best_model['polynomes']
    thetas = best_model['thetas']
    alpha = best_model['alpha']
    iter = best_model['iter']
    mylr = MyLR(thetas, alpha, iter, progress_bar=True)
    x_ = add_polynomial_features(x, hypo)
    x_test_ = add_polynomial_features(x_test, hypo)
    print(f"Model {yellow}{name}{reset} ... training")
    mse_list = mylr.fit_(x_, y)
    mse = MyLR.mse_(y_test, mylr.predict_(x_test_))

    y_hat = mylr.predict_(x_test_)
    plt.figure()
    plt.scatter(x_test[:, 0], y_test, c="b", marker='o', label="price")
    plt.scatter(x_test[:, 0], y_hat, c='r', marker='x', label="predicted price")
    plt.xlabel("Weight")
    plt.ylabel("Price")
    plt.title(f"Model : {name} MSE={mse:0.5f}")
    plt.figure()
    plt.scatter(x_test[:, 1], y_test, c="b", marker='o', label="price")
    plt.scatter(x_test[:, 1], y_hat, c='r', marker='x', label="predicted price")
    plt.title(f"Model : {name} MSE={mse:0.5f}")
    plt.xlabel("Distance")
    plt.ylabel("Price")
    plt.figure()
    plt.scatter(x_test[:, 2], y_test, c="b", marker='o', label="price")
    plt.scatter(x_test[:, 2], y_hat, c='r', marker='x', label="predicted price")
    plt.title(f"Model : {name} MSE={mse:0.5f}")
    plt.xlabel("Time")
    plt.ylabel("Price")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(iter), (mse_list))
    plt.title(f"Model : {name} MSE={mse:0.5f}")
    ax.set_xlabel("number iteration")
    ax.set_ylabel("mse")
    ax.grid()
    plt.show()

if __name__ == "__main__":
    print("space_avocado starting ...")
    main()
    print("Good by !")