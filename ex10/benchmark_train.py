from cmath import inf, nan
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ft_yaml import init_model_yaml, load_model, save_model

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

from polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

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

def graph_3D(data):
    """
    show the graphics of data in 3D
    """
    target = data.target.values.reshape(-1, 1) #price
    weight = data.weight.values.reshape(-1, 1)
    prod_distance = data.prod_distance.values.reshape(-1, 1)
    time_delivery = data.time_delivery.values.reshape(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Analyzer')
    ax.set_xlabel('weight')
    ax.set_ylabel('distance')
    ax.set_zlabel('time')
    min_target = target.min()
    max_target = target.max()
    taille = target - min_target
    taille = taille / (max_target - min_target) * 100
    p = ax.scatter(weight, prod_distance, time_delivery, s=taille, c=target, alpha = 0.9, cmap='viridis', vmin = min_target, vmax = max_target)
    cbar = plt.colorbar(p)
    cbar.set_label("price of the order (in trantorian unit)", labelpad=+1)
    plt.show()

def main():
    print("Loading models ...")
    try:
    # Importation of the dataset
        data = pd.read_csv("space_avocado.csv", dtype=np.float64)
        #init models.yaml
        try:
            with open('models.yaml'): pass
        except IOError:
            if not init_model_yaml():
                sys.exit()
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    target = data.target.values.reshape(-1, 1) #price
    Xs = data[['weight','prod_distance','time_delivery']].values # features
    
    # 3D
    graph_3D(data)
 
    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)

    #normalisation
    scaler_x = Normalizer(x_train)
    scaler_y = Normalizer(y_train)

    x = scaler_x.norme(x_train)
    y = scaler_y.norme(y_train)

    x_test = scaler_x.norme(x_test)
    y_test = scaler_y.norme(y_test)

    update_list = load_model()
    changed = False
    if update_list is not None:
        print(f"{green}Ok{reset}")
        nb_model = len(update_list)
        for idx, model in zip(range(nb_model), update_list):
            print(f"Model {idx + 1}/{nb_model} -> {yellow}{model['name']}{reset}", end=" ")
            if model['mse'] is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
                print(" ... training")
                hypo = model['polynomes']
                thetas = model['thetas']
                x_ = add_polynomial_features(x, hypo)
                x_test_ = add_polynomial_features(x_test, hypo)
                alpha = model['alpha']
                iter = model['iter']
                mylr = MyLR(thetas, alpha, iter, progress_bar=True)
                mse_list = mylr.fit_(x_, y)
                model['evol_mse'] = [float(m) for m in mse_list]
                mse = MyLR.mse_(y_test, mylr.predict_(x_test_))
                model['mse'] = float(mse)
                print(f"\tMSE = {green}{mse}{reset}")
                changed = True
            else:
                print("MSE = ", end="")
                if model['mse'] is not None or model['mse']!= inf or model['mse']!= nan:
                    print(green, end='')
                else:
                    print(red, end='')
                print(f"{model['mse']:0.8f}{reset}")
    if changed:
        print("Saving models ...", end=" ")
        if save_model(list_models=update_list):
            print(f"{green}OK{reset}")
        else:
            print(f"{red}\tKO{reset}")

    # controle
    print("model control ...", end='')
    best_model = None
    best_mse = inf
    for model in update_list:
        if float(model['mse']) is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
            print(f"Model {yellow}{model['name']}{reset} {red}not good{reset}")
        else:
            if float(model['mse']) < best_mse:
                best_mse = float(model['mse'])
                best_model = model
    print(" ended")
    print(f"Best Model -> {yellow}{best_model['name']}{reset} with MSE = {green}{best_model['mse']}{reset}")
    
    #graph mse
    fig = plt.figure()
    ax = fig.add_subplot()
    for model in update_list:
        ax.plot(np.arange(model['iter']), model['evol_mse'])
    ax.set_xlabel("number iteration")
    ax.set_ylabel("mse")
    ax.grid()
    plt.axis([0, 200, 0, 2])
    plt.show()

if __name__ == "__main__":
    print("Benchmar starting ...")
    main()
    print("Good by !")