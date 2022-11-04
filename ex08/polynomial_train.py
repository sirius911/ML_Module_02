import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)

from polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

if __name__ == "__main__":
     #main program
    nb_model = 6
     #loading data
    try:
        data = pd.read_csv('../are_blue_pills_magics.csv')
    except Exception as e:
        print(e)
        sys.exit(1)
    
    target = data.Score.values.reshape(-1,1)
    feature = data.Micrograms.values

    features = []
    for i in range(nb_model):
        features.append(add_polynomial_features(feature, i + 1))

    # feature1 = add_polynomial_features(feature, 1)
    # thetas
    np.random.seed = 42
    # theta1 = np.random.rand(2, 1) #t0 + t1x
    theta1 = np.array([[89.12387377], [-9.00946642]])
    # theta2 = np.random.rand(3, 1) #t0 + t1x + t2x^2
    theta2 = np.array([[ 91.9896726], [-10.88120698], [0.23854169]])
    # theta3 = np.random.rand(4, 1) #t0 + t1x + t2x^2 + t3x^3
    theta3 = np.array([[70.332973], [13.45587206], [-7.13417592], [0.64955749]])
    # theta4 = np.array([-20., 160., -80., 10., -1.]).reshape(-1, 1) #t0 + t1x + t2x^2 + t3x^3 + t4x^4
    theta4 = np.array([[-20.21868533], [160.04015269], [-78.81522318], [14.30207777], [-0.89077285]])
    # theta5 = np.array([1140., -1850., 1110., -305., 40., -2.]).reshape(-1, 1) #t0 + t1x + t2x^2 + t3x^3 + t4x^4 + t5x^5
    theta5 = np.array([[ 1140.15316087], [-1849.72463317], [ 1110.46138744], [ -304.58270338], [38.99629805], [-1.89404028]])
    # theta6 = np.array([9110., -18015., 13400., -4935., 966., -96.4, 3.86]).reshape(-1, 1)  #t0 + t1x + t2x^2 + t3x^3 + t4x^4 + t5x^5 + t6x^6
    theta6 = np.array([[ 9.10999607e+03], [-1.80150088e+04], [ 1.33999801e+04], [-4.93503821e+03], [ 9.65959917e+02], [-9.63268478e+01], [ 3.85235987e+00]])

    #Linear Regression models
    mylr = []
    mylr.append(MyLR(thetas=theta1, alpha=1e-3, max_iter=1000000, progress_bar=True))
    mylr.append(MyLR(thetas=theta2, alpha=1e-3, max_iter=1000000, progress_bar=True))
    mylr.append(MyLR(thetas=theta3, alpha=5e-5, max_iter=5000000, progress_bar=True))
    mylr.append(MyLR(thetas=theta4, alpha=1e-6, max_iter=1000000, progress_bar=True))
    mylr.append(MyLR(thetas=theta5, alpha=4e-8, max_iter=5000000, progress_bar=True))
    mylr.append(MyLR(thetas=theta6, alpha=1e-9, max_iter=5000000, progress_bar=True))

    dict = {}
    for i in range(nb_model):
        model = "model"+str(i + 1)
        print(f"Training {model}")
        # mylr[i].fit_(features[i], target)
        prediction = mylr[i].predict_(features[i])
        mse = MyLR.mse_(target,prediction)        
        dict[model] = mse
        print(f"MSE = {mse}")

     # creating the bar plot
    courses = list(dict.keys())
    values = list(dict.values())
    fig = plt.figure(figsize = (10, 5))
    colors=['black', 'red', 'green', 'blue', 'cyan', 'orange']
   
    plt.bar(courses, values, color=colors,\
        tick_label=courses, width = 0.4, label=[round(x, 2) for x in values])
    
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Score MSE by models")
    plt.show()
   
    # models and data points
    x_sampling = np.linspace(1, 7, 100)
    for i in range(nb_model):
        model = "model"+str(i + 1)
        plt.plot(x_sampling, mylr[i].predict_(add_polynomial_features(x_sampling, i+1)),
                label=model, color=colors[i] )
    plt.scatter(feature, target, label='targets', c='black')
    
    plt.grid()
    plt.legend()
    plt.show()