
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
print(path)
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../spacecraft_data.csv")

# test in subject
# X = np.array(data[['Age']])
# Y = np.array(data[['Sell_price']])
# myLR_age = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000, progress_bar=True)
# myLR_age.fit_(X[:,0].reshape(-1,1), Y)

# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# print(MyLR.mse_(y_pred,Y))


def plot(X,Y,Y_hat, Xlabel, title):
    """ plot the real values and predicted value
    args:
        X: X values
        Y: Real Y values
        Y_hat: Y predicted values
        Xlabel: label X axe
        title: title of graph    
    """

    # draw real values
    plt.scatter(X, Y, c='blue', label="Sell price")
    # draw predicted values
    plt.scatter(X, Y_hat, c='c', s=8, label='Predicted sell price')

    plt.xlabel(Xlabel)
    plt.ylabel("y: sell price (in keuros)")
    plt.legend(frameon=True)
    plt.title(title)
    plt.grid(True)
    plt.show()

print("****** First Part ******")
print(" with age")
Xage = np.array(data['Age']).reshape(-1,1)
Ysell = np.array(data['Sell_price']).reshape(-1,1)

myLR_age = MyLR([[0.0],[0.0]], alpha=0.01, max_iter=5000, progress_bar=True)

myLR_age.fit_(Xage, Ysell)
# myLR_age.thetas = np.array([[647.09274075], [-12.99506324]])

print(f"Thetas = {myLR_age.thetas}")
y_hat_age = myLR_age.predict_(Xage)
mse = MyLR.mse_(Ysell, y_hat_age)
print(f"MSE = {mse}")
title = "MSE = " + str(round(mse,2))
plot(Xage, Ysell, y_hat_age, "$x_1: Age~(in~years)$",title)

print(" With Thrust")
Xthrust = np.array(data['Thrust_power']).reshape(-1,1)
myLR_thrust = MyLR([[0.0],[0.0]], alpha=1e-4, max_iter=1000, progress_bar=True)

myLR_thrust.fit_(Xthrust, Ysell)
# myLR_thrust.thetas = np.array([[39.27654867],[ 4.33215864]])

print(f"Thetas = {myLR_thrust.thetas}")
y_hat_thrust = myLR_thrust.predict_(Xthrust)
mse = MyLR.mse_(Ysell, y_hat_thrust)
print(f"MSE = {mse}")
title = "MSE = " + str(round(mse,2))
plot(Xthrust, Ysell, y_hat_thrust, "$x_2: thrust~(in~10km/s)$",title)

print(" With Distance")
Xdistance = np.array(data['Terameters']).reshape(-1,1)
myLR_distance = MyLR([[0.0],[0.0]], alpha=2e-4, max_iter=100000, progress_bar=True)

myLR_distance.fit_(Xdistance, Ysell)
# myLR_distance.thetas = np.array([[744.64256348], [ -2.8623013 ]])

print(f"Thetas = {myLR_distance.thetas}")
y_hat_distance = myLR_distance.predict_(Xdistance)
mse = MyLR.mse_(Ysell, y_hat_distance)
print(f"MSE = {mse}")
title = "MSE = " + str(round(mse,2))
plot(Xdistance, Ysell, y_hat_distance, "$x_3: distance~totalizer~value~of~spacescraft~(in~Tmeters)$",title)

print("second part")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
theta=np.array( [1.0, 1.0, 1.0, 1.0]).reshape(-1,1)
my_lreg = MyLR(thetas = theta, alpha = 5e-5, max_iter = 75000, progress_bar=True)
y_hat = my_lreg.predict_(X)
print(MyLR.mse_(Y,y_hat))

my_lreg.fit_(X,Y)

#after 30mn with alpha = 1e-5, max_iter = 4000000
# my_lreg.thetas = np.array([[359.89514161],[-23.43288337],[5.76394932],[-2.62662224]])
print(my_lreg.thetas)

y_hat = my_lreg.predict_(X)
mse = MyLR.mse_(Y,y_hat)
print(MyLR.mse_(Y,y_hat))
title = "MSE = " + str(round(mse,2))
plot(Xage, Ysell, y_hat, "$x_1: Age~(in~years)$",title)
plot(Xthrust, Ysell, y_hat, "$x_2: thrust~(in~10km/s)$",title)
plot(Xdistance, Ysell, y_hat, "$x_3: distance~totalizer~value~of~spacescraft~(in~Tmeters)$",title)

