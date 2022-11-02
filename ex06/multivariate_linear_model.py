
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../spacecraft_data.csv")

# test in subject
# X = np.array(data[['Age']])
# Y = np.array(data[['Sell_price']])
# myLR_age = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000, progress_bar=True)
# myLR_age.fit_(X[:,0].reshape(-1,1), Y)

# y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
# print(MyLR.mse_(y_pred,Y))


# print("****** First Part ******")
print(" with age")
Xage = np.array(data['Age']).reshape(-1,1)
Ysell = np.array(data['Sell_price']).reshape(-1,1)

myLR_age = MyLR([[0.0],[0.0]], alpha=0.001, max_iter=150000, progress_bar=True)

# myLR_age.fit_(Xage, Ysell)
myLR_age.thetas = np.array([[647.09274075], [-12.99506324]])

print(f"Thetas = {myLR_age.thetas}")
y_hat_age = myLR_age.predict_(Xage)
mse = MyLR.mse_(Ysell, y_hat_age)
print(f"MSE = {mse}")
# draw real values
plt.scatter(Xage, Ysell, c='blue', label="Sell price")
# draw predicted values
plt.scatter(Xage, y_hat_age, c='c', s=8, label='Predicted sell price')
plt.xlabel("$x_1: Age~(in~years)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()

print(" With Thrust")
Xthrust = np.array(data['Thrust_power']).reshape(-1,1)
myLR_thrust = MyLR([[0.0],[0.0]], alpha=0.0001, max_iter=400000, progress_bar=True)

# myLR_thrust.fit_(Xthrust, Ysell)
myLR_thrust.thetas = np.array([[39.27654867],[ 4.33215864]])

print(f"Thetas = {myLR_thrust.thetas}")
y_hat_thrust = myLR_thrust.predict_(Xthrust)
mse = MyLR.mse_(Ysell, y_hat_thrust)
print(f"MSE = {mse}")
# draw real values
plt.scatter(Xthrust, Ysell, c='green', label="Sell price")
# draw predicted values
plt.scatter(Xthrust, y_hat_thrust, s=4, c="olive", label='Predicted sell price')
plt.xlabel("$x_2: thrust~(in~10km/s)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()

print(" With Distance")
Xdistance = np.array(data['Terameters']).reshape(-1,1)
myLR_distance = MyLR([[0.0],[0.0]], alpha=0.0001, max_iter=500000, progress_bar=True)

# myLR_distance.fit_(Xdistance, Ysell)
myLR_distance.thetas = np.array([[744.64256348], [ -2.8623013 ]])

print(f"Thetas = {myLR_distance.thetas}")
y_hat_distance = myLR_distance.predict_(Xdistance)
mse = MyLR.mse_(Ysell, y_hat_distance)
print(f"MSE = {mse}")
# draw real values
plt.scatter(Xdistance, Ysell, c='b', label="Sell price")
# draw predicted values
plt.scatter(Xdistance, y_hat_distance, s=4, c="cyan", label='Predicted sell price')
plt.xlabel("$x_3: distance~totalizer~value~of~spacescraft~(in~Tmeters)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()
print("second part")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
theta=np.array( [1.0, 1.0, 1.0, 1.0]).reshape(-1,1)
my_lreg = MyLR(thetas = theta, alpha = 1e-5, max_iter = 4000000, progress_bar=True)
y_hat = my_lreg.predict_(X)
print(MyLR.mse_(Y,y_hat))
print(MyLR.gradien_(X,Y,theta))

# my_lreg.fit_(X,Y)
# my_lreg.thetas = np.array([[334.994],[-22.535],[5.857],[-2.586]]) #thetas target

#after 30mn
my_lreg.thetas = np.array([[359.89514161],[-23.43288337],[5.76394932],[-2.62662224]])
print(my_lreg.thetas)
# Output:
# array([[334.994...],[-22.535...],[5.857...],[-2.586...]])
# Example 2:
y_hat = my_lreg.predict_(X)
mse = MyLR.mse_(Y,y_hat)
print(MyLR.mse_(Y,y_hat))

# # draw real values
plt.scatter(Xage, Y, c='m', label="Sell price")
# draw predicted values
plt.scatter(Xage, y_hat, c='b', s=6,label="Prédicted Sell price")
plt.xlabel("$x_1: Age~(in~years)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()

# # draw real values
plt.scatter(Xthrust, Y, c='m', label="Sell price")
# draw predicted values
plt.scatter(Xthrust, y_hat, c='b', s=6,label="Predicted Sell price")
plt.xlabel("$x_2: thrust~(in~10km/s)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()

# # draw real values
plt.scatter(Xdistance, Y, c='m', label="Sell price")
# draw predicted values
plt.scatter(Xdistance, y_hat, c='b', s=6,label="Prédicted Sell price")
plt.xlabel("$x_3: distance~totalizer~value~of~spacescraft~(in~Tmeters)$")
plt.ylabel("y: sell price (in keuros)")
plt.legend(frameon=True)
title = "MSE = " + str(round(mse,2))
plt.title(title)
plt.grid(True)
plt.show()