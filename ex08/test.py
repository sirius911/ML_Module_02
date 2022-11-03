import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298],
[ 3.88237651],
[ 4.37726357],
[ 4.63389049],
[ 7.79814439],
[ 6.41717461],
[ 8.63429886],
[ 8.19939795],
[10.37567392],
[10.68238222]])

# Build the model:
print(x.shape)
x_ = add_polynomial_features(x, 3)
print(x_)
print(x_.shape)
thetas = np.ones(4).reshape(-1,1)
print(thetas)
print(thetas.shape)
gradien = MyLR.gradien_(x_, y, thetas)
print(gradien)
my_lr = MyLR(thetas=thetas, alpha=0.00001, progress_bar=True).fit_(x_,y)
# my_lr.fit_(x_,y)
print(my_lr.thetas)
# Plot:
## To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
x_ = add_polynomial_features(continuous_x, 3)
print(continuous_x.shape)
print(my_lr.thetas.shape)
y_hat = my_lr.predict_(x_)
print(y_hat)

plt.scatter(x,y)
plt.plot(continuous_x, y_hat, color='orange')

plt.show()