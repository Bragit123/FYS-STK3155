
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def FrankeFunction(x, y):
    """ Calculates the Franke function at a point (x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

x = np.sort(np.random.rand(100))
y = np.sort(np.random.rand(100))
z = FrankeFunction(x, y)

dic = {
    "x": x,
    "y": y,
    "z": z
}

df = pd.DataFrame(dic)

xy = df[["x", "y"]] # Matrix combining x and y as columns
z = df["z"]

poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(xy)

X = pd.DataFrame(X[:,1:])
X = X - X.mean()

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

z_model_train = X_train @ beta
z_model_test = X_test @ beta

mse_train = mean_squared_error(z_train, z_model_train)
mse_test = mean_squared_error(z_test, z_model_test)

print(f"MSE_train = {mse_train}")
print(f"MSE_test = {mse_test}")

# ## with scikit
# clf = LinearRegression()
# clf.fit(X_train, z_train)
# z_model_train = clf.predict(X_train)
# z_model_test = clf.predict(X_test)

# z = np.asarray(z)
# # print(z)
# # print(z_predicted)

# mse_train = mean_squared_error(z_train, z_model_train)
# mse_test = mean_squared_error(z_test, z_model_test)
# print(mse_train)
# print(mse_test)