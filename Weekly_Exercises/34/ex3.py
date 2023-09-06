
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## a
np.random.seed()
n = 100

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
"""
poly5 = PolynomialFeatures(degree=5)
X = poly5.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## b
clf = LinearRegression()
clf.fit(X_train, y_train)

y_tilde_train = clf.predict(X_train)
y_tilde_test = clf.predict(X_test)

MSE_train = mean_squared_error(y_train, y_tilde_train)
MSE_test = mean_squared_error(y_test, y_tilde_test)
"""
## c
deg_num = 15
MSE_train_arr = np.zeros(deg_num-1)
MSE_test_arr = np.zeros(deg_num-1)
degrees = np.linspace(2, deg_num, deg_num-1, dtype=int)

for deg in degrees:
    poly15 = PolynomialFeatures(degree=deg)
    X = poly15.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_tilde_train = clf.predict(X_train)
    y_tilde_test = clf.predict(X_test)

    MSE_train_arr[deg-2] = mean_squared_error(y_train, y_tilde_train)
    MSE_test_arr[deg-2] = mean_squared_error(y_test, y_tilde_test)

plt.plot(degrees, MSE_train_arr, label="MSE_train")
plt.plot(degrees, MSE_test_arr, label="MSE_test")
plt.legend()
plt.savefig("mse.pdf")