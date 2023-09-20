
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

## Define functions for Least Squares and Ridge
def OLS(X, y):
    """ Ordinary Least Squares """
    XTX = X.transpose().dot(X)
    XTXinv = pd.DataFrame(inv(XTX.values))
    XTy = X.transpose().dot(y)
    beta = XTXinv.dot(XTy)
    return beta

def RDG(X, y, lambd):
    """ Ridge Regression """
    XTX = X.transpose().dot(X)
    I = np.diag(np.diag(np.ones(np.shape(XTX)))) # Identity matrix with same shape as XTX
    lambdI = pd.DataFrame(lambd * I)
    XTXlambdInv = pd.DataFrame(inv(XTX + lambdI))
    XTy = X.transpose().dot(y)
    beta = XTXlambdInv.dot(XTy)
    return beta

## Make data set.
np.random.seed()
n = 100
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(5*x) + 0.1 * np.random.normal(0, 0.1, x.shape)

## Do regression
deg_arr = np.arange(1, 16, 1) # Polynomial degree

mses = np.zeros(len(deg_arr))

for i in range(len(deg_arr)):
    deg = deg_arr[i]
    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(x)
    X_pandas = pd.DataFrame(X[:,1:])
    X_pandas = X_pandas - X_pandas.mean() # Scale design matrix by subtracting the mean.
    X_train, X_test, y_train, y_test = train_test_split(X_pandas, y, test_size=0.2)

    beta_ols = OLS(X_train, y_train)

    ## Predict y
    y_model_ols = X_test.dot(beta_ols)

    ## Compute Mean Squared Error
    n = len(y_model_ols)
    mse = mean_squared_error(y_test, y_model_ols) # 0th element is ols, 1-5 are rdg for the different lambdas
    mses[i] = mse

plt.figure()
plt.title(f"Degree {deg_arr[i]}")
plt.plot(deg_arr, mses)
plt.xlabel("Complexity (Degree of polynomial)")
plt.ylabel("Mean Square Error")
plt.savefig(f"mse.pdf")

