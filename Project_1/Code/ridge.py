import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from random import random, seed
seed = np.random.seed(200)

def FrankeFunction(x: float, y: float) -> float:
    """ Calculates the Franke function at a point (x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def ridgefit(x: np.ndarray, y: np.ndarray, z: np.ndarray, deg: int, lambda_val: float) -> tuple:
    """ Calculates a model fitting our data using Ridge regression.
    
    ## Parameters
        x (ndarray): x-values of data points.
        y (ndarray): y-values of data points.
        z (ndarray): z-values of data points.
        deg (int): Degree of polynomial fit.
        lambda_val (float): The lambda-value in Ridge regression.
    
    ## Returns
        MSE_train (float): Mean square error of model on training data.
        MSE_test (float): Mean square error of model on test data.
        R2_train (float): R2 value of model on training data.
        R2_test (float): R2 value of model on test data.
        beta (ndarray): Coefficients of model.
    
    """
    ## Create feature matrix 
    xy_dic = {
        "x": x,
        "y": y
    }
    xy = pd.DataFrame(xy_dic)

    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(xy) # Find feature matrix 

    # Scale X and z by subtracting mean (also ignore intercept of X to avoid
    # getting a singular matrix when calculating OLS).
    X = pd.DataFrame(X[:,1:])
    X = X - X.mean()
    z = pd.DataFrame(z)
    z = z - z.mean()
    
    ## Split X and z into train- and test data
    X_train, X_test, z_train, z_test = sklearn.model_selection.train_test_split(X, z, test_size= 0.2, random_state=0)

    ## Compute coefficients beta
    XTX_train = X_train.T @ X_train
    beta = np.linalg.pinv(np.add(XTX_train, lambda_val*np.identity(XTX_train.shape[0]))) @ X_train.T @ z_train
    
    ## Compute z_tilde from train data and z_predict from test_data
    ztilde = X_train @ beta
    zpredict = X_test @ beta

    ## Compute mean squared error (MSE) and R2-value from the model on train- and test data
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_test = sklearn.metrics.mean_squared_error(z_test,zpredict)
    R2_train = sklearn.metrics.r2_score(z_train,ztilde)
    R2_test = sklearn.metrics.r2_score(z_test,zpredict)
    return MSE_train, MSE_test, R2_train, R2_test, beta


## Creating data set
N = 50 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z_with_noise = z# + np.random.normal(0, 1, z.shape)

## Initiate arrays for the values that we want to compute
deg = 5 # Polynomial degree
lambda_exp_start = -20
lambda_exp_stop = -5
lambda_num = 50

lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)
MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num

for i in range(lambda_num):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = ridgefit(x,y,z, deg,lambdas[i])

plt.figure()
plt.plot(np.log10(lambdas),MSE_train_array,label="MSE_train, Ridge")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("MSE")
plt.legend()
plt.savefig("MSERidge.png")

plt.figure()
plt.plot(np.log10(lambdas),R2_train_array,label="R2_train, Ridge")
plt.plot(np.log10(lambdas),R2_test_array,label="R2_test, Ridge")
plt.xlabel("log10lambda")
plt.ylabel("R2-score")
plt.legend()
plt.savefig("R2Ridge.png")