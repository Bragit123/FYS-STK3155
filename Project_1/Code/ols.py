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

def FeatureMatrix(x: np.ndarray, y: np.ndarray, z: np.ndarray, deg: int) -> tuple:
    """ Calculates the feature matrix X, and scales X and z (if scale = True).

    ## Parameters
        x (ndarray): x-values of data points.
        y (ndarray): y-values of data points.
        z (ndarray): z-values of data points.
        deg (int): Degree of polynomial fit.
        scale (bool, default=True): If True, scale X and z before returning.
    
    ## Returns
        X (ndarray): Feature matrix (Scaled if scale=True).
        z (ndarray): z-values (scaled if scale=True).
    """

    ## Create feature matrix 
    xy_dic = {
        "x": x,
        "y": y
    }
    xy = pd.DataFrame(xy_dic)

    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(xy) # Find feature matrix 
    
    return X, z

def Scale(X_train: np.ndarray, X_test: np.ndarray, z_train: np.ndarray, z_test: np.ndarray) -> tuple:
    """ Scales X_train, X_test, z_train and z_test by subtracting the mean of
    the training data.

    ## Parameters
        X_train (ndarray): Feature matrix of training data.
        X_test (ndarray): Feature matrix of testing data.
        z_train (ndarray): z-values of training data.
        z_test (ndarray): z-values of test data.
    
    ## Returns
        X_train, X_test, z_train, z_test: Scaled versions of the input data.
    """

    # Compute the mean value of the training data.
    X_train = pd.DataFrame(X_train[:,1:])
    z_train = pd.DataFrame(z_train)
    X_train_mean = X_train.mean()
    z_train_mean = z_train.mean()

    # Scale training data
    X_train_scaled = X_train - X_train_mean
    z_train_scaled = z_train - z_train_mean
    
    # Scale test data
    X_test = pd.DataFrame(X_test[:,1:])
    z_test = pd.DataFrame(z_test)
    X_test_scaled = X_test - X_train_mean
    z_test_scaled = z_test - z_train_mean

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled
    

def OLSfit(X_train: np.ndarray, X_test: np.ndarray, z_train: np.ndarray, z_test: np.ndarray) -> tuple:
    """ Calculates a model fitting our data using ordinary least squares.
    
    ## Parameters
        X_train (ndarray): Feature matrix for training data.
        X_test (ndarray): Feature matrix for test data.
        z_train (ndarray): z-values of training data.
        z_test (ndarray): z-values of test data.
    
    ## Returns
        MSE_train (float): Mean square error of model on training data.
        MSE_test (float): Mean square error of model on test data.
        R2_train (float): R2 value of model on training data.
        R2_test (float): R2 value of model on test data.
        beta (ndarray): Coefficients of model.
    """

    ## Compute coefficients beta
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    
    ## Compute z_tilde from train data and z_predict from test_data
    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    ## Compute mean squared error (MSE) and R2-value from the model on train- and test data
    MSE_train = mean_squared_error(z_train,z_tilde)
    MSE_test = mean_squared_error(z_test,z_predict)
    R2_train = r2_score(z_train,z_tilde)
    R2_test = r2_score(z_test,z_predict)

    return MSE_train, MSE_test, R2_train, R2_test, beta

## Creating data set
N = 100 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

## Initiate arrays for the values that we want to compute
deg_num = 7
degs = np.linspace(1, deg_num, deg_num, dtype=int)
MSE_train_array = np.zeros(deg_num)
MSE_test_array = np.zeros(deg_num)
R2_train_array = np.zeros(deg_num)
R2_test_array = np.zeros(deg_num)
beta_list = [0]*deg_num

## Compute values from OLS
for i in range(deg_num):
    X, z = FeatureMatrix(x, y, z, degs[i]) # Compute feature matrix
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Split into training and test data
    
    X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test)

    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = OLSfit(X_train, X_test, z_train, z_test) # Compute model

plt.figure()
plt.title(f"Mean square error for ordinary least squares.")
plt.plot(degs,MSE_train_array,label="MSE_train")
plt.plot(degs,MSE_test_array,label="MSE_test")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.savefig("MSEOLS.pdf")

plt.figure()
plt.title(f"R2-values for ordinary least squares.")
plt.plot(degs,R2_train_array,label="R2_train")
plt.plot(degs,R2_test_array,label="R2_test")
plt.xlabel("degree")
plt.ylabel("R2-score")
plt.legend()
plt.savefig("R2OLS.pdf")