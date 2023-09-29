"""
# functions.py

This python script contains different functions that are frequently used in our
project. These functions are used for defining the FrankeFunction, finding the
feature matrix, scaling our data and performing different regression methods.
"""

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

def FeatureMatrix(x: np.ndarray, y: np.ndarray, z: np.ndarray, deg: int) -> np.ndarray:
    """ Calculates the feature matrix X, and scales X and z (if scale = True).

    ## Parameters
        x (ndarray): x-values of data points.
        y (ndarray): y-values of data points.
        z (ndarray): z-values of data points.
        deg (int): Degree of polynomial fit.
        scale (bool, default=True): If True, scale X and z before returning.

    ## Returns
        X (ndarray): Feature matrix (Scaled if scale=True).
    """

    ## Create feature matrix
    xy_dic = {
        "x": x,
        "y": y
    }
    xy = pd.DataFrame(xy_dic)

    poly = PolynomialFeatures(degree=deg)
    X = poly.fit_transform(xy) # Find feature matrix

    return X

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

def ridgefit(X_train: np.ndarray, X_test: np.ndarray, z_train: np.ndarray, z_test: np.ndarray, lambda_val: float) -> tuple:
    """ Calculates a model fitting our data using Ridge regression.

    ## Parameters
        X_train (ndarray): Feature matrix for training data.
        X_test (ndarray): Feature matrix for test data.
        z_train (ndarray): z-values of training data.
        z_test (ndarray): z-values of test data.
        lambda_val (float): lambda_value to use for the Ridge regression.

    ## Returns
        MSE_train (float): Mean square error of model on training data.
        MSE_test (float): Mean square error of model on test data.
        R2_train (float): R2 value of model on training data.
        R2_test (float): R2 value of model on test data.
        beta (ndarray): Coefficients of model.
    """

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

def Lassofit(X_train: np.ndarray, X_test: np.ndarray, z_train: np.ndarray, z_test: np.ndarray, lambda_val: float) -> tuple:
    """ Calculates a model fitting our data using Lasso regression.

    ## Parameters
        X_train (ndarray): Feature matrix for training data.
        X_test (ndarray): Feature matrix for test data.
        z_train (ndarray): z-values of training data.
        z_test (ndarray): z-values of test data.
        lambda_val (float): lambda_value to use for the Ridge regression.

    ## Returns
        MSE_train (float): Mean square error of model on training data.
        MSE_test (float): Mean square error of model on test data.
        R2_train (float): R2 value of model on training data.
        R2_test (float): R2 value of model on test data.
        beta (ndarray): Coefficients of model.
    """

    ## Compute Lasso model using scikit-learn
    clf = linear_model.Lasso(lambda_val, fit_intercept=False, max_iter=int(1e7)) # Increased max_iter to avoid ConvergenceWarning
    clf.fit(X_train,z_train)

    ## Compute z_tilde from train data and z_predict from test_data
    ztilde = clf.predict(X_train)
    zpredict = clf.predict(X_test)

    ## Compute mean squared error (MSE) and R2-value from the model on train- and test data
    MSE_train = sklearn.metrics.mean_squared_error(z_train,ztilde)
    MSE_test = sklearn.metrics.mean_squared_error(z_test,zpredict)
    R2_train = sklearn.metrics.r2_score(z_train,ztilde)
    R2_test = sklearn.metrics.r2_score(z_test,zpredict)

    return MSE_train, MSE_test, R2_train, R2_test
