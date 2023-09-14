import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

def FrankeFunction(x,y):
    """ Calculates the Franke function f(x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Calculate the Ordinary Least Squares

    ### Parameters:
        X (ndarray): Design/feature matrix
        y (ndarray): Data points/dependent variable

    ### Returns:
        beta (ndarray): Coefficients beta from the regression
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

print(OLS.__doc__)