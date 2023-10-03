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
from functions import *
plt.style.use("seaborn-v0_8")

from random import random, seed
seed = np.random.seed(200)

## Creating data set
N = 100 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z_with_noise = z + np.random.normal(0, 1, z.shape)

## Initiate arrays for the values that we want to compute
deg_num = 20
degs = np.linspace(1, deg_num, deg_num, dtype=int)
MSE_train_array = np.zeros(deg_num)
MSE_test_array = np.zeros(deg_num)
R2_train_array = np.zeros(deg_num)
R2_test_array = np.zeros(deg_num)
beta_list = [0]*deg_num

## Compute values from OLS
for i in range(deg_num):
    X = FeatureMatrix(x, y, z, degs[i]) # Compute feature matrix
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Split into training and test data
    X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test) # Scale data

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
