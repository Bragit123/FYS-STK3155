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
N = 30 # Number of data points
x = np.sort(np.random.rand(N))
y = np.sort(np.random.rand(N))
z = FrankeFunction(x, y)
z = z + 0.1*np.random.normal(0, 1, z.shape)

## Initiate arrays for the values that we want to compute
deg = 5 # Polynomial degree
lambda_exp_start = -10
lambda_exp_stop = -1
lambda_num = 30

lambdas = np.logspace(lambda_exp_start, lambda_exp_stop, num=lambda_num)
MSE_train_array = np.zeros(lambda_num)
MSE_test_array = np.zeros(lambda_num)
R2_train_array = np.zeros(lambda_num)
R2_test_array = np.zeros(lambda_num)
beta_list = [0]*lambda_num
X = FeatureMatrix(x, y, deg) # Compute feature matrix
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Split into training and test data
X_train, X_test, z_train, z_test = Scale(X_train, X_test, z_train, z_test) # Scale data

for i in range(lambda_num):
    MSE_train_array[i], MSE_test_array[i], R2_train_array[i], R2_test_array[i], beta_list[i] = Lassofit(X_train, X_test, z_train, z_test, lambdas[i])

plt.figure()
plt.title(f"Mean square error for Lasso regression with polynomial degree {deg}.", fontsize=20)
plt.plot(np.log10(lambdas),MSE_train_array,label="MSE_train, Lasso")
plt.plot(np.log10(lambdas),MSE_test_array,label="MSE_test, Lasso")
plt.xlabel("log10lambda", fontsize=20)
plt.ylabel("MSE", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)
plt.savefig("../Figures/MSELasso.pdf", bbox_inches='tight')

plt.figure()
plt.title(f"R2-values for Lasso regression with polynomial degree {deg}.", fontsize=20)
plt.plot(np.log10(lambdas),R2_train_array,label="R2_train, Lasso")
plt.plot(np.log10(lambdas),R2_test_array,label="R2_test, Lasso")
plt.xlabel("log10lambda", fontsize=20)
plt.ylabel("R2-score", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=20)
plt.savefig("../Figures/R2Lasso.pdf", bbox_inches='tight')
